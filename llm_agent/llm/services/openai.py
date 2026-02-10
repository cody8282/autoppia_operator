"""
OpenAI LLM service implementation.
"""

import json
import logging
from typing import Any, Dict, Iterator, List, Optional

from ..base import (
    BaseLLMService,
    CompletionResponse,
    Message,
    MessageRole,
    StreamChunk,
    Tool,
    ToolCall,
)

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAIService(BaseLLMService):
    """OpenAI chat-completion service with tool-calling support."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        api_base: Optional[str] = None,
        default_temperature: float = 0.7,
        default_max_tokens: int = 4096,
        organization: Optional[str] = None,
    ):
        super().__init__(api_key, model, api_base, default_temperature, default_max_tokens)
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required. Run: pip install openai")

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base
        if organization:
            client_kwargs["organization"] = organization
        self.client = OpenAI(**client_kwargs)

    # ── Format helpers ───────────────────────────────────────────

    def _to_api_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        result = []
        for msg in messages:
            d: Dict[str, Any] = {"role": msg.role.value}
            if msg.role == MessageRole.TOOL:
                d["role"] = "tool"
                d["content"] = msg.content or ""
                d["tool_call_id"] = msg.tool_call_id
            elif msg.role == MessageRole.ASSISTANT and msg.tool_calls:
                d["content"] = msg.content or ""
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": (
                                json.dumps(tc.arguments)
                                if isinstance(tc.arguments, dict)
                                else tc.arguments
                            ),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            else:
                d["content"] = msg.content or ""
            result.append(d)
        return result

    def _parse_tool_calls(self, raw: List[Any]) -> List[ToolCall]:
        result = []
        for tc in raw:
            try:
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": tc.function.arguments}
            result.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        return result

    # ── Core API ─────────────────────────────────────────────────

    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> CompletionResponse:
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": self._to_api_messages(messages),
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
        }
        if tools:
            params["tools"] = [t.to_openai_format() for t in tools]
            params["tool_choice"] = kwargs.pop("tool_choice", "auto")
        params.update(kwargs)

        response = self.client.chat.completions.create(**params)
        choice = response.choices[0]

        return CompletionResponse(
            content=choice.message.content,
            tool_calls=(
                self._parse_tool_calls(choice.message.tool_calls)
                if choice.message.tool_calls
                else None
            ),
            finish_reason=choice.finish_reason,
            usage=(
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                if response.usage
                else None
            ),
            model=response.model,
        )

    def complete_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": self._to_api_messages(messages),
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
            "stream": True,
        }
        if tools:
            params["tools"] = [t.to_openai_format() for t in tools]
            params["tool_choice"] = kwargs.pop("tool_choice", "auto")

        stream = self.client.chat.completions.create(**params)
        accumulators: Dict[int, Dict[str, str]] = {}

        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content:
                yield StreamChunk(content=delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in accumulators:
                        accumulators[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        accumulators[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            accumulators[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            accumulators[idx]["arguments"] += tc.function.arguments
                    yield StreamChunk(tool_call_chunk=accumulators[idx].copy())

            if choice.finish_reason:
                yield StreamChunk(finish_reason=choice.finish_reason, is_final=True)

    def is_available(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False
