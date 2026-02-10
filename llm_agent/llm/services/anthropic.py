"""
Anthropic LLM service implementation.
"""

import json
import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

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
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class AnthropicService(BaseLLMService):
    """Anthropic Claude service with tool-use support."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        api_base: Optional[str] = None,
        default_temperature: float = 0.7,
        default_max_tokens: int = 4096,
    ):
        super().__init__(api_key, model, api_base, default_temperature, default_max_tokens)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package required. Run: pip install anthropic")

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base
        self.client = anthropic.Anthropic(**client_kwargs)

    # ── Format helpers ───────────────────────────────────────────

    def _to_api_messages(
        self, messages: List[Message]
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Return (system_prompt, messages) in Anthropic format."""
        system_prompt = None
        result: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
            elif msg.role == MessageRole.USER:
                result.append({"role": "user", "content": msg.content or ""})
            elif msg.role == MessageRole.ASSISTANT:
                if msg.tool_calls:
                    content: List[Dict[str, Any]] = []
                    if msg.content:
                        content.append({"type": "text", "text": msg.content})
                    for tc in msg.tool_calls:
                        content.append(
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.arguments,
                            }
                        )
                    result.append({"role": "assistant", "content": content})
                else:
                    result.append({"role": "assistant", "content": msg.content or ""})
            elif msg.role == MessageRole.TOOL:
                result.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content or "",
                            }
                        ],
                    }
                )

        return system_prompt, result

    def _parse_tool_calls(self, content_blocks: List[Any]) -> List[ToolCall]:
        result = []
        for block in content_blocks:
            if getattr(block, "type", None) == "tool_use":
                result.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )
        return result

    def _extract_text(self, content_blocks: List[Any]) -> Optional[str]:
        parts = [
            block.text
            for block in content_blocks
            if getattr(block, "type", None) == "text"
        ]
        return "".join(parts) if parts else None

    # ── Core API ─────────────────────────────────────────────────

    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> CompletionResponse:
        system_prompt, api_messages = self._to_api_messages(messages)

        params: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
            "temperature": temperature if temperature is not None else self.default_temperature,
        }
        if system_prompt:
            params["system"] = system_prompt
        if tools:
            params["tools"] = [t.to_anthropic_format() for t in tools]
        params.update(kwargs)

        response = self.client.messages.create(**params)

        content = self._extract_text(response.content)
        tool_calls = self._parse_tool_calls(response.content)

        return CompletionResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=response.stop_reason,
            usage=(
                {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
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
        system_prompt, api_messages = self._to_api_messages(messages)

        params: Dict[str, Any] = {
            "model": self.model,
            "messages": api_messages,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
            "temperature": temperature if temperature is not None else self.default_temperature,
        }
        if system_prompt:
            params["system"] = system_prompt
        if tools:
            params["tools"] = [t.to_anthropic_format() for t in tools]

        with self.client.messages.stream(**params) as stream:
            current_tool_use: Optional[Dict[str, Any]] = None

            for event in stream:
                if event.type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if getattr(block, "type", None) == "tool_use":
                        current_tool_use = {
                            "id": block.id,
                            "name": block.name,
                            "arguments": "",
                        }

                elif event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        yield StreamChunk(content=event.delta.text)
                    elif hasattr(event.delta, "partial_json") and current_tool_use:
                        current_tool_use["arguments"] += event.delta.partial_json
                        yield StreamChunk(tool_call_chunk=current_tool_use.copy())

                elif event.type == "content_block_stop":
                    if current_tool_use:
                        try:
                            current_tool_use["arguments"] = json.loads(
                                current_tool_use["arguments"]
                            )
                        except json.JSONDecodeError:
                            pass
                        current_tool_use = None

                elif event.type == "message_stop":
                    yield StreamChunk(finish_reason="end_turn", is_final=True)

    def is_available(self) -> bool:
        try:
            self.client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception:
            return False
