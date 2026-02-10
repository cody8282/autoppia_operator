"""
Base LLM service types and abstract interface.

Defines the core types (Message, Tool, ToolCall, CompletionResponse) and
the BaseLLMService that all provider implementations must extend.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """A tool call requested by the LLM."""

    id: str
    name: str
    arguments: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class ToolResult:
    """Result of executing a tool."""

    tool_call_id: str
    name: str
    result: Any
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class Message:
    """A message in the conversation history."""

    role: MessageRole
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role.value}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d

    # ── Factory helpers ──────────────────────────────────────────

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(
        cls, content: str = None, tool_calls: List[ToolCall] = None
    ) -> "Message":
        return cls(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool_result(cls, tool_call_id: str, name: str, result: str) -> "Message":
        return cls(
            role=MessageRole.TOOL,
            content=result,
            tool_call_id=tool_call_id,
            name=name,
        )


@dataclass
class Tool:
    """Definition of a tool the LLM can invoke."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    handler: Optional[Callable] = None

    def to_openai_format(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_format(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass
class CompletionResponse:
    """Response from an LLM completion call."""

    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def to_message(self) -> Message:
        return Message.assistant(content=self.content, tool_calls=self.tool_calls)


@dataclass
class StreamChunk:
    """An incremental chunk from a streaming LLM response."""

    content: Optional[str] = None
    tool_call_chunk: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    is_final: bool = False


class BaseLLMService(ABC):
    """
    Abstract base for LLM provider implementations.

    Subclasses must implement ``complete``, ``complete_stream``, and ``is_available``.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        api_base: Optional[str] = None,
        default_temperature: float = 0.7,
        default_max_tokens: int = 4096,
    ):
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> CompletionResponse:
        ...

    @abstractmethod
    def complete_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Iterator[StreamChunk]:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...
