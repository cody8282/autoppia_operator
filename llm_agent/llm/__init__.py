"""
LLM service layer.

Quick start::

    from llm_agent.llm import create_llm_service, Message

    llm = create_llm_service("openai", api_key="sk-...", model="gpt-4o")
    response = llm.complete([
        Message.system("You are a helpful assistant."),
        Message.user("Hello!"),
    ])
    print(response.content)
"""

from .base import (
    BaseLLMService,
    CompletionResponse,
    Message,
    MessageRole,
    StreamChunk,
    Tool,
    ToolCall,
    ToolResult,
)
from .services import AnthropicService, OpenAIService


def create_llm_service(
    provider: str,
    api_key: str,
    model: str = None,
    api_base: str = None,
    **kwargs,
) -> BaseLLMService:
    """Factory: create an LLM service by provider name.

    Args:
        provider: ``"openai"`` or ``"anthropic"``.
        api_key: API key for the provider.
        model: Model identifier (uses provider default when *None*).
        api_base: Optional custom API base URL.

    Returns:
        A ready-to-use :class:`BaseLLMService` instance.
    """
    provider = provider.lower()

    if provider == "openai":
        return OpenAIService(
            api_key=api_key, model=model or "gpt-4o", api_base=api_base, **kwargs
        )
    elif provider == "anthropic":
        return AnthropicService(
            api_key=api_key,
            model=model or "claude-sonnet-4-20250514",
            api_base=api_base,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. Available: openai, anthropic"
        )


__all__ = [
    "create_llm_service",
    "BaseLLMService",
    "OpenAIService",
    "AnthropicService",
    "Message",
    "MessageRole",
    "Tool",
    "ToolCall",
    "ToolResult",
    "CompletionResponse",
    "StreamChunk",
]
