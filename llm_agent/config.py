"""
Configuration classes for the LLM agent.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMConfig:
    """Configuration for connecting to an LLM provider."""

    provider_type: str  # "openai", "anthropic"
    api_key: str
    model_name: str
    api_base: Optional[str] = None

    def __post_init__(self):
        if not self.provider_type:
            raise ValueError("provider_type is required")
        if not self.api_key:
            raise ValueError("api_key is required")
        if not self.model_name:
            raise ValueError("model_name is required")


@dataclass
class AgentConfig:
    """Configuration for agent execution behavior."""

    max_iterations: int = 10
    max_tool_calls_per_iteration: int = 5
    temperature: float = 0.7
    max_tokens: int = 4096
    tool_timeout: float = 30.0
    parallel_tool_execution: bool = True
    max_history_messages: int = 50
