"""
Data models for the LLM agent.
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskStatus(Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class TaskEnvelope:
    """Incoming task to be processed by the agent."""

    task_id: str
    capability: str
    input: Dict[str, Any]
    parent_task_id: Optional[str] = None
    workspace_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "workspace_id": self.workspace_id,
            "capability": self.capability,
            "input": self.input,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskEnvelope":
        return cls(
            task_id=data["task_id"],
            parent_task_id=data.get("parent_task_id"),
            workspace_id=data.get("workspace_id"),
            capability=data["capability"],
            input=data["input"],
            metadata=data.get("metadata"),
        )


@dataclass
class TaskResult:
    """Result returned after processing a task."""

    task_id: str
    status: TaskStatus
    output: Dict[str, Any]
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskResult":
        return cls(
            task_id=data["task_id"],
            status=TaskStatus(data["status"]),
            output=data["output"],
            error=data.get("error"),
            metrics=data.get("metrics"),
        )


@dataclass
class AgentResponse:
    """Detailed response from agent execution."""

    content: str
    tool_calls_made: List[Dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    tokens_used: Optional[Dict[str, int]] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None
