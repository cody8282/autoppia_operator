"""
LLM Agent – ReAct (Reason + Act) loop.

This is the core agent that:
1. Receives a user message
2. Calls the LLM with available tools
3. Executes any requested tool calls
4. Loops until the LLM produces a final answer
"""

import json
import logging
from typing import Any, Dict, List, Optional

from .config import AgentConfig, LLMConfig
from .llm import Message, create_llm_service
from .llm.base import BaseLLMService, Tool
from .models import AgentResponse, TaskEnvelope, TaskResult, TaskStatus
from .tools import ToolExecutor, ToolRegistry

logger = logging.getLogger(__name__)


class LLMAgent:
    """
    A basic LLM agent implementing the ReAct loop.

    Example::

        from llm_agent import LLMAgent, AgentConfig, LLMConfig

        agent = LLMAgent(
            name="my-agent",
            system_prompt="You are a helpful assistant.",
            llm_config=LLMConfig(
                provider_type="openai",
                api_key="sk-...",
                model_name="gpt-4o",
            ),
        )
        agent.start()
        result = agent.act("What is 2+2?")
        print(result.content)
        agent.stop()
    """

    def __init__(
        self,
        name: str,
        system_prompt: str = "",
        llm_config: Optional[LLMConfig] = None,
        llm_service: Optional[BaseLLMService] = None,
        agent_config: Optional[AgentConfig] = None,
        tools: Optional[ToolRegistry] = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.agent_config = agent_config or AgentConfig()

        # LLM – either inject a service directly or provide config
        self._llm_config = llm_config
        self._llm_service = llm_service

        # Tools
        self.tool_registry = tools or ToolRegistry()
        self.tool_executor: Optional[ToolExecutor] = None

        # Conversation history
        self._history: List[Message] = []
        self._started = False

    # ── Lifecycle ────────────────────────────────────────────────

    def start(self) -> None:
        """Initialise the agent (must be called before act)."""
        if self._started:
            return

        self.tool_executor = ToolExecutor(
            registry=self.tool_registry,
            default_timeout=self.agent_config.tool_timeout,
        )

        if self.system_prompt:
            self._history = [Message.system(self.system_prompt)]
        else:
            self._history = []

        self._started = True
        logger.info(f"Agent '{self.name}' started")

    def stop(self) -> None:
        """Release resources."""
        if not self._started:
            return
        self._history = []
        self.tool_executor = None
        self._started = False
        logger.info(f"Agent '{self.name}' stopped")

    # ── Main entry points ────────────────────────────────────────

    def act(self, message: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Run the ReAct loop for a single user message.

        This is the primary interface: the agent *observes* the user
        message, *reasons* via the LLM, and *acts* by calling tools,
        repeating until a final answer is produced.

        Args:
            message: User input.
            context: Optional metadata appended to the message.

        Returns:
            AgentResponse with the final content and execution metadata.
        """
        if not self._started:
            self.start()

        return self._react_loop(message, context)

    def call(self, task: TaskEnvelope, call_context: Optional[Dict] = None) -> TaskResult:
        """Process a TaskEnvelope (compatible with the Autoppia worker interface)."""
        if not self._started:
            self.start()

        response = self._react_loop(task.input["content"], call_context)
        return TaskResult(
            task_id=task.task_id,
            status=TaskStatus.SUCCEEDED if response.success else TaskStatus.FAILED,
            output={"content": response.content},
            error=response.error,
            metrics=response.tokens_used,
        )

    # ── ReAct loop ───────────────────────────────────────────────

    def _react_loop(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResponse:
        """Core Reason + Act loop."""
        full_message = message
        if context:
            full_message = f"{message}\n\nContext:\n{json.dumps(context, indent=2, default=str)}"

        self._history.append(Message.user(full_message))

        llm = self._get_llm()
        if llm is None:
            return AgentResponse(content="Error: No LLM service configured", error="no_llm")

        tools: Optional[List[Tool]] = (
            self.tool_registry.get_tools() if len(self.tool_registry) > 0 else None
        )

        iterations = 0
        tool_calls_made: List[Dict[str, Any]] = []
        total_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            while iterations < self.agent_config.max_iterations:
                iterations += 1

                response = llm.complete(
                    messages=self._history,
                    tools=tools,
                    temperature=self.agent_config.temperature,
                    max_tokens=self.agent_config.max_tokens,
                )

                # Accumulate token usage
                if response.usage:
                    for k in total_tokens:
                        total_tokens[k] += response.usage.get(k, 0)

                self._history.append(response.to_message())

                # No tool calls → final answer
                if not response.has_tool_calls:
                    self._trim_history()
                    return AgentResponse(
                        content=response.content or "",
                        tool_calls_made=tool_calls_made,
                        iterations=iterations,
                        tokens_used=total_tokens,
                    )

                # Execute each tool call
                for tc in response.tool_calls:
                    result = self.tool_executor.execute(tc, self.agent_config.tool_timeout)
                    tool_calls_made.append(
                        {
                            "name": tc.name,
                            "arguments": tc.arguments,
                            "result": result.result if not result.error else None,
                            "error": result.error,
                        }
                    )
                    result_str = result.result if result.result else f"Error: {result.error}"
                    self._history.append(
                        Message.tool_result(tc.id, tc.name, str(result_str))
                    )

            # Max iterations reached
            self._trim_history()
            return AgentResponse(
                content="Reached maximum reasoning steps.",
                tool_calls_made=tool_calls_made,
                iterations=iterations,
                tokens_used=total_tokens,
            )

        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return AgentResponse(
                content=f"Error during execution: {e}",
                tool_calls_made=tool_calls_made,
                iterations=iterations,
                tokens_used=total_tokens,
                error=str(e),
            )

    # ── Helpers ──────────────────────────────────────────────────

    def _get_llm(self) -> Optional[BaseLLMService]:
        if self._llm_service:
            return self._llm_service
        if self._llm_config:
            self._llm_service = create_llm_service(
                provider=self._llm_config.provider_type,
                api_key=self._llm_config.api_key,
                model=self._llm_config.model_name,
                api_base=self._llm_config.api_base,
            )
            return self._llm_service
        return None

    def _trim_history(self) -> None:
        max_msgs = self.agent_config.max_history_messages
        if len(self._history) <= max_msgs:
            return

        system = [m for m in self._history if m.role.value == "system"]
        others = [m for m in self._history if m.role.value != "system"]
        keep = max_msgs - len(system)
        self._history = system + others[-keep:] if keep > 0 else system

    def reset(self) -> None:
        """Clear conversation history."""
        if self.system_prompt:
            self._history = [Message.system(self.system_prompt)]
        else:
            self._history = []

    def get_history(self) -> List[Dict[str, Any]]:
        """Return conversation history as plain dicts."""
        return [msg.to_dict() for msg in self._history]

    @property
    def is_running(self) -> bool:
        return self._started
