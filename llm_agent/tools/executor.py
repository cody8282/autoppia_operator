"""
Tool executor – runs tool calls with timeout and error handling.
"""

import json
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional

from ..llm.base import ToolCall, ToolResult
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Executes tool calls safely with timeout protection and error handling.

    Example::

        executor = ToolExecutor(registry)
        result = executor.execute(tool_call)
    """

    def __init__(
        self,
        registry: ToolRegistry,
        default_timeout: float = 30.0,
        max_workers: int = 5,
    ):
        self.registry = registry
        self.default_timeout = default_timeout
        self.max_workers = max_workers

    def execute(
        self,
        tool_call: ToolCall,
        timeout: Optional[float] = None,
    ) -> ToolResult:
        timeout = timeout or self.default_timeout

        if not self.registry.has_tool(tool_call.name):
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=f"Tool not found: {tool_call.name}",
            )

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    self.registry.execute, tool_call.name, tool_call.arguments
                )
                raw = future.result(timeout=timeout)

            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=self._format(raw),
            )

        except FuturesTimeoutError:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=f"Tool execution timed out after {timeout}s",
            )
        except Exception as e:
            logger.debug(traceback.format_exc())
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=None,
                error=f"{type(e).__name__}: {e}",
            )

    def execute_many(
        self,
        tool_calls: List[ToolCall],
        timeout: Optional[float] = None,
        parallel: bool = True,
    ) -> List[ToolResult]:
        if not tool_calls:
            return []
        if not parallel or len(tool_calls) == 1:
            return [self.execute(tc, timeout) for tc in tool_calls]

        results: List[Optional[ToolResult]] = [None] * len(tool_calls)
        with ThreadPoolExecutor(max_workers=min(len(tool_calls), self.max_workers)) as pool:
            futures = {pool.submit(self.execute, tc, timeout): i for i, tc in enumerate(tool_calls)}
            for future in futures:
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    tc = tool_calls[idx]
                    results[idx] = ToolResult(
                        tool_call_id=tc.id, name=tc.name, result=None, error=str(e)
                    )
        return results  # type: ignore[return-value]

    @staticmethod
    def _format(result: Any) -> str:
        if result is None:
            return "null"
        if isinstance(result, str):
            return result
        if isinstance(result, (dict, list)):
            try:
                return json.dumps(result, indent=2, default=str)
            except (TypeError, ValueError):
                return str(result)
        return str(result)
