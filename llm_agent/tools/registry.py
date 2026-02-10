"""
Tool registry – stores tool definitions and their handlers.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from ..llm.base import Tool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for managing tool definitions and handlers.

    Example::

        registry = ToolRegistry()
        registry.register_function(
            name="get_weather",
            description="Get current weather for a location",
            function=get_weather,
            parameters={
                "location": {"type": "string", "required": True},
            },
        )
        tools = registry.get_tools()
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[str, List[str]] = {}

    # ── Registration ─────────────────────────────────────────────

    def register(self, tool: Tool, category: str = "general") -> None:
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")
        self._tools[tool.name] = tool
        self._categories.setdefault(category, [])
        if tool.name not in self._categories[category]:
            self._categories[category].append(tool.name)

    def register_function(
        self,
        name: str,
        description: str,
        function: Callable,
        parameters: Dict[str, Any] = None,
        category: str = "general",
    ) -> None:
        """Register a plain function as a tool.

        *parameters* uses a simplified format::

            {
                "location": {"type": "string", "description": "City", "required": True},
                "units": {"type": "string", "default": "celsius"},
            }
        """
        properties: Dict[str, Any] = {}
        required: List[str] = []

        if parameters:
            for pname, pdef in parameters.items():
                prop: Dict[str, Any] = {"type": pdef.get("type", "string")}
                if "description" in pdef:
                    prop["description"] = pdef["description"]
                if "default" in pdef:
                    prop["default"] = pdef["default"]
                if "enum" in pdef:
                    prop["enum"] = pdef["enum"]
                properties[pname] = prop
                if pdef.get("required", False):
                    required.append(pname)

        schema: Dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required

        self.register(
            Tool(name=name, description=description, parameters=schema, handler=function),
            category,
        )

    def unregister(self, name: str) -> bool:
        if name not in self._tools:
            return False
        del self._tools[name]
        for cat_tools in self._categories.values():
            if name in cat_tools:
                cat_tools.remove(name)
        return True

    # ── Lookup ───────────────────────────────────────────────────

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def get_tools(self, categories: List[str] = None) -> List[Tool]:
        if categories is None:
            return list(self._tools.values())
        tools = []
        for cat in categories:
            for name in self._categories.get(cat, []):
                if name in self._tools:
                    tools.append(self._tools[name])
        return tools

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    # ── Execution ────────────────────────────────────────────────

    def execute(self, name: str, arguments: Dict[str, Any]) -> Any:
        tool = self.get(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        if not tool.handler:
            raise ValueError(f"Tool '{name}' has no handler")
        return tool.handler(**arguments)

    def clear(self) -> None:
        self._tools.clear()
        self._categories.clear()

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
