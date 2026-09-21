"""Tool Registry — 统一注册和管理所有工具。"""

from typing import Any


class ToolRegistry:
    """
    工具注册表。

    用法:
        registry = ToolRegistry()
        registry.register(WebSearchTool())
        result = await registry.call("web_search", query="AI safety")
    """

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, tool: Any) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Any:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered")
        return self._tools[name]

    async def call(self, name: str, **kwargs) -> Any:
        tool = self.get(name)
        return await tool.execute(**kwargs)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())
