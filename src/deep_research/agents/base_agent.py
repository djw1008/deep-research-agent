"""Agent 抽象基类。"""

from abc import ABC, abstractmethod
from typing import Any

from ..core.schema import AgentResult, SubTask


class BaseAgent(ABC):
    """所有可执行 SubTask 的 Agent 必须继承此类。"""

    def __init__(self, name: str, policy, tools: list | None = None):
        self.name = name
        self.policy = policy
        self.tools = tools or []

    @abstractmethod
    async def run(self, task: SubTask, context: dict) -> AgentResult:
        """执行给定的 SubTask。"""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} tools={len(self.tools)}>"
