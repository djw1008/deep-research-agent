"""Agent 生命周期管理 (AgentPool)

负责 Worker Agent 的创建、复用、按 task_type 路由。
采用对象池模式减少重复创建开销。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_agent import BaseAgent


class AgentPool:
    """Agent 对象池。

    设计要点:
      - 延迟创建：首次请求某类型 Agent 时才实例化
      - 复用策略：同类型 Agent 释放后回到池中复用
      - 按类型路由：SEARCH/ANALYZE/VERIFY 映射到不同 Agent 实现
    """

    def __init__(
        self,
        policy_factory,
        tools_factory=None,
        max_idle: int = 3,
        agent_factory=None,
        config: dict | None = None,
        session_memory=None,
        knowledge_base=None,
        event_sink=None,
    ):
        self.policy_factory = policy_factory
        self.tools_factory = tools_factory
        self.max_idle = max(max_idle, 1)
        self.agent_factory = agent_factory
        self.config = config or {}
        self.session_memory = session_memory
        self.knowledge_base = knowledge_base
        self.event_sink = event_sink

        # 类型 -> 空闲 Agent 列表
        self._idle: dict[str, list[BaseAgent]] = {}
        # 类型 -> 活跃计数
        self._active_count: dict[str, int] = {}

    async def get_agent(self, task_type: str) -> BaseAgent:
        """根据任务类型获取可用的 Agent 实例。优先复用，无空闲则新建。"""
        type_key = task_type

        if type_key not in self._idle:
            self._idle[type_key] = []
            self._active_count[type_key] = 0

        # 尝试复用
        while self._idle[type_key]:
            agent = self._idle[type_key].pop()
            self._active_count[type_key] += 1
            return agent

        # 新建
        agent = self._create_agent(type_key)
        self._active_count[type_key] += 1
        return agent

    async def release_agent(self, agent: BaseAgent) -> None:
        """释放 Agent 回对象池。"""
        if agent is None:
            return

        type_key = self._infer_type_key(agent)
        self._active_count[type_key] = max(0, self._active_count.get(type_key, 0) - 1)

        idle_list = self._idle.setdefault(type_key, [])
        if len(idle_list) < self.max_idle:
            idle_list.append(agent)

    def _create_agent(self, type_key: str) -> BaseAgent:
        """根据类型键创建对应的 Agent 实例。"""
        if self.agent_factory is not None:
            return self.agent_factory(type_key)
        # 兼容：policy_factory 可接受 type_key 参数（按模块配置采样参数）
        try:
            policy = self.policy_factory(type_key)
        except TypeError:
            policy = self.policy_factory()
        tools = self.tools_factory() if self.tools_factory else []

        from .blue_agent import BlueTeamAgent
        from .red_agent import RedTeamAgent
        from .researcher import ResearchAgent
        from .summarizer import SummarizerAgent

        if type_key in ("search", "analyze", "verify"):
            return ResearchAgent(
                name=f"researcher_{type_key}",
                policy=policy,
                tools=tools,
                config=self.config,
                knowledge_base=self.knowledge_base,
                event_sink=self.event_sink,
            )
        if type_key == "synthesize":
            return SummarizerAgent(
                name="summarizer",
                policy=policy,
                tools=tools,
                session_memory=self.session_memory,
            )
        if type_key == "red_agent":
            return RedTeamAgent(name="red_agent", policy=policy, config=self.config)
        if type_key == "blue_agent":
            return BlueTeamAgent(name="blue_agent", policy=policy, tools=tools, config=self.config)
        # 默认降级
        return ResearchAgent(
            name=f"researcher_default",
            policy=policy,
            tools=tools,
            config=self.config,
            knowledge_base=self.knowledge_base,
            event_sink=self.event_sink,
        )

    async def execute(self, task, context):
        """便捷方法：获取 Agent，运行任务，释放 Agent。"""
        agent = await self.get_agent(task.task_type)
        try:
            return await agent.run(task, context)
        finally:
            await self.release_agent(agent)

    def _infer_type_key(self, agent: BaseAgent) -> str:
        """从 Agent 实例推断其类型键。"""
        cls_name = agent.__class__.__name__
        if "Summarizer" in cls_name:
            return "synthesize"
        if "RedTeam" in cls_name:
            return "red_agent"
        if "BlueTeam" in cls_name:
            return "blue_agent"
        return "search"
