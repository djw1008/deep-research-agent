"""Integration test: PLANNING + DISPATCHING with AgentPool."""

import asyncio

import pytest

from deep_research.agents import AgentPool, ResearchAgent
from deep_research.core import HandlerResult, Orchestrator, ResearchContext, WorkflowState
from deep_research.models import LLMClient, LLMResponse
from deep_research.planner import DAG, Planner


class FakeLLMClient(LLMClient):
    """Fake LLM that returns a predefined plan."""

    def __init__(self, response_content: str):
        super().__init__(model_name="fake", base_url="http://fake", api_key="fake")
        self._response_content = response_content

    def chat(self, messages, tools=None, **kwargs):
        return LLMResponse(content=self._response_content)


class FakeResearchAgent(ResearchAgent):
    """Fake agent that returns task.id as result."""

    def __init__(self, client):
        super().__init__(name="fake", policy=client, tools=[])

    async def run(self, task, context):
        await asyncio.sleep(0.001)
        from deep_research.core.schema import AgentResult, AgentStatus
        return AgentResult(
            task_id=task.id,
            status=AgentStatus.SUCCESS,
            output=f"result_for_{task.id}",
        )


PLAN_JSON = """{
  "sub_tasks": [
    {"task_id": "t1", "task_type": "search", "description": "搜索A", "dependencies": [], "search_hints": ["A"]},
    {"task_id": "t2", "task_type": "search", "description": "搜索B", "dependencies": [], "search_hints": ["B"]},
    {"task_id": "t3", "task_type": "analyze", "description": "分析C", "dependencies": ["t1", "t2"], "search_hints": []}
  ]
}"""


@pytest.mark.asyncio
async def test_planning_and_dispatching_handler():
    """Full flow: PLANNING generates DAG, DISPATCHING runs tasks via AgentPool."""

    orch = Orchestrator()
    ctx = ResearchContext(topic="test topic")

    # ---------- PLANNING handler ----------
    async def planning_handler(orch, ctx):
        client = FakeLLMClient(PLAN_JSON)
        planner = Planner(client)
        dag, subtasks = await planner.generate_plan(ctx.topic)
        ctx.plan = dag.to_dict()
        ctx.subtasks = subtasks
        return HandlerResult(status="success")

    # ---------- DISPATCHING handler ----------
    async def dispatching_handler(orch, ctx):
        # Rebuild DAG from subtasks
        dag = DAG()
        for task in ctx.subtasks:
            dag.add_node(task.id)
        for task in ctx.subtasks:
            for dep in task.dependencies:
                dag.add_edge(dep, task.id)

        # AgentPool with fake agents
        pool = AgentPool(
            policy_factory=lambda: FakeLLMClient(""),
            tools_factory=lambda: [],
            agent_factory=lambda _type_key: FakeResearchAgent(FakeLLMClient("")),
        )

        all_results = {}
        groups = dag.get_parallel_groups()

        for group in groups:
            tasks = [t for t in ctx.subtasks if t.id in group]
            coros = [pool.execute(t, {}) for t in tasks]
            results = await asyncio.gather(*coros)
            for t, r in zip(tasks, results):
                all_results[t.id] = r.output if hasattr(r, "output") else r

        ctx.metadata["results"] = all_results
        return HandlerResult(status="success")

    # ---------- COLLECTING handler ----------
    async def collecting_handler(orch, ctx):
        # 检查是否有超过半数失败
        results = ctx.metadata.get("results", {})
        failed = [k for k, v in results.items() if isinstance(v, Exception)]
        if len(failed) > len(ctx.subtasks) / 2:
            return HandlerResult(status="partial_failure", message="majority failed")
        return HandlerResult(status="success")

    # ---------- SYNTHESIZING handler ----------
    async def synthesizing_handler(orch, ctx):
        return HandlerResult(status="success")

    # Register handlers
    orch.on_state(WorkflowState.PLANNING, planning_handler)
    orch.on_state(WorkflowState.DISPATCHING, dispatching_handler)
    orch.on_state(WorkflowState.COLLECTING, collecting_handler)
    orch.on_state(WorkflowState.SYNTHESIZING, synthesizing_handler)

    # Run
    final = await orch.run(ctx)

    assert final == WorkflowState.DONE
    assert len(ctx.subtasks) == 3
    assert ctx.metadata["results"] == {
        "t1": "result_for_t1",
        "t2": "result_for_t2",
        "t3": "result_for_t3",
    }

    # Verify execution order via DAG groups
    dag = DAG()
    for task in ctx.subtasks:
        dag.add_node(task.id)
    for task in ctx.subtasks:
        for dep in task.dependencies:
            dag.add_edge(dep, task.id)
    groups = dag.get_parallel_groups()
    assert groups[0] == ["t1", "t2"]  # Layer 1: parallel
    assert groups[1] == ["t3"]        # Layer 2: depends on t1, t2
