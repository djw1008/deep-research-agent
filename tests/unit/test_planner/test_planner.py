"""Tests for Planner."""

import pytest

from deep_research.core.schema import SubTask
from deep_research.models import LLMClient, LLMResponse
from deep_research.planner import DAG, PlanParseError, Planner


class FakeLLMClient(LLMClient):
    """用于测试的伪造 LLM 客户端。"""

    def __init__(self, response_content: str):
        super().__init__(
            model_name="fake",
            base_url="http://fake",
            api_key="fake",
        )
        self._response_content = response_content

    def chat(self, messages, tools=None, **kwargs):
        return LLMResponse(content=self._response_content)


# ---------------------------------------------------------------------------
# _parse_plan
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_parse_plan_clean_json():
    client = FakeLLMClient('{"sub_tasks": [{"task_id": "t1", "task_type": "search", "description": "desc1", "dependencies": []}]}')
    planner = Planner(client)
    dag, tasks = planner._parse_plan(client._response_content)

    assert isinstance(dag, DAG)
    assert len(dag) == 1
    assert len(tasks) == 1
    assert tasks[0].id == "t1"
    assert tasks[0].task_type == "search"


@pytest.mark.asyncio
async def test_parse_plan_with_markdown_code_block():
    raw = '```json\n{"sub_tasks": [{"task_id": "t1", "task_type": "search", "description": "d", "dependencies": []}]}\n```'
    client = FakeLLMClient(raw)
    planner = Planner(client)
    dag, tasks = planner._parse_plan(raw)

    assert len(tasks) == 1
    assert tasks[0].id == "t1"


@pytest.mark.asyncio
async def test_parse_plan_with_dependencies():
    raw = '{"sub_tasks": [{"task_id": "t1", "task_type": "search", "description": "d1", "dependencies": []}, {"task_id": "t2", "task_type": "analyze", "description": "d2", "dependencies": ["t1"]}]}'
    client = FakeLLMClient(raw)
    planner = Planner(client)
    dag, tasks = planner._parse_plan(raw)

    assert len(tasks) == 2
    assert dag.get_dependencies("t2") == ["t1"]
    order = dag.topological_sort()
    assert order.index("t1") < order.index("t2")


@pytest.mark.asyncio
async def test_parse_plan_cycle_raises():
    raw = '{"sub_tasks": [{"task_id": "t1", "task_type": "search", "description": "d1", "dependencies": ["t2"]}, {"task_id": "t2", "task_type": "search", "description": "d2", "dependencies": ["t1"]}]}'
    client = FakeLLMClient(raw)
    planner = Planner(client)

    with pytest.raises(PlanParseError):
        planner._parse_plan(raw)


@pytest.mark.asyncio
async def test_parse_plan_invalid_json_raises():
    client = FakeLLMClient("this is not json")
    planner = Planner(client)

    with pytest.raises(PlanParseError):
        planner._parse_plan(client._response_content)


# ---------------------------------------------------------------------------
# generate_plan
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_generate_plan_success():
    raw = '{"sub_tasks": [{"task_id": "t1", "task_type": "search", "description": "how to find internship", "dependencies": [], "search_hints": ["internship"]}]}'
    client = FakeLLMClient(raw)
    planner = Planner(client)
    dag, tasks = await planner.generate_plan("how to find internship")

    assert len(tasks) == 1
    assert tasks[0].search_hints == ["internship"]


@pytest.mark.asyncio
async def test_generate_plan_llm_error():
    client = FakeLLMClient("Error: network timeout")
    planner = Planner(client)

    with pytest.raises(PlanParseError):
        await planner.generate_plan("test")


# ---------------------------------------------------------------------------
# replan
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_replan_success():
    raw = '{"sub_tasks": [{"task_id": "t1_new", "task_type": "search", "description": "fixed", "dependencies": []}]}'
    client = FakeLLMClient(raw)
    planner = Planner(client)

    failed = [SubTask(id="t1", description="old", task_type="search")]
    preserved = {"t1": "some result"}
    dag, tasks = await planner.replan("test", failed, preserved, "timeout")

    assert len(tasks) == 1
    assert tasks[0].id == "t1_new"
