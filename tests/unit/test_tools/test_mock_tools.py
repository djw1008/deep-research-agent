"""Test mock tools integration via ResearchAgent."""

import pytest

from deep_research.agents import ResearchAgent
from deep_research.core.schema import AgentStatus, SubTask
from deep_research.models import LLMClient, LLMResponse
from deep_research.tools import ToolRegistry
from deep_research.tools.calculator import CalculatorTool
from deep_research.tools.notepad import NotepadTool
from deep_research.tools.web_search import WebSearchTool


class FakeLLMClient(LLMClient):
    def __init__(self, response_content: str = "mock analysis"):
        super().__init__(model_name="fake", base_url="http://fake", api_key="fake")
        self._response_content = response_content

    def chat(self, messages, tools=None, **kwargs):
        return LLMResponse(content=self._response_content)


@pytest.fixture
def agent():
    tools = [WebSearchTool(mock_mode=True), CalculatorTool(), NotepadTool()]
    return ResearchAgent("test", FakeLLMClient("mock llm result"), tools=tools)


@pytest.mark.asyncio
async def test_search_task(agent):
    task = SubTask(
        id="t1",
        description="AI safety research",
        task_type="search",
        search_hints=["AI safety", "alignment"],
    )
    result = await agent.run(task, {})

    assert result.status == AgentStatus.SUCCESS
    assert result.task_id == "t1"
    assert isinstance(result.output, str)
    assert len(result.trajectory) > 0


@pytest.mark.asyncio
async def test_analyze_task(agent):
    task = SubTask(id="t2", description="Analyze trends", task_type="analyze")
    result = await agent.run(task, {})

    assert result.status == AgentStatus.SUCCESS
    assert result.task_id == "t2"


@pytest.mark.asyncio
async def test_verify_task(agent):
    task = SubTask(id="t3", description="Verify claim", task_type="verify")
    result = await agent.run(task, {})

    assert result.status == AgentStatus.SUCCESS
    assert result.task_id == "t3"


@pytest.mark.asyncio
async def test_calculator_tool():
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    result = await registry.call("calculator", expression="2 + 3 * 4")
    assert result["result"] == 14.0
    assert result["error"] is None

    result = await registry.call("calculator", expression="(10 - 2) / 4")
    assert result["result"] == 2.0


@pytest.mark.asyncio
async def test_notepad_tool():
    registry = ToolRegistry()
    registry.register(NotepadTool())

    await registry.call("notepad", action="write", content="note 1", tag="todo")
    await registry.call("notepad", action="write", content="note 2", tag="todo")

    result = await registry.call("notepad", action="read", tag="todo")
    assert len(result["notes"]) == 2
    assert result["notes"][0]["content"] == "note 1"


# ---------------------------------------------------------------------------
# Multi-turn tool-calling test
# ---------------------------------------------------------------------------
class MultiTurnPolicy:
    """Fake policy that returns tool_calls on first turn, then final answer."""

    def __init__(self):
        self._turn = 0

    def chat(self, messages, tools=None):
        self._turn += 1
        if self._turn == 1:
            return LLMResponse(
                content="I will search for this.",
                tool_calls=[{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query": "AI safety"}'}
                }]
            )
        return LLMResponse(content="Confidence: 0.85. Here is the summary.")


@pytest.mark.asyncio
async def test_multi_turn_tool_calling():
    tools = [WebSearchTool(mock_mode=True)]
    agent = ResearchAgent("test", MultiTurnPolicy(), tools=tools)
    task = SubTask(id="t1", description="test", task_type="search")

    result = await agent.run(task, {})

    assert result.status == AgentStatus.SUCCESS
    assert result.confidence == 0.85
    assert len(result.trajectory) >= 2  # assistant + tool + assistant


class _FailingBrowser:
    name = "browser"

    def __init__(self):
        self.calls = 0

    @staticmethod
    def get_schema():
        return {"type": "function", "function": {"name": "browser", "parameters": {"type": "object"}}}

    async def execute(self, **kwargs):
        self.calls += 1
        return {"error": "HTTPError: 403 Forbidden", "url": kwargs.get("url", "")}


class _FailOnceThenSummarizePolicy:
    def __init__(self):
        self.turn = 0

    def set_tools(self, tools):
        pass

    def chat(self, messages, **kwargs):
        self.turn += 1
        if self.turn == 1:
            return LLMResponse(tool_calls=[{
                "id": "call_browser",
                "type": "function",
                "function": {"name": "browser", "arguments": '{"url":"https://example.com"}'},
            }])
        return LLMResponse(content="基于已有资料完成总结。置信度：0.5")


@pytest.mark.asyncio
async def test_single_tool_failure_is_one_observation_then_agent_can_summarize():
    browser = _FailingBrowser()
    agent = ResearchAgent(
        "test",
        _FailOnceThenSummarizePolicy(),
        tools=[browser],
        config={"researcher": {"max_llm_turns": 8}},
    )

    result = await agent.run(SubTask(id="t-retry", description="test", task_type="search"), {})

    assert browser.calls == 1
    assert result.status == AgentStatus.SUCCESS
    tool_event = next(item for item in result.trajectory if item.get("role") == "tool")
    assert tool_event["failed"] is True
    assert tool_event["result"]["fallback"]


class _FailTwiceThenSummarizePolicy:
    def __init__(self):
        self.turn = 0
        self.tools = None

    def set_tools(self, tools):
        self.tools = tools

    def chat(self, messages, **kwargs):
        self.turn += 1
        if self.turn <= 2:
            return LLMResponse(tool_calls=[{
                "id": f"call_browser_{self.turn}",
                "type": "function",
                "function": {
                    "name": "browser",
                    "arguments": f'{{"url":"https://example.com/{self.turn}"}}',
                },
            }])
        assert kwargs.get("tool_choice") == "none"
        return LLMResponse(content="工具均失败，基于已有信息给出有限总结。置信度：0.3")


@pytest.mark.asyncio
async def test_second_failed_tool_invocation_forces_summary_instead_of_task_failure():
    browser = _FailingBrowser()
    agent = ResearchAgent(
        "test",
        _FailTwiceThenSummarizePolicy(),
        tools=[browser],
        config={"researcher": {"max_llm_turns": 8}},
    )

    result = await agent.run(SubTask(id="t-fallback", description="test", task_type="search"), {})

    assert browser.calls == 2
    assert result.status == AgentStatus.SUCCESS
    failed_events = [
        item for item in result.trajectory
        if item.get("role") == "tool" and item.get("failed") is True
    ]
    assert len(failed_events) == 2
    assert "有限总结" in result.output
