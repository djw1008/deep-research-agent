"""Tests for BlueTeamAgent."""

import json

import pytest

from deep_research.agents.blue_agent import BlueTeamAgent
from deep_research.core.schema import (
    AgentStatus,
    AttackDimension,
    FixType,
    Issue,
    ResearchReport,
    Severity,
    SubTask,
)


class FakeLLMResponse:
    def __init__(self, content: str, tool_calls: list | None = None, usage=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = None
        self.model = "fake"
        self.usage = usage


class FakePolicy:
    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or []
        self.call_count = 0
        self.tools = None

    def chat(self, messages):
        idx = self.call_count % len(self.responses) if self.responses else 0
        self.call_count += 1
        return FakeLLMResponse(self.responses[idx])

    def set_tools(self, tools):
        self.tools = tools


@pytest.fixture
def agent():
    return BlueTeamAgent(name="blue_agent", policy=FakePolicy())


# ---------------------------------------------------------------------------
# Issue filtering / sorting
# ---------------------------------------------------------------------------

def test_filter_issues_skips_invalid(agent):
    issues = [
        Issue(
            dimension=AttackDimension.FACTUAL,
            severity=Severity.MAJOR,
            location="第1段",
            description="问题 A",
            fix_type=FixType.REMOVAL,
        ),
        Issue(
            dimension=AttackDimension.FACTUAL,
            severity=Severity.MAJOR,
            location="第2段",
            description="",
            fix_type=FixType.IN_PLACE,
        ),
        "not an issue",
    ]
    filtered = agent._filter_issues(issues)
    assert len(filtered) == 1
    assert filtered[0].description == "问题 A"


def test_sort_issues_order(agent):
    issues = [
        Issue(
            dimension=AttackDimension.FACTUAL,
            severity=Severity.MINOR,
            location="第1段",
            description="in_place minor",
            fix_type=FixType.IN_PLACE,
        ),
        Issue(
            dimension=AttackDimension.FACTUAL,
            severity=Severity.CRITICAL,
            location="第2段",
            description="removal critical",
            fix_type=FixType.REMOVAL,
        ),
        Issue(
            dimension=AttackDimension.FACTUAL,
            severity=Severity.MAJOR,
            location="第3段",
            description="search major",
            fix_type=FixType.SEARCH,
        ),
        Issue(
            dimension=AttackDimension.FACTUAL,
            severity=Severity.MAJOR,
            location="第4段",
            description="removal major",
            fix_type=FixType.REMOVAL,
        ),
    ]
    sorted_issues = agent._sort_issues(issues)
    fix_types = [i.fix_type for i in sorted_issues]
    assert fix_types == [
        FixType.REMOVAL,
        FixType.REMOVAL,
        FixType.SEARCH,
        FixType.IN_PLACE,
    ]
    assert sorted_issues[0].severity == Severity.CRITICAL
    assert sorted_issues[1].severity == Severity.MAJOR


# ---------------------------------------------------------------------------
# Apply fix parsing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_apply_fix_removal(agent):
    report = ResearchReport(
        query="AI safety",
        content="第一段。第二段：有问题的论断。第三段。",
        sources=[{"title": "Source", "url": "https://example.com", "snippet": "snippet"}],
    )
    issue = Issue(
        dimension=AttackDimension.FACTUAL,
        severity=Severity.CRITICAL,
        location="第二段",
        description="有问题的论断",
        fix_type=FixType.REMOVAL,
        evidence="无来源支撑",
    )

    agent.policy.responses = [
        json.dumps(
            {
                "content": "第一段。第三段。",
                "changes": "删除第二段有问题的论断",
            },
            ensure_ascii=False,
        )
    ]

    result = await agent._apply_fix(report, issue, "AI safety", AttackDimension.FACTUAL)
    assert "第二段" not in result["content"]
    assert result["changes"] == "删除第二段有问题的论断"


@pytest.mark.asyncio
async def test_apply_fix_in_place(agent):
    report = ResearchReport(query="AI safety", content="AI 将必然取代所有工作。")
    issue = Issue(
        dimension=AttackDimension.LOGIC,
        severity=Severity.MAJOR,
        location="第一段",
        description="绝对化表述",
        fix_type=FixType.IN_PLACE,
        evidence="无法推出必然结论",
    )

    agent.policy.responses = [
        json.dumps(
            {
                "content": "AI 可能在部分岗位替代人工，但全面取代所有工作仍存争议[待进一步核实]。",
                "changes": "降低绝对化表述并添加待核实标注",
            },
            ensure_ascii=False,
        )
    ]

    result = await agent._apply_fix(report, issue, "AI safety", AttackDimension.LOGIC)
    assert "必然" not in result["content"]
    assert "[待进一步核实]" in result["content"]


# ---------------------------------------------------------------------------
# Self-verify
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_self_verify_detects_new_issue(agent):
    agent.policy.responses = [
        json.dumps(
            {
                "has_new_issue": True,
                "new_issues": [
                    {
                        "severity": "major",
                        "description": "新增逻辑矛盾",
                        "location": "第二段",
                    }
                ],
            },
            ensure_ascii=False,
        )
    ]

    result = await agent._self_verify(
        original_content="第一段。第二段：A。",
        revised_content="第一段。第二段：非A。",
        fixes=[{"fix_type": "in_place", "description": "修改A"}],
    )
    assert result["has_new_issue"] is True
    assert len(result["new_issues"]) == 1
    assert result["new_issues"][0]["description"] == "新增逻辑矛盾"


@pytest.mark.asyncio
async def test_self_verify_no_change(agent):
    result = await agent._self_verify(
        original_content="same",
        revised_content="same",
        fixes=[],
    )
    assert result["has_new_issue"] is False
    assert result["new_issues"] == []
    assert agent.policy.call_count == 0


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_no_issues():
    agent = BlueTeamAgent(name="blue_agent", policy=FakePolicy())
    report = ResearchReport(query="AI safety", content="Test report")
    task = SubTask(id="blue_1", description="blue repair", task_type="blue_agent")
    result = await agent.run(
        task,
        {
            "report": report,
            "query": "AI safety",
            "dimension": AttackDimension.FACTUAL,
            "issues": [],
        },
    )
    assert result.status == AgentStatus.SUCCESS
    assert result.output is report
    assert result.trajectory[0]["action"] == "no_issues_to_fix"


@pytest.mark.asyncio
async def test_run_with_issues_records_self_verify():
    responses = [
        # fix removal
        json.dumps(
            {"content": "修正后的报告。", "changes": "删除错误论断"},
            ensure_ascii=False,
        ),
        # self-verify
        json.dumps(
            {
                "has_new_issue": True,
                "new_issues": [
                    {"severity": "minor", "description": "语气稍硬", "location": "第一段"}
                ],
            },
            ensure_ascii=False,
        ),
    ]
    agent = BlueTeamAgent(name="blue_agent", policy=FakePolicy(responses=responses))
    report = ResearchReport(query="AI safety", content="原报告内容。")
    issues = [
        Issue(
            dimension=AttackDimension.HALLUCINATION,
            severity=Severity.CRITICAL,
            location="第一段",
            description="无来源论断",
            fix_type=FixType.REMOVAL,
        )
    ]
    task = SubTask(id="blue_1", description="blue repair", task_type="blue_agent")
    result = await agent.run(
        task,
        {
            "report": report,
            "query": "AI safety",
            "dimension": AttackDimension.HALLUCINATION,
            "issues": issues,
        },
    )
    assert result.status == AgentStatus.SUCCESS
    assert result.output.content == "修正后的报告。"
    trajectory = result.trajectory[0]
    assert trajectory["dimension"] == "hallucination"
    assert len(trajectory["fixes"]) == 1
    assert len(trajectory["self_verify_new_issues"]) == 1


@pytest.mark.asyncio
async def test_run_invalid_report():
    agent = BlueTeamAgent(name="blue_agent", policy=FakePolicy())
    task = SubTask(id="blue_1", description="blue repair", task_type="blue_agent")
    result = await agent.run(
        task,
        {
            "report": "not a report",
            "query": "AI safety",
            "dimension": AttackDimension.FACTUAL,
            "issues": [],
        },
    )
    assert result.status == AgentStatus.FAILED


@pytest.mark.asyncio
async def test_run_invalid_dimension():
    agent = BlueTeamAgent(name="blue_agent", policy=FakePolicy())
    report = ResearchReport(query="AI safety", content="Test report")
    task = SubTask(id="blue_1", description="blue repair", task_type="blue_agent")
    result = await agent.run(
        task,
        {
            "report": report,
            "query": "AI safety",
            "dimension": "not_a_dimension",
            "issues": [],
        },
    )
    assert result.status == AgentStatus.FAILED


# ---------------------------------------------------------------------------
# Function calling for SEARCH issues
# ---------------------------------------------------------------------------

class ToolCallingFakePolicy:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0
        self.tools = None

    def set_tools(self, tools):
        self.tools = tools

    def chat(self, messages):
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp


class FakeWebSearchTool:
    name = "web_search"

    @staticmethod
    def get_schema():
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "test web search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "num_results": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
        }

    async def execute(self, query: str, num_results: int = 5):
        self.last_query = query
        self.last_num_results = num_results
        return [
            {
                "title": f"Result for {query}",
                "url": "https://example.com/source",
                "snippet": "supporting snippet",
            }
        ]


@pytest.mark.asyncio
async def test_run_search_issue_uses_function_calling():
    web_search_tool = FakeWebSearchTool()
    policy = ToolCallingFakePolicy([
        # first call: model requests web_search
        FakeLLMResponse(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query": "AI safety regulation 2024", "num_results": 3}',
                    },
                }
            ],
        ),
        # second call: model returns the fixed report
        FakeLLMResponse(
            content=json.dumps(
                {"content": "AI 监管政策仍在演进[1]。", "changes": "补充来源并软化表述"},
                ensure_ascii=False,
            )
        ),
    ])
    agent = BlueTeamAgent(
        name="blue_agent",
        policy=policy,
        tools=[web_search_tool],
    )

    report = ResearchReport(query="AI safety", content="AI 监管政策已经完善。")
    issues = [
        Issue(
            dimension=AttackDimension.SOURCE,
            severity=Severity.MAJOR,
            location="第一段",
            description="来源不足",
            fix_type=FixType.SEARCH,
            evidence="缺少权威来源",
        )
    ]
    task = SubTask(id="blue_1", description="blue repair", task_type="blue_agent")
    result = await agent.run(
        task,
        {
            "report": report,
            "query": "AI safety",
            "dimension": AttackDimension.SOURCE,
            "issues": issues,
        },
    )

    assert result.status == AgentStatus.SUCCESS
    assert result.output.content == "AI 监管政策仍在演进[1]。"
    assert policy.call_count == 2
    assert web_search_tool.last_query == "AI safety regulation 2024"
    assert web_search_tool.last_num_results == 3
    # SEARCH 完成后应清理工具 schema，避免影响后续调用
    assert policy.tools is None
