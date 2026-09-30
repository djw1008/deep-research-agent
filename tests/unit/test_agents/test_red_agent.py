"""Tests for RedTeamAgent and adversarial data models."""

import json

import pytest

from deep_research.agents.red_agent import RedTeamAgent
from deep_research.core.schema import (
    AttackDimension,
    DimensionAttack,
    FixType,
    Issue,
    RedAttackResult,
    ResearchReport,
    Severity,
    SubTask,
)


class FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = []
        self.reasoning_content = None
        self.model = "fake"
        self.usage = None


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


# ---------------------------------------------------------------------------
# RedAttackResult tests
# ---------------------------------------------------------------------------

def test_compute_overall_score():
    dimension_attacks = {
        AttackDimension.SOURCE: DimensionAttack(
            dimension=AttackDimension.SOURCE, dimension_score=10.0
        ),
        AttackDimension.FACTUAL: DimensionAttack(
            dimension=AttackDimension.FACTUAL, dimension_score=5.0
        ),
        AttackDimension.LOGIC: DimensionAttack(
            dimension=AttackDimension.LOGIC, dimension_score=5.0
        ),
        AttackDimension.COVERAGE: DimensionAttack(
            dimension=AttackDimension.COVERAGE, dimension_score=5.0
        ),
        AttackDimension.HALLUCINATION: DimensionAttack(
            dimension=AttackDimension.HALLUCINATION, dimension_score=10.0
        ),
    }
    result = RedAttackResult(
        round_no=1,
        dimension_attacks=dimension_attacks,
    )
    # (10*0.25 + 5*0.20 + 5*0.20 + 5*0.20 + 10*0.15) / 1.0 = 7.0
    assert result.compute_overall_score() == 7.0


def test_outstanding_issues_filtering():
    dimension_attacks = {
        AttackDimension.FACTUAL: DimensionAttack(
            dimension=AttackDimension.FACTUAL,
            issues=[
                Issue(
                    dimension=AttackDimension.FACTUAL,
                    severity=Severity.CRITICAL,
                    location="第1段",
                    description="核心数据错误",
                    fix_type=FixType.REMOVAL,
                ),
                Issue(
                    dimension=AttackDimension.FACTUAL,
                    severity=Severity.MINOR,
                    location="第2段",
                    description="小笔误",
                    fix_type=FixType.IN_PLACE,
                ),
            ],
        ),
        AttackDimension.LOGIC: DimensionAttack(
            dimension=AttackDimension.LOGIC,
            issues=[
                Issue(
                    dimension=AttackDimension.LOGIC,
                    severity=Severity.MAJOR,
                    location="第3段",
                    description="因果谬误",
                    fix_type=FixType.IN_PLACE,
                ),
            ],
        ),
    }
    result = RedAttackResult(round_no=1, dimension_attacks=dimension_attacks)
    outstanding = result.outstanding_issues(Severity.MAJOR)
    assert len(outstanding) == 2
    assert all(i.severity in (Severity.CRITICAL, Severity.MAJOR) for i in outstanding)


# ---------------------------------------------------------------------------
# RedTeamAgent parsing tests
# ---------------------------------------------------------------------------

@pytest.fixture
def agent():
    return RedTeamAgent(name="red_agent", policy=FakePolicy())


def test_extract_json_from_code_block(agent):
    text = '```json\n{"score": 7.5, "issues": []}\n```'
    raw = agent._extract_json(text)
    assert raw == '{"score": 7.5, "issues": []}'


def test_extract_json_from_plain_text(agent):
    text = 'Some text\n{"score": 7.5, "issues": []}\nMore text'
    raw = agent._extract_json(text)
    assert raw == '{"score": 7.5, "issues": []}'


def test_extract_json_prefers_first_valid_object(agent):
    text = 'prefix {"score": 1, "issues": []} middle {"score": 2} suffix'
    raw = agent._extract_json(text)
    assert raw == '{"score": 1, "issues": []}'


def test_format_sources(agent):
    sources = [
        {"title": "Example", "url": "https://example.com", "snippet": "A snippet"},
        {"title": "", "url": "https://test.com", "snippet": ""},
    ]
    text = agent._format_sources(sources)
    assert "Example" in text
    assert "https://example.com" in text
    assert "A snippet" in text
    assert "https://test.com" in text


def test_format_sources_deduplicates_and_ignores_empty_urls(agent):
    sources = [
        {"title": "First", "url": "https://example.com/a", "snippet": "one"},
        {"title": "Duplicate", "url": "https://example.com/a", "snippet": "two"},
        {"title": "No URL", "url": "", "snippet": "three"},
        {"title": "Second", "url": "https://example.com/b", "snippet": "four"},
    ]

    text = agent._format_sources(sources)

    assert text.count("https://example.com/a") == 1
    assert text.count("https://example.com/b") == 1
    assert "Duplicate" not in text
    assert "No URL" not in text


def test_parse_dimension_json(agent):
    json_text = json.dumps(
        {
            "score": 6.5,
            "issues": [
                {
                    "severity": "major",
                    "description": "数据错误",
                    "location": "第3段",
                    "fix_type": "removal",
                    "evidence": "原文：2023年为2024年",
                }
            ],
        },
        ensure_ascii=False,
    )
    da = agent._parse_dimension_json(json_text, AttackDimension.FACTUAL)
    assert da.dimension == AttackDimension.FACTUAL
    assert da.dimension_score == 6.5
    assert len(da.issues) == 1
    issue = da.issues[0]
    assert issue.severity == Severity.MAJOR
    assert issue.fix_type == FixType.REMOVAL
    assert issue.description == "数据错误"


def test_parse_dimension_json_invalid_returns_empty(agent):
    da = agent._parse_dimension_json("not json", AttackDimension.LOGIC)
    assert da.dimension_score == 0.0
    assert len(da.issues) == 0


def test_parse_dimension_json_with_trailing_comma(agent):
    text = '{"score": 5.5, "issues": [{"severity": "major", "description": "错误", "location": "第1段", "fix_type": "in_place", "evidence": "原文",},],}'
    da = agent._parse_dimension_json(text, AttackDimension.FACTUAL)
    assert da.dimension_score == 5.5
    assert len(da.issues) == 1
    assert da.issues[0].severity == Severity.MAJOR


def test_parse_dimension_json_with_comments(agent):
    text = '''{"score": 4.0, // 评分
    "issues": [
        {"severity": "critical", "description": "幻觉", "location": "第2段", "fix_type": "removal", "evidence": "无来源"} // 关键问题
    ]}'''
    da = agent._parse_dimension_json(text, AttackDimension.HALLUCINATION)
    assert da.dimension_score == 4.0
    assert len(da.issues) == 1
    assert da.issues[0].severity == Severity.CRITICAL


def test_parse_dimension_json_with_markdown_and_extra_text(agent):
    text = '以下是评分结果：\n```json\n{"score": 3.0, "issues": []}\n```\n请继续下一个维度。'
    da = agent._parse_dimension_json(text, AttackDimension.LOGIC)
    assert da.dimension_score == 3.0
    assert len(da.issues) == 0


def test_parse_dimension_json_with_chinese_punctuation(agent):
    text = '｛"score"：6.0，"issues"：[｛"severity"："minor"，"description"："小问题"，"location"："第1段"，"fix_type"："in_place"，"evidence"："原文"｝]｝'
    da = agent._parse_dimension_json(text, AttackDimension.SOURCE)
    assert da.dimension_score == 6.0
    assert len(da.issues) == 1
    assert da.issues[0].severity == Severity.MINOR


def test_repair_json_returns_none_for_unfixable(agent):
    assert agent._repair_json("not json at all") is None


# ---------------------------------------------------------------------------
# RedTeamAgent.run tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_success_with_mock_policy():
    # 5 个维度的 mock 响应
    responses = [
        json.dumps({"score": 6.0, "issues": []}),
        json.dumps({"score": 7.0, "issues": []}),
        json.dumps({"score": 5.0, "issues": []}),
        json.dumps({"score": 8.0, "issues": []}),
        json.dumps({"score": 6.5, "issues": []}),
    ]
    policy = FakePolicy(responses=responses)
    agent = RedTeamAgent(name="red_agent", policy=policy)

    report = ResearchReport(query="AI safety", content="This is a report.")
    task = SubTask(id="red_1", description="red attack", task_type="red_agent")
    result = await agent.run(task, {"report": report, "query": "AI safety", "round_no": 1})

    assert result.status.value == "success"
    red_result = result.output
    assert isinstance(red_result, RedAttackResult)
    assert red_result.round_no == 1
    assert len(red_result.dimension_attacks) == 5
    # factual=6.0*0.20 + hallucination=7.0*0.15 + logic=5.0*0.20 +
    # source=8.0*0.25 + coverage=6.5*0.20 = 6.55
    assert red_result.overall_score == pytest.approx(6.55, rel=1e-2)
    # Red Agent 应禁用工具
    assert policy.tools is None


@pytest.mark.asyncio
async def test_run_returns_failed_without_report():
    agent = RedTeamAgent(name="red_agent", policy=FakePolicy())
    task = SubTask(id="red_1", description="red attack", task_type="red_agent")
    result = await agent.run(task, {"query": "AI safety"})
    assert result.status.value == "failed"


@pytest.mark.asyncio
async def test_run_parses_issues():
    responses = [
        json.dumps(
            {
                "score": 4.0,
                "issues": [
                    {
                        "severity": "critical",
                        "description": "事实错误",
                        "location": "第1段",
                        "fix_type": "removal",
                        "evidence": "来源中没有该数据",
                    }
                ],
            }
        ),
        json.dumps({"score": 7.0, "issues": []}),
        json.dumps({"score": 5.0, "issues": []}),
        json.dumps({"score": 8.0, "issues": []}),
        json.dumps({"score": 6.5, "issues": []}),
    ]
    policy = FakePolicy(responses=responses)
    agent = RedTeamAgent(name="red_agent", policy=policy)

    report = ResearchReport(
        query="AI safety",
        content="Report content",
        sources=[{"title": "Source", "url": "https://example.com", "snippet": "Snippet"}],
    )
    task = SubTask(id="red_1", description="red attack", task_type="red_agent")
    result = await agent.run(task, {"report": report, "query": "AI safety", "round_no": 1})

    red_result = result.output
    factual = red_result.dimension_attacks[AttackDimension.FACTUAL]
    assert len(factual.issues) == 1
    assert factual.issues[0].severity == Severity.CRITICAL
    assert factual.issues[0].fix_type == FixType.REMOVAL
