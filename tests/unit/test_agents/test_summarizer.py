"""Tests for SummarizerAgent."""

import pytest

from deep_research.agents import SummarizerAgent
from deep_research.core.schema import AgentResult, AgentStatus, ResearchReport, SubTask
from deep_research.models import LLMClient, LLMResponse


class FakeLLMClient(LLMClient):
    """Fake LLM that returns predefined content."""

    def __init__(self, response_content: str):
        super().__init__(model_name="fake", base_url="http://fake", api_key="fake")
        self._response_content = response_content

    def chat(self, messages, tools=None, **kwargs):
        return LLMResponse(content=self._response_content)


REPORT_CONTENT = """# Research Report: AI Safety

## Executive Summary
Artificial intelligence safety is a critical field.

## Background
AI systems are becoming more powerful.

## Key Findings
1. Alignment problem remains unsolved.
2. Interpretability is improving.

## Analysis
Detailed analysis goes here with enough length to satisfy the requirement.
More text to ensure the report is comprehensive and well-structured.
The field of AI safety encompasses technical research, governance, and ethics.

## Comparisons
Different approaches to AI safety have varying trade-offs.

## Implications
Societal implications are significant.

## Conclusion
Continued research is essential.

Overall Confidence: 0.85
"""


@pytest.mark.asyncio
async def test_empty_results_returns_failed():
    agent = SummarizerAgent(name="sum", policy=FakeLLMClient(""))
    task = SubTask(id="synthesize", description="合成报告", task_type="synthesize")
    result = await agent.run(task, {"query": "test", "results": []})

    assert result.status == AgentStatus.FAILED
    assert isinstance(result.output, ResearchReport)
    assert result.output.content == "无可用子任务结果进行合成。"


@pytest.mark.asyncio
async def test_synthesize_success():
    client = FakeLLMClient(REPORT_CONTENT)
    agent = SummarizerAgent(name="sum", policy=client)

    results = [
        AgentResult(
            task_id="t1",
            status=AgentStatus.SUCCESS,
            output="Found that AI alignment is hard.",
            confidence=0.9,
            trajectory=[
                {"role": "tool", "result": {"results": [{"url": "https://example.com/1", "title": "AI Alignment", "snippet": "Alignment is hard"}]}},
            ],
        ),
        AgentResult(
            task_id="t2",
            status=AgentStatus.SUCCESS,
            output="Interpretability is improving.",
            confidence=0.8,
            trajectory=[
                {"role": "tool", "result": {"papers": [{"pdf_url": "https://arxiv.org/abs/1234", "title": "Interp Paper", "summary": "We show improvements"}]}},
            ],
        ),
    ]

    task = SubTask(id="synthesize", description="合成报告", task_type="synthesize")
    result = await agent.run(task, {"query": "AI safety", "results": results})

    assert result.status == AgentStatus.SUCCESS
    assert isinstance(result.output, ResearchReport)
    report: ResearchReport = result.output

    assert report.query == "AI safety"
    assert "AI Safety" in report.content
    assert report.confidence > 0
    # 2 successes out of 2 → success_rate=1.0 → confidence = 0.85 * 1.0 = 0.85
    assert report.confidence == 0.85
    assert len(report.sources) == 2
    assert report.num_searches == 2

    # Verify source deduplication: same URL should not appear twice
    urls = [s["url"] for s in report.sources]
    assert len(urls) == len(set(urls))


@pytest.mark.asyncio
async def test_synthesize_llm_error():
    class BadClient(LLMClient):
        def __init__(self):
            super().__init__(model_name="bad", base_url="http://bad", api_key="bad")

        def chat(self, messages, tools=None, **kwargs):
            raise RuntimeError("LLM down")

    agent = SummarizerAgent(name="sum", policy=BadClient())
    task = SubTask(id="synthesize", description="合成报告", task_type="synthesize")
    result = await agent.run(
        task,
        {
            "query": "test",
            "results": [
                AgentResult(task_id="t1", status=AgentStatus.SUCCESS, output="ok", confidence=0.5)
            ],
        },
    )

    assert result.status == AgentStatus.FAILED
    assert "LLM down" in result.output


@pytest.mark.asyncio
async def test_confidence_with_partial_failures():
    """混合置信度 = LLM 自评 × sqrt(成功率)。"""
    client = FakeLLMClient("Report.\n\nOverall Confidence: 0.80")
    agent = SummarizerAgent(name="sum", policy=client)

    # 3 tasks, 1 failed → success_rate = 2/3 ≈ 0.667
    results = [
        AgentResult(task_id="t1", status=AgentStatus.SUCCESS, output="a", confidence=0.9),
        AgentResult(task_id="t2", status=AgentStatus.SUCCESS, output="b", confidence=0.8),
        AgentResult(task_id="t3", status=AgentStatus.FAILED, output="c", confidence=0.0),
    ]

    task = SubTask(id="synthesize", description="合成报告", task_type="synthesize")
    result = await agent.run(task, {"query": "test", "results": results})

    assert result.status == AgentStatus.SUCCESS
    report: ResearchReport = result.output
    expected = round(0.80 * ((2 / 3) ** 0.5), 2)
    assert report.confidence == expected


@pytest.mark.asyncio
async def test_failed_results_included_in_prompt():
    """失败结果应出现在 prompt 中（标记 ✗），但不应被提取为 source。"""
    client = FakeLLMClient("Report.\n\nOverall Confidence: 0.70")
    agent = SummarizerAgent(name="sum", policy=client)

    results = [
        AgentResult(task_id="t1", status=AgentStatus.FAILED, output="search failed", confidence=0.0),
    ]

    task = SubTask(id="synthesize", description="合成报告", task_type="synthesize")
    result = await agent.run(task, {"query": "test", "results": results})

    assert result.status == AgentStatus.SUCCESS
    report: ResearchReport = result.output
    # 失败结果不会被提取为 source
    assert len(report.sources) == 0
    # confidence: 0.70 * sqrt(0/1) = 0
    assert report.confidence == 0.0
