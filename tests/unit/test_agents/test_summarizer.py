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
1. Alignment problem remains unsolved [1].
2. Interpretability is improving [2].

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
            output="Found that AI alignment is hard [SRC-1].",
            confidence=0.9,
            trajectory=[{"role": "tool", "result": {}}],
            metadata={"sources": [{
                "url": "https://example.com/1",
                "title": "AI Alignment",
                "snippet": "Alignment is hard",
                "source_label": "SRC-1",
            }]},
        ),
        AgentResult(
            task_id="t2",
            status=AgentStatus.SUCCESS,
            output="Interpretability is improving [SRC-1].",
            confidence=0.8,
            trajectory=[{"role": "tool", "result": {}}],
            metadata={"sources": [{
                "url": "https://arxiv.org/abs/1234",
                "title": "Interp Paper",
                "snippet": "We show improvements",
                "source_label": "SRC-1",
            }]},
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
async def test_synthesizer_strips_llm_generated_reference_section():
    content = """# 报告

## 结论
正文结论。

## 引用来源
1. 模型自行生成的来源

Overall Confidence: 0.80
"""
    agent = SummarizerAgent(name="sum", policy=FakeLLMClient(content))
    task = SubTask(id="synthesize", description="合成报告", task_type="synthesize")
    source_result = AgentResult(
        task_id="t1",
        status=AgentStatus.SUCCESS,
        output="材料",
        confidence=1.0,
        trajectory=[],
    )

    result = await agent.run(task, {"query": "test", "results": [source_result]})

    assert "正文结论" in result.output.content
    assert "引用来源" not in result.output.content
    assert "模型自行生成的来源" not in result.output.content


@pytest.mark.asyncio
async def test_synthesizer_keeps_valid_citation_and_removes_dangling_one():
    content = """# 报告

## 结论
有效引用 [1]，无效引用 [99]。

Overall Confidence: 0.80
"""
    agent = SummarizerAgent(name="sum", policy=FakeLLMClient(content))
    task = SubTask(id="synthesize", description="合成报告", task_type="synthesize")
    source_result = AgentResult(
        task_id="t1",
        status=AgentStatus.SUCCESS,
        output="材料 [SRC-1]",
        confidence=1.0,
        metadata={"sources": [
            {
                "title": "Source",
                "url": "https://example.com/source",
                "snippet": "evidence",
                "source_label": "SRC-1",
            }
        ]},
    )

    result = await agent.run(task, {"query": "test", "results": [source_result]})

    assert "有效引用 [1]" in result.output.content
    assert "[99]" not in result.output.content
    assert result.output.sources[0]["citation_id"] == 1


def test_synthesis_prompt_binds_material_to_registered_source_number():
    agent = SummarizerAgent(name="sum", policy=FakeLLMClient(""))
    source_result = AgentResult(
        task_id="t1",
        status=AgentStatus.SUCCESS,
        output="该来源支持结论 [SRC-1]。",
        confidence=0.9,
        metadata={"sources": [
            {
                "title": "Primary Source",
                "url": "https://example.com/primary",
                "snippet": "supporting evidence",
                "source_label": "SRC-1",
            }
        ]},
    )
    sources = agent._collect_sources([source_result])

    prompt = agent._build_synthesis_prompt("test", [source_result], sources)

    assert "该材料关联的可用引用：[1]" in prompt
    assert "[1] Primary Source" in prompt
    assert "不得创造编号" in prompt
    assert "摘要: supporting evidence" not in prompt


def test_uncited_researcher_url_is_not_registered():
    agent = SummarizerAgent(name="sum", policy=FakeLLMClient(""))
    source_result = AgentResult(
        task_id="t1",
        status=AgentStatus.SUCCESS,
        output="Only the first source is used [SRC-1].",
        confidence=0.9,
        metadata={"sources": [
            {"title": "Used", "url": "https://example.com/used", "source_label": "SRC-1"},
            {"title": "Unused", "url": "https://example.com/unused", "source_label": "SRC-2"},
        ]},
    )

    sources = agent._collect_sources([source_result])

    assert [source["url"] for source in sources] == ["https://example.com/used"]


def test_legacy_malformed_citation_is_registered_and_replaced():
    agent = SummarizerAgent(name="sum", policy=FakeLLMClient(""))
    source_result = AgentResult(
        task_id="t1",
        status=AgentStatus.SUCCESS,
        output="Legacy recalled result [SRC-13 摘要].",
        metadata={"sources": [{
            "title": "Legacy source",
            "url": "https://example.com/legacy",
            "source_label": "SRC-13",
        }]},
    )

    sources = agent._collect_sources([source_result])

    assert [source["url"] for source in sources] == ["https://example.com/legacy"]
    assert agent._replace_research_citations(
        source_result.output, "t1", sources
    ) == "Legacy recalled result [1]."


def test_namespaced_dependency_citation_is_registered_and_replaced():
    agent = SummarizerAgent(name="sum", policy=FakeLLMClient(""))
    source_result = AgentResult(
        task_id="t2",
        status=AgentStatus.SUCCESS,
        output="Inherited evidence [T1:SRC-3].",
        metadata={"sources": [{
            "title": "Upstream source",
            "url": "https://example.com/upstream",
            "source_label": "T1:SRC-3",
        }]},
    )

    sources = agent._collect_sources([source_result])

    assert sources[0]["bindings"] == [{"task_id": "t2", "label": "T1:SRC-3"}]
    assert agent._replace_research_citations(
        source_result.output, "t2", sources
    ) == "Inherited evidence [1]."


def test_same_url_from_multiple_researchers_gets_one_global_citation():
    agent = SummarizerAgent(name="sum", policy=FakeLLMClient(""))
    results = [
        AgentResult(
            task_id="t1",
            status=AgentStatus.SUCCESS,
            output="First finding [SRC-1].",
            metadata={"sources": [{
                "source_label": "SRC-1",
                "url": "https://example.com/shared",
                "title": "Shared source",
            }]},
        ),
        AgentResult(
            task_id="t2",
            status=AgentStatus.SUCCESS,
            output="Second finding [SRC-3].",
            metadata={"sources": [{
                "source_label": "SRC-3",
                "url": "https://example.com/shared",
                "title": "Shared source",
            }]},
        ),
    ]

    sources = agent._collect_sources(results)

    assert len(sources) == 1
    assert sources[0]["citation_id"] == 1
    assert sources[0]["bindings"] == [
        {"task_id": "t1", "label": "SRC-1"},
        {"task_id": "t2", "label": "SRC-3"},
    ]
    assert agent._replace_research_citations(results[0].output, "t1", sources) == (
        "First finding [1]."
    )
    assert agent._replace_research_citations(results[1].output, "t2", sources) == (
        "Second finding [1]."
    )


@pytest.mark.asyncio
async def test_final_report_keeps_only_summarizer_citations_and_compacts_numbers():
    content = "Conclusion uses the second registered source [2].\n\nOverall Confidence: 0.80"
    agent = SummarizerAgent(name="sum", policy=FakeLLMClient(content))
    result = AgentResult(
        task_id="t1",
        status=AgentStatus.SUCCESS,
        output="First [SRC-1], second [SRC-2].",
        confidence=0.9,
        metadata={"sources": [
            {"title": "First", "url": "https://example.com/first", "source_label": "SRC-1"},
            {"title": "Second", "url": "https://example.com/second", "source_label": "SRC-2"},
        ]},
    )

    synthesized = await agent.run(
        SubTask(id="synthesize", description="report", task_type="synthesize"),
        {"query": "test", "results": [result]},
    )

    assert synthesized.output.content == "Conclusion uses the second registered source [1]."
    assert [source["url"] for source in synthesized.output.sources] == [
        "https://example.com/second"
    ]


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
