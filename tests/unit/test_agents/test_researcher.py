"""Tests for ResearchAgent knowledge-base recall."""

from __future__ import annotations

from pathlib import Path

import pytest

from deep_research.agents.researcher import ResearchAgent
from deep_research.core.schema import AgentResult, AgentStatus, SubTask
from deep_research.memory import KnowledgeBase, KnowledgeEntry
from deep_research.memory.embedder import MemoryEmbedder


_NGRAM_EMBEDDER = MemoryEmbedder(model_name=None, fallback_n=2)


class _DummyPolicy:
    def chat(self, messages, **kwargs):
        class Resp:
            content = "dummy"
            tool_calls = []
        return Resp()

    def set_tools(self, tools):
        pass


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "researcher_kb.db")


async def test_recall_from_knowledge_base(db_path: str) -> None:
    """ResearchAgent 对高相似子任务直接返回知识库结果，跳过搜索。"""
    topic = "GPT-4o"
    task_description = "GPT-4o OpenAI 多模态模型"
    kb = KnowledgeBase(db_path=db_path, embedder=_NGRAM_EMBEDDER)
    await kb.initialize()
    try:
        entry = KnowledgeEntry(
            id=KnowledgeBase.make_id(topic, task_description),
            content="GPT-4o 于 2024 年 5 月发布，支持 128K 上下文 [SRC-1]。",
            task_type="search",
            topic=topic,
            confidence=0.85,
            sources=[{
                "source_label": "SRC-1",
                "url": "https://example.com/gpt-4o",
                "title": "GPT-4o announcement",
                "snippet": "Released in May 2024.",
            }],
            metadata={"task_description": task_description},
        )
        await kb.add(entry)

        agent = ResearchAgent(
            name="researcher_search",
            policy=_DummyPolicy(),
            tools=[],
            config={
                "memory": {
                    "recall_before_research": {
                        "enabled": True,
                        "top_k": 1,
                        "threshold": 0.0,  # use 0 to ensure hit in tiny n-gram space
                    }
                }
            },
            knowledge_base=kb,
        )

        task = SubTask(
            id="task_002",
            description=task_description,
            task_type="search",
        )
        result: AgentResult = await agent.run(task, {"query": topic})

        assert result.status == AgentStatus.SUCCESS
        assert "2024 年 5 月发布" in result.output
        assert result.token_usage == 0
        assert result.confidence == 0.85
        assert result.metadata["sources"] == entry.sources
    finally:
        await kb.close()


async def test_recall_disabled(db_path: str) -> None:
    """recall_before_research 关闭时执行正常搜索路径。"""
    topic = "GPT-4o"
    task_description = "GPT-4o OpenAI 多模态模型"
    kb = KnowledgeBase(db_path=db_path, embedder=_NGRAM_EMBEDDER)
    await kb.initialize()
    try:
        entry = KnowledgeEntry(
            id=KnowledgeBase.make_id(topic, task_description),
            content="cached content",
            task_type="search",
            topic=topic,
            confidence=0.85,
            metadata={"task_description": task_description},
        )
        await kb.add(entry)

        agent = ResearchAgent(
            name="researcher_search",
            policy=_DummyPolicy(),
            tools=[],
            config={
                "memory": {
                    "recall_before_research": {"enabled": False}
                }
            },
            knowledge_base=kb,
        )

        task = SubTask(
            id="task_002",
            description=task_description,
            task_type="search",
        )
        # The dummy policy returns "dummy", so we know it went through search.
        result: AgentResult = await agent.run(task, {"query": topic})
        assert result.output == "dummy"
    finally:
        await kb.close()


async def test_recall_only_for_search(db_path: str) -> None:
    """只有 search 类型做知识库召回，analyze/verify 不走召回。"""
    topic = "GPT-4o"
    task_description = "GPT-4o OpenAI 多模态模型"
    kb = KnowledgeBase(db_path=db_path, embedder=_NGRAM_EMBEDDER)
    await kb.initialize()
    try:
        entry = KnowledgeEntry(
            id=KnowledgeBase.make_id(topic, task_description),
            content="cached content",
            task_type="search",
            topic=topic,
            confidence=0.85,
            metadata={"task_description": task_description},
        )
        await kb.add(entry)

        agent = ResearchAgent(
            name="researcher_search",
            policy=_DummyPolicy(),
            tools=[],
            config={
                "memory": {
                    "recall_before_research": {
                        "enabled": True,
                        "top_k": 1,
                        "threshold": 0.0,
                    }
                }
            },
            knowledge_base=kb,
        )

        for task_type in ("analyze", "verify"):
            task = SubTask(
                id=f"task_{task_type}",
                description=task_description,
                task_type=task_type,
            )
            result: AgentResult = await agent.run(task, {"query": topic})
            assert result.output == "dummy", f"{task_type} should not recall"
    finally:
        await kb.close()


async def test_recall_skipped_for_time_sensitive_query(db_path: str) -> None:
    """原始问题含"最近/现在/当前"等时效性词时，search 子任务跳过知识库召回，直接执行。"""
    topic = "GPT-4o"
    task_description = "GPT-4o OpenAI 多模态模型"
    kb = KnowledgeBase(db_path=db_path, embedder=_NGRAM_EMBEDDER)
    await kb.initialize()
    try:
        entry = KnowledgeEntry(
            id=KnowledgeBase.make_id(topic, task_description),
            content="cached content",
            task_type="search",
            topic=topic,
            confidence=0.85,
            metadata={"task_description": task_description},
        )
        await kb.add(entry)

        agent = ResearchAgent(
            name="researcher_search",
            policy=_DummyPolicy(),
            tools=[],
            config={
                "memory": {
                    "recall_before_research": {
                        "enabled": True,
                        "top_k": 1,
                        "threshold": 0.0,
                    }
                }
            },
            knowledge_base=kb,
        )

        task = SubTask(
            id="task_002",
            description=task_description,
            task_type="search",
        )
        # "最近" 命中时效性关键词，应跳过召回，走 DummyPolicy 的搜索路径。
        result: AgentResult = await agent.run(task, {"query": "GPT-4o 最近动态"})
        assert result.output == "dummy"
    finally:
        await kb.close()


def test_web_search_url_deduplication_normalizes_tracking_and_markdown_urls() -> None:
    results = [
        {"title": "first", "url": "https://Example.com/post/?utm_source=news#section", "snippet": "a"},
        {"title": "duplicate", "url": "[mirror](http://example.com/post)", "snippet": "b"},
        {"title": "different query", "url": "https://example.com/post?id=2", "snippet": "c"},
        {"title": "same query reordered", "url": "https://example.com/post?b=2&a=1", "snippet": "d"},
        {"title": "same query reordered duplicate", "url": "https://example.com/post?a=1&b=2&utm_medium=x", "snippet": "e"},
    ]

    deduplicated = ResearchAgent._deduplicate_search_results(results)

    assert [item["title"] for item in deduplicated] == [
        "first",
        "different query",
        "same query reordered",
    ]


def test_web_search_url_deduplication_across_rounds() -> None:
    seen: set[str] = set()
    first = ResearchAgent._deduplicate_search_results(
        [{"title": "first", "url": "https://example.com/a?utm_source=x"}], seen
    )
    second = ResearchAgent._deduplicate_search_results(
        [
            {"title": "duplicate", "url": "http://example.com/a#part"},
            {"title": "new", "url": "https://example.com/b"},
        ],
        seen,
    )

    assert [item["title"] for item in first] == ["first"]
    assert [item["title"] for item in second] == ["new"]


def test_each_loop_accepts_only_one_budget_eligible_tool_call() -> None:
    calls = [
        {"function": {"name": "web_search"}},
        {"function": {"name": "web_search"}},
        {"function": {"name": "browser_batch"}},
        {"function": {"name": "calculator"}},
    ]

    accepted = ResearchAgent._limit_tool_calls(
        calls, remaining_total=3, remaining_search=1, remaining_browser=1
    )

    assert [call["function"]["name"] for call in accepted] == ["web_search"]


def test_source_labels_are_stable_and_attached_only_to_url_items() -> None:
    labels: dict[str, str] = {}
    first = [
        {"title": "A", "url": "https://example.com/a"},
        {"title": "No URL"},
    ]
    second = {
        "results": [
            {"title": "A again", "url": "https://example.com/a"},
            {"title": "B", "url": "https://example.com/b"},
        ]
    }

    ResearchAgent._attach_source_labels(first, labels)
    ResearchAgent._attach_source_labels(second, labels)

    assert first[0]["source_label"] == "SRC-1"
    assert "source_label" not in first[1]
    assert second["results"][0]["source_label"] == "SRC-1"
    assert second["results"][1]["source_label"] == "SRC-2"


def test_select_cited_sources_keeps_only_labels_used_by_output() -> None:
    trajectory = [{
        "role": "tool",
        "result": [
            {
                "source_label": "SRC-1",
                "url": "https://example.com/used",
                "title": "Used",
                "snippet": "evidence",
            },
            {
                "source_label": "SRC-2",
                "url": "https://example.com/unused",
                "title": "Unused",
                "snippet": "unused evidence",
            },
        ],
    }]

    sources = ResearchAgent._select_cited_sources(
        "Only this claim is cited [SRC-1].", trajectory
    )

    assert sources == [{
        "source_label": "SRC-1",
        "url": "https://example.com/used",
        "title": "Used",
        "snippet": "evidence",
    }]


def test_available_sources_preserves_search_metadata_when_browser_result_is_empty() -> None:
    trajectory = [
        {
            "role": "tool",
            "result": [{
                "source_label": "SRC-1",
                "url": "https://example.com/article",
                "title": "有效标题",
                "snippet": "有效搜索摘要",
            }],
        },
        {
            "role": "tool",
            "result": {
                "results": [{
                    "source_label": "SRC-1",
                    "url": "https://example.com/article",
                    "title": "å\u008d\u008eå°\u0094è¡\u0097è§\u0081é\u0097»",
                    "content": "",
                }],
            },
        },
    ]

    source = ResearchAgent._available_sources(trajectory)["SRC-1"]

    assert source["title"] == "有效标题"
    assert source["snippet"] == "有效搜索摘要"


def test_prepare_result_normalizes_citation_before_persistence() -> None:
    task = SubTask(id="t1", description="research", task_type="search")
    trajectory = [{
        "role": "tool",
        "result": [{
            "source_label": "SRC-13",
            "url": "https://example.com/source",
            "title": "Source",
        }],
    }]

    output, metadata = ResearchAgent._prepare_result(
        task,
        "Supported finding [SRC-13 摘要]. Unknown [SRC-99 说明].",
        trajectory,
        from_memory=False,
    )

    assert output == "Supported finding [SRC-13]. Unknown [SRC-99]."
    assert [source["source_label"] for source in metadata["sources"]] == ["SRC-13"]
    assert metadata["citation_diagnostics"] == {
        "normalized_count": 2,
        "invalid_labels": ["SRC-99"],
    }


def test_prepare_result_inherits_namespaced_dependency_source() -> None:
    task = SubTask(id="t2", description="analyze", task_type="analyze")
    context = {
        "dep:t1": "Upstream finding [SRC-3].",
        "dep_sources:t1": [{
            "source_label": "SRC-3",
            "url": "https://example.com/upstream",
            "title": "Upstream source",
        }],
    }

    output, metadata = ResearchAgent._prepare_result(
        task,
        "Inherited finding [t1:SRC-3 摘要].",
        [],
        context,
        from_memory=False,
    )

    assert output == "Inherited finding [T1:SRC-3]."
    assert metadata["sources"] == [{
        "source_label": "T1:SRC-3",
        "url": "https://example.com/upstream",
        "title": "Upstream source",
        "snippet": "",
    }]


def test_task_prompt_namespaces_dependency_citations() -> None:
    agent = ResearchAgent(name="researcher", policy=_DummyPolicy())
    task = SubTask(
        id="t2",
        description="analyze",
        task_type="analyze",
        dependencies=["t1"],
    )
    context = {
        "dep:t1": "Upstream finding [SRC-3 摘要].",
        "dep_sources:t1": [{
            "source_label": "SRC-3",
            "url": "https://example.com/upstream",
            "title": "Upstream source",
        }],
    }

    prompt = agent._build_task_prompt(task, context)

    assert "Upstream finding [T1:SRC-3]." in prompt
    assert "[T1:SRC-3] Upstream source" in prompt


def test_transitive_dependency_keeps_original_source_namespace() -> None:
    agent = ResearchAgent(name="researcher", policy=_DummyPolicy())
    task = SubTask(
        id="task_7",
        description="deeper analysis",
        task_type="analyze",
        dependencies=["task_5"],
    )
    context = {
        "dep:task_5": "Transitive evidence [TASK_1:SRC-3]. Local [SRC-2].",
        "dep_sources:task_5": [
            {
                "source_label": "TASK_1:SRC-3",
                "url": "https://example.com/original",
                "title": "Original task 1 source",
            },
            {
                "source_label": "SRC-2",
                "url": "https://example.com/task-5",
                "title": "Task 5 source",
            },
        ],
    }

    prompt = agent._build_task_prompt(task, context)
    output, metadata = ResearchAgent._prepare_result(
        task,
        "Use inherited [TASK_1:SRC-3] and direct [TASK_5:SRC-2].",
        [],
        context,
        from_memory=False,
    )

    assert "[TASK_1:SRC-3] Original task 1 source" in prompt
    assert "[TASK_5:SRC-2] Task 5 source" in prompt
    assert "TASK_5:TASK_1" not in prompt
    assert output == "Use inherited [TASK_1:SRC-3] and direct [TASK_5:SRC-2]."
    assert [source["source_label"] for source in metadata["sources"]] == [
        "TASK_1:SRC-3",
        "TASK_5:SRC-2",
    ]
    assert metadata["citation_diagnostics"]["invalid_labels"] == []
