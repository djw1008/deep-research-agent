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
            content="GPT-4o 于 2024 年 5 月发布，支持 128K 上下文。",
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
