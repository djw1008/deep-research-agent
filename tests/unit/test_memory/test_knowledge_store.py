"""Tests for KnowledgeBase."""

from __future__ import annotations

import pytest

from deep_research.memory.embedder import MemoryEmbedder
from deep_research.memory.knowledge_store import KnowledgeBase
from deep_research.memory.models import KnowledgeEntry


_NGRAM_EMBEDDER = MemoryEmbedder(model_name=None, fallback_n=2)


@pytest.fixture
def knowledge_base(tmp_path):
    db = tmp_path / "knowledge_base.db"
    return KnowledgeBase(str(db), embedder=_NGRAM_EMBEDDER, config={"similarity_threshold_dup": 0.92})


@pytest.mark.asyncio
async def test_add_and_search(knowledge_base):
    await knowledge_base.initialize()

    entry = KnowledgeEntry(
        id=KnowledgeBase.make_id("GPT-4o", "release date"),
        content="GPT-4o was released in May 2024.",
        task_type="search",
        topic="GPT-4o",
        confidence=0.85,
        metadata={"task_description": "release date"},
    )
    stored = await knowledge_base.add(entry)
    assert stored.id == entry.id
    assert stored.metadata.get("action") == "inserted"

    matches = await knowledge_base.search("release date", task_type="search", threshold=0.0)
    assert len(matches) == 1
    assert "May 2024" in matches[0].entry.content


@pytest.mark.asyncio
async def test_global_deduplication_keeps_higher_confidence(knowledge_base):
    await knowledge_base.initialize()

    entry1 = KnowledgeEntry(
        id=KnowledgeBase.make_id("Topic", "description"),
        content="low confidence content",
        task_type="search",
        topic="Topic",
        confidence=0.5,
        metadata={"task_description": "description"},
    )
    await knowledge_base.add(entry1)

    entry2 = KnowledgeEntry(
        id=KnowledgeBase.make_id("Topic", "description"),
        content="high confidence content",
        task_type="search",
        topic="Topic",
        confidence=0.9,
        metadata={"task_description": "description"},
    )
    stored = await knowledge_base.add(entry2)

    assert stored.confidence == 0.9

    matches = await knowledge_base.search("description", task_type="search", threshold=0.0)
    assert len(matches) == 1
    assert matches[0].entry.confidence == 0.9


@pytest.mark.asyncio
async def test_search_filters_by_task_type(knowledge_base):
    await knowledge_base.initialize()

    await knowledge_base.add(KnowledgeEntry(
        id=KnowledgeBase.make_id("t", "search task"),
        content="search result",
        task_type="search",
        topic="t",
        confidence=0.8,
        metadata={"task_description": "search task"},
    ))
    await knowledge_base.add(KnowledgeEntry(
        id=KnowledgeBase.make_id("t", "analyze task"),
        content="analyze result",
        task_type="analyze",
        topic="t",
        confidence=0.8,
        metadata={"task_description": "analyze task"},
    ))

    matches = await knowledge_base.search("search task", task_type="search", threshold=0.0)
    assert len(matches) == 1
    assert matches[0].entry.task_type == "search"


@pytest.mark.asyncio
async def test_hybrid_search_prefers_matching_model_entity(knowledge_base):
    await knowledge_base.initialize()
    common = "在中文推理、代码生成和长上下文任务上的性能表现数据"
    for model in ("GPT-4o", "Claude 3.5", "Gemini 1.5", "Qwen2.5"):
        description = f"收集 {model} {common}"
        await knowledge_base.add(KnowledgeEntry(
            id=KnowledgeBase.make_id("comparison", description),
            content=f"result for {model}",
            task_type="search",
            topic="comparison",
            confidence=0.8,
            metadata={"task_description": description},
        ))

    matches = await knowledge_base.search(
        f"整理 Gemini 1.5 {common}",
        task_type="search",
        top_k=1,
        threshold=0.0,
    )

    assert len(matches) == 1
    assert matches[0].entry.content == "result for Gemini 1.5"


@pytest.mark.asyncio
async def test_hybrid_search_rejects_different_model_with_shared_template(knowledge_base):
    await knowledge_base.initialize()
    qwen_description = "收集 Qwen2.5 在中文推理、代码生成和长上下文任务上的性能表现数据"
    await knowledge_base.add(KnowledgeEntry(
        id=KnowledgeBase.make_id("comparison", qwen_description),
        content="Qwen result",
        task_type="search",
        topic="comparison",
        confidence=0.8,
        metadata={"task_description": qwen_description},
    ))

    matches = await knowledge_base.search(
        "收集 Gemini 1.5 在中文推理、代码生成和长上下文任务上的性能表现数据",
        task_type="search",
        top_k=1,
        threshold=0.0,
    )

    assert matches == []
