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

    matches = await knowledge_base.search("GPT-4o release", task_type="search", threshold=0.0)
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

    matches = await knowledge_base.search("Topic description", task_type="search", threshold=0.0)
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

    matches = await knowledge_base.search("t", task_type="search", threshold=0.0)
    assert len(matches) == 1
    assert matches[0].entry.task_type == "search"
