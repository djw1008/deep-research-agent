"""Tests for SessionMemory."""

from __future__ import annotations

import pytest

from deep_research.memory.session_store import SessionMemory


@pytest.fixture
def session_memory(tmp_path):
    db = tmp_path / "session_memory.db"
    return SessionMemory(str(db))


@pytest.mark.asyncio
async def test_add_and_get_history(session_memory):
    await session_memory.initialize()

    entry = await session_memory.add_round(
        session_id="s1",
        round=1,
        query="GPT-4o capabilities",
        dag={"nodes": ["task_1"], "edges": {}},
        report="GPT-4o supports 128K context.",
    )

    assert entry.session_id == "s1"
    assert entry.round == 1
    assert entry.query == "GPT-4o capabilities"

    history = await session_memory.get_session_history("s1")
    assert len(history) == 1
    assert history[0].report == "GPT-4o supports 128K context."


@pytest.mark.asyncio
async def test_history_sorted_by_round(session_memory):
    await session_memory.initialize()

    await session_memory.add_round("s1", 2, "q2", {"nodes": [], "edges": {}}, "report2")
    await session_memory.add_round("s1", 1, "q1", {"nodes": [], "edges": {}}, "report1")

    history = await session_memory.get_session_history("s1")
    assert [e.round for e in history] == [1, 2]
    assert [e.query for e in history] == ["q1", "q2"]


@pytest.mark.asyncio
async def test_list_sessions(session_memory):
    await session_memory.initialize()

    await session_memory.add_round("s1", 1, "q1", {"nodes": [], "edges": {}}, "r1")
    await session_memory.add_round("s2", 1, "q2", {"nodes": [], "edges": {}}, "r2")
    await session_memory.add_round("s1", 2, "q3", {"nodes": [], "edges": {}}, "r3")

    sessions = await session_memory.list_sessions()
    assert len(sessions) == 2
    by_id = {s["session_id"]: s for s in sessions}
    assert by_id["s1"]["count"] == 2
    assert by_id["s2"]["count"] == 1
