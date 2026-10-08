"""Session Memory storage.

Stores per-session research rounds (query + DAG + report + sources) for follow-up
"continue research" interactions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import aiosqlite

from .models import SessionMemoryEntry

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS session_rounds (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    query TEXT NOT NULL,
    dag TEXT NOT NULL,
    report TEXT NOT NULL,
    sources TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL
);
"""

_CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_session_rounds_session_id ON session_rounds(session_id);
CREATE INDEX IF NOT EXISTS idx_session_rounds_round ON session_rounds(round);
CREATE INDEX IF NOT EXISTS idx_session_rounds_created_at ON session_rounds(created_at);
"""


class SessionMemory:
    """CRUD for research rounds scoped by ``session_id``."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def _connect(self) -> aiosqlite.Connection:
        if self._connection is None:
            parent = os.path.dirname(self.db_path)
            if parent:
                await asyncio.to_thread(os.makedirs, parent, exist_ok=True)
            self._connection = await aiosqlite.connect(self.db_path)
            self._connection.row_factory = aiosqlite.Row
            await self._connection.execute("PRAGMA journal_mode=WAL")
            await self._connection.execute("PRAGMA synchronous=NORMAL")
        return self._connection

    async def initialize(self) -> None:
        """Create tables and indexes."""
        async with self._lock:
            conn = await self._connect()
            await conn.execute(_CREATE_TABLE_SQL)
            cursor = await conn.execute("PRAGMA table_info(session_rounds)")
            columns = {
                row[1] for row in await cursor.fetchall()
            }
            if "sources" not in columns:
                await conn.execute(
                    "ALTER TABLE session_rounds "
                    "ADD COLUMN sources TEXT NOT NULL DEFAULT '[]'"
                )
            for stmt in _CREATE_INDEXES_SQL.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    await conn.execute(stmt)
            await conn.commit()

    async def add_round(
        self,
        session_id: str,
        round: int,
        query: str,
        dag: dict[str, Any],
        report: str,
        sources: list[dict] | None = None,
    ) -> SessionMemoryEntry:
        """Persist a single research round."""
        import time

        entry = SessionMemoryEntry(
            id=f"{session_id}:round:{round}",
            session_id=session_id,
            round=round,
            query=query,
            dag=dag,
            report=report,
            created_at=time.time(),
            sources=list(sources or []),
        )

        async with self._lock:
            conn = await self._connect()
            await conn.execute(
                """
                INSERT OR REPLACE INTO session_rounds
                (id, session_id, round, query, dag, report, sources, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.session_id,
                    entry.round,
                    entry.query,
                    json.dumps(entry.dag, ensure_ascii=False),
                    entry.report,
                    json.dumps(entry.sources, ensure_ascii=False),
                    entry.created_at,
                ),
            )
            await conn.commit()

        logger.info(
            "Session round persisted [session=%s round=%d]",
            session_id,
            round,
        )
        return entry

    async def get_session_history(self, session_id: str) -> list[SessionMemoryEntry]:
        """Return all rounds for a session, sorted by round ascending."""
        async with self._lock:
            conn = await self._connect()
            cursor = await conn.execute(
                "SELECT * FROM session_rounds WHERE session_id = ? ORDER BY round ASC",
                (session_id,),
            )
            rows = await cursor.fetchall()
            return [self._row_to_entry(row) for row in rows]

    async def list_sessions(self) -> list[dict[str, Any]]:
        """List active sessions with topic, created_at and round count."""
        async with self._lock:
            conn = await self._connect()
            cursor = await conn.execute(
                """
                SELECT
                    session_id,
                    MIN(query) AS topic,
                    MIN(created_at) AS created_at,
                    COUNT(*) AS count,
                    MAX(round) AS max_round
                FROM session_rounds
                GROUP BY session_id
                ORDER BY MIN(created_at) DESC
                """
            )
            rows = await cursor.fetchall()
            return [
                {
                    "session_id": row["session_id"],
                    "topic": row["topic"],
                    "created_at": row["created_at"],
                    "count": row["count"],
                    "max_round": row["max_round"],
                }
                for row in rows
            ]

    @staticmethod
    def _row_to_entry(row: aiosqlite.Row) -> SessionMemoryEntry:
        return SessionMemoryEntry(
            id=row["id"],
            session_id=row["session_id"],
            round=row["round"],
            query=row["query"],
            dag=json.loads(row["dag"]),
            report=row["report"],
            created_at=row["created_at"],
            sources=json.loads(row["sources"]) if row["sources"] else [],
        )

    async def close(self) -> None:
        async with self._lock:
            if self._connection is not None:
                await self._connection.close()
                self._connection = None

    async def __aenter__(self) -> SessionMemory:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
