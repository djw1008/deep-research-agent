"""Knowledge Base storage.

Globally shared, cross-session store for reusable sub-task outputs.
Provides vector similarity search for future ``search`` sub-tasks.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any

import aiosqlite
import numpy as np

from .embedder import MemoryEmbedder
from .models import KnowledgeEntry, KnowledgeMatch

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS knowledge (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    task_type TEXT NOT NULL,
    topic TEXT NOT NULL,
    confidence REAL NOT NULL,
    embedding BLOB NOT NULL,
    sources TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    access_count INTEGER DEFAULT 0
);
"""

_CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_knowledge_task_type ON knowledge(task_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_topic ON knowledge(topic);
CREATE INDEX IF NOT EXISTS idx_knowledge_access_count ON knowledge(access_count);
"""


class KnowledgeBase:
    """Persistent knowledge store with in-memory vector index.

    Design choices:
      - All active entries are loaded into memory at startup for fast
        similarity search and deduplication.
      - Deduplication is global (cross-session) based on the embedding of
        ``topic + task_description``.
      - Higher-confidence versions replace lower-confidence duplicates.
    """

    _DEFAULT_MAX_ENTRIES = 10000
    _DEFAULT_EVICT_INTERVAL = 100
    _DEFAULT_EVICTED_RETENTION_DAYS = 30

    def __init__(
        self,
        db_path: str,
        embedder: MemoryEmbedder | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.db_path = db_path
        self.embedder = embedder or MemoryEmbedder()
        self.config = config or {}

        self._connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

        # In-memory vector index
        self._entry_ids: list[str] = []
        self._embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._entries_cache: dict[str, KnowledgeEntry] = {}
        self._embedding_dim: int = 0

        self._dedup_threshold: float = float(
            self.config.get("similarity_threshold_dup", 0.92)
        )
        self._max_entries: int = self._DEFAULT_MAX_ENTRIES
        self._evict_interval: int = self._DEFAULT_EVICT_INTERVAL

        # Write batching
        self._batch_size: int = max(1, self.config.get("batch_size", 10))
        self._pending_deletes: set[str] = set()
        self._pending_evictions: set[str] = set()
        self._pending_access_updates: dict[str, tuple[int, float]] = {}
        self._pending_inserts: list[KnowledgeEntry] = []
        self._write_count: int = 0

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
        """Create tables, indexes and load all entries into memory."""
        async with self._lock:
            conn = await self._connect()
            await conn.execute(_CREATE_TABLE_SQL)
            for stmt in _CREATE_INDEXES_SQL.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    await conn.execute(stmt)
            await conn.commit()
            await self._clear_stale_embeddings_locked()
            await self._rebuild_index_locked()

    async def _clear_stale_embeddings_locked(self) -> None:
        """If existing embeddings have a different dimension, clear the table."""
        if self.embedder.model_name is None:
            return

        conn = await self._connect()
        cursor = await conn.execute(
            "SELECT embedding FROM knowledge WHERE embedding IS NOT NULL LIMIT 1"
        )
        row = await cursor.fetchone()
        if row is None or row["embedding"] is None:
            return

        old_emb = np.frombuffer(row["embedding"], dtype=np.float32)
        old_dim = old_emb.shape[0]

        sample_emb = await asyncio.to_thread(self.embedder.encode_one, "dimension probe")
        if self.embedder.backend != "sentence_transformers":
            return
        expected_dim = sample_emb.shape[0]

        if old_dim != expected_dim:
            logger.warning(
                "Knowledge embedding dimension mismatch: db=%d model=%d. Clearing knowledge.",
                old_dim,
                expected_dim,
            )
            await conn.execute("DELETE FROM knowledge")
            await conn.commit()

    async def _rebuild_index_locked(self) -> None:
        """Load active entries from SQLite and rebuild the in-memory index."""
        conn = self._connection
        if conn is None:
            self._entry_ids = []
            self._entries_cache = {}
            self._embeddings = np.zeros((0, 0), dtype=np.float32)
            return

        cursor = await conn.execute("SELECT * FROM knowledge ORDER BY created_at")
        rows = await cursor.fetchall()
        entries = [self._row_to_entry(row) for row in rows]

        self._entries_cache = {e.id: e for e in entries}
        entries_with_emb = [e for e in entries if e.embedding is not None]

        if entries_with_emb:
            dims = [e.embedding.shape[0] for e in entries_with_emb]
            expected_dim = max(set(dims), key=dims.count)
            skipped = sum(1 for d in dims if d != expected_dim)
            if skipped:
                logger.warning(
                    "Knowledge embedding dimension mismatch: skipped %d entries (expected %d)",
                    skipped,
                    expected_dim,
                )
            entries_with_emb = [
                e for e in entries_with_emb if e.embedding.shape[0] == expected_dim
            ]

        self._entry_ids = [e.id for e in entries_with_emb]

        if entries_with_emb:
            mat = np.array([e.embedding for e in entries_with_emb], dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms < 1e-9] = 1.0
            self._embeddings = mat / norms
            self._embedding_dim = self._embeddings.shape[1]
        else:
            self._embeddings = np.zeros((0, max(self._embedding_dim, 0)), dtype=np.float32)

        logger.info("Knowledge index rebuilt: %d entries", len(entries))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def search(
        self,
        query: str,
        task_type: str | None = "search",
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[KnowledgeMatch]:
        """Vector similarity search over knowledge entries."""
        async with self._lock:
            matches = self._search_locked(query, task_type, top_k, threshold)
            for match in matches:
                entry = match.entry
                entry.access_count += 1
                entry.updated_at = time.time()
                self._pending_access_updates[entry.id] = (
                    entry.access_count,
                    entry.updated_at,
                )
            await self._flush_locked()
            return matches

    def _search_locked(
        self,
        query: str,
        task_type: str | None,
        top_k: int,
        threshold: float,
    ) -> list[KnowledgeMatch]:
        if self._embeddings.shape[0] == 0:
            return []

        query_emb = self.embedder.encode_one(query)
        query_emb = query_emb.astype(np.float32, copy=False)
        norm = np.linalg.norm(query_emb)
        if norm > 0:
            query_emb = query_emb / norm

        all_sims = self._embeddings.dot(query_emb)
        candidates: list[tuple[float, str]] = []

        for idx, eid in enumerate(self._entry_ids):
            entry = self._entries_cache[eid]
            if task_type is not None and entry.task_type != task_type:
                continue
            score = float(all_sims[idx])
            if score >= threshold:
                candidates.append((score, eid))

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[0], reverse=True)
        k = min(top_k, len(candidates))
        return [
            KnowledgeMatch(entry=self._entries_cache[eid], score=score)
            for score, eid in candidates[:k]
        ]

    async def add(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        """Add a knowledge entry with global deduplication.

        Deduplication is based on the embedding of ``topic + task_description``,
        where ``task_description`` is read from ``entry.metadata``.
        """
        task_description = entry.metadata.get("task_description", "")
        dedup_text = f"{entry.topic}\n{task_description}".strip() or entry.content[:500]
        embedding = await asyncio.to_thread(self.embedder.encode_one, dedup_text)
        entry.embedding = embedding

        async with self._lock:
            insert_action = "inserted"

            # 1) Exact ID collision: same topic + task_description produced the same id.
            existing_by_id = self._entries_cache.get(entry.id)
            if existing_by_id is not None:
                if existing_by_id.confidence >= entry.confidence:
                    logger.info(
                        "Knowledge id already exists, keeping existing %s (%.2f >= %.2f)",
                        entry.id,
                        existing_by_id.confidence,
                        entry.confidence,
                    )
                    existing_by_id.metadata["action"] = "kept_id"
                    return existing_by_id
                self._remove_from_index(entry.id)
                self._pending_deletes.add(entry.id)
                logger.info("Replaced lower-confidence knowledge %s with new entry", entry.id)
                insert_action = "replaced_id"

            # 2) Global semantic deduplication
            dup_id = self._find_duplicate_id(embedding)
            if dup_id is not None and dup_id != entry.id:
                existing = self._entries_cache[dup_id]
                if existing.confidence >= entry.confidence:
                    logger.info(
                        "Duplicate knowledge detected, keeping existing %s (%.2f >= %.2f)",
                        dup_id,
                        existing.confidence,
                        entry.confidence,
                    )
                    existing.metadata["action"] = "kept_dup"
                    return existing
                self._remove_from_index(dup_id)
                self._pending_deletes.add(dup_id)
                logger.info(
                    "Replaced lower-confidence duplicate %s with new entry %s",
                    dup_id,
                    entry.id,
                )
                insert_action = "replaced_dup"

            now = time.time()
            entry.created_at = now
            entry.updated_at = now
            entry.metadata["action"] = insert_action

            self._pending_inserts.append(entry)
            self._write_count += 1
            self._append_to_index(entry, embedding)

            await self._maybe_evict_locked()
            await self._flush_locked()

        return entry

    def _find_duplicate_id(self, embedding: np.ndarray) -> str | None:
        """Find a duplicate in the global knowledge index."""
        if self._embeddings.shape[0] == 0:
            return None

        query_emb = embedding.astype(np.float32, copy=False)
        norm = np.linalg.norm(query_emb)
        if norm < 1e-9:
            return None
        query_emb = query_emb / norm

        sims = self._embeddings.dot(query_emb)
        best_idx = int(np.argmax(sims))
        if float(sims[best_idx]) >= self._dedup_threshold:
            return self._entry_ids[best_idx]
        return None

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------
    def _remove_from_index(self, eid: str) -> None:
        if eid not in self._entries_cache:
            return
        idx = self._entry_ids.index(eid)
        self._entry_ids.pop(idx)
        del self._entries_cache[eid]
        if self._embeddings.shape[0] > 0:
            self._embeddings = np.delete(self._embeddings, idx, axis=0)

    def _append_to_index(self, entry: KnowledgeEntry, embedding: np.ndarray) -> None:
        self._entry_ids.append(entry.id)
        self._entries_cache[entry.id] = entry

        emb = embedding.reshape(1, -1).astype(np.float32, copy=False)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        if self._embeddings.shape[0] == 0:
            self._embeddings = emb
            self._embedding_dim = emb.shape[1]
        else:
            self._embeddings = np.vstack([self._embeddings, emb])

    # ------------------------------------------------------------------
    # Write batching
    # ------------------------------------------------------------------
    async def _flush_locked(self, force: bool = False) -> None:
        pending_total = (
            len(self._pending_deletes)
            + len(self._pending_evictions)
            + len(self._pending_access_updates)
            + len(self._pending_inserts)
        )
        if pending_total == 0:
            return
        if not force and pending_total < self._batch_size:
            return

        conn = await self._connect()
        try:
            await conn.execute("BEGIN IMMEDIATE")

            if self._pending_deletes:
                placeholders = ",".join("?" * len(self._pending_deletes))
                await conn.execute(
                    f"DELETE FROM knowledge WHERE id IN ({placeholders})",
                    tuple(self._pending_deletes),
                )
                self._pending_deletes.clear()

            if self._pending_evictions:
                # Knowledge base currently does not evict; placeholder for future.
                self._pending_evictions.clear()

            retention_seconds = self._DEFAULT_EVICTED_RETENTION_DAYS * 24 * 3600
            cutoff = time.time() - retention_seconds
            cursor = await conn.execute(
                "DELETE FROM knowledge WHERE created_at < ?",
                (cutoff,),
            )
            if cursor.rowcount:
                logger.info("Pruned stale knowledge rows: %d", cursor.rowcount)

            if self._pending_access_updates:
                await conn.executemany(
                    "UPDATE knowledge SET access_count = ?, updated_at = ? WHERE id = ?",
                    [
                        (count, updated_at, eid)
                        for eid, (count, updated_at) in self._pending_access_updates.items()
                    ],
                )
                self._pending_access_updates.clear()

            if self._pending_inserts:
                await conn.executemany(
                    """
                    INSERT OR REPLACE INTO knowledge
                    (id, content, task_type, topic, confidence, embedding, sources,
                     metadata, created_at, updated_at, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            entry.id,
                            entry.content,
                            entry.task_type,
                            entry.topic,
                            entry.confidence,
                            entry.embedding.tobytes() if entry.embedding is not None else b"",
                            json.dumps(entry.sources, ensure_ascii=False),
                            json.dumps(entry.metadata, ensure_ascii=False),
                            entry.created_at,
                            entry.updated_at,
                            entry.access_count,
                        )
                        for entry in self._pending_inserts
                    ],
                )
                self._pending_inserts.clear()

            await conn.commit()
        except Exception:
            await conn.rollback()
            raise

    async def _maybe_evict_locked(self) -> None:
        if self._max_entries <= 0:
            return
        if self._evict_interval > 0 and self._write_count % self._evict_interval != 0:
            return
        await self._evict_locked()

    async def _evict_locked(self) -> None:
        """LRU eviction when entry count exceeds the cap."""
        conn = await self._connect()
        cursor = await conn.execute("SELECT COUNT(*) FROM knowledge")
        row = await cursor.fetchone()
        db_count = row[0] if row else 0
        pending_insert_count = len(self._pending_inserts)
        total_active = db_count + pending_insert_count

        if total_active <= self._max_entries:
            return

        to_evict = total_active - self._max_entries
        pending_ids = {entry.id for entry in self._pending_inserts}
        excluded = pending_ids | self._pending_deletes

        cursor = await conn.execute(
            "SELECT id FROM knowledge ORDER BY access_count ASC, updated_at ASC, created_at ASC"
        )
        rows = await cursor.fetchall()
        evicted_ids: list[str] = []
        for row in rows:
            eid = row["id"]
            if eid in excluded:
                continue
            evicted_ids.append(eid)
            if len(evicted_ids) >= to_evict:
                break

        for eid in evicted_ids:
            self._remove_from_index(eid)
            self._pending_deletes.add(eid)

        logger.info("Knowledge LRU eviction: removed %d entries", len(evicted_ids))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_entry(row: aiosqlite.Row) -> KnowledgeEntry:
        embedding = None
        if row["embedding"] is not None:
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)

        return KnowledgeEntry(
            id=row["id"],
            content=row["content"],
            task_type=row["task_type"],
            topic=row["topic"],
            confidence=row["confidence"] or 0.0,
            embedding=embedding,
            sources=json.loads(row["sources"]) if row["sources"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=row["created_at"] or 0.0,
            updated_at=row["updated_at"] or 0.0,
            access_count=row["access_count"] or 0,
        )

    @staticmethod
    def make_id(topic: str, task_description: str) -> str:
        """Generate a stable id from topic + task_description."""
        text = f"{topic}\n{task_description}".strip()
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    async def close(self) -> None:
        async with self._lock:
            try:
                await self._flush_locked(force=True)
            finally:
                if self._connection is not None:
                    await self._connection.close()
                    self._connection = None

    async def __aenter__(self) -> KnowledgeBase:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
