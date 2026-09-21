"""Data models for the persistent memory layer."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SessionMemoryEntry:
    """A single round of research within a session.

    Session Memory stores the full context of one research round:
    - what the user asked
    - how the system planned to answer (DAG)
    - what the final report was

    Multiple entries with the same ``session_id`` form a conversation-like
    research history that can be used for "continue research" follow-ups.
    """

    id: str
    session_id: str
    round: int
    query: str
    dag: dict[str, Any]
    report: str
    created_at: float


@dataclass
class KnowledgeEntry:
    """A reusable piece of knowledge extracted from a successful sub-task.

    Knowledge Base is shared across sessions.  It stores facts/outputs that
    can be recalled by future ``search`` sub-tasks to avoid redundant searches.
    """

    id: str
    content: str
    task_type: str  # "search" | "analyze" | "verify"
    topic: str
    confidence: float = 0.0
    embedding: np.ndarray | None = None
    sources: list[dict] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0
    access_count: int = 0


@dataclass
class KnowledgeMatch:
    """A knowledge entry returned by vector similarity search."""

    entry: KnowledgeEntry
    score: float
