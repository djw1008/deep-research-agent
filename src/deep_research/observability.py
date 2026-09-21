"""Structured event recording for the local Agent Observatory dashboard."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable


EventSink = Callable[[str, dict[str, Any]], None]


def _safe(value: Any, max_string: int = 12_000) -> Any:
    """Make arbitrary agent payloads JSON-safe and keep event files bounded."""
    if isinstance(value, str):
        return value if len(value) <= max_string else value[:max_string] + "…[truncated]"
    if isinstance(value, dict):
        return {str(key): _safe(item, max_string) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item, max_string) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _safe(str(value), max_string)


class RunEventRecorder:
    """Append-only JSONL event recorder, safe to call from concurrent agents."""

    def __init__(self, query: str, root: str | Path = "outputs/debug_runs") -> None:
        requested_id = os.getenv("DEEP_RESEARCH_RUN_ID", "")
        self.run_id = requested_id if requested_id.isalnum() else uuid.uuid4().hex[:12]
        self.query = query
        self.started_at = time.time()
        self.run_dir = Path(root) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "events.jsonl"
        self._sequence = 0
        self._lock = threading.Lock()
        self.emit("run_started", {"query": query, "run_id": self.run_id})

    def emit(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._sequence += 1
            event = {
                "seq": self._sequence,
                "ts": time.time(),
                "elapsed_ms": round((time.time() - self.started_at) * 1000),
                "run_id": self.run_id,
                "type": event_type,
                "payload": _safe(payload or {}),
            }
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def __call__(self, event_type: str, payload: dict[str, Any]) -> None:
        self.emit(event_type, payload)


class ObservableTrajectory(list[dict[str, Any]]):
    """A normal trajectory list that mirrors every append to the event stream."""

    def __init__(self, task_id: str, sink: EventSink | None = None) -> None:
        super().__init__()
        self.task_id = task_id
        self.sink = sink

    def append(self, item: dict[str, Any]) -> None:
        super().append(item)
        if self.sink is not None:
            self.sink("agent_loop_event", {"task_id": self.task_id, "event": item})
