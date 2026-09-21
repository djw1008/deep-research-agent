#!/usr/bin/env python3
"""Local JSON API for the Deep Research Agent Observatory.

This intentionally uses only the Python standard library so the dashboard can
be started without changing the research runtime dependencies.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "outputs" / "debug_runs"
LEGACY_DIR = ROOT / "outputs"
PROCESS_LOCK = threading.Lock()
PROCESSES: dict[str, subprocess.Popen] = {}


def _read_events(run_id: str) -> list[dict]:
    if not run_id.replace("-", "").replace("_", "").isalnum():
        return []
    path = RUNS_DIR / run_id / "events.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _run_summary(run_id: str, events: list[dict]) -> dict:
    first = events[0] if events else {}
    last = events[-1] if events else {}
    query = first.get("payload", {}).get("query", run_id)
    terminal = next((e for e in reversed(events) if e.get("type") in {"run_completed", "run_failed"}), None)
    status = terminal.get("payload", {}).get("state", "running") if terminal else "running"
    if status == "done":
        status = "success"
    return {
        "id": run_id,
        "query": query,
        "status": status,
        "started_at": first.get("ts", 0),
        "elapsed_ms": last.get("elapsed_ms", 0),
        "event_count": len(events),
    }


def list_runs() -> list[dict]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for path in RUNS_DIR.iterdir():
        if path.is_dir():
            events = _read_events(path.name)
            if events:
                runs.append(_run_summary(path.name, events))
    return sorted(runs, key=lambda item: item["started_at"], reverse=True)


def run_detail(run_id: str) -> dict | None:
    events = _read_events(run_id)
    if not events:
        return None
    summary = _run_summary(run_id, events)
    dag_event = next((e for e in reversed(events) if e.get("type") == "dag_created"), None)
    dag_payload = dag_event.get("payload", {}) if dag_event else {}
    tasks = {task["id"]: task for task in dag_payload.get("tasks", []) if task.get("id")}
    task_state = {task_id: "waiting" for task_id in tasks}
    loops: dict[str, list[dict]] = {task_id: [] for task_id in tasks}
    completed: dict[str, dict] = {}
    states = []

    for event in events:
        payload = event.get("payload", {})
        if event.get("type") == "state_transition":
            states.append({**payload, "ts": event.get("ts"), "elapsed_ms": event.get("elapsed_ms")})
        elif event.get("type") == "task_started":
            task_id = payload.get("task_id")
            if task_id and task_id not in tasks:
                tasks[task_id] = {
                    "id": task_id,
                    "description": payload.get("description", task_id),
                    "task_type": payload.get("task_type", "agent"),
                    "dependencies": payload.get("dependencies", []),
                }
                loops[task_id] = []
            task_state[task_id] = "running"
        elif event.get("type") == "agent_loop_event":
            task_id = payload.get("task_id")
            loops.setdefault(task_id, []).append(payload.get("event", {}))
        elif event.get("type") == "task_completed":
            task_id = payload.get("task_id")
            raw_status = payload.get("status", "failed")
            task_state[task_id] = "success" if raw_status == "success" else "failed"
            completed[task_id] = payload

    layers = [list(layer) for layer in dag_payload.get("layers", [])]
    assigned = {task_id for layer in layers for task_id in layer}
    for task_type in ("synthesize", "red_agent", "blue_agent", "agent"):
        extra = [task_id for task_id, task in tasks.items() if task_id not in assigned and task.get("task_type") == task_type]
        if extra:
            layers.append(extra)
            assigned.update(extra)
    nodes = []
    for layer_index, layer in enumerate(layers):
        for row_index, task_id in enumerate(layer):
            task = tasks.get(task_id, {"id": task_id, "description": task_id, "dependencies": []})
            result = completed.get(task_id, {})
            nodes.append({
                "id": task_id,
                "label": task.get("description", task_id),
                "type": task.get("task_type", "search"),
                "dependencies": task.get("dependencies", []),
                "status": task_state.get(task_id, "waiting"),
                "confidence": result.get("confidence"),
                "tokens": result.get("token_usage", 0),
                "layer": layer_index,
                "row": row_index,
            })

    return {
        **summary,
        "nodes": nodes,
        "layers": layers,
        "loops": loops,
        "outputs": completed,
        "states": states,
        "events": events,
    }


def start_run(query: str) -> dict:
    query = query.strip()
    if not query or len(query) > 1000:
        raise ValueError("query must contain 1-1000 characters")
    run_id = uuid.uuid4().hex[:12]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    child_env = os.environ.copy()
    child_env["DEEP_RESEARCH_RUN_ID"] = run_id
    command = [sys.executable, str(ROOT / "run.py"), "--query", query]
    log_stream = (run_dir / "process.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=child_env,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
    )
    log_stream.close()
    with PROCESS_LOCK:
        PROCESSES[run_id] = process
    return {"run_id": run_id, "pid": process.pid, "status": "started"}


class Handler(BaseHTTPRequestHandler):
    server_version = "AgentObservatory/0.1"

    def _send(self, status: int, data: dict | list) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        origin = self.headers.get("Origin", "")
        if origin in {"http://localhost:3000", "http://127.0.0.1:3000"}:
            self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send(204, {})

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send(200, {"status": "healthy", "runs_dir": str(RUNS_DIR)})
            return
        if path == "/api/runs":
            self._send(200, list_runs())
            return
        if path.startswith("/api/runs/"):
            detail = run_detail(unquote(path.removeprefix("/api/runs/")))
            self._send(200 if detail else 404, detail or {"error": "run not found"})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/runs":
            self._send(404, {"error": "not found"})
            return
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 4096)
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._send(202, start_run(str(payload.get("query", ""))))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})

    def log_message(self, fmt: str, *args) -> None:
        print(f"[dashboard-api] {self.address_string()} {fmt % args}")


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Agent Observatory API: http://127.0.0.1:8765")
    server.serve_forever()


if __name__ == "__main__":
    main()
