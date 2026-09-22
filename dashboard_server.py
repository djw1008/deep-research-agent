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
    terminal_index = next(
        (i for i in range(len(events) - 1, -1, -1) if events[i].get("type") in {"run_completed", "run_failed"}),
        None,
    )
    # 终态事件之后又追加了新事件（如对抗升级），说明 run 重新进入运行中
    if terminal_index is None or terminal_index < len(events) - 1:
        status = "running"
    else:
        status = events[terminal_index].get("payload", {}).get("state", "running")
        if status == "done":
            status = "success"
    adversarial = False
    meta_path = RUNS_DIR / run_id / "meta.json"
    if meta_path.exists():
        try:
            adversarial = bool(json.loads(meta_path.read_text(encoding="utf-8")).get("adversarial"))
        except (json.JSONDecodeError, AttributeError):
            adversarial = False
    return {
        "id": run_id,
        "query": query,
        "status": status,
        "started_at": first.get("ts", 0),
        "elapsed_ms": last.get("elapsed_ms", 0),
        "event_count": len(events),
        "adversarial": adversarial,
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
    first_started: dict[str, float] = {}
    completed_ts: dict[str, float] = {}
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
            if task_id and task_id not in first_started:
                first_started[task_id] = event.get("ts") or 0.0
            task_state[task_id] = "running"
        elif event.get("type") == "agent_loop_event":
            task_id = payload.get("task_id")
            loops.setdefault(task_id, []).append(payload.get("event", {}))
        elif event.get("type") == "task_completed":
            task_id = payload.get("task_id")
            raw_status = payload.get("status", "failed")
            task_state[task_id] = "success" if raw_status == "success" else "failed"
            completed[task_id] = payload
            completed_ts[task_id] = event.get("ts") or 0.0

    layers = [list(layer) for layer in dag_payload.get("layers", [])]
    assigned = {task_id for layer in layers for task_id in layer}
    # DAG 之外的任务（synthesize / 对抗轮次的 red / blue）按真实启动时间排序，
    # 任务类型变化或串行执行（前一个同类型任务已结束）时开启新层，
    # 让时间线还原 "red×5 → blue 修复 → 下一轮 red×5 → …" 的实际顺序。
    extras = sorted(
        (task_id for task_id in tasks if task_id not in assigned),
        key=lambda tid: (first_started.get(tid, 0.0), tid),
    )
    current_type: str | None = None
    layer_end_ts = 0.0
    for task_id in extras:
        task_type = tasks[task_id].get("task_type", "agent")
        start_ts = first_started.get(task_id, 0.0)
        if current_type is None or task_type != current_type or (layer_end_ts > 0.0 and start_ts >= layer_end_ts):
            layers.append([task_id])
            current_type = task_type
            layer_end_ts = 0.0
        else:
            layers[-1].append(task_id)
        assigned.add(task_id)
        end_ts = completed_ts.get(task_id)
        if end_ts is not None:
            layer_end_ts = max(layer_end_ts, end_ts)
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


def start_run(query: str, adversarial: bool = False) -> dict:
    query = query.strip()
    if not query or len(query) > 1000:
        raise ValueError("query must contain 1-1000 characters")
    run_id = uuid.uuid4().hex[:12]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "meta.json").write_text(
        json.dumps({"adversarial": adversarial}, ensure_ascii=False),
        encoding="utf-8",
    )
    child_env = os.environ.copy()
    child_env["DEEP_RESEARCH_RUN_ID"] = run_id
    command = [sys.executable, str(ROOT / "run.py"), "--query", query]
    if adversarial:
        command.append("--adversarial")
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


def start_adversarial_upgrade(run_id: str) -> tuple[int, dict]:
    """对已完成的 run 只启动 Red/Blue 对抗升级进程。"""
    if not run_id.replace("-", "").replace("_", "").isalnum():
        return 404, {"error": "report not found"}
    run_dir = RUNS_DIR / run_id
    if not (run_dir / "report.md").exists():
        return 404, {"error": "report not found"}
    with PROCESS_LOCK:
        process = PROCESSES.get(run_id)
        if process is not None and process.poll() is None:
            return 409, {"error": "run already in progress"}

    events_path = run_dir / "events.jsonl"
    last_elapsed = 0
    if events_path.exists():
        for line in reversed(events_path.read_text(encoding="utf-8", errors="replace").splitlines()):
            try:
                last_elapsed = json.loads(line).get("elapsed_ms", 0)
                break
            except json.JSONDecodeError:
                continue

    # 首次升级前快照原始报告，供前端做对抗前后对比
    before_path = run_dir / "report_before_adversarial.md"
    if not before_path.exists():
        before_path.write_text(
            (run_dir / "report.md").read_text(encoding="utf-8", errors="replace"),
            encoding="utf-8",
        )

    # 立即写入状态事件，保证前端下一次轮询即为 running，消除 spawn 竞态
    with events_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "ts": time.time(),
            "elapsed_ms": last_elapsed,
            "run_id": run_id,
            "type": "state_transition",
            "payload": {"from": "done", "to": "adversarial"},
        }, ensure_ascii=False) + "\n")

    meta_path = run_dir / "meta.json"
    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    meta["adversarial"] = True
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    child_env = os.environ.copy()
    child_env["DEEP_RESEARCH_RUN_ID"] = run_id
    command = [sys.executable, str(ROOT / "run.py"), "--upgrade-adversarial", run_id]
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
    return 202, {"run_id": run_id, "status": "upgrade_started"}


def _pre_adversarial_report(run_dir: Path) -> dict | None:
    """取对抗前报告：优先升级流程的快照文件；直接开对抗的 run 没有快照，
    回退到 synthesize 节点的结构化输出（即对抗前的初版报告）。

    返回 {"markdown": ..., "confidence": float | None, "round_scores": [float, ...]}，
    无对抗历史返回 None。round_scores 来自每轮 merge 节点的加权评分，
    首个即对抗前质量分、最后一个即对抗后质量分（同一度量，可直接对比）。
    """
    before_path = run_dir / "report_before_adversarial.md"
    events_path = run_dir / "events.jsonl"
    if not before_path.exists() and not events_path.exists():
        return None
    synthesize_done: dict | None = None
    saw_adversarial = before_path.exists()
    round_scores: list[tuple[int, float]] = []
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload = event.get("payload", {})
            if event.get("type") == "task_started" and payload.get("task_type") in ("red_agent", "blue_agent"):
                saw_adversarial = True
            if event.get("type") != "task_completed":
                continue
            if payload.get("task_id") == "synthesize" and payload.get("status") == "success":
                synthesize_done = payload
                continue
            task_id = str(payload.get("task_id") or "")
            if task_id.startswith("merge_round_") and isinstance(payload.get("output"), dict):
                score = payload["output"].get("overall_score")
                if isinstance(score, (int, float)):
                    try:
                        round_scores.append((int(task_id.rsplit("_", 1)[1]), float(score)))
                    except (ValueError, IndexError):
                        continue
    if not saw_adversarial:
        return None
    confidence = None
    fallback_content = None
    if synthesize_done and isinstance(synthesize_done.get("output"), dict):
        output = synthesize_done["output"]
        raw_confidence = output.get("confidence")
        if isinstance(raw_confidence, (int, float)):
            confidence = raw_confidence
        content = output.get("content")
        if isinstance(content, str) and content:
            fallback_content = content
    scores = [score for _, score in sorted(round_scores)]
    if before_path.exists():
        return {
            "markdown": before_path.read_text(encoding="utf-8", errors="replace"),
            "confidence": confidence,
            "round_scores": scores,
        }
    if fallback_content:
        return {"markdown": fallback_content, "confidence": confidence, "round_scores": scores}
    return None


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
        if path.startswith("/api/runs/") and path.endswith("/report"):
            run_id = unquote(path.removeprefix("/api/runs/").removesuffix("/report"))
            report = None
            if run_id.replace("-", "").replace("_", "").isalnum():
                report_path = RUNS_DIR / run_id / "report.md"
                if report_path.exists():
                    before = _pre_adversarial_report(RUNS_DIR / run_id)
                    report = {
                        "run_id": run_id,
                        "markdown": report_path.read_text(encoding="utf-8", errors="replace"),
                        # 对抗前的原始版本：升级流程用快照文件，直接开对抗的 run 回退到 synthesize 输出；无对抗为 None
                        "before_markdown": before["markdown"] if before else None,
                        "before_confidence": before["confidence"] if before else None,
                        # 每轮红队加权评分（同一度量）：首项=对抗前，末项=对抗后
                        "adversarial_scores": before["round_scores"] if before else [],
                    }
            self._send(200 if report else 404, report or {"error": "report not ready"})
            return
        if path.startswith("/api/runs/"):
            detail = run_detail(unquote(path.removeprefix("/api/runs/")))
            self._send(200 if detail else 404, detail or {"error": "run not found"})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path.startswith("/api/runs/") and path.endswith("/adversarial"):
            run_id = unquote(path.removeprefix("/api/runs/").removesuffix("/adversarial"))
            status, payload = start_adversarial_upgrade(run_id)
            self._send(status, payload)
            return
        if path != "/api/runs":
            self._send(404, {"error": "not found"})
            return
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 4096)
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._send(202, start_run(str(payload.get("query", "")), adversarial=bool(payload.get("adversarial"))))
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
