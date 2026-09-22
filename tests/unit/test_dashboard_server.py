import json

import dashboard_server


class _FakeProcess:
    pid = 4321


def test_start_run_returns_stable_id_and_sets_child_env(tmp_path, monkeypatch):
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return _FakeProcess()

    monkeypatch.setattr(dashboard_server, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(dashboard_server.subprocess, "Popen", fake_popen)
    result = dashboard_server.start_run("研究问题")

    assert result["run_id"] != "pending"
    assert result["pid"] == 4321
    assert captured["env"]["DEEP_RESEARCH_RUN_ID"] == result["run_id"]
    assert (tmp_path / result["run_id"] / "process.log").exists()


def test_run_summary_reports_failed_terminal_event():
    events = [
        {"ts": 1, "elapsed_ms": 0, "payload": {"query": "q"}, "type": "run_started"},
        {"ts": 2, "elapsed_ms": 10, "payload": {"state": "failed"}, "type": "run_failed"},
    ]
    assert dashboard_server._run_summary("abc", events)["status"] == "failed"


def test_start_run_with_adversarial_appends_flag_and_writes_meta(tmp_path, monkeypatch):
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return _FakeProcess()

    monkeypatch.setattr(dashboard_server, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(dashboard_server.subprocess, "Popen", fake_popen)
    result = dashboard_server.start_run("研究问题", adversarial=True)

    assert captured["command"][-1] == "--adversarial"
    meta = json.loads((tmp_path / result["run_id"] / "meta.json").read_text(encoding="utf-8"))
    assert meta == {"adversarial": True}
    assert dashboard_server._run_summary(result["run_id"], [])["adversarial"] is True


def test_run_summary_defaults_adversarial_false(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_server, "RUNS_DIR", tmp_path)
    assert dashboard_server._run_summary("no-meta", [])["adversarial"] is False


def test_run_summary_reports_running_when_events_follow_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_server, "RUNS_DIR", tmp_path)
    events = [
        {"ts": 1, "elapsed_ms": 0, "payload": {"query": "q"}, "type": "run_started"},
        {"ts": 2, "elapsed_ms": 10, "payload": {"status": "success"}, "type": "run_completed"},
        {"ts": 3, "elapsed_ms": 10, "payload": {"from": "done", "to": "adversarial"}, "type": "state_transition"},
    ]
    assert dashboard_server._run_summary("abc", events)["status"] == "running"


def test_start_adversarial_upgrade_404_when_report_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard_server, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(dashboard_server, "PROCESSES", {})
    (tmp_path / "abc123").mkdir()
    status, payload = dashboard_server.start_adversarial_upgrade("abc123")
    assert status == 404
    assert "error" in payload


def test_start_adversarial_upgrade_spawns_process_and_updates_state(tmp_path, monkeypatch):
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return _FakeProcess()

    monkeypatch.setattr(dashboard_server, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(dashboard_server, "PROCESSES", {})
    monkeypatch.setattr(dashboard_server.subprocess, "Popen", fake_popen)
    run_dir = tmp_path / "run42"
    run_dir.mkdir()
    (run_dir / "report.md").write_text("# report", encoding="utf-8")
    (run_dir / "meta.json").write_text(json.dumps({"adversarial": False}), encoding="utf-8")
    (run_dir / "events.jsonl").write_text(
        json.dumps({"ts": 1, "elapsed_ms": 0, "type": "run_started", "payload": {}}) + "\n"
        + json.dumps({"ts": 2, "elapsed_ms": 99, "type": "run_completed", "payload": {"status": "success"}}) + "\n",
        encoding="utf-8",
    )

    status, payload = dashboard_server.start_adversarial_upgrade("run42")

    assert status == 202
    assert payload == {"run_id": "run42", "status": "upgrade_started"}
    assert captured["command"][-2:] == ["--upgrade-adversarial", "run42"]
    assert captured["env"]["DEEP_RESEARCH_RUN_ID"] == "run42"
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["adversarial"] is True
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[-1]["type"] == "state_transition"
    assert events[-1]["payload"] == {"from": "done", "to": "adversarial"}
    assert events[-1]["elapsed_ms"] == 99
    assert dashboard_server.PROCESSES["run42"].pid == 4321


def test_start_adversarial_upgrade_snapshots_before_version(tmp_path, monkeypatch):
    """首次升级时应把当前 report.md 快照为 report_before_adversarial.md，且不覆盖已有快照。"""
    def fake_popen(command, **kwargs):
        return _FakeProcess()

    monkeypatch.setattr(dashboard_server, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(dashboard_server, "PROCESSES", {})
    monkeypatch.setattr(dashboard_server.subprocess, "Popen", fake_popen)
    run_dir = tmp_path / "run99"
    run_dir.mkdir()
    (run_dir / "report.md").write_text("# 原始报告", encoding="utf-8")

    status, _ = dashboard_server.start_adversarial_upgrade("run99")

    assert status == 202
    before_path = run_dir / "report_before_adversarial.md"
    assert before_path.read_text(encoding="utf-8") == "# 原始报告"

    # 再次升级（报告已被改写）不应覆盖首次快照
    dashboard_server.PROCESSES.clear()
    (run_dir / "report.md").write_text("# 升级后报告", encoding="utf-8")
    status, _ = dashboard_server.start_adversarial_upgrade("run99")
    assert status == 202
    assert before_path.read_text(encoding="utf-8") == "# 原始报告"
