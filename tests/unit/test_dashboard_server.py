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
