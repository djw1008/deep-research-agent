import json

from deep_research.observability import ObservableTrajectory, RunEventRecorder


def test_recorder_writes_ordered_jsonl(tmp_path):
    recorder = RunEventRecorder("test query", root=tmp_path)
    recorder.emit("state_transition", {"from": "idle", "to": "planning"})
    events = [json.loads(line) for line in recorder.path.read_text(encoding="utf-8").splitlines()]
    assert [event["type"] for event in events] == ["run_started", "state_transition"]
    assert [event["seq"] for event in events] == [1, 2]
    assert events[0]["payload"]["query"] == "test query"


def test_observable_trajectory_mirrors_append():
    events = []
    trajectory = ObservableTrajectory("task_1", lambda event_type, payload: events.append((event_type, payload)))
    trajectory.append({"turn": 1, "role": "assistant"})
    assert len(trajectory) == 1
    assert events == [("agent_loop_event", {"task_id": "task_1", "event": {"turn": 1, "role": "assistant"}})]
