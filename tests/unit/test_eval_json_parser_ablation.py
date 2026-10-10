import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import eval_json_parser_ablation as evaluation


def test_usage_tracking_client_is_transparent_and_counts_usage():
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18)
    )

    class Delegate:
        model = "unchanged-model"

        def __init__(self):
            self.call = None

        def chat(self, messages, tools=None, **kwargs):
            self.call = (messages, tools, kwargs)
            return response

    delegate = Delegate()
    bucket = {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    client = evaluation._UsageTrackingClient(delegate, bucket)

    messages = [{"role": "user", "content": "test"}]
    tools = [{"type": "function"}]
    result = client.chat(messages, tools=tools, temperature=0.2)

    assert result is response
    assert delegate.call == (messages, tools, {"temperature": 0.2})
    assert client.model == "unchanged-model"
    assert bucket == {
        "calls": 1,
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }


@pytest.mark.asyncio
async def test_failed_run_preserves_parse_records_and_saves_result(monkeypatch, tmp_path):
    record = evaluation.JsonParseRecord(
        stage="planner",
        raw_output="not-json",
        baseline_error="JSONDecodeError",
        raw_len=8,
    )

    class FakeInterceptor:
        def __init__(self):
            self.records = [record]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return None

    class FailingOrchestrator:
        async def run(self, context, timeout_seconds=None):
            raise ValueError("forced failure")

    class FakeRecorder:
        run_id = "test-run"

        def __init__(self, query):
            self.query = query

        def emit(self, event, payload):
            return None

    monkeypatch.setattr(evaluation, "JsonParseInterceptor", FakeInterceptor)
    monkeypatch.setattr(evaluation, "RunEventRecorder", FakeRecorder)
    monkeypatch.setattr(
        evaluation,
        "build_orchestrator",
        lambda *args, **kwargs: FailingOrchestrator(),
    )

    bench = evaluation.ResearchBench()
    result = await evaluation.run_single_question(
        query="forced failure query",
        question_id="custom",
        bench=bench,
        config={"orchestrator": {"global_timeout_seconds": 1}},
        sm=object(),
        kb=object(),
        out_dir=tmp_path,
    )

    assert result["status"] == "failed"
    assert result["json_parse_summary"]["total_records"] == 1
    assert result["output_file"] is not None

    payload = json.loads(Path(result["output_file"]).read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error"] == "ValueError: forced failure"
    assert payload["json_parse_records"][0]["stage"] == "planner"
