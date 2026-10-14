from types import SimpleNamespace

from observability import (
    TurnRecorder,
    model_usage,
    tool_verdict,
    turn_metadata,
    verdict_for_blocked,
)
from prompts.messages import TOOL_LIMIT_MESSAGE


def test_recorder_disabled_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setattr("observability.recorder.RUNS_DIR", tmp_path)
    recorder = TurnRecorder()

    assert recorder.enabled is True
    assert recorder.start_turn("hi", "model") is None
    assert recorder.log_model_call(None, {}, 0.1, 1) is None
    assert recorder.log_tool_call(None, "bash", {}, "executed", 10, 0.1) is None
    recorder.end_turn(None, "answer", {})
    assert (tmp_path / "turns.jsonl").exists()


def test_recorder_never_raises(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake-key")

    class BrokenClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("offline")

    monkeypatch.setitem(__import__("sys").modules, "langsmith", SimpleNamespace(Client=BrokenClient))
    recorder = TurnRecorder()

    assert recorder.start_turn("hi", "model") is None


def test_verdict_mapping():
    assert verdict_for_blocked(TOOL_LIMIT_MESSAGE) == "limit:total"
    assert verdict_for_blocked("something else") == "blocked"
    assert tool_verdict("bash", "Command was not run; nope", False) == "denied"
    assert tool_verdict("bash", "Error: boom", True) == "failed"
    assert tool_verdict("bash", "Exit code: 0\nok", False) == "executed"


def test_model_usage_tolerates_missing_fields():
    assert model_usage(SimpleNamespace(usage=None)) == {
        "prompt_tokens": None,
        "completion_tokens": None,
    }


def test_turn_metadata_extracts_counts():
    assert turn_metadata({"model_calls": 2, "tool_calls": 1}, "m") == {
        "model": "m",
        "model_calls": 2,
        "tool_calls": 1,
    }
