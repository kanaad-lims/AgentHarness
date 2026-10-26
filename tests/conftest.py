"""Shared test isolation: keep tests off the real user database and logs."""

import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver


def _fresh_saver() -> SqliteSaver:
    saver = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
    saver.setup()
    return saver


@pytest.fixture(autouse=True)
def isolated_persistence(monkeypatch, tmp_path):
    """Redirect all persistence/observability writes to a temp dir.

    - agent.py DB helpers are stubbed so no session/message rows hit
      the real ~/.boreas/state.db.
    - the LangGraph checkpointer uses a throwaway in-memory saver, so
      checkpoint state cannot leak between tests or into the real DB.
    - recorder JSONL sink points at a temp dir instead of runs/.
    """
    monkeypatch.setattr("agent.get_session", lambda _session_id: True)
    monkeypatch.setattr("agent.create_db_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("agent.add_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "persistence.checkpoint.get_saver",
        lambda: _fresh_saver(),
    )
    monkeypatch.setattr("observability.recorder.RUNS_DIR", tmp_path / "runs")
