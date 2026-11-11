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
    - the LangGraph checkpointer uses ONE throwaway in-memory saver per
      test (shared across graph builds, so multi-turn checkpoint state
      works inside a single test) — and never the real DB.
    - run_turn's self-summary-load returns None (no prior state in tests
      unless a test explicitly writes checkpoints through the same saver).
    - recorder JSONL sink points at a temp dir instead of runs/.
    """
    saver = _fresh_saver()
    # Delegate to the REAL loader, but through the shared in-memory saver:
    # unknown threads return None (no real-DB reads), known threads return
    # actual state — so multi-turn checkpoint flows work inside tests.
    from persistence.checkpoint import load_thread_state as _real_load_thread_state
    monkeypatch.setattr("agent.get_session", lambda _session_id: True)
    monkeypatch.setattr("agent.create_db_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("agent.add_message", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("agent.load_thread_state", lambda _tid: _real_load_thread_state(_tid))
    monkeypatch.setattr("persistence.checkpoint.get_saver", lambda: saver)
    monkeypatch.setattr("observability.recorder.RUNS_DIR", tmp_path / "runs")
    return saver
