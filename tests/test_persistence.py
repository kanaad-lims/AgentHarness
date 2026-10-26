"""Test persistence layer: database, checkpointer, compaction."""
import tempfile
import os
import time
from pathlib import Path

# Use a temporary database for testing
os.environ["BOREAS_TEST_DB"] = "1"

import pytest


def cleanup_test_data():
    """Clean up test data from the database."""
    from persistence.db import get_connection
    conn = get_connection()
    conn.execute("DELETE FROM messages WHERE session_id LIKE 'test-%' OR session_id LIKE 'archive-%'")
    conn.execute("DELETE FROM sessions WHERE session_id LIKE 'test-%' OR session_id LIKE 'archive-%'")
    conn.commit()


@pytest.fixture(autouse=True)
def clean_db():
    cleanup_test_data()
    yield
    cleanup_test_data()


def test_db_schema_and_crud():
    """Test database initialization and basic CRUD."""
    from persistence.db import (
        get_connection,
        create_session,
        get_session,
        list_sessions,
        add_message,
        get_messages,
        get_message_count,
        search_messages,
    )

    conn = get_connection()
    # Tables should exist
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    table_names = [t[0] for t in tables]
    assert "sessions" in table_names
    assert "messages" in table_names
    assert "messages_fts" in table_names

    # Create session
    session_id = "test-session-123"
    create_session(session_id, "Test Session")
    session = get_session(session_id)
    assert session is not None
    assert session["session_id"] == session_id
    assert session["title"] == "Test Session"
    assert session["status"] == "active"

    # Add messages
    seq1 = add_message(session_id, "user", "Hello, world!")
    seq2 = add_message(session_id, "assistant", "Hi there! How can I help?")
    seq3 = add_message(session_id, "user", "What is Python?")
    assert seq1 == 1
    assert seq2 == 2
    assert seq3 == 3

    # Get messages
    msgs = get_messages(session_id)
    assert len(msgs) == 3
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Hello, world!"
    assert msgs[1]["role"] == "assistant"
    assert msgs[2]["role"] == "user"

    # Count
    assert get_message_count(session_id) == 3

    # List sessions
    sessions = list_sessions()
    assert any(s["session_id"] == session_id for s in sessions)

    # Search
    results = search_messages("Python", session_id=session_id)
    assert len(results) == 1
    assert results[0]["content"] == "What is Python?"


def test_checkpointer_integration():
    """Test that SqliteSaver works on the same connection."""
    from persistence.checkpoint import get_saver, make_config
    from persistence.db import get_connection

    saver = get_saver()
    assert saver is not None

    # Same connection should be used
    conn = get_connection()
    # The saver should have run its migrations
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = [t[0] for t in tables]
    # Checkpointer tables should exist (names vary by version)
    checkpoint_tables = [t for t in table_names if "checkpoint" in t.lower()]
    assert len(checkpoint_tables) > 0, f"No checkpoint tables found in {table_names}"

    # make_config should produce valid config
    config = make_config("test-thread", recursion_limit=10)
    assert config["configurable"]["thread_id"] == "test-thread"
    assert config["recursion_limit"] == 10


def test_compaction_helpers():
    """Test token estimation and turn splitting."""
    from persistence.compaction import (
        estimate_tokens,
        estimate_messages_tokens,
        split_into_turns,
        count_complete_turns,
        should_compact,
        compact_state_messages_sync,
    )

    # Token estimation
    assert estimate_tokens("hello") == 2  # ~4 chars per token
    assert estimate_tokens("") == 0

    # Message token estimation
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    tokens = estimate_messages_tokens(messages)
    assert tokens > 0

    # Turn splitting
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "search"}}]},
        {"role": "tool", "content": "result"},
        {"role": "assistant", "content": "A2"},
    ]
    turns = split_into_turns(messages)
    assert len(turns) == 2
    assert len(turns[0]) == 2  # Q1 + A1
    assert len(turns[1]) == 4  # Q2 + tool call + tool result + A2

    # Complete turn count
    assert count_complete_turns(messages) == 2

    # Compaction decision
    small_messages = [{"role": "user", "content": "Hi"}]
    assert not should_compact(small_messages)

    # Large messages should trigger (TOKEN_BUDGET = 8000)
    large_content = "x" * 35000  # ~8750 tokens
    large_messages = [{"role": "user", "content": large_content}]
    assert should_compact(large_messages)


def test_compaction_sync():
    """Test synchronous compaction (heuristic summary)."""
    from persistence.compaction import compact_state_messages_sync

    # Build messages with many turns
    messages = []
    for i in range(10):
        messages.append({"role": "user", "content": f"Question {i}"})
        messages.append({"role": "assistant", "content": f"Answer {i}"})

    summary = "Previous summary"
    new_messages, new_summary = compact_state_messages_sync(messages, summary)

    # Should keep summary + recent turns
    assert "Previous summary" in new_summary
    assert "Question" in new_summary  # heuristic includes topics
    # Should have system message with summary + 2 recent turns (4 messages)
    assert new_messages[0]["role"] == "system"
    assert "Conversation summary" in new_messages[0]["content"]
    # Last 4 messages should be the last 2 turns
    assert len(new_messages) == 5  # 1 system + 4 messages (2 turns)


def test_full_turn_archive():
    """Test that a full turn archives correctly."""
    import time
    from persistence.db import create_session, add_message, get_messages

    # This would require a real model call, so we just test the archive part
    session_id = "archive-test-" + str(int(time.time()))
    create_session(session_id)

    # Simulate a turn
    user_msg = "What is the capital of France?"
    assistant_msg = "The capital of France is Paris."

    add_message(session_id, "user", user_msg)
    add_message(session_id, "assistant", assistant_msg)

    msgs = get_messages(session_id)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == user_msg
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == assistant_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])