"""Boreas persistence package."""

from persistence.db import (
    create_session,
    get_session,
    list_sessions,
    update_session_timestamp,
    archive_session,
    add_message,
    get_messages,
    get_last_messages,
    search_messages,
    get_message_count,
    get_connection,
    close_connection,
    transaction,
)

from persistence.checkpoint import (
    get_saver,
    compile_with_checkpointer,
    make_config,
    list_threads,
    get_latest_checkpoint,
)

from persistence.compaction import (
    should_compact,
    compact_state_messages,
    compact_state_messages_sync,
    estimate_tokens,
    estimate_messages_tokens,
    split_into_turns,
    count_complete_turns,
    TOKEN_BUDGET,
    MESSAGE_BUDGET,
    MIN_RECENT_TURNS,
)

__all__ = [
    # db
    "create_session",
    "get_session",
    "list_sessions",
    "update_session_timestamp",
    "archive_session",
    "add_message",
    "get_messages",
    "get_last_messages",
    "search_messages",
    "get_message_count",
    "get_connection",
    "close_connection",
    "transaction",
    # checkpoint
    "get_saver",
    "compile_with_checkpointer",
    "make_config",
    "list_threads",
    "get_latest_checkpoint",
    # compaction
    "should_compact",
    "compact_state_messages",
    "compact_state_messages_sync",
    "estimate_tokens",
    "estimate_messages_tokens",
    "split_into_turns",
    "count_complete_turns",
    "TOKEN_BUDGET",
    "MESSAGE_BUDGET",
    "MIN_RECENT_TURNS",
]