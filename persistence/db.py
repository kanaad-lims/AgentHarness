"""Boreas persistence layer: SQLite connection, schema, and CRUD.

Single database file at ~/.boreas/state.db containing two logical layers:
1. Boreas archive tables (sessions, messages, messages_fts) - our schema
2. LangGraph checkpoint tables - managed by SqliteSaver on the SAME connection
"""
from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

# ─── Paths ──────────────────────────────────────────────────────────────
BOREAS_DIR = Path.home() / ".boreas"
DB_PATH = BOREAS_DIR / "state.db"

# ─── Schema ─────────────────────────────────────────────────────────────
SCHEMA = """
-- Boreas conversation archive (our tables)
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    title        TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    status       TEXT DEFAULT 'active' CHECK (status IN ('active', 'archived'))
);

CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    seq          INTEGER NOT NULL,
    role         TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content      TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session_seq ON messages(session_id, seq);

-- FTS5 virtual table for full-text search on archived messages
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    session_id UNINDEXED,
    seq UNINDEXED,
    content='messages',
    content_rowid='id'
);

-- Triggers to keep FTS in sync with messages table
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content, session_id, seq)
    VALUES (new.id, new.content, new.session_id, new.seq);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, session_id, seq)
    VALUES ('delete', old.id, old.content, old.session_id, old.seq);
END;

CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, session_id, seq)
    VALUES ('delete', old.id, old.content, old.session_id, old.seq);
    INSERT INTO messages_fts(rowid, content, session_id, seq)
    VALUES (new.id, new.content, new.session_id, new.seq);
END;
"""

# ─── Connection management ──────────────────────────────────────────────
_connection: Optional[sqlite3.Connection] = None
_connection_lock = threading.Lock()


def get_connection() -> sqlite3.Connection:
    """Return a shared SQLite connection (thread-safe, check_same_thread=False)."""
    global _connection
    with _connection_lock:
        if _connection is None:
            BOREAS_DIR.mkdir(parents=True, exist_ok=True)
            _connection = sqlite3.connect(
                DB_PATH,
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            _connection.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrent access
            _connection.execute("PRAGMA journal_mode=WAL;")
            _connection.execute("PRAGMA foreign_keys=ON;")
            # Initialize our schema
            _connection.executescript(SCHEMA)
            _connection.commit()
        return _connection


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    """Context manager for a transaction on the shared connection."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def close_connection() -> None:
    """Close the shared connection (for cleanup/shutdown)."""
    global _connection
    with _connection_lock:
        if _connection is not None:
            _connection.close()
            _connection = None


# ─── Session CRUD ───────────────────────────────────────────────────────
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_session(session_id: str, title: Optional[str] = None) -> None:
    """Create a new session record."""
    with transaction() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, title, created_at, updated_at, status) VALUES (?, ?, ?, ?, 'active')",
            (session_id, title or f"Session {session_id[:8]}", now_iso(), now_iso()),
        )


def get_session(session_id: str) -> Optional[sqlite3.Row]:
    """Fetch a session by ID."""
    conn = get_connection()
    return conn.execute(
        "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()


def list_sessions(limit: int = 50, offset: int = 0) -> list[sqlite3.Row]:
    """List sessions ordered by most recent update."""
    conn = get_connection()
    return conn.execute(
        "SELECT session_id, title, created_at, updated_at, status FROM sessions ORDER BY updated_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()


def update_session_timestamp(session_id: str) -> None:
    """Update the session's updated_at timestamp."""
    with transaction() as conn:
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (now_iso(), session_id),
        )


def archive_session(session_id: str) -> None:
    """Mark a session as archived."""
    with transaction() as conn:
        conn.execute(
            "UPDATE sessions SET status = 'archived', updated_at = ? WHERE session_id = ?",
            (now_iso(), session_id),
        )


# ─── Message CRUD ───────────────────────────────────────────────────────
def add_message(session_id: str, role: str, content: str) -> int:
    """Append a user/assistant message to the archive. Returns the message seq."""
    if role not in ("user", "assistant"):
        raise ValueError("role must be 'user' or 'assistant'")
    with transaction() as conn:
        # Get next sequence number
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        seq = row[0] if row else 1

        conn.execute(
            "INSERT INTO messages (session_id, seq, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, seq, role, content, now_iso()),
        )
        update_session_timestamp(session_id)
        return seq


def get_messages(session_id: str, limit: Optional[int] = None, offset: int = 0) -> list[sqlite3.Row]:
    """Retrieve messages for a session in order."""
    conn = get_connection()
    sql = "SELECT id, session_id, seq, role, content, created_at FROM messages WHERE session_id = ? ORDER BY seq"
    params: list[Any] = [session_id]
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])
    return conn.execute(sql, params).fetchall()


def get_last_messages(session_id: str, count: int) -> list[sqlite3.Row]:
    """Get the most recent N messages for a session."""
    conn = get_connection()
    return conn.execute(
        "SELECT id, session_id, seq, role, content, created_at FROM messages WHERE session_id = ? ORDER BY seq DESC LIMIT ?",
        (session_id, count),
    ).fetchall()


# ─── FTS Search ─────────────────────────────────────────────────────────
def search_messages(query: str, session_id: Optional[str] = None, limit: int = 20) -> list[sqlite3.Row]:
    """Full-text search on archived messages."""
    conn = get_connection()
    if session_id:
        return conn.execute(
            """
            SELECT m.id, m.session_id, m.seq, m.role, m.content, m.created_at
            FROM messages_fts f
            JOIN messages m ON m.id = f.rowid
            WHERE messages_fts MATCH ? AND f.session_id = ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, session_id, limit),
        ).fetchall()
    return conn.execute(
        """
        SELECT m.id, m.session_id, m.seq, m.role, m.content, m.created_at
        FROM messages_fts f
        JOIN messages m ON m.id = f.rowid
        WHERE messages_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (query, limit),
    ).fetchall()


# ─── Utility ────────────────────────────────────────────────────────────
def get_message_count(session_id: str) -> int:
    """Total message count for a session."""
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
    ).fetchone()
    return row[0] if row else 0


# ─── Module init ────────────────────────────────────────────────────────
# Ensure directory exists on import
BOREAS_DIR.mkdir(parents=True, exist_ok=True)