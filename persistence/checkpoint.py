"""LangGraph checkpoint integration: shared SqliteSaver on the Boreas database."""
from __future__ import annotations

from langgraph.checkpoint.sqlite import SqliteSaver

from persistence.db import get_connection

# ─── Singleton SqliteSaver ──────────────────────────────────────────────
_saver: SqliteSaver | None = None


def get_saver() -> SqliteSaver:
    """Return a shared SqliteSaver instance using the Boreas database connection.

    The saver runs its own migrations on the shared connection, creating its
    internal checkpoint tables alongside our Boreas archive tables.
    """
    global _saver
    if _saver is None:
        conn = get_connection()
        _saver = SqliteSaver(conn)
        _saver.setup()  # Create checkpoint tables
    return _saver


def compile_with_checkpointer(graph, *, saver: SqliteSaver | None = None):
    """Compile a LangGraph StateGraph with the Boreas checkpointer.

    Args:
        graph: A StateGraph instance (not yet compiled)
        saver: Optional SqliteSaver; uses shared instance if not provided

    Returns:
        Compiled graph with checkpointing enabled
    """
    if saver is None:
        saver = get_saver()
    return graph.compile(checkpointer=saver)


# ─── Thread/Session binding ─────────────────────────────────────────────
def make_config(thread_id: str, *, recursion_limit: int = 25) -> dict:
    """Create the config dict for invoking a graph with a specific thread_id.

    The thread_id MUST match the session_id for 1:1 correspondence.
    """
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }


# ─── Inspection helpers (optional, for debugging) ───────────────────────
def list_threads(saver: SqliteSaver | None = None) -> list[dict]:
    """List all threads (sessions) that have checkpoints."""
    if saver is None:
        saver = get_saver()
    # SqliteSaver doesn't expose a direct list method; we query our sessions table
    from persistence.db import list_sessions
    return [{"thread_id": s["session_id"], "title": s["title"], "updated_at": s["updated_at"]} for s in list_sessions()]


def get_latest_checkpoint(thread_id: str, saver: SqliteSaver | None = None) -> dict | None:
    """Fetch the latest checkpoint for a thread (for inspection)."""
    if saver is None:
        saver = get_saver()
    # Use the checkpointer's get_tuple method
    checkpoint_tuple = saver.get({"configurable": {"thread_id": thread_id}})
    if checkpoint_tuple is None:
        return None
    return {
        "checkpoint": checkpoint_tuple.checkpoint,
        "metadata": checkpoint_tuple.metadata,
        "parent_config": checkpoint_tuple.parent_config,
    }