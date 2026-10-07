"""Policy predicates: failure classification and per-call limit verdicts.

Pure functions only — no graph, no I/O, no execution. agent.py imports
these and keeps orchestration; the rules live here and are unit-testable.
"""

import json

from config import (
    MAX_ARXIV_CALLS_PER_TURN,
    MAX_SAME_CALL_ATTEMPTS,
    MAX_TOOL_CALLS_PER_TURN,
    MAX_WEB_CALLS_PER_TURN,
)
from prompts.messages import (
    ARXIV_LIMIT_MESSAGE,
    DUPLICATE_TOOL_MESSAGE,
    RETRY_LIMIT_MESSAGE,
    TOOL_LIMIT_MESSAGE,
    WEB_LIMIT_MESSAGE,
)


def approx_tokens(text: str) -> int:
    """Rough character-based estimate; provider tokenization is authoritative."""
    return (len(text) + 3) // 4


def tool_call_fields(tool_call) -> tuple:
    """Return (call_id, name, arguments_json_string) for object or dict calls."""
    if hasattr(tool_call, "function"):
        function = tool_call.function
        if isinstance(function, dict):
            name = function.get("name", "")
            args_text = function.get("arguments", "{}")
        else:
            name = getattr(function, "name", "")
            args_text = getattr(function, "arguments", "{}")
        call_id = getattr(tool_call, "id", None)
        if not isinstance(args_text, str):
            args_text = json.dumps(args_text, ensure_ascii=False, default=str)
        return call_id, name, args_text or "{}"
    if isinstance(tool_call, dict):
        function = tool_call.get("function", {})
        if isinstance(function, dict):
            return (
                tool_call.get("id"),
                function.get("name", ""),
                function.get("arguments", "{}") or "{}",
            )
        return (
            tool_call.get("id"),
            tool_call.get("name", ""),
            json.dumps(tool_call.get("args", {}), ensure_ascii=False, default=str),
        )
    return None, "", "{}"


def parse_tool_arguments(args_text: str) -> dict:
    """Parse tool arguments JSON, falling back to an empty object."""
    try:
        arguments = json.loads(args_text or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return arguments if isinstance(arguments, dict) else {}


def call_key(name: str, arguments: dict) -> str:
    """Canonical identity of one tool call for dedup and attempt counting."""
    return json.dumps([name, arguments], sort_keys=True, ensure_ascii=False)


def is_tool_failure(name: str, result: str) -> bool:
    """Transient execution failures may be retried; successes must reformulate."""
    if not isinstance(result, str):
        return False
    if (
        result.startswith("Error:")
        or result.startswith("Tool error")
        or result.startswith("Timed out after")
        or result.startswith("ArXiv error:")
        or result.startswith("Web error:")
    ):
        return True
    stripped = result.strip()
    if stripped.startswith("{") and '"error"' in stripped:
        return True
    return False


def check_tool_call(
    name: str,
    arguments: dict,
    *,
    tool_calls_used: int,
    arxiv_calls_used: int,
    web_calls_used: int,
    attempts_made: int,
    already_succeeded: bool,
) -> tuple[bool, str | None]:
    """Decide allow-vs-block for one requested call.

    Returns (blocked, message). A None message means allowed.
    """
    if tool_calls_used >= MAX_TOOL_CALLS_PER_TURN:
        return True, TOOL_LIMIT_MESSAGE
    if already_succeeded:
        return True, DUPLICATE_TOOL_MESSAGE
    if attempts_made >= MAX_SAME_CALL_ATTEMPTS:
        return True, RETRY_LIMIT_MESSAGE
    if name == "arxiv_search" and arxiv_calls_used >= MAX_ARXIV_CALLS_PER_TURN:
        return True, ARXIV_LIMIT_MESSAGE
    if name == "web_search" and web_calls_used >= MAX_WEB_CALLS_PER_TURN:
        return True, WEB_LIMIT_MESSAGE
    return False, None
