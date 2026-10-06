"""Central policy values: loop caps, budgets, and enforcement flags.

Values are defined in config.py (single source of truth) and re-exported
here. Rule logic that interprets them lives with the enforcers for now and
moves into this package as policies/checks.py next.
"""

from config import (
    ARXIV_ABSTRACT_CHARS,
    ARXIV_FORMAT_MAX_CHARS,
    BASH_HEAD_CHARS,
    BASH_RENDER_MAX_CHARS,
    BASH_STDERR_TAIL_CHARS,
    BASH_TAIL_CHARS,
    MAX_ARXIV_CALLS_PER_TURN,
    MAX_ARXIV_RESULTS_PER_CALL,
    MAX_MODEL_CALLS_PER_TURN,
    MAX_SAME_CALL_ATTEMPTS,
    MAX_TOOL_CALLS_PER_TURN,
    MAX_TOOL_RESULT_CHARS,
    RECURSION_LIMIT,
    REQUIRE_BASH_APPROVAL,
)

__all__ = [
    "ARXIV_ABSTRACT_CHARS",
    "ARXIV_FORMAT_MAX_CHARS",
    "BASH_HEAD_CHARS",
    "BASH_RENDER_MAX_CHARS",
    "BASH_STDERR_TAIL_CHARS",
    "BASH_TAIL_CHARS",
    "MAX_ARXIV_CALLS_PER_TURN",
    "MAX_ARXIV_RESULTS_PER_CALL",
    "MAX_MODEL_CALLS_PER_TURN",
    "MAX_SAME_CALL_ATTEMPTS",
    "MAX_TOOL_CALLS_PER_TURN",
    "MAX_TOOL_RESULT_CHARS",
    "RECURSION_LIMIT",
    "REQUIRE_BASH_APPROVAL",
]
