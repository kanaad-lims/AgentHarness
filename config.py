"""Single source of truth for all configuration values.

Every env read and default lives here. Policies, tools, presenters, and the
LLM layer import from this module; nothing else calls os.getenv directly.
"""

import os

# --- Model / provider ---
MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
_COMPLETION_TOKENS = os.getenv("MAX_COMPLETION_TOKENS")
MAX_COMPLETION_TOKENS = int(_COMPLETION_TOKENS) if _COMPLETION_TOKENS else None
DEBUG_TOKEN_USAGE = True

# --- Loop caps (per user turn) ---
MAX_MODEL_CALLS_PER_TURN = 8
MAX_TOOL_CALLS_PER_TURN = 6
MAX_ARXIV_CALLS_PER_TURN = 3
MAX_ARXIV_RESULTS_PER_CALL = 10
MAX_TOOL_RESULT_CHARS = 5000
MAX_SAME_CALL_ATTEMPTS = 3

# --- Permissions ---
REQUIRE_BASH_APPROVAL = True

# --- Presenter budgets ---
ARXIV_ABSTRACT_CHARS = int(os.getenv("ARXIV_ABSTRACT_CHARS", "300"))
ARXIV_FORMAT_MAX_CHARS = int(os.getenv("ARXIV_FORMAT_MAX_CHARS", "1500"))
BASH_RENDER_MAX_CHARS = int(os.getenv("BASH_RENDER_MAX_CHARS", "2000"))

# --- Bash execution ---
DEFAULT_TIMEOUT_SECONDS = 30
BASH_HEAD_CHARS = int(os.getenv("BASH_HEAD_CHARS", "8000"))
BASH_TAIL_CHARS = int(os.getenv("BASH_TAIL_CHARS", "8000"))
BASH_STDERR_TAIL_CHARS = int(os.getenv("BASH_STDERR_TAIL_CHARS", "4000"))

# --- arXiv tool ---
DEFAULT_MAX_RESULTS = 5
MAX_RESULTS_LIMIT = 10

# --- Derived ---
RECURSION_LIMIT = 2 * MAX_MODEL_CALLS_PER_TURN + 10
