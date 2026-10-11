"""Pure helpers: usage extraction, verdict mapping, turn metadata."""

from prompts.messages import (
    ARXIV_LIMIT_MESSAGE,
    DUPLICATE_TOOL_MESSAGE,
    RETRY_LIMIT_MESSAGE,
    SKILL_LIMIT_MESSAGE,
    TOOL_LIMIT_MESSAGE,
    WEB_LIMIT_MESSAGE,
)

_BLOCK_REASONS = {
    TOOL_LIMIT_MESSAGE: "limit:total",
    DUPLICATE_TOOL_MESSAGE: "duplicate",
    RETRY_LIMIT_MESSAGE: "limit:retries",
    ARXIV_LIMIT_MESSAGE: "limit:arxiv",
    WEB_LIMIT_MESSAGE: "limit:web",
    SKILL_LIMIT_MESSAGE: "limit:skills",
}


def verdict_for_blocked(message: str) -> str:
    """Map a block message to a short machine-readable reason."""
    return _BLOCK_REASONS.get(message, "blocked")


def tool_verdict(name: str, result: str, failed: bool) -> str:
    """Verdict for an executed tool call."""
    if name == "bash" and result.startswith("Command was not run"):
        return "denied"
    return "failed" if failed else "executed"


def model_usage(response) -> dict:
    """Extract token usage from a chat response; tolerant of fakes."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None}
    completion_details = getattr(usage, "completion_tokens_details", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": getattr(completion_details, "reasoning_tokens", None),
        "cached_tokens": getattr(prompt_details, "cached_tokens", None),
    }


def turn_metadata(result: dict, model: str) -> dict:
    """Small metadata payload attached when a turn closes."""
    return {
        "model": model,
        "model_calls": result.get("model_calls", 0),
        "tool_calls": result.get("tool_calls", 0),
    }
