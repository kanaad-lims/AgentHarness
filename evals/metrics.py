"""In-house code metrics over recorded turn envelopes (runs/turns.jsonl).

Judge-based quality lives in LangSmith; everything here is deterministic
and free: budgets, tool discipline, citations, recovery.
"""

from __future__ import annotations


def within_budgets(turn: dict, *, max_tokens_in: int = 7000) -> tuple[bool, str]:
    """Summed prompt tokens stay under budget; tool calls have no global cap."""
    tokens_in = sum(call.get("prompt_tokens") or 0 for call in turn.get("model_calls", []))
    if tokens_in > max_tokens_in:
        return False, f"input tokens {tokens_in} over budget {max_tokens_in}"
    return True, "ok"


def no_unknown_tools(turn: dict) -> tuple[bool, str]:
    """Every executed tool resolves (unknown tools surface as Error results)."""
    for trace in turn.get("tool_trace", []):
        if trace.get("verdict") == "failed" and "unknown tool" in str(trace.get("arguments", "")):
            return False, f"unknown tool: {trace.get('name')}"
    return True, "ok"


def cites_sources_when_factual(turn: dict) -> tuple[bool, str]:
    """Answers on search-backed turns must carry at least one URL."""
    searched = any(
        trace.get("name") in {"arxiv_search", "web_search"}
        and trace.get("verdict") == "executed"
        for trace in turn.get("tool_trace", [])
    )
    if not searched:
        return True, "no search in turn"
    answer = turn.get("answer", "")
    if "http" in answer:
        return True, "ok"
    return False, "search-backed answer without a cited URL"


def recovered_or_declared(turn: dict) -> tuple[bool, str]:
    """Denials and failures must end in an answer, never silence or a crash."""
    if turn.get("errors"):
        return False, f"turn errors: {turn['errors']}"
    if not turn.get("answer", "").strip():
        return False, "empty answer"
    return True, "ok"


METRICS = {
    "within_budgets": within_budgets,
    "no_unknown_tools": no_unknown_tools,
    "cites_sources_when_factual": cites_sources_when_factual,
    "recovered_or_declared": recovered_or_declared,
}


def grade_turn(turn: dict) -> dict:
    """Run every metric over one envelope; returns {metric: (pass, detail)}."""
    return {name: func(turn) for name, func in METRICS.items()}
