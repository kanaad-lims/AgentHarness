"""Turn observability: LangSmith mirror for runs, never in the critical path.

Every public method here is a no-op without LANGSMITH_API_KEY and swallows
its own errors. A missing trace is acceptable; a missing answer is not.
"""

from observability.recorder import TurnRecorder, current_turn
from observability.turn import (
    model_usage,
    tool_verdict,
    turn_metadata,
    verdict_for_blocked,
)

__all__ = [
    "TurnRecorder",
    "current_turn",
    "model_usage",
    "tool_verdict",
    "turn_metadata",
    "verdict_for_blocked",
]
