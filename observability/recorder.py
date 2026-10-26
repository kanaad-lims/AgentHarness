"""Turn observability: local JSONL always, LangSmith mirror when keyed.

Every public method here is a no-op without its sink and swallows its own
errors. A missing trace is acceptable; a missing answer is not.
"""

import json
import os
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

# (recorder, turn_run_id) for the currently executing turn, if any.
current_turn: ContextVar = ContextVar("boreas_turn", default=None)

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class TurnRecorder:
    def __init__(self) -> None:
        self.api_key = os.getenv("LANGSMITH_API_KEY", "")
        self.project = os.getenv("LANGSMITH_PROJECT", "boreas-agent")
        self._client = None
        self._buffer: dict | None = None

    @property
    def enabled(self) -> bool:
        return True

    def _client_or_none(self):
        if not self.api_key:
            return None
        if self._client is None:
            try:
                from langsmith import Client

                self._client = Client(api_key=self.api_key)
            except Exception:
                return None
        return self._client

    def _safe(self, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            return None

    # -- local sink --

    def _write_jsonl(self) -> None:
        def _default(value):
            return str(value)

        try:
            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            with open(RUNS_DIR / "turns.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(self._buffer, ensure_ascii=False, default=_default) + "\n")
        except Exception:
            pass

    # -- turn lifecycle --

    def start_turn(self, user_text: str, model: str):
        """Open a turn: local buffer always, LangSmith run when keyed."""
        self._buffer = {
            "turn_id": None,
            "started_at": _utcnow(),
            "ended_at": None,
            "model": model,
            "user": user_text,
            "answer": "",
            "model_calls": [],
            "tool_trace": [],
            "limits_hit": [],
            "denials": 0,
            "errors": [],
            "metadata": {},
        }
        client = self._client_or_none()
        if client is None:
            return None
        import uuid

        run_id = str(uuid.uuid4())
        self._buffer["turn_id"] = run_id
        self._safe(
            client.create_run,
            name="boreas-turn",
            run_type="chain",
            inputs={"user": user_text, "model": model},
            project_name=self.project,
            id=run_id,
        )
        return run_id

    def log_model_call(self, turn_id, usage: dict, latency_s: float, n_messages: int):
        if self._buffer is not None:
            self._buffer["model_calls"].append(
                {**usage, "latency_s": round(latency_s, 3), "n_messages": n_messages}
            )
        client = self._client_or_none()
        if client is None or turn_id is None:
            return None
        from datetime import timedelta

        end = datetime.now(timezone.utc)
        start = end - timedelta(seconds=max(latency_s, 0))

        import uuid

        run_id = str(uuid.uuid4())
        self._safe(
            client.create_run,
            name="groq-chat",
            run_type="llm",
            inputs={"n_messages": n_messages},
            outputs={**usage, "latency_s": round(latency_s, 3)},
            parent_run_ids=[turn_id],
            project_name=self.project,
            start_time=start,
            id=run_id,
        )
        self._safe(client.update_run, run_id=run_id, end_time=end)
        return run_id

    def log_tool_call(self, turn_id, name, arguments, verdict, result_chars, latency_s):
        if self._buffer is not None:
            self._buffer["tool_trace"].append(
                {
                    "name": name,
                    "arguments": arguments,
                    "verdict": verdict,
                    "result_chars": result_chars,
                    "latency_s": round(latency_s, 3),
                }
            )
            if verdict == "denied":
                self._buffer["denials"] += 1
            if verdict.startswith("limit"):
                limit = verdict.split(":", 1)[1] if ":" in verdict else verdict
                if limit not in self._buffer["limits_hit"]:
                    self._buffer["limits_hit"].append(limit)
        client = self._client_or_none()
        if client is None or turn_id is None:
            return None
        from datetime import timedelta

        end = datetime.now(timezone.utc)
        start = end - timedelta(seconds=max(latency_s, 0))

        import uuid

        run_id = str(uuid.uuid4())
        self._safe(
            client.create_run,
            name=f"tool:{name}",
            run_type="tool",
            inputs={"arguments": arguments},
            outputs={
                "verdict": verdict,
                "result_chars": result_chars,
                "latency_s": round(latency_s, 3),
            },
            parent_run_ids=[turn_id],
            project_name=self.project,
            start_time=start,
            id=run_id,
        )
        self._safe(client.update_run, run_id=run_id, end_time=end)
        return run_id

    def log_error(self, message: str) -> None:
        if self._buffer is not None:
            self._buffer["errors"].append(message)

    def end_turn(self, turn_id, answer: str, metadata: dict):
        if self._buffer is not None:
            self._buffer["ended_at"] = _utcnow()
            self._buffer["answer"] = answer
            self._buffer["metadata"] = metadata
            self._write_jsonl()
            self._buffer = None
        client = self._client_or_none()
        if client is None or turn_id is None:
            return None
        return self._safe(
            client.update_run,
            run_id=turn_id,
            outputs={"answer": answer},
            extra_metadata=metadata,
        )
