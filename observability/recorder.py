"""LangSmith client wrapper. All methods are total: they return None on any
failure (no key, offline, API error) so tracing can never break a turn."""

import os
from contextvars import ContextVar

# (recorder, turn_run_id) for the currently executing turn, if any.
current_turn: ContextVar = ContextVar("boreas_turn", default=None)


class TurnRecorder:
    def __init__(self) -> None:
        self.api_key = os.getenv("LANGSMITH_API_KEY", "")
        self.project = os.getenv("LANGSMITH_PROJECT", "boreas-agent")
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _client_or_none(self):
        if not self.enabled:
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

    def start_turn(self, user_text: str, model: str):
        """Open a chain run for one user turn. Returns the run id or None."""
        client = self._client_or_none()
        if client is None:
            return None
        import uuid

        run_id = str(uuid.uuid4())
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
        client = self._client_or_none()
        if client is None or turn_id is None:
            return None
        from datetime import datetime, timedelta, timezone

        import uuid

        end = datetime.now(timezone.utc)
        start = end - timedelta(seconds=max(latency_s, 0))
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
        client = self._client_or_none()
        if client is None or turn_id is None:
            return None
        from datetime import datetime, timedelta, timezone

        import uuid

        end = datetime.now(timezone.utc)
        start = end - timedelta(seconds=max(latency_s, 0))
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

    def end_turn(self, turn_id, answer: str, metadata: dict):
        client = self._client_or_none()
        if client is None or turn_id is None:
            return None
        return self._safe(
            client.update_run,
            run_id=turn_id,
            outputs={"answer": answer},
            extra_metadata=metadata,
        )
