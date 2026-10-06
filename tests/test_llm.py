from copy import deepcopy
from types import SimpleNamespace

import llm


def test_call_llm_uses_native_groq_sdk_and_returns_assistant_message(monkeypatch):
    requests = []
    response_message = SimpleNamespace(content="Hello", tool_calls=[])

    class FakeCompletions:
        def create(self, **kwargs):
            requests.append(deepcopy(kwargs))
            return SimpleNamespace(
                choices=[SimpleNamespace(message=response_message)],
                usage=None,
            )

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setattr(llm, "DEBUG_TOKEN_USAGE", False)

    result = llm.call_llm(
        [{"role": "user", "content": "Hi"}],
        client,
        tools=[{"type": "function", "function": {"name": "example"}}],
    )

    assert result is response_message
    assert requests[0]["model"] == llm.MODEL
    assert requests[0]["tools"] == [{"type": "function", "function": {"name": "example"}}]
    assert requests[0]["tool_choice"] == "auto"
    if llm.MAX_COMPLETION_TOKENS is None:
        assert "max_completion_tokens" not in requests[0]
    else:
        assert requests[0]["max_completion_tokens"] == llm.MAX_COMPLETION_TOKENS
    assert requests[0]["messages"] == [{"role": "user", "content": "Hi"}]


def test_create_client_uses_groq_sdk(monkeypatch):
    created = object()
    monkeypatch.setattr(llm, "Groq", lambda: created)

    assert llm.create_client() is created
