from copy import deepcopy
from types import SimpleNamespace

import llm


def test_chat_sends_history_and_prints_streamed_reply(monkeypatch, capsys):
    requests = []

    def make_chunk(content):
        return SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
        )

    class FakeCompletions:
        def create(self, **kwargs):
            # The chat loop mutates its history after each API call.
            requests.append(deepcopy(kwargs))
            return iter([make_chunk("Hello"), make_chunk(" there!")])

    class FakeGroq:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    answers = iter(["Hi", "How are you?", "/exit"])
    monkeypatch.setattr(llm, "Groq", FakeGroq)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setenv("GROQ_MODEL", "test-model")

    llm.main()

    output = capsys.readouterr().out
    assert "Hello there!" in output
    assert len(requests) == 2
    assert requests[0]["model"] == "test-model"
    assert requests[0]["messages"] == [
        {"role": "system", "content": llm.SYSTEM_PROMPT},
        {"role": "user", "content": "Hi"},
    ]
    assert requests[1]["messages"] == [
        {"role": "system", "content": llm.SYSTEM_PROMPT},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello there!"},
        {"role": "user", "content": "How are you?"},
    ]
