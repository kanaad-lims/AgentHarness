from copy import deepcopy
from types import SimpleNamespace

import llm


def test_chat_sends_history_and_prints_reply(monkeypatch, capsys):
    requests = []

    class FakeCompletions:
        def create(self, **kwargs):
            # The chat loop mutates its history after each API call.
            requests.append(deepcopy(kwargs))
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Hello there!", tool_calls=[])
                    )
                ]
            )

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
    assert requests[0]["tools"] == [llm.BASH_TOOL_SCHEMA]
    assert requests[0]["tool_choice"] == "auto"
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


def test_bash_tool_result_is_shown_and_sent_back_to_model(monkeypatch, capsys):
    requests = []
    tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(
            name="bash",
            arguments='{"command":"pwd"}',
        ),
    )
    assistant_tool_request = SimpleNamespace(
        content=None,
        tool_calls=[tool_call],
        model_dump=lambda exclude_none=True: {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
                }
            ],
        },
    )

    class FakeCompletions:
        def create(self, **kwargs):
            requests.append(deepcopy(kwargs))
            message = assistant_tool_request if len(requests) == 1 else SimpleNamespace(
                content="You are in the project directory.", tool_calls=[]
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeGroq:
        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    answers = iter(["Show the current directory", "y", "/exit"])
    monkeypatch.setattr(llm, "Groq", FakeGroq)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr(llm, "run_bash", lambda command: "Exit code: 0\n/work/project")

    llm.main()

    output = capsys.readouterr().out
    assert "Bash output:" in output
    assert "/work/project" in output
    assert "You are in the project directory." in output
    assert len(requests) == 2
    assert requests[1]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "name": "bash",
        "content": "Exit code: 0\n/work/project",
    }
