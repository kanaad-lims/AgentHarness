from types import SimpleNamespace

import agent
from tools import todo_tool


def _tool_call(name, arguments, call_id="call-1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_run_turn_returns_final_model_answer(monkeypatch):
    response = SimpleNamespace(content="Hello!", tool_calls=[])
    monkeypatch.setattr(agent, "call_llm", lambda *_args: response)
    messages = [{"role": "system", "content": agent.SYSTEM_PROMPT}, {"role": "user", "content": "Hi"}]

    answer = agent.run_turn(messages, client=object())

    assert answer == "Hello!"
    assert messages[-1] == {"role": "assistant", "content": "Hello!"}


def test_run_turn_dispatches_tool_then_sends_result_back(monkeypatch):
    monkeypatch.setattr(todo_tool, "_TODOS", [])
    plan = [{"task": "Inspect project", "status": "in_progress"}]
    responses = iter(
        [
            SimpleNamespace(
                content=None,
                tool_calls=[
                    _tool_call(
                        "write_todos",
                        '{"todos":[{"task":"Inspect project","status":"in_progress"}]}',
                    )
                ],
                model_dump=lambda exclude_none=True: {
                    "role": "assistant",
                    "tool_calls": [{"id": "call-1", "type": "function"}],
                },
            ),
            SimpleNamespace(content="Plan saved.", tool_calls=[]),
        ]
    )
    requests = []
    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda messages, client, tools: requests.append((messages, tools)) or next(responses),
    )
    messages = [{"role": "system", "content": agent.SYSTEM_PROMPT}, {"role": "user", "content": "Plan this"}]

    answer = agent.run_turn(messages, client=object())

    assert answer == "Plan saved."
    assert todo_tool.get_todos() == plan
    assert messages[-2]["role"] == "tool"
    assert "Inspect project" in messages[-2]["content"]
    assert requests[0][1] == agent.TOOL_SCHEMAS
    assert "Current plan (authoritative)" in requests[1][0][-1]["content"]


def test_run_turn_enforces_arxiv_limit_for_parallel_tool_calls(monkeypatch):
    searches = []
    monkeypatch.setattr(agent, "DEBUG_TOKEN_USAGE", False)
    monkeypatch.setattr(agent, "MAX_ARXIV_CALLS_PER_TURN", 1)
    monkeypatch.setitem(
        agent.TOOLS,
        "arxiv_search",
        lambda query, **_kwargs: searches.append(query) or [{"title": "Paper"}],
    )
    responses = iter(
        [
            SimpleNamespace(
                content=None,
                tool_calls=[
                    _tool_call("arxiv_search", '{"query":"attention 2026"}', "a"),
                    _tool_call("arxiv_search", '{"query":"transformer attention"}', "b"),
                ],
                model_dump=lambda exclude_none=True: {"role": "assistant", "tool_calls": []},
            ),
            SimpleNamespace(content="Here are the papers.", tool_calls=[]),
        ]
    )
    monkeypatch.setattr(agent, "call_llm", lambda *_args: next(responses))
    messages = [{"role": "system", "content": agent.SYSTEM_PROMPT}]

    agent.run_turn(messages, client=object())

    assert searches == ["attention 2026"]
    tool_results = [message["content"] for message in messages if message.get("role") == "tool"]
    assert any("Arxiv search limit reached" in result for result in tool_results)


def test_duplicate_tool_call_is_not_executed_twice(monkeypatch):
    calls = []
    monkeypatch.setattr(agent, "DEBUG_TOKEN_USAGE", False)
    monkeypatch.setitem(agent.TOOLS, "arxiv_search", lambda query: calls.append(query) or "results")
    duplicate_call = _tool_call("arxiv_search", '{"query":"attention"}')
    responses = iter(
        [
            SimpleNamespace(
                content=None,
                tool_calls=[duplicate_call, duplicate_call],
                model_dump=lambda exclude_none=True: {"role": "assistant", "tool_calls": []},
            ),
            SimpleNamespace(content="Done.", tool_calls=[]),
        ]
    )
    monkeypatch.setattr(agent, "call_llm", lambda *_args: next(responses))
    messages = [{"role": "system", "content": agent.SYSTEM_PROMPT}]

    agent.run_turn(messages, client=object())

    assert calls == ["attention"]
    tool_results = [message["content"] for message in messages if message.get("role") == "tool"]
    assert tool_results[1].startswith("Duplicate tool call skipped")


def test_failed_tool_call_may_retry_identical_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(agent, "DEBUG_TOKEN_USAGE", False)
    monkeypatch.setattr(agent, "MAX_SAME_CALL_ATTEMPTS", 3)
    monkeypatch.setitem(
        agent.TOOLS, "bash", lambda command: calls.append(command) or "Error: boom"
    )
    failing_call = _tool_call("bash", '{"command":"pwd"}')
    responses = iter(
        [
            SimpleNamespace(
                content=None,
                tool_calls=[failing_call],
                model_dump=lambda exclude_none=True: {"role": "assistant", "tool_calls": []},
            ),
            SimpleNamespace(
                content=None,
                tool_calls=[failing_call],
                model_dump=lambda exclude_none=True: {"role": "assistant", "tool_calls": []},
            ),
            SimpleNamespace(content="Gave up.", tool_calls=[]),
        ]
    )
    monkeypatch.setattr(agent, "call_llm", lambda *_args: next(responses))
    messages = [{"role": "system", "content": agent.SYSTEM_PROMPT}]

    agent.run_turn(messages, client=object())

    assert calls == ["pwd", "pwd"]
    tool_results = [message["content"] for message in messages if message.get("role") == "tool"]
    assert all(result == "Error: boom" for result in tool_results)


def test_format_arxiv_results_shortens_abstracts_and_keeps_links():
    papers = [
        {
            "title": "Full Title One",
            "authors": "A One, B Two, C Three, D Four",
            "published": "2026-10-01",
            "summary": "word " * 500,
            "url": "http://arxiv.org/abs/0001",
            "pdf_url": "",
        }
    ]

    text = agent._format_arxiv_results(papers)

    assert "Full Title One" in text
    assert "http://arxiv.org/abs/0001" in text
    assert "et al." in text
    assert "[truncated]" in text
    assert len(text) <= agent.ARXIV_FORMAT_MAX_CHARS + 100
