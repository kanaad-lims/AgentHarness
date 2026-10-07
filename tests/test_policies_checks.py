from policies import checks


def _allow(**overrides):
    kwargs = {
        "tool_calls_used": 0,
        "arxiv_calls_used": 0,
        "web_calls_used": 0,
        "attempts_made": 0,
        "already_succeeded": False,
    }
    kwargs.update(overrides)
    return checks.check_tool_call("bash", {"command": "ls"}, **kwargs)


def test_allows_normal_call():
    assert _allow() == (False, None)


def test_blocks_over_total_limit():
    assert _allow(tool_calls_used=99)[0] is True


def test_blocks_duplicate_success_but_allows_failure_retry():
    assert _allow(already_succeeded=True)[0] is True
    assert _allow(attempts_made=99)[0] is True


def test_blocks_arxiv_over_its_limit():
    blocked, message = checks.check_tool_call(
        "arxiv_search",
        {"query": "x"},
        tool_calls_used=0,
        arxiv_calls_used=99,
        web_calls_used=0,
        attempts_made=0,
        already_succeeded=False,
    )
    assert blocked is True
    assert "ArXiv" in message


def test_blocks_web_over_its_limit():
    blocked, message = checks.check_tool_call(
        "web_search",
        {"query": "x"},
        tool_calls_used=0,
        arxiv_calls_used=0,
        web_calls_used=99,
        attempts_made=0,
        already_succeeded=False,
    )
    assert blocked is True
    assert "Web search limit" in message


def test_failure_classifier():
    assert checks.is_tool_failure("bash", "Error: boom") is True
    assert checks.is_tool_failure("bash", "Timed out after 3 seconds.") is True
    assert checks.is_tool_failure("arxiv_search", '{"error": "nope"}') is True
    assert checks.is_tool_failure("bash", "Exit code: 0\nok") is False
    assert checks.is_tool_failure("bash", "Command was not run; the user did not approve it.") is False
    assert checks.is_tool_failure("web_search", "Web error: Web search failed: boom") is True
