from evals.metrics import grade_turn


def _turn(**overrides):
    turn = {
        "turn_id": "t1",
        "answer": "See http://arxiv.org/abs/0001 for details.",
        "model_calls": [{"prompt_tokens": 100, "completion_tokens": 10}],
        "tool_trace": [
            {"name": "arxiv_search", "arguments": {}, "verdict": "executed",
             "result_chars": 100, "latency_s": 1.0}
        ],
        "limits_hit": [],
        "denials": 0,
        "errors": [],
    }
    turn.update(overrides)
    return turn


def test_grade_turn_passes_healthy_turn():
    grades = grade_turn(_turn())

    assert all(passed for passed, _ in grades.values())


def test_budget_metric_fails_on_overuse():
    grades = grade_turn(_turn(model_calls=[{"prompt_tokens": 99999, "completion_tokens": 1}]))

    assert grades["within_budgets"][0] is False


def test_citation_metric_fails_without_url():
    grades = grade_turn(_turn(answer="Here are some papers I remember."))

    assert grades["cites_sources_when_factual"][0] is False


def test_recovery_metric_fails_on_errors_or_empty_answer():
    assert grade_turn(_turn(errors=["boom"]))["recovered_or_declared"][0] is False
    assert grade_turn(_turn(answer="  "))["recovered_or_declared"][0] is False
