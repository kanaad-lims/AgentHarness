import json

from tools import todo_tool


def test_write_todos_replaces_plan_and_returns_todos_context(monkeypatch):
    monkeypatch.setattr(todo_tool, "_TODOS", [])
    todos = [
        {"task": "Inspect project", "status": "completed"},
        {"task": "Implement changes", "status": "in_progress"},
        {"task": "Run tests", "status": "pending"},
    ]

    result = todo_tool.write_todos(todos)

    assert todo_tool.get_todos() == todos
    assert result == f"<todos>{json.dumps(todos, ensure_ascii=False)}</todos>"


def test_write_todos_rejects_multiple_in_progress_tasks(monkeypatch):
    monkeypatch.setattr(todo_tool, "_TODOS", [])

    result = todo_tool.write_todos(
        [
            {"task": "First task", "status": "in_progress"},
            {"task": "Second task", "status": "in_progress"},
        ]
    )

    assert result.startswith("Error:")
    assert todo_tool.get_todos() == []


def test_write_todos_allows_completed_plan(monkeypatch):
    monkeypatch.setattr(todo_tool, "_TODOS", [])
    todos = [{"task": "Done", "status": "completed"}]

    todo_tool.write_todos(todos)

    assert todo_tool.get_todos() == todos
