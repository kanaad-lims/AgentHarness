"""In-memory planning/todo tool for the current agent session."""

import json
from typing import Any


TODO_STATUSES = {"pending", "in_progress", "completed"}
_TODOS: list[dict[str, str]] = []

TODO_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_todos",
        "description": (
            "Create or replace the current task plan. Send the complete list "
            "on every update. Use one in_progress task while work remains."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["task", "status"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["todos"],
            "additionalProperties": False,
        },
    },
}


def get_todos() -> list[dict[str, str]]:
    """Return a copy of the current plan."""
    return [todo.copy() for todo in _TODOS]


def write_todos(todos: list[dict[str, Any]]) -> str:
    """Replace the current plan after validating task names and statuses."""
    if not isinstance(todos, list):
        return "Error: todos must be a list."

    updated: list[dict[str, str]] = []
    for index, todo in enumerate(todos, start=1):
        if not isinstance(todo, dict):
            return f"Error: todo #{index} must be an object."

        task = todo.get("task")
        status = todo.get("status")
        if not isinstance(task, str) or not task.strip():
            return f"Error: todo #{index} needs a non-empty task."
        if status not in TODO_STATUSES:
            return (
                f"Error: todo #{index} has invalid status {status!r}; "
                f"use one of {', '.join(sorted(TODO_STATUSES))}."
            )

        updated.append({"task": task.strip(), "status": status})

    active_count = sum(todo["status"] == "in_progress" for todo in updated)
    unfinished = any(todo["status"] != "completed" for todo in updated)
    if active_count > 1 or (unfinished and active_count != 1):
        return "Error: keep exactly one task in_progress while tasks remain."

    _TODOS[:] = updated
    return f"<todos>{json.dumps(get_todos(), ensure_ascii=False)}</todos>"
