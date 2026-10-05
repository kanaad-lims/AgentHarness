from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import build_agent
from tools import todo_tool


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, _messages):
        return next(self.responses)


def test_graph_returns_model_answer_without_tool_calls(monkeypatch):
    monkeypatch.setattr(todo_tool, "_TODOS", [])
    model = FakeModel([AIMessage(content="Hello from the graph.")])
    graph = build_agent(model=model)

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="Say hello")],
            "model_calls": 0,
            "tool_calls": 0,
        }
    )

    assert result["messages"][-1].content == "Hello from the graph."
    assert result["model_calls"] == 1


def test_graph_executes_todo_tool_and_continues_to_final_answer(monkeypatch):
    monkeypatch.setattr(todo_tool, "_TODOS", [])
    plan = [
        {"task": "Inspect the project", "status": "in_progress"},
        {"task": "Summarize findings", "status": "pending"},
    ]
    model = FakeModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_todos",
                        "args": {"todos": plan},
                        "id": "todo-call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Plan saved; I’ll inspect the project."),
        ]
    )
    graph = build_agent(model=model)

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="Inspect and summarize this project")],
            "model_calls": 0,
            "tool_calls": 0,
        }
    )

    assert todo_tool.get_todos() == plan
    assert any(isinstance(message, ToolMessage) for message in result["messages"])
    assert result["messages"][-1].content == "Plan saved; I’ll inspect the project."
    assert result["model_calls"] == 2
