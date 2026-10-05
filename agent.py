"""LangGraph-based orchestration loop for the Droid coding assistant."""

import json
import os
import time
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from llm import DEBUG_TOKEN_USAGE, MODEL, SYSTEM_PROMPT
from tools.arxiv_search_tool import arxiv_search as search_arxiv
from tools.bash_tool import run_bash
from tools.browser_tool import run_browser
from tools.todo_tool import get_todos, write_todos

load_dotenv()

MAX_MODEL_CALLS_PER_TURN = 8
MAX_TOOL_CALLS_PER_TURN = 6
MAX_COMPLETION_TOKENS = int(os.getenv("MAX_COMPLETION_TOKENS", "800"))


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    model_calls: int
    tool_calls: int


class BashArgs(BaseModel):
    command: str = Field(description="Bash command to run in the project directory.")


class ArxivSearchArgs(BaseModel):
    query: str = Field(description="Research topic or arXiv query.")
    max_results: int = Field(default=5, ge=1, le=10)


class TodoEntry(BaseModel):
    task: str = Field(description="A concise task in the plan.")
    status: Literal["pending", "in_progress", "completed"]


class WriteTodosArgs(BaseModel):
    todos: list[TodoEntry]


@tool("bash", args_schema=BashArgs)
def bash_tool(command: str) -> str:
    """Run a Bash command. Commands execute on the host and are not sandboxed."""
    print(f"\nBash command requested:\n{command}")
    approval = input("Run this command? [y/N]: ").strip().lower()
    if approval not in {"y", "yes"}:
        return "Command was not run; the user did not approve it."
    return run_bash(command)


@tool("arxiv_search", args_schema=ArxivSearchArgs)
def arxiv_search_tool(query: str, max_results: int = 5) -> str:
    """Find the newest arXiv papers matching a topic and return their metadata."""
    result = search_arxiv(query=query, max_results=max_results)
    return json.dumps(result, ensure_ascii=False)


@tool("write_todos", args_schema=WriteTodosArgs)
def write_todos_tool(todos: list[TodoEntry]) -> str:
    """Create or replace the complete plan for the current task."""
    normalized = [
        todo.model_dump() if isinstance(todo, BaseModel) else todo
        for todo in todos
    ]
    return write_todos(normalized)


TOOLS: list[StructuredTool] = [bash_tool, arxiv_search_tool, write_todos_tool]
TOOLS_BY_NAME = {registered_tool.name: registered_tool for registered_tool in TOOLS}


def _approx_tokens(text: str) -> int:
    """Rough character-based estimate; exact tokenization is provider-specific."""
    return (len(text) + 3) // 4


def _render_todos() -> str:
    return (
        "Current plan (authoritative): <todos>"
        f"{json.dumps(get_todos(), ensure_ascii=False)}"
        "</todos>"
    )


def build_agent(model=None):
    """Build and compile the LangGraph agent; model can be injected for tests."""
    if model is None:
        from langchain_groq import ChatGroq

        model = ChatGroq(
            model=MODEL,
            temperature=0,
            max_tokens=MAX_COMPLETION_TOKENS,
        )

    model_with_tools = model.bind_tools(TOOLS)

    def call_model(state: AgentState) -> dict:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            SystemMessage(content=_render_todos()),
            *state["messages"],
        ]

        if DEBUG_TOKEN_USAGE:
            serialized = "\n".join(
                f"{message.type}: {message.content}" for message in messages
            )
            print(
                f"[token-debug] LangGraph model call {state['model_calls'] + 1}; "
                f"{len(serialized):,} chars (~{_approx_tokens(serialized):,} tokens)"
            )

        started = time.perf_counter()
        response = model_with_tools.invoke(messages)
        elapsed = time.perf_counter() - started

        if DEBUG_TOKEN_USAGE:
            usage = getattr(response, "usage_metadata", None) or {}
            if usage:
                print(
                    "[token-debug] actual usage: "
                    f"input={usage.get('input_tokens', 'n/a')}, "
                    f"output={usage.get('output_tokens', 'n/a')}, "
                    f"total={usage.get('total_tokens', 'n/a')}; latency={elapsed:.2f}s"
                )
            else:
                print(f"[token-debug] usage unavailable; latency={elapsed:.2f}s")

        return {
            "messages": [response],
            "model_calls": state["model_calls"] + 1,
        }

    def dispatch_tools(state: AgentState) -> dict:
        assistant_message = state["messages"][-1]
        tool_messages = []
        calls_used = state["tool_calls"]

        for tool_call in assistant_message.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            started = time.perf_counter()

            if calls_used >= MAX_TOOL_CALLS_PER_TURN:
                result = "Tool-call limit reached for this user request."
            elif name not in TOOLS_BY_NAME:
                result = f"Error: unknown tool '{name}'."
            else:
                calls_used += 1
                if DEBUG_TOKEN_USAGE:
                    args_text = json.dumps(args, ensure_ascii=False, default=str)
                    print(
                        f"[tool-debug] selected {name}; arguments "
                        f"~{_approx_tokens(args_text):,} tokens"
                    )
                try:
                    result = TOOLS_BY_NAME[name].invoke(args)
                except Exception as error:
                    result = f"Tool error ({name}): {error}"

            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False, default=str)

            if DEBUG_TOKEN_USAGE:
                print(
                    f"[tool-debug] {name} completed in "
                    f"{time.perf_counter() - started:.2f}s; "
                    f"result {len(result):,} chars (~{_approx_tokens(result):,} tokens)"
                )

            tool_messages.append(
                ToolMessage(content=result, tool_call_id=tool_call["id"], name=name)
            )

        return {"messages": tool_messages, "tool_calls": calls_used}

    def route_after_model(state: AgentState) -> str:
        latest = state["messages"][-1]
        if not getattr(latest, "tool_calls", None):
            return "end"
        if state["model_calls"] >= MAX_MODEL_CALLS_PER_TURN:
            return "limit"
        return "tools"

    def stop_at_limit(_state: AgentState) -> dict:
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I reached the model-call limit for this request. "
                        "Please ask me to continue if more work is needed."
                    )
                )
            ]
        }

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", dispatch_tools)
    graph.add_node("limit", stop_at_limit)
    graph.add_edge(START, "model")
    graph.add_conditional_edges(
        "model",
        route_after_model,
        {"tools": "tools", "limit": "limit", "end": END},
    )
    graph.add_edge("tools", "model")
    graph.add_edge("limit", END)
    return graph.compile()


def main() -> None:
    """Run an interactive LangGraph-backed chat session."""
    load_dotenv()
    app = None
    conversation: list[AnyMessage] = []
    print(f"LangGraph Agent ({MODEL}). Type /bye to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"/bye", "/exit", "/quit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        if user_input == "/browser" or user_input.startswith("/browser "):
            browser_command = user_input[len("/browser"):].strip()
            print(f"\n{run_browser(browser_command)}\n")
            continue

        if app is None:
            app = build_agent()

        conversation.append(HumanMessage(content=user_input))
        try:
            result = app.invoke(
                {"messages": conversation, "model_calls": 0, "tool_calls": 0},
                config={"recursion_limit": 20},
            )
            conversation = result["messages"]
            final_message = conversation[-1]
            answer = final_message.content
            if isinstance(answer, list):
                answer = "\n".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in answer
                )
            print(f"\nAgent: {answer}\n")
        except Exception as error:
            print(f"\nRequest failed: {error}\n")


if __name__ == "__main__":
    main()
