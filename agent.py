"""LangGraph agent loop reusing the pure-Python tool implementation."""

import json
import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from llm import DEBUG_TOKEN_USAGE, MODEL, call_llm, create_client
from policies import (
    MAX_ARXIV_CALLS_PER_TURN,
    MAX_MODEL_CALLS_PER_TURN,
    MAX_SAME_CALL_ATTEMPTS,
    MAX_TOOL_CALLS_PER_TURN,
    MAX_TOOL_RESULT_CHARS,
    RECURSION_LIMIT,
    REQUIRE_BASH_APPROVAL,
)
from prompts import SYSTEM_PROMPT
from prompts.messages import (
    ARXIV_LIMIT_MESSAGE,
    BASH_APPROVAL_PROMPT,
    BASH_REQUEST_HEADER,
    DUPLICATE_TOOL_MESSAGE,
    MODEL_LIMIT_ANSWER,
    RETRY_LIMIT_MESSAGE,
    TOOL_LIMIT_MESSAGE,
    plan_message,
)
from tools.browser_tool import run_browser
from tools.registry import TOOL_SCHEMAS, TOOLS
from tools.todo_tool import get_todos


class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    model_calls: int
    tool_calls: int
    arxiv_calls: int
    seen_keys: Annotated[list[str], operator.add]
    attempts: dict
    pending: list


def _is_tool_failure(name: str, result: str) -> bool:
    """Transient execution failures may be retried; successes must reformulate."""
    if not isinstance(result, str):
        return False
    if (
        result.startswith("Error:")
        or result.startswith("Tool error")
        or result.startswith("Timed out after")
        or result.startswith("ArXiv error:")
    ):
        return True
    stripped = result.strip()
    if stripped.startswith("{") and '"error"' in stripped:
        return True
    return False


def _approx_tokens(text: str) -> int:
    return (len(text) + 3) // 4


def _tool_call_fields(tool_call) -> tuple:
    """Return (call_id, name, arguments_json_string) for object or dict calls."""
    if hasattr(tool_call, "function"):
        function = tool_call.function
        if isinstance(function, dict):
            name = function.get("name", "")
            args_text = function.get("arguments", "{}")
        else:
            name = getattr(function, "name", "")
            args_text = getattr(function, "arguments", "{}")
        call_id = getattr(tool_call, "id", None)
        if not isinstance(args_text, str):
            args_text = json.dumps(args_text, ensure_ascii=False, default=str)
        return call_id, name, args_text or "{}"
    if isinstance(tool_call, dict):
        function = tool_call.get("function", {})
        if isinstance(function, dict):
            return (
                tool_call.get("id"),
                function.get("name", ""),
                function.get("arguments", "{}") or "{}",
            )
        return (
            tool_call.get("id"),
            tool_call.get("name", ""),
            json.dumps(tool_call.get("args", {}), ensure_ascii=False, default=str),
        )
    return None, "", "{}"


def _execute_tool_call(tool_call) -> tuple[str, str, dict]:
    """Validate and execute one model-requested tool call."""
    _, name, args_text = _tool_call_fields(tool_call)
    try:
        arguments = json.loads(args_text or "{}")
        if not isinstance(arguments, dict):
            return name, "Error: tool arguments must be a JSON object.", {}
    except (TypeError, json.JSONDecodeError) as error:
        return name, f"Error: invalid arguments for {name}: {error}", {}

    if DEBUG_TOKEN_USAGE:
        args_text = json.dumps(arguments, ensure_ascii=False, default=str)
        print(
            f"[tool-debug] selected {name}; arguments "
            f"{len(args_text):,} chars (~{_approx_tokens(args_text):,} tokens)"
        )

    if name not in TOOLS:
        return name, f"Error: unknown tool '{name}'.", arguments

    if name == "bash" and REQUIRE_BASH_APPROVAL:
        print(f"\n{BASH_REQUEST_HEADER}\n{arguments.get('command', '')}")
        approved = input(f"{BASH_APPROVAL_PROMPT}").strip().lower()
        if approved not in {"y", "yes"}:
            return name, "Command was not run; the user did not approve it.", arguments

    started = time.perf_counter()
    try:
        result = TOOLS[name](**arguments)
    except Exception as error:
        result = f"Tool error ({name}): {type(error).__name__}: {error}"

    if not isinstance(result, str):
        result = json.dumps(result, ensure_ascii=False, default=str)
    if len(result) > MAX_TOOL_RESULT_CHARS:
        result = result[:MAX_TOOL_RESULT_CHARS] + "\n... tool result truncated to limit"
    if DEBUG_TOKEN_USAGE:
        print(
            f"[tool-debug] {name} completed in {time.perf_counter() - started:.2f}s; "
            f"result {len(result):,} chars (~{_approx_tokens(result):,} tokens)"
        )
    return name, result, arguments


def build_graph(client):
    """Build the LangGraph workflow around the existing tool implementation."""

    def call_model(state: AgentState) -> dict:
        current_plan = get_todos()
        request_messages = list(state["messages"])
        if current_plan:
            request_messages.append(plan_message(current_plan))

        response = call_llm(request_messages, client, TOOL_SCHEMAS)
        requested_tools = list(getattr(response, "tool_calls", None) or [])
        try:
            dumped = response.model_dump(exclude_none=True)
        except Exception:
            dumped = {
                "role": "assistant",
                "content": getattr(response, "content", None) or "",
            }
        return {
            "messages": [dumped],
            "model_calls": state["model_calls"] + 1,
            "pending": requested_tools,
        }

    def run_tools(state: AgentState) -> dict:
        pending = state.get("pending", []) or []
        tool_messages = []
        new_seen: list[str] = []
        tool_calls_used = state.get("tool_calls", 0)
        arxiv_used = state.get("arxiv_calls", 0)
        successful = set(state.get("seen_keys", []))
        attempts = dict(state.get("attempts", {}) or {})

        for tool_call in pending:
            call_id, name, args_text = _tool_call_fields(tool_call)
            try:
                arguments = json.loads(args_text or "{}")
            except (TypeError, json.JSONDecodeError):
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            call_key = json.dumps(
                [name, arguments], sort_keys=True, ensure_ascii=False
            )
            attempts_made = attempts.get(call_key, 0)
            if tool_calls_used >= MAX_TOOL_CALLS_PER_TURN:
                result = TOOL_LIMIT_MESSAGE
            elif call_key in successful:
                result = DUPLICATE_TOOL_MESSAGE
            elif attempts_made >= MAX_SAME_CALL_ATTEMPTS:
                result = RETRY_LIMIT_MESSAGE
            elif name == "arxiv_search" and arxiv_used >= MAX_ARXIV_CALLS_PER_TURN:
                result = ARXIV_LIMIT_MESSAGE
            else:
                tool_calls_used += 1
                attempts[call_key] = attempts_made + 1
                if name == "arxiv_search":
                    arxiv_used += 1
                dispatched_name, result, _ = _execute_tool_call(tool_call)
                name = dispatched_name
                if not _is_tool_failure(name, result):
                    successful.add(call_key)
                    new_seen.append(call_key)

            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": result,
                }
            )

        return {
            "messages": tool_messages,
            "tool_calls": tool_calls_used,
            "arxiv_calls": arxiv_used,
            "seen_keys": new_seen,
            "attempts": attempts,
            "pending": [],
        }

    def stop_at_limit(_state: AgentState) -> dict:
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": MODEL_LIMIT_ANSWER,
                }
            ],
            "pending": [],
        }

    def route_after_model(state: AgentState) -> str:
        if not (state.get("pending", []) or []):
            return "end"
        return "tools"

    def route_after_tools(state: AgentState) -> str:
        if state["model_calls"] >= MAX_MODEL_CALLS_PER_TURN:
            return "limit"
        return "model"

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", run_tools)
    graph.add_node("limit", stop_at_limit)
    graph.add_edge(START, "model")
    graph.add_conditional_edges(
        "model", route_after_model, {"tools": "tools", "end": END}
    )
    graph.add_conditional_edges(
        "tools", route_after_tools, {"model": "model", "limit": "limit"}
    )
    graph.add_edge("limit", END)
    return graph.compile()


def run_turn(messages: list[dict], client) -> str:
    """Run one user turn through the LangGraph workflow."""
    app = build_graph(client)
    result = app.invoke(
        {
            "messages": list(messages),
            "model_calls": 0,
            "tool_calls": 0,
            "arxiv_calls": 0,
            "seen_keys": [],
            "attempts": {},
            "pending": [],
        },
        config={"recursion_limit": RECURSION_LIMIT},
    )
    messages[:] = result["messages"]
    for message in reversed(result["messages"]):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return message.get("content", "") or ""
    return ""


def main() -> None:
    """Start the interactive agent."""
    client = None
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"Agent ({MODEL}). Type /bye to exit.\n")

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
            command = user_input[len("/browser"):].strip()
            print(f"\n{run_browser(command)}\n")
            continue

        if client is None:
            client = create_client()
        messages.append({"role": "user", "content": user_input})
        try:
            answer = run_turn(messages, client)
            print(f"\nAgent: {answer}\n")
        except Exception as error:
            print(f"\nRequest failed: {error}\n")


if __name__ == "__main__":
    main()
