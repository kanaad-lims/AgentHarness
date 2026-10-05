"""LangGraph agent loop reusing the pure-Python tool implementation."""

import json
import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from llm import DEBUG_TOKEN_USAGE, MODEL, call_llm, create_client
from presenters.arxiv import format_arxiv_results
from presenters.bash import format_bash_result
from tools.arxiv_search_tool import ARXIV_SEARCH_TOOL_SCHEMA, arxiv_search
from tools.bash_tool import BASH_TOOL_SCHEMA, run_bash
from tools.browser_tool import run_browser
from tools.todo_tool import TODO_TOOL_SCHEMA, get_todos, write_todos

MAX_MODEL_CALLS_PER_TURN = 8
MAX_TOOL_CALLS_PER_TURN = 6
MAX_ARXIV_CALLS_PER_TURN = 3
MAX_ARXIV_RESULTS_PER_CALL = 5
MAX_TOOL_RESULT_CHARS = 5000
MAX_SAME_CALL_ATTEMPTS = 3

SYSTEM_PROMPT = """You are a helpful and careful coding assistant.

Help the user understand, write, debug, and improve software. Give accurate,
practical answers and explain important trade-offs briefly. When writing code,
provide complete, runnable examples when appropriate, use clear names, and
follow the language and conventions of the user's project. Do not invent files,
APIs, test results, or actions you have not performed. Ask a concise clarifying
question when essential requirements are missing; otherwise state reasonable
assumptions and proceed. Point out security, data-loss, or compatibility risks
before recommending risky changes. Keep responses focused on the user's request.
Use arxiv_search for recent arXiv papers or research topics. Use write_todos only
for multi-step tasks needing 2 or more tool actions in this turn; skip planning
for single tool calls, including retries. When planning, send the complete list
on each update and keep exactly one unfinished task in_progress.
RULE: NEVER execute the bash tool if asked to delete or modify any file.
RULE: Keep answers focused and under 700 tokens where practical.
"""

TOOL_SCHEMAS = [BASH_TOOL_SCHEMA, ARXIV_SEARCH_TOOL_SCHEMA, TODO_TOOL_SCHEMA]


def _search_arxiv(**arguments):
    """Bound arXiv result count and return compact text for the model."""
    requested = arguments.get("max_results", 5)
    if not isinstance(requested, int) or isinstance(requested, bool):
        requested = 5
    arguments["max_results"] = min(max(requested, 1), MAX_ARXIV_RESULTS_PER_CALL)
    return format_arxiv_results(arxiv_search(**arguments))


TOOLS = {
    "bash": run_bash,
    "arxiv_search": _search_arxiv,
    "write_todos": write_todos,
}


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


def _tool_result(tool_call) -> tuple[str, str, dict]:
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

    if name == "bash":
        print(f"\nBash command requested:\n{arguments.get('command', '')}")
        approved = input("Run this command? [y/N]: ").strip().lower()
        if approved not in {"y", "yes"}:
            return name, "Command was not run; the user did not approve it.", arguments

    started = time.perf_counter()
    try:
        result = TOOLS[name](**arguments)
    except Exception as error:
        result = f"Tool error ({name}): {type(error).__name__}: {error}"

    if name == "bash" and isinstance(result, dict):
        result = format_bash_result(result)
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


def _tool_result_metadata(tool_call) -> tuple[str, dict]:
    """Read a tool call's name and arguments for limit/deduplication checks."""
    _, name, args_text = _tool_call_fields(tool_call)
    try:
        arguments = json.loads(args_text or "{}")
    except (TypeError, json.JSONDecodeError):
        arguments = {}
    return name, arguments if isinstance(arguments, dict) else {}


def build_graph(client):
    """Build the LangGraph workflow around the existing tool implementation."""

    def call_model(state: AgentState) -> dict:
        current_plan = get_todos()
        request_messages = list(state["messages"])
        if current_plan:
            request_messages.append(
                {
                    "role": "system",
                    "content": "Current plan (authoritative): <todos>"
                    f"{json.dumps(current_plan, ensure_ascii=False)}</todos>",
                },
            )

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
                result = "Tool-call limit reached for this user request."
            elif call_key in successful:
                result = "Duplicate tool call skipped; use the earlier result in this conversation."
            elif attempts_made >= MAX_SAME_CALL_ATTEMPTS:
                result = (
                    "Retry limit reached for this exact tool call. "
                    "Change the arguments or use results already in the conversation."
                )
            elif name == "arxiv_search" and arxiv_used >= MAX_ARXIV_CALLS_PER_TURN:
                result = (
                    "ArXiv search limit reached for this request. Use the arXiv results "
                    "already in the conversation to answer."
                )
            else:
                tool_calls_used += 1
                attempts[call_key] = attempts_made + 1
                if name == "arxiv_search":
                    arxiv_used += 1
                dispatched_name, result, _ = _tool_result(tool_call)
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
                    "content": (
                        "I reached the model-call limit for this request. "
                        "Please ask me to continue if more work is needed."
                    ),
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
        config={"recursion_limit": 2 * MAX_MODEL_CALLS_PER_TURN + 10},
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
