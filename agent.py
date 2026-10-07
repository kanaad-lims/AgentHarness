"""LangGraph agent loop reusing the pure-Python tool implementation."""

import json
import operator
import time
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from llm import DEBUG_TOKEN_USAGE, MODEL, call_llm, create_client
from policies import (
    MAX_MODEL_CALLS_PER_TURN,
    MAX_TOOL_RESULT_CHARS,
    RECURSION_LIMIT,
    REQUIRE_BASH_APPROVAL,
)
from prompts import SYSTEM_PROMPT
from prompts.messages import (
    BASH_APPROVAL_PROMPT,
    BASH_REQUEST_HEADER,
    MODEL_LIMIT_ANSWER,
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
    web_calls: int
    seen_keys: Annotated[list[str], operator.add]
    attempts: dict
    pending: list


from policies.checks import (
    approx_tokens,
    call_key,
    check_tool_call,
    is_tool_failure,
    parse_tool_arguments,
    tool_call_fields,
)


def _execute_tool_call(tool_call) -> tuple[str, str, dict]:
    """Validate and execute one model-requested tool call."""
    _, name, args_text = tool_call_fields(tool_call)
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
            f"{len(args_text):,} chars (~{approx_tokens(args_text):,} tokens)"
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
            f"result {len(result):,} chars (~{approx_tokens(result):,} tokens)"
        )
    return name, result, arguments


def build_graph(client, on_phase=None):
    """Build the LangGraph workflow around the existing tool implementation."""
    emit = on_phase or (lambda _phase: None)

    def call_model(state: AgentState) -> dict:
        current_plan = get_todos()
        emit("thinking")
        request_messages = list(state["messages"])
        if current_plan:
            emit("planning")
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
        if pending:
            emit("selecting")
        tool_messages = []
        new_seen: list[str] = []
        tool_calls_used = state.get("tool_calls", 0)
        arxiv_used = state.get("arxiv_calls", 0)
        web_used = state.get("web_calls", 0)
        successful = set(state.get("seen_keys", []))
        attempts = dict(state.get("attempts", {}) or {})

        for tool_call in pending:
            call_id, name, args_text = tool_call_fields(tool_call)
            arguments = parse_tool_arguments(args_text)
            key = call_key(name, arguments)
            attempts_made = attempts.get(key, 0)
            blocked, message = check_tool_call(
                name,
                arguments,
                tool_calls_used=tool_calls_used,
                arxiv_calls_used=arxiv_used,
                web_calls_used=web_used,
                attempts_made=attempts_made,
                already_succeeded=key in successful,
            )
            if blocked:
                result = message
            else:
                tool_calls_used += 1
                attempts[key] = attempts_made + 1
                if name == "arxiv_search":
                    arxiv_used += 1
                if name == "web_search":
                    web_used += 1
                dispatched_name, result, _ = _execute_tool_call(tool_call)
                name = dispatched_name
                if not is_tool_failure(name, result):
                    successful.add(key)
                    new_seen.append(key)

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
            "web_calls": web_used,
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


def run_turn(messages: list[dict], client, on_phase=None) -> str:
    """Run one user turn through the LangGraph workflow."""
    emit = on_phase or (lambda _phase: None)
    app = build_graph(client, on_phase=emit)
    result = app.invoke(
        {
            "messages": list(messages),
            "model_calls": 0,
            "tool_calls": 0,
            "arxiv_calls": 0,
            "web_calls": 0,
            "seen_keys": [],
            "attempts": {},
            "pending": [],
        },
        config={"recursion_limit": RECURSION_LIMIT},
    )
    messages[:] = result["messages"]
    emit("accepted")
    for message in reversed(result["messages"]):
        if isinstance(message, dict) and message.get("role") == "assistant":
            return message.get("content", "") or ""
    return ""


def main() -> None:
    """Start the orchestrator agent."""
    import uuid

    from rich.console import Console

    from cli.prompt import create_session
    from cli.ui import (
        APP_VERSION,
        HELP_TEXT,
        render_splash,
        show_phase,
        show_session_header,
        show_welcome,
    )
    from tools.registry import TOOL_GROUPS, TOOLS

    console = Console()
    session_id = uuid.uuid4().hex[:12]
    render_splash(console, TOOL_GROUPS, model=MODEL, session_id=session_id)
    show_welcome(console)
    show_session_header(console, session_id)
    console.print(f"[dim]Model: {MODEL}  •  {APP_VERSION}  •  /help for commands[/]\n")

    client = None
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    session = create_session()

    while True:
        try:
            user_input = session.prompt().strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!🧃")
            break

        if user_input.lower() in {"/bye", "/exit", "/quit"}:
            print("Goodbye!🧃")
            break
        if not user_input:
            continue

        if user_input == "/help":
            console.print(HELP_TEXT)
            continue

        if user_input == "/tools":
            console.print(f"[dim]tools:[/] {', '.join(sorted(TOOLS))}")
            continue

        if user_input == "/model":
            console.print(f"[dim]model:[/] {MODEL}")
            continue

        if user_input == "/browser" or user_input.startswith("/browser "):
            command = user_input[len("/browser"):].strip()
            print(f"\n{run_browser(command)}\n")
            continue

        if client is None:
            client = create_client()
        messages.append({"role": "user", "content": user_input})
        try:
            answer = run_turn(messages, client, on_phase=lambda phase: show_phase(console, phase))
            print(f"\nAgent: {answer}\n")
        except Exception as error:
            print(f"\nRequest failed: {error}\n")


if __name__ == "__main__":
    main()
