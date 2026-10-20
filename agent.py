"""LangGraph agent loop with persistence, checkpointer, and compaction."""

import json
import operator
import time
import uuid
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from llm import DEBUG_TOKEN_USAGE, MODEL, call_llm, create_client
from observability import (
    TurnRecorder,
    current_turn,
    model_usage,
    tool_verdict,
    turn_metadata,
    verdict_for_blocked,
)
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

from policies.checks import (
    approx_tokens,
    call_key,
    check_tool_call,
    is_tool_failure,
    parse_tool_arguments,
    tool_call_fields,
)

from persistence import (
    get_saver,
    compile_with_checkpointer,
    make_config,
    should_compact,
    compact_state_messages_sync,
    add_message,
    create_session as create_db_session,
    get_session,
)


class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    model_calls: int
    tool_calls: int
    arxiv_calls: int
    web_calls: int
    skill_calls: int
    seen_keys: Annotated[list[str], operator.add]
    attempts: dict
    pending: list
    conversation_summary: str  # Persisted via checkpointer


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
    """Build the LangGraph workflow with checkpointing."""
    emit = on_phase or (lambda _phase: None)

    def call_model(state: AgentState) -> dict:
        # Compaction check before model call
        messages = state["messages"]
        summary = state.get("conversation_summary", "")
        if should_compact(messages):
            messages, summary = compact_state_messages_sync(messages, summary)

        current_plan = get_todos()
        emit("thinking")
        request_messages = list(messages)
        if current_plan:
            emit("planning")
            request_messages.append(plan_message(current_plan))

        started = time.perf_counter()
        response = call_llm(request_messages, client, TOOL_SCHEMAS)
        model_latency = time.perf_counter() - started
        requested_tools = list(getattr(response, "tool_calls", None) or [])
        turn = current_turn.get()
        if turn is not None:
            recorder, turn_id = turn
            recorder.log_model_call(
                turn_id,
                model_usage(response),
                model_latency,
                len(request_messages),
            )
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
            "conversation_summary": summary,
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
        skill_used = state.get("skill_calls", 0)
        successful = set(state.get("seen_keys", []))
        attempts = dict(state.get("attempts", {}) or {})

        for tool_call in pending:
            call_id, name, args_text = tool_call_fields(tool_call)
            arguments = parse_tool_arguments(args_text)
            key = call_key(name, arguments)
            attempts_made = attempts.get(key, 0)
            tool_started = time.perf_counter()
            blocked, message = check_tool_call(
                name,
                arguments,
                arxiv_calls_used=arxiv_used,
                web_calls_used=web_used,
                skill_reads_used=skill_used,
                attempts_made=attempts_made,
                already_succeeded=key in successful,
            )
            if blocked:
                result = message
                verdict = verdict_for_blocked(message)
            else:
                tool_calls_used += 1
                attempts[key] = attempts_made + 1
                if name == "arxiv_search":
                    arxiv_used += 1
                if name == "web_search":
                    web_used += 1
                if name == "read_skill":
                    skill_used += 1
                dispatched_name, result, _ = _execute_tool_call(tool_call)
                name = dispatched_name
                failed = is_tool_failure(name, result)
                verdict = tool_verdict(name, result, failed)
                if not failed:
                    successful.add(key)
                    new_seen.append(key)
            turn = current_turn.get()
            if turn is not None:
                recorder, turn_id = turn
                recorder.log_tool_call(
                    turn_id,
                    name,
                    arguments,
                    verdict,
                    len(result),
                    time.perf_counter() - tool_started,
                )

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
            "skill_calls": skill_used,
            "seen_keys": new_seen,
            "attempts": attempts,
            "pending": [],
            "conversation_summary": state.get("conversation_summary", ""),
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
            "conversation_summary": _state.get("conversation_summary", ""),
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

    # Compile with checkpointer
    return compile_with_checkpointer(graph)


def run_turn(messages: list[dict], client, *, session_id: str, on_phase=None) -> str:
    """Run one user turn through the LangGraph workflow with checkpointing."""
    emit = on_phase or (lambda _phase: None)

    # Ensure session exists in archive
    if not get_session(session_id):
        create_db_session(session_id)

    app = build_graph(client, on_phase=emit)
    recorder = TurnRecorder()

    user_text = next(
        (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    turn_id = recorder.start_turn(user_text, MODEL)
    token = current_turn.set((recorder, turn_id))

    try:
        config = make_config(session_id, recursion_limit=RECURSION_LIMIT)
        result = app.invoke(
            {
                "messages": list(messages),
                "model_calls": 0,
                "tool_calls": 0,
                "arxiv_calls": 0,
                "web_calls": 0,
                "skill_calls": 0,
                "seen_keys": [],
                "attempts": {},
                "pending": [],
                "conversation_summary": "",
            },
            config=config,
        )
    except Exception as error:
        recorder.log_error(f"{type(error).__name__}: {error}")
        recorder.end_turn(turn_id, "", turn_metadata({}, MODEL))
        raise
    finally:
        current_turn.reset(token)

    messages[:] = result["messages"]
    emit("accepted")

    # Extract final assistant answer
    answer = ""
    for message in reversed(result["messages"]):
        if isinstance(message, dict) and message.get("role") == "assistant":
            answer = message.get("content", "") or ""
            break

    # Archive user message and final answer
    if user_text:
        add_message(session_id, "user", user_text)
    if answer:
        add_message(session_id, "assistant", answer)

    recorder.end_turn(turn_id, answer, turn_metadata(result, MODEL))
    return answer


def main() -> None:
    """Start the orchestrator agent with session management."""
    from rich.console import Console

    from cli.prompt import create_session
    from cli.ui import (
        APP_VERSION,
        HELP_TEXT,
        PLACEHOLDER_SKILLS,
        render_splash,
        show_answer,
        show_phase,
        show_session_header,
        show_welcome,
    )
    from tools.registry import TOOL_GROUPS, TOOLS

    console = Console()

    # Generate or load session_id
    session_id = uuid.uuid4().hex[:12]

    try:
        from skills.loader import load_skills, skill_groups

        splash_skills = skill_groups(load_skills())
    except Exception:
        splash_skills = PLACEHOLDER_SKILLS

    render_splash(console, TOOL_GROUPS, splash_skills, model=MODEL, session_id=session_id)
    show_welcome(console)
    show_session_header(console, session_id)
    console.print(f"[dim]Model: {MODEL}  •  {APP_VERSION}  •  /help for commands[/]\n")

    client = None
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    prompt_session = create_session()

    while True:
        try:
            user_input = prompt_session.prompt().strip()
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
            command = user_input[len("/browser") :].strip()
            print(f"\n{run_browser(command)}\n")
            continue

        # Session commands
        if user_input == "/session new":
            session_id = uuid.uuid4().hex[:12]
            create_db_session(session_id)  # Persist immediately
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            console.print(f"[green]New session:[/] {session_id}")
            show_session_header(console, session_id)
            continue

        if user_input == "/session list":
            from persistence import list_sessions
            sessions = list_sessions()
            if not sessions:
                console.print("[dim]No sessions yet.[/]")
            else:
                console.print("[bold]Sessions:[/]")
                for s in sessions:
                    marker = " ← current" if s["session_id"] == session_id else ""
                    console.print(f"  {s['session_id']}  {s['title']}  ({s['updated_at']}){marker}")
            continue

        if user_input.startswith("/session switch "):
            target_id = user_input[len("/session switch "):].strip()
            if not target_id:
                console.print("[red]Usage: /session switch <session_id>[/]")
                continue
            if not get_session(target_id):
                console.print(f"[red]Session not found:[/] {target_id}")
                continue
            session_id = target_id
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            console.print(f"[green]Switched to session:[/] {session_id}")
            show_session_header(console, session_id)
            continue

        if client is None:
            client = create_client()

        messages.append({"role": "user", "content": user_input})
        try:
            answer = run_turn(messages, client, session_id=session_id, on_phase=lambda phase: show_phase(console, phase))
            show_answer(console, answer)
        except Exception as error:
            print(f"\nRequest failed: {error}\n")


if __name__ == "__main__":
    main()