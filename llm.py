import json
import os
import time

from dotenv import load_dotenv
from groq import Groq

from tools.bash_tool import BASH_TOOL_SCHEMA, run_bash
from tools.arxiv_search_tool import ARXIV_SEARCH_TOOL_SCHEMA, arxiv_search
from tools.browser_tool import run_browser
from tools.todo_tool import TODO_TOOL_SCHEMA, get_todos, write_todos

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
TOOL_SCHEMAS = [BASH_TOOL_SCHEMA, ARXIV_SEARCH_TOOL_SCHEMA, TODO_TOOL_SCHEMA]
TOOLS = {
    "bash": run_bash,
    "arxiv_search": arxiv_search,
    "write_todos": write_todos,
}

# Temporary diagnostics: set to False (or comment out the debug calls below)
# once token usage and tool latency have been investigated.
DEBUG_TOKEN_USAGE = True
_MODEL_CALL_COUNT = 0

SYSTEM_PROMPT = """You are a helpful and careful coding assistant.

Help the user understand, write, debug, and improve software. Give accurate,
practical answers and explain important trade-offs briefly. When writing code,
provide complete, runnable examples when appropriate, use clear names, and
follow the language and conventions of the user's project. Do not invent files,
APIs, test results, or actions you have not performed. Ask a concise clarifying
question when essential requirements are missing; otherwise state reasonable
assumptions and proceed. Point out security, data-loss, or compatibility risks
before recommending risky changes. Keep responses focused on the user's request.
Use the bash tool for inspecting the files.
Use arxiv_search when the user asks for recent arXiv papers or research on a topic.
For tasks that take multiple steps, call write_todos first with the full plan.
On every update, send the complete replacement list, keep exactly one unfinished
task in_progress, and mark work completed as soon as it is done. Skip planning
for single-step tasks. The current plan is provided in <todos> tags.
RULE: NEVER execute the bash tool if you are asked to delete or modify any file inside any directory.
RULE: ALWAYS limit your output to <700 tokens plus or minus 100. So adjust the answers accordingly. Do not give incomplete answers just because you ran out of output tokens.
"""


def _approx_tokens(text: str) -> int:
    """Rough estimate only; provider tokenization is model-specific."""
    return (len(text) + 3) // 4


def _log_request_size(messages) -> None:
    if not DEBUG_TOKEN_USAGE:
        return

    serialized_messages = json.dumps(messages, ensure_ascii=False, default=str)
    serialized_tools = json.dumps(TOOL_SCHEMAS, ensure_ascii=False, default=str)
    print(
        f"[token-debug] request input: {len(messages)} messages, "
        f"{len(serialized_messages):,} message chars (~{_approx_tokens(serialized_messages):,} tokens), "
        f"{len(serialized_tools):,} tool-schema chars (~{_approx_tokens(serialized_tools):,} tokens)"
    )

    role_sizes = {}
    for item in messages:
        role = item.get("role", "unknown")
        encoded = json.dumps(item, ensure_ascii=False, default=str)
        role_sizes[role] = role_sizes.get(role, 0) + len(encoded)
    for role, char_count in role_sizes.items():
        print(
            f"[token-debug]   {role}: {char_count:,} chars "
            f"(~{(char_count + 3) // 4:,} tokens)"
        )


def _log_response_usage(response, elapsed: float) -> None:
    if not DEBUG_TOKEN_USAGE:
        return

    usage = getattr(response, "usage", None)
    if usage is None:
        print(f"[token-debug] response usage unavailable ({elapsed:.2f}s)")
        return

    completion_details = getattr(usage, "completion_tokens_details", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    print(
        "[token-debug] actual usage: "
        f"prompt={getattr(usage, 'prompt_tokens', 'n/a')}, "
        f"completion={getattr(usage, 'completion_tokens', 'n/a')}, "
        f"reasoning={getattr(completion_details, 'reasoning_tokens', 'n/a')}, "
        f"cached_input={getattr(prompt_details, 'cached_tokens', 'n/a')}; "
        f"latency={elapsed:.2f}s"
    )


def call_llm(messages, client, tool_choice="auto"):
    global _MODEL_CALL_COUNT
    _MODEL_CALL_COUNT += 1
    if DEBUG_TOKEN_USAGE:
        print(f"[token-debug] model call #{_MODEL_CALL_COUNT} ({MODEL})")

    request_messages = list(messages)
    if request_messages and request_messages[0].get("role") == "system":
        request_messages.insert(
            1,
            {
                "role": "system",
                "content": (
                    "Current plan (authoritative): <todos>"
                    f"{json.dumps(get_todos(), ensure_ascii=False)}"
                    "</todos>"
                ),
            },
        )
    _log_request_size(request_messages)

    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=request_messages,
            tools=TOOL_SCHEMAS,
            tool_choice=tool_choice,
            max_completion_tokens=2048,
        )
    except Exception as error:
        elapsed = time.perf_counter() - started
        if DEBUG_TOKEN_USAGE:
            print(
                f"[token-debug] model request failed after {elapsed:.2f}s: "
                f"{type(error).__name__}: {error}"
            )
        raise

    _log_response_usage(response, time.perf_counter() - started)
    return response.choices[0].message


def main():
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
            browser_command = user_input[len("/browser"):].strip()
            print(f"\n{run_browser(browser_command)}\n")
            continue

        if client is None:
            client = Groq()

        messages.append({"role": "user", "content": user_input})

        try:
            for _ in range(5):  # Prevent unbounded tool-call loops.
                message = call_llm(messages, client)
                tool_calls = message.tool_calls or []

                if not tool_calls:
                    answer = message.content or ""
                    print(f"\nAgent: {answer}\n")
                    messages.append({"role": "assistant", "content": answer})
                    break

                # Preserve the assistant's tool-call request in the history.
                messages.append(message.model_dump(exclude_none=True))

                for tool_call in tool_calls:
                    name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)

                    if DEBUG_TOKEN_USAGE:
                        args_text = json.dumps(args, ensure_ascii=False, default=str)
                        print(
                            f"[tool-debug] selected {name}; arguments: "
                            f"{len(args_text):,} chars (~{_approx_tokens(args_text):,} tokens)"
                        )

                    if name not in TOOLS:
                        result = f"Error: unknown tool '{name}'."
                    elif name == "bash":
                        command = args.get("command", "")
                        print(f"\nBash command requested:\n{command}")
                        approved = input("Run this command? [y/N]: ").strip().lower()

                        if approved in {"y", "yes"}:
                            tool_started = time.perf_counter()
                            result = TOOLS[name](**args)
                            if DEBUG_TOKEN_USAGE:
                                print(
                                    f"[tool-debug] {name} completed in "
                                    f"{time.perf_counter() - tool_started:.2f}s"
                                )
                            # removed printing raw bash output. Directly passed to the llm as context.
                        else:
                            result = "Command was not run; the user did not approve it."

                    else:
                        tool_started = time.perf_counter()
                        result = TOOLS[name](**args)
                        if DEBUG_TOKEN_USAGE:
                            print(
                                f"[tool-debug] {name} completed in "
                                f"{time.perf_counter() - tool_started:.2f}s"
                            )

                    if not isinstance(result, str):
                        result = json.dumps(result, ensure_ascii=False)

                    if DEBUG_TOKEN_USAGE:
                        print(
                            f"[tool-debug] {name} result: {len(result):,} chars "
                            f"(~{_approx_tokens(result):,} tokens); added to conversation"
                        )

                    # Return tool output to the model for its follow-up answer.
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": name,
                            "content": result,
                        }
                    )
            else:
                print("Agent: Stopped after reaching the tool-call limit.\n")

        except Exception as error:
            print(f"\nRequest failed: {error}\n")


if __name__ == "__main__":
    main()
