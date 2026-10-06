"""Groq SDK setup and single-request model communication."""

import json
import time

from dotenv import load_dotenv
from groq import Groq

from config import DEBUG_TOKEN_USAGE, MAX_COMPLETION_TOKENS, MODEL

load_dotenv()


def create_client() -> Groq:
    """Create the native Groq SDK client."""
    return Groq()


def call_llm(messages, client, tools, tool_choice="auto"):
    """Send one chat-completion request and return its assistant message.

    Conversation state, tool dispatch, and follow-up calls belong to agent.py.
    """
    if DEBUG_TOKEN_USAGE:
        serialized_messages = json.dumps(messages, ensure_ascii=False, default=str)
        serialized_tools = json.dumps(tools, ensure_ascii=False, default=str)
        print(
            f"[token-debug] request: {len(messages)} messages, "
            f"{len(serialized_messages):,} message chars "
            f"(~{(len(serialized_messages) + 3) // 4:,} tokens), "
            f"{len(serialized_tools):,} tool-schema chars "
            f"(~{(len(serialized_tools) + 3) // 4:,} tokens)"
        )

    started = time.perf_counter()
    try:
        request_kwargs = {
            "model": MODEL,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        if MAX_COMPLETION_TOKENS is not None:
            request_kwargs["max_completion_tokens"] = MAX_COMPLETION_TOKENS
        response = client.chat.completions.create(**request_kwargs)
    except Exception as error:
        if DEBUG_TOKEN_USAGE:
            print(
                f"[token-debug] Groq request failed after "
                f"{time.perf_counter() - started:.2f}s: "
                f"{type(error).__name__}: {error}"
            )
        raise

    elapsed = time.perf_counter() - started
    if DEBUG_TOKEN_USAGE:
        usage = getattr(response, "usage", None)
        if usage is None:
            print(f"[token-debug] response usage unavailable ({elapsed:.2f}s)")
        else:
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

    return response.choices[0].message
