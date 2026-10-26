"""Conversation compaction: token estimation, turn detection, summarization."""
from __future__ import annotations

import json
from typing import Any

from llm import create_client, call_llm
from config import MODEL

# ─── Configuration ──────────────────────────────────────────────────────
TOKEN_BUDGET = 8000          # Primary trigger: estimated tokens in messages
MESSAGE_BUDGET = 20          # Fallback trigger: message count
MIN_RECENT_TURNS = 2         # Always keep at least this many complete turns


# ─── Token estimation ───────────────────────────────────────────────────
def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return (len(text) + 3) // 4


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate total tokens for a message list."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            # Handle structured content (e.g., tool results)
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += estimate_tokens(part.get("text", ""))
    return total


# ─── Turn detection ─────────────────────────────────────────────────────
def is_tool_message(msg: dict) -> bool:
    return msg.get("role") == "tool"


def is_assistant_with_tool_calls(msg: dict) -> bool:
    """Check if assistant message has tool_calls."""
    return msg.get("role") == "assistant" and bool(msg.get("tool_calls"))


def split_into_turns(messages: list[dict]) -> list[list[dict]]:
    """Split messages into complete conversation turns.

    A turn = user message + assistant response (+ any tool call/result pairs).
    """
    turns: list[list[dict]] = []
    current: list[dict] = []

    for msg in messages:
        current.append(msg)

        # A turn ends when we see an assistant message WITHOUT tool_calls
        # (i.e., a final answer), or when we see a user message after an assistant
        if msg.get("role") == "assistant" and not msg.get("tool_calls"):
            turns.append(current)
            current = []
        elif msg.get("role") == "user" and current and current[0].get("role") != "user":
            # New user message starts a new turn
            turns.append(current[:-1])
            current = [msg]

    if current:
        turns.append(current)

    return turns


def count_complete_turns(messages: list[dict]) -> int:
    """Count complete turns (user + assistant final) in messages."""
    turns = split_into_turns(messages)
    return len([t for t in turns if t and t[-1].get("role") == "assistant" and not t[-1].get("tool_calls")])


# ─── Compaction decision ────────────────────────────────────────────────
def should_compact(messages: list[dict]) -> bool:
    """Check if compaction should trigger."""
    if len(messages) > MESSAGE_BUDGET:
        return True
    if estimate_messages_tokens(messages) > TOKEN_BUDGET:
        return True
    return False


# ─── Summarization ──────────────────────────────────────────────────────
SUMMARIZATION_PROMPT = """Summarize the following conversation turns into a concise summary.
Preserve key facts, decisions, and context that would be useful for continuing the conversation.
Do not include verbose details, tool outputs, or internal reasoning.
Focus on: what was asked, what was found/decided, and any outstanding items.

Turns to summarize:
{turns_json}

Summary:"""


def build_turns_json(turns: list[list[dict]]) -> str:
    """Convert turns to a compact JSON string for the summarization prompt."""
    simplified = []
    for turn in turns:
        turn_summary = []
        for msg in turn:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                turn_summary.append(f"User: {content[:500]}")
            elif role == "assistant":
                if msg.get("tool_calls"):
                    tool_names = [tc.get("function", {}).get("name", "") for tc in msg["tool_calls"]]
                    turn_summary.append(f"Assistant: [tool calls: {', '.join(tool_names)}]")
                else:
                    turn_summary.append(f"Assistant: {content[:500]}")
            elif role == "tool":
                name = msg.get("name", "")
                turn_summary.append(f"Tool({name}): {content[:200]}")
        simplified.append(" | ".join(turn_summary))
    return "\n".join(simplified)


async def summarize_turns(turns: list[list[dict]]) -> str:
    """Call the LLM to summarize a list of turns."""
    if not turns:
        return ""

    turns_json = build_turns_json(turns)
    prompt = SUMMARIZATION_PROMPT.format(turns_json=turns_json)

    client = create_client()
    messages = [{"role": "user", "content": prompt}]
    response = await call_llm(messages, client, tools=None, tool_choice="none")
    return (response.content or "").strip()


async def compact_state_messages(messages: list[dict], conversation_summary: str = "") -> tuple[list[dict], str]:
    """Compact messages by summarizing old turns, keeping recent turns intact.

    Returns:
        (new_messages, updated_conversation_summary)

    Preserves system prompt at index 0 for prefix caching.
    """
    # Preserve system prompt (index 0) for prefix caching
    system_prompt = messages[0] if messages and messages[0].get("role") == "system" else None
    other_messages = messages[1:] if system_prompt else messages

    turns = split_into_turns(other_messages)
    if len(turns) <= MIN_RECENT_TURNS:
        return messages, conversation_summary

    # Split into old turns to summarize + recent turns to keep
    recent_turns = turns[-MIN_RECENT_TURNS:]
    old_turns = turns[:-MIN_RECENT_TURNS]

    # Summarize old turns
    new_summary_part = ""
    if old_turns:
        new_summary_part = await summarize_turns(old_turns)

    # Combine with existing summary
    if conversation_summary and new_summary_part:
        updated_summary = f"{conversation_summary}\n\n---\n\n{new_summary_part}"
    else:
        updated_summary = conversation_summary or new_summary_part

    # Rebuild with system prompt FIRST (prefix cache), then summary, then recent turns
    new_messages = []
    if system_prompt:
        new_messages.append(system_prompt)
    if updated_summary:
        new_messages.append({
            "role": "system",
            "content": f"Conversation summary (previous turns):\n{updated_summary}"
        })

    for turn in recent_turns:
        new_messages.extend(turn)

    return new_messages, updated_summary


# ─── Sync wrapper for use in non-async contexts ─────────────────────────
def compact_state_messages_sync(messages: list[dict], conversation_summary: str = "") -> tuple[list[dict], str]:
    """Synchronous version using a simple heuristic summary (no LLM call).

    Use this when you can't await (e.g., inside the graph node).
    For production, replace with the async version called separately.

    Preserves system prompt at index 0 for prefix caching.
    """
    # Preserve system prompt (index 0) for prefix caching
    system_prompt = messages[0] if messages and messages[0].get("role") == "system" else None
    other_messages = messages[1:] if system_prompt else messages

    turns = split_into_turns(other_messages)
    if len(turns) <= MIN_RECENT_TURNS:
        return messages, conversation_summary

    recent_turns = turns[-MIN_RECENT_TURNS:]
    old_turns = turns[:-MIN_RECENT_TURNS]

    # Heuristic summary without LLM
    summary_parts = []
    if conversation_summary:
        summary_parts.append(conversation_summary)

    if old_turns:
        topics = []
        for turn in old_turns:
            for msg in turn:
                if msg.get("role") == "user":
                    topics.append(msg.get("content", "")[:100])
                    break
        if topics:
            summary_parts.append(f"Previous topics: {'; '.join(topics)}")

    updated_summary = "\n\n---\n\n".join(summary_parts) if summary_parts else ""

    # Rebuild with system prompt FIRST (prefix cache), then summary, then recent turns
    new_messages = []
    if system_prompt:
        new_messages.append(system_prompt)
    if updated_summary:
        new_messages.append({
            "role": "system",
            "content": f"Conversation summary (previous turns):\n{updated_summary}"
        })

    for turn in recent_turns:
        new_messages.extend(turn)

    return new_messages, updated_summary