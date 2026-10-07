"""Control and interaction strings returned to the model or shown to the user."""

import json

TOOL_LIMIT_MESSAGE = "Tool-call limit reached for this user request."
DUPLICATE_TOOL_MESSAGE = (
    "Duplicate tool call skipped; use the earlier result in this conversation."
)
RETRY_LIMIT_MESSAGE = (
    "Retry limit reached for this exact tool call. "
    "Change the arguments or use results already in the conversation."
)
ARXIV_LIMIT_MESSAGE = (
    "ArXiv search limit reached for this request. Use the arXiv results "
    "already in the conversation to answer."
)
WEB_LIMIT_MESSAGE = (
    "Web search limit reached for this request. Use the web results "
    "already in the conversation to answer."
)
MODEL_LIMIT_ANSWER = (
    "I reached the model-call limit for this request. "
    "Please ask me to continue if more work is needed."
)

BASH_APPROVAL_PROMPT = "Run this command? [y/N]: "
BASH_REQUEST_HEADER = "Bash command requested:"


def plan_message(todos: list[dict]) -> dict:
    """Build the authoritative plan block appended to model requests."""
    return {
        "role": "system",
        "content": "Current plan (authoritative): <todos>"
        f"{json.dumps(todos, ensure_ascii=False)}</todos>",
    }
