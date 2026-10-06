"""Bash tool implementation and function schema for model tool calling.

This executes commands on the host machine; it is not a sandbox. Only invoke
it through the agent's tool-call and permission flow.
"""

import os
import re
import subprocess

from config import DEFAULT_TIMEOUT_SECONDS
from policies import BASH_HEAD_CHARS as HEAD_CHARS
from policies import BASH_STDERR_TAIL_CHARS as STDERR_TAIL_CHARS
from policies import BASH_TAIL_CHARS as TAIL_CHARS

_QUOTED_WINDOWS_PATH = re.compile(
    r"""(?P<quote>[\"'])(?P<path>[A-Za-z]:[\\/](?![\\/])[^\"']+)(?P=quote)"""
)
_UNQUOTED_WINDOWS_PATH = re.compile(
    r"""(?<![\w/:])(?P<drive>[A-Za-z]):[\\/](?![\\/])(?P<rest>[^\s\"'`;|&<>]+)"""
)


def _windows_path_to_wsl(path: str) -> str:
    """Convert a Windows drive path to WSL's /mnt/<drive>/ format."""
    drive = path[0].lower()
    remainder = path[3:].replace("\\", "/").lstrip("/")
    return f"/mnt/{drive}/{remainder}"


def _normalize_windows_paths(command: str) -> str:
    """Translate Windows drive paths in a Bash command to WSL paths."""
    command = _QUOTED_WINDOWS_PATH.sub(
        lambda match: (
            match.group("quote")
            + _windows_path_to_wsl(match.group("path"))
            + match.group("quote")
        ),
        command,
    )

    return _UNQUOTED_WINDOWS_PATH.sub(
        lambda match: (
            f"/mnt/{match.group('drive').lower()}/"
            f"{match.group('rest').replace(chr(92), '/')}"
        ),
        command,
    )


BASH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": (
            "Run a shell command using Bash in the current project directory. "
            "Commands execute on the host machine and are not sandboxed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The Bash command to execute.",
                }
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _split_head_tail(text: str, head: int, tail: int) -> tuple[str, str, int]:
    """Split text into head/tail, returning (head, tail, omitted_chars)."""
    if len(text) <= head + tail:
        return text, "", 0
    return text[:head], text[-tail:], len(text) - head - tail


def _format_output(stdout: str | bytes | None, stderr: str | bytes | None) -> dict:
    """Return structured head/tail output so tails are never silently dropped."""
    stdout_text = _as_text(stdout)
    stderr_text = _as_text(stderr)
    stdout_head, stdout_tail, omitted_stdout = _split_head_tail(
        stdout_text, HEAD_CHARS, TAIL_CHARS
    )
    if len(stderr_text) <= STDERR_TAIL_CHARS:
        stderr_tail, omitted_stderr = stderr_text, 0
    else:
        stderr_tail, omitted_stderr = stderr_text[-STDERR_TAIL_CHARS:], (
            len(stderr_text) - STDERR_TAIL_CHARS
        )
    omitted = omitted_stdout + omitted_stderr
    return {
        "stdout_head": stdout_head,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "omitted_chars": omitted,
        "truncated": omitted > 0,
    }


def run_bash(
    command: str,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    """Run ``command`` with Bash and return structured output.

    This function does not sandbox commands, so callers should apply their
    permission policy before dispatching a model-requested tool call.
    """
    if not isinstance(command, str) or not command.strip():
        return {"error": "Error: command must be a non-empty string."}
    if timeout <= 0:
        return {"error": "Error: timeout must be a positive number of seconds."}

    command = _normalize_windows_paths(command)

    try:
        result = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return {"error": "Error: Bash was not found on PATH. Install Bash or configure its path."}
    except subprocess.TimeoutExpired as error:
        output = _format_output(error.stdout, error.stderr)
        return {
            "command": command,
            "exit_code": None,
            "timed_out": True,
            "timeout_seconds": timeout,
            "cwd": os.getcwd(),
            **output,
        }
    except OSError as error:
        return {"error": f"Error launching Bash: {error}"}

    output = _format_output(result.stdout, result.stderr)
    return {
        "command": command,
        "exit_code": result.returncode,
        "timed_out": False,
        "cwd": os.getcwd(),
        **output,
    }
