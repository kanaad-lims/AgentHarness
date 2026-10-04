"""Bash tool implementation and function schema for model tool calling.

This executes commands on the host machine; it is not a sandbox. Only invoke
it through the agent's tool-call and permission flow.
"""

import re
import subprocess

DEFAULT_TIMEOUT_SECONDS = 30
MAX_OUTPUT_CHARACTERS = 20_000

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


def _format_output(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    output = _as_text(stdout)
    error_output = _as_text(stderr)
    if error_output:
        output += ("\n" if output else "") + error_output

    if len(output) > MAX_OUTPUT_CHARACTERS:
        output = output[:MAX_OUTPUT_CHARACTERS] + "\n... output truncated"
    return output or "(no output)"


def run_bash(
    command: str,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Run ``command`` with Bash and return its exit code and output.

    Bash must be installed and available on PATH. This function does not
    sandbox commands, so callers should apply their permission policy before
    dispatching a model-requested tool call.
    """
    if not isinstance(command, str) or not command.strip():
        return "Error: command must be a non-empty string."
    if timeout <= 0:
        return "Error: timeout must be a positive number of seconds."

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
        return "Error: Bash was not found on PATH. Install Bash or configure its path."
    except subprocess.TimeoutExpired as error:
        output = _format_output(error.stdout, error.stderr)
        return f"Timed out after {timeout} seconds.\n{output}"
    except OSError as error:
        return f"Error launching Bash: {error}"

    output = _format_output(result.stdout, result.stderr)
    return f"Exit code: {result.returncode}\n{output}"
