"""Compact bash formatting for model consumption."""

from policies import BASH_RENDER_MAX_CHARS


def format_bash_result(result: dict) -> str:
    """Render structured bash output with head + tail inside a small budget."""
    if not isinstance(result, dict):
        return str(result)
    if result.get("error"):
        return str(result["error"])
    command = result.get("command", "")
    cwd = result.get("cwd", "")
    exit_code = result.get("exit_code")
    header = f"$ {command} (cwd: {cwd}) -> exit {exit_code}"
    if result.get("timed_out"):
        header += f" [timed out after {result.get('timeout_seconds', '?')}s]"
    stdout_head = result.get("stdout_head", "")
    stdout_tail = result.get("stdout_tail", "")
    stderr_tail = result.get("stderr_tail", "")
    omitted = result.get("omitted_chars", 0)
    parts = [header]
    if stdout_head:
        parts.append(f"stdout head:\n{stdout_head}")
    if stdout_tail:
        parts.append(f"stdout tail:\n{stdout_tail}")
    if stderr_tail:
        parts.append(f"stderr tail:\n{stderr_tail}")
    if not stdout_head and not stdout_tail and not stderr_tail:
        parts.append("(no output)")
    if omitted:
        parts.append(f"...[omitted {omitted:,} chars]...")
    text = "\n".join(parts)
    if len(text) > BASH_RENDER_MAX_CHARS:
        keep_tail = BASH_RENDER_MAX_CHARS // 2
        keep_head = BASH_RENDER_MAX_CHARS - keep_tail
        text = (
            text[:keep_head]
            + f"\n...[omitted {len(text) - BASH_RENDER_MAX_CHARS:,} chars]...\n"
            + text[-keep_tail:]
        )
    return text
