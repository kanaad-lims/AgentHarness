from tools.bash_tool import _format_output, _normalize_windows_paths


def test_normalizes_unquoted_windows_path_to_wsl():
    command = r"cat D:\Kanaad\temp\automation_arxiv.py"

    assert _normalize_windows_paths(command) == (
        "cat /mnt/d/Kanaad/temp/automation_arxiv.py"
    )


def test_normalizes_quoted_windows_path_with_spaces():
    command = 'cat "D:/Kanaad/temp/my file.py"'

    assert _normalize_windows_paths(command) == 'cat "/mnt/d/Kanaad/temp/my file.py"'


def test_does_not_convert_urls():
    command = "curl https://example.com/search?q=test"

    assert _normalize_windows_paths(command) == command


def test_format_output_preserves_tail_and_counts_omitted(monkeypatch):
    import tools.bash_tool as bash_tool

    monkeypatch.setattr(bash_tool, "HEAD_CHARS", 10)
    monkeypatch.setattr(bash_tool, "TAIL_CHARS", 10)
    monkeypatch.setattr(bash_tool, "STDERR_TAIL_CHARS", 10)

    formatted = _format_output("x" * 100, "e" * 50)

    assert formatted["truncated"] is True
    assert formatted["omitted_chars"] == (100 - 20) + (50 - 10)
    assert formatted["stdout_head"] == "x" * 10
    assert formatted["stdout_tail"] == "x" * 10
    assert formatted["stderr_tail"] == "e" * 10
