from tools.bash_tool import _normalize_windows_paths


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
