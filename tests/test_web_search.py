from presenters import web as web_presenter
from tools.registry import TOOL_SCHEMAS, TOOLS


def test_web_search_registered_with_schema():
    assert "web_search" in TOOLS
    names = [schema["function"]["name"] for schema in TOOL_SCHEMAS]
    assert "web_search" in names


def test_format_web_results_shortens_snippets_and_keeps_links():
    results = [
        {"title": "Title One", "url": "https://example.com/1", "snippet": "word " * 500},
        {"title": "Title Two", "url": "https://example.com/2", "snippet": "short snippet"},
    ]

    text = web_presenter.format_web_results(results)

    assert "Title One" in text
    assert "https://example.com/1" in text
    assert "[truncated]" in text
    assert len(text) <= web_presenter.WEB_FORMAT_MAX_CHARS + 100


def test_format_web_single_result_returns_full_snippet():
    results = [{"title": "T", "url": "https://example.com", "snippet": "word " * 500}]

    text = web_presenter.format_web_results(results)

    assert "full snippet" in text
    assert "[truncated]" not in text


def test_format_web_error_passthrough():
    assert web_presenter.format_web_results({"error": "boom"}).startswith("Web error:")
    assert web_presenter.format_web_results([]).startswith("No web results")
