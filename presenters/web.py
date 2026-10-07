"""Compact web-search formatting for model consumption."""

import json

from policies import WEB_FORMAT_MAX_CHARS, WEB_SNIPPET_CHARS


def shorten_at_word(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ...[truncated]"


def format_web_results(results) -> str:
    """Format raw web results as compact valid text for the model."""
    if isinstance(results, dict):
        if "error" in results:
            return f"Web error: {results['error']}"
        return f"Web result: {json.dumps(results, ensure_ascii=False)[:500]}"
    if not results:
        return "No web results found for this query."
    if len(results) == 1:
        item = results[0]
        snippet = " ".join((item.get("snippet", "") or "").split())
        return (
            "Web result (1 result, full snippet):\n"
            f"Title: {(item.get('title', '') or '(no title)').strip()}\n"
            f"URL: {(item.get('url', '') or '').strip()}\n"
            f"Snippet: {snippet or '(no snippet)'}"
        )
    lines = [f"Web results ({len(results)} results, snippets shortened):"]
    for index, item in enumerate(results, start=1):
        title = (item.get("title", "") or "(no title)").strip()
        url = (item.get("url", "") or "").strip()
        snippet = shorten_at_word(item.get("snippet", "") or "", WEB_SNIPPET_CHARS)
        lines.append(f"{index}. {title} | URL: {url} | Snippet: {snippet}")
    text = "\n".join(lines)
    if len(text) > WEB_FORMAT_MAX_CHARS:
        text = text[:WEB_FORMAT_MAX_CHARS].rsplit(" ", 1)[0] + "\n...[web results truncated]"
    return text
