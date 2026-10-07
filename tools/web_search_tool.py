"""DuckDuckGo web search tool (raw execution; formatting lives in presenters)."""

from config import DEFAULT_MAX_RESULTS, MAX_RESULTS_LIMIT

WEB_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the general web via DuckDuckGo. Use for current events, "
            "facts, docs, and anything that is not an arXiv paper lookup. "
            "Returns titles, URLs, and snippets for the model to summarize."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Web search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1-10).",
                    "minimum": 1,
                    "maximum": MAX_RESULTS_LIMIT,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def web_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]] | dict[str, str]:
    """Search the web and return raw result dicts for the presenter."""
    query = query.strip() if isinstance(query, str) else ""
    if not query:
        return {"error": "Search query must not be empty."}

    if not isinstance(max_results, int) or isinstance(max_results, bool):
        return {"error": "max_results must be an integer."}
    max_results = max(1, min(max_results, MAX_RESULTS_LIMIT))

    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = ddgs.text(query, region="wt-wt", max_results=max_results)
    except Exception as error:
        return {"error": f"Web search failed: {error}"}

    if not results:
        return []
    return [
        {
            "title": (item.get("title") or "No title").strip(),
            "url": (item.get("href") or "").strip(),
            "snippet": (item.get("body") or "").strip(),
        }
        for item in results
    ]
