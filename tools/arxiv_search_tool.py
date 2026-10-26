import arxiv
from concurrent.futures import TimeoutError as FuturesTimeoutError

from config import DEFAULT_MAX_RESULTS, MAX_RESULTS_LIMIT, TOOL_TIMEOUT_SECONDS
from tools._timeout import run_with_timeout

ARXIV_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "arxiv_search",
        "description": (
            "Search arXiv for the newest research papers matching a topic or query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Research topic or arXiv search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of papers to return (1-10).",
                    "minimum": 1,
                    "maximum": MAX_RESULTS_LIMIT,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def arxiv_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, str]] | dict[str, str]:
    """Search arXiv for the newest matching papers and return their metadata."""
    query = query.strip() if isinstance(query, str) else ""
    if not query:
        return {"error": "Search query must not be empty."}

    if not isinstance(max_results, int) or isinstance(max_results, bool):
        return {"error": "max_results must be an integer."}
    max_results = max(1, min(max_results, MAX_RESULTS_LIMIT))

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )

    try:
        client = arxiv.Client()
        return run_with_timeout(_fetch_results, TOOL_TIMEOUT_SECONDS, client, search)
    except Exception as error:
        if isinstance(error, FuturesTimeoutError):
            return {"error": f"arXiv search timed out after {TOOL_TIMEOUT_SECONDS}s; refine the query or retry later."}
        return {"error": f"arXiv search failed: {error}"}


def _fetch_results(client: "arxiv.Client", search: "arxiv.Search") -> list[dict[str, str]]:
    """Fetch and format results (runs in a worker thread under the timeout)."""
    return [
        {
            "title": result.title.strip(),
            "authors": ", ".join(author.name for author in result.authors),
            "published": result.published.date().isoformat(),
            "summary": result.summary.strip(),
            "url": result.entry_id,
            "pdf_url": result.pdf_url or "",
        }
        for result in client.results(search)
    ]
