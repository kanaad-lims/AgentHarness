"""Compact arXiv formatting for model consumption"""

import json
import os

ARXIV_ABSTRACT_CHARS = int(os.getenv("ARXIV_ABSTRACT_CHARS", "300"))
ARXIV_FORMAT_MAX_CHARS = int(os.getenv("ARXIV_FORMAT_MAX_CHARS", "1500"))


def shorten_at_word(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ...[truncated]"


def format_arxiv_results(results) -> str:
    """Format raw arXiv results as compact valid text for the model."""
    if isinstance(results, dict):
        if "error" in results:
            return f"ArXiv error: {results['error']}"
        return f"ArXiv result: {json.dumps(results, ensure_ascii=False)[:500]}"
    if not results:
        return "No papers found for this query."
    lines = [f"ArXiv results ({len(results)} papers, abstracts shortened):"]
    for index, paper in enumerate(results, start=1):
        title = (paper.get("title", "") or "(no title)").strip()
        authors = (paper.get("authors", "") or "").strip()
        parts = [part.strip() for part in authors.split(",") if part.strip()]
        if len(parts) > 3:
            authors = ", ".join(parts[:3]) + " et al."
        else:
            authors = ", ".join(parts) or "unknown"
        published = paper.get("published", "") or ""
        url = paper.get("url", "") or ""
        abstract = shorten_at_word(paper.get("summary", "") or "", ARXIV_ABSTRACT_CHARS)
        lines.append(
            f"{index}. {title} | Authors: {authors} | Published: {published} | "
            f"URL: {url} | Abstract: {abstract}"
        )
    text = "\n".join(lines)
    if len(text) > ARXIV_FORMAT_MAX_CHARS:
        text = text[:ARXIV_FORMAT_MAX_CHARS].rsplit(" ", 1)[0] + "\n...[arXiv results truncated]"
    return text
