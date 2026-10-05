"""Model-facing presenters for tool outputs."""

from presenters.arxiv import format_arxiv_results
from presenters.bash import format_bash_result

__all__ = ["format_arxiv_results", "format_bash_result"]
