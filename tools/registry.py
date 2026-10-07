"""Tool registry: names, schemas, and model-ready callables.

Each tool module stays pure execution; presenters stay pure formatting.
This module composes them into the callables the agent dispatches, so
agent.py never hand-maintains tool lists or inline wrappers.
"""

from config import MAX_ARXIV_RESULTS_PER_CALL, WEB_MAX_RESULTS_PER_CALL
from presenters.arxiv import format_arxiv_results
from presenters.bash import format_bash_result
from presenters.web import format_web_results
from tools.arxiv_search_tool import ARXIV_SEARCH_TOOL_SCHEMA, arxiv_search
from tools.bash_tool import BASH_TOOL_SCHEMA, run_bash
from tools.todo_tool import TODO_TOOL_SCHEMA, write_todos
from tools.web_search_tool import WEB_SEARCH_TOOL_SCHEMA, web_search

# Callable function for the search_arxiv function.
# Contains the actual function call and output formatter.
def search_arxiv(**arguments):
    """Clamp result count, call arXiv, return compact model-ready text."""
    requested = arguments.get("max_results", 5)
    if not isinstance(requested, int) or isinstance(requested, bool):
        requested = 5
    arguments["max_results"] = min(max(requested, 1), MAX_ARXIV_RESULTS_PER_CALL)
    return format_arxiv_results(arxiv_search(**arguments))

# Callable for the bash tool.
#Contains the tool call function and output formatter (head/tail and )
def run_bash_tool(**arguments):
    """Run bash and return compact model-ready text (timeouts stay detectable)."""
    result = run_bash(**arguments)
    if isinstance(result, dict):
        if result.get("timed_out"):
            return (
                f"Timed out after {result.get('timeout_seconds', '?')} seconds.\n"
                f"{format_bash_result(result)}"
            )
        return format_bash_result(result)
    return result


def search_web(**arguments):
    """Clamp result count, search the web, return compact model-ready text."""
    requested = arguments.get("max_results", 5)
    if not isinstance(requested, int) or isinstance(requested, bool):
        requested = 5
    arguments["max_results"] = min(max(requested, 1), WEB_MAX_RESULTS_PER_CALL)
    return format_web_results(web_search(**arguments))


TOOL_SCHEMAS = [
    BASH_TOOL_SCHEMA,
    ARXIV_SEARCH_TOOL_SCHEMA,
    WEB_SEARCH_TOOL_SCHEMA,
    TODO_TOOL_SCHEMA,
]
TOOLS = {
    "bash": run_bash_tool,
    "arxiv_search": search_arxiv,
    "web_search": search_web,
    "write_todos": write_todos,
}
TOOL_NAMES = list(TOOLS)
TOOL_GROUPS = {
    "shell": ["bash"],
    "search": ["arxiv_search", "web_search"],
    "productivity": ["write_todos"],
}
