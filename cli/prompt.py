"""prompt_toolkit session: history, slash-command completion, styling."""

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

COMMANDS = [
    "/bye",
    "/exit",
    "/quit",
    "/help",
    "/browser",
    "/model",
    "/tools",
]

STYLE = Style.from_dict(
    {
        "prompt": "bold #FFB52E",
        "completion-menu.completion": "bg:#1c1c1c #FFB52E",
        "completion-menu.completion.current": "bg:#FFB52E #000000",
    }
)


def create_session(extra_commands: list[str] | None = None) -> PromptSession:
    """Build the interactive prompt session with command completion."""
    words = COMMANDS + (extra_commands or [])
    completer = WordCompleter(words, ignore_case=True, sentence=True)
    return PromptSession(
        message=[("class:prompt", "> ")],
        history=InMemoryHistory(),
        completer=completer,
        style=STYLE,
    )
