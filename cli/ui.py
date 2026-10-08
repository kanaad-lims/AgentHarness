"""BOREAS."""

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.text import Text

from cli.art import EMBLEM

APP_NAME = "BOREAS-AGENT"
APP_VERSION = "v0.1.0"
ACCENT = "#2BD97C"
DIM = "#5E9C7B"

BANNER = r"""
██████╗  ██████╗ ██████╗ ███████╗ █████╗ ███████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗
██╔══██╗██╔═══██╗██╔══██╗██╔════╝██╔══██╗██╔════╝      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
██████╔╝██║   ██║██████╔╝█████╗  ███████║███████╗█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   
██╔══██╗██║   ██║██╔══██╗██╔══╝  ██╔══██║╚════██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   
██████╔╝╚██████╔╝██║  ██║███████╗██║  ██║███████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   
╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   
                                                                                                   
                                                                                                     
                                                                                                  
""".strip("\n")


BANNER_GRADIENT = [
    "#D2FAD8",
    "#9BF2B4",
    "#54DE8B",
    "#1FBD66",
    "#0B7A43",
]


# Placeholder skill listing until the skills layer lands.
PLACEHOLDER_SKILLS = {
    "research": ["arxiv", "web-research"],
    "software-development": ["code-review", "debugging"],
    "productivity": ["task-planning"],
}


MAX_ROW_NAMES = 4


def _group_rows(groups: dict[str, list[str]]) -> list[str]:
    rows = []
    for group, names in sorted(groups.items()):
        shown = names[:MAX_ROW_NAMES]
        extra = len(names) - len(shown)
        row = f"[dim]{group}[/]:  " + ", ".join(shown)
        if extra > 0:
            row += f"  [dim]+{extra} more[/]"
        rows.append(row)
    return rows


def render_splash(
    console: Console,
    tool_groups: dict[str, list[str]],
    skills: dict[str, list[str]] | None = None,
    *,
    model: str = "",
    session_id: str = "",
    working_dir: str = "",
) -> None:
    """Hermes-style splash: full-width banner, titled panel, emblem left."""
    import os

    skills = skills if skills is not None else PLACEHOLDER_SKILLS
    tool_count = sum(len(names) for names in tool_groups.values())
    skill_count = sum(len(names) for names in skills.values())
    working_dir = working_dir or os.getcwd()

    console.print()

    left = Text()
    left.append("\n".join(EMBLEM.splitlines()) + "\n\n", style=ACCENT)
    left.append(f"{model}  •  local runtime\n", style=f"bold {ACCENT}")
    left.append(f"{working_dir}\n", style=DIM)
    left.append(f"Session: {session_id}\n", style=DIM)

    right = Text(no_wrap=False)
    banner_lines = BANNER.splitlines()
    for index, line in enumerate(banner_lines):
        color = BANNER_GRADIENT[index % len(BANNER_GRADIENT)]
        right.append(line, style=f"bold {color}")
        if index < len(banner_lines) - 1:
            right.append("\n")
    right.append(f"\n\n{APP_NAME} {APP_VERSION}\n", style=f"bold {ACCENT}")
    right.append("\nAvailable Tool Groups\n", style=f"bold {ACCENT}")
    for row in _group_rows(tool_groups):
        right.append_text(Text.from_markup(row + "\n"))
    right.append("\nAvailable Skills\n", style=f"bold {ACCENT}")
    for row in _group_rows(skills):
        right.append_text(Text.from_markup(row + "\n"))
    right.append(
        f"\n{tool_count} tools  •  {skill_count} skills  •  /help for commands",
        style=DIM,
    )

    title = f"{APP_NAME} {APP_VERSION}  •  experimental harness  •  local runtime"
    columns = Columns([left, right], equal=False, expand=True)
    console.print(Panel(columns, title=title, border_style=ACCENT, padding=(1, 2)))


def show_welcome(console: Console) -> None:
    """Print the welcome + tip lines below the splash panel."""
    console.print(
        f"Welcome to {APP_NAME}! Type your message or /help for commands."
    )
    console.print(
        "[dim]◆ Tip: approvals, tools and runtime state are managed by the harness.[/]"
    )


def approval_card(console: Console, header: str, command: str) -> None:
    """Render a tool-approval request as a highlighted card."""
    body = Text()
    body.append(header + "\n", style=DIM)
    body.append(command, style=f"bold {ACCENT}")
    console.print(Panel(body, border_style=ACCENT, padding=(0, 1)))


def show_tool_event(console: Console, text: str) -> None:
    console.print(f"[{DIM}]{text}[/]")


def show_answer(console: Console, text: str) -> None:
    console.print(Panel(text, title="◆Boreas", border_style=ACCENT, padding=(0, 1)))


def run_with_spinner(console: Console, label: str):
    """Spinner context for model/tool runs."""
    return Status(f"[{ACCENT}]{label}...", spinner="dots")


HELP_TEXT = """[bold]Commands[/]
  /help              Show this help
  /browser <cmd>     Run a browser command directly (open <url> | search <query>)
  /tools             List registered tools
  /model             Show current model
  /bye, /exit, /quit  Leave

Tool shortcuts (/<tool>) land with the skills pass; tools run via the model until then."""

PHASE_LABELS = {
    "thinking": "thinking...",
    "planning": "planning...",
    "selecting": "selecting tool...",
    "accepted": "task accepted",
}


def show_phase(console: Console, phase: str) -> None:
    """Print one agent-phase line (thinking / planning / selecting / accepted)."""
    label = PHASE_LABELS.get(phase, phase)
    if phase == "accepted":
        console.print(f"[bold green]✓ {label}[/]")
    elif phase == "thinking":
        console.print(f"[{ACCENT}]◆ {label}[/]")
    else:
        console.print(f"[{DIM}]  → {label}[/]")


def show_session_header(console: Console, session_id: str) -> None:
    """Print the boreas:local session status bar."""
    from datetime import datetime

    clock = datetime.now().strftime("%H:%M:%S")
    console.print(
        f"[{ACCENT}]◆ boreas:local[/]"
        f"  [dim]session {session_id}[/]"
        f"  [{ACCENT}]{clock}  •  READY[/]"
    )
    console.rule(style=DIM)
