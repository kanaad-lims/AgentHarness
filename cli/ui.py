"""NEXUS."""

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.text import Text

from cli.art import EMBLEM_LINES

APP_NAME = "NEXUS-AGENT"
APP_VERSION = "v0.1.0"
ACCENT = "#FFB52E"
DIM = "yellow3"

BANNER = r"""
███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗███████╗
████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██╔════╝
██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ███████╗
██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ╚════██║
██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ███████║
╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝
                                                                                                     
""".strip("\n")


BANNER_GRADIENT = [
    "#FFE955",
    "#FFD93B",
    "#FFC93C",
    "#FFB52E",
    "#FF9E2C",
    "#FF8C1A",
]


def _tool_groups(tool_names: list[str]) -> list[str]:
    groups: dict[str, list[str]] = {}
    for name in tool_names:
        groups.setdefault("general", []).append(name)
    return [f"[dim]{group}[/]:  " + ", ".join(sorted(names)) for group, names in sorted(groups.items())]


def _skill_groups(skills: dict[str, list[str]]) -> list[str]:
    if not skills:
        return ["[dim]none yet[/] — skills land after the injection layer"]
    return [f"[dim]{group}[/]:  " + ", ".join(sorted(names)) for group, names in sorted(skills.items())]


def render_splash(
    console: Console,
    tool_names: list[str],
    skills: dict[str, list[str]] | None = None,
) -> None:
    """Render the two-column splash: banner/tools/skills left, emblem right."""
    skills = skills or {}
    left = Text(no_wrap=False)
    for index, line in enumerate(BANNER.splitlines()):
        color = BANNER_GRADIENT[index % len(BANNER_GRADIENT)]
        left.append(line + "\n", style=color)
    left.append(f"\n{APP_NAME} {APP_VERSION} ", style=f"bold {ACCENT}")
    left.append("•  experimental harness  •  local runtime\n", style=ACCENT)
    left.append("\nAvailable Tools\n", style=f"bold {ACCENT}")
    for row in _tool_groups(tool_names):
        left.append(row + "\n")
    left.append("\nAvailable Skills\n", style=f"bold {ACCENT}")
    for row in _skill_groups(skills):
        left.append(row + "\n")
    left.append(
        f"\n{len(tool_names)} tools  •  "
        f"{sum(len(v) for v in skills.values())} skills  •  /help for commands",
        style=DIM,
    )

    right = Text("\n".join(EMBLEM_LINES), style=ACCENT)
    columns = Columns([left, right], equal=False, expand=True)
    console.print(Panel(columns, border_style=ACCENT, padding=(1, 2)))


def approval_card(console: Console, header: str, command: str) -> None:
    """Render a tool-approval request as a highlighted card."""
    body = Text()
    body.append(header + "\n", style=DIM)
    body.append(command, style=f"bold {ACCENT}")
    console.print(Panel(body, border_style=ACCENT, padding=(0, 1)))


def show_tool_event(console: Console, text: str) -> None:
    console.print(f"[{DIM}]{text}[/]")


def show_answer(console: Console, text: str) -> None:
    console.print(f"\n[bold]Agent:[/] {text}\n")


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
    """Print the nexus:local session status bar."""
    from datetime import datetime

    clock = datetime.now().strftime("%H:%M:%S")
    console.print(
        f"[{ACCENT}]◆ nexus:local[/]"
        f"  [dim]session {session_id}[/]"
        f"  [{ACCENT}]{clock}  •  READY[/]"
    )
    console.rule(style=DIM)


def echo_user_input(console: Console, text: str) -> None:
    """Echo the user's message with the > prefix."""
    console.print(f"[bold {ACCENT}]>[/] {text}")
