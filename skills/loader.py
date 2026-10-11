"""Skill loading, validation, and prompt rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

SKILLS_DIR = Path(__file__).resolve().parent

REQUIRED_KEYS = {"name", "description", "triggers", "tools"}


@dataclass
class Skill:
    name: str
    description: str
    triggers: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    category: str = "general"
    version: str = "0.1.0"
    body: str = ""


def _split_front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        raise ValueError("SKILL.md must start with a --- front-matter block.")
    end = text.find("\n---", 3)
    if end == -1:
        raise ValueError("SKILL.md front-matter block is not closed with ---.")
    front = yaml.safe_load(text[3:end]) or {}
    return front, text[end + 4 :].lstrip("\n")


def load_skills(skills_dir: Path = SKILLS_DIR) -> dict[str, Skill]:
    """Load and validate every skills/*/SKILL.md file."""
    from tools.registry import TOOL_NAMES

    skills: dict[str, Skill] = {}
    for path in sorted(skills_dir.glob("*/SKILL.md")):
        front, body = _split_front_matter(path.read_text(encoding="utf-8"))
        missing = REQUIRED_KEYS - set(front)
        if missing:
            raise ValueError(f"{path}: missing front-matter keys {sorted(missing)}.")
        unknown_tools = [tool for tool in front["tools"] if tool not in TOOL_NAMES]
        if unknown_tools:
            raise ValueError(f"{path}: unknown tools {unknown_tools}.")
        name = front["name"]
        if name in skills:
            raise ValueError(f"Duplicate skill name: {name}.")
        if len(body) > 4000:
            raise ValueError(f"{path}: body exceeds 4000 chars; keep skills concise.")
        skills[name] = Skill(
            name=name,
            description=front["description"],
            triggers=[str(trigger).lower() for trigger in front["triggers"]],
            tools=list(front["tools"]),
            category=str(front.get("category", "general")),
            version=str(front.get("version", "0.1.0")),
            body=body,
        )
    return skills


def match_skills(skills: dict[str, Skill], text: str, limit: int = 2) -> list[Skill]:
    """Score user text against skill triggers; best first, capped."""
    words = set(text.lower().split())
    scored = []
    for skill in skills.values():
        hits = sum(1 for trigger in skill.triggers if trigger in words or trigger in text.lower())
        if hits:
            scored.append((hits, skill.name, skill))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [skill for _, _, skill in scored[:limit]]


def skill_index_block(skills: dict[str, Skill]) -> str:
    """Compact index injected into the system prompt."""
    lines = ["Available skills (call read_skill with the exact name to load one):"]
    for skill in sorted(skills.values(), key=lambda item: item.name):
        lines.append(f"- {skill.name}: {skill.description}")
    return "\n".join(lines)


def skill_groups(skills: dict[str, Skill]) -> dict[str, list[str]]:
    """Group skill names by category for the splash listing."""
    groups: dict[str, list[str]] = {}
    for skill in skills.values():
        groups.setdefault(skill.category, []).append(skill.name)
    return {group: sorted(names) for group, names in sorted(groups.items())}
