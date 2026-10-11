"""Skill tool: load one skill body by exact name."""

from config import SKILL_BODY_MAX_CHARS

READ_SKILL_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_skill",
        "description": (
            "Load a skill's full instructions by exact name. Call this when a "
            "task matches an available skill, then follow the loaded protocol."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Exact skill name from the available-skills index.",
                }
            },
            "required": ["skill_name"],
            "additionalProperties": False,
        },
    },
}


def read_skill(skill_name: str, *, skills: dict | None = None) -> str:
    """Return a skill body, capped, or an error listing valid names."""
    from skills.loader import load_skills

    loaded = skills if skills is not None else load_skills()
    name = skill_name.strip() if isinstance(skill_name, str) else ""
    skill = loaded.get(name)
    if skill is None:
        valid = ", ".join(sorted(loaded))
        return f"Error: unknown skill '{skill_name}'. Valid names: {valid}."
    body = skill.body.strip()
    if len(body) > SKILL_BODY_MAX_CHARS:
        body = body[:SKILL_BODY_MAX_CHARS].rsplit(" ", 1)[0] + "\n...[skill truncated]"
    return f"Skill '{skill.name}' (v{skill.version}):\n{body}"
