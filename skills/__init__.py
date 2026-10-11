"""Skills: loadable capability packs (protocol text + tool bindings)."""

from skills.loader import Skill, load_skills, match_skills, skill_groups, skill_index_block
from skills.skill_tool import READ_SKILL_TOOL_SCHEMA, read_skill

__all__ = [
    "READ_SKILL_TOOL_SCHEMA",
    "Skill",
    "load_skills",
    "match_skills",
    "read_skill",
    "skill_groups",
    "skill_index_block",
]
