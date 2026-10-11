from skills.loader import load_skills, match_skills, skill_groups, skill_index_block
from skills.skill_tool import read_skill
from tools.registry import TOOL_SCHEMAS, TOOLS


def test_skills_load_with_valid_front_matter():
    skills = load_skills()

    assert {"arxiv-research", "web-research", "task-planning", "shell-usage"} <= set(skills)
    assert skills["arxiv-research"].tools == ["arxiv_search"]


def test_skill_index_lists_every_skill():
    skills = load_skills()
    index = skill_index_block(skills)

    for name in skills:
        assert name in index


def test_skill_groups_cover_all_skills():
    skills = load_skills()
    groups = skill_groups(skills)

    assert sum(len(names) for names in groups.values()) == len(skills)


def test_match_skills_finds_research_skill():
    skills = load_skills()

    matched = match_skills(skills, "find papers on attention mechanisms")

    assert matched and matched[0].name == "arxiv-research"


def test_read_skill_returns_body_and_rejects_unknown():
    skills = load_skills()

    body = read_skill("web-research", skills=skills)
    assert "grounding" in body.lower() or "cite" in body.lower()

    error = read_skill("nope", skills=skills)
    assert error.startswith("Error: unknown skill")
    assert "web-research" in error


def test_read_skill_registered_with_schema():
    assert "read_skill" in TOOLS
    names = [schema["function"]["name"] for schema in TOOL_SCHEMAS]
    assert "read_skill" in names
