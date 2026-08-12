"""Anthropic-style project skill loading and selection."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from travel_agent.agent.schemas import TravelRequest
from travel_agent.memory.models import UserProfile
from travel_agent.skills.models import SkillDefinition, SkillSelection

ANTHROPIC_SKILL_FILENAME = "SKILL.md"
DEFAULT_PROJECT_SKILLS_DIR = Path(".claude") / "skills"
MAX_SKILL_FILE_BYTES = 30_000
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
RESERVED_SKILL_NAME_PARTS = {"anthropic", "claude"}


class SkillRegistry:
    """Load project skills from ``.claude/skills`` and select active skills."""

    def __init__(self, skill_dirs: Iterable[Path]) -> None:
        self.skill_dirs = tuple(Path(path) for path in skill_dirs)

    @classmethod
    def from_project(
        cls,
        project_root: Path | None = None,
        extra_dirs: Iterable[Path] | None = None,
    ) -> SkillRegistry:
        root = project_root or Path.cwd()
        dirs = [root / DEFAULT_PROJECT_SKILLS_DIR]
        env_dirs = os.getenv("TRAVEL_AGENT_SKILLS_DIRS", "").strip()
        if env_dirs:
            dirs.extend(Path(item.strip()) for item in env_dirs.split(",") if item.strip())
        if extra_dirs:
            dirs.extend(Path(path) for path in extra_dirs)
        return cls(dirs)

    def list_skills(self) -> list[SkillDefinition]:
        skills: dict[str, SkillDefinition] = {}
        for skill_dir in self.skill_dirs:
            if not skill_dir.exists():
                continue
            for skill_file in sorted(skill_dir.glob(f"*/{ANTHROPIC_SKILL_FILENAME}")):
                skill = _read_skill(skill_file)
                if skill is not None:
                    skills[skill.name] = skill
        return sorted(skills.values(), key=lambda item: item.name)

    def load_skill(self, name: str) -> SkillDefinition:
        for skill in self.list_skills():
            if skill.name == name:
                return skill
        raise KeyError(f"skill not found: {name}")

    def select_skills(
        self,
        request: TravelRequest,
        question: str,
        user_profile: UserProfile | None = None,
        max_skills: int = 4,
    ) -> SkillSelection:
        scored: list[tuple[int, SkillDefinition, list[str]]] = []
        text = _selection_text(request, question, user_profile)
        for skill in self.list_skills():
            score, matched = _score_skill(skill, request, text)
            if score > 0:
                scored.append((score, skill, matched))

        if not scored:
            fallback = _find_default_skill(self.list_skills())
            if fallback is None:
                return SkillSelection(selection_reason="No project skills matched.")
            scored.append((1, fallback, ["default"]))

        scored.sort(key=lambda item: (item[0], item[1].priority, item[1].name), reverse=True)
        active = [skill.to_applied(matched) for _, skill, matched in scored[:max_skills]]
        reason_parts = [
            f"{skill.name} matched {', '.join(matched)}"
            for _, skill, matched in scored[:max_skills]
        ]
        reason = "; ".join(reason_parts)
        return SkillSelection(active_skills=active, selection_reason=reason)


def build_skill_registry(
    project_root: Path | None = None,
    skill_dirs: Iterable[Path] | None = None,
    enabled: bool = True,
) -> SkillRegistry | None:
    if not enabled:
        return None
    return SkillRegistry.from_project(project_root=project_root, extra_dirs=skill_dirs)


@lru_cache(maxsize=128)
def _read_skill(skill_file: Path) -> SkillDefinition | None:
    if not skill_file.is_file() or skill_file.stat().st_size > MAX_SKILL_FILE_BYTES:
        return None
    text = skill_file.read_text(encoding="utf-8")
    parsed = _parse_frontmatter(text)
    if parsed is None:
        return None
    metadata, body = parsed
    name = str(metadata.get("name", "")).strip()
    description = str(metadata.get("description", "")).strip()
    if not _valid_skill_name(name) or not description:
        return None
    return SkillDefinition(
        name=name,
        description=description,
        body=body.strip(),
        path=str(skill_file),
        priority=_parse_priority(body),
        keywords=_parse_csv_field(body, "Keywords"),
        required_tools=_parse_csv_field(body, "Required tools"),
        retrieval_focus=_extract_section(body, "Retrieval Focus"),
        planner_instructions=_extract_section(body, "Planner Instructions"),
        reflection_checks=_extract_section(body, "Reflection Checks"),
    )


def _parse_frontmatter(text: str) -> tuple[dict[str, object], str] | None:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    end_line = -1
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_line = index
            break
    if end_line < 0:
        return None
    frontmatter = "\n".join(lines[1:end_line])
    body = "\n".join(lines[end_line + 1:]).lstrip()
    metadata = yaml.safe_load(frontmatter) or {}
    if not isinstance(metadata, dict):
        return None
    return metadata, body


def _valid_skill_name(name: str) -> bool:
    if not SKILL_NAME_PATTERN.match(name):
        return False
    return not any(part in name for part in RESERVED_SKILL_NAME_PARTS)


def _parse_priority(body: str) -> int:
    match = re.search(r"(?im)^\s*-\s*Priority:\s*(\d+)\s*$", body)
    if not match:
        return 50
    return max(0, min(100, int(match.group(1))))


def _parse_csv_field(body: str, label: str) -> list[str]:
    match = re.search(rf"(?im)^\s*-\s*{re.escape(label)}:\s*(.+)$", body)
    if not match:
        return []
    raw = match.group(1).replace("，", ",")
    return [item.strip() for item in raw.split(",") if item.strip()]


def _extract_section(body: str, title: str) -> str:
    pattern = re.compile(rf"(?ims)^##\s+{re.escape(title)}\s*\n(.*?)(?=^##\s+|\Z)")
    match = pattern.search(body)
    if not match:
        return ""
    return match.group(1).strip()


def _selection_text(
    request: TravelRequest,
    question: str,
    user_profile: UserProfile | None,
) -> str:
    parts = [
        question,
        request.raw_query,
        request.destination,
        request.budget_preference,
        " ".join(request.audience),
    ]
    if user_profile is not None:
        parts.extend([
            " ".join(user_profile.audience_types),
            user_profile.budget_preference,
            user_profile.preferences_summary,
        ])
    return " ".join(part for part in parts if part).lower()


def _score_skill(
    skill: SkillDefinition,
    request: TravelRequest,
    text: str,
) -> tuple[int, list[str]]:
    score = 0
    matched: list[str] = []
    for keyword in skill.keywords:
        if keyword.lower() in text:
            score += 5
            matched.append(keyword)

    name = skill.name
    audience = set(request.audience)
    if name == "family-travel-planner" and audience & {"family_with_children", "elderly"}:
        score += 8
        matched.append("audience")
    if name == "budget-aware-planner" and request.budget_preference == "economy":
        score += 8
        matched.append("economy-budget")
    if name == "crowd-avoidance-planner" and any(
        token in text for token in ("周末", "节假日", "五一", "十一", "春节", "crowd", "holiday")
    ):
        score += 8
        matched.append("crowd-context")
    if name == "weather-risk-planner" and any(
        token in text for token in ("雨", "下雨", "天气", "高温", "台风", "weather", "rain")
    ):
        score += 8
        matched.append("weather-context")
    if name == "free-independent-planner" and any(
        token in text for token in ("自由行", "个人", "一个人", "solo", "friends", "朋友")
    ):
        score += 6
        matched.append("independent-travel")

    return score, matched


def _find_default_skill(skills: list[SkillDefinition]) -> SkillDefinition | None:
    for skill in skills:
        if skill.name == "free-independent-planner":
            return skill
    return skills[0] if skills else None
