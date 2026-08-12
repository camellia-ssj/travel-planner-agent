"""面向 Skill 驱动旅行规划的项目 Skill 模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SkillDefinition(BaseModel):
    """解析后的 Anthropic 风格项目 Skill。"""

    name: str
    description: str
    body: str
    path: str
    priority: int = 50
    keywords: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    retrieval_focus: str = ""
    planner_instructions: str = ""
    reflection_checks: str = ""

    def to_applied(self, matched_keywords: list[str]) -> AppliedSkill:
        return AppliedSkill(
            name=self.name,
            description=self.description,
            priority=self.priority,
            matched_keywords=matched_keywords,
            required_tools=self.required_tools,
            retrieval_focus=self.retrieval_focus,
            planner_instructions=self.planner_instructions,
            reflection_checks=self.reflection_checks,
            source_path=self.path,
        )


class AppliedSkill(BaseModel):
    """单次 Agent 运行中选中的 Skill。"""

    name: str
    description: str
    priority: int = 50
    matched_keywords: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    retrieval_focus: str = ""
    planner_instructions: str = ""
    reflection_checks: str = ""
    source_path: str = ""


class SkillSelection(BaseModel):
    """存储在 LangGraph 状态中的 Skill 选择结果。"""

    active_skills: list[AppliedSkill] = Field(default_factory=list)
    selection_reason: str = ""

    @property
    def names(self) -> list[str]:
        return [skill.name for skill in self.active_skills]

    def required_tools(self) -> list[str]:
        tools: list[str] = []
        for skill in self.active_skills:
            for tool in skill.required_tools:
                if tool and tool not in tools:
                    tools.append(tool)
        return tools

    def retrieval_focus_text(self) -> str:
        lines = [
            f"- {skill.name}: {skill.retrieval_focus}"
            for skill in self.active_skills
            if skill.retrieval_focus
        ]
        return "\n".join(lines)

    def planner_prompt_text(self) -> str:
        if not self.active_skills:
            return ""
        sections = ["## 当前激活的旅行规划 Skill"]
        for skill in self.active_skills:
            sections.append(f"### {skill.name}")
            if skill.planner_instructions:
                sections.append(skill.planner_instructions)
            if skill.required_tools:
                sections.append(f"所需工具信号: {', '.join(skill.required_tools)}")
        return "\n".join(sections)

    def reflection_prompt_text(self) -> str:
        checks = [
            f"- {skill.name}: {skill.reflection_checks}"
            for skill in self.active_skills
            if skill.reflection_checks
        ]
        return "\n".join(checks)
