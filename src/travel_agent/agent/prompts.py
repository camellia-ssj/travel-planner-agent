"""结构化旅行规划与反思的提示词辅助工具。"""

from __future__ import annotations

from travel_agent.agent.schemas import TravelPlan, TravelRequest
from travel_agent.memory.models import UserProfile
from travel_agent.rag.models import EvidenceBundle, SearchResult

PLANNER_SYSTEM_PROMPT = (
    "你是一个严谨的旅行规划代理。请仅根据提供的 RAG 证据生成结构化的 TravelPlan。"
    "不要编造不可用的事实。每个日计划和风险提示都必须引用提供的 evidence_sources。"
    "如果证据不足，请保持计划保守，并包含备选方案。"
)

REFLECTION_SYSTEM_PROMPT = (
    "你是一个一丝不苟的事实检查员，负责审查旅行计划。你的任务是找出计划中所有"
    "没有 RAG 证据支持的声明。逐一交叉核验每个活动、预算项、风险提示和备选方案"
    "是否与证据相符。标记任何编造的 POI 名称、虚构的价格、没有依据的人群声明，"
    "或与证据相矛盾的活动。请务必全面——漏报（遗漏幻觉）比误报（标记了实际有"
    "证据支持的内容）更严重。"
)


def build_planner_prompt(
    request: TravelRequest,
    evidence: EvidenceBundle,
    user_feedback: list[str] | None = None,
    tool_results: dict[str, object] | None = None,
    user_profile: UserProfile | None = None,
) -> str:
    """构建用于结构化旅行规划的用户提示词。

    当提供带有出行历史的 *user_profile* 时，会将其作为附加上下文纳入，
    以便规划器能够提供个性化推荐。
    """

    feedback = user_feedback or []
    evidence_text = "\n\n".join(_format_result(index, result) for index, result in enumerate(
        evidence.results,
        start=1,
    ))
    sources = ", ".join(_evidence_sources(evidence.results)) or "none"
    audience = ", ".join(request.audience)
    prompt = (
        "请为此请求创建一个 TravelPlan。\n\n"
        f"原始请求: {request.raw_query}\n"
        f"目的地: {request.destination or evidence.query_analysis.get('destination', '')}\n"
        f"天数: {request.days}\n"
        f"出行人员: {audience}\n"
        f"预算偏好: {request.budget_preference}\n"
        f"用户后续反馈: {feedback or '无'}\n"
        f"必需的 evidence_sources: {sources}\n\n"
    )
    if user_profile is not None and user_profile.total_trips > 0:
        profile_text = user_profile.to_context_text()
        if profile_text:
            prompt += f"{profile_text}\n\n"
    prompt += (
        "RAG 证据:\n"
        f"{evidence_text or '未检索到证据。'}\n\n"
    )
    tool_section = _format_tool_results(tool_results)
    if tool_section:
        prompt += tool_section + "\n\n"
    prompt += (
        "约束条件:\n"
        "- 输出必须符合 TravelPlan schema。\n"
        "- 当解析出的请求中存在 destination 和 days 时，必须与之匹配。\n"
        "- day_plans 的长度必须等于 days。\n"
        "- evidence_sources 只能包含所提供的来源名称。\n"
        "- alternatives 在有 section=alternatives 的证据时，应优先使用该部分证据。\n"
        "- risk_notices 在有证据支持时，必须包含人流、天气或一般风险提醒。"
    )
    return prompt


def _format_result(index: int, result: SearchResult) -> str:
    section = str(result.metadata.get("section", ""))
    return (
        f"[{index}] source={result.source}; destination={result.destination}; "
        f"section={section}; score={result.score:.4f}\n{result.content}"
    )


def _evidence_sources(results: list[SearchResult]) -> list[str]:
    sources: list[str] = []
    for result in results:
        if result.source and result.source not in sources:
            sources.append(result.source)
    return sources


def _format_tool_results(tool_results: dict[str, object] | None) -> str:
    if not tool_results:
        return ""
    import json

    sections: list[str] = ["## 确定性工具计算结果（作为 ground truth 使用，请勿自行编造预算数字）"]
    budget = tool_results.get("tool_budget")
    if budget is not None:
        sections.append(f"Budget estimate: {json.dumps(budget.model_dump(), ensure_ascii=False)}")
    crowd = tool_results.get("tool_crowd_risk")
    if crowd is not None:
        sections.append(f"Crowd risk: {json.dumps(crowd.model_dump(), ensure_ascii=False)}")
    alt = tool_results.get("tool_alternatives")
    if alt is not None:
        sections.append(f"Alternatives: {json.dumps(alt.model_dump(), ensure_ascii=False)}")
    return "\n".join(sections)


def build_reflection_prompt(
    plan: TravelPlan,
    evidence: EvidenceBundle,
    tool_results: dict[str, object] | None = None,
) -> str:
    """构建用于计划反思/事实性审查的用户提示词。

    包含完整的计划文本和所有 RAG 证据，以便事实检查员可以将每项声明
    与其支撑来源进行交叉对照。
    """
    evidence_text = "\n\n".join(
        _format_result(index, result)
        for index, result in enumerate(evidence.results, start=1)
    )
    plan_text = _format_plan_for_review(plan)
    prompt = (
        "请根据提供的 RAG 证据审查以下旅行计划的事实准确性。\n\n"
        f"=== 旅行计划 ===\n{plan_text}\n\n"
        f"=== RAG 证据 ===\n{evidence_text or '未检索到证据。'}\n\n"
    )
    tool_section = _format_tool_results(tool_results)
    if tool_section:
        prompt += (
            f"{tool_section}\n\n"
            "上述工具结果是确定性的 ground truth。"
            "请标记任何与之矛盾的计划内容。\n\n"
        )
    prompt += (
        "指令:\n"
        "- 找出计划中所有没有证据支持的声明。\n"
        "- 标记编造的 POI 名称、虚构的价格、没有依据的人群声明。\n"
        "- 检查活动是否与目的地匹配"
        "（例如，不应在杭州计划中出现北京的 POI）。\n"
        "- 在有工具预算结果时，将预算项与 tool_budget 结果交叉核验。\n"
        "- 在有人流风险工具结果时，将风险提示与 tool_crowd_risk 结果交叉核验。\n"
        "- 对每个被标记的声明，请注明：位置、声明文本、标记原因、严重程度（高/中/低）。\n"
        "- 计算 evidence_coverage（0.0-1.0）：计划中有证据支撑的声明占比。\n"
        "- 计算 confidence_score（0.0-1.0）：审查后的整体置信度。\n"
        "- 提供可操作的改进建议。\n"
        "- 如未发现问题，请返回空的 hallucination_flags 列表，并将 passed 设为 true。"
    )
    return prompt


def _format_plan_for_review(plan: TravelPlan) -> str:
    """将 TravelPlan 格式化为便于事实检查员阅读的文本。"""
    import json

    lines: list[str] = [
        f"Destination: {plan.destination}",
        f"Days: {plan.days}",
        f"Summary: {plan.summary}",
        f"Evidence Sources: {plan.evidence_sources}",
        f"Fallback Used: {plan.fallback_used}",
        "",
        "Day Plans:",
    ]
    for i, day in enumerate(plan.day_plans):
        lines.append(f"  Day {day.day} — {day.title}")
        for j, activity in enumerate(day.activities):
            lines.append(f"    [{i}.activities[{j}]] {activity}")
    lines.append("")
    lines.append("Budget Items:")
    for i, item in enumerate(plan.budget_items):
        lines.append(f"  [{i}] {item.category}: {item.preference} — {item.note}")
    lines.append("")
    lines.append("Risk Notices:")
    for i, notice in enumerate(plan.risk_notices):
        lines.append(f"  [{i}] [{notice.severity}] {notice.risk_type}: {notice.message}")
    lines.append("")
    lines.append(f"Alternatives: {json.dumps(plan.alternatives, ensure_ascii=False)}")
    return "\n".join(lines)
