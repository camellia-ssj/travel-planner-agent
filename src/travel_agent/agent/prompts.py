"""Prompt helpers for structured travel planning and reflection."""

from __future__ import annotations

from travel_agent.agent.schemas import TravelPlan, TravelRequest
from travel_agent.memory.models import UserProfile
from travel_agent.rag.models import EvidenceBundle, SearchResult

PLANNER_SYSTEM_PROMPT = (
    "You are a careful travel planning agent. Generate a structured TravelPlan only from the "
    "provided RAG evidence. Do not invent unavailable facts. Every day plan and risk notice must "
    "reference the provided evidence_sources. If evidence is thin, keep the plan conservative and "
    "include fallback alternatives."
)

REFLECTION_SYSTEM_PROMPT = (
    "You are a meticulous fact-checker reviewing a travel plan. Your task is to identify every "
    "claim in the plan that is NOT supported by the provided RAG evidence. Cross-check each "
    "activity, budget item, risk notice, and alternative against the evidence. Flag any invented "
    "POI names, fabricated prices, unsupported crowd claims, or activities that contradict the "
    "evidence. Be thorough — a false negative (missing a hallucination) is worse than a false "
    "positive (flagging something that is actually supported)."
)


def build_planner_prompt(
    request: TravelRequest,
    evidence: EvidenceBundle,
    user_feedback: list[str] | None = None,
    tool_results: dict[str, object] | None = None,
    user_profile: UserProfile | None = None,
) -> str:
    """Build the user prompt for structured travel planning.

    When *user_profile* is provided with trip history, it is included as
    additional context so the planner can personalize recommendations.
    """

    feedback = user_feedback or []
    evidence_text = "\n\n".join(_format_result(index, result) for index, result in enumerate(
        evidence.results,
        start=1,
    ))
    sources = ", ".join(_evidence_sources(evidence.results)) or "none"
    audience = ", ".join(request.audience)
    prompt = (
        "Create a TravelPlan for this request.\n\n"
        f"Raw request: {request.raw_query}\n"
        f"Destination: {request.destination or evidence.query_analysis.get('destination', '')}\n"
        f"Days: {request.days}\n"
        f"Audience: {audience}\n"
        f"Budget preference: {request.budget_preference}\n"
        f"User follow-up feedback: {feedback or 'none'}\n"
        f"Required evidence_sources: {sources}\n\n"
    )
    if user_profile is not None and user_profile.total_trips > 0:
        profile_text = user_profile.to_context_text()
        if profile_text:
            prompt += f"{profile_text}\n\n"
    prompt += (
        "RAG evidence:\n"
        f"{evidence_text or 'No evidence retrieved.'}\n\n"
    )
    tool_section = _format_tool_results(tool_results)
    if tool_section:
        prompt += tool_section + "\n\n"
    prompt += (
        "Constraints:\n"
        "- Output must match the TravelPlan schema.\n"
        "- destination and days must match the parsed request when present.\n"
        "- day_plans length must equal days.\n"
        "- evidence_sources must include only the provided source names.\n"
        "- alternatives should prefer evidence from section=alternatives when available.\n"
        "- risk_notices must include crowd, weather or general risk reminders when supported."
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
    """Build the user prompt for plan reflection / factuality review.

    Includes the full plan text and all RAG evidence so the fact-checker can
    cross-reference every claim against its supporting sources.
    """
    evidence_text = "\n\n".join(
        _format_result(index, result)
        for index, result in enumerate(evidence.results, start=1)
    )
    plan_text = _format_plan_for_review(plan)
    prompt = (
        "Review the following travel plan for factual accuracy against the provided RAG"
        " evidence.\n\n"
        f"=== TRAVEL PLAN ===\n{plan_text}\n\n"
        f"=== RAG EVIDENCE ===\n{evidence_text or 'No evidence retrieved.'}\n\n"
    )
    tool_section = _format_tool_results(tool_results)
    if tool_section:
        prompt += (
            f"{tool_section}\n\n"
            "The tool results above are deterministic ground truth. "
            "Flag any plan content that contradicts them.\n\n"
        )
    prompt += (
        "Instructions:\n"
        "- Identify every claim in the plan that is NOT supported by the evidence.\n"
        "- Flag invented POI names, fabricated prices, unsupported crowd claims.\n"
        "- Check that activities match the destination"
        " (e.g., no Beijing POIs in a Hangzhou plan).\n"
        "- Cross-check budget items against tool_budget results when available.\n"
        "- Cross-check risk notices against tool_crowd_risk results when available.\n"
        "- For each flagged claim, specify: location, claim text, why it's flagged,"
        " severity (high/medium/low).\n"
        "- Compute evidence_coverage (0.0-1.0): fraction of plan claims grounded in evidence.\n"
        "- Compute confidence_score (0.0-1.0): overall confidence after review.\n"
        "- Provide actionable suggestions for improving the plan.\n"
        "- If no issues are found, return an empty hallucination_flags list and passed=true."
    )
    return prompt


def _format_plan_for_review(plan: TravelPlan) -> str:
    """Format a TravelPlan as readable text for the fact-checker."""
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
