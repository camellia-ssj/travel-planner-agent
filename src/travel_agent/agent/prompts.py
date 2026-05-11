"""Prompt helpers for structured travel planning."""

from __future__ import annotations

from travel_agent.agent.schemas import TravelRequest
from travel_agent.rag.models import EvidenceBundle, SearchResult

PLANNER_SYSTEM_PROMPT = (
    "You are a careful travel planning agent. Generate a structured TravelPlan only from the "
    "provided RAG evidence. Do not invent unavailable facts. Every day plan and risk notice must "
    "reference the provided evidence_sources. If evidence is thin, keep the plan conservative and "
    "include fallback alternatives."
)


def build_planner_prompt(
    request: TravelRequest,
    evidence: EvidenceBundle,
    user_feedback: list[str] | None = None,
    tool_results: dict[str, object] | None = None,
) -> str:
    """Build the user prompt for structured travel planning."""

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
