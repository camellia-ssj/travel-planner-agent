"""Deterministic alternative suggestion tool."""

from __future__ import annotations

from travel_agent.agent.schemas import AlternativePlan, AlternativeSuggestion, CrowdRiskAssessment
from travel_agent.rag.models import EvidenceBundle

_MAX_ALTERNATIVES = 5

_FALLBACK = AlternativeSuggestion(
    original_scenario="行程受限或天气不佳",
    suggested_alternative="选择室内景点或文化体验活动",
    reason="作为通用备选方案",
)


def suggest_alternatives(
    destination: str,
    evidence: EvidenceBundle,
    crowd_assessment: CrowdRiskAssessment | None = None,
) -> AlternativePlan:
    """Generate alternative suggestions based on evidence and crowd risk."""

    suggestions: list[AlternativeSuggestion] = []
    weather_parts: list[str] = []

    # Collect high-risk POI names from crowd assessment
    high_risk_pois = set()
    if crowd_assessment:
        for poi_risk in crowd_assessment.poi_risks:
            if poi_risk.risk_level == "high":
                high_risk_pois.add(poi_risk.poi_name)

    for result in evidence.results:
        section = str(result.metadata.get("section", ""))

        if section == "alternatives":
            scenario = _build_scenario(result.content, high_risk_pois)
            suggestions.append(AlternativeSuggestion(
                original_scenario=scenario,
                suggested_alternative=_clean_content(result.content),
                reason=f"来自{destination}备选方案知识",
            ))

        elif section == "weather_risk":
            weather_parts.append(_clean_content(result.content))
            suggestions.append(AlternativeSuggestion(
                original_scenario="天气不佳时",
                suggested_alternative=_clean_content(result.content),
                reason="天气风险应对",
            ))

    if not suggestions:
        suggestions.append(_FALLBACK)

    weather_note = (
        "；".join(weather_parts[:2])
        if weather_parts
        else f"{destination}暂无特别天气风险提示"
    )

    return AlternativePlan(
        destination=destination,
        alternatives=suggestions[:_MAX_ALTERNATIVES],
        weather_note=weather_note,
    )


def _build_scenario(content: str, high_risk_pois: set[str]) -> str:
    if high_risk_pois:
        poi = next(iter(high_risk_pois))
        return f"热门景点{poi}拥挤时"
    return "原定行程受限时"


def _clean_content(content: str) -> str:
    cleaned = " ".join(content.split())
    return cleaned[:200]
