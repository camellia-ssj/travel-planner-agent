"""Deterministic crowd risk assessment tool."""

from __future__ import annotations

import re

from travel_agent.agent.schemas import CrowdRiskAssessment, POICrowdRisk
from travel_agent.rag.models import EvidenceBundle

_CONNECTOR_RE = re.compile(r"[和与及、，,。.！!？?\s]+")

_POI_PATTERN = re.compile(
    r"([一-鿿]{1,4}(?:寺|公园|景区|山|湖|塔|街|路|桥|城|庄|馆))"
)

_CONNECTOR_CHARS = set("和与及、，,。.！!？?\t\n\r ")

_HIGH_KEYWORDS = {"拥挤", "高峰", "排队", "人多", "爆满", "拥堵"}
_MEDIUM_KEYWORDS = {"适中", "一般", "较多"}


def _extract_pois(content: str) -> list[str]:
    """Extract POI names by splitting on connectors first, then regex matching."""
    seen: set[str] = set()
    pois: list[str] = []
    for segment in _CONNECTOR_RE.split(content):
        for match in _POI_PATTERN.findall(segment):
            if match not in seen and not any(c in _CONNECTOR_CHARS for c in match):
                seen.add(match)
                pois.append(match)
    return pois

_RISK_RANK = {"low": 0, "medium": 1, "high": 2}
_RANK_TO_NAME = {0: "low", 1: "medium", 2: "high"}

_ADVICE_BY_RISK = {
    "high": "建议早上9点前到达或考虑改期，避开高峰时段",
    "medium": "预计有一定客流，建议错峰出行",
    "low": "无需特别担心拥挤问题",
}


def assess_crowd_risk(
    destination: str,
    evidence: EvidenceBundle,
    is_weekend_holiday: bool = False,
) -> CrowdRiskAssessment:
    """Assess crowd risk for a destination based on RAG evidence."""

    crowd_results = [
        r for r in evidence.results
        if str(r.metadata.get("section", "")) == "crowd_risk"
    ]

    poi_risks: list[POICrowdRisk] = []
    for result in crowd_results:
        content = result.content
        first_sentence = _first_sentence(content)
        pois = _extract_pois(content)
        risk_level = _score_risk(content)

        if is_weekend_holiday:
            risk_level = _promote(risk_level)

        for poi in pois:
            poi_risks.append(POICrowdRisk(
                poi_name=poi,
                risk_level=risk_level,
                peak_times="weekends and holidays" if is_weekend_holiday else "peak hours",
                source_evidence=first_sentence,
            ))

        if not pois:
            poi_risks.append(POICrowdRisk(
                poi_name=f"{destination} popular areas",
                risk_level=risk_level,
                peak_times="weekends and holidays" if is_weekend_holiday else "peak hours",
                source_evidence=first_sentence,
            ))

    overall_risk = _compute_overall(poi_risks)
    advice = _ADVICE_BY_RISK.get(overall_risk, _ADVICE_BY_RISK["low"])

    return CrowdRiskAssessment(
        destination=destination,
        is_weekend_holiday=is_weekend_holiday,
        poi_risks=poi_risks,
        overall_risk=overall_risk,
        advice=advice,
    )


def _first_sentence(text: str) -> str:
    for sep in ("\n", "。", ".", "！", "!", "？", "?"):
        idx = text.find(sep)
        if 0 < idx < 200:
            return text[:idx].strip()
    return text[:200].strip()


def _score_risk(content: str) -> str:
    if any(kw in content for kw in _HIGH_KEYWORDS):
        return "high"
    if any(kw in content for kw in _MEDIUM_KEYWORDS):
        return "medium"
    return "low"


def _promote(level: str) -> str:
    rank = _RISK_RANK.get(level, 0)
    return _RANK_TO_NAME[min(rank + 1, 2)]


def _compute_overall(poi_risks: list[POICrowdRisk]) -> str:
    if not poi_risks:
        return "low"
    max_rank = max(_RISK_RANK.get(p.risk_level, 0) for p in poi_risks)
    return _RANK_TO_NAME[max_rank]
