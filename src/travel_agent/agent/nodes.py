"""Rule-based nodes for the LangGraph travel agent MVP."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Protocol

from travel_agent.agent.planner import RuleBasedTravelPlanner, TravelPlanner
from travel_agent.agent.schemas import TravelRequest
from travel_agent.agent.state import TravelAgentState
from travel_agent.rag.models import EvidenceBundle, SearchResult
from travel_agent.tools.alternatives import suggest_alternatives
from travel_agent.tools.budget import estimate_budget
from travel_agent.tools.crowd import assess_crowd_risk


class EvidenceService(Protocol):
    """Minimal service contract needed by the agent retrieval node."""

    def retrieve_evidence(
        self,
        query: str,
        top_k: int | None = None,
        destination: str | None = None,
        section: str | None = None,
        travel_type: str | None = None,
        season: str | None = None,
        retrieval_mode: str | None = None,
    ) -> EvidenceBundle:
        """Return structured RAG evidence for the given query."""


DESTINATION_ALIASES = {
    "杭州": "Hangzhou",
    "hangzhou": "Hangzhou",
    "东京": "Tokyo",
    "tokyo": "Tokyo",
    "苏州": "Suzhou",
    "suzhou": "Suzhou",
    "大理": "Dali",
    "dali": "Dali",
    "长沙": "Changsha",
    "changsha": "Changsha",
    "巴黎": "Paris",
    "paris": "Paris",
    "成都": "Chengdu",
    "chengdu": "Chengdu",
    "北京": "Beijing",
    "beijing": "Beijing",
}

CHINESE_DAY_NUMBERS = {
    "一": 1,
    "两": 2,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def parse_user_request_node(state: TravelAgentState) -> TravelAgentState:
    """Parse a natural-language request into a structured travel request."""

    question = state.get("question", "")
    parsed_destination = _parse_destination(question)
    parsed_days = _parse_days(question)
    destination = state.get("destination_override") or parsed_destination
    days = state.get("days_override") or parsed_days
    request = TravelRequest(
        raw_query=question,
        destination=destination,
        days=max(1, int(days)),
        audience=_parse_audience(question),
        budget_preference=_parse_budget_preference(question),
    )
    return {
        "original_user_request": question,
        "request": request,
        "user_feedback": state.get("user_feedback", []),
    }


def apply_feedback_node(state: TravelAgentState) -> TravelAgentState:
    """Append follow-up user feedback to checkpointed state."""

    feedback = state.get("latest_user_feedback", "").strip()
    existing = list(state.get("user_feedback", []))
    if feedback:
        existing.append(feedback)
    return {"user_feedback": existing}


def retrieve_evidence_node(
    state: TravelAgentState,
    rag_service: EvidenceService,
) -> TravelAgentState:
    """Retrieve RAG evidence across multiple sections for diverse day plans."""

    question = state.get("question", "")
    request = state.get("request")
    destination = request.destination if request and request.destination else None

    # Fetch evidence from multiple sections so the planner can vary content
    # across days.  We request itinerary first, then fall back to a broader
    # unfiltered retrieval so budget/crowd-risk mentions in the query don't
    # starve the planner of itinerary content.
    evidence = rag_service.retrieve_evidence(
        question,
        destination=destination,
        top_k=10,
    )
    # If section inference narrowed results too much, supplement with a
    # broader retrieval that skips section filtering.
    itinerary_results = [
        r for r in evidence.results
        if str(r.metadata.get("section", "")) == "itinerary"
    ]
    if len(itinerary_results) < 3:
        supplementary = rag_service.retrieve_evidence(
            question,
            destination=destination,
            section="itinerary",
            top_k=8,
        )
        if supplementary.results:
            merged = list(evidence.results)
            seen = {_result_dedup_key(r) for r in merged}
            for r in supplementary.results:
                key = _result_dedup_key(r)
                if key not in seen:
                    seen.add(key)
                    merged.append(r)
            evidence = replace(evidence, results=merged)

    return {"evidence": evidence}


def tool_node(state: TravelAgentState) -> TravelAgentState:
    """Run deterministic tools: budget, crowd risk, alternatives.

    Uses query text to detect weekend/holiday keywords for crowd risk
    and to extract explicit headcount for budget estimation.
    """

    request = state.get("request")
    evidence = state.get("evidence")
    if request is None or evidence is None:
        return {
            "tool_budget": None,
            "tool_crowd_risk": None,
            "tool_alternatives": None,
        }

    question = state.get("question", request.raw_query if request else "")
    people_count = _parse_people_count(question, request.audience)
    is_weekend_holiday = _detect_weekend_holiday(question)

    budget = estimate_budget(
        people_count=people_count,
        days=request.days,
        budget_level=request.budget_preference,
        evidence=evidence,
    )
    crowd = assess_crowd_risk(
        destination=request.destination,
        evidence=evidence,
        is_weekend_holiday=is_weekend_holiday,
    )
    alternatives = suggest_alternatives(
        destination=request.destination,
        evidence=evidence,
        crowd_assessment=crowd,
    )
    return {
        "tool_budget": budget,
        "tool_crowd_risk": crowd,
        "tool_alternatives": alternatives,
    }


def generate_plan_node(state: TravelAgentState) -> TravelAgentState:
    """Generate a deterministic structured travel plan from request and evidence."""

    return generate_plan_with_planner_node(state, RuleBasedTravelPlanner())


def generate_plan_with_planner_node(
    state: TravelAgentState,
    planner: TravelPlanner,
) -> TravelAgentState:
    """Generate a structured travel plan with an injected planner."""

    request = state.get("request") or TravelRequest(raw_query=state.get("question", ""))
    evidence = state.get("evidence")
    if evidence is None:
        raise ValueError("generate_plan_node requires evidence in state")
    tool_results: dict[str, object] = {
        "tool_budget": state.get("tool_budget"),
        "tool_crowd_risk": state.get("tool_crowd_risk"),
        "tool_alternatives": state.get("tool_alternatives"),
    }
    plan = planner.plan(
        request, evidence,
        user_feedback=state.get("user_feedback", []),
        tool_results=tool_results,
    )
    return {"plan": plan}


def validate_plan_node(state: TravelAgentState) -> TravelAgentState:
    """Validate that the generated plan is complete enough for downstream use."""

    plan = state.get("plan")
    errors: list[str] = []
    if plan is None:
        return {"is_valid": False, "validation_errors": ["missing plan"]}
    if not plan.destination:
        errors.append("missing destination")
    if plan.days <= 0:
        errors.append("days must be positive")
    if len(plan.day_plans) != plan.days:
        errors.append("daily plan count does not match requested days")
    for day_plan in plan.day_plans:
        if not day_plan.activities:
            errors.append(f"day {day_plan.day} has no activities")
    if not plan.risk_notices:
        errors.append("missing risk notices")
    return {"is_valid": not errors, "validation_errors": errors}


def _parse_destination(text: str) -> str:
    normalized = text.lower()
    for alias, destination in DESTINATION_ALIASES.items():
        if alias.lower() in normalized:
            return destination
    return ""


def _parse_days(text: str) -> int:
    digit_match = re.search(r"(\d+)\s*(?:天|日|days?|d)", text, flags=re.IGNORECASE)
    if digit_match:
        return max(1, int(digit_match.group(1)))
    chinese_match = re.search(r"([一二两三四五六七八九十])\s*(?:天|日)", text)
    if chinese_match:
        return CHINESE_DAY_NUMBERS[chinese_match.group(1)]
    return 1


def _parse_audience(text: str) -> list[str]:
    normalized = text.lower()
    audience: list[str] = []
    if any(token in normalized for token in ("亲子", "孩子", "儿童", "family", "kids")):
        audience.append("family_with_children")
    if any(token in normalized for token in ("老人", "父母", "elderly", "senior")):
        audience.append("elderly")
    if any(token in normalized for token in ("情侣", "夫妻", "couple")):
        audience.append("couple")
    if any(token in normalized for token in ("朋友", "同学", "friends")):
        audience.append("friends")
    if any(token in normalized for token in ("独自", "一个人", "solo")):
        audience.append("solo")
    return audience or ["general"]


def _parse_budget_preference(text: str) -> str:
    normalized = text.lower()
    if any(token in normalized for token in ("省钱", "经济", "低预算", "穷游", "cheap", "budget")):
        return "economy"
    if any(token in normalized for token in ("舒适", "高端", "豪华", "luxury", "premium")):
        return "premium"
    if any(token in normalized for token in ("适中", "中等", "standard", "mid")):
        return "standard"
    return "standard"


_WEEKEND_HOLIDAY_KEYWORDS = [
    "周末", "双休", "周六", "周日", "礼拜六", "礼拜天", "星期六", "星期日",
    "周末游", "小长假", "长假", "黄金周", "国庆", "五一", "十一",
    "清明节", "劳动节", "端午节", "中秋节", "元旦", "春节", "端午",
    "清明", "中秋", "寒假", "暑假", "春节假期", "国定假日", "法定假日",
    "节假日", "假期", "节假", "休假", "放假",
    "weekend", "holiday", "vacation", "national day", "golden week",
    "spring festival", "christmas", "new year",
]

_WEEKEND_HOLIDAY_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw in _WEEKEND_HOLIDAY_KEYWORDS),
    flags=re.IGNORECASE,
)


def _detect_weekend_holiday(text: str) -> bool:
    """Return True if the query contains weekend or holiday keywords."""
    return bool(_WEEKEND_HOLIDAY_PATTERN.search(text))


_PEOPLE_COUNT_PATTERNS = [
    (re.compile(r"([一-鿿]+)(\d+)\s*个?\s*人"), lambda m: int(m.group(2))),
    (re.compile(r"(\d+)\s*个?\s*人"), lambda m: int(m.group(1))),
    (re.compile(r"我们\s*(\d+)\s*个"), lambda m: int(m.group(1))),
    (re.compile(r"([\d]+)\s*(?:位|名|adults?|people|persons?)"), lambda m: int(m.group(1))),
    (re.compile(r"一家\s*([\d一二两三])\s*口"), lambda m: _cn_digit_to_int(m.group(1))),
    (re.compile(r"([一二两三四五六七八九十])\s*个?\s*人"), lambda m: _cn_digit_to_int(m.group(1))),
]

_PEOPLE_IMPLICIT: dict[str, int] = {
    "一个人": 1, "独自": 1, "一个人去": 1, "单独": 1, "solo": 1,
    "我和父母": 3, "我和爸妈": 3, "带父母": 3, "带爸妈": 3,
    "我们俩": 2, "两个人": 2, "两人": 2, "二人": 2, "两口子": 2,
    "一家三口": 3, "一家四口": 4, "一家五口": 5,
    "三口之家": 3, "四口之家": 4,
    "亲子": 3, "一家": 3,
}


def _cn_digit_to_int(ch: str) -> int:
    mapping = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
               "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    return mapping.get(ch, 1)


def _parse_people_count(text: str, audience: list[str]) -> int:
    """Extract headcount from query text. Falls back to audience length."""
    if not text:
        return max(1, len(audience))

    for pattern, extractor in _PEOPLE_COUNT_PATTERNS:
        match = pattern.search(text)
        if match:
            return max(1, extractor(match))

    normalized = text.lower()
    for phrase, count in _PEOPLE_IMPLICIT.items():
        if phrase in normalized:
            return count

    # Fallback: audience type based — "couple" = 2, else 1 per type
    if "couple" in audience or "情侣" in text or "夫妻" in text:
        return 2
    return max(1, len(audience))


def _result_dedup_key(result: SearchResult) -> str:
    """Stable dedup key so supplementary retrieval doesn't add duplicates."""
    chunk_id = result.metadata.get("chunk_id")
    if isinstance(chunk_id, str) and chunk_id:
        return chunk_id
    # Fallback: hash the content prefix + source
    content_prefix = result.content[:80]
    return f"{result.source}:{content_prefix}"
