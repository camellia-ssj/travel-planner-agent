"""Rule-based nodes for the LangGraph travel agent MVP."""

from __future__ import annotations

import re
from typing import Protocol

from travel_agent.agent.planner import RuleBasedTravelPlanner, TravelPlanner
from travel_agent.agent.schemas import TravelRequest
from travel_agent.agent.state import TravelAgentState
from travel_agent.rag.models import EvidenceBundle


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
    """Retrieve RAG evidence using the existing pure RAG service contract."""

    question = state.get("question", "")
    request = state.get("request")
    evidence = rag_service.retrieve_evidence(
        question,
        destination=request.destination if request and request.destination else None,
    )
    return {"evidence": evidence}


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
    plan = planner.plan(request, evidence, user_feedback=state.get("user_feedback", []))
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
