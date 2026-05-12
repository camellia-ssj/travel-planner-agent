"""Rule-based nodes for the LangGraph travel agent MVP."""

from __future__ import annotations

import re
import uuid
from contextlib import suppress
from dataclasses import replace
from typing import Protocol

from travel_agent.agent.planner import RuleBasedTravelPlanner, TravelPlanner
from travel_agent.agent.schemas import (
    ReflectionReport,
    TravelPlan,
    TravelRequest,
)
from travel_agent.agent.state import TravelAgentState
from travel_agent.knowledge import (
    CHINESE_DAY_NUMBERS,
    DESTINATION_ALIASES,
    PEOPLE_IMPLICIT,
    WEEKEND_HOLIDAY_PATTERN,
)
from travel_agent.memory.models import TripRecord, UserProfile
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


class MemoryService(Protocol):
    """Minimal service contract needed by memory nodes."""

    def get_profile(self, user_id: str) -> UserProfile: ...

    def save_trip(self, record: TripRecord) -> None: ...


def parse_user_request_node(state: TravelAgentState) -> TravelAgentState:
    """Parse a natural-language request into a structured travel request.

    When the user has a stored profile (from long-term memory), profile
    preferences are used as fallback for fields not explicitly mentioned
    in the query.
    """

    question = state.get("question", "")
    parsed_destination = _parse_destination(question)
    parsed_days = _parse_days(question)
    destination_override = state.get("destination_override")
    days_override = state.get("days_override")
    destination = destination_override if destination_override is not None else parsed_destination
    days = days_override if days_override is not None else parsed_days
    parsed_audience = _parse_audience(question)
    parsed_budget = _parse_budget_preference(question)

    # Use profile as fallback for unspecified fields
    profile = state.get("user_profile")
    audience = _resolve_audience(question, parsed_audience, profile)
    budget_preference = _resolve_budget(question, parsed_budget, profile)

    request = TravelRequest(
        raw_query=question,
        destination=destination,
        days=int(days),
        audience=audience,
        budget_preference=budget_preference,
    )
    return {
        "original_user_request": question,
        "request": request,
        "user_feedback": state.get("user_feedback", []),
    }


_CHANGE_INTENT_RE = re.compile(
    r"(?:"
    r"改(?:成|为|到|去)|换成|换到|变成|转到|调整到|修改到|换成去|"
    r"change\s+(?:the\s+)?(?:trip\s+|plan\s+|destination\s+|days\s+|budget\s+)?to|"
    r"switch\s+(?:the\s+)?(?:trip\s+|plan\s+|destination\s+|days\s+|budget\s+)?to|"
    r"update\s+(?:the\s+)?(?:trip\s+|plan\s+|destination\s+|days\s+|budget\s+)?to|"
    r"move\s+(?:the\s+)?(?:trip\s+|plan\s+|destination\s+)?to"
    r")",
    re.IGNORECASE,
)


def _has_change_intent(text: str) -> bool:
    """Return True when *text* explicitly asks to change a trip parameter.

    Without this guard, incidental matches (e.g. "第二天少走路" where
    "二"+"天" → 2) would incorrectly overwrite the checkpointed request.
    """
    return bool(_CHANGE_INTENT_RE.search(text))


def apply_feedback_node(state: TravelAgentState) -> TravelAgentState:
    """Append follow-up user feedback and re-parse for parameter changes.

    When the user asks to change destination, days, or budget in a resume
    request, this node updates ``request`` and ``question`` so downstream
    nodes (re-retrieval, tools, planner) operate on the new intent instead
    of the stale checkpointed values.

    Parsed values are only applied when the feedback contains explicit
    change-intent language (``改成``, ``改为``, ``换成``, etc.) to avoid
    false positives like ``第二天少走路`` overwriting ``days=3`` with 2.
    """

    feedback = state.get("latest_user_feedback", "").strip()
    existing = list(state.get("user_feedback", []))
    if feedback:
        existing.append(feedback)

    result: dict[str, object] = {"user_feedback": existing}
    if not feedback:
        return result

    old_request = state.get("request")
    if old_request is None:
        return result

    if not _has_change_intent(feedback):
        return result

    new_destination = _parse_destination(feedback)
    new_days = _parse_days(feedback)
    new_budget = _parse_budget_preference(feedback)

    updated_fields: dict[str, object] = {}
    if new_destination and new_destination != old_request.destination:
        updated_fields["destination"] = new_destination
    if _days_explicitly_mentioned(feedback) and new_days >= 1 and new_days != old_request.days:
        updated_fields["days"] = max(1, int(new_days))
    if new_budget and new_budget != old_request.budget_preference:
        updated_fields["budget_preference"] = new_budget

    if updated_fields:
        combined_query = _resume_query_with_feedback(
            state,
            old_request=old_request,
            feedback=feedback,
        )
        updated_fields["raw_query"] = combined_query
        result["request"] = old_request.model_copy(update=updated_fields)
        # Preserve original trip constraints in the regenerated retrieval query
        # so feedback like "change destination to Beijing" does not drop
        # people-count, weekend, or other context from the original request.
        result["question"] = combined_query

    return result


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
            added = 0
            for r in supplementary.results:
                key = _result_dedup_key(r)
                if key not in seen:
                    seen.add(key)
                    merged.append(r)
                    added += 1
            if added:
                evidence = replace(
                    evidence,
                    results=merged,
                    trace=replace(
                        evidence.trace,
                        returned_results=len(merged),
                        reranked_hits=(
                            evidence.trace.reranked_hits
                            + supplementary.trace.reranked_hits
                        ),
                        average_score=_avg_score(merged),
                        empty_result=False,
                        total_latency_ms=round(
                            evidence.trace.total_latency_ms
                            + supplementary.trace.total_latency_ms,
                            3,
                        ),
                    ),
                    confidence=round(
                        max(evidence.confidence, supplementary.confidence), 4
                    ),
                )

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
    user_profile = state.get("user_profile")
    plan = planner.plan(
        request, evidence,
        user_feedback=state.get("user_feedback", []),
        tool_results=tool_results,
        user_profile=user_profile,
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


_DAYS_DIGIT_RE = re.compile(r"(\d+)\s*(?:天|日|days?|d)", re.IGNORECASE)
_DAYS_CHINESE_RE = re.compile(r"([一二两三四五六七八九十])\s*(?:天|日)")


def _days_explicitly_mentioned(text: str) -> bool:
    """Return True when *text* contains an explicit day-count mention.

    Used so ``_parse_days`` defaulting to 1 doesn't trigger spurious
    day-changes when the user didn't actually mention a day count.
    """
    return bool(_DAYS_DIGIT_RE.search(text) or _DAYS_CHINESE_RE.search(text))


def _parse_days(text: str) -> int:
    digit_match = _DAYS_DIGIT_RE.search(text)
    if digit_match:
        return max(1, int(digit_match.group(1)))
    chinese_match = _DAYS_CHINESE_RE.search(text)
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
    if any(
        token in normalized
        for token in ("省钱", "经济", "低预算", "穷游", "cheap", "economy", "budget-friendly")
    ):
        return "economy"
    if any(token in normalized for token in ("舒适", "高端", "豪华", "luxury", "premium")):
        return "premium"
    if any(
        token in normalized
        for token in ("适中", "中等", "standard", "mid", "mid-range", "moderate")
    ):
        return "standard"
    return "standard"


def _detect_weekend_holiday(text: str) -> bool:
    """Return True if the query contains weekend or holiday keywords."""
    return bool(WEEKEND_HOLIDAY_PATTERN.search(text))


_PEOPLE_COUNT_PATTERNS = [
    (re.compile(r"([一-鿿]+)(\d+)\s*个?\s*人"), lambda m: int(m.group(2))),
    (re.compile(r"(\d+)\s*个?\s*人"), lambda m: int(m.group(1))),
    (re.compile(r"我们\s*(\d+)\s*个"), lambda m: int(m.group(1))),
    (re.compile(r"([\d]+)\s*(?:位|名|adults?|people|persons?)"), lambda m: int(m.group(1))),
    (re.compile(r"一家\s*([\d一二两三])\s*口"), lambda m: _cn_digit_to_int(m.group(1))),
    (re.compile(r"([一二两三四五六七八九十])\s*个?\s*人"), lambda m: _cn_digit_to_int(m.group(1))),
]


def _cn_digit_to_int(ch: str) -> int:
    return CHINESE_DAY_NUMBERS.get(ch, 1)


def _parse_people_count(text: str, audience: list[str]) -> int:
    """Extract headcount from query text. Falls back to audience length."""
    if not text:
        return max(1, len(audience))

    for pattern, extractor in _PEOPLE_COUNT_PATTERNS:
        match = pattern.search(text)
        if match:
            return max(1, extractor(match))

    normalized = text.lower()
    for phrase, count in PEOPLE_IMPLICIT.items():
        if phrase in normalized:
            return count

    # Fallback: audience type based — "couple" = 2, else 1 per type
    if "couple" in audience or "情侣" in text or "夫妻" in text:
        return 2
    return max(1, len(audience))


def _resolve_audience(
    question: str, parsed: list[str], profile: UserProfile | None
) -> list[str]:
    """Use profile audience as fallback when no audience is mentioned in query."""
    if parsed and parsed != ["general"]:
        return parsed
    if profile is not None and profile.audience_types:
        return profile.audience_types
    return parsed


def _resolve_budget(
    question: str, parsed: str, profile: UserProfile | None
) -> str:
    """Use profile budget preference as fallback when not mentioned."""
    if _budget_explicitly_mentioned(question):
        return parsed
    if profile is not None and profile.total_trips > 0:
        return profile.budget_preference
    return parsed


def _budget_explicitly_mentioned(text: str) -> bool:
    """Return True when the query explicitly mentions budget preference."""
    normalized = text.lower()
    budget_keywords = [
        "省钱", "经济", "低预算", "穷游", "cheap", "economy", "budget-friendly",
        "舒适", "高端", "豪华", "luxury", "premium",
        "适中", "中等", "standard", "mid", "mid-range", "moderate",
        "预算", "费用", "budget",
    ]
    return any(kw in normalized for kw in budget_keywords)


def _result_dedup_key(result: SearchResult) -> str:
    """Stable dedup key so supplementary retrieval doesn't add duplicates."""
    chunk_id = result.metadata.get("chunk_id")
    if isinstance(chunk_id, str) and chunk_id:
        return chunk_id
    # Fallback: hash the content prefix + source
    content_prefix = result.content[:80]
    return f"{result.source}:{content_prefix}"


def _resume_query_with_feedback(
    state: TravelAgentState,
    old_request: TravelRequest,
    feedback: str,
) -> str:
    original_query = str(
        state.get("original_user_request")
        or old_request.raw_query
        or state.get("question", "")
    ).strip()
    if not original_query:
        return feedback
    return (
        f"{original_query}\n"
        f"Follow-up change request: {feedback}"
    )


def load_user_profile_node(
    state: TravelAgentState,
    memory_service: MemoryService,
) -> TravelAgentState:
    """Load the user profile from long-term memory into graph state.

    No-op when ``user_id`` is not set in state. The profile is later
    passed to the planner so it can personalize recommendations.
    """
    user_id = state.get("user_id", "").strip()
    if not user_id:
        return {}
    try:
        profile = memory_service.get_profile(user_id)
    except Exception:
        return {}
    return {"user_profile": profile}


def save_trip_memory_node(
    state: TravelAgentState,
    memory_service: MemoryService,
) -> TravelAgentState:
    """Persist the completed trip into long-term memory and rebuild the profile.

    No-op when ``user_id`` is not set or no plan exists in state.
    """
    user_id = state.get("user_id", "").strip()
    if not user_id:
        return {}

    request = state.get("request")
    plan: TravelPlan | None = state.get("plan")  # type: ignore[assignment]
    if request is None or plan is None:
        return {}

    thread_id = str(state.get("thread_id", ""))
    memory_id = uuid.uuid4().hex

    record = TripRecord(
        memory_id=memory_id,
        user_id=user_id,
        thread_id=thread_id,
        destination=plan.destination or request.destination,
        days=request.days,
        audience=request.audience,
        budget_preference=request.budget_preference,
        plan_summary=plan.summary,
        user_feedback=list(state.get("user_feedback", [])),
    )
    with suppress(Exception):
        memory_service.save_trip(record)

    # Return the updated profile so downstream can use it
    try:
        updated_profile = memory_service.get_profile(user_id)
        return {"user_profile": updated_profile}
    except Exception:
        return {}


def _avg_score(results: list[SearchResult]) -> float:
    """Average score across a merged result list."""
    if not results:
        return 0.0
    return round(sum(r.score for r in results) / len(results), 4)


# ---------------------------------------------------------------------------
# Reflection / factuality review node
# ---------------------------------------------------------------------------


def reflect_node(
    state: TravelAgentState,
    reflection_service: object | None = None,
) -> TravelAgentState:
    """Review the generated plan against RAG evidence for factual consistency.

    When *reflection_service* (a ``ReflectionService``) is provided, the
    review uses an LLM structured-output call as the primary checker, with
    deterministic text-overlap as fallback.  Without it the review is
    purely deterministic (no API key needed).

    Produces a ``ReflectionReport`` with hallucination flags, evidence
    coverage, confidence score, and actionable suggestions.
    """

    plan = state.get("plan")
    evidence = state.get("evidence")

    if plan is None:
        return {
            "reflection_report": ReflectionReport(
                issues=["No plan to review"],
                passed=False,
            ),
            "reflection_issues": ["No plan to review"],
            "reflection_retry_count": state.get("reflection_retry_count", 0),
        }

    tool_results: dict[str, object] = {
        "tool_budget": state.get("tool_budget"),
        "tool_crowd_risk": state.get("tool_crowd_risk"),
        "tool_alternatives": state.get("tool_alternatives"),
    }

    if reflection_service is not None and hasattr(reflection_service, "reflect"):
        report = reflection_service.reflect(plan, evidence, tool_results)  # type: ignore[union-attr]
    else:
        from travel_agent.agent.reflection import deterministic_reflect

        report = deterministic_reflect(plan, evidence, tool_results)

    retry_count = state.get("reflection_retry_count", 0)
    if not report.passed:
        retry_count += 1
        report.issues.insert(
            0,
            f"Reflection attempt {retry_count}: plan did not pass factuality review.",
        )

    return {
        "reflection_report": report,
        "reflection_issues": report.issues,
        "reflection_retry_count": retry_count,
    }
