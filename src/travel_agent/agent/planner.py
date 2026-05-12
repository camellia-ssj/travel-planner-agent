"""Planner abstractions for the LangGraph travel agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from travel_agent.agent.prompts import PLANNER_SYSTEM_PROMPT, build_planner_prompt
from travel_agent.agent.schemas import (
    AlternativePlan,
    BudgetEstimate,
    BudgetItem,
    CrowdRiskAssessment,
    DayPlan,
    RiskNotice,
    TravelPlan,
    TravelRequest,
)
from travel_agent.memory.models import UserProfile
from travel_agent.rag.models import EvidenceBundle, SearchResult

DEFAULT_AGENT_MODEL = "qwen3-max"


class TravelPlanner(Protocol):
    """Planner contract used by the agent graph."""

    def plan(
        self,
        request: TravelRequest,
        evidence: EvidenceBundle,
        user_feedback: list[str] | None = None,
        tool_results: dict[str, object] | None = None,
        user_profile: UserProfile | None = None,
    ) -> TravelPlan:
        """Generate a structured travel plan."""


@dataclass
class RuleBasedTravelPlanner:
    """Deterministic fallback planner from the stage 1 MVP."""

    def plan(
        self,
        request: TravelRequest,
        evidence: EvidenceBundle,
        user_feedback: list[str] | None = None,
        tool_results: dict[str, object] | None = None,
        user_profile: UserProfile | None = None,
    ) -> TravelPlan:
        destination = _destination_from(request, evidence)
        results = evidence.results
        evidence_sources = _evidence_sources(results)
        feedback = user_feedback or []
        day_plans = _build_day_plans(destination, request.days, results, feedback, evidence_sources)

        budget_items = _budget_items(request)
        risk_notices = _risk_notices(results)
        alternatives = _alternatives(results, feedback)

        if tool_results:
            if tool_results.get("tool_budget") is not None:
                budget_items = _budget_estimate_to_items(tool_results["tool_budget"])
            if tool_results.get("tool_crowd_risk") is not None:
                risk_notices = _crowd_to_risk_notices(tool_results["tool_crowd_risk"])
            if tool_results.get("tool_alternatives") is not None:
                alternatives = _alternatives_to_strings(tool_results["tool_alternatives"])

        return TravelPlan(
            request=request,
            destination=destination,
            days=request.days,
            summary=_summary(destination, request, feedback, user_profile),
            day_plans=day_plans,
            budget_items=budget_items,
            risk_notices=risk_notices,
            alternatives=alternatives,
            evidence_sources=evidence_sources,
            evidence_trace_id=evidence.trace.trace_id,
        )


@dataclass
class LangChainStructuredPlanner:
    """Planner backed by a LangChain chat model with Pydantic structured output."""

    chat_model: BaseChatModel
    fallback: TravelPlanner | None = None

    def plan(
        self,
        request: TravelRequest,
        evidence: EvidenceBundle,
        user_feedback: list[str] | None = None,
        tool_results: dict[str, object] | None = None,
        user_profile: UserProfile | None = None,
    ) -> TravelPlan:
        try:
            structured_model = self.chat_model.with_structured_output(TravelPlan)
            response = structured_model.invoke(
                [
                    SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                    HumanMessage(
                        content=build_planner_prompt(
                            request,
                            evidence,
                            user_feedback=user_feedback or [],
                            tool_results=tool_results,
                            user_profile=user_profile,
                        )
                    ),
                ]
            )
            plan = _coerce_plan(response)
            plan = _apply_tool_overrides(plan, tool_results)
            return _ensure_evidence_contract(plan, request, evidence)
        except Exception:
            if self.fallback is None:
                raise
            plan = self.fallback.plan(
                request, evidence, user_feedback=user_feedback, tool_results=tool_results,
            )
            plan.fallback_used = True
            return plan


@dataclass(frozen=True)
class AgentPlannerSettings:
    """Runtime settings for the default LLM planner."""

    llm_provider: str = "qwen"
    model: str = DEFAULT_AGENT_MODEL

    @classmethod
    def from_env(cls) -> AgentPlannerSettings:
        return cls(
            llm_provider=os.getenv("TRAVEL_AGENT_LLM_PROVIDER", "qwen").strip().lower(),
            model=os.getenv("TRAVEL_AGENT_MODEL", DEFAULT_AGENT_MODEL).strip()
            or DEFAULT_AGENT_MODEL,
        )


def build_default_planner(settings: AgentPlannerSettings | None = None) -> TravelPlanner:
    """Build the default planner, falling back to rules when no API key is configured."""

    active_settings = settings or AgentPlannerSettings.from_env()
    fallback = RuleBasedTravelPlanner()
    chat_model = _build_chat_model(active_settings)
    if chat_model is None:
        return fallback
    return LangChainStructuredPlanner(chat_model=chat_model, fallback=fallback)


def _build_chat_model(settings: AgentPlannerSettings) -> BaseChatModel | None:
    provider = settings.llm_provider
    if provider in {"qwen", "dashscope"}:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            return None
        return _chat_openai(
            model=settings.model,
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return _chat_openai(model=settings.model, api_key=api_key)
    return None


def _chat_openai(
    model: str,
    api_key: str,
    base_url: str | None = None,
) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, object] = {
        "model": model,
        "api_key": api_key,
        "temperature": 0,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _coerce_plan(response: object) -> TravelPlan:
    if isinstance(response, TravelPlan):
        return response
    if isinstance(response, dict):
        return TravelPlan.model_validate(response)
    return TravelPlan.model_validate(response)


def _ensure_evidence_contract(
    plan: TravelPlan,
    request: TravelRequest,
    evidence: EvidenceBundle,
) -> TravelPlan:
    sources = _evidence_sources(evidence.results)
    destination = request.destination or plan.destination or evidence.query_analysis.get(
        "destination",
        "",
    )
    evidence_sources = [source for source in plan.evidence_sources if source in sources]
    if not evidence_sources:
        evidence_sources = sources
    day_plans = [
        day_plan.model_copy(
            update={
                "evidence_sources": [
                    source for source in day_plan.evidence_sources if source in sources
                ]
                or evidence_sources
            }
        )
        for day_plan in plan.day_plans
    ]
    return plan.model_copy(
        update={
            "request": request,
            "destination": destination,
            "days": request.days,
            "day_plans": day_plans,
            "evidence_sources": evidence_sources,
            "evidence_trace_id": evidence.trace.trace_id,
        }
    )


def _destination_from(request: TravelRequest, evidence: EvidenceBundle) -> str:
    if request.destination:
        return request.destination
    if evidence.query_analysis.get("destination"):
        return evidence.query_analysis["destination"]
    if evidence.results:
        return evidence.results[0].destination
    return ""


def _summary(
    destination: str,
    request: TravelRequest,
    user_feedback: list[str],
    user_profile: UserProfile | None = None,
) -> str:
    audience = ", ".join(request.audience)
    summary = (
        f"{destination} {request.days}-day rule-based plan for {audience} "
        f"with {request.budget_preference} budget preference."
    )
    if user_profile is not None and user_profile.total_trips > 0:
        summary += f" (returning user, {user_profile.total_trips} previous trips)"
    if user_feedback:
        summary += f" Updated with feedback: {user_feedback[-1]}"
    return summary


_DAY_TITLES: dict[int, str] = {
    1: "Arrival & Exploration",
    2: "Deep Dive & Discovery",
    3: "Cultural Highlights",
    4: "Hidden Gems",
    5: "Leisure & Local Life",
    6: "Scenic Routes",
    7: "Farewell & Favorites",
}


_DAY_TIME_SLOTS: dict[int, list[str]] = {
    1: ["上午: 抵达后安顿", "下午: 轻松漫步", "傍晚: 初探当地美食"],
    2: ["上午: 深度探索", "下午: 文化体验", "傍晚: 特色街区"],
    3: ["上午: 经典景点", "下午: 小众秘境", "傍晚: 夜生活"],
    4: ["上午: 周边探索", "下午: 户外活动", "傍晚: 市集闲逛"],
    5: ["上午: 慢生活体验", "下午: 手作或茶歇", "傍晚: 江/湖边散步"],
    6: ["上午: 周边一日", "下午: 自然风光", "傍晚: 夜景打卡"],
    7: ["上午: 自由安排", "下午: 购物或纪念品", "傍晚: 告别晚餐"],
}


def _build_day_plans(
    destination: str,
    days: int,
    results: list[SearchResult],
    feedback: list[str],
    evidence_sources: list[str],
) -> list[DayPlan]:
    """Build day plans with meaningful variation across days.

    Evidence content is broken into sentences and distributed across days
    using stride-based allocation so every day sees different material.
    When evidence is sparse, synthetic day-themed slots fill the gaps.
    """
    sentences = _evidence_sentences(results)
    day_plans: list[DayPlan] = []

    for day in range(1, days + 1):
        title = f"{destination} Day {day} — {_DAY_TITLES.get(day, f'Day {day}')}"
        activities = _activities_for_day_v3(
            day=day,
            days=days,
            destination=destination,
            sentences=sentences,
            user_feedback=feedback,
        )
        day_plans.append(DayPlan(
            day=day,
            title=title,
            activities=activities,
            evidence_sources=evidence_sources,
        ))

    return day_plans


def _evidence_sentences(results: list[SearchResult]) -> list[str]:
    """Break evidence results into individual sentences, preserving order."""
    sentences: list[str] = []
    for r in results:
        text = r.content.strip()
        # Split on Chinese/English sentence boundaries
        parts = _split_sentences(text)
        for part in parts:
            clean = part.strip()
            if len(clean) >= 4:
                sentences.append(clean)
    return sentences


def _split_sentences(text: str) -> list[str]:
    """Split text on common sentence delimiters."""
    result: list[str] = []
    current: list[str] = []
    for ch in text:
        current.append(ch)
        if ch in "。！？!?.\n":
            s = "".join(current).strip()
            if s:
                result.append(s)
            current = []
    tail = "".join(current).strip()
    if tail:
        result.append(tail)
    return result


def _activities_for_day_v3(
    day: int,
    days: int,
    destination: str,
    sentences: list[str],
    user_feedback: list[str],
) -> list[str]:
    """Build day-specific activities by distributing evidence sentences
    across days with stride-based allocation so every day gets different
    material, even when the evidence pool is small."""

    activities: list[str] = []
    slots = _DAY_TIME_SLOTS.get(day, [
        f"上午: {destination} Day {day}",
        f"下午: {destination} Day {day}",
        f"傍晚: {destination} Day {day}",
    ])

    # Assign evidence sentences to this day via stride distribution.
    # Day 1 gets sentences[0], sentences[days], sentences[2*days], ...
    # Day 2 gets sentences[1], sentences[1+days], sentences[1+2*days], ...
    assigned: list[str] = []
    for i in range(day - 1, len(sentences), days):
        assigned.append(sentences[i])

    # If we got nothing (sparse evidence), take a window per day
    if not assigned and sentences:
        window = max(1, len(sentences) // max(days, 1))
        start = (day - 1) * window % len(sentences)
        assigned = sentences[start:start + window]

    # Build activities by pairing time slots with evidence — one evidence
    # sentence per slot, no cycling.
    for idx, slot in enumerate(slots):
        if idx < len(assigned):
            activities.append(f"{slot} — {assigned[idx]}")
        elif not assigned:
            activities.append(f"{slot} — 探索{destination}周边")
            break  # no evidence at all → one synthetic slot is enough

    # If we have more evidence than slots, append the extras
    if len(assigned) > len(slots):
        for extra in assigned[len(slots):]:
            activities.append(f"推荐: {extra}")

    # Always ensure at least 2 activities
    if len(activities) < 2:
        activities.append(f"自由探索: 根据天气调整{destination}行程")

    if user_feedback:
        activities.append(f"根据反馈调整: {user_feedback[-1]}")

    return activities


def _budget_items(request: TravelRequest) -> list[BudgetItem]:
    preference = request.budget_preference
    return [
        BudgetItem(
            category="transport",
            preference=preference,
            note="Prefer routes that match the requested budget level.",
        ),
        BudgetItem(
            category="dining",
            preference=preference,
            note="Reserve daily meal choices around the budget preference.",
        ),
        BudgetItem(
            category="tickets",
            preference=preference,
            note="Leave room for attraction tickets and booking changes.",
        ),
    ]


def _risk_notices(results: list[SearchResult]) -> list[RiskNotice]:
    notices: list[RiskNotice] = []
    risk_sections = {"crowd_risk", "weather_risk", "risk", "alternatives"}
    for result in results:
        section = str(result.metadata.get("section", "risk"))
        if section not in risk_sections:
            continue
        preview = " ".join(result.content.split())[:180]
        notices.append(RiskNotice(risk_type=section, message=preview))
    if notices:
        return notices[:3]
    return [
        RiskNotice(
            risk_type="general",
            message="Review crowd, weather and fallback options before departure.",
        )
    ]


def _alternatives(results: list[SearchResult], user_feedback: list[str]) -> list[str]:
    alternatives = [
        " ".join(result.content.split())[:180]
        for result in results
        if str(result.metadata.get("section", "")) == "alternatives"
    ]
    if user_feedback:
        alternatives.append(f"Feedback-aware backup: {user_feedback[-1]}")
    return alternatives[:3] or ["Keep one lower-crowd indoor or nearby backup option available."]


def _evidence_sources(results: list[SearchResult]) -> list[str]:
    sources: list[str] = []
    for result in results:
        if result.source and result.source not in sources:
            sources.append(result.source)
    return sources


# ---------------------------------------------------------------------------
# Tool result -> TravelPlan field conversion helpers
# ---------------------------------------------------------------------------


def _budget_estimate_to_items(estimate: BudgetEstimate) -> list[BudgetItem]:
    """Convert a deterministic BudgetEstimate into BudgetItem list for TravelPlan."""
    return [
        BudgetItem(
            category="accommodation",
            preference=estimate.budget_level,
            note=f"Estimated {estimate.accommodation:.0f} CNY total",
        ),
        BudgetItem(
            category="dining",
            preference=estimate.budget_level,
            note=f"Estimated {estimate.dining:.0f} CNY total",
        ),
        BudgetItem(
            category="transport",
            preference=estimate.budget_level,
            note=f"Estimated {estimate.transport:.0f} CNY total",
        ),
        BudgetItem(
            category="tickets",
            preference=estimate.budget_level,
            note=f"Estimated {estimate.tickets:.0f} CNY total",
        ),
        BudgetItem(
            category="total",
            preference=estimate.budget_level,
            note=f"Total {estimate.total:.0f} CNY, daily average {estimate.daily_average:.0f} CNY",
        ),
    ]


def _crowd_to_risk_notices(assessment: CrowdRiskAssessment) -> list[RiskNotice]:
    """Convert a CrowdRiskAssessment into RiskNotice list for TravelPlan."""
    notices: list[RiskNotice] = []
    for poi in assessment.poi_risks:
        notices.append(RiskNotice(
            risk_type="crowd_risk",
            message=f"{poi.poi_name}: {poi.risk_level} -- {poi.peak_times}",
            severity=poi.risk_level,
        ))
    if assessment.overall_risk == "high":
        notices.insert(0, RiskNotice(
            risk_type="crowd_risk",
            message=f"{assessment.destination} overall crowd risk is high. {assessment.advice}",
            severity="high",
        ))
    return notices or [RiskNotice(
        risk_type="crowd_risk",
        message=f"{assessment.destination} no significant crowd concerns",
        severity="low",
    )]


def _alternatives_to_strings(plan: AlternativePlan) -> list[str]:
    """Convert an AlternativePlan into string list for TravelPlan.alternatives."""
    result: list[str] = []
    for alt in plan.alternatives:
        result.append(f"{alt.original_scenario} -> {alt.suggested_alternative} ({alt.reason})")
    return result or ([plan.weather_note] if plan.weather_note else ["暂无备选方案"])


def _apply_tool_overrides(
    plan: TravelPlan,
    tool_results: dict[str, object] | None,
) -> TravelPlan:
    """Post-LLM override: force tool results onto plan fields."""
    if not tool_results:
        return plan
    updates: dict[str, object] = {}
    if tool_results.get("tool_budget") is not None:
        updates["budget_items"] = _budget_estimate_to_items(tool_results["tool_budget"])
    if tool_results.get("tool_crowd_risk") is not None:
        updates["risk_notices"] = _crowd_to_risk_notices(tool_results["tool_crowd_risk"])
    if tool_results.get("tool_alternatives") is not None:
        updates["alternatives"] = _alternatives_to_strings(tool_results["tool_alternatives"])
    return plan.model_copy(update=updates) if updates else plan
