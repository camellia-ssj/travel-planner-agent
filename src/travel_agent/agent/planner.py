"""Planner abstractions for the LangGraph travel agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from travel_agent.agent.prompts import PLANNER_SYSTEM_PROMPT, build_planner_prompt
from travel_agent.agent.schemas import BudgetItem, DayPlan, RiskNotice, TravelPlan, TravelRequest
from travel_agent.rag.models import EvidenceBundle, SearchResult

DEFAULT_AGENT_MODEL = "qwen3-max"


class TravelPlanner(Protocol):
    """Planner contract used by the agent graph."""

    def plan(
        self,
        request: TravelRequest,
        evidence: EvidenceBundle,
        user_feedback: list[str] | None = None,
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
    ) -> TravelPlan:
        destination = _destination_from(request, evidence)
        results = evidence.results
        evidence_sources = _evidence_sources(results)
        feedback = user_feedback or []
        day_plans = [
            DayPlan(
                day=day,
                title=f"{destination} Day {day}",
                activities=_activities_for_day(day, results, feedback),
                evidence_sources=evidence_sources,
            )
            for day in range(1, request.days + 1)
        ]
        return TravelPlan(
            request=request,
            destination=destination,
            days=request.days,
            summary=_summary(destination, request, feedback),
            day_plans=day_plans,
            budget_items=_budget_items(request),
            risk_notices=_risk_notices(results),
            alternatives=_alternatives(results, feedback),
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
                        )
                    ),
                ]
            )
            plan = _coerce_plan(response)
            return _ensure_evidence_contract(plan, request, evidence)
        except Exception:
            if self.fallback is None:
                raise
            return self.fallback.plan(request, evidence, user_feedback=user_feedback)


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


def _summary(destination: str, request: TravelRequest, user_feedback: list[str]) -> str:
    audience = ", ".join(request.audience)
    summary = (
        f"{destination} {request.days}-day rule-based plan for {audience} "
        f"with {request.budget_preference} budget preference."
    )
    if user_feedback:
        summary += f" Updated with feedback: {user_feedback[-1]}"
    return summary


def _activities_for_day(
    day: int,
    results: list[SearchResult],
    user_feedback: list[str],
) -> list[str]:
    if not results:
        activities = ["Review destination constraints and keep the schedule flexible."]
    else:
        selected = results[(day - 1) % len(results)]
        section = str(selected.metadata.get("section", "overview"))
        preview = " ".join(selected.content.split())[:160]
        activities = [f"Use {section} evidence: {preview}"]
    if user_feedback:
        activities.append(f"Adjust for user feedback: {user_feedback[-1]}")
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
