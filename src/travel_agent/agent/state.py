"""LangGraph state definitions for the travel agent MVP."""

from __future__ import annotations

from typing import TypedDict

from travel_agent.agent.schemas import (
    AlternativePlan,
    BudgetEstimate,
    CrowdRiskAssessment,
    TravelPlan,
    TravelRequest,
)
from travel_agent.rag.models import EvidenceBundle


class TravelAgentState(TypedDict, total=False):
    """State passed between travel agent graph nodes."""

    question: str
    original_user_request: str
    destination_override: str
    days_override: int
    latest_user_feedback: str
    user_feedback: list[str]
    request: TravelRequest
    evidence: EvidenceBundle
    tool_budget: BudgetEstimate
    tool_crowd_risk: CrowdRiskAssessment
    tool_alternatives: AlternativePlan
    plan: TravelPlan
    is_valid: bool
    validation_errors: list[str]
