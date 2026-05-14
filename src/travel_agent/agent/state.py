"""旅行智能体MVP的LangGraph状态定义。"""

from __future__ import annotations

from typing import TypedDict

from travel_agent.agent.schemas import (
    AlternativePlan,
    BudgetEstimate,
    CrowdRiskAssessment,
    ReflectionReport,
    TravelPlan,
    TravelRequest,
)
from travel_agent.memory.models import UserProfile
from travel_agent.rag.models import EvidenceBundle


class TravelAgentState(TypedDict, total=False):
    """在旅行智能体图节点之间传递的状态。"""

    question: str
    original_user_request: str
    destination_override: str
    days_override: int
    latest_user_feedback: str
    user_feedback: list[str]
    user_id: str
    thread_id: str
    user_profile: UserProfile
    request: TravelRequest
    evidence: EvidenceBundle
    tool_budget: BudgetEstimate
    tool_crowd_risk: CrowdRiskAssessment
    tool_alternatives: AlternativePlan
    plan: TravelPlan
    is_valid: bool
    validation_errors: list[str]
    reflection_report: ReflectionReport
    reflection_issues: list[str]
    reflection_retry_count: int
