"""Structured schemas for the LangGraph travel agent."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TravelRequest(BaseModel):
    """Rule-parsed user travel intent."""

    raw_query: str
    destination: str = ""
    days: int = 1
    audience: list[str] = Field(default_factory=list)
    budget_preference: str = "standard"


class DayPlan(BaseModel):
    """A single day in the generated itinerary."""

    day: int
    title: str
    activities: list[str]
    evidence_sources: list[str] = Field(default_factory=list)


class BudgetItem(BaseModel):
    """Budget guidance for one travel spending category."""

    category: str
    preference: str
    note: str


class RiskNotice(BaseModel):
    """A rule-based risk reminder derived from retrieved evidence."""

    risk_type: str
    message: str
    severity: str = "medium"


class TravelPlan(BaseModel):
    """Structured output produced by the travel agent graph."""

    request: TravelRequest
    destination: str
    days: int
    summary: str
    day_plans: list[DayPlan]
    budget_items: list[BudgetItem]
    risk_notices: list[RiskNotice]
    alternatives: list[str] = Field(default_factory=list)
    evidence_sources: list[str] = Field(default_factory=list)
    evidence_trace_id: str = ""
