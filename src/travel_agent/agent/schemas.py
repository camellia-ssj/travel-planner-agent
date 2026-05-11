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


class BudgetEstimate(BaseModel):
    """Deterministic per-category budget breakdown from budget_tool."""

    accommodation: float = Field(description="Total estimated accommodation cost in CNY")
    dining: float = Field(description="Total estimated dining cost in CNY")
    transport: float = Field(description="Total estimated local transport cost in CNY")
    tickets: float = Field(description="Total estimated attraction tickets cost in CNY")
    total: float = Field(description="Sum of all categories")
    daily_average: float = Field(description="Average daily spend")
    budget_level: str = Field(description="economy / standard / premium")
    notes: list[str] = Field(default_factory=list)


class POICrowdRisk(BaseModel):
    """Crowd risk assessment for a single point of interest."""

    poi_name: str
    risk_level: str = Field(description="low / medium / high")
    peak_times: str = Field(description="When crowds peak")
    source_evidence: str = Field(description="Snippet from RAG evidence")


class CrowdRiskAssessment(BaseModel):
    """Deterministic crowd risk assessment for the trip."""

    destination: str
    is_weekend_holiday: bool
    poi_risks: list[POICrowdRisk] = Field(default_factory=list)
    overall_risk: str = Field(description="low / medium / high")
    advice: str = Field(description="One-line actionable advice")


class AlternativeSuggestion(BaseModel):
    """A single alternative recommendation."""

    original_scenario: str = Field(description="What situation triggers this alternative")
    suggested_alternative: str
    reason: str


class AlternativePlan(BaseModel):
    """Deterministic alternative suggestions based on risks and weather."""

    destination: str
    alternatives: list[AlternativeSuggestion] = Field(default_factory=list)
    weather_note: str = Field(default="", description="Weather-based recommendation summary")


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
