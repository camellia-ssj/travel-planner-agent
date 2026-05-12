"""Pydantic models for Memory long-term user profiles and trip records."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class TripRecord(BaseModel):
    """A single trip planning session stored in long-term memory."""

    memory_id: str
    user_id: str
    thread_id: str = ""
    destination: str
    days: int
    audience: list[str] = Field(default_factory=list)
    budget_preference: str = "standard"
    plan_summary: str = ""
    user_feedback: list[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


class UserProfile(BaseModel):
    """Learned user profile aggregated from trip history.

    This profile serves as the Agent's "user portrait" — it captures
    travel preferences, patterns, and style so the planner can
    personalize recommendations across sessions.
    """

    user_id: str
    preferred_destinations: list[str] = Field(default_factory=list)
    budget_preference: str = "standard"
    audience_types: list[str] = Field(default_factory=list)
    trip_length_avg: float = 0.0
    total_trips: int = 0
    last_destination: str = ""
    preferences_summary: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    def to_context_text(self) -> str:
        """Render the profile as natural-language context for the planner."""
        if self.total_trips == 0:
            return ""
        parts: list[str] = [f"用户画像 (基于 {self.total_trips} 次历史行程):"]
        if self.preferred_destinations:
            parts.append(
                f"- 常去目的地: {', '.join(self.preferred_destinations[:5])}"
            )
        if self.audience_types:
            parts.append(f"- 出行方式: {', '.join(self.audience_types)}")
        if self.budget_preference != "standard":
            budget_labels = {"economy": "经济实惠", "premium": "高端舒适", "standard": "中等标准"}
            label = budget_labels.get(self.budget_preference, self.budget_preference)
            parts.append(f"- 预算偏好: {label}")
        if self.trip_length_avg > 0:
            parts.append(f"- 平均行程天数: {self.trip_length_avg:.1f} 天")
        if self.preferences_summary:
            parts.append(f"- 偏好总结: {self.preferences_summary}")
        return "\n".join(parts)
