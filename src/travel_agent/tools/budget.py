"""Deterministic budget estimation tool."""

from __future__ import annotations

import re

from travel_agent.agent.schemas import BudgetEstimate
from travel_agent.rag.models import EvidenceBundle

# Per-person-per-day base cost in CNY by budget level
_BASE_COST_PER_PERSON_PER_DAY: dict[str, float] = {
    "economy": 300,
    "standard": 600,
    "premium": 1200,
}

# Category allocation percentages
_CATEGORY_WEIGHTS: dict[str, float] = {
    "accommodation": 0.40,
    "dining": 0.25,
    "transport": 0.15,
    "tickets": 0.20,
}

_PRICE_PATTERN = re.compile(r"(\d[\d,]*)\s*(?:元|¥|CNY|/人)")


def estimate_budget(
    people_count: int,
    days: int,
    budget_level: str,
    evidence: EvidenceBundle,
) -> BudgetEstimate:
    """Compute a deterministic per-category budget estimate."""

    effective_days = max(days, 1)
    effective_people = max(people_count, 1)
    normalized_level = budget_level.strip().lower() if budget_level else "standard"
    base = _BASE_COST_PER_PERSON_PER_DAY.get(normalized_level, 600)
    if normalized_level not in _BASE_COST_PER_PERSON_PER_DAY:
        normalized_level = "standard"

    notes: list[str] = [
        f"Base cost: {base} CNY/person/day for {normalized_level} level",
        f"People: {effective_people}, Days: {effective_days}",
    ]

    # Extract price hints from budget-section evidence and nudge base rate
    price_hints = _extract_price_hints(evidence)
    if price_hints:
        avg_hint = sum(price_hints) / len(price_hints)
        # Nudge: if evidence average is far from base, adjust ±15%
        if avg_hint > base * 1.3:
            base *= 1.15
            notes.append(f"Evidence price hints ({avg_hint:.0f} CNY)高于基准, adjusted +15%")
        elif avg_hint < base * 0.7:
            base *= 0.85
            notes.append(f"Evidence price hints ({avg_hint:.0f} CNY)低于基准, adjusted -15%")

    total_per_person_per_day = base
    accommodation = (
        total_per_person_per_day
        * _CATEGORY_WEIGHTS["accommodation"]
        * effective_people
        * effective_days
    )
    dining = (
        total_per_person_per_day
        * _CATEGORY_WEIGHTS["dining"]
        * effective_people
        * effective_days
    )
    transport = (
        total_per_person_per_day
        * _CATEGORY_WEIGHTS["transport"]
        * effective_people
        * effective_days
    )
    tickets = (
        total_per_person_per_day
        * _CATEGORY_WEIGHTS["tickets"]
        * effective_people
        * effective_days
    )
    total = accommodation + dining + transport + tickets
    daily_average = total / effective_days

    return BudgetEstimate(
        accommodation=round(accommodation, 2),
        dining=round(dining, 2),
        transport=round(transport, 2),
        tickets=round(tickets, 2),
        total=round(total, 2),
        daily_average=round(daily_average, 2),
        budget_level=normalized_level,
        notes=notes,
    )


def _extract_price_hints(evidence: EvidenceBundle) -> list[float]:
    """Extract numeric price values from budget-section evidence."""
    hints: list[float] = []
    for result in evidence.results:
        section = str(result.metadata.get("section", ""))
        if section != "budget":
            continue
        for match in _PRICE_PATTERN.finditer(result.content):
            raw = match.group(1).replace(",", "")
            try:
                hints.append(float(raw))
            except ValueError:
                continue
    return hints
