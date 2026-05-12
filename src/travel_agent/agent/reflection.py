"""Post-generation factuality review (审校) with LLM + deterministic fallback.

Cross-checks every claim in a TravelPlan against RAG evidence and
deterministic tool results. Uses an LLM structured-output call as the
primary checker, with a deterministic text-overlap fallback when no API
key is configured or the LLM call fails.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from difflib import SequenceMatcher

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from travel_agent.agent.prompts import REFLECTION_SYSTEM_PROMPT, build_reflection_prompt
from travel_agent.agent.schemas import (
    HallucinationFlag,
    ReflectionReport,
    TravelPlan,
)
from travel_agent.knowledge import DESTINATION_ALIASES
from travel_agent.rag.models import EvidenceBundle

# ---------------------------------------------------------------------------
# LLM-backed reflection service
# ---------------------------------------------------------------------------


class ReflectionService:
    """Post-generation fact-checker using LLM structured output.

    Falls back to ``deterministic_reflect()`` when no chat model is
    available or the LLM call raises.
    """

    def __init__(
        self,
        chat_model: BaseChatModel | None = None,
        coverage_threshold: float = 0.5,
    ) -> None:
        self._chat_model = chat_model
        self.coverage_threshold = coverage_threshold

    @property
    def has_llm(self) -> bool:
        return self._chat_model is not None

    def reflect(
        self,
        plan: TravelPlan,
        evidence: EvidenceBundle,
        tool_results: dict[str, object] | None = None,
    ) -> ReflectionReport:
        """Run factuality review and return a structured report."""
        if self._chat_model is not None:
            try:
                return self._llm_reflect(plan, evidence, tool_results)
            except Exception:
                pass
        return deterministic_reflect(plan, evidence, tool_results)

    def _llm_reflect(
        self,
        plan: TravelPlan,
        evidence: EvidenceBundle,
        tool_results: dict[str, object] | None,
    ) -> ReflectionReport:
        """Invoke the LLM with structured output for factuality review."""
        structured_model = self._chat_model.with_structured_output(ReflectionReport)  # type: ignore[union-attr]
        response = structured_model.invoke(
            [
                SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
                HumanMessage(
                    content=build_reflection_prompt(plan, evidence, tool_results)
                ),
            ]
        )
        report = _coerce_reflection_report(response)
        report = _clamp_report_scores(report)
        # Merge deterministic destination-consistency flags — the LLM may miss
        # cross-destination contamination that the alias map catches reliably.
        dest_flags = _check_destination_consistency(plan)
        existing_flags = {_flag_key(flag) for flag in report.hallucination_flags}
        for flag in dest_flags:
            flag_key = _flag_key(flag)
            if flag_key not in existing_flags:
                report.hallucination_flags.append(flag)
                existing_flags.add(flag_key)
        report.checked_claims = max(
            report.checked_claims,
            report.grounded_claims + len(report.hallucination_flags),
        )
        report.passed = (
            len(report.hallucination_flags) == 0
            and report.evidence_coverage >= self.coverage_threshold
        )
        return report


def _flag_key(flag: HallucinationFlag) -> tuple[str, str, str, str]:
    return (
        flag.location,
        flag.claim,
        flag.issue,
        flag.severity,
    )


def _clamp_report_scores(report: ReflectionReport) -> ReflectionReport:
    """Ensure coverage and confidence stay in 0.0-1.0 range."""
    coverage = max(0.0, min(1.0, report.evidence_coverage))
    confidence = max(0.0, min(1.0, report.confidence_score))
    if coverage != report.evidence_coverage or confidence != report.confidence_score:
        return report.model_copy(
            update={"evidence_coverage": coverage, "confidence_score": confidence}
        )
    return report


def _coerce_reflection_report(response: object) -> ReflectionReport:
    if isinstance(response, ReflectionReport):
        return response
    if isinstance(response, dict):
        return ReflectionReport.model_validate(response)
    return ReflectionReport.model_validate(response)


# ---------------------------------------------------------------------------
# Deterministic fallback reflection
# ---------------------------------------------------------------------------


def deterministic_reflect(
    plan: TravelPlan,
    evidence: EvidenceBundle | None,
    tool_results: dict[str, object] | None = None,
) -> ReflectionReport:
    """Deterministic, no-LLM factuality review.

    Uses text-overlap (SequenceMatcher) for activity/evidence matching
    and cross-destination alias checks.  Serves as the fallback when
    no LLM API key is configured.
    """
    if plan is None:
        return ReflectionReport(
            issues=["No plan to review"],
            passed=False,
        )

    evidence_snippets = _collect_evidence_snippets(evidence)
    flags: list[HallucinationFlag] = []
    suggestions: list[str] = []
    checked = 0
    grounded = 0

    # 1. Review day plan activities
    for day_idx, day_plan in enumerate(plan.day_plans):
        for act_idx, activity in enumerate(day_plan.activities):
            checked += 1
            location = f"day_plans[{day_idx}].activities[{act_idx}]"
            overlap = _best_overlap(activity, evidence_snippets)
            if overlap < 0.15:
                severity = "medium" if overlap > 0.05 else "high"
                flags.append(
                    HallucinationFlag(
                        location=location,
                        claim=activity[:200],
                        issue=(
                            "Activity text has low overlap with retrieved "
                            "evidence — may be unsupported."
                        ),
                        severity=severity,
                    )
                )
                if overlap <= 0.05:
                    suggestions.append(
                        f"Replace or ground activity at {location} with evidence-backed content."
                    )
            else:
                grounded += 1

    # 2. Review budget items against tool_budget
    tool_budget = (tool_results or {}).get("tool_budget")
    if tool_budget is not None and plan.budget_items:
        for item_idx, item in enumerate(plan.budget_items):
            checked += 1
            location = f"budget_items[{item_idx}]"
            budget_overlap = _check_budget_item(item, tool_budget, evidence_snippets)
            if budget_overlap < 0.1:
                flags.append(
                    HallucinationFlag(
                        location=location,
                        claim=f"{item.category}: {item.note}",
                        issue="Budget item not grounded in tool results or evidence.",
                        severity="low",
                    )
                )
            else:
                grounded += 1

    # 3. Review risk notices against tool_crowd_risk and evidence
    tool_crowd = (tool_results or {}).get("tool_crowd_risk")
    if plan.risk_notices:
        for notice_idx, notice in enumerate(plan.risk_notices):
            checked += 1
            location = f"risk_notices[{notice_idx}]"
            crowd_overlap = _check_risk_notice(notice, tool_crowd, evidence_snippets)
            if crowd_overlap < 0.1:
                flags.append(
                    HallucinationFlag(
                        location=location,
                        claim=notice.message[:200],
                        issue="Risk notice not supported by tool results or evidence.",
                        severity="medium",
                    )
                )
            else:
                grounded += 1

    # 4. Review alternatives against tool_alternatives and evidence
    tool_alt = (tool_results or {}).get("tool_alternatives")
    if plan.alternatives:
        for alt_idx, alt in enumerate(plan.alternatives):
            checked += 1
            location = f"alternatives[{alt_idx}]"
            alt_overlap = _check_alternative(alt, tool_alt, evidence_snippets)
            if alt_overlap < 0.1:
                flags.append(
                    HallucinationFlag(
                        location=location,
                        claim=alt[:200],
                        issue="Alternative not grounded in tool results or evidence.",
                        severity="low",
                    )
                )
            else:
                grounded += 1

    # 5. Cross-destination check
    dest_issues = _check_destination_consistency(plan)
    for issue in dest_issues:
        flags.append(issue)
        checked += 1

    # 6. Build the report
    evidence_coverage = round(grounded / max(checked, 1), 4)
    high_severity = sum(1 for f in flags if f.severity == "high")
    medium_severity = sum(1 for f in flags if f.severity == "medium")
    confidence_score = round(
        max(0.0, evidence_coverage - 0.15 * high_severity - 0.05 * medium_severity),
        4,
    )

    issues: list[str] = []
    if high_severity > 0:
        issues.append(f"{high_severity} high-severity hallucination flag(s) found.")
    if medium_severity > 0:
        issues.append(f"{medium_severity} medium-severity flag(s) found.")
    if evidence_coverage < 0.5:
        issues.append(
            f"Low evidence coverage ({evidence_coverage:.0%}) — plan may contain fabrications."
        )
        suggestions.append(
            "Consider re-generating with stricter evidence constraints."
        )
    if plan.fallback_used:
        issues.append(
            "Plan was generated by rule-based fallback — review manually."
        )
        suggestions.append(
            "Rule-based plans are templates; verify all POIs and "
            "schedules against current conditions."
        )

    passed = len(flags) == 0 and evidence_coverage >= 0.3

    return ReflectionReport(
        hallucination_flags=flags,
        evidence_coverage=evidence_coverage,
        confidence_score=confidence_score,
        issues=issues,
        suggestions=suggestions[:5],
        passed=passed,
        checked_claims=checked,
        grounded_claims=grounded,
    )


# ---------------------------------------------------------------------------
# Deterministic helper functions (shared with nodes.py)
# ---------------------------------------------------------------------------


def _collect_evidence_snippets(evidence: EvidenceBundle | None) -> list[str]:
    if evidence is None or not evidence.results:
        return []
    snippets: list[str] = []
    for r in evidence.results:
        content = r.content.strip()
        if content:
            snippets.append(content)
        if r.destination:
            snippets.append(r.destination)
    return snippets


def _best_overlap(text: str, snippets: list[str]) -> float:
    if not snippets:
        return 0.0
    best = 0.0
    for snippet in snippets:
        ratio = SequenceMatcher(None, text, snippet).ratio()
        if ratio > best:
            best = ratio
    return best


def _check_budget_item(
    item: object,
    tool_budget: object,
    snippets: list[str],
) -> float:
    text_parts: list[str] = []
    if isinstance(item, dict):
        text_parts.extend([str(item.get("category", "")), str(item.get("note", ""))])
    elif getattr(item, "category", None) is not None:
        text_parts.extend([str(item.category), str(item.note)])  # type: ignore[union-attr]
    else:
        text_parts.append(str(item))
    budget_text = " ".join(text_parts)

    if isinstance(tool_budget, dict):
        tool_text = str(tool_budget.get("budget_level", ""))
    elif getattr(tool_budget, "budget_level", None) is not None:
        tool_text = str(tool_budget.budget_level)  # type: ignore[union-attr]
    else:
        tool_text = str(tool_budget)

    return _best_overlap(budget_text, snippets + [tool_text])


def _check_risk_notice(
    notice: object,
    tool_crowd: object | None,
    snippets: list[str],
) -> float:
    if isinstance(notice, dict):
        notice_text = f"{notice.get('risk_type', '')} {notice.get('message', '')}"
    elif getattr(notice, "message", None) is not None:
        notice_text = f"{notice.risk_type} {notice.message}"  # type: ignore[union-attr]
    else:
        notice_text = str(notice)

    combined = list(snippets)
    if tool_crowd is not None:
        if isinstance(tool_crowd, dict):
            combined.append(str(tool_crowd.get("overall_risk", "")))
        elif getattr(tool_crowd, "overall_risk", None) is not None:
            combined.append(str(tool_crowd.overall_risk))  # type: ignore[union-attr]
    return _best_overlap(notice_text, combined)


def _check_alternative(
    alt_text: str,
    tool_alt: object | None,
    snippets: list[str],
) -> float:
    combined = list(snippets)
    if tool_alt is not None:
        if isinstance(tool_alt, dict):
            combined.append(str(tool_alt.get("weather_note", "")))
        elif getattr(tool_alt, "weather_note", None) is not None:
            combined.append(str(tool_alt.weather_note))  # type: ignore[union-attr]
    return _best_overlap(alt_text, combined)


def _check_destination_consistency(plan: TravelPlan) -> list[HallucinationFlag]:
    """Flag any plan content that references the wrong destination."""
    flags: list[HallucinationFlag] = []
    plan_dest = plan.destination.lower() if plan.destination else ""

    other_dest_aliases: dict[str, list[str]] = {}
    for alias, canonical in DESTINATION_ALIASES.items():
        if canonical.lower() != plan_dest:
            other_dest_aliases.setdefault(canonical, []).append(alias.lower())

    all_activity_text = " ".join(
        activity
        for day_plan in plan.day_plans
        for activity in day_plan.activities
    ).lower()

    for canonical, aliases in other_dest_aliases.items():
        for alias in aliases:
            if (
                len(alias) >= 2
                and alias in all_activity_text
                and (
                    len(alias) >= 3
                    or alias in {
                        "北京",
                        "东京",
                        "巴黎",
                        "成都",
                        "长沙",
                        "大理",
                        "苏州",
                        "杭州",
                    }
                )
            ):
                flags.append(
                    HallucinationFlag(
                        location="day_plans[*].activities[*]",
                        claim=(
                            f"Mentions '{alias}' which belongs to "
                            f"{canonical}, not {plan.destination}"
                        ),
                        issue=(
                            f"Cross-destination contamination: {canonical} "
                            f"POI in {plan.destination} plan."
                        ),
                        severity="medium",
                    )
                )
                break
    return flags


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReflectionSettings:
    llm_provider: str = "qwen"
    model: str = "qwen3-max"
    coverage_threshold: float = 0.5
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> ReflectionSettings:
        return cls(
            llm_provider=os.getenv("TRAVEL_AGENT_LLM_PROVIDER", "qwen").strip().lower(),
            model=os.getenv("TRAVEL_AGENT_MODEL", "qwen3-max").strip() or "qwen3-max",
            coverage_threshold=float(
                os.getenv("TRAVEL_AGENT_REFLECTION_COVERAGE_THRESHOLD", "0.5")
            ),
            max_retries=int(
                os.getenv("TRAVEL_AGENT_REFLECTION_MAX_RETRIES", "2")
            ),
        )


def build_reflection_service(
    settings: ReflectionSettings | None = None,
) -> ReflectionService:
    """Build a ReflectionService with the same LLM config as the planner.

    Returns a service with only deterministic fallback when no API key
    is configured.
    """
    active_settings = settings or ReflectionSettings.from_env()
    chat_model = _build_chat_model(active_settings)
    return ReflectionService(
        chat_model=chat_model,
        coverage_threshold=active_settings.coverage_threshold,
    )


def _build_chat_model(settings: ReflectionSettings) -> BaseChatModel | None:
    from langchain_openai import ChatOpenAI

    provider = settings.llm_provider
    if provider in {"qwen", "dashscope"}:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            return None
        return ChatOpenAI(
            model=settings.model,
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0,
        )
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return ChatOpenAI(model=settings.model, api_key=api_key, temperature=0)
    return None
