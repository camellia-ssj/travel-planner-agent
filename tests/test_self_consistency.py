"""Self-Consistency 多路径投票单元测试。

使用 RuleBasedTravelPlanner 作为 inner planner（确定性，无需 API key），
测试候选生成、评分、择优和降级逻辑。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from travel_agent.agent.planner import RuleBasedTravelPlanner, TravelPlanner
from travel_agent.agent.reflection import (
    ReflectionReport,
    ReflectionService,
    deterministic_reflect,
)
from travel_agent.agent.schemas import (
    DayPlan,
    HallucinationFlag,
    TravelPlan,
    TravelRequest,
)
from travel_agent.agent.self_consistency import (
    SelfConsistencyPlanner,
    SelfConsistencySettings,
    _plan_fingerprint,
)
from travel_agent.rag.models import EvidenceBundle, RetrievalTrace, SearchResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_evidence(destination: str = "Hangzhou") -> EvidenceBundle:
    results = [
        SearchResult(
            content=f"{destination} West Lake is a scenic freshwater lake ideal for walking.",
            source=f"{destination.lower()}.md",
            destination=destination,
            score=0.92,
            metadata={"section": "itinerary"},
        ),
        SearchResult(
            content=f"{destination} Lingyin Temple features Buddhist carvings and shaded paths.",
            source=f"{destination.lower()}.md",
            destination=destination,
            score=0.89,
            metadata={"section": "itinerary"},
        ),
        SearchResult(
            content=f"{destination} Yunnan-Guizhou Plateau has unique geological formations.",
            source=f"{destination.lower()}.md",
            destination=destination,
            score=0.85,
            metadata={"section": "itinerary"},
        ),
    ]
    return EvidenceBundle(
        question=f"{destination} trip plan",
        results=results,
        trace=RetrievalTrace.create(
            retrieval_mode="mock",
            requested_top_k=3,
            candidate_k=3,
            returned_results=3,
            empty_result=False,
            destination=destination,
            section="",
            travel_type="",
            season="",
            embedding_provider="mock",
            reranker="keyword",
            collection_version="test",
            metadata_filters={},
            vector_hits=[],
            keyword_hits=[],
            fused_hits=[],
            reranked_hits=[],
        ),
        query_analysis={"destination": destination},
        confidence=0.9,
    )


def _make_request(destination: str = "Hangzhou", days: int = 3) -> TravelRequest:
    return TravelRequest(
        raw_query=f"{destination} {days}日游",
        destination=destination,
        days=days,
        audience=["general"],
        budget_preference="standard",
    )


# ---------------------------------------------------------------------------
# Plan fingerprint
# ---------------------------------------------------------------------------


class TestPlanFingerprint:
    def test_same_plan_same_fingerprint(self):
        plan1 = TravelPlan(
            request=_make_request(),
            destination="Hangzhou",
            days=3,
            summary="Same plan.",
            day_plans=[
                DayPlan(day=1, title="Day 1", activities=["Activity A", "Activity B"])
            ],
            budget_items=[],
            risk_notices=[],
            alternatives=[],
            evidence_sources=[],
            evidence_trace_id="",
        )
        plan2 = TravelPlan(
            request=_make_request(),
            destination="Hangzhou",
            days=3,
            summary="Same plan.",
            day_plans=[
                DayPlan(day=1, title="Day 1", activities=["Activity A", "Activity B"])
            ],
            budget_items=[],
            risk_notices=[],
            alternatives=[],
            evidence_sources=[],
            evidence_trace_id="",
        )
        assert _plan_fingerprint(plan1) == _plan_fingerprint(plan2)

    def test_different_activities_different_fingerprint(self):
        plan1 = TravelPlan(
            request=_make_request(),
            destination="Hangzhou",
            days=3,
            summary="Plan A.",
            day_plans=[
                DayPlan(day=1, title="Day 1", activities=["Activity A"])
            ],
            budget_items=[],
            risk_notices=[],
            alternatives=[],
            evidence_sources=[],
            evidence_trace_id="",
        )
        plan2 = TravelPlan(
            request=_make_request(),
            destination="Hangzhou",
            days=3,
            summary="Plan B.",
            day_plans=[
                DayPlan(day=1, title="Day 1", activities=["Activity B"])
            ],
            budget_items=[],
            risk_notices=[],
            alternatives=[],
            evidence_sources=[],
            evidence_trace_id="",
        )
        assert _plan_fingerprint(plan1) != _plan_fingerprint(plan2)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestSelfConsistencySettings:
    def test_default_settings(self):
        settings = SelfConsistencySettings()
        assert settings.num_candidates == 3
        assert settings.selection_criterion == "confidence_score"
        assert settings.enabled is True

    def test_disabled_settings(self):
        settings = SelfConsistencySettings(enabled=False, num_candidates=5)
        assert settings.enabled is False


# ---------------------------------------------------------------------------
# SelfConsistencyPlanner
# ---------------------------------------------------------------------------


class TestSelfConsistencyPlanner:
    def test_disabled_passes_through_directly(self):
        """禁用时直接透传 inner planner，不做多路生成。"""
        request = _make_request()
        evidence = _make_evidence()

        inner = RuleBasedTravelPlanner()
        settings = SelfConsistencySettings(enabled=False, num_candidates=3)
        sc_planner = SelfConsistencyPlanner(
            inner_planner=inner, settings=settings
        )

        result = sc_planner.plan(request, evidence)
        assert result.destination == "Hangzhou"
        assert result.days == 3
        # 禁用时不带 SC 标记
        assert "[SC:" not in result.summary

    def test_single_candidate_no_voting(self):
        """候选数=1 时直接返回，不加标记。"""
        request = _make_request()
        evidence = _make_evidence()

        inner = RuleBasedTravelPlanner()
        settings = SelfConsistencySettings(num_candidates=1)
        sc_planner = SelfConsistencyPlanner(
            inner_planner=inner, settings=settings
        )

        result = sc_planner.plan(request, evidence)
        assert result.destination == "Hangzhou"
        assert "[SC:" not in result.summary

    def test_generates_multiple_candidates_with_summary_tag(self):
        """多候选时摘要包含 SC 标记。"""
        request = _make_request()
        evidence = _make_evidence()

        inner = RuleBasedTravelPlanner()
        settings = SelfConsistencySettings(num_candidates=3)
        sc_planner = SelfConsistencyPlanner(
            inner_planner=inner, settings=settings
        )

        result = sc_planner.plan(request, evidence)
        assert result.destination == "Hangzhou"
        # RuleBasedTravelPlanner 是确定性的 — 3 个候选去重后仅 1 个
        # SC 标记格式: [SC:N/M candidates, best=...score...]
        assert "[SC:" in result.summary
        assert "candidates" in result.summary

    def test_selects_by_confidence_score_default(self):
        """默认按 confidence_score 择优。使用 mock 生成不同候选。"""
        request = _make_request()
        evidence = _make_evidence()

        # 创建产生不同计划的 mock inner planner
        inner = MagicMock(spec=TravelPlanner)
        inner.plan.side_effect = [
            # 候选 1 — 中等覆盖率
            TravelPlan(
                request=request,
                destination="Hangzhou",
                days=3,
                summary="Plan A",
                day_plans=[
                    DayPlan(day=1, title="Day 1", activities=[
                        "Hangzhou West Lake is a scenic freshwater lake ideal for walking."
                    ], evidence_sources=["hangzhou.md"]),
                ],
                budget_items=[], risk_notices=[], alternatives=[],
                evidence_sources=["hangzhou.md"],
                evidence_trace_id=evidence.trace.trace_id,
            ),
            # 候选 2 — 更好覆盖率（同时匹配两条证据）
            TravelPlan(
                request=request,
                destination="Hangzhou",
                days=3,
                summary="Plan B",
                day_plans=[
                    DayPlan(day=1, title="Day 1", activities=[
                        "Hangzhou West Lake is a scenic freshwater lake ideal for walking.",
                        "Hangzhou Lingyin Temple features Buddhist carvings and shaded paths.",
                    ], evidence_sources=["hangzhou.md"]),
                ],
                budget_items=[], risk_notices=[], alternatives=[],
                evidence_sources=["hangzhou.md"],
                evidence_trace_id=evidence.trace.trace_id,
            ),
        ]
        settings = SelfConsistencySettings(
            num_candidates=2, selection_criterion="confidence_score"
        )
        sc_planner = SelfConsistencyPlanner(
            inner_planner=inner, settings=settings
        )

        result = sc_planner.plan(request, evidence)
        assert result.destination == "Hangzhou"
        assert "confidence_score" in result.summary

    def test_selects_by_evidence_coverage(self):
        """按 evidence_coverage 择优。使用 mock 生成不同候选。"""
        request = _make_request()
        evidence = _make_evidence()

        inner = MagicMock(spec=TravelPlanner)
        inner.plan.side_effect = [
            TravelPlan(
                request=request,
                destination="Hangzhou",
                days=3,
                summary="Plan A",
                day_plans=[
                    DayPlan(day=1, title="Day 1", activities=[
                        "Hangzhou West Lake is a scenic freshwater lake ideal for walking."
                    ], evidence_sources=["hangzhou.md"]),
                ],
                budget_items=[], risk_notices=[], alternatives=[],
                evidence_sources=["hangzhou.md"],
                evidence_trace_id=evidence.trace.trace_id,
            ),
            TravelPlan(
                request=request,
                destination="Hangzhou",
                days=3,
                summary="Plan B",
                day_plans=[
                    DayPlan(day=1, title="Day 1", activities=[
                        "Hangzhou West Lake is a scenic freshwater lake ideal for walking.",
                        "Hangzhou Lingyin Temple features Buddhist carvings and shaded paths.",
                    ], evidence_sources=["hangzhou.md"]),
                ],
                budget_items=[], risk_notices=[], alternatives=[],
                evidence_sources=["hangzhou.md"],
                evidence_trace_id=evidence.trace.trace_id,
            ),
        ]
        settings = SelfConsistencySettings(
            num_candidates=2, selection_criterion="evidence_coverage"
        )
        sc_planner = SelfConsistencyPlanner(
            inner_planner=inner, settings=settings
        )

        result = sc_planner.plan(request, evidence)
        assert result.destination == "Hangzhou"
        assert "evidence_coverage" in result.summary

    def test_uses_deterministic_reflect_when_no_service_provided(self):
        """未注入 ReflectionService 时使用确定性回退。"""
        request = _make_request()
        evidence = _make_evidence()
        inner = RuleBasedTravelPlanner()
        settings = SelfConsistencySettings(num_candidates=2)

        planner = SelfConsistencyPlanner(
            inner_planner=inner,
            reflection_service=None,
            settings=settings,
        )

        result = planner.plan(request, evidence)
        assert result.destination == "Hangzhou"

    def test_returns_single_candidate_when_all_are_duplicates(self):
        """所有候选相同时（确定性规划器），返回单个去重后的计划。"""
        request = _make_request()
        evidence = _make_evidence()

        inner = RuleBasedTravelPlanner()  # 完全确定性
        settings = SelfConsistencySettings(num_candidates=5)
        sc_planner = SelfConsistencyPlanner(
            inner_planner=inner, settings=settings
        )

        result = sc_planner.plan(request, evidence)
        # 5 个候选经过去重变为 1 个
        assert "[SC:1/5" in result.summary
