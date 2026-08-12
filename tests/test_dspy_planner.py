"""DSPy 优化规划器单元测试。

使用 mock DSPy LM 避免真实 API 调用，测试模块、metric、编译和降级逻辑。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from travel_agent.agent.dspy_planner import (
    DSPyCompileSettings,
    DSPyOptimizedPlanner,
    TravelPlanningModule,
    _parse_plan_json,
    _serialize_evidence,
    _serialize_request,
    _serialize_tool_results,
    build_reflection_metric,
    load_compiled_planner,
)
from travel_agent.agent.planner import RuleBasedTravelPlanner
from travel_agent.agent.schemas import DayPlan, TravelPlan, TravelRequest
from travel_agent.rag.models import EvidenceBundle, RetrievalTrace, SearchResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_evidence(destination: str = "Hangzhou") -> EvidenceBundle:
    """构建基础测试证据。"""
    results = [
        SearchResult(
            content=f"{destination} West Lake is a scenic freshwater lake suitable for a relaxing walk.",
            source=f"{destination.lower()}.md",
            destination=destination,
            score=0.92,
            metadata={"section": "itinerary", "chunk_id": "chunk_001"},
        ),
        SearchResult(
            content=f"{destination} Lingyin Temple features ancient Buddhist carvings and shaded paths.",
            source=f"{destination.lower()}.md",
            destination=destination,
            score=0.89,
            metadata={"section": "itinerary", "chunk_id": "chunk_002"},
        ),
    ]
    return EvidenceBundle(
        question=f"{destination} trip plan",
        results=results,
        trace=RetrievalTrace.create(
            retrieval_mode="mock",
            requested_top_k=5,
            candidate_k=2,
            returned_results=2,
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


def _make_request(
    destination: str = "Hangzhou",
    days: int = 3,
    audience: list[str] | None = None,
    budget: str = "standard",
) -> TravelRequest:
    return TravelRequest(
        raw_query=f"{destination} {days}日游",
        destination=destination,
        days=days,
        audience=audience or ["general"],
        budget_preference=budget,
    )


def _make_valid_plan(request: TravelRequest, evidence: EvidenceBundle) -> TravelPlan:
    """构建一个与证据匹配的有效计划。"""
    return TravelPlan(
        request=request,
        destination=request.destination,
        days=request.days,
        summary=f"{request.destination} {request.days}-day test plan.",
        day_plans=[
            DayPlan(
                day=1,
                title="Day 1",
                activities=[
                    "上午: West Lake scenic walk — a scenic freshwater lake suitable for a relaxing walk.",
                    "下午: Lingyin Temple — ancient Buddhist carvings and shaded paths.",
                ],
                evidence_sources=[evidence.results[0].source],
            )
        ],
        budget_items=[],
        risk_notices=[],
        alternatives=["Backup indoor option available."],
        evidence_sources=[r.source for r in evidence.results],
        evidence_trace_id=evidence.trace.trace_id,
    )


# ---------------------------------------------------------------------------
# 序列化测试
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_serialize_request_produces_valid_json(self):
        req = _make_request("Hangzhou", 3, ["family_with_children"], "standard")
        result = _serialize_request(req)
        data = json.loads(result)
        assert data["destination"] == "Hangzhou"
        assert data["days"] == 3
        assert "family_with_children" in data["audience"]

    def test_serialize_evidence_formats_correctly(self):
        evidence = _make_evidence("Beijing")
        result = _serialize_evidence(evidence)
        assert "Beijing" in result
        assert "source=beijing.md" in result
        assert "section=itinerary" in result
        assert "score=0.920" in result

    def test_serialize_tool_results_handles_none(self):
        assert _serialize_tool_results(None) == "{}"

    def test_serialize_tool_results_handles_empty_dict(self):
        assert _serialize_tool_results({}) == "{}"

    def test_parse_plan_json_valid(self):
        request = _make_request()
        evidence = _make_evidence()
        plan = _make_valid_plan(request, evidence)
        json_str = plan.model_dump_json()
        parsed = _parse_plan_json(json_str)
        assert parsed is not None
        assert parsed.destination == "Hangzhou"
        assert parsed.days == 3

    def test_parse_plan_json_with_markdown_code_block(self):
        request = _make_request()
        evidence = _make_evidence()
        plan = _make_valid_plan(request, evidence)
        json_str = f"```json\n{plan.model_dump_json()}\n```"
        parsed = _parse_plan_json(json_str)
        assert parsed is not None
        assert parsed.destination == "Hangzhou"

    def test_parse_plan_json_invalid_returns_none(self):
        assert _parse_plan_json("not valid json at all") is None

    def test_parse_plan_json_extracts_from_surrounding_text(self):
        request = _make_request()
        evidence = _make_evidence()
        plan = _make_valid_plan(request, evidence)
        json_str = f"Here is your plan: {plan.model_dump_json()} Hope you like it!"
        parsed = _parse_plan_json(json_str)
        assert parsed is not None
        assert parsed.destination == "Hangzhou"


# ---------------------------------------------------------------------------
# DSPy 模块测试
# ---------------------------------------------------------------------------


class TestTravelPlanningModule:
    def test_ensure_module_creates_internal_module(self):
        module = TravelPlanningModule()
        assert module._module is None
        module.ensure_module()
        assert module._module is not None

    def test_forward_requires_ensure_module_first(self):
        module = TravelPlanningModule()
        # forward 会自动调用 ensure_module，然后调用内部 DSPy 模块
        # 我们 mock 内部 _module 避免需要真实 LM
        mock_inner = MagicMock()
        mock_pred = MagicMock()
        mock_pred.plan = _make_valid_plan(_make_request(), _make_evidence()).model_dump_json()
        mock_inner.forward.return_value = mock_pred
        module._module = mock_inner  # 预先注入避免 ensure_module 创建真实模块
        pred = module.forward(
            request=_serialize_request(_make_request()),
            evidence=_serialize_evidence(_make_evidence()),
            tools="{}",
            profile="No history.",
        )
        assert pred is not None
        assert pred.plan is not None

    def test_uncompiled_module_is_not_compiled(self):
        module = TravelPlanningModule()
        module.ensure_module()
        assert not module.is_compiled


# ---------------------------------------------------------------------------
# Metric 测试
# ---------------------------------------------------------------------------


class TestReflectionMetric:
    def test_metric_scores_high_for_good_plan(self):
        request = _make_request()
        evidence = _make_evidence()
        plan = _make_valid_plan(request, evidence)

        metric_fn = build_reflection_metric(evidence)

        # 创建 mock prediction
        pred = MagicMock()
        pred.plan = plan.model_dump_json()

        score = metric_fn(None, pred)
        # 好计划应与证据高匹配 — 分数应 > 0.3
        assert score > 0.3, f"Expected score > 0.3, got {score}"

    def test_metric_scores_zero_for_parse_failure(self):
        evidence = _make_evidence()
        metric_fn = build_reflection_metric(evidence)

        pred = MagicMock()
        pred.plan = "invalid json {{{"

        score = metric_fn(None, pred)
        assert score == 0.0

    def test_metric_scores_low_for_hallucinated_plan(self):
        evidence = _make_evidence("Hangzhou")
        request = _make_request("Hangzhou")
        # 创建含虚假活动的计划 — 使用极短且完全无关的活动文本
        # 确保 deterministic_reflect 的 SequenceMatcher 相似度 < 0.15
        plan = TravelPlan(
            request=request,
            destination="Hangzhou",
            days=1,
            summary="Hangzhou day trip.",
            day_plans=[
                DayPlan(
                    day=1,
                    title="Day 1",
                    activities=[
                        "xyzzy_fake_12345",  # 完全无关的短文本
                    ],
                    evidence_sources=[],
                )
            ],
            budget_items=[],
            risk_notices=[],
            alternatives=[],
            evidence_sources=[],
            evidence_trace_id="",
        )

        metric_fn = build_reflection_metric(evidence)
        pred = MagicMock()
        pred.plan = plan.model_dump_json()

        score = metric_fn(None, pred)
        # 完全编造的计划应有低分
        assert score < 0.3, f"Expected low score for hallucinated plan, got {score}"


# ---------------------------------------------------------------------------
# DSPyOptimizedPlanner 测试
# ---------------------------------------------------------------------------


class FakeDspyModule:
    """返回预设 plan JSON 的 mock DSPy 模块。"""

    def __init__(self, plan_json: str, should_fail: bool = False):
        self.plan_json = plan_json
        self.should_fail = should_fail
        self.ensure_called = False

    def ensure_module(self) -> None:
        self.ensure_called = True

    def forward(self, **kwargs: object) -> object:
        if self.should_fail:
            raise RuntimeError("DSPy prediction failed")
        pred = MagicMock()
        pred.plan = self.plan_json
        return pred


class TestDSPyOptimizedPlanner:
    def test_uses_dspy_module_when_available(self):
        request = _make_request()
        evidence = _make_evidence()
        plan = _make_valid_plan(request, evidence)

        fake_module = FakeDspyModule(plan.model_dump_json(), should_fail=False)
        fallback = RuleBasedTravelPlanner()
        planner = DSPyOptimizedPlanner(
            dspy_module=fake_module,  # type: ignore[arg-type]
            fallback=fallback,
        )

        result = planner.plan(request, evidence)
        assert result.destination == "Hangzhou"
        assert fake_module.ensure_called

    def test_falls_back_on_dspy_failure(self):
        request = _make_request()
        evidence = _make_evidence()

        fake_module = FakeDspyModule("{}", should_fail=True)
        fallback = RuleBasedTravelPlanner()
        planner = DSPyOptimizedPlanner(
            dspy_module=fake_module,  # type: ignore[arg-type]
            fallback=fallback,
        )

        result = planner.plan(request, evidence)
        # 回退到 RuleBasedTravelPlanner — 应生成有效计划
        assert result.destination == "Hangzhou"
        assert result.days == 3

    def test_falls_back_on_invalid_json(self):
        request = _make_request()
        evidence = _make_evidence()

        fake_module = FakeDspyModule("not valid json", should_fail=False)
        fallback = RuleBasedTravelPlanner()
        planner = DSPyOptimizedPlanner(
            dspy_module=fake_module,  # type: ignore[arg-type]
            fallback=fallback,
        )

        result = planner.plan(request, evidence)
        assert result.destination == "Hangzhou"


# ---------------------------------------------------------------------------
# 编译设置测试
# ---------------------------------------------------------------------------


class TestDSPyCompileSettings:
    def test_default_settings(self):
        settings = DSPyCompileSettings()
        assert settings.model == "qwen3-max"
        assert settings.optimizer == "BootstrapFewShot"
        assert settings.max_bootstrapped_demos == 4
        assert settings.max_labeled_demos == 4

    def test_save_path_default(self):
        settings = DSPyCompileSettings()
        assert settings.save_path == Path("data/dspy/compiled_planner.json")


# ---------------------------------------------------------------------------
# 持久化测试
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_load_missing_file_returns_none(self, tmp_path: Path):
        result = load_compiled_planner(tmp_path / "does_not_exist.json")
        assert result is None
