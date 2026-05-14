"""旅行智能体流水线的离线评估。

在不需要LLM的情况下评估计划质量 — 使用基于规则的规划器，
使评估保持确定性且无需API密钥。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from travel_agent.agent.graph import build_travel_agent_graph
from travel_agent.agent.planner import RuleBasedTravelPlanner, TravelPlanner
from travel_agent.agent.schemas import TravelPlan
from travel_agent.rag.models import EvidenceBundle, RetrievalTrace, SearchResult

# ---------------------------------------------------------------------------
# 评估用例定义
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentEvalCase:
    query: str
    destination: str = ""
    expected_days: int = 1
    expect_budget: bool = True
    expect_risk_notices: bool = True
    expected_evidence_sources: tuple[str, ...] = ()
    low_confidence_ok: bool = False  # 预期证据为空/薄弱的用例
    expected_empty: bool = False  # 预期不返回任何证据的用例
    category: str = ""


# ---------------------------------------------------------------------------
# 评估指标
# ---------------------------------------------------------------------------


@dataclass
class AgentEvalMetrics:
    total_cases: int = 0
    days_match: int = 0
    budget_present: int = 0
    risk_notices_present: int = 0
    evidence_source_coverage: float = 0.0
    low_confidence_handled: int = 0
    low_confidence_total: int = 0
    empty_result_handled: int = 0
    empty_result_total: int = 0
    validation_passed: int = 0
    avg_latency_ms: float = 0.0

    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "days_match_rate": self.days_match / self.total_cases if self.total_cases else 1.0,
            "budget_present_rate": (
                self.budget_present / self.total_cases if self.total_cases else 1.0
            ),
            "risk_notices_rate": (
                self.risk_notices_present / self.total_cases if self.total_cases else 1.0
            ),
            "evidence_source_coverage": (
                self.evidence_source_coverage / self.total_cases if self.total_cases else 1.0
            ),
            "low_confidence_handling_rate": (
                self.low_confidence_handled / self.low_confidence_total
                if self.low_confidence_total
                else 1.0
            ),
            "empty_result_handling_rate": (
                self.empty_result_handled / self.empty_result_total
                if self.empty_result_total
                else 1.0
            ),
            "validation_pass_rate": (
                self.validation_passed / self.total_cases if self.total_cases else 1.0
            ),
            "avg_latency_ms": self.avg_latency_ms,
            "failures": self.failures,
        }


# ---------------------------------------------------------------------------
# Mock证据服务
# ---------------------------------------------------------------------------


class _MockEvidenceService:
    """基于字典的确定性证据服务。

    如果查询不在映射表中，则返回空证据。
    """

    def __init__(self, evidence_map: dict[str, list[SearchResult]]) -> None:
        self._map = evidence_map

    def retrieve_evidence(
        self,
        query: str,
        top_k: int | None = None,
        destination: str | None = None,
        section: str | None = None,
        travel_type: str | None = None,
        season: str | None = None,
        retrieval_mode: str | None = None,
    ) -> EvidenceBundle:
        results = self._map.get(query, [])
        confidence = 0.85 if results else 0.1
        return EvidenceBundle(
            question=query,
            results=results,
            trace=RetrievalTrace.create(
                retrieval_mode="mock",
                requested_top_k=top_k or 5,
                candidate_k=len(results),
                returned_results=len(results),
                empty_result=not results,
                destination=destination or "",
                section=section or "",
                travel_type=travel_type or "",
                season=season or "",
                embedding_provider="mock",
                reranker="keyword",
                collection_version="eval",
                metadata_filters={},
                vector_hits=[],
                keyword_hits=[],
                fused_hits=[],
                reranked_hits=[],
            ),
            query_analysis={"destination": destination or ""},
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# 核心评估逻辑
# ---------------------------------------------------------------------------


def evaluate_agent_plans(
    cases: list[AgentEvalCase],
    evidence_map: dict[str, list[SearchResult]],
    planner: TravelPlanner | None = None,
) -> AgentEvalMetrics:
    """在不调用任何LLM的情况下运行智能体评估。

    使用Mock证据服务和基于规则的规划器，
    以确定性方式评估计划质量。
    """
    import time

    active_planner = planner or RuleBasedTravelPlanner()
    rag_service = _MockEvidenceService(evidence_map)
    graph = build_travel_agent_graph(rag_service, planner=active_planner)
    metrics = AgentEvalMetrics(total_cases=len(cases))
    total_latency = 0.0

    for case in cases:
        t0 = time.perf_counter()
        state = graph.invoke({"question": case.query})
        latency = (time.perf_counter() - t0) * 1000
        total_latency += latency

        plan: TravelPlan = state["plan"]
        is_valid: bool = state.get("is_valid", False)
        errors: list[str] = state.get("validation_errors", [])

        # --- 指标：天数匹配 ---
        if plan.days == case.expected_days:
            metrics.days_match += 1
        else:
            metrics.failures.append(
                f"days_mismatch: query={case.query!r}, "
                f"expected={case.expected_days}, actual={plan.days}"
            )

        # --- 指标：预算存在 ---
        if (plan.budget_items and len(plan.budget_items) > 0) == case.expect_budget:
            metrics.budget_present += 1
        else:
            metrics.failures.append(
                f"budget_mismatch: query={case.query!r}, expected={case.expect_budget}, "
                f"actual={len(plan.budget_items)} items"
            )

        # --- 指标：风险提示存在 ---
        if (plan.risk_notices and len(plan.risk_notices) > 0) == case.expect_risk_notices:
            metrics.risk_notices_present += 1
        else:
            metrics.failures.append(
                f"risk_mismatch: query={case.query!r}, expected={case.expect_risk_notices}, "
                f"actual={len(plan.risk_notices)} notices"
            )

        # --- 指标：证据来源覆盖率 ---
        if case.expected_evidence_sources:
            expected_set = set(case.expected_evidence_sources)
            actual_set = set(plan.evidence_sources)
            coverage = len(expected_set & actual_set) / len(expected_set) if expected_set else 1.0
            if coverage < 1.0:
                metrics.failures.append(
                    f"evidence_coverage: query={case.query!r}, expected={expected_set}, "
                    f"actual={actual_set}, coverage={coverage:.2f}"
                )
        else:
            coverage = 1.0
        metrics.evidence_source_coverage += coverage

        # --- 指标：低置信度处理 ---
        if case.low_confidence_ok:
            metrics.low_confidence_total += 1
            # 合理处理：计划仍生成包含备选/回退方案的输出
            if plan.alternatives or (plan.risk_notices and len(plan.risk_notices) >= 1):
                metrics.low_confidence_handled += 1
            else:
                metrics.failures.append(
                    f"low_confidence_mishandled: query={case.query!r}"
                )

        # --- 指标：空结果处理 ---
        if case.expected_empty:
            metrics.empty_result_total += 1
            # 合理处理：即使没有可用证据，计划在结构上仍然有效（有天数和预算/风险）
            if plan.days > 0 and len(plan.budget_items) > 0:
                metrics.empty_result_handled += 1
            else:
                metrics.failures.append(
                    f"empty_result_mishandled: query={case.query!r}, errors={errors}"
                )

        # --- 指标：校验通过 ---
        if is_valid:
            metrics.validation_passed += 1
        else:
            metrics.failures.append(
                f"validation_failed: query={case.query!r}, errors={errors}"
            )

    metrics.avg_latency_ms = total_latency / len(cases) if cases else 0.0
    return metrics


# ---------------------------------------------------------------------------
# 评估用例加载
# ---------------------------------------------------------------------------


def load_agent_eval_cases(path: Path) -> list[AgentEvalCase]:
    cases: list[AgentEvalCase] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        try:
            cases.append(
                AgentEvalCase(
                    query=payload["query"],
                    destination=payload.get("destination", ""),
                    expected_days=payload.get("expected_days", 1),
                    expect_budget=payload.get("expect_budget", True),
                    expect_risk_notices=payload.get("expect_risk_notices", True),
                    expected_evidence_sources=tuple(
                        payload.get("expected_evidence_sources", ())
                    ),
                    low_confidence_ok=payload.get("low_confidence_ok", False),
                    expected_empty=payload.get("expected_empty", False),
                    category=payload.get("category", ""),
                )
            )
        except KeyError as exc:
            raise ValueError(f"missing field {exc!s} in {path}:{line_number}") from exc
    return cases


def build_eval_report(
    metrics: AgentEvalMetrics,
    case_count: int,
    planner_name: str = "RuleBasedTravelPlanner",
) -> dict[str, Any]:
    return {
        "metrics": metrics.as_dict(),
        "run": {
            "total_cases": case_count,
            "planner": planner_name,
            "mode": "offline_deterministic",
        },
    }
