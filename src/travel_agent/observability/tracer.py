"""LLMOps tracing abstraction — LangSmith primary, Langfuse optional.

No API keys required. When keys are absent, tracing is a silent no-op.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from travel_agent.agent.schemas import ReflectionReport, TravelPlan, TravelRequest
from travel_agent.rag.models import EvidenceBundle


@dataclass
class AgentTraceContext:
    """Metrics collected during one agent planning run."""

    user_request: str = ""
    parsed_request: TravelRequest | None = None
    retrieved_evidence_count: int = 0
    retrieval_confidence: float = 0.0
    planner_model: str = ""
    planner_fallback_used: bool = False
    final_plan: TravelPlan | None = None
    latency_ms: float = 0.0
    validation_passed: bool = False
    validation_errors: list[str] = field(default_factory=list)
    reflection_passed: bool | None = None
    reflection_evidence_coverage: float = 0.0
    reflection_confidence: float = 0.0
    reflection_flag_count: int = 0

    thread_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "user_request": self.user_request,
            "parsed_request": self.parsed_request.model_dump() if self.parsed_request else None,
            "retrieved_evidence_count": self.retrieved_evidence_count,
            "retrieval_confidence": self.retrieval_confidence,
            "planner_model": self.planner_model,
            "planner_fallback_used": self.planner_fallback_used,
            "latency_ms": self.latency_ms,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "reflection_passed": self.reflection_passed,
            "reflection_evidence_coverage": self.reflection_evidence_coverage,
            "reflection_confidence": self.reflection_confidence,
            "reflection_flag_count": self.reflection_flag_count,
            "final_plan_summary": self.final_plan.summary if self.final_plan else "",
            "final_plan_days": self.final_plan.days if self.final_plan else 0,
            "final_plan_has_budget": bool(self.final_plan and self.final_plan.budget_items),
            "final_plan_has_risks": bool(self.final_plan and self.final_plan.risk_notices),
            "final_plan_evidence_sources": (
                self.final_plan.evidence_sources if self.final_plan else []
            ),
            "thread_id": self.thread_id,
        }


class AgentTracer:
    """Tracing facade that writes to LangSmith and/or Langfuse when configured.

    Design:
    - LangSmith is primary (LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY)
    - Langfuse is optional (LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY)
    - When neither is configured, all calls are no-ops.
    """

    def __init__(self) -> None:
        self._langsmith_client: object = None
        self._langfuse_client: object = None
        self._langsmith_ok: bool | None = None
        self._langfuse_ok: bool | None = None
        self._start_ms: float = 0.0
        self._run_name: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_run(self, run_name: str, user_request: str) -> AgentTraceContext:
        ctx = AgentTraceContext(user_request=user_request)
        self._start_ms = time.perf_counter()
        self._run_name = run_name
        self._maybe_start_langsmith(run_name, user_request)
        self._maybe_start_langfuse(run_name, user_request)
        return ctx

    def record_parse(self, ctx: AgentTraceContext, request: TravelRequest) -> None:
        ctx.parsed_request = request
        self._maybe_langsmith_metadata("parse", {
            "destination": request.destination,
            "days": request.days,
            "audience": request.audience,
            "budget_preference": request.budget_preference,
        })

    def record_retrieval(self, ctx: AgentTraceContext, evidence: EvidenceBundle) -> None:
        ctx.retrieved_evidence_count = len(evidence.results)
        ctx.retrieval_confidence = evidence.confidence
        self._maybe_langsmith_metadata("retrieval", {
            "evidence_count": len(evidence.results),
            "confidence": evidence.confidence,
            "trace_id": evidence.trace.trace_id,
        })

    def record_planner(self, ctx: AgentTraceContext, model: str, fallback: bool = False) -> None:
        ctx.planner_model = model
        ctx.planner_fallback_used = fallback
        self._maybe_langsmith_metadata("planner", {
            "model": model,
            "fallback_used": fallback,
        })

    def record_validation(self, ctx: AgentTraceContext, passed: bool, errors: list[str]) -> None:
        ctx.validation_passed = passed
        ctx.validation_errors = errors

    def record_reflection(self, ctx: AgentTraceContext, report: ReflectionReport) -> None:
        ctx.reflection_passed = report.passed
        ctx.reflection_evidence_coverage = report.evidence_coverage
        ctx.reflection_confidence = report.confidence_score
        ctx.reflection_flag_count = len(report.hallucination_flags)
        self._maybe_langsmith_metadata("reflection", {
            "passed": report.passed,
            "evidence_coverage": report.evidence_coverage,
            "confidence": report.confidence_score,
            "hallucination_flags": len(report.hallucination_flags),
        })

    def finish_run(self, ctx: AgentTraceContext, plan: TravelPlan, thread_id: str = "") -> None:
        ctx.final_plan = plan
        ctx.latency_ms = (time.perf_counter() - self._start_ms) * 1000
        ctx.thread_id = thread_id
        payload = ctx.as_dict()
        self._maybe_finish_langsmith(payload)
        self._maybe_finish_langfuse(payload)

    # ------------------------------------------------------------------
    # LangSmith (primary)
    # ------------------------------------------------------------------

    def _langsmith_enabled(self) -> bool:
        if self._langsmith_ok is not None:
            return self._langsmith_ok
        self._langsmith_ok = False
        try:
            if os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1") and (
                os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGCHAIN_ENDPOINT")
            ):
                import langsmith  # type: ignore[import-untyped]

                self._langsmith_client = langsmith.Client()
                self._langsmith_ok = True
        except Exception:
            pass
        return self._langsmith_ok

    def _maybe_start_langsmith(self, run_name: str, user_input: str) -> None:
        if not self._langsmith_enabled():
            return

    def _maybe_langsmith_metadata(self, step: str, metadata: dict[str, Any]) -> None:
        if not self._langsmith_enabled():
            return
        try:
            run = self._langsmith_client  # type: ignore[union-attr]
            if run is not None and hasattr(run, "update_run"):
                pass  # The run will be populated at finish via create_run
        except Exception:
            pass

    def _maybe_finish_langsmith(self, payload: dict[str, Any]) -> None:
        if not self._langsmith_enabled():
            return
        try:
            client = self._langsmith_client  # type: ignore[union-attr]
            if client is not None:
                client.create_run(
                    name=self._run_name,
                    run_type="chain",
                    inputs={"user_request": payload["user_request"]},
                    outputs={
                        "plan_summary": payload["final_plan_summary"],
                        "plan_days": payload["final_plan_days"],
                        "has_budget": payload["final_plan_has_budget"],
                        "has_risks": payload["final_plan_has_risks"],
                    },
                    extra=payload,
                    end_time=time.time(),
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Langfuse (optional)
    # ------------------------------------------------------------------

    def _langfuse_enabled(self) -> bool:
        if self._langfuse_ok is not None:
            return self._langfuse_ok
        self._langfuse_ok = False
        try:
            if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
                import langfuse  # type: ignore[import-untyped]

                self._langfuse_client = langfuse.Langfuse(
                    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                    secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
                    host=os.getenv("LANGFUSE_HOST"),
                )
                self._langfuse_ok = True
        except Exception:
            pass
        return self._langfuse_ok

    def _maybe_start_langfuse(self, run_name: str, user_input: str) -> None:
        if not self._langfuse_enabled():
            return

    def _maybe_finish_langfuse(self, payload: dict[str, Any]) -> None:
        if not self._langfuse_enabled():
            return
        try:
            client = self._langfuse_client  # type: ignore[union-attr]
            if client is not None:
                trace = client.trace(
                    name=self._run_name,
                    input={"user_request": payload["user_request"]},
                    output={
                        "plan_summary": payload["final_plan_summary"],
                        "plan_days": payload["final_plan_days"],
                    },
                    metadata=payload,
                )
                # Span for trace grouping
                trace.span(
                    name="plan_generation",
                    metadata={
                        "latency_ms": payload["latency_ms"],
                        "model": payload["planner_model"],
                        "confidence": payload["retrieval_confidence"],
                    },
                )
                client.flush()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Global tracer singleton
# ---------------------------------------------------------------------------

_tracer: AgentTracer | None = None


def get_tracer() -> AgentTracer:
    global _tracer
    if _tracer is None:
        _tracer = AgentTracer()
    return _tracer


def reset_tracer() -> None:
    global _tracer
    _tracer = None


def build_tracer() -> AgentTracer:
    return get_tracer()


@contextmanager
def trace_agent_run(run_name: str, user_request: str) -> Iterator[AgentTraceContext]:
    """Context manager for instrumenting one agent planning run."""
    tracer = get_tracer()
    ctx = tracer.start_run(run_name=run_name, user_request=user_request)
    try:
        yield ctx
    finally:
        pass  # Caller must call tracer.finish_run
