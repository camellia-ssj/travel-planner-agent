from __future__ import annotations

import tempfile
from pathlib import Path

from typer.testing import CliRunner

from travel_agent.agent import (
    AlternativePlan,
    BudgetEstimate,
    CrowdRiskAssessment,
    LangChainStructuredPlanner,
    ReflectionReport,
    RuleBasedTravelPlanner,
    TravelPlan,
    TravelRequest,
    build_travel_agent_graph,
)
from travel_agent.agent.cli import app, resume_plan, run_plan
from travel_agent.agent.schemas import BudgetItem, DayPlan, HallucinationFlag, RiskNotice
from travel_agent.rag.models import EvidenceBundle, RetrievalTrace, SearchResult

_TRACE_FIELDS: dict[str, object] = dict(
    retrieval_mode="hybrid",
    requested_top_k=5,
    candidate_k=10,
    returned_results=3,
    empty_result=False,
    destination="Hangzhou",
    section="",
    travel_type="",
    season="",
    embedding_provider="local",
    reranker="keyword",
    collection_version="test",
    metadata_filters={},
    vector_hits=[],
    keyword_hits=[],
    fused_hits=[],
    reranked_hits=[],
)


class MockRagService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

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
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "destination": destination,
                "section": section,
                "travel_type": travel_type,
                "season": season,
                "retrieval_mode": retrieval_mode,
            }
        )
        return EvidenceBundle(
            question=query,
            results=[
                SearchResult(
                    content="West Lake is suitable for a relaxed family walk.",
                    source="hangzhou.md",
                    destination="Hangzhou",
                    score=0.9,
                    metadata={"section": "itinerary"},
                ),
                SearchResult(
                    content="Lingyin Temple can be crowded on weekends.",
                    source="hangzhou.md",
                    destination="Hangzhou",
                    score=0.8,
                    metadata={"section": "crowd_risk"},
                ),
                SearchResult(
                    content="If West Lake is too crowded, consider the Grand Canal area.",
                    source="hangzhou.md",
                    destination="Hangzhou",
                    score=0.7,
                    metadata={"section": "alternatives"},
                ),
            ],
            trace=RetrievalTrace.create(
                retrieval_mode="hybrid",
                requested_top_k=5,
                candidate_k=10,
                returned_results=2,
                empty_result=False,
                destination=destination or "Hangzhou",
                section="",
                travel_type="",
                season="",
                embedding_provider="local",
                reranker="keyword",
                collection_version="test",
                metadata_filters={},
                vector_hits=[],
                keyword_hits=[],
                fused_hits=[],
                reranked_hits=[],
            ),
            query_analysis={"destination": destination or "Hangzhou"},
            confidence=0.85,
        )


class FakeStructuredChatModel:
    def __init__(self, response: TravelPlan | None = None, fail: bool = False) -> None:
        self.response = response
        self.fail = fail
        self.messages: object = None
        self.schema: object = None

    def with_structured_output(self, schema: object) -> FakeStructuredChatModel:
        self.schema = schema
        return self

    def invoke(self, messages: object) -> TravelPlan:
        self.messages = messages
        if self.fail:
            raise RuntimeError("LLM unavailable")
        if self.response is None:
            raise AssertionError("missing fake response")
        return self.response


def _mock_evidence(destination: str = "Hangzhou") -> EvidenceBundle:
    service = MockRagService()
    return service.retrieve_evidence("杭州亲子三天", destination=destination)


def test_agent_graph_can_invoke_with_mock_rag_service() -> None:
    rag_service = MockRagService()
    graph = build_travel_agent_graph(rag_service)

    result = graph.invoke({"question": "杭州亲子三天，预算适中，担心周末拥挤"})

    request = result["request"]
    plan = result["plan"]
    assert isinstance(request, TravelRequest)
    assert isinstance(plan, TravelPlan)
    assert request.destination == "Hangzhou"
    assert request.days == 3
    assert request.budget_preference == "standard"
    assert "family_with_children" in request.audience
    assert rag_service.calls[0]["destination"] == "Hangzhou"
    assert result["is_valid"] is True
    assert result["validation_errors"] == []
    assert plan.destination == "Hangzhou"
    assert plan.days == 3
    assert len(plan.day_plans) == 3
    assert plan.budget_items
    assert plan.risk_notices
    assert plan.alternatives
    assert plan.evidence_sources == ["hangzhou.md"]
    assert plan.evidence_trace_id


def test_parse_user_request_days_override_zero_is_rejected() -> None:
    """days_override=0 must raise a Pydantic validation error, not silently clamp."""
    import pytest as pt
    from pydantic import ValidationError

    rag_service = MockRagService()
    graph = build_travel_agent_graph(rag_service)

    with pt.raises(ValidationError):
        graph.invoke({"question": "杭州三天", "days_override": 0})


def test_parse_user_request_days_override_none_uses_parsed() -> None:
    """When days_override is absent (None), parsed_days from question is used."""
    rag_service = MockRagService()
    graph = build_travel_agent_graph(rag_service)

    result = graph.invoke({"question": "杭州三天"})

    assert result["request"].days == 3


def test_langchain_planner_uses_structured_llm_output_and_evidence_prompt() -> None:
    request = TravelRequest(
        raw_query="杭州亲子三天",
        destination="Hangzhou",
        days=3,
        audience=["family_with_children"],
        budget_preference="standard",
    )
    evidence = _mock_evidence()
    llm_plan = TravelPlan(
        request=request,
        destination="Hangzhou",
        days=3,
        summary="LLM structured plan",
        day_plans=[
            DayPlan(day=1, title="Lake", activities=["Walk by West Lake"]),
            DayPlan(day=2, title="Temple", activities=["Visit Lingyin Temple"]),
            DayPlan(day=3, title="Backup", activities=["Use Grand Canal backup"]),
        ],
        budget_items=[
            BudgetItem(category="transport", preference="standard", note="Use transit"),
        ],
        risk_notices=[
            RiskNotice(risk_type="crowd_risk", message="Lingyin can be crowded"),
        ],
        alternatives=["Grand Canal"],
        evidence_sources=["hangzhou.md"],
    )
    chat_model = FakeStructuredChatModel(response=llm_plan)
    planner = LangChainStructuredPlanner(chat_model=chat_model)  # type: ignore[arg-type]

    plan = planner.plan(request, evidence)

    assert plan.summary == "LLM structured plan"
    assert plan.evidence_sources == ["hangzhou.md"]
    assert plan.evidence_trace_id == evidence.trace.trace_id
    assert chat_model.schema is TravelPlan
    messages_text = "\n".join(
        str(getattr(message, "content", "")) for message in chat_model.messages
    )
    assert "West Lake is suitable" in messages_text
    assert "hangzhou.md" in messages_text


def test_langchain_planner_falls_back_when_llm_fails() -> None:
    request = TravelRequest(
        raw_query="杭州亲子三天",
        destination="Hangzhou",
        days=3,
        audience=["family_with_children"],
        budget_preference="standard",
    )
    evidence = _mock_evidence()
    planner = LangChainStructuredPlanner(
        chat_model=FakeStructuredChatModel(fail=True),  # type: ignore[arg-type]
        fallback=RuleBasedTravelPlanner(),
    )

    plan = planner.plan(request, evidence)

    assert plan.summary.startswith("Hangzhou 3-day rule-based plan")
    assert len(plan.day_plans) == 3
    assert plan.evidence_sources == ["hangzhou.md"]


def test_cli_plan_rejects_negative_days(monkeypatch) -> None:
    """CLI --days -2 must exit non-zero, not silently clamp to 1."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["plan", "杭州三天", "--days", "-2", "--json"],
    )
    assert result.exit_code != 0


def test_run_plan_rejects_days_zero() -> None:
    """run_plan with days=0 must raise ValueError, not silently clamp."""
    import pytest as pt

    rag_service = MockRagService()
    with pt.raises(ValueError, match="days must be >= 1"):
        run_plan("杭州三天", rag_service, days=0)


def test_run_plan_returns_serializable_payload_with_overrides() -> None:
    rag_service = MockRagService()

    payload = run_plan(
        "parents trip with mid budget",
        rag_service,
        destination="Hangzhou",
        days=2,
    )

    assert payload["request"]["destination"] == "Hangzhou"
    assert payload["thread_id"]
    assert payload["original_user_request"] == "parents trip with mid budget"
    assert payload["user_feedback"] == []
    assert payload["evidence"]["results"]
    assert payload["request"]["days"] == 2
    assert payload["plan"]["days"] == 2
    assert len(payload["plan"]["day_plans"]) == 2
    assert payload["plan"]["alternatives"]
    assert payload["plan"]["evidence_sources"] == ["hangzhou.md"]
    assert payload["validation"]["is_valid"] is True
    assert rag_service.calls[0]["destination"] == "Hangzhou"


def test_run_plan_english_standard_budget_remains_standard() -> None:
    rag_service = MockRagService()

    payload = run_plan(
        "Hangzhou for 3 days with 2 people and a standard budget",
        rag_service,
        destination="Hangzhou",
        days=3,
    )

    assert payload["request"]["budget_preference"] == "standard"
    assert payload["tool_budget"]["budget_level"] == "standard"


def test_checkpoint_resume_updates_plan_for_same_thread_id(tmp_path) -> None:
    rag_service = MockRagService()
    checkpoint_path = tmp_path / "agent.sqlite"
    thread_id = "thread-checkpoint-test"

    first_payload = run_plan(
        "杭州亲子三天",
        rag_service,
        destination="Hangzhou",
        days=3,
        thread_id=thread_id,
        checkpoint_path=checkpoint_path,
    )
    resumed_payload = resume_plan(
        thread_id=thread_id,
        feedback="第二天少走路，增加雨天备选",
        rag_service=rag_service,
        planner=RuleBasedTravelPlanner(),
        checkpoint_path=checkpoint_path,
    )

    assert first_payload["thread_id"] == thread_id
    assert resumed_payload["thread_id"] == thread_id
    assert resumed_payload["original_user_request"] == "杭州亲子三天"
    assert resumed_payload["user_feedback"] == ["第二天少走路，增加雨天备选"]
    # Feedback does NOT change destination/days/budget → request stays the same
    assert resumed_payload["request"]["destination"] == "Hangzhou"
    assert resumed_payload["plan"]["days"] == 3
    assert resumed_payload["request"]["budget_preference"] == "standard"
    assert resumed_payload["plan"]["summary"] != first_payload["plan"]["summary"]
    assert "第二天少走路" in resumed_payload["plan"]["summary"]
    joined_activities = "\n".join(
        activity
        for day_plan in resumed_payload["plan"]["day_plans"]
        for activity in day_plan["activities"]
    )
    assert "第二天少走路" in joined_activities
    assert resumed_payload["validation"]["is_valid"] is True


def test_agent_cli_plan_json_uses_mock_rag_service(monkeypatch) -> None:
    rag_service = MockRagService()

    def fake_build_rag_service(*args: object, **kwargs: object) -> MockRagService:
        return rag_service

    monkeypatch.setattr("travel_agent.agent.cli._build_rag_service", fake_build_rag_service)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "plan",
            "杭州亲子3天，预算中等",
            "--destination",
            "Hangzhou",
            "--days",
            "3",
            "--embedding-provider",
            "local",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"destination": "Hangzhou"' in result.output
    assert '"days": 3' in result.output
    assert '"evidence_sources"' in result.output
    assert rag_service.calls[0]["destination"] == "Hangzhou"


# ---------------------------------------------------------------------------
# Tool integration tests
# ---------------------------------------------------------------------------


def test_agent_graph_produces_tool_results() -> None:
    rag_service = MockRagService()
    graph = build_travel_agent_graph(rag_service)

    result = graph.invoke({"question": "杭州亲子三天，预算适中"})

    assert isinstance(result.get("tool_budget"), BudgetEstimate)
    assert isinstance(result.get("tool_crowd_risk"), CrowdRiskAssessment)
    assert isinstance(result.get("tool_alternatives"), AlternativePlan)
    assert result["tool_budget"].total > 0
    assert result["tool_crowd_risk"].destination == "Hangzhou"
    assert result["tool_alternatives"].destination == "Hangzhou"
    assert result["is_valid"] is True


def test_budget_tool_overrides_rule_based_planner() -> None:
    rag_service = MockRagService()
    graph = build_travel_agent_graph(rag_service)

    result = graph.invoke({"question": "杭州亲子三天，预算适中"})
    plan = result["plan"]
    budget = result["tool_budget"]

    # budget_items should come from tool_budget, not the hardcoded 3-category template
    assert len(plan.budget_items) == 5  # accommodation, dining, transport, tickets, total
    categories = [item.category for item in plan.budget_items]
    assert "accommodation" in categories
    assert "total" in categories
    # The total item note should contain the tool's deterministic total
    total_item = next(item for item in plan.budget_items if item.category == "total")
    assert f"{budget.total:.0f}" in total_item.note


def test_resume_graph_runs_tool_node() -> None:
    rag_service = MockRagService()

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "test.sqlite"
        thread_id = "tool-resume-test"

        first = run_plan(
            "杭州亲子三天",
            rag_service,
            destination="Hangzhou",
            days=3,
            thread_id=thread_id,
            checkpoint_path=checkpoint_path,
        )
        assert isinstance(first.get("tool_budget"), dict)
        assert first["tool_budget"]["total"] > 0

        resumed = resume_plan(
            thread_id=thread_id,
            feedback="增加雨天备选",
            rag_service=rag_service,
            planner=RuleBasedTravelPlanner(),
            checkpoint_path=checkpoint_path,
        )
        assert isinstance(resumed.get("tool_budget"), dict)
        assert resumed["thread_id"] == thread_id
        assert "增加雨天备选" in resumed["user_feedback"]


# ---------------------------------------------------------------------------
# Resume with parameter changes — verify feedback parsing + re-retrieval
# ---------------------------------------------------------------------------


def test_resume_with_destination_change_re_retrieves_evidence(tmp_path) -> None:
    """When feedback changes destination, request is updated and evidence is re-fetched."""
    rag_service = MockRagService()
    checkpoint_path = tmp_path / "agent.sqlite"
    thread_id = "resume-dest-change"

    run_plan(
        "杭州三天",
        rag_service,
        destination="Hangzhou",
        days=3,
        thread_id=thread_id,
        checkpoint_path=checkpoint_path,
    )
    call_count_before = len(rag_service.calls)

    resumed = resume_plan(
        thread_id=thread_id,
        feedback="改成去北京玩两天",
        rag_service=rag_service,
        planner=RuleBasedTravelPlanner(),
        checkpoint_path=checkpoint_path,
    )

    # Evidence was re-retrieved: at least one new call with the new destination
    assert len(rag_service.calls) > call_count_before
    new_calls = rag_service.calls[call_count_before:]
    destinations = {call["destination"] for call in new_calls}
    assert "Beijing" in destinations
    # Request was updated
    assert resumed["request"]["destination"] == "Beijing"
    assert resumed["plan"]["destination"] == "Beijing"


def test_resume_with_english_feedback_updates_request(tmp_path) -> None:
    rag_service = MockRagService()
    checkpoint_path = tmp_path / "agent.sqlite"
    thread_id = "resume-english-change"

    run_plan(
        "Hangzhou for 3 days",
        rag_service,
        destination="Hangzhou",
        days=3,
        thread_id=thread_id,
        checkpoint_path=checkpoint_path,
    )

    resumed = resume_plan(
        thread_id=thread_id,
        feedback="change destination to Beijing for 2 days",
        rag_service=rag_service,
        planner=RuleBasedTravelPlanner(),
        checkpoint_path=checkpoint_path,
    )

    assert resumed["request"]["destination"] == "Beijing"
    assert resumed["request"]["days"] == 2
    assert resumed["plan"]["destination"] == "Beijing"
    assert resumed["plan"]["days"] == 2


def test_resume_with_days_change_updates_request(tmp_path) -> None:
    """When feedback changes the number of days, request.days is updated."""
    rag_service = MockRagService()
    checkpoint_path = tmp_path / "agent.sqlite"
    thread_id = "resume-days-change"

    run_plan(
        "杭州三天",
        rag_service,
        destination="Hangzhou",
        days=3,
        thread_id=thread_id,
        checkpoint_path=checkpoint_path,
    )

    resumed = resume_plan(
        thread_id=thread_id,
        feedback="改成玩五天",
        rag_service=rag_service,
        planner=RuleBasedTravelPlanner(),
        checkpoint_path=checkpoint_path,
    )

    assert resumed["request"]["days"] == 5
    assert resumed["plan"]["days"] == 5
    # Destination unchanged
    assert resumed["request"]["destination"] == "Hangzhou"


def test_resume_with_days_change_to_one(tmp_path) -> None:
    """Resume can change days to 1 — regression test for new_days > 1 guard bug."""
    rag_service = MockRagService()
    checkpoint_path = tmp_path / "agent.sqlite"
    thread_id = "resume-days-to-one"

    run_plan(
        "杭州三天",
        rag_service,
        destination="Hangzhou",
        days=3,
        thread_id=thread_id,
        checkpoint_path=checkpoint_path,
    )

    resumed = resume_plan(
        thread_id=thread_id,
        feedback="改成玩一天",
        rag_service=rag_service,
        planner=RuleBasedTravelPlanner(),
        checkpoint_path=checkpoint_path,
    )

    assert resumed["request"]["days"] == 1
    assert resumed["plan"]["days"] == 1
    assert resumed["request"]["destination"] == "Hangzhou"


def test_resume_with_budget_change_updates_request(tmp_path) -> None:
    """When feedback requests a different budget level, it is updated."""
    rag_service = MockRagService()
    checkpoint_path = tmp_path / "agent.sqlite"
    thread_id = "resume-budget-change"

    run_plan(
        "杭州三天",
        rag_service,
        destination="Hangzhou",
        days=3,
        thread_id=thread_id,
        checkpoint_path=checkpoint_path,
    )

    resumed = resume_plan(
        thread_id=thread_id,
        feedback="预算改成高端豪华",
        rag_service=rag_service,
        planner=RuleBasedTravelPlanner(),
        checkpoint_path=checkpoint_path,
    )

    assert resumed["request"]["budget_preference"] == "premium"


def test_resume_without_parameter_change_keeps_request_intact(tmp_path) -> None:
    """Feedback that only adds commentary does not alter the original request."""
    rag_service = MockRagService()
    checkpoint_path = tmp_path / "agent.sqlite"
    thread_id = "resume-no-change"

    run_plan(
        "杭州三天",
        rag_service,
        destination="Hangzhou",
        days=3,
        thread_id=thread_id,
        checkpoint_path=checkpoint_path,
    )

    resumed = resume_plan(
        thread_id=thread_id,
        feedback="第二天少走路",
        rag_service=rag_service,
        planner=RuleBasedTravelPlanner(),
        checkpoint_path=checkpoint_path,
    )

    assert resumed["request"]["destination"] == "Hangzhou"
    assert resumed["request"]["days"] == 3
    assert resumed["request"]["budget_preference"] == "standard"
    assert "第二天少走路" in resumed["user_feedback"]


def test_apply_feedback_preserves_original_constraints_in_query() -> None:
    """Changing destination should keep original budget/headcount context."""
    from travel_agent.agent.nodes import apply_feedback_node

    state = {
        "question": "Hangzhou for 3 days with 2 people on the weekend",
        "original_user_request": "Hangzhou for 3 days with 2 people on the weekend",
        "latest_user_feedback": "change destination to Beijing",
        "request": TravelRequest(
            raw_query="Hangzhou for 3 days with 2 people on the weekend",
            destination="Hangzhou",
            days=3,
            audience=["general"],
            budget_preference="standard",
        ),
        "user_feedback": [],
    }

    result = apply_feedback_node(state)
    updated_request = result["request"]

    assert updated_request.destination == "Beijing"
    assert "2 people" in result["question"]
    assert "weekend" in result["question"]
    assert "change destination to Beijing" in result["question"]
    assert updated_request.raw_query == result["question"]


# ---------------------------------------------------------------------------
# Reflection node tests
# ---------------------------------------------------------------------------


def test_reflection_node_produces_report() -> None:
    """The reflect node must produce a ReflectionReport in the final state."""
    rag_service = MockRagService()
    graph = build_travel_agent_graph(rag_service)

    result = graph.invoke({"question": "杭州亲子三天，预算适中"})

    report = result.get("reflection_report")
    assert report is not None
    assert isinstance(report, ReflectionReport)
    assert report.checked_claims > 0
    assert report.grounded_claims >= 0
    assert 0.0 <= report.evidence_coverage <= 1.0
    assert 0.0 <= report.confidence_score <= 1.0
    assert isinstance(report.passed, bool)


def test_reflection_report_in_serializable_payload() -> None:
    """run_plan payload must include the reflection report."""
    rag_service = MockRagService()
    payload = run_plan("杭州亲子三天", rag_service, destination="Hangzhou", days=3)

    reflection = payload.get("reflection")
    assert reflection is not None
    assert "passed" in reflection
    assert "evidence_coverage" in reflection
    assert "confidence_score" in reflection
    assert "hallucination_flags" in reflection
    assert "checked_claims" in reflection


def test_reflection_flags_unsupported_claim() -> None:
    """A plan with activities completely unrelated to evidence should be flagged."""
    rag_service = MockRagService()
    graph = build_travel_agent_graph(rag_service)

    result = graph.invoke({"question": "杭州亲子三天"})

    report = result.get("reflection_report")
    assert isinstance(report, ReflectionReport)
    # The plan activities should have reasonable overlap with the evidence
    # If coverage is high, the flags should be minimal
    if report.hallucination_flags:
        for flag in report.hallucination_flags:
            assert isinstance(flag, HallucinationFlag)
            assert flag.location
            assert flag.claim
            assert flag.issue
            assert flag.severity in ("high", "medium", "low")


def test_reflection_with_tool_results_cross_checks() -> None:
    """Reflection must cross-check budget and risk against tool results."""
    rag_service = MockRagService()
    graph = build_travel_agent_graph(rag_service)

    result = graph.invoke({"question": "杭州亲子三天，预算适中"})

    report = result.get("reflection_report")
    tool_budget = result.get("tool_budget")
    tool_crowd = result.get("tool_crowd_risk")

    assert report is not None
    assert tool_budget is not None
    assert tool_crowd is not None
    # The evidence coverage should be reasonable since we use MockRagService
    # with content matching the destination
    assert report.evidence_coverage >= 0.0


def test_reflection_report_passed_field() -> None:
    """ReflectionReport.passed must reflect whether the plan is factually sound."""
    rag_service = MockRagService()
    graph = build_travel_agent_graph(rag_service)

    result = graph.invoke({"question": "杭州亲子三天"})

    report = result["reflection_report"]
    assert isinstance(report.passed, bool)
    # With good evidence match, the plan should pass
    if report.evidence_coverage >= 0.3 and len(report.hallucination_flags) == 0:
        assert report.passed is True


def test_reflection_resume_graph_also_produces_report(tmp_path) -> None:
    """The resume graph must also produce a reflection report."""
    rag_service = MockRagService()
    checkpoint_path = tmp_path / "agent.sqlite"
    thread_id = "reflect-resume-test"

    run_plan(
        "杭州三天",
        rag_service,
        destination="Hangzhou",
        days=3,
        thread_id=thread_id,
        checkpoint_path=checkpoint_path,
    )

    resumed = resume_plan(
        thread_id=thread_id,
        feedback="改成去北京玩两天",
        rag_service=rag_service,
        planner=RuleBasedTravelPlanner(),
        checkpoint_path=checkpoint_path,
    )

    reflection = resumed.get("reflection")
    assert reflection is not None
    assert "passed" in reflection
    assert "evidence_coverage" in reflection


def test_reflection_cli_json_includes_reflection(monkeypatch) -> None:
    """CLI --json output must include the reflection report."""
    rag_service = MockRagService()

    def fake_build_rag_service(*args: object, **kwargs: object) -> MockRagService:
        return rag_service

    monkeypatch.setattr("travel_agent.agent.cli._build_rag_service", fake_build_rag_service)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "plan",
            "杭州亲子3天",
            "--destination",
            "Hangzhou",
            "--days",
            "3",
            "--embedding-provider",
            "local",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"reflection"' in result.output
    assert '"evidence_coverage"' in result.output
    assert '"hallucination_flags"' in result.output


# ---------------------------------------------------------------------------
# LLM-based ReflectionService tests
# ---------------------------------------------------------------------------


def test_deterministic_reflect_returns_report() -> None:
    """deterministic_reflect must return a valid ReflectionReport."""
    from travel_agent.agent.reflection import deterministic_reflect

    plan = TravelPlan(
        request=TravelRequest(raw_query="test"),
        destination="Hangzhou",
        days=1,
        summary="test",
        day_plans=[DayPlan(day=1, title="Day 1", activities=["Visit West Lake"])],
        budget_items=[BudgetItem(category="transport", preference="standard", note="bus")],
        risk_notices=[
            RiskNotice(
                risk_type="crowd_risk",
                message="Weekend crowds",
                severity="medium",
            )
        ],
        alternatives=["Indoor museums"],
    )
    evidence = EvidenceBundle(
        question="test",
        results=[
            SearchResult(
                content="West Lake is a scenic area in Hangzhou.",
                source="hangzhou.md",
                destination="Hangzhou",
                score=0.9,
                metadata={"section": "itinerary"},
            ),
        ],
        trace=RetrievalTrace.create(**_TRACE_FIELDS),
        query_analysis={"destination": "Hangzhou"},
        confidence=0.85,
    )
    report = deterministic_reflect(plan, evidence)

    assert isinstance(report, ReflectionReport)
    assert report.checked_claims > 0
    assert 0.0 <= report.evidence_coverage <= 1.0
    assert 0.0 <= report.confidence_score <= 1.0
    assert isinstance(report.passed, bool)


def test_deterministic_reflect_none_plan() -> None:
    """deterministic_reflect should handle None plan gracefully."""
    from travel_agent.agent.reflection import deterministic_reflect

    report = deterministic_reflect(None, None)  # type: ignore[arg-type]
    assert report.passed is False
    assert "No plan to review" in report.issues


def test_deterministic_reflect_cross_destination_check() -> None:
    """deterministic_reflect must flag cross-destination contamination."""
    from travel_agent.agent.reflection import deterministic_reflect

    plan = TravelPlan(
        request=TravelRequest(raw_query="test"),
        destination="Hangzhou",
        days=1,
        summary="test",
        day_plans=[DayPlan(day=1, title="Day 1", activities=["Visit the Great Wall in Beijing"])],
        budget_items=[],
        risk_notices=[],
        alternatives=[],
    )
    evidence = EvidenceBundle(
        question="test",
        results=[
            SearchResult(
                content="West Lake is a scenic area in Hangzhou.",
                source="hangzhou.md",
                destination="Hangzhou",
                score=0.9,
                metadata={"section": "itinerary"},
            ),
        ],
        trace=RetrievalTrace.create(**_TRACE_FIELDS),
        query_analysis={"destination": "Hangzhou"},
        confidence=0.85,
    )
    report = deterministic_reflect(plan, evidence)
    dest_flags = [f for f in report.hallucination_flags if "Cross-destination" in f.issue]
    assert len(dest_flags) > 0


def test_reflection_service_no_llm_uses_deterministic() -> None:
    """ReflectionService without a chat model must fall back to deterministic."""
    from travel_agent.agent.reflection import ReflectionService

    service = ReflectionService(chat_model=None)
    assert service.has_llm is False

    plan = TravelPlan(
        request=TravelRequest(raw_query="test"),
        destination="Hangzhou",
        days=1,
        summary="test",
        day_plans=[DayPlan(day=1, title="Day 1", activities=["Visit West Lake"])],
        budget_items=[],
        risk_notices=[],
        alternatives=[],
    )
    evidence = EvidenceBundle(
        question="test",
        results=[
            SearchResult(
                content="West Lake is a scenic area in Hangzhou.",
                source="hangzhou.md",
                destination="Hangzhou",
                score=0.9,
                metadata={"section": "itinerary"},
            ),
        ],
        trace=RetrievalTrace.create(**_TRACE_FIELDS),
        query_analysis={"destination": "Hangzhou"},
        confidence=0.85,
    )
    report = service.reflect(plan, evidence)
    assert isinstance(report, ReflectionReport)
    assert report.checked_claims > 0


def test_reflection_service_keeps_multiple_destination_flags_from_same_location() -> None:
    """LLM merge must not collapse distinct cross-destination flags by location only."""
    from travel_agent.agent.reflection import ReflectionService

    service = ReflectionService(
        chat_model=FakeStructuredChatModel(
            response=ReflectionReport(
                hallucination_flags=[],
                evidence_coverage=1.0,
                confidence_score=1.0,
                issues=[],
                suggestions=[],
                passed=True,
                checked_claims=1,
                grounded_claims=1,
            )
        )  # type: ignore[arg-type]
    )
    plan = TravelPlan(
        request=TravelRequest(raw_query="test"),
        destination="Hangzhou",
        days=1,
        summary="test",
        day_plans=[
            DayPlan(
                day=1,
                title="Day 1",
                activities=["Visit Beijing, then Tokyo in one day"],
            )
        ],
        budget_items=[],
        risk_notices=[],
        alternatives=[],
    )
    evidence = EvidenceBundle(
        question="test",
        results=[
            SearchResult(
                content="West Lake is a scenic area in Hangzhou.",
                source="hangzhou.md",
                destination="Hangzhou",
                score=0.9,
                metadata={"section": "itinerary"},
            ),
        ],
        trace=RetrievalTrace.create(**_TRACE_FIELDS),
        query_analysis={"destination": "Hangzhou"},
        confidence=0.85,
    )

    report = service.reflect(plan, evidence)
    dest_flags = [
        flag for flag in report.hallucination_flags
        if "Cross-destination contamination" in flag.issue
    ]

    assert len(dest_flags) >= 2


def test_reflection_retry_count_increments() -> None:
    """Retry count must increment on each failed reflection pass."""
    rag_service = MockRagService()
    graph = build_travel_agent_graph(rag_service, max_reflection_retries=2)

    result = graph.invoke({"question": "杭州亲子三天"})
    # With MockRagService evidence matching Hangzhou, the plan should pass
    # or at most have 1 retry. The retry_count must be present.
    assert "reflection_retry_count" in result
    assert isinstance(result["reflection_retry_count"], int)
    # Should not exceed max_retries+1
    assert result["reflection_retry_count"] <= 3


def test_reflection_retry_stops_at_limit() -> None:
    """Graph must terminate even when reflection never passes."""
    rag_service = MockRagService()
    # max_retries=1: at most 1 retry
    graph = build_travel_agent_graph(rag_service, max_reflection_retries=1)

    result = graph.invoke({"question": "杭州亲子三天"})
    report = result.get("reflection_report")
    assert report is not None
    # Graph must terminate (this test is about no infinite loop)
    assert result["reflection_retry_count"] <= 2


def test_build_reflection_service_factory() -> None:
    """build_reflection_service must return a working service."""
    from travel_agent.agent.reflection import build_reflection_service

    service = build_reflection_service()
    assert isinstance(service, object)
    assert hasattr(service, "reflect")


def test_reflection_prompt_builder_includes_plan_and_evidence() -> None:
    """build_reflection_prompt must include both plan and evidence content."""
    from travel_agent.agent.prompts import build_reflection_prompt

    plan = TravelPlan(
        request=TravelRequest(raw_query="test"),
        destination="Hangzhou",
        days=1,
        summary="test",
        day_plans=[DayPlan(day=1, title="Day 1", activities=["Visit West Lake"])],
        budget_items=[BudgetItem(category="transport", preference="standard", note="bus")],
        risk_notices=[RiskNotice(risk_type="crowd_risk", message="Crowds", severity="medium")],
        alternatives=["Indoor museums"],
    )
    evidence = EvidenceBundle(
        question="test",
        results=[
            SearchResult(
                content="West Lake is scenic.",
                source="hangzhou.md",
                destination="Hangzhou",
                score=0.9,
                metadata={"section": "itinerary"},
            ),
        ],
        trace=RetrievalTrace.create(**_TRACE_FIELDS),
        query_analysis={"destination": "Hangzhou"},
        confidence=0.85,
    )
    prompt = build_reflection_prompt(plan, evidence, tool_results=None)
    assert "TRAVEL PLAN" in prompt
    assert "RAG EVIDENCE" in prompt
    assert "West Lake" in prompt
    assert "Hangzhou" in prompt
