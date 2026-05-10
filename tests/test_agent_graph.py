from __future__ import annotations

from typer.testing import CliRunner

from travel_agent.agent import (
    LangChainStructuredPlanner,
    RuleBasedTravelPlanner,
    TravelPlan,
    TravelRequest,
    build_travel_agent_graph,
)
from travel_agent.agent.cli import app, resume_plan, run_plan
from travel_agent.agent.schemas import BudgetItem, DayPlan, RiskNotice
from travel_agent.rag.models import EvidenceBundle, RetrievalTrace, SearchResult


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
        planner=RuleBasedTravelPlanner(),
        checkpoint_path=checkpoint_path,
    )

    assert first_payload["thread_id"] == thread_id
    assert resumed_payload["thread_id"] == thread_id
    assert resumed_payload["original_user_request"] == "杭州亲子三天"
    assert resumed_payload["user_feedback"] == ["第二天少走路，增加雨天备选"]
    assert resumed_payload["request"]["destination"] == "Hangzhou"
    assert resumed_payload["plan"]["days"] == 3
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
            "鏉窞浜插瓙3澶╋紝棰勭畻涓瓑",
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
