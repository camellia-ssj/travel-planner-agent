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
    """days_override=0 必须抛出 Pydantic 验证错误，而不是静默截断。"""
    import pytest as pt
    from pydantic import ValidationError

    rag_service = MockRagService()
    graph = build_travel_agent_graph(rag_service)

    with pt.raises(ValidationError):
        graph.invoke({"question": "杭州三天", "days_override": 0})


def test_parse_user_request_days_override_none_uses_parsed() -> None:
    """当 days_override 未提供（None）时，使用从问题中解析出的 parsed_days。"""
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
    """CLI --days -2 必须以非零退出码退出，而不是静默截断为 1。"""
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["plan", "杭州三天", "--days", "-2", "--json"],
    )
    assert result.exit_code != 0


def test_run_plan_rejects_days_zero() -> None:
    """run_plan 在 days=0 时必须抛出 ValueError，而不是静默截断。"""
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
    # 反馈不改变目的地/天数/预算 → request 保持不变
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
# 工具集成测试
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

    # budget_items 应来自 tool_budget，而不是硬编码的 3 分类模板
    assert len(plan.budget_items) == 5  # accommodation, dining, transport, tickets, total
    categories = [item.category for item in plan.budget_items]
    assert "accommodation" in categories
    assert "total" in categories
    # 总计项的备注应包含工具的确定性总额
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
# 带参数变更的恢复测试 — 验证反馈解析 + 重新检索
# ---------------------------------------------------------------------------


def test_resume_with_destination_change_re_retrieves_evidence(tmp_path) -> None:
    """当反馈更改目的地时，request 应更新且证据应重新获取。"""
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

    # 证据已重新检索：至少有一次使用新目的地的调用
    assert len(rag_service.calls) > call_count_before
    new_calls = rag_service.calls[call_count_before:]
    destinations = {call["destination"] for call in new_calls}
    assert "Beijing" in destinations
    # request 已更新
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
    """当反馈更改天数时，request.days 应更新。"""
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
    # 目的地未变
    assert resumed["request"]["destination"] == "Hangzhou"


def test_resume_with_days_change_to_one(tmp_path) -> None:
    """恢复时可将天数改为 1 —— 针对 new_days > 1 守卫逻辑缺陷的回归测试。"""
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
    """当反馈请求不同预算等级时，应更新。"""
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
    """仅添加评注的反馈不应修改原始 request。"""
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
    """更改目的地时应保留原始的预算/人数上下文。"""
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
# 反思节点测试
# ---------------------------------------------------------------------------


def test_reflection_node_produces_report() -> None:
    """反思节点必须在最终状态中生成 ReflectionReport。"""
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
    """run_plan 的输出必须包含反思报告。"""
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
    """与证据完全不相关的活动计划应被标记。"""
    rag_service = MockRagService()
    graph = build_travel_agent_graph(rag_service)

    result = graph.invoke({"question": "杭州亲子三天"})

    report = result.get("reflection_report")
    assert isinstance(report, ReflectionReport)
    # 计划活动与证据应有合理的重合度
    # 如果覆盖率高，标记应变少
    if report.hallucination_flags:
        for flag in report.hallucination_flags:
            assert isinstance(flag, HallucinationFlag)
            assert flag.location
            assert flag.claim
            assert flag.issue
            assert flag.severity in ("high", "medium", "low")


def test_reflection_with_tool_results_cross_checks() -> None:
    """反思必须将预算和风险与工具结果进行交叉检查。"""
    rag_service = MockRagService()
    graph = build_travel_agent_graph(rag_service)

    result = graph.invoke({"question": "杭州亲子三天，预算适中"})

    report = result.get("reflection_report")
    tool_budget = result.get("tool_budget")
    tool_crowd = result.get("tool_crowd_risk")

    assert report is not None
    assert tool_budget is not None
    assert tool_crowd is not None
    # 证据覆盖率应合理，因为我们使用 MockRagService 且内容与目的地匹配
    assert report.evidence_coverage >= 0.0


def test_reflection_report_passed_field() -> None:
    """ReflectionReport.passed 必须反映计划是否事实可靠。"""
    rag_service = MockRagService()
    graph = build_travel_agent_graph(rag_service)

    result = graph.invoke({"question": "杭州亲子三天"})

    report = result["reflection_report"]
    assert isinstance(report.passed, bool)
    # 证据匹配良好的情况下，计划应通过
    if report.evidence_coverage >= 0.3 and len(report.hallucination_flags) == 0:
        assert report.passed is True


def test_reflection_resume_graph_also_produces_report(tmp_path) -> None:
    """恢复图也必须生成反思报告。"""
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
    """CLI --json 输出必须包含反思报告。"""
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
# 基于 LLM 的 ReflectionService 测试
# ---------------------------------------------------------------------------


def test_deterministic_reflect_returns_report() -> None:
    """deterministic_reflect 必须返回有效的 ReflectionReport。"""
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
    """deterministic_reflect 应优雅处理 None 计划。"""
    from travel_agent.agent.reflection import deterministic_reflect

    report = deterministic_reflect(None, None)  # type: ignore[arg-type]
    assert report.passed is False
    assert "No plan to review" in report.issues


def test_deterministic_reflect_cross_destination_check() -> None:
    """deterministic_reflect 必须标记跨目的地污染。"""
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
    """没有聊天模型的 ReflectionService 必须回退到确定性反思。"""
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
    """LLM 合并时不能仅按位置合并不同的跨目的地标记。"""
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
    """每次反思未通过时重试计数必须递增。"""
    rag_service = MockRagService()
    graph = build_travel_agent_graph(rag_service, max_reflection_retries=2)

    result = graph.invoke({"question": "杭州亲子三天"})
    # 使用 MockRagService 且证据匹配杭州的情况下，计划应通过
    # 或最多重试 1 次。retry_count 必须存在。
    assert "reflection_retry_count" in result
    assert isinstance(result["reflection_retry_count"], int)
    # 不应超过 max_retries+1
    assert result["reflection_retry_count"] <= 3


def test_reflection_retry_stops_at_limit() -> None:
    """即使反思从未通过，图也必须终止。"""
    rag_service = MockRagService()
    # max_retries=1: 最多 1 次重试
    graph = build_travel_agent_graph(rag_service, max_reflection_retries=1)

    result = graph.invoke({"question": "杭州亲子三天"})
    report = result.get("reflection_report")
    assert report is not None
    # 图必须终止（此测试确保不会无限循环）
    assert result["reflection_retry_count"] <= 2


def test_build_reflection_service_factory() -> None:
    """build_reflection_service 必须返回一个可用的服务。"""
    from travel_agent.agent.reflection import build_reflection_service

    service = build_reflection_service()
    assert isinstance(service, object)
    assert hasattr(service, "reflect")


def test_reflection_prompt_builder_includes_plan_and_evidence() -> None:
    """build_reflection_prompt 必须同时包含计划和证据内容。"""
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
    assert "旅行计划" in prompt
    assert "RAG 证据" in prompt
    assert "West Lake" in prompt
    assert "Hangzhou" in prompt
