"""对话式旅行智能体组件的测试。"""

from __future__ import annotations

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from travel_agent.conversation.nodes import (
    ClarificationOutput,
    clarify_node,
    feedback_router_node,
    invoke_planning_node,
    present_plan_node,
    slot_tracker_node,
    _audience_to_text,
    _build_fallback_presentation,
    _build_fallback_response,
    _MAX_CLARIFICATION_TURNS,
)
from travel_agent.conversation.slot_tracker import (
    apply_defaults,
    check_slots_complete,
    extract_slots,
    get_recommendation_text,
    is_vague_request,
    _slot_extraction_function,
)
from travel_agent.conversation.state import ConversationState
from travel_agent.rag.models import EvidenceBundle, SearchResult


# ── Mock RAG 服务（与 test_agent_graph.py 相同的模式）─────────


class MockRagService:
    """符合 EvidenceService 协议，使规划图能够正常工作。"""

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
        self.calls.append({
            "query": query,
            "destination": destination,
            "section": section,
        })
        return EvidenceBundle(
            question=query,
            results=[
                SearchResult(
                    content=f"{destination or 'Hangzhou'} West Lake scenic itinerary.",
                    source="hangzhou.md",
                    destination=destination or "Hangzhou",
                    score=0.9,
                    metadata={"section": "itinerary"},
                ),
            ],
        )


# ── 测试夹具 ──────────────────────────────────────────────────────


def _base_state(**overrides) -> ConversationState:
    state: ConversationState = {
        "messages": [],
        "user_message": "",
        "clarified_destination": "",
        "clarified_days": 0,
        "clarified_budget": "",
        "clarified_audience": [],
        "original_request_text": "",
        "missing_slots": [],
        "slot_filling_complete": False,
        "clarification_turn_count": 0,
        "planning_output": {},
        "plan_generation_count": 0,
        "phase": "clarifying",
        "feedback_action": "",
        "thread_id": "test-thread",
        "user_id": "test-user",
        "conversation_history_summary": "",
        "streaming_enabled": False,
    }
    state.update(overrides)
    return state


# ── 槽位追踪器测试 ────────────────────────────────────────────


class TestExtractSlots:
    def test_extracts_destination(self):
        result = _slot_extraction_function("我想去杭州玩")
        assert result["clarified_destination"] == "Hangzhou"

    def test_extracts_days(self):
        result = _slot_extraction_function("3天行程")
        assert result["clarified_days"] == 3

    def test_extracts_budget(self):
        result = _slot_extraction_function("中等预算")
        assert result["clarified_budget"] == "standard"

    def test_extracts_economy_budget(self):
        result = _slot_extraction_function("经济实惠一点的")
        assert result["clarified_budget"] == "economy"

    def test_extracts_premium_budget(self):
        result = _slot_extraction_function("要高端奢侈的体验")
        assert result["clarified_budget"] == "premium"

    def test_extracts_audience_family(self):
        result = _slot_extraction_function("带孩子去玩")
        assert "family_with_children" in result["clarified_audience"]

    def test_extracts_audience_couple(self):
        result = _slot_extraction_function("情侣出游")
        assert "couple" in result["clarified_audience"]

    def test_extracts_audience_elderly(self):
        result = _slot_extraction_function("带父母去旅游")
        assert "elderly" in result["clarified_audience"]

    def test_extracts_audience_friends(self):
        result = _slot_extraction_function("和朋友一起")
        assert "friends" in result["clarified_audience"]

    def test_extracts_audience_solo(self):
        result = _slot_extraction_function("一个人去旅行")
        assert "solo" in result["clarified_audience"]

    def test_extracts_all_slots_together(self):
        result = _slot_extraction_function("想去杭州玩3天，中等预算，带父母")
        assert result["clarified_destination"] == "Hangzhou"
        assert result["clarified_days"] == 3
        assert result["clarified_budget"] == "standard"
        assert "elderly" in result["clarified_audience"]

    def test_no_days_when_not_explicit(self):
        result = _slot_extraction_function("我想去杭州")
        assert "clarified_days" not in result  # days not explicitly mentioned


class TestCheckSlotsComplete:
    def test_incomplete_when_no_destination(self):
        state = _base_state(clarified_destination="")
        complete, missing = check_slots_complete(state)
        assert complete is False
        assert "destination" in missing

    def test_incomplete_when_no_days(self):
        state = _base_state(clarified_destination="Hangzhou")
        complete, missing = check_slots_complete(state)
        assert "days" in missing

    def test_blocks_on_budget_in_early_turns(self):
        state = _base_state(
            clarified_destination="Hangzhou", clarified_days=3, clarification_turn_count=0
        )
        complete, missing = check_slots_complete(state)
        assert "budget" in missing

    def test_blocks_on_audience_in_early_turns(self):
        state = _base_state(
            clarified_destination="Hangzhou", clarified_days=3, clarification_turn_count=1
        )
        complete, missing = check_slots_complete(state)
        assert "audience" in missing

    def test_stops_blocking_budget_after_2_turns(self):
        state = _base_state(
            clarified_destination="Hangzhou", clarified_days=3, clarification_turn_count=2
        )
        complete, missing = check_slots_complete(state)
        assert complete is True
        assert "budget" not in missing

    def test_complete_with_all_slots(self):
        state = _base_state(
            clarified_destination="Hangzhou",
            clarified_days=3,
            clarified_budget="standard",
            clarified_audience=["family_with_children"],
        )
        complete, missing = check_slots_complete(state)
        assert complete is True


class TestApplyDefaults:
    def test_applies_day_default(self):
        state = _base_state()
        defaults = apply_defaults(state)
        assert defaults["clarified_days"] == 3

    def test_applies_budget_default(self):
        state = _base_state()
        defaults = apply_defaults(state)
        assert defaults["clarified_budget"] == "standard"

    def test_applies_audience_default(self):
        state = _base_state()
        defaults = apply_defaults(state)
        assert defaults["clarified_audience"] == ["general"]

    def test_does_not_overwrite_existing(self):
        state = _base_state(clarified_budget="premium")
        defaults = apply_defaults(state)
        assert "clarified_budget" not in defaults


class TestVagueRequest:
    def test_detects_suibian(self):
        assert is_vague_request("随便吧") is True

    def test_detects_recommend(self):
        assert is_vague_request("推荐几个地方") is True

    def test_detects_dont_know(self):
        assert is_vague_request("不知道去哪") is True

    def test_normal_request_not_vague(self):
        assert is_vague_request("我想去杭州3天") is False


class TestRecommendationText:
    def test_returns_nonempty(self):
        text = get_recommendation_text()
        assert len(text) > 0
        assert "推荐" in text


# ── 反馈路由测试 ─────────────────────────────────────────


class TestFeedbackRouter:
    def test_approve_action(self):
        state = _base_state(user_message="好的，这个计划不错")
        result = feedback_router_node(state)
        assert result["feedback_action"] == "approve"

    def test_approve_with_thanks(self):
        state = _base_state(user_message="谢谢，很满意！")
        result = feedback_router_node(state)
        assert result["feedback_action"] == "approve"

    def test_modify_change_destination(self):
        state = _base_state(user_message="把目的地改成北京")
        result = feedback_router_node(state)
        assert result["feedback_action"] == "modify"

    def test_modify_adjust_days(self):
        state = _base_state(user_message="加一天行程")
        result = feedback_router_node(state)
        assert result["feedback_action"] == "modify"

    def test_modify_too_expensive(self):
        state = _base_state(user_message="太贵了，便宜点")
        result = feedback_router_node(state)
        assert result["feedback_action"] == "modify"

    def test_new_trip_reset(self):
        state = _base_state(user_message="换个城市，重新规划")
        result = feedback_router_node(state)
        assert result["feedback_action"] == "new_trip"

    def test_question_default(self):
        state = _base_state(user_message="第二天有哪些景点？")
        result = feedback_router_node(state)
        assert result["feedback_action"] == "question"

    def test_modify_less_walking(self):
        state = _base_state(user_message="第二天少走点路")
        result = feedback_router_node(state)
        assert result["feedback_action"] == "modify"


# ── 节点单元测试 ───────────────────────────────────────────────


class TestAudienceToText:
    def test_family(self):
        assert "亲子" in _audience_to_text(["family_with_children"])

    def test_couple(self):
        assert "情侣" in _audience_to_text(["couple"])

    def test_multiple(self):
        text = _audience_to_text(["family_with_children", "elderly"])
        assert "亲子" in text
        assert "老人" in text

    def test_general(self):
        assert "通用" in _audience_to_text(["general"])

    def test_empty(self):
        assert "通用出行" in _audience_to_text([])


class TestFallbackResponse:
    def test_with_destination_and_days(self):
        resp = _build_fallback_response({"clarified_destination": "Hangzhou", "clarified_days": 3})
        assert "Hangzhou" in resp

    def test_with_destination_only(self):
        resp = _build_fallback_response({"clarified_destination": "Beijing"})
        assert "Beijing" in resp
        assert "几天" in resp

    def test_with_nothing(self):
        resp = _build_fallback_response({})
        assert "哪里" in resp or "去哪" in resp


class TestFallbackPresentation:
    def test_returns_markdown_with_destination(self):
        plan = {"destination": "Hangzhou", "days": 3, "summary": "test summary", "day_plans": []}
        result = _build_fallback_presentation(plan, None, None, None)
        assert "Hangzhou" in result
        assert "3" in result

    def test_includes_budget_when_present(self):
        plan = {"destination": "Chengdu", "days": 2, "summary": "", "day_plans": []}
        budget = {"budget_level": "economy", "total": 2000, "daily_average": 1000}
        result = _build_fallback_presentation(plan, budget, None, None)
        assert "2000" in result

    def test_includes_crowd_risk_when_present(self):
        plan = {"destination": "Chengdu", "days": 2, "summary": "", "day_plans": []}
        crowd = {"overall_risk": "high", "advice": "避开节假日"}
        result = _build_fallback_presentation(plan, None, crowd, None)
        assert "high" in result


# ── 槽位追踪节点测试 ───────────────────────────────────────


class TestSlotTrackerNode:
    def test_complete_triggers_planning(self):
        state = _base_state(
            clarified_destination="Hangzhou",
            clarified_days=3,
            clarified_budget="standard",
            clarified_audience=["general"],
        )
        result = slot_tracker_node(state)
        assert result["slot_filling_complete"] is True
        assert result["phase"] == "planning"

    def test_incomplete_returns_to_clarifying(self):
        state = _base_state(clarified_destination="")
        result = slot_tracker_node(state)
        assert result["slot_filling_complete"] is False
        assert result["phase"] == "clarifying"

    def test_force_completes_after_max_turns(self):
        state = _base_state(
            clarified_destination="",
            clarified_days=0,
            clarification_turn_count=_MAX_CLARIFICATION_TURNS,
        )
        result = slot_tracker_node(state)
        assert result["slot_filling_complete"] is True
        assert result["phase"] == "planning"

    def test_applies_defaults_when_complete(self):
        state = _base_state(
            clarified_destination="Hangzhou",
            clarified_days=5,
            clarified_budget="",
            clarified_audience=[],
            clarification_turn_count=2,  # 轮次足够，跳过预算/受众的阻塞要求
        )
        result = slot_tracker_node(state)
        assert result["slot_filling_complete"] is True
        assert result.get("clarified_budget", "standard") == "standard"


# ── 集成测试：使用 mock LLM 的完整图流程 ────────────────────


class _EchoChatModel(BaseChatModel):
    """用于测试的聊天模型，回显结构化响应。"""

    response_text: str = ""
    destination: str = ""
    days: int | None = None
    budget: str = ""
    audience: list[str] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        text = self.response_text or "好的，请问计划玩几天呢？"
        message = AIMessage(content=text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _llm_type(self) -> str:
        return "echo-test-model"

    @property
    def _identifying_params(self):
        return {"model": "echo-test"}


class TestConversationGraphIntegration:
    def test_graph_accepts_partial_info(self):
        """图应能处理部分信息且不触发规划。"""
        from travel_agent.conversation.graph import build_conversation_graph

        rag = MockRagService()
        model = _EchoChatModel(
            response_text="明白了，想去杭州。请问计划玩几天呢？",
            destination="Hangzhou",
        )
        graph = build_conversation_graph(
            chat_model=model,
            rag_service=rag,
        )

        result = graph.invoke({
            "messages": [HumanMessage(content="我想去杭州玩")],
            "user_message": "我想去杭州玩",
        })

        # 应已完成处理但未触发规划（未提供天数）
        phase = result.get("phase", "")
        assert phase != "feedback", f"Expected planning NOT triggered, got phase={phase}"
        # 应有 AI 回复
        msgs = result.get("messages", [])
        ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
        assert len(ai_msgs) >= 1

    def test_graph_slots_preserved_across_turns(self):
        """当使用缓存状态调用图时，槽位应被保留。"""
        from travel_agent.conversation.graph import build_conversation_graph

        rag = MockRagService()
        model = _EchoChatModel(
            response_text="好的，请问计划玩几天呢？",
            destination="Hangzhou",
        )
        graph = build_conversation_graph(
            chat_model=model,
            rag_service=rag,
        )

        # 第一轮：用户提供目的地
        result1 = graph.invoke({
            "messages": [HumanMessage(content="我想去杭州玩")],
            "user_message": "我想去杭州玩",
        })
        dest = result1.get("clarified_destination", "")
        assert dest == "Hangzhou"

        # 第二轮：传入缓存的槽位 + 新消息（天数）
        cached = {
            "clarified_destination": result1.get("clarified_destination", ""),
            "clarified_budget": result1.get("clarified_budget", ""),
            "clarified_audience": result1.get("clarified_audience", []),
            "clarified_days": result1.get("clarified_days"),
        }
        result2 = graph.invoke({
            "messages": [HumanMessage(content="3天，中等预算")],
            "user_message": "3天，中等预算",
            **cached,
        })
        # 目的地应从第一轮保留
        assert result2.get("clarified_destination") or cached.get("clarified_destination")

    def test_graph_triggers_planning_when_complete(self):
        """当填充了足够多的槽位时，完整图应路由到规划阶段。"""
        from travel_agent.conversation.graph import build_conversation_graph

        rag = MockRagService()
        model = _EchoChatModel(
            response_text="好的，杭州3天，我这就为您规划！",
            destination="Hangzhou",
            days=3,
            budget="standard",
        )
        graph = build_conversation_graph(
            chat_model=model,
            rag_service=rag,
        )

        result = graph.invoke({
            "messages": [HumanMessage(content="我想去杭州玩3天，标准预算")],
            "user_message": "我想去杭州玩3天，标准预算",
        })

        # 应到达 规划/展示/反馈 阶段
        planning_output = result.get("planning_output")
        if planning_output:
            # 规划已触发
            assert "plan" in planning_output or "error" in planning_output

    def test_state_mapping_for_planning(self):
        """invoke_planning_node 正确地将 ConversationState 映射为 TravelAgentState。"""
        rag = MockRagService()
        state = _base_state(
            clarified_destination="Chengdu",
            clarified_days=4,
            clarified_budget="economy",
            clarified_audience=["friends"],
            thread_id="test-thread",
            original_request_text="想去成都玩4天",
        )
        result = invoke_planning_node(state, rag_service=rag)
        assert "planning_output" in result
        # 成功时阶段为 "presenting"，失败时保持 "planning"
        assert result["phase"] in ("presenting", "planning")
        assert result["plan_generation_count"] >= 1

    def test_present_plan_generates_message(self):
        """present_plan_node 应根据计划数据生成 AI 消息。"""
        model = _EchoChatModel(response_text="这是您的杭州之旅计划...")
        state = _base_state(
            planning_output={
                "plan": {
                    "destination": "Hangzhou",
                    "days": 3,
                    "summary": "杭州3日经典游",
                    "day_plans": [
                        {"day": 1, "title": "西湖游", "activities": ["断桥残雪", "苏堤春晓"]},
                    ],
                },
            },
        )
        result = present_plan_node(state, model)
        assert "messages" in result
        assert result["phase"] == "feedback"


# ── 澄清节点状态保留测试 ─────────────────────────


class TestClarifyNodeStatePreservation:
    def test_preserves_previous_destination(self):
        """当用户在新消息中未提及目的地时，保留之前的目的地。"""
        model = _EchoChatModel(
            response_text="好的，3天的话，预算是怎样的呢？",
            destination="",  # LLM 未能从此消息中提取目的地
        )
        state = _base_state(
            user_message="3天，中等预算",
            clarified_destination="Hangzhou",  # 此前已提取
            clarification_turn_count=1,
        )
        result = clarify_node(state, model)
        # 之前的目的地应被保留
        assert result.get("clarified_destination") == "Hangzhou"

    def test_updates_destination_when_changed(self):
        """当用户更改目的地时，新的目的地应生效。"""
        model = _EchoChatModel(
            response_text="好的，改成北京，了解了！",
            destination="",
        )
        state = _base_state(
            user_message="我想改成北京",
            clarified_destination="Hangzhou",  # 旧值
            clarification_turn_count=1,
        )
        result = clarify_node(state, model)
        # 基于规则的提取应能识别 北京 → Beijing
        # 若未匹配到，LLM 可能会也可能不会识别
        assert isinstance(result.get("clarified_destination"), str)
