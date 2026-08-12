"""对话式旅行规划 Agent 的 LangGraph 图编排。

图每次运行一个"回合"——处理用户消息、澄清需求、检查信息槽位是否完整，
然后要么返回等待更多信息，要么生成计划。REPL 为每条用户消息调用一次图。
状态通过 checkpointer 在多次调用之间持久化。

流程:
  START → clarify → summarize → slot_tracker
    ├── (信息不完整) → END
    └── (信息完整)
          ├── (无现有计划) → invoke_planning → present_plan → feedback_router → END
          ├── (有计划 + 槽位变更/修改意图) → invoke_planning → ... → END
          ├── (有计划 + 追问) → present_plan → feedback_router → END
          └── (有计划 + 确认) → feedback_router → END
"""

from __future__ import annotations

import warnings
from typing import Any

import langchain_core._api.deprecation  # noqa: F401

# 抑制 langgraph 导入时的弃用警告
warnings.simplefilter("ignore")
from langgraph.graph import END, START, StateGraph  # noqa: E402

warnings.filters.pop(0)

from travel_agent.agent.nodes import EvidenceService, MemoryService  # noqa: E402
from travel_agent.agent.planner import TravelPlanner  # noqa: E402
from travel_agent.agent.reflection import ReflectionService  # noqa: E402
from travel_agent.conversation.nodes import (  # noqa: E402
    clarify_node,
    classify_feedback_intent,
    conversation_summary_node,
    feedback_router_node,
    invoke_planning_node,
    present_plan_node,
    slot_tracker_node,
)
from travel_agent.conversation.state import ConversationState  # noqa: E402
from travel_agent.skills.registry import SkillRegistry  # noqa: E402


def _route_after_slot_tracker(state: ConversationState) -> str:
    """槽位完整则决定下一步：规划、展示现有计划、或直接反馈。

    已有计划时，根据槽位变更和用户意图避免不必要的重规划：
    - 槽位变更（目的地/天数变化）或明确修改/新行程意图 → 重新规划
    - 追问现有计划 → 重新展示计划
    - 确认/感谢 → 直接进入反馈路由
    """
    if not state.get("slot_filling_complete"):
        return END

    plan_output = state.get("planning_output", {}) or {}
    plan_dict = plan_output.get("plan")
    if not plan_dict:
        return "invoke_planning"

    # 检查关键槽位是否与现有计划不一致
    new_dest = state.get("clarified_destination", "")
    new_days = state.get("clarified_days", 0)
    plan_dest = plan_dict.get("destination", "")
    plan_days = plan_dict.get("days", 0)
    slots_changed = (new_dest and new_dest != plan_dest) or (new_days and new_days != plan_days)

    intent = classify_feedback_intent(state.get("user_message", ""))

    if slots_changed or intent in ("modify", "new_trip"):
        return "invoke_planning"
    if intent == "approve":
        return "feedback_router"
    return "present_plan"


def _route_after_feedback(state: ConversationState) -> str:
    """根据反馈分类进行路由。

    所有反馈路径都返回 END——REPL 将为下一条用户消息重新调用图。
    之前对追问（question）会循环回 present_plan，但因两个节点之间无状态变化，
    导致 GraphRecursionError，因此统一返回 END。
    """
    return END


def build_conversation_graph(
    chat_model: Any,
    rag_service: EvidenceService,
    planner: TravelPlanner | None = None,
    checkpointer: Any | None = None,
    memory_service: MemoryService | None = None,
    reflection_service: ReflectionService | None = None,
    skill_registry: SkillRegistry | None = None,
) -> Any:
    """构建并编译对话式旅行规划 Agent 图。

    参数
    ----------
    chat_model:
        LangChain 聊天模型，用于对话澄清和计划展示。
    rag_service:
        RAG 证据检索服务。
    planner:
        旅行规划器（默认使用 ``build_default_planner()``）。
    checkpointer:
        SQLite 检查点，用于跨回合状态持久化。
    memory_service:
        用户长期记忆服务。
    reflection_service:
        事实性审核服务。
    """
    graph = StateGraph(ConversationState)

    # ── 节点 ──────────────────────────────────────────────────────
    graph.add_node(
        "clarify",
        lambda state: clarify_node(state, chat_model),
    )
    graph.add_node(
        "summarize",
        lambda state: conversation_summary_node(state, chat_model),
    )
    graph.add_node("slot_tracker", slot_tracker_node)
    graph.add_node(
        "invoke_planning",
        lambda state: invoke_planning_node(
            state,
            rag_service=rag_service,
            planner=planner,
            memory_service=memory_service,
            reflection_service=reflection_service,
            skill_registry=skill_registry,
        ),
    )
    graph.add_node(
        "present_plan",
        lambda state: present_plan_node(state, chat_model),
    )
    graph.add_node("feedback_router", feedback_router_node)

    # ── 边 ──────────────────────────────────────────────────────
    graph.add_edge(START, "clarify")
    graph.add_edge("clarify", "summarize")
    graph.add_edge("summarize", "slot_tracker")

    # 槽位追踪 → 决策：规划 / 展示现有计划 / 直接反馈 / 结束
    graph.add_conditional_edges(
        "slot_tracker",
        _route_after_slot_tracker,
        {
            "invoke_planning": "invoke_planning",
            "present_plan": "present_plan",
            "feedback_router": "feedback_router",
            END: END,
        },
    )

    graph.add_edge("invoke_planning", "present_plan")
    graph.add_edge("present_plan", "feedback_router")

    # 反馈：所有路径均结束，REPL 将为下一条用户消息重新调用图
    graph.add_conditional_edges(
        "feedback_router",
        _route_after_feedback,
        {
            END: END,
        },
    )

    return graph.compile(checkpointer=checkpointer)
