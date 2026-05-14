"""对话式旅行规划 Agent 的 LangGraph 图编排。

图每次运行一个"回合"——处理用户消息、澄清需求、检查信息槽位是否完整，
然后要么返回等待更多信息，要么生成计划。REPL 为每条用户消息调用一次图。
状态通过 checkpointer 在多次调用之间持久化。

流程:
  START → clarify → summarize → slot_tracker
    ├── (信息不完整) → END → (用户补充信息 → 再次调用)
    └── (信息完整) → invoke_planning → present_plan → feedback_router
          ├── (修改/新行程) → END → (用户表达新意图 → 再次调用)
          ├── (确认) → END
          └── (追问) → present_plan
"""

from __future__ import annotations

import warnings
from typing import Any

from travel_agent.agent.nodes import EvidenceService, MemoryService
from travel_agent.agent.planner import TravelPlanner
from travel_agent.agent.reflection import ReflectionService
from travel_agent.conversation.nodes import (
    clarify_node,
    conversation_summary_node,
    feedback_router_node,
    invoke_planning_node,
    present_plan_node,
    slot_tracker_node,
)
from travel_agent.conversation.state import ConversationState

import langchain_core._api.deprecation  # noqa: E402,F401

warnings.simplefilter("ignore")

from langgraph.graph import END, START, StateGraph  # noqa: E402

warnings.filters.pop(0)


def _route_after_slot_tracker(state: ConversationState) -> str:
    """信息槽位完整则进入规划，否则结束等待用户补充。"""
    if state.get("slot_filling_complete"):
        return "invoke_planning"
    return END


def _route_after_feedback(state: ConversationState) -> str:
    """根据反馈分类进行路由。

    所有反馈路径最终都返回 END——REPL 将为下一条用户消息重新调用图。
    present_plan → feedback_router 路径仅在内部追问处理时循环，其他情况结束。
    """
    action = state.get("feedback_action", "question")
    if action == "question" and state.get("plan_generation_count", 0) > 1:
        return "present_plan"
    return END


def build_conversation_graph(
    chat_model: Any,
    rag_service: EvidenceService,
    planner: TravelPlanner | None = None,
    checkpointer: Any | None = None,
    memory_service: MemoryService | None = None,
    reflection_service: ReflectionService | None = None,
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

    # 槽位追踪：完整 → 规划，不完整 → 结束（返回用户）
    graph.add_conditional_edges(
        "slot_tracker",
        _route_after_slot_tracker,
        {
            "invoke_planning": "invoke_planning",
            END: END,
        },
    )

    graph.add_edge("invoke_planning", "present_plan")
    graph.add_edge("present_plan", "feedback_router")

    # 反馈：修改/新行程/确认 → 结束（返回用户），追问 → 重新展示计划
    graph.add_conditional_edges(
        "feedback_router",
        _route_after_feedback,
        {
            END: END,
            "present_plan": "present_plan",
        },
    )

    return graph.compile(checkpointer=checkpointer)
