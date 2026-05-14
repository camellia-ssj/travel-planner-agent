"""对话式旅行规划 Agent 的扩展状态模式。"""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from travel_agent.memory.models import UserProfile


class ConversationState(TypedDict, total=False):
    """对话式 Agent 图的状态。

    包含对话历史、澄清槽位、规划桥接字段、阶段路由和会话元数据。
    """

    # ── 消息历史 ────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]
    user_message: str  # 用户最新原始输入

    # ── 澄清槽位 ────────────────────────────────────────────
    clarified_destination: str  # 已确认目的地
    clarified_days: int  # 已确认天数
    clarified_budget: str  # "economy" | "standard" | "premium"
    clarified_audience: list[str]  # 出行人员类型
    original_request_text: str  # 用户原始需求文本

    # ── 槽位追踪 ────────────────────────────────────────────
    missing_slots: list[str]  # 缺失的信息项
    slot_filling_complete: bool  # 信息是否收集完整
    clarification_turn_count: int  # 澄清轮次计数

    # ── 规划桥接 ────────────────────────────────────────────
    planning_output: dict[str, Any]  # 规划结果
    plan_generation_count: int  # 计划生成次数

    # ── 阶段与路由 ──────────────────────────────────────────
    phase: str  # "greeting" | "clarifying" | "planning" | "presenting" | "feedback"
    feedback_action: str  # "modify" | "approve" | "question" | "new_trip"

    # ── 会话元数据 ──────────────────────────────────────────
    thread_id: str
    user_id: str
    user_profile: UserProfile
    conversation_history_summary: str

    # ── 流式控制 ────────────────────────────────────────────
    streaming_enabled: bool
