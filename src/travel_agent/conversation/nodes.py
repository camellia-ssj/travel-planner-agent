"""对话式旅行规划 Agent 图的节点实现。"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from travel_agent.agent.graph import build_travel_agent_graph
from travel_agent.agent.nodes import EvidenceService, MemoryService
from travel_agent.agent.planner import TravelPlanner
from travel_agent.agent.reflection import ReflectionService
from travel_agent.agent.schemas import TravelPlan
from travel_agent.conversation.prompts import (
    CLARIFICATION_SYSTEM_PROMPT,
    FEEDBACK_CLASSIFICATION_PROMPT,
    PRESENTATION_SYSTEM_PROMPT,
)
from travel_agent.conversation.slot_tracker import (
    apply_defaults,
    check_slots_complete,
    extract_slots,
    get_recommendation_text,
    is_vague_request,
)
from travel_agent.conversation.state import ConversationState
from travel_agent.memory.models import UserProfile
from travel_agent.rag.models import EvidenceBundle

logger = logging.getLogger(__name__)


# ── 澄清阶段的结构化输出模型 ──────────────────────────────────


class ClarificationOutput(BaseModel):
    """澄清步骤的 LLM 输出。"""
    extracted_destination: str = Field(default="", description="提取的目的地城市名称")
    extracted_days: int | None = Field(default=None, description="提取的游玩天数")
    extracted_budget: str = Field(default="", description="提取的预算: economy/standard/premium")
    extracted_audience: list[str] = Field(default_factory=list, description="提取的出行人员类型")
    response_text: str = Field(description="给用户的中文自然语言回复")
    user_intent: str = Field(
        default="providing_info",
        description="用户意图: providing_info/asking_question/ready_to_plan/vague",
    )
    missing_info_hint: list[str] = Field(default_factory=list, description="还缺少哪些信息")


# ── 欢迎节点 ─────────────────────────────────────────────────────


def greet_node(state: ConversationState) -> dict[str, object]:
    """根据用户画像生成开场问候语。"""
    profile = state.get("user_profile")
    if profile is not None and getattr(profile, "total_trips", 0) > 0:
        last_dest = getattr(profile, "last_destination", "")
        total = getattr(profile, "total_trips", 0)
        if last_dest:
            greeting = (
                f"欢迎回来！👋 我是您的旅行规划顾问小旅。"
                f"之前您去过{last_dest}，已经累计{total}次旅行了。"
                f"这次想去哪里呢？"
            )
        else:
            greeting = (
                f"欢迎回来！👋 我是您的旅行规划顾问小旅。"
                f"您已经累计{total}次旅行了。这次想去哪里呢？"
            )
    else:
        greeting = (
            "您好！👋 我是您的旅行规划顾问**小旅**。\n\n"
            "我可以帮您规划旅行路线，提供预算估算、拥挤风险提醒和备选方案。\n"
            "只需告诉我您的需求，比如：\n"
            "- 目的地（如杭州、成都、东京...）\n"
            "- 游玩天数\n"
            "- 预算偏好（经济/标准/高端）\n"
            "- 出行人员（亲子/情侣/朋友...）\n\n"
            "现在，告诉我您想去哪里吧！"
        )

    return {
        "messages": [AIMessage(content=greeting)],
        "phase": "clarifying",
        "clarification_turn_count": 0,
    }


# ── 澄清节点 ───────────────────────────────────────────────────


def clarify_node(
    state: ConversationState,
    chat_model: BaseChatModel,
) -> dict[str, object]:
    """分析用户消息并生成自然的澄清回复。

    使用 LLM 结构化输出提取旅行槽位并决定下一步询问什么。
    """
    user_message = state.get("user_message", "").strip()
    messages = list(state.get("messages", []))

    # 首先进行确定性槽位提取
    rule_slots = extract_slots(user_message, state)

    # 追踪原始需求文本
    original_text = state.get("original_request_text", "")
    if not original_text and user_message:
        original_text = user_message

    turn_count = state.get("clarification_turn_count", 0)

    # 用户表达模糊需求且还没有目的地时，直接给推荐，不走 LLM
    if is_vague_request(user_message) and not rule_slots.get("clarified_destination"):
        return {
            "messages": [AIMessage(content=get_recommendation_text())],
            "original_request_text": original_text,
            "clarification_turn_count": turn_count + 1,
            "phase": "clarifying",
        }

    # 将规则槽位合并到临时状态中，供 LLM 上下文使用
    merged_slots: dict[str, object] = dict(state)
    merged_slots.update(rule_slots)

    # 构建 LLM 提示词上下文
    context_lines = _build_clarification_context(messages, user_message, merged_slots)

    response = _try_structured_clarify(chat_model, context_lines)
    if response is None:
        response = _rule_based_clarify(user_message, rule_slots, merged_slots)

    # 合并 LLM 提取的槽位与规则槽位
    llm_destination = response.extracted_destination.strip()
    llm_days = response.extracted_days
    llm_budget = response.extracted_budget.strip()
    llm_audience = response.extracted_audience

    # 规则解析对中国城市名称更可靠，优先采用
    # 当新消息中没有提取到时，保留之前的槽位值
    prev_dest = state.get("clarified_destination", "")
    prev_days = state.get("clarified_days")
    prev_budget = state.get("clarified_budget", "")
    prev_audience = state.get("clarified_audience")

    final_slots: dict[str, object] = {}
    if rule_slots.get("clarified_destination"):
        final_slots["clarified_destination"] = rule_slots["clarified_destination"]
    elif llm_destination:
        final_slots["clarified_destination"] = llm_destination
    elif prev_dest:
        final_slots["clarified_destination"] = prev_dest

    if rule_slots.get("clarified_days"):
        final_slots["clarified_days"] = rule_slots["clarified_days"]
    elif llm_days is not None and llm_days >= 1:
        final_slots["clarified_days"] = llm_days
    elif prev_days:
        final_slots["clarified_days"] = prev_days

    if llm_budget:
        final_slots["clarified_budget"] = llm_budget
    elif rule_slots.get("clarified_budget"):
        final_slots["clarified_budget"] = rule_slots["clarified_budget"]
    elif prev_budget:
        final_slots["clarified_budget"] = prev_budget

    if llm_audience:
        final_slots["clarified_audience"] = llm_audience
    elif rule_slots.get("clarified_audience"):
        final_slots["clarified_audience"] = rule_slots["clarified_audience"]
    elif prev_audience:
        final_slots["clarified_audience"] = prev_audience

    ai_text = response.response_text.strip() or _build_fallback_response(final_slots)

    return {
        "messages": [AIMessage(content=ai_text)],
        "original_request_text": original_text,
        "clarification_turn_count": turn_count + 1,
        "phase": "clarifying",
        **final_slots,
    }


def _build_fallback_response(slots: dict[str, object]) -> str:
    """当 LLM 输出为空时，构建通用的兜底回复。"""
    dest = slots.get("clarified_destination", "")
    days = slots.get("clarified_days")
    if dest and days:
        return f"好的，{dest}{days}天的旅行，我来为您规划！"
    if dest:
        return f"明白了，想去{dest}。请问计划玩几天呢？"
    return "请问您想去哪里旅行呢？我可以帮您推荐几个热门目的地~"


# ── 槽位追踪节点 ──────────────────────────────────────────────


def slot_tracker_node(state: ConversationState) -> dict[str, object]:
    """检查是否收集到足够信息以进入规划阶段。

    超过 _MAX_CLARIFICATION_TURNS 后强制完成并使用默认值，
    防止无限澄清循环。
    """
    complete, missing = check_slots_complete(state)
    turn_count = state.get("clarification_turn_count", 0)

    # 超过最大轮次后强制完成
    if turn_count >= _MAX_CLARIFICATION_TURNS:
        defaults = apply_defaults(state)
        if not state.get("clarified_destination") and not defaults.get("clarified_destination"):
            defaults["clarified_destination"] = "杭州"
        return {
            "slot_filling_complete": True,
            "missing_slots": [],
            "phase": "planning",
            **defaults,
        }

    if complete:
        defaults = apply_defaults(state)
        return {
            "slot_filling_complete": True,
            "missing_slots": missing,
            "phase": "planning",
            **defaults,
        }

    return {
        "slot_filling_complete": False,
        "missing_slots": missing,
        "phase": "clarifying",
    }


# ── 调用规划节点 ───────────────────────────────────────────────


def invoke_planning_node(
    state: ConversationState,
    rag_service: EvidenceService,
    planner: TravelPlanner | None = None,
    memory_service: MemoryService | None = None,
    reflection_service: ReflectionService | None = None,
) -> dict[str, object]:
    """将对话槽位映射到 TravelAgentState 并调用规划图。

    现有规划图以函数方式调用（而非 LangGraph 子图），
    以避免 TypedDict 兼容性问题。
    """
    destination = state.get("clarified_destination", "")
    days = state.get("clarified_days", 3)
    budget = state.get("clarified_budget", "standard")
    audience = state.get("clarified_audience", ["general"])
    thread_id = state.get("thread_id", "")
    original_text = state.get("original_request_text", state.get("user_message", ""))

    # 构建自然的查询文本
    audience_text = _audience_to_text(audience)
    budget_text = {"economy": "经济实惠", "standard": "中等标准", "premium": "高端舒适"}.get(
        budget, "中等标准"
    )
    query = f"我想去{destination}玩{days}天，{budget_text}预算，{audience_text}"

    planning_input: dict[str, object] = {
        "question": original_text or query,
        "original_user_request": original_text,
        "destination_override": destination,
        "days_override": days,
        "user_id": state.get("user_id", ""),
        "thread_id": thread_id,
    }

    # 调用规划图
    planning_graph = build_travel_agent_graph(
        rag_service,
        planner=planner,
        memory_service=memory_service,
        reflection_service=reflection_service,
    )

    try:
        result = planning_graph.invoke(planning_input)
    except Exception:
        logger.warning("Planning graph invocation failed, returning error to user")
        return {
            "planning_output": {"error": "规划生成失败，请重试"},
            "phase": "planning",
            "plan_generation_count": state.get("plan_generation_count", 0) + 1,
        }

    # 捕获规划结果
    plan = result.get("plan")
    plan_summary = ""
    if isinstance(plan, TravelPlan):
        plan_summary = plan.summary

    planning_output = {
        "plan": plan.model_dump() if isinstance(plan, TravelPlan) else None,
        "request": result.get("request").model_dump() if result.get("request") else None,
        "evidence_count": (
            len(result["evidence"].results) if result.get("evidence") else 0
        ),
        "tool_budget": (
            result["tool_budget"].model_dump() if result.get("tool_budget") else None
        ),
        "tool_crowd_risk": (
            result["tool_crowd_risk"].model_dump() if result.get("tool_crowd_risk") else None
        ),
        "tool_alternatives": (
            result["tool_alternatives"].model_dump() if result.get("tool_alternatives") else None
        ),
        "reflection_report": (
            result["reflection_report"].model_dump() if result.get("reflection_report") else None
        ),
        "is_valid": result.get("is_valid", False),
        "validation_errors": result.get("validation_errors", []),
        "plan_summary": plan_summary,
    }

    return {
        "planning_output": planning_output,
        "phase": "presenting",
        "plan_generation_count": state.get("plan_generation_count", 0) + 1,
    }


# ── 展示计划节点 ──────────────────────────────────────────────


def present_plan_node(
    state: ConversationState,
    chat_model: BaseChatModel,
) -> dict[str, object]:
    """将旅行计划以自然语言形式展示给用户。"""
    planning_output = state.get("planning_output", {})
    plan_dict = planning_output.get("plan")
    budget = planning_output.get("tool_budget")
    crowd = planning_output.get("tool_crowd_risk")
    alternatives = planning_output.get("tool_alternatives")

    if plan_dict is None:
        error_msg = "抱歉，规划生成时遇到了问题。请再试一次或者告诉我不同的需求。"
        return {
            "messages": [AIMessage(content=error_msg)],
            "phase": "feedback",
        }

    # 构建精简的计划摘要供 LLM 展示
    plan_text = _format_plan_for_presentation(plan_dict, budget, crowd, alternatives)

    try:
        response = chat_model.invoke([
            SystemMessage(content=PRESENTATION_SYSTEM_PROMPT),
            HumanMessage(content=f"请用自然语言向用户介绍这个旅行计划：\n\n{plan_text}"),
        ])
        content = response.content if hasattr(response, "content") else str(response)
    except Exception:
        logger.warning("Plan presentation LLM call failed, using fallback")
        content = _build_fallback_presentation(plan_dict, budget, crowd, alternatives)

    return {
        "messages": [AIMessage(content=str(content))],
        "phase": "feedback",
    }


# ── 反馈路由节点 ───────────────────────────────────────────────


def feedback_router_node(state: ConversationState) -> dict[str, object]:
    """分类用户反馈意图并决定下一步。"""
    user_message = state.get("user_message", "").strip().lower()

    from travel_agent.agent.nodes import _has_change_intent

    action: str

    # 修改意图关键词
    modify_keywords = {
        "改", "换", "调整", "修改", "变成", "换成", "不要", "去掉",
        "加一天", "减一天", "多一天", "少一天", "太贵", "便宜",
        "贵一点", "慢一点", "少走", "多走", "增加", "减少",
    }
    # 确认意图关键词
    approve_keywords = {
        "好的", "可以", "不错", "很好", "满意", "就这样", "没问题",
        "行", "ok", "yes", "好", "棒", "完美", "喜欢",
        "谢谢", "感谢", "辛苦了",
    }
    # 新建行程意图关键词
    new_trip_keywords = {
        "重新", "换一个目的地", "换地方", "再去", "下一次", "另一个",
        "新行程", "全新", "换个城市", "不去了",
    }

    if _has_change_intent(state.get("user_message", "")):
        action = "modify"
    elif any(kw in user_message for kw in new_trip_keywords):
        action = "new_trip"
    elif any(kw in user_message for kw in modify_keywords):
        action = "modify"
    elif any(kw in user_message for kw in approve_keywords):
        action = "approve"
    else:
        action = "question"

    return {
        "feedback_action": action,
        "phase": "feedback",
    }


# ── 对话摘要节点 ──────────────────────────────────────────────


_MAX_MESSAGES_BEFORE_SUMMARY = 20


def conversation_summary_node(
    state: ConversationState,
    chat_model: BaseChatModel,
) -> dict[str, object]:
    """当消息过长时总结对话历史并裁剪消息列表。"""
    messages = list(state.get("messages", []))
    if len(messages) <= _MAX_MESSAGES_BEFORE_SUMMARY:
        return {}

    # 保留最后6条消息，总结前面的
    to_summarize = messages[:-6]
    summary_lines = []
    for msg in to_summarize:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        content = msg.content if hasattr(msg, "content") else str(msg)
        if isinstance(content, str) and len(content) > 100:
            content = content[:100] + "..."
        summary_lines.append(f"[{role}] {content}")

    prev_summary = state.get("conversation_history_summary", "")
    new_summary = "\n".join(summary_lines)

    try:
        summary_response = chat_model.invoke([
            SystemMessage(content="请用2-3句话总结以下对话的关键信息（目的地、天数、预算、人员等）："),
            HumanMessage(content=new_summary),
        ])
        summary = summary_response.content if hasattr(summary_response, "content") else str(summary_response)
    except Exception:
        summary = f"对话摘要: 讨论了{state.get('clarified_destination', '未知目的地')}旅行"

    full_summary = f"{prev_summary}\n{summary}".strip() if prev_summary else summary

    return {
        "conversation_history_summary": full_summary,
        "messages": messages[-6:],
    }


# ── 最大澄清轮次，超过后强制使用默认值 ─────────────────────────

_MAX_CLARIFICATION_TURNS = 8


def _try_structured_clarify(
    chat_model: BaseChatModel,
    context: str,
) -> ClarificationOutput | None:
    """尝试获取结构化澄清输出，含降级链。

    先尝试 structured output，失败则尝试普通调用 + 解析，
    再失败则返回 None 让调用方使用规则兜底。
    """
    try:
        structured_model = chat_model.with_structured_output(ClarificationOutput)
        return structured_model.invoke([
            SystemMessage(content=CLARIFICATION_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ])
    except (NotImplementedError, Exception):
        pass

    # 降级：普通调用 + 文本解析
    try:
        response = chat_model.invoke([
            SystemMessage(content=CLARIFICATION_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ])
        content = response.content if hasattr(response, "content") else str(response)
        return _parse_clarification_from_text(str(content))
    except Exception:
        logger.warning("All clarification methods failed, using rule-based fallback")
        return None


def _parse_clarification_from_text(text: str) -> ClarificationOutput:
    """将 LLM 纯文本回复解析为 ClarificationOutput。"""
    return ClarificationOutput(
        extracted_destination="",
        extracted_days=None,
        extracted_budget="",
        extracted_audience=[],
        response_text=text,
        user_intent="providing_info",
        missing_info_hint=[],
    )


# ── 辅助函数 ───────────────────────────────────────────────


def _build_clarification_context(
    messages: list,
    user_message: str,
    merged_slots: dict[str, object],
) -> str:
    """构建澄清步骤的 LLM 提示词上下文。"""
    parts: list[str] = []

    # 最近对话历史（最后6条消息）
    recent = messages[-6:] if len(messages) > 6 else messages
    if recent:
        parts.append("## 对话历史")
        for msg in recent:
            role = "用户" if isinstance(msg, HumanMessage) else "助手"
            content = msg.content if hasattr(msg, "content") else str(msg)
            if isinstance(content, str) and len(content) > 200:
                content = content[:200] + "..."
            parts.append(f"{role}: {content}")

    # 当前已提取的槽位
    parts.append("")
    parts.append("## 已提取的信息")
    dest = merged_slots.get("clarified_destination", "")
    days = merged_slots.get("clarified_days")
    budget = merged_slots.get("clarified_budget", "")
    audience = merged_slots.get("clarified_audience", [])
    parts.append(f"- 目的地: {dest or '未提及'}")
    parts.append(f"- 天数: {days if days else '未提及'}")
    parts.append(f"- 预算: {budget or '未提及'}")
    parts.append(f"- 出行人员: {', '.join(audience) if audience else '未提及'}")

    parts.append("")
    parts.append("## 用户最新消息")
    parts.append(user_message)

    return "\n".join(parts)


def _rule_based_clarify(
    user_message: str,
    rule_slots: dict[str, object],
    merged_slots: dict[str, object],
) -> ClarificationOutput:
    """LLM 调用失败时的规则兜底澄清。"""
    dest = rule_slots.get("clarified_destination", "")
    days = rule_slots.get("clarified_days")
    audience = rule_slots.get("clarified_audience", [])

    missing: list[str] = []
    if not dest:
        missing.append("目的地")
    if not days:
        missing.append("游玩天数")
    if not audience:
        missing.append("出行人员")

    if missing:
        response = f"好的！还想再了解一下：{'、'.join(missing[:2])}是怎样的呢？"
    else:
        response = "收到！信息很完整，我这就为您规划行程~"

    return ClarificationOutput(
        extracted_destination=str(dest),
        extracted_days=int(days) if days else None,
        extracted_budget=str(rule_slots.get("clarified_budget", "")),
        extracted_audience=list(audience) if audience else [],
        response_text=response,
        user_intent="ready_to_plan" if not missing else "providing_info",
        missing_info_hint=missing,
    )




def _audience_to_text(audience: list[str]) -> str:
    """将出行人员列表转为自然中文描述。"""
    mapping: dict[str, str] = {
        "family_with_children": "亲子家庭出行",
        "elderly": "带老人出行",
        "couple": "情侣出行",
        "friends": "朋友结伴出行",
        "solo": "独自出行",
        "general": "通用出行",
    }
    labels = [mapping.get(a, a) for a in audience]
    return "、".join(labels) if labels else "通用出行"


def _format_plan_for_presentation(
    plan: dict[str, Any],
    budget: dict[str, Any] | None,
    crowd: dict[str, Any] | None,
    alternatives: dict[str, Any] | None,
) -> str:
    """将计划数据紧凑格式化，供展示 LLM 使用。"""
    lines: list[str] = [
        f"目的地: {plan.get('destination', '')}",
        f"天数: {plan.get('days', 0)}",
        f"摘要: {plan.get('summary', '')}",
        "",
        "每日行程:",
    ]

    for day in plan.get("day_plans", []):
        lines.append(f"  第{day['day']}天 — {day['title']}")
        for act in day.get("activities", []):
            lines.append(f"    - {act}")

    if budget:
        lines.append("")
        lines.append(f"预算 ({budget.get('budget_level', '')}):")
        lines.append(f"  住宿: {budget.get('accommodation', 0):.0f} CNY")
        lines.append(f"  餐饮: {budget.get('dining', 0):.0f} CNY")
        lines.append(f"  交通: {budget.get('transport', 0):.0f} CNY")
        lines.append(f"  门票: {budget.get('tickets', 0):.0f} CNY")
        lines.append(f"  总计: {budget.get('total', 0):.0f} CNY")
        lines.append(f"  日均: {budget.get('daily_average', 0):.0f} CNY")

    if crowd:
        lines.append("")
        lines.append(f"拥挤风险 ({crowd.get('overall_risk', 'unknown')}):")
        lines.append(f"  {crowd.get('advice', '')}")

    if alternatives:
        lines.append("")
        lines.append(f"天气/备选: {alternatives.get('weather_note', '')}")

    risk_notices = plan.get("risk_notices", [])
    if risk_notices:
        lines.append("")
        lines.append("风险提醒:")
        for notice in risk_notices[:3]:
            lines.append(f"  [{notice.get('severity', '')}] {notice.get('message', '')}")

    return "\n".join(lines)


def _build_fallback_presentation(
    plan: dict[str, Any],
    budget: dict[str, Any] | None,
    crowd: dict[str, Any] | None,
    alternatives: dict[str, Any] | None,
) -> str:
    """LLM 不可用时的兜底计划展示。"""
    dest = plan.get("destination", "")
    days = plan.get("days", 0)
    summary = plan.get("summary", "")

    lines: list[str] = [
        f"## {dest} {days}天旅行计划",
        "",
        summary,
        "",
        "### 每日行程",
    ]

    for day in plan.get("day_plans", []):
        lines.append(f"**第{day['day']}天 — {day['title']}**")
        for act in day.get("activities", [])[:4]:
            lines.append(f"  - {act}")
        lines.append("")

    if budget:
        lines.append(f"### 预算概览（{budget.get('budget_level', '')}）")
        lines.append(f"- 总计: {budget.get('total', 0):.0f} CNY，日均 {budget.get('daily_average', 0):.0f} CNY")

    if crowd:
        lines.append(f"### 拥挤风险")
        lines.append(f"- 整体风险: {crowd.get('overall_risk', '')}")
        advice = crowd.get("advice", "")
        if advice:
            lines.append(f"- 建议: {advice}")

    if alternatives:
        weather = alternatives.get("weather_note", "")
        if weather:
            lines.append(f"### 天气/备选")
            lines.append(f"- {weather}")

    lines.append("")
    lines.append("这个行程怎么样？需要调整的话随时告诉我~")

    return "\n".join(lines)
