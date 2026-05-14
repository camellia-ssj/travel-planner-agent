"""LangGraph旅行智能体MVP的基于规则的节点。"""

from __future__ import annotations

import re
import uuid
from contextlib import suppress
from dataclasses import replace
from typing import Protocol

from travel_agent.agent.planner import RuleBasedTravelPlanner, TravelPlanner
from travel_agent.agent.schemas import (
    ReflectionReport,
    TravelPlan,
    TravelRequest,
)
from travel_agent.agent.state import TravelAgentState
from travel_agent.knowledge import (
    CHINESE_DAY_NUMBERS,
    DESTINATION_ALIASES,
    PEOPLE_IMPLICIT,
    WEEKEND_HOLIDAY_PATTERN,
)
from travel_agent.memory.models import TripRecord, UserProfile
from travel_agent.rag.models import EvidenceBundle, SearchResult
from travel_agent.tools.alternatives import suggest_alternatives
from travel_agent.tools.budget import estimate_budget
from travel_agent.tools.crowd import assess_crowd_risk


class EvidenceService(Protocol):
    """智能体检索节点所需的最小服务契约。"""

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
        """返回给定查询的结构化RAG证据。"""


class MemoryService(Protocol):
    """记忆节点所需的最小服务契约。"""

    def get_profile(self, user_id: str) -> UserProfile: ...

    def save_trip(self, record: TripRecord) -> None: ...


def parse_user_request_node(state: TravelAgentState) -> TravelAgentState:
    """将自然语言请求解析为结构化的旅行请求。

    当用户有存储的画像（来自长期记忆）时，画像偏好将作为
    查询中未明确提及字段的回退值。
    """

    question = state.get("question", "")
    parsed_destination = _parse_destination(question)
    parsed_days = _parse_days(question)
    destination_override = state.get("destination_override")
    days_override = state.get("days_override")
    destination = destination_override if destination_override is not None else parsed_destination
    days = days_override if days_override is not None else parsed_days
    parsed_audience = _parse_audience(question)
    parsed_budget = _parse_budget_preference(question)

    # 使用画像作为未指定字段的回退值
    profile = state.get("user_profile")
    audience = _resolve_audience(question, parsed_audience, profile)
    budget_preference = _resolve_budget(question, parsed_budget, profile)

    request = TravelRequest(
        raw_query=question,
        destination=destination,
        days=int(days),
        audience=audience,
        budget_preference=budget_preference,
    )
    return {
        "original_user_request": question,
        "request": request,
        "user_feedback": state.get("user_feedback", []),
    }


_CHANGE_INTENT_RE = re.compile(
    r"(?:"
    r"改(?:成|为|到|去)|换成|换到|变成|转到|调整到|修改到|换成去|"
    r"change\s+(?:the\s+)?(?:trip\s+|plan\s+|destination\s+|days\s+|budget\s+)?to|"
    r"switch\s+(?:the\s+)?(?:trip\s+|plan\s+|destination\s+|days\s+|budget\s+)?to|"
    r"update\s+(?:the\s+)?(?:trip\s+|plan\s+|destination\s+|days\s+|budget\s+)?to|"
    r"move\s+(?:the\s+)?(?:trip\s+|plan\s+|destination\s+)?to"
    r")",
    re.IGNORECASE,
)


def _has_change_intent(text: str) -> bool:
    """当 *text* 明确要求更改行程参数时返回 True。

    没有这个守卫，偶发性匹配（例如 "第二天少走路" 中
    "二"+"天" → 2）会错误地覆盖检查点中的请求。
    """
    return bool(_CHANGE_INTENT_RE.search(text))


def apply_feedback_node(state: TravelAgentState) -> TravelAgentState:
    """追加用户的后续反馈并重新解析参数变化。

    当用户在恢复请求中要求更改目的地、天数或预算时，此节点会更新
    ``request`` 和 ``question``，以便下游节点（重新检索、工具、规划器）
    基于新的意图来工作，而非使用过期的检查点值。

    解析值仅在反馈中包含明确的更改意图措辞（``改成``、``改为``、
    ``换成`` 等）时才生效，以避免像 ``第二天少走路`` 这样的误报
    将 ``days=3`` 覆盖为 2。
    """

    feedback = state.get("latest_user_feedback", "").strip()
    existing = list(state.get("user_feedback", []))
    if feedback:
        existing.append(feedback)

    result: dict[str, object] = {"user_feedback": existing}
    if not feedback:
        return result

    old_request = state.get("request")
    if old_request is None:
        return result

    if not _has_change_intent(feedback):
        return result

    new_destination = _parse_destination(feedback)
    new_days = _parse_days(feedback)
    new_budget = _parse_budget_preference(feedback)

    updated_fields: dict[str, object] = {}
    if new_destination and new_destination != old_request.destination:
        updated_fields["destination"] = new_destination
    if _days_explicitly_mentioned(feedback) and new_days >= 1 and new_days != old_request.days:
        updated_fields["days"] = max(1, int(new_days))
    if new_budget and new_budget != old_request.budget_preference:
        updated_fields["budget_preference"] = new_budget

    if updated_fields:
        combined_query = _resume_query_with_feedback(
            state,
            old_request=old_request,
            feedback=feedback,
        )
        updated_fields["raw_query"] = combined_query
        result["request"] = old_request.model_copy(update=updated_fields)
        # 在重新生成的检索查询中保留原始行程约束，
        # 这样像 "改目的地到北京" 这样的反馈不会丢失
        # 原始请求中的人数、周末或其他上下文信息。
        result["question"] = combined_query

    return result


def retrieve_evidence_node(
    state: TravelAgentState,
    rag_service: EvidenceService,
) -> TravelAgentState:
    """跨多个板块检索RAG证据，为每天的行程提供多样化内容。"""

    question = state.get("question", "")
    request = state.get("request")
    destination = request.destination if request and request.destination else None

    # 从多个板块获取证据，使规划器能够跨天变化内容。
    # 我们先请求行程相关板块，然后回退到更广泛的未过滤检索，
    # 这样查询中的预算/拥挤风险关键词不会使规划器缺少行程内容。
    evidence = rag_service.retrieve_evidence(
        question,
        destination=destination,
        top_k=10,
    )
    # 如果板块推断导致结果太少，则通过跳过板块过滤的
    # 更广泛检索来补充。
    itinerary_results = [
        r for r in evidence.results
        if str(r.metadata.get("section", "")) == "itinerary"
    ]
    if len(itinerary_results) < 3:
        supplementary = rag_service.retrieve_evidence(
            question,
            destination=destination,
            section="itinerary",
            top_k=8,
        )
        if supplementary.results:
            merged = list(evidence.results)
            seen = {_result_dedup_key(r) for r in merged}
            added = 0
            for r in supplementary.results:
                key = _result_dedup_key(r)
                if key not in seen:
                    seen.add(key)
                    merged.append(r)
                    added += 1
            if added:
                evidence = replace(
                    evidence,
                    results=merged,
                    trace=replace(
                        evidence.trace,
                        returned_results=len(merged),
                        reranked_hits=(
                            evidence.trace.reranked_hits
                            + supplementary.trace.reranked_hits
                        ),
                        average_score=_avg_score(merged),
                        empty_result=False,
                        total_latency_ms=round(
                            evidence.trace.total_latency_ms
                            + supplementary.trace.total_latency_ms,
                            3,
                        ),
                    ),
                    confidence=round(
                        max(evidence.confidence, supplementary.confidence), 4
                    ),
                )

    return {"evidence": evidence}


def tool_node(state: TravelAgentState) -> TravelAgentState:
    """运行确定性工具：预算、拥挤风险、备选方案。

    使用查询文本来检测周末/节假日关键词（用于拥挤风险评估），
    并提取明确的人数（用于预算估算）。
    """

    request = state.get("request")
    evidence = state.get("evidence")
    if request is None or evidence is None:
        return {
            "tool_budget": None,
            "tool_crowd_risk": None,
            "tool_alternatives": None,
        }

    question = state.get("question", request.raw_query if request else "")
    people_count = _parse_people_count(question, request.audience)
    is_weekend_holiday = _detect_weekend_holiday(question)

    budget = estimate_budget(
        people_count=people_count,
        days=request.days,
        budget_level=request.budget_preference,
        evidence=evidence,
    )
    crowd = assess_crowd_risk(
        destination=request.destination,
        evidence=evidence,
        is_weekend_holiday=is_weekend_holiday,
    )
    alternatives = suggest_alternatives(
        destination=request.destination,
        evidence=evidence,
        crowd_assessment=crowd,
    )
    return {
        "tool_budget": budget,
        "tool_crowd_risk": crowd,
        "tool_alternatives": alternatives,
    }


def generate_plan_node(state: TravelAgentState) -> TravelAgentState:
    """根据请求和证据生成确定性的结构化旅行计划。"""

    return generate_plan_with_planner_node(state, RuleBasedTravelPlanner())


def generate_plan_with_planner_node(
    state: TravelAgentState,
    planner: TravelPlanner,
) -> TravelAgentState:
    """使用注入的规划器生成结构化的旅行计划。"""

    request = state.get("request") or TravelRequest(raw_query=state.get("question", ""))
    evidence = state.get("evidence")
    if evidence is None:
        raise ValueError("generate_plan_node requires evidence in state")
    tool_results: dict[str, object] = {
        "tool_budget": state.get("tool_budget"),
        "tool_crowd_risk": state.get("tool_crowd_risk"),
        "tool_alternatives": state.get("tool_alternatives"),
    }
    user_profile = state.get("user_profile")
    plan = planner.plan(
        request, evidence,
        user_feedback=state.get("user_feedback", []),
        tool_results=tool_results,
        user_profile=user_profile,
    )
    return {"plan": plan}


def validate_plan_node(state: TravelAgentState) -> TravelAgentState:
    """验证生成的计划是否足够完整以供下游使用。"""

    plan = state.get("plan")
    errors: list[str] = []
    if plan is None:
        return {"is_valid": False, "validation_errors": ["missing plan"]}
    if not plan.destination:
        errors.append("missing destination")
    if plan.days <= 0:
        errors.append("days must be positive")
    if len(plan.day_plans) != plan.days:
        errors.append("daily plan count does not match requested days")
    for day_plan in plan.day_plans:
        if not day_plan.activities:
            errors.append(f"day {day_plan.day} has no activities")
    if not plan.risk_notices:
        errors.append("missing risk notices")
    return {"is_valid": not errors, "validation_errors": errors}


def _parse_destination(text: str) -> str:
    normalized = text.lower()
    for alias, destination in DESTINATION_ALIASES.items():
        if alias.lower() in normalized:
            return destination
    return ""


_DAYS_DIGIT_RE = re.compile(r"(\d+)\s*(?:天|日|days?|d)", re.IGNORECASE)
_DAYS_CHINESE_RE = re.compile(r"([一二两三四五六七八九十])\s*(?:天|日)")


def _days_explicitly_mentioned(text: str) -> bool:
    """当 *text* 包含明确的天数提及时返回 True。

    用于防止 ``_parse_days`` 默认为 1 时，在用户实际未提及天数的情况下
    触发虚假的天数变更。
    """
    return bool(_DAYS_DIGIT_RE.search(text) or _DAYS_CHINESE_RE.search(text))


def _parse_days(text: str) -> int:
    digit_match = _DAYS_DIGIT_RE.search(text)
    if digit_match:
        return max(1, int(digit_match.group(1)))
    chinese_match = _DAYS_CHINESE_RE.search(text)
    if chinese_match:
        return CHINESE_DAY_NUMBERS[chinese_match.group(1)]
    return 1


def _parse_audience(text: str) -> list[str]:
    normalized = text.lower()
    audience: list[str] = []
    if any(token in normalized for token in ("亲子", "孩子", "儿童", "family", "kids")):
        audience.append("family_with_children")
    if any(token in normalized for token in ("老人", "父母", "elderly", "senior")):
        audience.append("elderly")
    if any(token in normalized for token in ("情侣", "夫妻", "couple")):
        audience.append("couple")
    if any(token in normalized for token in ("朋友", "同学", "friends")):
        audience.append("friends")
    if any(token in normalized for token in ("独自", "一个人", "solo")):
        audience.append("solo")
    return audience or ["general"]


def _parse_budget_preference(text: str) -> str:
    normalized = text.lower()
    if any(
        token in normalized
        for token in ("省钱", "经济", "低预算", "穷游", "cheap", "economy", "budget-friendly")
    ):
        return "economy"
    if any(token in normalized for token in ("舒适", "高端", "豪华", "luxury", "premium")):
        return "premium"
    if any(
        token in normalized
        for token in ("适中", "中等", "standard", "mid", "mid-range", "moderate")
    ):
        return "standard"
    return "standard"


def _detect_weekend_holiday(text: str) -> bool:
    """如果查询包含周末或节假日关键词则返回 True。"""
    return bool(WEEKEND_HOLIDAY_PATTERN.search(text))


_PEOPLE_COUNT_PATTERNS = [
    (re.compile(r"([一-鿿]+)(\d+)\s*个?\s*人"), lambda m: int(m.group(2))),
    (re.compile(r"(\d+)\s*个?\s*人"), lambda m: int(m.group(1))),
    (re.compile(r"我们\s*(\d+)\s*个"), lambda m: int(m.group(1))),
    (re.compile(r"([\d]+)\s*(?:位|名|adults?|people|persons?)"), lambda m: int(m.group(1))),
    (re.compile(r"一家\s*([\d一二两三])\s*口"), lambda m: _cn_digit_to_int(m.group(1))),
    (re.compile(r"([一二两三四五六七八九十])\s*个?\s*人"), lambda m: _cn_digit_to_int(m.group(1))),
]


def _cn_digit_to_int(ch: str) -> int:
    return CHINESE_DAY_NUMBERS.get(ch, 1)


def _parse_people_count(text: str, audience: list[str]) -> int:
    """从查询文本中提取人数。回退到同行人员数量。"""
    if not text:
        return max(1, len(audience))

    for pattern, extractor in _PEOPLE_COUNT_PATTERNS:
        match = pattern.search(text)
        if match:
            return max(1, extractor(match))

    normalized = text.lower()
    for phrase, count in PEOPLE_IMPLICIT.items():
        if phrase in normalized:
            return count

    # 回退：基于同行类型 — "情侣" = 2，否则每种类型算 1 人
    if "couple" in audience or "情侣" in text or "夫妻" in text:
        return 2
    return max(1, len(audience))


def _resolve_audience(
    question: str, parsed: list[str], profile: UserProfile | None
) -> list[str]:
    """当查询中未提及同行人员时，使用画像中的同行人员作为回退值。"""
    if parsed and parsed != ["general"]:
        return parsed
    if profile is not None and profile.audience_types:
        return profile.audience_types
    return parsed


def _resolve_budget(
    question: str, parsed: str, profile: UserProfile | None
) -> str:
    """当未提及时，使用画像中的预算偏好作为回退值。"""
    if _budget_explicitly_mentioned(question):
        return parsed
    if profile is not None and profile.total_trips > 0:
        return profile.budget_preference
    return parsed


def _budget_explicitly_mentioned(text: str) -> bool:
    """当查询明确提及预算偏好时返回 True。"""
    normalized = text.lower()
    budget_keywords = [
        "省钱", "经济", "低预算", "穷游", "cheap", "economy", "budget-friendly",
        "舒适", "高端", "豪华", "luxury", "premium",
        "适中", "中等", "standard", "mid", "mid-range", "moderate",
        "预算", "费用", "budget",
    ]
    return any(kw in normalized for kw in budget_keywords)


def _result_dedup_key(result: SearchResult) -> str:
    """稳定的去重键，使补充检索不会添加重复结果。"""
    chunk_id = result.metadata.get("chunk_id")
    if isinstance(chunk_id, str) and chunk_id:
        return chunk_id
    # 回退：对内容前缀 + 来源进行哈希
    content_prefix = result.content[:80]
    return f"{result.source}:{content_prefix}"


def _resume_query_with_feedback(
    state: TravelAgentState,
    old_request: TravelRequest,
    feedback: str,
) -> str:
    original_query = str(
        state.get("original_user_request")
        or old_request.raw_query
        or state.get("question", "")
    ).strip()
    if not original_query:
        return feedback
    return (
        f"{original_query}\n"
        f"Follow-up change request: {feedback}"
    )


def load_user_profile_node(
    state: TravelAgentState,
    memory_service: MemoryService,
) -> TravelAgentState:
    """从长期记忆中加载用户画像到图状态中。

    当状态中的 ``user_id`` 未设置时不执行任何操作。画像随后会
    传递给规划器，以便个性化推荐。
    """
    user_id = state.get("user_id", "").strip()
    if not user_id:
        return {}
    try:
        profile = memory_service.get_profile(user_id)
    except Exception:
        return {}
    return {"user_profile": profile}


def save_trip_memory_node(
    state: TravelAgentState,
    memory_service: MemoryService,
) -> TravelAgentState:
    """将已完成的行程持久化到长期记忆中并重建画像。

    当 ``user_id`` 未设置或状态中不存在计划时不执行任何操作。
    """
    user_id = state.get("user_id", "").strip()
    if not user_id:
        return {}

    request = state.get("request")
    plan: TravelPlan | None = state.get("plan")  # type: ignore[assignment]
    if request is None or plan is None:
        return {}

    thread_id = str(state.get("thread_id", ""))
    memory_id = uuid.uuid4().hex

    record = TripRecord(
        memory_id=memory_id,
        user_id=user_id,
        thread_id=thread_id,
        destination=plan.destination or request.destination,
        days=request.days,
        audience=request.audience,
        budget_preference=request.budget_preference,
        plan_summary=plan.summary,
        user_feedback=list(state.get("user_feedback", [])),
    )
    with suppress(Exception):
        memory_service.save_trip(record)

    # 返回更新后的画像，供下游使用
    try:
        updated_profile = memory_service.get_profile(user_id)
        return {"user_profile": updated_profile}
    except Exception:
        return {}


def _avg_score(results: list[SearchResult]) -> float:
    """合并结果列表的平均分数。"""
    if not results:
        return 0.0
    return round(sum(r.score for r in results) / len(results), 4)


# ---------------------------------------------------------------------------
# 审查 / 事实性校验节点
# ---------------------------------------------------------------------------


def reflect_node(
    state: TravelAgentState,
    reflection_service: object | None = None,
) -> TravelAgentState:
    """将生成的计划与RAG证据进行对比审查，检查事实一致性。

    提供 *reflection_service*（一个 ``ReflectionService``）时，
    审查使用LLM结构化输出调用作为主要检查器，确定性文本重叠作为回退。
    不提供时，审查完全以确定性方式运行（无需API密钥）。

    生成一个 ``ReflectionReport``，包含幻觉标记、证据覆盖率、
    置信度分数和可操作的建议。
    """

    plan = state.get("plan")
    evidence = state.get("evidence")

    if plan is None:
        return {
            "reflection_report": ReflectionReport(
                issues=["No plan to review"],
                passed=False,
            ),
            "reflection_issues": ["No plan to review"],
            "reflection_retry_count": state.get("reflection_retry_count", 0),
        }

    tool_results: dict[str, object] = {
        "tool_budget": state.get("tool_budget"),
        "tool_crowd_risk": state.get("tool_crowd_risk"),
        "tool_alternatives": state.get("tool_alternatives"),
    }

    if reflection_service is not None and hasattr(reflection_service, "reflect"):
        report = reflection_service.reflect(plan, evidence, tool_results)  # type: ignore[union-attr]
    else:
        from travel_agent.agent.reflection import deterministic_reflect

        report = deterministic_reflect(plan, evidence, tool_results)

    retry_count = state.get("reflection_retry_count", 0)
    if not report.passed:
        retry_count += 1
        report.issues.insert(
            0,
            f"Reflection attempt {retry_count}: plan did not pass factuality review.",
        )

    return {
        "reflection_report": report,
        "reflection_issues": report.issues,
        "reflection_retry_count": retry_count,
    }
