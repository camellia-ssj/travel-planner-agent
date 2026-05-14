"""旅行智能体MVP的LangGraph图组装。"""

from __future__ import annotations

import warnings
from typing import Any

from travel_agent.agent.nodes import (
    EvidenceService,
    MemoryService,
    apply_feedback_node,
    generate_plan_node,
    generate_plan_with_planner_node,
    load_user_profile_node,
    parse_user_request_node,
    reflect_node,
    retrieve_evidence_node,
    save_trip_memory_node,
    tool_node,
    validate_plan_node,
)
from travel_agent.agent.planner import TravelPlanner
from travel_agent.agent.state import TravelAgentState

# langchain_core._api.deprecation 在首次导入时会在位置0注册一个
# 针对 LangChainPendingDeprecationWarning 的 ``simplefilter("default")``。
# 我们先触发该导入，然后在位置0插入我们自己的 "ignore"，
# 这样当真正的 langgraph 模块加载时它就会生效。
import langchain_core._api.deprecation  # noqa: E402,F401

warnings.simplefilter("ignore")

from langgraph.graph import END, START, StateGraph  # noqa: E402

# 恢复默认过滤器，防止我们的全局 "ignore" 泄露到无关代码中。
warnings.filters.pop(0)

DEFAULT_MAX_RETRIES = 1


def _after_reflect(
    state: TravelAgentState,
    max_retries: int = DEFAULT_MAX_RETRIES,
    has_memory: bool = False,
) -> str:
    """决定是重试（补充检索 + 重新规划）还是结束。

    当审查报告显示计划需要更多证据且尚未耗尽重试次数时，
    返回 ``"retry"``。否则返回相应的终止节点。
    """
    report = state.get("reflection_report")
    retry_count = state.get("reflection_retry_count", 0)

    if (
        report is not None
        and not report.passed
        and retry_count <= max_retries
    ):
        return "retry"

    return "save_trip_memory" if has_memory else "end"


def build_travel_agent_graph(
    rag_service: EvidenceService,
    planner: TravelPlanner | None = None,
    checkpointer: Any | None = None,
    memory_service: MemoryService | None = None,
    reflection_service: object | None = None,
    max_reflection_retries: int = DEFAULT_MAX_RETRIES,
) -> Any:
    """构建并编译确定性MVP旅行智能体图。

    提供 *memory_service* 时，图会在规划前加载用户画像，
    并在校验后将行程保存到长期记忆中。

    提供 *reflection_service*（一个 ``ReflectionService``）时，
    审查节点使用基于LLM的事实检查器，而非纯确定性文本重叠匹配。
    不提供时，节点仍可通过确定性回退方案正常工作。

    *max_reflection_retries* 控制当审查报告失败时，图在
    检索 → 工具 → 规划 → 校验 → 审查 之间循环重试的次数。
    默认值为 1（一次重试）。
    """

    graph = StateGraph(TravelAgentState)

    has_memory = memory_service is not None

    if has_memory:
        graph.add_node(
            "load_user_profile",
            lambda state: load_user_profile_node(state, memory_service),
        )
    graph.add_node("parse_user_request", parse_user_request_node)
    graph.add_node(
        "retrieve_evidence",
        lambda state: retrieve_evidence_node(state, rag_service),
    )
    graph.add_node("deterministic_tools", tool_node)
    if planner is None:
        graph.add_node("generate_plan", generate_plan_node)
    else:
        graph.add_node(
            "generate_plan",
            lambda state: generate_plan_with_planner_node(state, planner),
        )
    graph.add_node("validate_plan", validate_plan_node)
    graph.add_node(
        "reflect",
        lambda state: reflect_node(state, reflection_service=reflection_service),
    )
    if has_memory:
        graph.add_node(
            "save_trip_memory",
            lambda state: save_trip_memory_node(state, memory_service),
        )

    # ── 边 ──────────────────────────────────────────────────────────
    if has_memory:
        graph.add_edge(START, "load_user_profile")
        graph.add_edge("load_user_profile", "parse_user_request")
    else:
        graph.add_edge(START, "parse_user_request")
    graph.add_edge("parse_user_request", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "deterministic_tools")
    graph.add_edge("deterministic_tools", "generate_plan")
    graph.add_edge("generate_plan", "validate_plan")
    graph.add_edge("validate_plan", "reflect")

    # 条件边：审查 → 重试（循环）或继续
    graph.add_conditional_edges(
        "reflect",
        lambda state: _after_reflect(
            state, max_retries=max_reflection_retries, has_memory=has_memory
        ),
        {
            "retry": "retrieve_evidence",
            "save_trip_memory": "save_trip_memory" if has_memory else END,
            "end": END,
        },
    )
    if has_memory:
        graph.add_edge("save_trip_memory", END)

    return graph.compile(checkpointer=checkpointer)


def build_travel_agent_resume_graph(
    rag_service: EvidenceService | None = None,
    planner: TravelPlanner | None = None,
    checkpointer: Any | None = None,
    memory_service: MemoryService | None = None,
    reflection_service: object | None = None,
    max_reflection_retries: int = DEFAULT_MAX_RETRIES,
) -> Any:
    """构建并编译一个恢复检查点状态并重新规划的图。

    提供 *rag_service* 时，会在 ``apply_feedback`` 之后插入
    ``retrieve_evidence`` 节点，以便反馈中的目的地/天数变更
    触发新的RAG检索。

    关于 *reflection_service* 和 *max_reflection_retries* 的详细说明，
    请参见 ``build_travel_agent_graph``。
    当 *rag_service* 为 ``None`` 时，由于没有检索节点可供回退，审查重试将被禁用。
    """

    graph = StateGraph(TravelAgentState)

    has_memory = memory_service is not None

    # 当没有可回退的检索节点时，禁用重试。
    if rag_service is None:
        max_reflection_retries = 0

    if has_memory:
        graph.add_node(
            "load_user_profile",
            lambda state: load_user_profile_node(state, memory_service),
        )
    graph.add_node("apply_feedback", apply_feedback_node)
    if rag_service is not None:
        graph.add_node(
            "retrieve_evidence",
            lambda state: retrieve_evidence_node(state, rag_service),
        )
    graph.add_node("deterministic_tools", tool_node)
    if planner is None:
        graph.add_node("generate_plan", generate_plan_node)
    else:
        graph.add_node(
            "generate_plan",
            lambda state: generate_plan_with_planner_node(state, planner),
        )
    graph.add_node("validate_plan", validate_plan_node)
    graph.add_node(
        "reflect",
        lambda state: reflect_node(state, reflection_service=reflection_service),
    )
    if has_memory:
        graph.add_node(
            "save_trip_memory",
            lambda state: save_trip_memory_node(state, memory_service),
        )

    # ── 边 ──────────────────────────────────────────────────────────
    if has_memory:
        graph.add_edge(START, "load_user_profile")
        graph.add_edge("load_user_profile", "apply_feedback")
    else:
        graph.add_edge(START, "apply_feedback")
    if rag_service is not None:
        graph.add_edge("apply_feedback", "retrieve_evidence")
        graph.add_edge("retrieve_evidence", "deterministic_tools")
    else:
        graph.add_edge("apply_feedback", "deterministic_tools")
    graph.add_edge("deterministic_tools", "generate_plan")
    graph.add_edge("generate_plan", "validate_plan")
    graph.add_edge("validate_plan", "reflect")

    # 条件边：审查 → 重试（循环）或继续
    graph.add_conditional_edges(
        "reflect",
        lambda state: _after_reflect(
            state, max_retries=max_reflection_retries, has_memory=has_memory
        ),
        {
            "retry": "retrieve_evidence",
            "save_trip_memory": "save_trip_memory" if has_memory else END,
            "end": END,
        },
    )
    if has_memory:
        graph.add_edge("save_trip_memory", END)

    return graph.compile(checkpointer=checkpointer)
