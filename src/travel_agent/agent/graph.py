"""LangGraph assembly for the travel agent MVP."""

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

# langchain_core._api.deprecation registers a ``simplefilter("default")``
# for LangChainPendingDeprecationWarning at position 0 during its first
# import.  We trigger that import first, then insert our own "ignore" at
# position 0 so it wins when the actual langgraph modules load.
import langchain_core._api.deprecation  # noqa: E402,F401

warnings.simplefilter("ignore")

from langgraph.graph import END, START, StateGraph  # noqa: E402

# Restore the default filter so our blanket "ignore" doesn't leak to
# unrelated code.
warnings.filters.pop(0)

DEFAULT_MAX_RETRIES = 1


def _after_reflect(
    state: TravelAgentState,
    max_retries: int = DEFAULT_MAX_RETRIES,
    has_memory: bool = False,
) -> str:
    """Decide whether to retry (supplementary retrieval + re-plan) or finish.

    Returns ``"retry"`` when the reflection report indicates the plan
    needs more evidence and we haven't exhausted the retry budget.
    Otherwise returns the appropriate terminal node.
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
    """Build and compile the deterministic MVP travel agent graph.

    When *memory_service* is provided, the graph loads the user profile
    before planning and saves the trip to long-term memory after validation.

    When *reflection_service* (a ``ReflectionService``) is provided, the
    reflection node uses an LLM-based fact-checker instead of pure
    deterministic text-overlap.  Without it the node still works with
    deterministic fallback.

    *max_reflection_retries* controls how many times the graph will loop
    back through retrieval → tools → plan → validate → reflect when the
    reflection report fails.  Default is 1 (one retry).
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

    # ── edges ──────────────────────────────────────────────────────────
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

    # Conditional edge: reflect → retry (loop) or proceed
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
    """Build and compile a graph that resumes checkpointed state and replans.

    When *rag_service* is provided a ``retrieve_evidence`` node is inserted
    after ``apply_feedback`` so that destination / day changes in the
    feedback trigger fresh RAG retrieval.

    See ``build_travel_agent_graph`` for details on *reflection_service*
    and *max_reflection_retries*.  When *rag_service* is ``None``,
    reflection retries are disabled since no retrieval node exists.
    """

    graph = StateGraph(TravelAgentState)

    has_memory = memory_service is not None

    # Disable retry when there is no retrieval node to loop back to.
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

    # ── edges ──────────────────────────────────────────────────────────
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

    # Conditional edge: reflect → retry (loop) or proceed
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
