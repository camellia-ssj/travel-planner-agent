"""LangGraph assembly for the travel agent MVP."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from travel_agent.agent.nodes import (
    EvidenceService,
    apply_feedback_node,
    generate_plan_node,
    generate_plan_with_planner_node,
    parse_user_request_node,
    retrieve_evidence_node,
    validate_plan_node,
)
from travel_agent.agent.planner import TravelPlanner
from travel_agent.agent.state import TravelAgentState


def build_travel_agent_graph(
    rag_service: EvidenceService,
    planner: TravelPlanner | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Build and compile the deterministic MVP travel agent graph."""

    graph = StateGraph(TravelAgentState)
    graph.add_node("parse_user_request", parse_user_request_node)
    graph.add_node(
        "retrieve_evidence",
        lambda state: retrieve_evidence_node(state, rag_service),
    )
    if planner is None:
        graph.add_node("generate_plan", generate_plan_node)
    else:
        graph.add_node(
            "generate_plan",
            lambda state: generate_plan_with_planner_node(state, planner),
        )
    graph.add_node("validate_plan", validate_plan_node)

    graph.add_edge(START, "parse_user_request")
    graph.add_edge("parse_user_request", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "generate_plan")
    graph.add_edge("generate_plan", "validate_plan")
    graph.add_edge("validate_plan", END)
    return graph.compile(checkpointer=checkpointer)


def build_travel_agent_resume_graph(
    planner: TravelPlanner | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Build and compile a graph that resumes checkpointed state and replans."""

    graph = StateGraph(TravelAgentState)
    graph.add_node("apply_feedback", apply_feedback_node)
    if planner is None:
        graph.add_node("generate_plan", generate_plan_node)
    else:
        graph.add_node(
            "generate_plan",
            lambda state: generate_plan_with_planner_node(state, planner),
        )
    graph.add_node("validate_plan", validate_plan_node)

    graph.add_edge(START, "apply_feedback")
    graph.add_edge("apply_feedback", "generate_plan")
    graph.add_edge("generate_plan", "validate_plan")
    graph.add_edge("validate_plan", END)
    return graph.compile(checkpointer=checkpointer)
