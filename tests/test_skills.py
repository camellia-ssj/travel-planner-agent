from __future__ import annotations

from pathlib import Path

from travel_agent.agent.graph import build_travel_agent_graph
from travel_agent.agent.schemas import TravelRequest
from travel_agent.rag.models import EvidenceBundle, RetrievalTrace, SearchResult
from travel_agent.skills.registry import SkillRegistry


class SkillAwareMockRagService:
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
                "destination": destination,
                "section": section,
                "top_k": top_k,
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
                    content="If rain affects West Lake, use the Grand Canal area backup.",
                    source="hangzhou.md",
                    destination="Hangzhou",
                    score=0.7,
                    metadata={"section": "alternatives"},
                ),
            ],
            trace=RetrievalTrace.create(
                retrieval_mode="hybrid",
                requested_top_k=top_k or 5,
                candidate_k=3,
                returned_results=3,
                empty_result=False,
                destination=destination or "Hangzhou",
                section=section or "",
                travel_type=travel_type or "",
                season=season or "",
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


def test_project_skill_registry_loads_anthropic_style_skills() -> None:
    registry = SkillRegistry.from_project(Path.cwd())

    skills = registry.list_skills()
    names = {skill.name for skill in skills}

    assert "family-travel-planner" in names
    assert "budget-aware-planner" in names
    assert "crowd-avoidance-planner" in names
    assert all(skill.description for skill in skills)
    assert all(skill.path.endswith("SKILL.md") for skill in skills)


def test_skill_registry_selects_behavior_skills_from_request() -> None:
    registry = SkillRegistry.from_project(Path.cwd())
    request = TravelRequest(
        raw_query="带父母和孩子五一去杭州三天，预算别太贵，还担心下雨和人多",
        destination="Hangzhou",
        days=3,
        audience=["family_with_children", "elderly"],
        budget_preference="economy",
    )

    selection = registry.select_skills(request, request.raw_query)

    assert "family-travel-planner" in selection.names
    assert "budget-aware-planner" in selection.names
    assert "crowd-avoidance-planner" in selection.names
    assert selection.required_tools()


def test_agent_graph_injects_selected_skills_into_retrieval_and_state() -> None:
    registry = SkillRegistry.from_project(Path.cwd())
    rag_service = SkillAwareMockRagService()
    graph = build_travel_agent_graph(rag_service, skill_registry=registry)

    result = graph.invoke({"question": "带父母杭州三天，预算别太贵，周末避开人多"})

    selection = result["active_skills"]
    assert "family-travel-planner" in selection.names
    assert "budget-aware-planner" in selection.names
    assert result["tool_policy"]["active_skills"] == selection.names
    assert "Skill retrieval focus" in str(rag_service.calls[0]["query"])
    assert "Active skills:" in result["plan"].summary
