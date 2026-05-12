"""Tests for the Memory long-term memory module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from travel_agent.memory import MemoryStore, TripRecord, UserProfile


def _build_trace(**overrides: object):
    from travel_agent.rag.models import RetrievalTrace

    kwargs: dict[str, object] = {
        "retrieval_mode": "hybrid",
        "requested_top_k": 5,
        "candidate_k": 10,
        "returned_results": 0,
        "empty_result": True,
        "destination": "",
        "section": "",
        "travel_type": "",
        "season": "",
        "embedding_provider": "local",
        "reranker": "keyword",
        "collection_version": "test",
        "metadata_filters": {},
        "vector_hits": [],
        "keyword_hits": [],
        "fused_hits": [],
        "reranked_hits": [],
    }
    kwargs.update(overrides)
    return RetrievalTrace.create(**kwargs)


def _make_evidence(question: str = "", results=None, trace=None, confidence: float = 0.5):
    from travel_agent.rag.models import EvidenceBundle

    return EvidenceBundle(
        question=question,
        results=results if results is not None else [],
        trace=trace if trace is not None else _build_trace(),
        query_analysis={},
        confidence=confidence,
    )


class TestUserProfile:
    def test_blank_profile_is_empty(self) -> None:
        profile = UserProfile(user_id="u1")
        assert profile.total_trips == 0
        assert profile.to_context_text() == ""

    def test_profile_with_trips_has_context(self) -> None:
        profile = UserProfile(
            user_id="u1",
            preferred_destinations=["Tokyo", "Paris"],
            budget_preference="premium",
            audience_types=["couple"],
            trip_length_avg=4.5,
            total_trips=3,
            preferences_summary="prefers luxury",
        )
        ctx = profile.to_context_text()
        assert "用户画像" in ctx
        assert "3 次历史行程" in ctx
        assert "Tokyo" in ctx
        assert "couple" in ctx


class TestMemoryStore:
    def test_create_and_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "test.sqlite")
            try:
                assert store.get_profile("u1").total_trips == 0
            finally:
                store.close()

    def test_save_trip_updates_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "test.sqlite")
            try:
                store.save_trip(
                    TripRecord(
                        memory_id="m1",
                        user_id="u1",
                        destination="Tokyo",
                        days=5,
                        audience=["family_with_children"],
                        budget_preference="standard",
                        plan_summary="Tokyo 5-day family trip",
                    )
                )
                profile = store.get_profile("u1")
                assert profile.total_trips == 1
                assert profile.last_destination == "Tokyo"
                assert "Tokyo" in profile.preferred_destinations
                assert profile.trip_length_avg == 5.0
            finally:
                store.close()

    def test_multiple_trips_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "test.sqlite")
            try:
                store.save_trip(
                    TripRecord(
                        memory_id="m1",
                        user_id="u1",
                        destination="Tokyo",
                        days=3,
                        audience=["couple"],
                        budget_preference="economy",
                    )
                )
                store.save_trip(
                    TripRecord(
                        memory_id="m2",
                        user_id="u1",
                        destination="Paris",
                        days=7,
                        audience=["couple"],
                        budget_preference="economy",
                    )
                )
                store.save_trip(
                    TripRecord(
                        memory_id="m3",
                        user_id="u1",
                        destination="Tokyo",
                        days=4,
                        audience=["couple", "friends"],
                        budget_preference="premium",
                    )
                )

                profile = store.get_profile("u1")
                assert profile.total_trips == 3
                assert profile.preferred_destinations[0] == "Tokyo"
                assert profile.budget_preference == "economy"
                assert "couple" in profile.audience_types
                assert profile.trip_length_avg == pytest.approx(14 / 3, 0.1)
                pref_text = profile.preferences_summary
                assert "Tokyo" in pref_text or "东京" in pref_text
            finally:
                store.close()

    def test_list_user_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "test.sqlite")
            try:
                store.save_trip(TripRecord(
                    memory_id="m1", user_id="u1", destination="Tokyo", days=3,
                ))
                store.save_trip(TripRecord(
                    memory_id="m2", user_id="u1", destination="Paris", days=5,
                ))
                trips = store.list_user_trips("u1")
                assert len(trips) == 2
                assert trips[0].destination == "Paris"
                assert trips[1].destination == "Tokyo"
            finally:
                store.close()

    def test_different_users_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "test.sqlite")
            try:
                store.save_trip(TripRecord(
                    memory_id="m1", user_id="u1", destination="Tokyo", days=3,
                ))
                store.save_trip(TripRecord(
                    memory_id="m2", user_id="u2", destination="Paris", days=5,
                ))
                p1 = store.get_profile("u1")
                p2 = store.get_profile("u2")
                assert p1.total_trips == 1
                assert p2.total_trips == 1
                assert p1.last_destination == "Tokyo"
                assert p2.last_destination == "Paris"
            finally:
                store.close()

    def test_trip_record_fields_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "test.sqlite")
            try:
                store.save_trip(
                    TripRecord(
                        memory_id="m1",
                        user_id="u1",
                        thread_id="th1",
                        destination="Hangzhou",
                        days=3,
                        audience=["family_with_children", "elderly"],
                        budget_preference="standard",
                        plan_summary="A nice trip",
                        user_feedback=["Day 2 too tiring"],
                    )
                )
                trips = store.list_user_trips("u1")
                assert len(trips) == 1
                t = trips[0]
                assert t.destination == "Hangzhou"
                assert t.days == 3
                assert "family_with_children" in t.audience
                assert "elderly" in t.audience
                assert t.budget_preference == "standard"
                assert t.plan_summary == "A nice trip"
                assert len(t.user_feedback) == 1
            finally:
                store.close()

    def test_empty_profile_for_unknown_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "test.sqlite")
            try:
                profile = store.get_profile("no_such_user")
                assert profile.total_trips == 0
                assert profile.preferred_destinations == []
                assert profile.to_context_text() == ""
            finally:
                store.close()

    def test_budget_preference_most_common(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "test.sqlite")
            try:
                for i, bp in enumerate(["economy", "premium", "economy", "standard", "economy"]):
                    store.save_trip(
                        TripRecord(
                            memory_id=f"m{i}", user_id="u1", destination="Tokyo", days=1,
                            budget_preference=bp,
                        )
                    )
                profile = store.get_profile("u1")
                assert profile.budget_preference == "economy"
            finally:
                store.close()


class TestMemoryGraphIntegration:
    """Tests that the graph integrates memory correctly."""

    def test_graph_with_memory_store_loads_profile(self) -> None:
        import uuid

        from travel_agent.agent.graph import build_travel_agent_graph

        user_id = f"test-{uuid.uuid4().hex[:8]}"
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            store = MemoryStore(tmpdir / "memory.sqlite")
            try:
                store.save_trip(
                    TripRecord(
                        memory_id="m1",
                        user_id=user_id,
                        destination="Tokyo",
                        days=5,
                        audience=["couple"],
                        budget_preference="premium",
                        plan_summary="Tokyo luxury trip",
                    )
                )

                class FakeRag:
                    def retrieve_evidence(self, query, **kwargs):
                        return _make_evidence(
                            question=query,
                            trace=_build_trace(
                                empty_result=True, returned_results=0,
                                destination="Tokyo", section="itinerary",
                            ),
                        )

                graph = build_travel_agent_graph(FakeRag(), memory_service=store)
                result = graph.invoke({
                    "question": "我和女朋友去东京5天高端游",
                    "user_id": user_id,
                    "thread_id": "th1",
                })

                plan = result["plan"]
                assert plan is not None
                assert plan.destination == "Tokyo"
                assert plan.days == 5

                profile = result.get("user_profile")
                assert profile is not None
                assert profile.total_trips >= 1
                assert "Tokyo" in profile.preferred_destinations

                updated = store.get_profile(user_id)
                assert updated.total_trips >= 2
            finally:
                store.close()

    def test_graph_without_memory_still_works(self) -> None:
        from travel_agent.agent.graph import build_travel_agent_graph

        class FakeRag:
            def retrieve_evidence(self, query, **kwargs):
                return _make_evidence(
                    question=query,
                    trace=_build_trace(
                        empty_result=True, returned_results=0,
                        destination="Beijing", section="itinerary",
                    ),
                )

        graph = build_travel_agent_graph(FakeRag())
        result = graph.invoke({"question": "北京3天"})
        assert result["plan"].destination == "Beijing"
        assert result["plan"].days == 3
        assert result.get("user_profile") is None

    def test_graph_no_user_id_skips_memory(self) -> None:
        from travel_agent.agent.graph import build_travel_agent_graph

        class FakeRag:
            def retrieve_evidence(self, query, **kwargs):
                return _make_evidence(
                    question=query,
                    trace=_build_trace(
                        empty_result=True, returned_results=0,
                        destination="Beijing", section="itinerary",
                    ),
                )

        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.sqlite")
            try:
                graph = build_travel_agent_graph(FakeRag(), memory_service=store)
                result = graph.invoke({"question": "北京3天", "thread_id": "th1"})
                assert result["plan"].destination == "Beijing"
                profile_after = store.get_profile("no_such_user")
                assert profile_after.total_trips == 0
            finally:
                store.close()

    def test_profile_used_for_preference_fallback(self) -> None:
        import uuid

        from travel_agent.agent.graph import build_travel_agent_graph
        from travel_agent.rag.models import SearchResult

        user_id = f"test-{uuid.uuid4().hex[:8]}"
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            store = MemoryStore(tmpdir / "memory.sqlite")
            try:
                store.save_trip(
                    TripRecord(
                        memory_id="m1",
                        user_id=user_id,
                        destination="Tokyo",
                        days=5,
                        audience=["couple"],
                        budget_preference="premium",
                        plan_summary="Tokyo luxury trip",
                    )
                )

                results = [
                    SearchResult(
                        content="杭州西湖美景，适合情侣漫步。雷峰塔门票40元，灵隐寺门票30元。",
                        source="hangzhou.md",
                        destination="Hangzhou",
                        score=0.9,
                        metadata={"section": "itinerary"},
                    ),
                    SearchResult(
                        content="杭州高端酒店推荐：四季酒店、安缦法云。",
                        source="hangzhou.md",
                        destination="Hangzhou",
                        score=0.85,
                        metadata={"section": "lodging"},
                    ),
                    SearchResult(
                        content="杭州美食：楼外楼、知味观。",
                        source="hangzhou.md",
                        destination="Hangzhou",
                        score=0.8,
                        metadata={"section": "dining"},
                    ),
                ]

                class FakeRag:
                    def retrieve_evidence(self, query, **kwargs):
                        return _make_evidence(
                            question=query,
                            results=results,
                            trace=_build_trace(
                                empty_result=False, returned_results=3,
                                destination="Hangzhou", section="itinerary",
                            ),
                            confidence=0.9,
                        )

                graph = build_travel_agent_graph(FakeRag(), memory_service=store)
                result = graph.invoke({
                    "question": "去杭州玩3天",
                    "user_id": user_id,
                    "thread_id": "th2",
                })

                request = result["request"]
                plan = result["plan"]
                assert plan.destination == "Hangzhou"
                assert plan.days == 3
                assert request.budget_preference == "premium"
            finally:
                store.close()

    def test_parse_with_explicit_budget_overrides_profile(self) -> None:
        import uuid

        from travel_agent.agent.graph import build_travel_agent_graph

        user_id = f"test-{uuid.uuid4().hex[:8]}"
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            store = MemoryStore(tmpdir / "memory.sqlite")
            try:
                store.save_trip(
                    TripRecord(
                        memory_id="m1",
                        user_id=user_id,
                        destination="Tokyo",
                        days=5,
                        budget_preference="premium",
                        audience=["couple"],
                    )
                )

                class FakeRag:
                    def retrieve_evidence(self, query, **kwargs):
                        return _make_evidence(
                            question=query,
                            trace=_build_trace(
                                empty_result=True, returned_results=0,
                                destination="Beijing", section="itinerary",
                            ),
                        )

                graph = build_travel_agent_graph(FakeRag(), memory_service=store)
                result = graph.invoke({
                    "question": "北京3天穷游",
                    "user_id": user_id,
                    "thread_id": "th3",
                })

                request = result["request"]
                assert request.budget_preference == "economy"
            finally:
                store.close()
