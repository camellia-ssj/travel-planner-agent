"""确定性工具函数的单元测试。"""

from __future__ import annotations

import pytest

from travel_agent.agent.nodes import (
    _detect_weekend_holiday,
    _parse_people_count,
    tool_node,
)
from travel_agent.agent.schemas import (
    AlternativePlan,
    BudgetEstimate,
    CrowdRiskAssessment,
    TravelRequest,
)
from travel_agent.rag.models import EvidenceBundle, RetrievalTrace, SearchResult
from travel_agent.tools.alternatives import suggest_alternatives
from travel_agent.tools.budget import estimate_budget
from travel_agent.tools.crowd import assess_crowd_risk

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _trace() -> RetrievalTrace:
    return RetrievalTrace.create(
        retrieval_mode="hybrid",
        requested_top_k=5,
        candidate_k=10,
        returned_results=3,
        empty_result=False,
        destination="Hangzhou",
        section="",
        travel_type="",
        season="",
        embedding_provider="local",
        reranker="keyword",
        collection_version="test",
        metadata_filters={},
        vector_hits=[],
        keyword_hits=[],
        fused_hits=[],
        reranked_hits=[],
    )


def _evidence_with_sections(sections: dict[str, str]) -> EvidenceBundle:
    """构建一个每个分类键对应一条结果的 EvidenceBundle。"""
    results = []
    for section, content in sections.items():
        results.append(SearchResult(
            content=content,
            source=f"{section}.md",
            destination="Hangzhou",
            score=0.8,
            metadata={"section": section},
        ))
    return EvidenceBundle(
        question="test",
        results=results,
        trace=_trace(),
        query_analysis={"destination": "Hangzhou"},
        confidence=0.85,
    )


def _empty_evidence() -> EvidenceBundle:
    return EvidenceBundle(
        question="test",
        results=[],
        trace=_trace(),
        query_analysis={},
        confidence=0.0,
    )


def _crowd_evidence() -> EvidenceBundle:
    return _evidence_with_sections({
        "crowd_risk": "节假日西湖断桥和灵隐寺拥挤风险较高，高峰时段排队人多。",
        "itinerary": "Day 1: 西湖湖滨、断桥、白堤。",
    })


def _full_evidence() -> EvidenceBundle:
    return _evidence_with_sections({
        "budget": "经济型旅行每日餐饮约80到150元，门票约120元/人。",
        "crowd_risk": "节假日西湖断桥和灵隐寺拥挤风险较高，高峰时段排队人多。",
        "alternatives": "若西湖人流过高，可改走浴鹄湾、茅家埠。",
        "weather_risk": "春季和梅雨季雨水较多，需要准备防滑鞋和雨具。",
        "itinerary": "第一天安排西湖湖滨、断桥、白堤。",
    })


# ---------------------------------------------------------------------------
# 预算工具测试
# ---------------------------------------------------------------------------


class TestBudgetTool:
    def test_economy_1person_1day(self) -> None:
        ev = _empty_evidence()
        result = estimate_budget(people_count=1, days=1, budget_level="economy", evidence=ev)
        assert result.budget_level == "economy"
        assert result.total == pytest.approx(300.0)
        assert result.daily_average == pytest.approx(300.0)
        assert result.accommodation == pytest.approx(120.0)
        assert result.dining == pytest.approx(75.0)
        assert result.transport == pytest.approx(45.0)
        assert result.tickets == pytest.approx(60.0)

    def test_standard_2people_3days(self) -> None:
        ev = _empty_evidence()
        result = estimate_budget(people_count=2, days=3, budget_level="standard", evidence=ev)
        assert result.budget_level == "standard"
        assert result.total == pytest.approx(600.0 * 2 * 3)
        assert result.daily_average == pytest.approx(600.0 * 2)

    def test_premium_has_higher_costs(self) -> None:
        ev = _empty_evidence()
        economy = estimate_budget(1, 1, "economy", ev)
        premium = estimate_budget(1, 1, "premium", ev)
        assert premium.total > economy.total
        assert premium.accommodation > economy.accommodation

    def test_extracts_price_hints_from_evidence(self) -> None:
        ev = _evidence_with_sections({
            "budget": "门票约300元/人，餐饮每天约200元。",
        })
        result = estimate_budget(1, 1, "standard", ev)
        assert any(
            "上调" in note or "下调" in note or "证据" in note
            for note in result.notes
        )

    def test_zero_days_clamps_to_minimum(self) -> None:
        ev = _empty_evidence()
        result = estimate_budget(1, 0, "standard", ev)
        assert result.daily_average == result.total
        assert result.daily_average > 0

    def test_unknown_level_falls_back_to_standard(self) -> None:
        ev = _empty_evidence()
        result = estimate_budget(1, 1, "unknown_level", ev)
        assert result.budget_level == "standard"
        assert result.total == pytest.approx(600.0)


# ---------------------------------------------------------------------------
# 拥挤风险工具测试
# ---------------------------------------------------------------------------


class TestCrowdRiskTool:
    def test_no_crowd_evidence_returns_low(self) -> None:
        ev = _evidence_with_sections({"itinerary": "Day 1: 西湖"})
        result = assess_crowd_risk("Hangzhou", ev)
        assert result.overall_risk == "low"
        assert result.poi_risks == []

    def test_extract_poi_from_content(self) -> None:
        result = assess_crowd_risk("Hangzhou", _crowd_evidence())
        poi_names = [p.poi_name for p in result.poi_risks]
        assert "断桥" in poi_names or any("断桥" in name for name in poi_names)
        assert "灵隐寺" in poi_names

    def test_weekend_promotes_risk(self) -> None:
        weekday = assess_crowd_risk("Hangzhou", _crowd_evidence(), is_weekend_holiday=False)
        weekend = assess_crowd_risk("Hangzhou", _crowd_evidence(), is_weekend_holiday=True)
        for w_poi in weekend.poi_risks:
            matching = [d for d in weekday.poi_risks if d.poi_name == w_poi.poi_name]
            if matching:
                from travel_agent.tools.crowd import _RISK_RANK
                assert _RISK_RANK[w_poi.risk_level] >= _RISK_RANK[matching[0].risk_level]

    def test_high_weekend_overall_high(self) -> None:
        result = assess_crowd_risk("Hangzhou", _crowd_evidence(), is_weekend_holiday=True)
        assert result.overall_risk == "high"

    def test_advice_matches_level(self) -> None:
        result = assess_crowd_risk("Hangzhou", _crowd_evidence(), is_weekend_holiday=True)
        assert result.advice
        assert len(result.advice) > 5


# ---------------------------------------------------------------------------
# 备选方案工具测试
# ---------------------------------------------------------------------------


class TestAlternativeTool:
    def test_from_evidence_alternatives_section(self) -> None:
        ev = _evidence_with_sections({
            "alternatives": "若西湖人流过高，可改走浴鹄湾和茅家埠。",
        })
        result = suggest_alternatives("Hangzhou", ev)
        assert len(result.alternatives) >= 1
        assert "浴鹄湾" in result.alternatives[0].suggested_alternative

    def test_from_weather_risk_section(self) -> None:
        ev = _evidence_with_sections({
            "weather_risk": "梅雨季需要准备雨具。",
        })
        result = suggest_alternatives("Hangzhou", ev)
        assert result.weather_note
        assert "雨具" in result.weather_note or any(
            "雨" in a.suggested_alternative for a in result.alternatives
        )

    def test_cross_references_crowd_high_risk_pois(self) -> None:
        ev = _evidence_with_sections({
            "crowd_risk": "灵隐寺拥挤风险较高。",
            "alternatives": "可改走浴鹄湾。",
        })
        crowd = assess_crowd_risk("Hangzhou", ev)
        result = suggest_alternatives("Hangzhou", ev, crowd_assessment=crowd)
        assert len(result.alternatives) >= 1

    def test_fallback_when_no_evidence(self) -> None:
        result = suggest_alternatives("Hangzhou", _empty_evidence())
        assert len(result.alternatives) >= 1
        assert result.weather_note

    def test_cap_at_five(self) -> None:
        sections = {f"alternatives_{i}": f"备选方案{i}: 地点{i}" for i in range(10)}
        ev = _evidence_with_sections(sections)
        result = suggest_alternatives("Hangzhou", ev)
        assert len(result.alternatives) <= 5


# ---------------------------------------------------------------------------
# tool_node 集成测试
# ---------------------------------------------------------------------------


class TestToolNode:
    def test_sets_all_three_results(self) -> None:
        state = {
            "request": TravelRequest(
                raw_query="杭州三天",
                destination="Hangzhou",
                days=3,
                audience=["family_with_children"],
                budget_preference="standard",
            ),
            "evidence": _full_evidence(),
        }
        result = tool_node(state)
        assert result["tool_budget"] is not None
        assert result["tool_crowd_risk"] is not None
        assert result["tool_alternatives"] is not None
        assert isinstance(result["tool_budget"], BudgetEstimate)
        assert isinstance(result["tool_crowd_risk"], CrowdRiskAssessment)
        assert isinstance(result["tool_alternatives"], AlternativePlan)

    def test_returns_none_when_request_missing(self) -> None:
        result = tool_node({"evidence": _empty_evidence()})
        assert result["tool_budget"] is None
        assert result["tool_crowd_risk"] is None
        assert result["tool_alternatives"] is None

    def test_returns_none_when_evidence_missing(self) -> None:
        result = tool_node({
            "request": TravelRequest(raw_query="test", destination="Hangzhou"),
        })
        assert result["tool_budget"] is None

    def test_budget_people_count_from_audience(self) -> None:
        state = {
            "request": TravelRequest(
                raw_query="杭州",
                destination="Hangzhou",
                days=2,
                audience=["family_with_children", "elderly"],
            ),
            "evidence": _empty_evidence(),
        }
        result = tool_node(state)
        budget = result["tool_budget"]
        assert budget is not None
        # 2 人 * 2 天 * 600 标准 = 2400
        assert budget.total == pytest.approx(2400.0)

    def test_weekend_keyword_sets_crowd_flag(self) -> None:
        state = {
            "question": "杭州周末拥挤吗？",
            "request": TravelRequest(
                raw_query="杭州周末拥挤吗？",
                destination="Hangzhou",
                days=2,
                audience=["general"],
            ),
            "evidence": _crowd_evidence(),
        }
        result = tool_node(state)
        crowd = result["tool_crowd_risk"]
        assert crowd is not None
        assert crowd.is_weekend_holiday is True

    def test_holiday_keyword_sets_crowd_flag(self) -> None:
        state = {
            "question": "国庆去杭州怎么玩",
            "request": TravelRequest(
                raw_query="国庆去杭州怎么玩",
                destination="Hangzhou",
                days=3,
                audience=["general"],
            ),
            "evidence": _crowd_evidence(),
        }
        result = tool_node(state)
        crowd = result["tool_crowd_risk"]
        assert crowd is not None
        assert crowd.is_weekend_holiday is True

    def test_no_weekend_keyword_leaves_flag_false(self) -> None:
        state = {
            "question": "杭州工作日拥挤吗",
            "request": TravelRequest(
                raw_query="杭州工作日拥挤吗",
                destination="Hangzhou",
                days=2,
                audience=["general"],
            ),
            "evidence": _crowd_evidence(),
        }
        result = tool_node(state)
        crowd = result["tool_crowd_risk"]
        assert crowd is not None
        assert crowd.is_weekend_holiday is False

    def test_explicit_people_count_from_query(self) -> None:
        state = {
            "question": "我们三个人去杭州",
            "request": TravelRequest(
                raw_query="我们三个人去杭州",
                destination="Hangzhou",
                days=2,
                audience=["general"],
            ),
            "evidence": _empty_evidence(),
        }
        result = tool_node(state)
        budget = result["tool_budget"]
        assert budget is not None
        # 3 人 * 2 天 * 600 = 3600
        assert budget.total == pytest.approx(3600.0)

    def test_couple_implicit_count(self) -> None:
        state = {
            "question": "夫妻二人去杭州蜜月",
            "request": TravelRequest(
                raw_query="夫妻二人去杭州蜜月",
                destination="Hangzhou",
                days=3,
                audience=["couple"],
            ),
            "evidence": _empty_evidence(),
        }
        result = tool_node(state)
        budget = result["tool_budget"]
        assert budget is not None
        assert budget.total == pytest.approx(3600.0)  # 2 * 3 * 600

    def test_question_from_state_falls_back_to_request_raw_query(self) -> None:
        state = {
            "request": TravelRequest(
                raw_query="春节去大理",
                destination="Hangzhou",
                days=2,
                audience=["general"],
            ),
            "evidence": _crowd_evidence(),
        }
        result = tool_node(state)
        crowd = result["tool_crowd_risk"]
        assert crowd is not None
        assert crowd.is_weekend_holiday is True


# ---------------------------------------------------------------------------
# 周末 / 节假日关键词检测
# ---------------------------------------------------------------------------


class TestWeekendHolidayDetection:
    @pytest.mark.parametrize("text", [
        "周末去杭州",
        "国庆去北京",
        "五一去玩",
        "端午节人多吗",
        "小长假去哪儿",
        "春节假期安排",
        "暑假亲子游",
        "weekend trip to Hangzhou",
        "national day holiday",
        "golden week travel",
    ])
    def test_detects_positive(self, text: str) -> None:
        assert _detect_weekend_holiday(text) is True

    @pytest.mark.parametrize("text", [
        "工作日去杭州",
        "周三下午",
        "周一开会",
        "一般日子",
        "weekday trip",
        "",
    ])
    def test_detects_negative(self, text: str) -> None:
        assert _detect_weekend_holiday(text) is False


# ---------------------------------------------------------------------------
# 人数解析
# ---------------------------------------------------------------------------


class TestPeopleCountParsing:
    @pytest.mark.parametrize("text, audience, expected", [
        ("我们3个人去杭州", [], 3),
        ("2人出行", [], 2),
        ("五个人一起去", [], 5),
        ("一家三口去杭州", [], 3),
        ("我们俩去苏州", [], 2),
        ("独自旅行", [], 1),
        ("我和父母去杭州", [], 3),
        ("我们八个人", [], 8),
        ("4 adults", [], 4),
        ("solo trip", [], 1),
    ])
    def test_explicit_counts(self, text: str, audience: list[str], expected: int) -> None:
        assert _parse_people_count(text, audience) == expected

    def test_couple_audience_fallback(self) -> None:
        assert _parse_people_count("去杭州", ["couple"]) == 2

    def test_single_audience_fallback(self) -> None:
        assert _parse_people_count("去杭州", ["solo"]) == 1

    def test_empty_text_uses_audience_len(self) -> None:
        assert _parse_people_count("", ["family_with_children", "elderly"]) == 2

    def test_empty_text_empty_audience_returns_one(self) -> None:
        assert _parse_people_count("", []) == 1
