"""Unit tests for the centralized travel knowledge module."""

from __future__ import annotations

from travel_agent.knowledge import (
    BUILTIN_CHINESE_TERMS,
    CHINESE_DAY_NUMBERS,
    DESTINATION_ALIASES,
    DESTINATION_DISPLAY_NAMES,
    KEYWORD_ANSWER_TOKENS,
    PEOPLE_IMPLICIT,
    SECTION_QUERY_ALIASES,
    WEEKEND_HOLIDAY_KEYWORDS,
    WEEKEND_HOLIDAY_PATTERN,
)


class TestDestinationAliases:
    def test_all_destinations_have_display_names(self) -> None:
        canonical = set(DESTINATION_ALIASES.values())
        assert canonical == set(DESTINATION_DISPLAY_NAMES.keys()), (
            "Every canonical destination must have a display name"
        )

    def test_chinese_alias_maps_to_english(self) -> None:
        assert DESTINATION_ALIASES["杭州"] == "Hangzhou"
        assert DESTINATION_ALIASES["东京"] == "Tokyo"
        assert DESTINATION_ALIASES["北京"] == "Beijing"
        assert DESTINATION_ALIASES["巴黎"] == "Paris"

    def test_lowercase_alias_maps_to_canonical(self) -> None:
        assert DESTINATION_ALIASES["hangzhou"] == "Hangzhou"
        assert DESTINATION_ALIASES["tokyo"] == "Tokyo"
        assert DESTINATION_ALIASES["paris"] == "Paris"

    def test_display_name_round_trip(self) -> None:
        for alias, canonical in DESTINATION_ALIASES.items():
            if canonical in DESTINATION_DISPLAY_NAMES:
                assert DESTINATION_DISPLAY_NAMES[canonical] in DESTINATION_ALIASES


class TestSectionQueryAliases:
    def test_every_section_has_aliases(self) -> None:
        expected = {
            "traffic", "budget", "lodging", "dining",
            "crowd_risk", "weather_risk", "itinerary", "audience", "alternatives",
        }
        assert set(SECTION_QUERY_ALIASES.keys()) == expected

    def test_section_aliases_are_non_empty_tuples(self) -> None:
        for section, aliases in SECTION_QUERY_ALIASES.items():
            assert isinstance(aliases, tuple), f"{section} aliases must be a tuple"
            assert len(aliases) > 0, f"{section} aliases must not be empty"


class TestChineseDayNumbers:
    def test_basic_mappings(self) -> None:
        assert CHINESE_DAY_NUMBERS["一"] == 1
        assert CHINESE_DAY_NUMBERS["两"] == 2
        assert CHINESE_DAY_NUMBERS["三"] == 3
        assert CHINESE_DAY_NUMBERS["十"] == 10

    def test_one_to_ten_coverage(self) -> None:
        for i in range(1, 11):
            values = list(CHINESE_DAY_NUMBERS.values())
            assert i in values, f"Missing day number {i}"


class TestWeekendHolidayDetection:
    def test_pattern_matches_chinese_weekend(self) -> None:
        assert WEEKEND_HOLIDAY_PATTERN.search("杭州周末拥挤吗")
        assert WEEKEND_HOLIDAY_PATTERN.search("国庆节去北京")

    def test_pattern_matches_english(self) -> None:
        assert WEEKEND_HOLIDAY_PATTERN.search("weekend trip to Hangzhou")
        assert WEEKEND_HOLIDAY_PATTERN.search("christmas in Paris")

    def test_pattern_no_false_positive(self) -> None:
        assert not WEEKEND_HOLIDAY_PATTERN.search("工作日去杭州")


class TestBuiltinChineseTerms:
    def test_contains_destinations(self) -> None:
        assert "杭州" in BUILTIN_CHINESE_TERMS
        assert "东京" in BUILTIN_CHINESE_TERMS
        assert "北京" in BUILTIN_CHINESE_TERMS

    def test_contains_travel_terms(self) -> None:
        assert "酒店" in BUILTIN_CHINESE_TERMS
        assert "地铁" in BUILTIN_CHINESE_TERMS
        assert "门票" in BUILTIN_CHINESE_TERMS
        assert "拥挤" in BUILTIN_CHINESE_TERMS

    def test_contains_poi_names(self) -> None:
        assert "西湖" in BUILTIN_CHINESE_TERMS
        assert "灵隐寺" in BUILTIN_CHINESE_TERMS
        assert "故宫" in BUILTIN_CHINESE_TERMS


class TestKeywordAnswerTokens:
    def test_is_tuple(self) -> None:
        assert isinstance(KEYWORD_ANSWER_TOKENS, tuple)

    def test_contains_core_tokens(self) -> None:
        assert "周末" in KEYWORD_ANSWER_TOKENS
        assert "拥挤" in KEYWORD_ANSWER_TOKENS
        assert "亲子" in KEYWORD_ANSWER_TOKENS


class TestPeopleImplicit:
    def test_solo_patterns(self) -> None:
        assert PEOPLE_IMPLICIT["一个人"] == 1
        assert PEOPLE_IMPLICIT["独自"] == 1
        assert PEOPLE_IMPLICIT["solo"] == 1

    def test_family_patterns(self) -> None:
        assert PEOPLE_IMPLICIT["我和父母"] == 3
        assert PEOPLE_IMPLICIT["一家三口"] == 3
        assert PEOPLE_IMPLICIT["一家四口"] == 4
        assert PEOPLE_IMPLICIT["我们俩"] == 2


class TestWeekendHolidayKeywords:
    def test_is_list(self) -> None:
        assert isinstance(WEEKEND_HOLIDAY_KEYWORDS, list)

    def test_contains_major_holidays(self) -> None:
        assert "春节" in WEEKEND_HOLIDAY_KEYWORDS
        assert "国庆" in WEEKEND_HOLIDAY_KEYWORDS
        assert "五一" in WEEKEND_HOLIDAY_KEYWORDS
        assert "周末" in WEEKEND_HOLIDAY_KEYWORDS
