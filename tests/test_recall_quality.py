import json
import shutil
import unittest
from dataclasses import dataclass
from pathlib import Path

from travel_agent.rag import TravelRag
from travel_agent.rag.config import EmbeddingProviderName
from travel_agent.rag.models import SearchResult


@dataclass(frozen=True)
class RecallCase:
    query: str
    expected_source: str
    expected_keywords: tuple[str, ...]
    destination: str | None = None
    section: str | None = None
    travel_type: str | None = None
    season: str | None = None
    expected_empty: bool = False
    hard_negative: bool = False


@dataclass(frozen=True)
class EvalMetrics:
    recall_at_3: float
    mrr_at_3: float
    keyword_hit_rate_at_3: float
    metadata_filter_accuracy: float


class RecallQualityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.cases = _load_cases(cls.root / "tests" / "fixtures" / "rag_eval_cases.jsonl")
        cls.persist_dir = cls.root / "data" / "test_tmp" / "recall_quality" / "chroma"
        if cls.persist_dir.parent.exists():
            shutil.rmtree(cls.persist_dir.parent, ignore_errors=True)

        cls.rag = TravelRag.create(
            persist_dir=cls.persist_dir,
            embedding_provider=EmbeddingProviderName.LOCAL,
            top_k=3,
        )
        cls.rag.ingest(cls.root / "docs" / "destinations")
        cls.case_results = {
            case.query: _search_case(cls.rag, case)
            for case in cls.cases
        }
        cls.metrics = _compute_metrics(cls.cases, cls.case_results)

    def test_recall_at_3_hits_expected_source_for_all_cases(self) -> None:
        failures = [
            _source_failure(case, self.case_results[case.query])
            for case in self.cases
            if not case.expected_empty
            and not case.hard_negative
            and _rank_of_expected_source(
                self.case_results[case.query],
                case.expected_source,
            )
            is None
        ]

        self.assertGreaterEqual(
            self.metrics.recall_at_3,
            1.0,
            _metric_failure("recall@3", self.metrics.recall_at_3, failures),
        )

    def test_mrr_at_3_is_perfect_for_fixture_cases(self) -> None:
        failures = []
        for case in self.cases:
            if case.expected_empty or case.hard_negative:
                continue
            results = self.case_results[case.query]
            rank = _rank_of_expected_source(results, case.expected_source)
            if rank != 1:
                failures.append(
                    f"query={case.query!r}, expected_source={case.expected_source!r}, "
                    f"expected_rank=1, actual_rank={rank!r}, actual_sources={_sources(results)!r}"
                )

        self.assertGreaterEqual(
            self.metrics.mrr_at_3,
            1.0,
            _metric_failure("MRR@3", self.metrics.mrr_at_3, failures),
        )

    def test_keyword_hit_rate_at_3_is_high_enough(self) -> None:
        misses: list[str] = []
        for case in self.cases:
            if case.expected_empty:
                continue
            results = self.case_results[case.query]
            joined_content = "\n".join(result.content for result in results)
            for keyword in case.expected_keywords:
                if keyword not in joined_content:
                    misses.append(
                        f"query={case.query!r}, missing_keyword={keyword!r}, "
                        f"expected_source={case.expected_source!r}, "
                        f"actual_sources={_sources(results)!r}"
                    )

        self.assertGreaterEqual(
            self.metrics.keyword_hit_rate_at_3,
            0.9,
            _metric_failure("keyword_hit_rate@3", self.metrics.keyword_hit_rate_at_3, misses),
        )

    def test_metadata_filter_accuracy_is_perfect(self) -> None:
        failures: list[str] = []
        for case in self.cases:
            results = self.case_results[case.query]
            if case.expected_empty:
                if results:
                    failures.append(f"query={case.query!r}, expected empty results")
                continue
            if not results:
                failures.append(f"query={case.query!r}, no results returned")
                continue
            for result in results:
                mismatches = _metadata_mismatches(case, result)
                if mismatches:
                    failures.append(
                        f"query={case.query!r}, source={result.source!r}, "
                        f"mismatches={mismatches!r}, metadata={result.metadata!r}"
                    )

        self.assertGreaterEqual(
            self.metrics.metadata_filter_accuracy,
            1.0,
            _metric_failure(
                "metadata_filter_accuracy",
                self.metrics.metadata_filter_accuracy,
                failures,
            ),
        )


def _load_cases(path: Path) -> list[RecallCase]:
    cases: list[RecallCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        try:
            cases.append(
                RecallCase(
                    query=payload["query"],
                    destination=payload.get("destination"),
                    section=payload.get("section"),
                    travel_type=payload.get("travel_type"),
                    season=payload.get("season"),
                    expected_source=payload["expected_source"],
                    expected_keywords=tuple(payload["expected_keywords"]),
                    expected_empty=bool(payload.get("expected_empty", False)),
                    hard_negative=bool(payload.get("hard_negative", False)),
                )
            )
        except KeyError as exc:
            raise AssertionError(f"missing field {exc!s} in {path}:{line_number}") from exc
    return cases


def _search_case(rag: TravelRag, case: RecallCase) -> list[SearchResult]:
    return rag.search(
        case.query,
        destination=case.destination,
        section=case.section,
        travel_type=case.travel_type,
        season=case.season,
        top_k=3,
    )


def _compute_metrics(
    cases: list[RecallCase],
    case_results: dict[str, list[SearchResult]],
) -> EvalMetrics:
    answerable_cases = [
        case for case in cases if not case.expected_empty and not case.hard_negative
    ]
    total_cases = len(answerable_cases)
    total_keywords = 0
    hit_keywords = 0
    reciprocal_rank_sum = 0.0
    recalled_cases = 0
    metadata_checks = 0
    metadata_hits = 0

    for case in cases:
        if case.expected_empty or case.hard_negative:
            continue
        results = case_results[case.query]
        rank = _rank_of_expected_source(results, case.expected_source)
        if rank is not None:
            recalled_cases += 1
            reciprocal_rank_sum += 1.0 / rank

        joined_content = "\n".join(result.content for result in results)
        for keyword in case.expected_keywords:
            total_keywords += 1
            if keyword in joined_content:
                hit_keywords += 1

        for result in results:
            expected_filters = _expected_metadata_filters(case)
            metadata_checks += len(expected_filters)
            metadata_hits += sum(
                1
                for key, expected_value in expected_filters.items()
                if _matches_metadata_value(result.metadata.get(key), expected_value)
            )

    return EvalMetrics(
        recall_at_3=recalled_cases / total_cases,
        mrr_at_3=reciprocal_rank_sum / total_cases,
        keyword_hit_rate_at_3=hit_keywords / total_keywords,
        metadata_filter_accuracy=metadata_hits / metadata_checks if metadata_checks else 1.0,
    )


def _rank_of_expected_source(results: list[SearchResult], expected_source: str) -> int | None:
    if not expected_source:
        return None
    for index, result in enumerate(results[:3], start=1):
        if result.source.endswith(expected_source):
            return index
    return None


def _metadata_mismatches(case: RecallCase, result: SearchResult) -> dict[str, object]:
    mismatches: dict[str, object] = {}
    for key, expected_value in _expected_metadata_filters(case).items():
        actual_value = result.metadata.get(key)
        if not _matches_metadata_value(actual_value, expected_value):
            mismatches[key] = {"expected": expected_value, "actual": actual_value}
    return mismatches


def _expected_metadata_filters(case: RecallCase) -> dict[str, str]:
    filters: dict[str, str | None] = {
        "destination": case.destination,
        "section": case.section,
        "travel_type": case.travel_type,
        "season": case.season,
    }
    expected: dict[str, str] = {}
    for key, value in filters.items():
        if value:
            expected[key] = value
    return expected


def _matches_metadata_value(actual: object, expected: str) -> bool:
    if actual is None:
        return False
    actual_text = str(actual).strip()
    if actual_text == expected:
        return True
    return expected in {
        item.strip()
        for item in actual_text.split(",")
        if item.strip()
    }


def _source_failure(case: RecallCase, results: list[SearchResult]) -> str:
    return (
        f"query={case.query!r}, expected_source={case.expected_source!r}, "
        f"expected_filters={_expected_metadata_filters(case)!r}, "
        f"actual_sources={_sources(results)!r}"
    )


def _metric_failure(metric_name: str, value: float, failures: list[str]) -> str:
    return f"{metric_name}={value:.3f} failed:\n" + "\n".join(failures)


def _sources(results: list[SearchResult]) -> list[str]:
    return [result.source for result in results]


if __name__ == "__main__":
    unittest.main()
