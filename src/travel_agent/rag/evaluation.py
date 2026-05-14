"""纯 RAG 检索评估工具。"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from travel_agent.rag.api import TravelRag
from travel_agent.rag.config import EmbeddingProviderName, RetrievalMode
from travel_agent.rag.models import SearchResult

DEFAULT_QUALITY_THRESHOLDS = {
    "recall_at_k": 0.95,
    "mrr_at_k": 0.9,
    "keyword_hit_rate_at_k": 0.9,
    "metadata_filter_accuracy": 1.0,
    "expected_empty_accuracy": 0.5,
}


@dataclass(frozen=True)
class EvalCase:
    query: str
    expected_keywords: tuple[str, ...]
    expected_source: str = ""
    destination: str | None = None
    section: str | None = None
    travel_type: str | None = None
    season: str | None = None
    expected_empty: bool = False
    hard_negative: bool = False


@dataclass(frozen=True)
class EvalReport:
    metrics: dict[str, float]
    run: dict[str, str | int | float]
    failures: list[str]
    quality_gate: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "run": self.run,
            "failures": self.failures,
            "quality_gate": self.quality_gate,
        }


def evaluate_rag(
    docs_path: Path,
    cases_path: Path,
    persist_dir: Path,
    collection_name: str,
    embedding_provider: str | EmbeddingProviderName,
    retrieval_mode: str | RetrievalMode,
    top_k: int,
    thresholds: dict[str, float] | None = None,
) -> EvalReport:
    """构建临时索引并在不调用任何 LLM 的情况下评估检索质量。"""

    if persist_dir.exists():
        shutil.rmtree(persist_dir, ignore_errors=True)

    rag = TravelRag.create(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
    )
    ingest_report = rag.ingest(docs_path)
    cases = load_eval_cases(cases_path)

    case_results = []
    for case in cases:
        evidence = rag.retrieve_evidence(
            case.query,
            destination=case.destination,
            section=case.section,
            travel_type=case.travel_type,
            season=case.season,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        )
        case_results.append((case, evidence.results, evidence.trace.total_latency_ms))

    metrics, failures = _metrics(case_results, top_k)
    quality_gate = evaluate_quality_gate(metrics, thresholds)
    run = {
        "cases": len(cases),
        "top_k": top_k,
        "embedding_provider": str(embedding_provider),
        "retrieval_mode": str(retrieval_mode),
        "indexed_chunks": ingest_report.indexed_chunks,
        "persist_dir": str(persist_dir),
        "collection_name": collection_name,
    }
    return EvalReport(
        metrics=metrics,
        run=run,
        failures=failures,
        quality_gate=quality_gate,
    )


def evaluate_quality_gate(
    metrics: dict[str, float],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """返回配置的 RAG 质量阈值的通过/失败状态。"""

    active_thresholds = thresholds or DEFAULT_QUALITY_THRESHOLDS
    failures = []
    for metric_name, threshold in active_thresholds.items():
        value = metrics.get(metric_name, 0.0)
        if value < threshold:
            failures.append(f"{metric_name}={value:.4f} below threshold {threshold:.4f}")

    return {
        "passed": not failures,
        "thresholds": active_thresholds,
        "failures": failures,
    }


def load_eval_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        try:
            cases.append(
                EvalCase(
                    query=payload["query"],
                    destination=payload.get("destination"),
                    section=payload.get("section"),
                    travel_type=payload.get("travel_type"),
                    season=payload.get("season"),
                    expected_source=payload.get("expected_source", ""),
                    expected_keywords=tuple(payload.get("expected_keywords", ())),
                    expected_empty=bool(payload.get("expected_empty", False)),
                    hard_negative=bool(payload.get("hard_negative", False)),
                )
            )
        except KeyError as exc:
            raise AssertionError(f"missing field {exc!s} in {path}:{line_number}") from exc
    return cases


def _metrics(
    case_results: list[tuple[EvalCase, list[SearchResult], float]],
    top_k: int,
) -> tuple[dict[str, float], list[str]]:
    non_empty_cases = [item for item in case_results if not item[0].expected_empty]
    expected_empty_cases = [item for item in case_results if item[0].expected_empty]
    total_keywords = 0
    hit_keywords = 0
    recalled = 0
    reciprocal_rank_sum = 0.0
    precision_sum = 0.0
    ndcg_sum = 0.0
    metadata_checks = 0
    metadata_hits = 0
    empty_hits = 0
    failures: list[str] = []

    for case, results, _ in non_empty_cases:
        rank = _rank_of_expected_source(results, case.expected_source)
        if rank is None:
            failures.append(
                f"miss source query={case.query!r}, expected={case.expected_source!r}, "
                f"actual={[result.source for result in results]!r}"
            )
        else:
            recalled += 1
            reciprocal_rank_sum += 1.0 / rank
            ndcg_sum += 1.0 / _log2(rank + 1)
            precision_sum += 1.0 / top_k

        joined_content = "\n".join(result.content for result in results)
        for keyword in case.expected_keywords:
            total_keywords += 1
            if keyword in joined_content:
                hit_keywords += 1

        for result in results:
            for key, value in _expected_filters(case).items():
                metadata_checks += 1
                metadata_hits += int(_matches_metadata_value(result.metadata.get(key), value))

    for case, results, _ in expected_empty_cases:
        if not results:
            empty_hits += 1
        else:
            failures.append(
                f"expected empty query={case.query!r}, actual_sources="
                f"{[result.source for result in results]!r}"
            )

    total_cases = len(non_empty_cases)
    latencies = [latency for _, _, latency in case_results]
    metrics = {
        "recall_at_k": recalled / total_cases if total_cases else 1.0,
        "mrr_at_k": reciprocal_rank_sum / total_cases if total_cases else 1.0,
        "precision_at_k": precision_sum / total_cases if total_cases else 1.0,
        "ndcg_at_k": ndcg_sum / total_cases if total_cases else 1.0,
        "keyword_hit_rate_at_k": hit_keywords / total_keywords if total_keywords else 1.0,
        "metadata_filter_accuracy": metadata_hits / metadata_checks if metadata_checks else 1.0,
        "empty_result_rate": sum(1 for _, results, _ in case_results if not results)
        / len(case_results),
        "expected_empty_accuracy": empty_hits / len(expected_empty_cases)
        if expected_empty_cases
        else 1.0,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
    }
    return metrics, failures


def _rank_of_expected_source(results: list[SearchResult], expected_source: str) -> int | None:
    if not expected_source:
        return None
    for index, result in enumerate(results, start=1):
        if result.source.endswith(expected_source):
            return index
    return None


def _expected_filters(case: EvalCase) -> dict[str, str]:
    filters = {
        "destination": case.destination,
        "section": case.section,
        "travel_type": case.travel_type,
        "season": case.season,
    }
    return {key: value for key, value in filters.items() if value}


def _matches_metadata_value(actual: object, expected: str) -> bool:
    if actual is None:
        return False
    actual_text = str(actual).strip()
    if actual_text == expected:
        return True
    return expected in {item.strip() for item in actual_text.split(",") if item.strip()}


def _log2(value: int) -> float:
    import math

    return math.log2(value)
