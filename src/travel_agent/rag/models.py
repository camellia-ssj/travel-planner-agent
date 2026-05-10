"""Domain models returned by the travel RAG module."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

Metadata = dict[str, str | int | float | bool | None]


@dataclass(frozen=True)
class SearchResult:
    """A retrieved Chroma chunk with user-facing metadata."""

    content: str
    source: str
    destination: str
    score: float
    metadata: Metadata


@dataclass(frozen=True)
class IngestReport:
    """Summary returned after ingestion."""

    scanned_files: int
    loaded_documents: int
    skipped_unchanged: int
    deleted_chunks: int
    indexed_chunks: int
    persist_dir: str
    collection_name: str
    manifest_path: str


@dataclass(frozen=True)
class QueryResponse:
    """Retrieval response returned by the pure RAG module."""

    query: str
    results: list[SearchResult]

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "results": [
                {
                    "content": result.content,
                    "source": result.source,
                    "destination": result.destination,
                    "score": result.score,
                    "metadata": result.metadata,
                }
                for result in self.results
            ],
        }


@dataclass(frozen=True)
class RetrievalTrace:
    """Operational metadata for one pure-RAG retrieval call."""

    trace_id: str
    retrieval_mode: str
    requested_top_k: int
    candidate_k: int
    returned_results: int
    empty_result: bool
    destination: str
    section: str
    travel_type: str
    season: str
    embedding_provider: str
    collection_version: str
    metadata_filters: dict[str, Any]
    vector_hits: list[dict[str, Any]]
    keyword_hits: list[dict[str, Any]]
    fused_hits: list[dict[str, Any]]
    reranked_hits: list[dict[str, Any]]
    empty_result_reason: str = ""
    average_score: float = 0.0
    vector_latency_ms: float = 0.0
    keyword_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    created_at: str = ""

    @classmethod
    def create(cls, **values: Any) -> RetrievalTrace:
        values.setdefault("trace_id", uuid4().hex)
        values.setdefault("created_at", datetime.now(UTC).isoformat())
        return cls(**values)


@dataclass(frozen=True)
class EvidenceBundle:
    """Structured retrieval context for downstream LangGraph nodes."""

    question: str
    results: list[SearchResult]
    trace: RetrievalTrace
    query_analysis: dict[str, str]
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "trace": self.trace.__dict__,
            "query_analysis": self.query_analysis,
            "confidence": self.confidence,
            "results": [
                {
                    "content": result.content,
                    "source": result.source,
                    "destination": result.destination,
                    "score": result.score,
                    "metadata": result.metadata,
                }
                for result in self.results
            ],
        }


@dataclass(frozen=True)
class AnswerResponse:
    """Extractive answer assembled from retrieved knowledge chunks."""

    question: str
    answer: str
    results: list[SearchResult]

    def as_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [
                {
                    "source": result.source,
                    "destination": result.destination,
                    "score": result.score,
                }
                for result in self.results
            ],
            "results": [
                {
                    "content": result.content,
                    "source": result.source,
                    "destination": result.destination,
                    "score": result.score,
                    "metadata": result.metadata,
                }
                for result in self.results
            ],
        }
