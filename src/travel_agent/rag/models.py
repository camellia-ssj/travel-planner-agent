"""旅行 RAG 模块返回的领域模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


class QueryRewriteMode(StrEnum):
    """检索的查询重写策略。"""

    OFF = "off"
    REWRITE_ONLY = "rewrite_only"
    MULTI_QUERY = "multi_query"
    ON = "on"  # 别名指向 MULTI_QUERY


@dataclass(frozen=True)
class QueryRewriteResult:
    """LLM 驱动的旅行领域检索查询重写结果。"""

    original_query: str
    rewritten_query: str
    search_queries: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    raw_response: str = ""

    def __post_init__(self) -> None:
        if not self.search_queries:
            object.__setattr__(self, "search_queries", [self.rewritten_query])

Metadata = dict[str, str | int | float | bool | None]


@dataclass(frozen=True)
class SearchResult:
    """检索到的 Chroma 文档块，附带面向用户的元数据。"""

    content: str
    source: str
    destination: str
    score: float
    metadata: Metadata


@dataclass(frozen=True)
class IngestReport:
    """入库后返回的摘要。"""

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
    """纯 RAG 模块返回的检索响应。"""

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
    """单次纯 RAG 检索调用的操作元数据。"""

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
    reranker: str
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
    """供下游 LangGraph 节点使用的结构化检索上下文。"""

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
    """从检索到的知识块中组合而成的抽取式答案。"""

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
