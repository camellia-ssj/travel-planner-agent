"""Reranker extension points for retrieval results."""

from __future__ import annotations

import importlib
import re
from typing import Any, Protocol

from travel_agent.rag.config import RagSettings, RerankerName
from travel_agent.rag.models import SearchResult


class Reranker(Protocol):
    """Protocol for optional pure-RAG rerankers."""

    name: str

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Return results in reranked order."""


class NoOpReranker:
    """Reranker placeholder that preserves retrieval order."""

    name = "none"

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        return results


class KeywordOverlapReranker:
    """Deterministic lexical reranker for pure-RAG operation.

    This avoids any LLM or cross-encoder dependency while still promoting chunks
    that share concrete query terms with the user's question.
    """

    name = "keyword"

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return results

        ranked: list[tuple[float, int, SearchResult]] = []
        for index, result in enumerate(results):
            content_tokens = _tokens(result.content)
            overlap = len(query_tokens & content_tokens)
            overlap_score = overlap / max(len(query_tokens), 1)
            combined_score = result.score + overlap_score
            ranked.append((combined_score, -index, result))

        ranked.sort(reverse=True)
        return [
            SearchResult(
                content=result.content,
                source=result.source,
                destination=result.destination,
                score=score,
                metadata=result.metadata,
            )
            for score, _, result in ranked
        ]


class CrossEncoderReranker:
    """Sentence-transformers cross-encoder reranker loaded lazily."""

    name = "cross-encoder"

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model: Any | None = None

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not query or len(results) <= 1:
            return results

        model = self._load_model()
        pairs = [(query, result.content) for result in results]
        scores = model.predict(pairs, batch_size=self.batch_size)

        ranked: list[tuple[float, int, SearchResult]] = []
        for index, (score, result) in enumerate(zip(scores, results, strict=False)):
            ranked.append((float(score), -index, result))

        ranked.sort(reverse=True)
        return [
            SearchResult(
                content=result.content,
                source=result.source,
                destination=result.destination,
                score=score,
                metadata=result.metadata,
            )
            for score, _, result in ranked
        ]

    def _load_model(self) -> Any:
        if self._model is None:
            module = importlib.import_module("sentence_transformers")
            cross_encoder_class = module.CrossEncoder
            kwargs: dict[str, object] = {}
            if self.device:
                kwargs["device"] = self.device
            self._model = cross_encoder_class(self.model_name, **kwargs)
        return self._model


class FallbackReranker:
    """Use a primary reranker, then fallback if model loading or scoring fails."""

    def __init__(self, primary: Reranker, fallback: Reranker) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}+fallback"

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        try:
            return self.primary.rerank(query, results)
        except (ImportError, OSError, RuntimeError, ValueError):
            return self.fallback.rerank(query, results)


def build_reranker(settings: RagSettings) -> Reranker:
    """Build the configured reranker without loading optional models up front."""

    fallback = KeywordOverlapReranker()
    if settings.reranker is RerankerName.KEYWORD:
        return fallback

    if settings.reranker in {RerankerName.CROSS_ENCODER, RerankerName.BGE_RERANKER}:
        primary: Reranker = CrossEncoderReranker(
            model_name=settings.reranker_model,
            batch_size=settings.reranker_batch_size,
            device=settings.reranker_device,
        )
        if settings.reranker_fallback:
            return FallbackReranker(primary=primary, fallback=fallback)
        return primary

    return fallback


def _tokens(text: str) -> set[str]:
    normalized = text.lower()
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    cjk = [char for char in normalized if "\u4e00" <= char <= "\u9fff"]
    tokens.update(cjk)
    tokens.update("".join(cjk[index : index + 2]) for index in range(len(cjk) - 1))
    return tokens
