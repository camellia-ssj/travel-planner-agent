"""Reranker extension points for retrieval results."""

from __future__ import annotations

import re
from typing import Protocol

from travel_agent.rag.models import SearchResult


class Reranker(Protocol):
    """Protocol for optional rerankers.

    Implementations may call a local model or external service in the future.
    The default implementation intentionally performs no model calls.
    """

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """Return results in reranked order."""


class NoOpReranker:
    """Reranker placeholder that preserves retrieval order."""

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        return results


class KeywordOverlapReranker:
    """Deterministic lexical reranker for pure-RAG operation.

    This avoids any LLM or cross-encoder dependency while still promoting chunks
    that share concrete query terms with the user's question.
    """

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


def _tokens(text: str) -> set[str]:
    normalized = text.lower()
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    cjk = [char for char in normalized if "\u4e00" <= char <= "\u9fff"]
    tokens.update(cjk)
    tokens.update("".join(cjk[index : index + 2]) for index in range(len(cjk) - 1))
    return tokens
