"""Lightweight keyword retrieval for travel RAG chunks."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from langchain_core.documents import Document

_WORD_RE = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)


def bm25_search(query: str, documents: list[Document], top_k: int) -> list[tuple[Document, float]]:
    """Return BM25-ranked documents for a query.

    This intentionally avoids an extra dependency. The tokenizer supports
    English words, individual CJK characters, and adjacent CJK bigrams, which
    is sufficient for short Chinese travel queries such as "预算" or "交通".
    """

    if top_k <= 0 or not documents:
        return []

    query_terms = _tokenize(query)
    if not query_terms:
        return []

    doc_terms = [_tokenize(document.page_content) for document in documents]
    avgdl = sum(len(terms) for terms in doc_terms) / len(doc_terms)
    doc_freq: Counter[str] = Counter()
    for terms in doc_terms:
        doc_freq.update(set(terms))

    scored: list[tuple[float, int, Document]] = []
    for index, (document, terms) in enumerate(zip(documents, doc_terms, strict=False)):
        score = _bm25_score(query_terms, terms, doc_freq, len(documents), avgdl)
        if score > 0:
            scored.append((score, -index, document))

    scored.sort(reverse=True)
    return [(document, score) for score, _, document in scored[:top_k]]


@dataclass
class BM25Index:
    """In-memory BM25 index reused across keyword and hybrid retrieval calls."""

    documents: list[Document]
    document_terms: list[list[str]]
    doc_freq: Counter[str]
    avgdl: float

    @classmethod
    def build(cls, documents: list[Document]) -> BM25Index:
        document_terms = [_tokenize(document.page_content) for document in documents]
        avgdl = (
            sum(len(terms) for terms in document_terms) / len(document_terms)
            if documents
            else 0.0
        )
        doc_freq: Counter[str] = Counter()
        for terms in document_terms:
            doc_freq.update(set(terms))
        return cls(
            documents=documents,
            document_terms=document_terms,
            doc_freq=doc_freq,
            avgdl=avgdl,
        )

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, str | None] | None = None,
    ) -> list[tuple[Document, float]]:
        if top_k <= 0 or not self.documents:
            return []

        query_terms = _tokenize(query)
        if not query_terms:
            return []

        scored: list[tuple[float, int, Document]] = []
        for index, (document, terms) in enumerate(
            zip(self.documents, self.document_terms, strict=False)
        ):
            if filters and not _matches_filters(document.metadata, filters):
                continue
            score = _bm25_score(query_terms, terms, self.doc_freq, len(self.documents), self.avgdl)
            if score > 0:
                scored.append((score, -index, document))

        scored.sort(reverse=True)
        return [(document, score) for score, _, document in scored[:top_k]]


def _bm25_score(
    query_terms: list[str],
    document_terms: list[str],
    doc_freq: Counter[str],
    total_documents: int,
    avgdl: float,
) -> float:
    if not document_terms:
        return 0.0

    term_freq = Counter(document_terms)
    doc_length = len(document_terms)
    k1 = 1.5
    b = 0.75
    score = 0.0

    for term in query_terms:
        tf = term_freq.get(term, 0)
        if tf <= 0:
            continue
        df = doc_freq.get(term, 0)
        idf = math.log(1 + (total_documents - df + 0.5) / (df + 0.5))
        denominator = tf + k1 * (1 - b + b * doc_length / avgdl)
        score += idf * (tf * (k1 + 1) / denominator)

    return score


def _tokenize(text: str) -> list[str]:
    normalized = text.lower()
    tokens = _WORD_RE.findall(normalized)

    cjk_runs: list[str] = []
    current: list[str] = []
    for char in normalized:
        if "\u4e00" <= char <= "\u9fff":
            current.append(char)
        elif current:
            cjk_runs.append("".join(current))
            current = []
    if current:
        cjk_runs.append("".join(current))

    for run in cjk_runs:
        tokens.extend(run)
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))

    return tokens


def _matches_filters(metadata: dict[str, object], filters: dict[str, str | None]) -> bool:
    return all(
        _matches_filter_value(metadata.get(key), value)
        for key, value in filters.items()
        if value
    )


def _matches_filter_value(actual: object, expected: str | None) -> bool:
    if not expected:
        return True
    if actual is None:
        return False
    if isinstance(actual, bool):
        return actual is (expected.lower() == "true")

    expected_text = expected.strip().lower()
    actual_text = str(actual).strip().lower()
    if actual_text == expected_text:
        return True
    return expected_text in {
        value.strip()
        for value in re.split(r"[,;|]", actual_text)
        if value.strip()
    }
