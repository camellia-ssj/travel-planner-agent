"""旅行 RAG 文档块的关键词检索。"""

from __future__ import annotations

import importlib
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from travel_agent.knowledge import BUILTIN_CHINESE_TERMS
from travel_agent.rag.config import KeywordTokenizerName

_WORD_RE = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_USER_DICT_CACHE: set[Path] = set()


def bm25_search(
    query: str,
    documents: list[Document],
    top_k: int,
    tokenizer: KeywordTokenizerName | str = KeywordTokenizerName.AUTO,
    user_dict: str | Path | None = None,
) -> list[tuple[Document, float]]:
    """返回查询的 BM25 排序文档。

    默认分词器在 jieba 已安装时使用它，同时保留一个无依赖的内置分词器
    作为测试和简单演示的回退。
    """

    if top_k <= 0 or not documents:
        return []

    tokenize = _build_tokenizer(tokenizer, user_dict)
    query_terms = tokenize(query)
    if not query_terms:
        return []

    doc_terms = [tokenize(document.page_content) for document in documents]
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
    """跨关键词和混合检索调用重用的内存 BM25 索引。"""

    documents: list[Document]
    document_terms: list[list[str]]
    doc_freq: Counter[str]
    avgdl: float
    tokenizer: Callable[[str], list[str]] = field(repr=False)

    @classmethod
    def build(
        cls,
        documents: list[Document],
        tokenizer: KeywordTokenizerName | str = KeywordTokenizerName.AUTO,
        user_dict: str | Path | None = None,
    ) -> BM25Index:
        tokenize = _build_tokenizer(tokenizer, user_dict)
        document_terms = [tokenize(document.page_content) for document in documents]
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
            tokenizer=tokenize,
        )

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, str | None] | None = None,
    ) -> list[tuple[Document, float]]:
        if top_k <= 0 or not self.documents:
            return []

        query_terms = self.tokenizer(query)
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


def _build_tokenizer(
    tokenizer: KeywordTokenizerName | str,
    user_dict: str | Path | None,
) -> Callable[[str], list[str]]:
    name = KeywordTokenizerName(tokenizer)
    if name in {KeywordTokenizerName.AUTO, KeywordTokenizerName.JIEBA}:
        jieba_tokenize = _jieba_tokenizer(user_dict)
        if jieba_tokenize is not None:
            return jieba_tokenize
        if name is KeywordTokenizerName.JIEBA:
            raise RuntimeError(
                "jieba is not installed. Install travel-agent[keyword] or set "
                "TRAVEL_RAG_KEYWORD_TOKENIZER=builtin."
            )
    return _tokenize_builtin


def _jieba_tokenizer(user_dict: str | Path | None) -> Callable[[str], list[str]] | None:
    try:
        import io
        import sys
        import warnings as _warnings

        # jieba 在导入时会向 stdout 打印加载信息并发出 pkg_resources
        # 弃用警告 —— 在导入期间抑制这两者
        _old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        _warnings.filterwarnings("ignore", category=UserWarning, module="jieba")
        try:
            jieba = importlib.import_module("jieba")
        finally:
            sys.stdout = _old_stdout
            _warnings.resetwarnings()
    except ImportError:
        return None

    if user_dict:
        dictionary = Path(user_dict)
        if dictionary.exists() and dictionary not in _USER_DICT_CACHE:
            jieba.load_userdict(str(dictionary))
            _USER_DICT_CACHE.add(dictionary)

    # 在分词期间抑制 jieba 的 INFO 日志消息
    import logging
    logging.getLogger("jieba").setLevel(logging.WARNING)

    def tokenize(text: str) -> list[str]:
        normalized = text.lower()
        tokens = _WORD_RE.findall(normalized)
        for token in jieba.cut(normalized, cut_all=False):
            token = token.strip()
            if token and not token.isspace():
                tokens.append(token)
        tokens.extend(_cjk_character_ngrams(normalized))
        return _dedupe_preserving_order(tokens)

    return tokenize


def _tokenize_builtin(text: str) -> list[str]:
    normalized = text.lower()
    tokens = _WORD_RE.findall(normalized)
    for run in _CJK_RE.findall(normalized):
        tokens.extend(_dictionary_terms(run))
        tokens.extend(_cjk_character_ngrams(run))
    return _dedupe_preserving_order(tokens)


def _dictionary_terms(run: str) -> list[str]:
    return [term for term in BUILTIN_CHINESE_TERMS if term in run]


def _cjk_character_ngrams(text: str) -> list[str]:
    tokens: list[str] = []
    cjk_runs: list[str] = []
    current: list[str] = []
    for char in text:
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


def _dedupe_preserving_order(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for token in tokens:
        token = token.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


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
