"""Application service for LangChain + Chroma destination RAG."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from time import perf_counter

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from travel_agent.rag.config import RagSettings, RetrievalMode
from travel_agent.rag.embeddings import build_embeddings
from travel_agent.rag.keyword import BM25Index
from travel_agent.rag.loaders import discover_document_files, load_documents
from travel_agent.rag.manifest import load_manifest, manifest_document
from travel_agent.rag.metadata import ensure_chunk_metadata, split_documents_by_markdown_section
from travel_agent.rag.models import (
    AnswerResponse,
    EvidenceBundle,
    IngestReport,
    QueryResponse,
    RetrievalTrace,
    SearchResult,
)
from travel_agent.rag.rerankers import Reranker, build_reranker
from travel_agent.rag.splitters import build_text_splitter
from travel_agent.rag.vector_store import (
    build_chroma_filter,
    build_vector_store,
    delete_documents_by_document_id,
    delete_documents_by_source,
    reset_chroma,
    vector_store_count,
    vector_store_documents,
    vector_store_metadatas,
)

DESTINATION_ALIASES = {
    "杭州": "Hangzhou",
    "hangzhou": "Hangzhou",
    "东京": "Tokyo",
    "東京": "Tokyo",
    "tokyo": "Tokyo",
    "苏州": "Suzhou",
    "suzhou": "Suzhou",
    "大理": "Dali",
    "dali": "Dali",
    "长沙": "Changsha",
    "changsha": "Changsha",
    "巴黎": "Paris",
    "paris": "Paris",
    "成都": "Chengdu",
    "chengdu": "Chengdu",
    "北京": "Beijing",
    "beijing": "Beijing",
}

SECTION_QUERY_ALIASES = {
    "traffic": (
        "交通",
        "地铁",
        "公交",
        "机场",
        "高铁",
        "火车",
        "换乘",
        "停车",
        "打车",
        "怎么去",
        "如何去",
        "到达",
        "transit",
        "transport",
        "subway",
        "metro",
        "airport",
        "rail",
    ),
    "budget": ("预算", "费用", "花费", "价格", "门票", "多少钱", "budget", "cost", "price"),
    "lodging": ("住宿", "酒店", "住哪", "住在哪里", "民宿", "hotel", "lodging", "stay"),
    "dining": ("餐饮", "吃饭", "美食", "餐厅", "小吃", "吃什么", "dining", "food", "restaurant"),
    "crowd_risk": ("拥挤", "排队", "人多", "人流", "高峰", "crowd", "queue", "busy"),
    "weather_risk": (
        "天气",
        "下雨",
        "雨天",
        "高温",
        "寒冷",
        "台风",
        "weather",
        "rain",
        "hot",
        "cold",
    ),
    "itinerary": ("玩法", "怎么玩", "路线", "行程", "安排", "itinerary", "route", "plan"),
    "audience": ("适合人群", "亲子", "老人", "带孩子", "audience"),
    "alternatives": ("备选", "替代", "改去", "下雨去哪", "alternatives", "backup"),
}

DESTINATION_DISPLAY_NAMES = {
    "Beijing": "北京",
    "Changsha": "长沙",
    "Chengdu": "成都",
    "Dali": "大理",
    "Hangzhou": "杭州",
    "Paris": "巴黎",
    "Suzhou": "苏州",
    "Tokyo": "东京",
}


class RagService:
    """Coordinates LangChain loading, splitting, embedding, Chroma and retrieval."""

    def __init__(
        self,
        settings: RagSettings | None = None,
        embeddings: Embeddings | None = None,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings or RagSettings()
        self.settings.ensure_parent_dirs()
        self.embeddings = embeddings or build_embeddings(self.settings)
        self.vector_store = vector_store or build_vector_store(self.settings, self.embeddings)
        self.text_splitter = build_text_splitter(self.settings)
        self.reranker = reranker or build_reranker(self.settings)
        self._bm25_index: BM25Index | None = None
        self._bm25_collection_count = -1

    def ingest_documents(
        self,
        path: Path,
        destination: str | None = None,
        incremental: bool = False,
    ) -> IngestReport:
        files = discover_document_files(path)
        documents = load_documents(path, destination=destination)
        manifest = load_manifest(self.settings)
        documents_to_index = [
            document
            for document in documents
            if not incremental
            or not manifest.unchanged(
                str(document.metadata.get("source", "")),
                str(document.metadata.get("document_hash", "")),
            )
        ]
        skipped_unchanged = len(documents) - len(documents_to_index)

        deleted_chunks = delete_documents_by_source(
            self.vector_store,
            sorted({str(document.metadata.get("source", "")) for document in documents_to_index}),
        )
        deleted_chunks += delete_documents_by_document_id(
            self.vector_store,
            sorted(
                {str(document.metadata.get("document_id", "")) for document in documents_to_index}
            ),
        )
        chunks = self._split_documents(documents_to_index)
        if chunks:
            self.vector_store.add_documents(chunks, ids=[self._chunk_id(chunk) for chunk in chunks])

        chunks_by_document = _chunks_by_document(chunks)
        for document in documents_to_index:
            metadata = dict(document.metadata)
            source = str(metadata.get("source", ""))
            manifest.update(manifest_document(metadata, chunks_by_document.get(source, 0)))
        manifest.save()
        self._invalidate_keyword_cache()

        return IngestReport(
            scanned_files=len(files),
            loaded_documents=len(documents),
            skipped_unchanged=skipped_unchanged,
            deleted_chunks=deleted_chunks,
            indexed_chunks=len(chunks),
            persist_dir=str(self.settings.persist_dir),
            collection_name=self.settings.collection_name,
            manifest_path=str(manifest.path),
        )

    def ingest_markdown(
        self,
        path: Path,
        destination: str | None = None,
        incremental: bool = False,
    ) -> IngestReport:
        """Backward-compatible alias for ingesting supported knowledge documents."""

        return self.ingest_documents(path, destination=destination, incremental=incremental)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        destination: str | None = None,
        section: str | None = None,
        travel_type: str | None = None,
        season: str | None = None,
        retrieval_mode: str | RetrievalMode | None = None,
    ) -> list[SearchResult]:
        return self.retrieve_evidence(
            query,
            top_k=top_k,
            destination=destination,
            section=section,
            travel_type=travel_type,
            season=season,
            retrieval_mode=retrieval_mode,
        ).results

    def retrieve_evidence(
        self,
        query: str,
        top_k: int | None = None,
        destination: str | None = None,
        section: str | None = None,
        travel_type: str | None = None,
        season: str | None = None,
        retrieval_mode: str | RetrievalMode | None = None,
    ) -> EvidenceBundle:
        started_at = perf_counter()
        k = top_k or self.settings.default_top_k
        mode = _retrieval_mode(retrieval_mode or self.settings.retrieval_mode)
        destination = destination or self._infer_destination(query)
        explicit_section = section
        inferred_section = None if section else self._infer_section(query)
        section = section or inferred_section
        python_filters = {
            "section": section,
            "travel_type": travel_type,
            "season": season,
        }
        metadata_filters = _metadata_filters(destination, explicit_section, travel_type, season)
        candidate_k = self._candidate_k(k, mode is not RetrievalMode.VECTOR)
        retrieval = self._retrieve_with_mode(
            query=query,
            top_k=candidate_k,
            metadata_filters=metadata_filters,
            mode=mode,
        )
        results = retrieval["results"]
        vector_hits = retrieval["vector_hits"]
        keyword_hits = retrieval["keyword_hits"]
        fused_hits = retrieval["fused_hits"]
        vector_latency_ms = float(retrieval["vector_latency_ms"])
        keyword_latency_ms = float(retrieval["keyword_latency_ms"])
        pre_filter_results = results

        if any(value for value in python_filters.values()):
            filtered_results = [
                result
                for result in results
                if _matches_metadata_filters(result.metadata, python_filters)
            ]
            if filtered_results or not inferred_section:
                results = filtered_results
            else:
                python_filters["section"] = None
                results = [
                    result
                    for result in results
                    if _matches_metadata_filters(result.metadata, python_filters)
                ]

        filtered_results = results
        rerank_started_at = perf_counter()
        results = [
            result
            for result in self.reranker.rerank(query, results)
            if result.score >= self.settings.min_score
        ][:k]
        rerank_latency_ms = (perf_counter() - rerank_started_at) * 1000
        average_score = sum(result.score for result in results) / len(results) if results else 0.0
        manifest = load_manifest(self.settings)
        trace = RetrievalTrace.create(
            retrieval_mode=mode.value,
            requested_top_k=k,
            candidate_k=candidate_k,
            returned_results=len(results),
            empty_result=not results,
            destination=destination or "",
            section=section or "",
            travel_type=travel_type or "",
            season=season or "",
            embedding_provider=self.settings.embedding_provider.value,
            reranker=getattr(self.reranker, "name", self.settings.reranker.value),
            collection_version=manifest.collection_version,
            metadata_filters={
                "retriever": _clean_filters(metadata_filters),
                "post_filter": _clean_filters(python_filters),
            },
            vector_hits=_trace_hits(vector_hits),
            keyword_hits=_trace_hits(keyword_hits),
            fused_hits=_trace_hits(fused_hits),
            reranked_hits=_trace_hits(results),
            empty_result_reason=_empty_result_reason(
                collection_count=vector_store_count(self.vector_store),
                pre_filter_results=pre_filter_results,
                filtered_results=filtered_results,
                final_results=results,
            ),
            average_score=average_score,
            vector_latency_ms=vector_latency_ms,
            keyword_latency_ms=keyword_latency_ms,
            rerank_latency_ms=rerank_latency_ms,
            total_latency_ms=(perf_counter() - started_at) * 1000,
        )
        return EvidenceBundle(
            question=query,
            results=results,
            trace=trace,
            query_analysis={
                "destination": destination or "",
                "section": section or "",
                "travel_type": travel_type or "",
                "season": season or "",
            },
            confidence=min(max(average_score, 0.0), 1.0),
        )

    def as_retriever(
        self,
        top_k: int | None = None,
        destination: str | None = None,
        section: str | None = None,
    ) -> RagRetriever:
        return RagRetriever(
            service=self,
            top_k=top_k or self.settings.default_top_k,
            destination=destination,
            section=section,
        )

    def query(
        self,
        query: str,
        top_k: int | None = None,
        destination: str | None = None,
        section: str | None = None,
        travel_type: str | None = None,
        season: str | None = None,
        retrieval_mode: str | RetrievalMode | None = None,
    ) -> QueryResponse:
        return QueryResponse(
            query=query,
            results=self.retrieve(
                query,
                top_k=top_k,
                destination=destination,
                section=section,
                travel_type=travel_type,
                season=season,
                retrieval_mode=retrieval_mode,
            ),
        )

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        destination: str | None = None,
        section: str | None = None,
        travel_type: str | None = None,
        season: str | None = None,
        retrieval_mode: str | RetrievalMode | None = None,
    ) -> AnswerResponse:
        results = self.retrieve(
            question,
            top_k=top_k,
            destination=destination,
            section=section,
            travel_type=travel_type,
            season=season,
            retrieval_mode=retrieval_mode,
        )
        return AnswerResponse(
            question=question,
            answer=self._build_extractive_answer(question, results),
            results=results,
        )

    def stats(self) -> dict[str, int | str]:
        manifest = load_manifest(self.settings)
        return {
            "chunks": vector_store_count(self.vector_store),
            "persist_dir": str(self.settings.persist_dir),
            "collection_name": self.settings.collection_name,
            "collection_version": manifest.collection_version,
            "manifest_path": str(manifest.path),
            "embedding_provider": self.settings.embedding_provider.value,
        }

    def reset(self) -> None:
        reset_chroma(self.settings)
        self.vector_store = build_vector_store(self.settings, self.embeddings)
        self._invalidate_keyword_cache()

    def _split_documents(self, documents: list[Document]) -> list[Document]:
        section_documents = split_documents_by_markdown_section(documents)
        chunks = self.text_splitter.split_documents(section_documents)
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
            ensure_chunk_metadata(chunk.metadata)
        return chunks

    def _retrieve_with_mode(
        self,
        query: str,
        top_k: int,
        metadata_filters: dict[str, object],
        mode: RetrievalMode,
    ) -> dict[str, object]:
        if mode is RetrievalMode.VECTOR:
            vector_results, vector_latency_ms = self._timed_vector_search(
                query,
                top_k,
                metadata_filters,
            )
            if vector_results:
                return {
                    "results": vector_results,
                    "vector_hits": vector_results,
                    "keyword_hits": [],
                    "fused_hits": vector_results,
                    "vector_latency_ms": vector_latency_ms,
                    "keyword_latency_ms": 0.0,
                }
            keyword_results, keyword_latency_ms = self._timed_keyword_search(
                query,
                top_k,
                metadata_filters,
            )
            return {
                "results": keyword_results,
                "vector_hits": vector_results,
                "keyword_hits": keyword_results,
                "fused_hits": keyword_results,
                "vector_latency_ms": vector_latency_ms,
                "keyword_latency_ms": keyword_latency_ms,
            }
        if mode is RetrievalMode.KEYWORD:
            keyword_results, keyword_latency_ms = self._timed_keyword_search(
                query,
                top_k,
                metadata_filters,
            )
            return {
                "results": keyword_results,
                "vector_hits": [],
                "keyword_hits": keyword_results,
                "fused_hits": keyword_results,
                "vector_latency_ms": 0.0,
                "keyword_latency_ms": keyword_latency_ms,
            }
        vector_results, vector_latency_ms = self._timed_vector_search(
            query,
            top_k,
            metadata_filters,
        )
        keyword_results, keyword_latency_ms = self._timed_keyword_search(
            query,
            top_k,
            metadata_filters,
        )
        fused_results = _rrf_fuse(
            [vector_results, keyword_results],
            weights=[self.settings.vector_weight, self.settings.keyword_weight],
            k=self.settings.rrf_k,
        )[:top_k]
        return {
            "results": fused_results,
            "vector_hits": vector_results,
            "keyword_hits": keyword_results,
            "fused_hits": fused_results,
            "vector_latency_ms": vector_latency_ms,
            "keyword_latency_ms": keyword_latency_ms,
        }

    def _timed_vector_search(
        self,
        query: str,
        top_k: int,
        metadata_filters: dict[str, object],
    ) -> tuple[list[SearchResult], float]:
        started_at = perf_counter()
        results = self._vector_search(query, top_k=top_k, metadata_filters=metadata_filters)
        return results, (perf_counter() - started_at) * 1000

    def _timed_keyword_search(
        self,
        query: str,
        top_k: int,
        metadata_filters: dict[str, object],
    ) -> tuple[list[SearchResult], float]:
        started_at = perf_counter()
        results = self._keyword_search(query, top_k=top_k, metadata_filters=metadata_filters)
        return results, (perf_counter() - started_at) * 1000

    def _vector_search(
        self,
        query: str,
        top_k: int,
        metadata_filters: dict[str, object],
    ) -> list[SearchResult]:
        search_kwargs: dict[str, object] = {"k": top_k}
        chroma_filter = build_chroma_filter(metadata_filters)
        if chroma_filter:
            search_kwargs["filter"] = chroma_filter
        pairs = self.vector_store.similarity_search_with_score(query, **search_kwargs)
        return [
            self._to_search_result(document, self._distance_to_score(distance))
            for document, distance in pairs
        ]

    def _keyword_search(
        self,
        query: str,
        top_k: int,
        metadata_filters: dict[str, object],
    ) -> list[SearchResult]:
        filters = {key: str(value) for key, value in metadata_filters.items()}
        pairs = self._keyword_index().search(query, top_k=top_k, filters=filters)
        return [self._to_search_result(document, score) for document, score in pairs]

    def _candidate_k(self, requested_k: int, expand_to_collection: bool) -> int:
        count = vector_store_count(self.vector_store)
        if count <= 0:
            return requested_k
        if expand_to_collection:
            expanded_k = requested_k * self.settings.retrieval_candidate_multiplier
            return min(max(expanded_k, requested_k), count)
        return min(requested_k, count)

    def _keyword_index(self) -> BM25Index:
        count = vector_store_count(self.vector_store)
        if self._bm25_index is None or self._bm25_collection_count != count:
            self._bm25_index = BM25Index.build(
                vector_store_documents(self.vector_store),
                tokenizer=self.settings.keyword_tokenizer,
                user_dict=self.settings.keyword_user_dict,
            )
            self._bm25_collection_count = count
        return self._bm25_index

    def _invalidate_keyword_cache(self) -> None:
        self._bm25_index = None
        self._bm25_collection_count = -1

    @staticmethod
    def _to_search_result(document: Document, score: float) -> SearchResult:
        metadata = dict(document.metadata)
        return SearchResult(
            content=document.page_content,
            source=str(metadata.get("source", "")),
            destination=str(metadata.get("destination", "")),
            score=float(score),
            metadata=metadata,
        )

    @staticmethod
    def _distance_to_score(distance: float) -> float:
        return 1.0 / (1.0 + max(float(distance), 0.0))

    @staticmethod
    def _build_extractive_answer(question: str, results: list[SearchResult]) -> str:
        if not results:
            return "当前知识库没有检索到足够相关的资料，无法基于已有文档回答。"

        bullets: list[str] = []
        seen: set[str] = set()
        ranked_sentences = _rank_sentences(question, results)
        for result, sentence in ranked_sentences:
            normalized = sentence.strip()
            destination = result.destination or str(result.metadata.get("destination", ""))
            section_title = str(
                result.metadata.get("section_title", "") or result.metadata.get("section", "")
            )
            label = _answer_bullet_label(destination, section_title)
            bullet = f"{label} {normalized}" if label else normalized
            if not normalized or bullet in seen:
                continue
            seen.add(bullet)
            bullets.append(bullet)
            if len(bullets) >= 5:
                break

        source_lines = []
        for result in results:
            source = f"{result.source} / {result.destination}"
            if source not in source_lines:
                source_lines.append(source)

        answer_lines = [
            "基于当前目的地知识库，检索到以下相关信息：",
            *[f"- {bullet}" for bullet in bullets],
            "",
            "资料来源：",
            *[f"- {source}" for source in source_lines],
        ]
        return "\n".join(answer_lines)

    def _infer_destination(self, query: str) -> str | None:
        normalized_query = query.lower()
        known_destinations = self._known_destinations()

        for destination in known_destinations:
            if destination.lower() in normalized_query:
                return destination

        for alias, destination in DESTINATION_ALIASES.items():
            if alias.lower() in normalized_query and destination in known_destinations:
                return destination

        return None

    @staticmethod
    def _infer_section(query: str) -> str | None:
        normalized_query = query.lower()
        for section, aliases in SECTION_QUERY_ALIASES.items():
            if any(alias.lower() in normalized_query for alias in aliases):
                return section
        return None

    def _known_destinations(self) -> set[str]:
        destinations: set[str] = set()
        for metadata in vector_store_metadatas(self.vector_store):
            destination = metadata.get("destination")
            if isinstance(destination, str) and destination:
                destinations.add(destination)
        return destinations

    @staticmethod
    def _chunk_id(document: Document) -> str:
        source = str(document.metadata.get("source", ""))
        destination = str(document.metadata.get("destination", ""))
        start_index = str(document.metadata.get("start_index", ""))
        digest = hashlib.sha256(
            f"{source}:{destination}:{start_index}:{document.page_content}".encode()
        ).hexdigest()
        document.metadata["chunk_id"] = digest[:24]
        return digest[:24]


class RagRetriever:
    """Small invoke-compatible retriever that uses the service fallback stack."""

    def __init__(
        self,
        service: RagService,
        top_k: int,
        destination: str | None = None,
        section: str | None = None,
    ) -> None:
        self.service = service
        self.top_k = top_k
        self.destination = destination
        self.section = section

    def invoke(self, query: str) -> list[Document]:
        return [
            Document(page_content=result.content, metadata=result.metadata)
            for result in self.service.retrieve(
                query,
                top_k=self.top_k,
                destination=self.destination,
                section=self.section,
            )
        ]


def _split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    sentences: list[str] = []
    current = []
    for char in normalized:
        current.append(char)
        if char in "。！？!?":
            sentence = "".join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []
    tail = "".join(current).strip()
    if tail:
        sentences.append(tail)
    return sentences


def _rank_sentences(question: str, results: list[SearchResult]) -> list[tuple[SearchResult, str]]:
    query_tokens = _keyword_tokens(question)
    scored: list[tuple[int, int, SearchResult, str]] = []
    order = 0

    for result in results:
        for sentence in _split_sentences(result.content):
            tokens = _keyword_tokens(sentence)
            score = len(query_tokens & tokens)
            scored.append((score, -order, result, sentence))
            order += 1

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [(result, sentence) for _, _, result, sentence in scored]


def _rrf_fuse(
    result_lists: list[list[SearchResult]],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[SearchResult]:
    """Fuse ranked result lists using Reciprocal Rank Fusion."""

    scores: dict[str, float] = {}
    results_by_key: dict[str, SearchResult] = {}
    first_seen_order: dict[str, int] = {}
    order = 0

    weights = weights or [1.0] * len(result_lists)
    for list_index, results in enumerate(result_lists):
        weight = weights[list_index] if list_index < len(weights) else 1.0
        for rank, result in enumerate(results, start=1):
            key = _result_key(result)
            results_by_key.setdefault(key, result)
            if key not in first_seen_order:
                first_seen_order[key] = order
                order += 1
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)

    ranked_keys = sorted(
        scores,
        key=lambda key: (scores[key], -first_seen_order[key]),
        reverse=True,
    )
    return [
        SearchResult(
            content=results_by_key[key].content,
            source=results_by_key[key].source,
            destination=results_by_key[key].destination,
            score=scores[key],
            metadata=results_by_key[key].metadata,
        )
        for key in ranked_keys
    ]


def _result_key(result: SearchResult) -> str:
    chunk_id = result.metadata.get("chunk_id")
    if isinstance(chunk_id, str) and chunk_id:
        return chunk_id
    return f"{result.source}:{result.metadata.get('start_index', '')}:{result.content}"


def _retrieval_mode(value: str | RetrievalMode) -> RetrievalMode:
    return value if isinstance(value, RetrievalMode) else RetrievalMode(value)


def _answer_bullet_label(destination: str, section_title: str) -> str:
    display_destination = DESTINATION_DISPLAY_NAMES.get(destination, destination)
    parts = [part for part in (display_destination, section_title) if part]
    return f"[{' / '.join(parts)}]" if parts else ""


def _matches_metadata_filters(metadata: dict[str, object], filters: dict[str, str | None]) -> bool:
    return all(
        _matches_metadata_value(metadata.get(key), value)
        for key, value in filters.items()
        if value
    )


def _matches_metadata_value(actual: object, expected: str | None) -> bool:
    if not expected:
        return True
    if actual is None:
        return False

    expected_text = expected.strip().lower()
    actual_text = str(actual).strip().lower()
    if actual_text == expected_text:
        return True

    return expected_text in {
        value.strip()
        for value in re.split(r"[,;|]", actual_text)
        if value.strip()
    }


def _metadata_filters(
    destination: str | None,
    section: str | None,
    travel_type: str | None,
    season: str | None,
) -> dict[str, object]:
    filters: dict[str, object] = {}
    if destination:
        filters["destination"] = destination
    if section:
        filters["section"] = section
    if travel_type:
        filters["travel_type"] = travel_type
    season_key = _season_flag_key(season)
    if season_key:
        filters[season_key] = "true"
    return filters


def _clean_filters(filters: dict[str, object]) -> dict[str, str]:
    return {key: str(value) for key, value in filters.items() if value}


def _trace_hits(results: object) -> list[dict[str, object]]:
    if not isinstance(results, list):
        return []
    hits = []
    for rank, result in enumerate(results, start=1):
        if not isinstance(result, SearchResult):
            continue
        hits.append(
            {
                "rank": rank,
                "source": result.source,
                "section": str(result.metadata.get("section", "")),
                "score": result.score,
            }
        )
    return hits


def _empty_result_reason(
    collection_count: int,
    pre_filter_results: list[SearchResult],
    filtered_results: list[SearchResult],
    final_results: list[SearchResult],
) -> str:
    if final_results:
        return ""
    if collection_count <= 0:
        return "empty_collection"
    if not pre_filter_results:
        return "no_candidates_from_retrievers"
    if not filtered_results:
        return "metadata_filters_removed_all"
    return "rerank_or_min_score_removed_all"


def _season_flag_key(season: str | None) -> str | None:
    if not season:
        return None
    normalized = season.strip().lower()
    aliases = {
        "spring": "season_spring",
        "春": "season_spring",
        "春季": "season_spring",
        "summer": "season_summer",
        "夏": "season_summer",
        "夏季": "season_summer",
        "autumn": "season_autumn",
        "fall": "season_autumn",
        "秋": "season_autumn",
        "秋季": "season_autumn",
        "winter": "season_winter",
        "冬": "season_winter",
        "冬季": "season_winter",
    }
    return aliases.get(normalized)


def _chunks_by_document(chunks: list[Document]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        source = str(chunk.metadata.get("source", ""))
        counts[source] = counts.get(source, 0) + 1
    return counts


def _keyword_tokens(text: str) -> set[str]:
    normalized = text.lower()
    tokens = {char for char in normalized if "\u4e00" <= char <= "\u9fff"}
    for keyword in ("周末", "拥挤", "亲子", "酒店", "灵隐寺", "东京", "杭州", "迪士尼"):
        if keyword.lower() in normalized:
            tokens.add(keyword.lower())
    return tokens
