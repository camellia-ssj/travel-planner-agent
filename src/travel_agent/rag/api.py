"""Public Python API for the travel destination RAG module.

External applications should prefer this module over importing CLI internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from travel_agent.rag.config import (
    EmbeddingProviderName,
    KeywordTokenizerName,
    RagSettings,
    RerankerName,
    RetrievalMode,
)
from travel_agent.rag.models import (
    AnswerResponse,
    EvidenceBundle,
    IngestReport,
    QueryResponse,
    QueryRewriteMode,
    SearchResult,
)
from travel_agent.rag.service import RagRetriever, RagService


@dataclass
class TravelRag:
    """Small facade for external Python callers."""

    service: RagService

    @classmethod
    def create(
        cls,
        persist_dir: str | Path | None = None,
        collection_name: str | None = None,
        embedding_provider: str | EmbeddingProviderName | None = None,
        top_k: int | None = None,
        retrieval_mode: str | RetrievalMode | None = None,
        min_score: float | None = None,
        keyword_tokenizer: str | KeywordTokenizerName | None = None,
        keyword_user_dict: str | Path | None = None,
        reranker: str | RerankerName | None = None,
        reranker_model: str | None = None,
        reranker_fallback: bool | None = None,
        query_rewrite: str | QueryRewriteMode | None = None,
    ) -> TravelRag:
        """Create a reusable RAG client."""

        return cls(
            service=create_rag_service(
                persist_dir=persist_dir,
                collection_name=collection_name,
                embedding_provider=embedding_provider,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                min_score=min_score,
                keyword_tokenizer=keyword_tokenizer,
                keyword_user_dict=keyword_user_dict,
                reranker=reranker,
                reranker_model=reranker_model,
                reranker_fallback=reranker_fallback,
                query_rewrite=query_rewrite,
            )
        )

    def ingest(
        self,
        path: str | Path,
        destination: str | None = None,
        incremental: bool = False,
    ) -> IngestReport:
        """Import supported documents into the local Chroma knowledge base."""

        return self.service.ingest_documents(
            Path(path),
            destination=destination,
            incremental=incremental,
        )

    def search(
        self,
        question: str,
        destination: str | None = None,
        top_k: int | None = None,
        section: str | None = None,
        travel_type: str | None = None,
        season: str | None = None,
        retrieval_mode: str | RetrievalMode | None = None,
    ) -> list[SearchResult]:
        """Retrieve relevant chunks."""

        return self.service.retrieve(
            question,
            destination=destination,
            top_k=top_k,
            section=section,
            travel_type=travel_type,
            season=season,
            retrieval_mode=retrieval_mode,
        )

    def retrieve_evidence(
        self,
        question: str,
        destination: str | None = None,
        top_k: int | None = None,
        section: str | None = None,
        travel_type: str | None = None,
        season: str | None = None,
        retrieval_mode: str | RetrievalMode | None = None,
    ) -> EvidenceBundle:
        """Return structured retrieval evidence and trace metadata."""

        return self.service.retrieve_evidence(
            question,
            destination=destination,
            top_k=top_k,
            section=section,
            travel_type=travel_type,
            season=season,
            retrieval_mode=retrieval_mode,
        )

    def query(
        self,
        question: str,
        destination: str | None = None,
        top_k: int | None = None,
        section: str | None = None,
        travel_type: str | None = None,
        season: str | None = None,
        retrieval_mode: str | RetrievalMode | None = None,
    ) -> QueryResponse:
        """Retrieve relevant chunks with the original query attached."""

        return self.service.query(
            question,
            destination=destination,
            top_k=top_k,
            section=section,
            travel_type=travel_type,
            season=season,
            retrieval_mode=retrieval_mode,
        )

    def ask(
        self,
        question: str,
        destination: str | None = None,
        top_k: int | None = None,
        section: str | None = None,
        travel_type: str | None = None,
        season: str | None = None,
        retrieval_mode: str | RetrievalMode | None = None,
    ) -> AnswerResponse:
        """Answer a question using retrieved chunks from the current knowledge base."""

        return self.service.answer(
            question,
            destination=destination,
            top_k=top_k,
            section=section,
            travel_type=travel_type,
            season=season,
            retrieval_mode=retrieval_mode,
        )

    def as_retriever(
        self,
        destination: str | None = None,
        top_k: int | None = None,
        section: str | None = None,
    ) -> RagRetriever:
        """Return a LangChain retriever for external retrieval pipelines."""

        return self.service.as_retriever(destination=destination, top_k=top_k, section=section)

    def stats(self) -> dict[str, int | str]:
        """Return local knowledge-base stats."""

        return self.service.stats()

    def reset(self) -> None:
        """Clear the local Chroma knowledge base."""

        self.service.reset()


def create_rag_service(
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
    embedding_provider: str | EmbeddingProviderName | None = None,
    top_k: int | None = None,
    retrieval_mode: str | RetrievalMode | None = None,
    min_score: float | None = None,
    keyword_tokenizer: str | KeywordTokenizerName | None = None,
    keyword_user_dict: str | Path | None = None,
    reranker: str | RerankerName | None = None,
    reranker_model: str | None = None,
    reranker_fallback: bool | None = None,
    query_rewrite: str | QueryRewriteMode | None = None,
) -> RagService:
    """Create a configured `RagService` for external callers."""

    settings = RagSettings()
    updates: dict[str, object] = {}
    if persist_dir is not None:
        updates["persist_dir"] = Path(persist_dir)
    if collection_name is not None:
        updates["collection_name"] = collection_name
    if embedding_provider is not None:
        updates["embedding_provider"] = _embedding_provider(embedding_provider)
    if top_k is not None:
        updates["default_top_k"] = top_k
    if retrieval_mode is not None:
        updates["retrieval_mode"] = _retrieval_mode(retrieval_mode)
    if min_score is not None:
        updates["min_score"] = min_score
    if keyword_tokenizer is not None:
        updates["keyword_tokenizer"] = _keyword_tokenizer(keyword_tokenizer)
    if keyword_user_dict is not None:
        updates["keyword_user_dict"] = Path(keyword_user_dict)
    if reranker is not None:
        updates["reranker"] = _reranker(reranker)
    if reranker_model is not None:
        updates["reranker_model"] = reranker_model
    if reranker_fallback is not None:
        updates["reranker_fallback"] = reranker_fallback
    if query_rewrite is not None:
        updates["query_rewrite"] = _query_rewrite_mode(query_rewrite)

    if updates:
        settings = settings.model_copy(update=updates)
    return RagService(settings=settings)


def ingest_destination_documents(
    path: str | Path,
    destination: str | None = None,
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
    embedding_provider: str | EmbeddingProviderName | None = None,
    incremental: bool = False,
) -> IngestReport:
    """One-shot helper for importing destination knowledge documents."""

    service = create_rag_service(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
    )
    return service.ingest_documents(Path(path), destination=destination, incremental=incremental)


def search_destination_knowledge(
    question: str,
    destination: str | None = None,
    top_k: int | None = None,
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
    embedding_provider: str | EmbeddingProviderName | None = None,
    section: str | None = None,
    travel_type: str | None = None,
    season: str | None = None,
    retrieval_mode: str | RetrievalMode | None = None,
) -> list[SearchResult]:
    """One-shot helper for retrieving destination knowledge chunks."""

    service = create_rag_service(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
    )
    return service.retrieve(
        question,
        destination=destination,
        top_k=top_k,
        section=section,
        travel_type=travel_type,
        season=season,
        retrieval_mode=retrieval_mode,
    )


def query_destination_knowledge(
    question: str,
    destination: str | None = None,
    top_k: int | None = None,
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
    embedding_provider: str | EmbeddingProviderName | None = None,
    section: str | None = None,
    travel_type: str | None = None,
    season: str | None = None,
    retrieval_mode: str | RetrievalMode | None = None,
) -> QueryResponse:
    """One-shot helper for retrieving destination knowledge with query metadata."""

    service = create_rag_service(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
    )
    return service.query(
        question,
        destination=destination,
        top_k=top_k,
        section=section,
        travel_type=travel_type,
        season=season,
        retrieval_mode=retrieval_mode,
    )


def retrieve_destination_evidence(
    question: str,
    destination: str | None = None,
    top_k: int | None = None,
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
    embedding_provider: str | EmbeddingProviderName | None = None,
    section: str | None = None,
    travel_type: str | None = None,
    season: str | None = None,
    retrieval_mode: str | RetrievalMode | None = None,
) -> EvidenceBundle:
    """One-shot helper returning structured retrieval evidence for Agent nodes."""

    service = create_rag_service(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
    )
    return service.retrieve_evidence(
        question,
        destination=destination,
        top_k=top_k,
        section=section,
        travel_type=travel_type,
        season=season,
        retrieval_mode=retrieval_mode,
    )


def answer_destination_question(
    question: str,
    destination: str | None = None,
    top_k: int | None = None,
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
    embedding_provider: str | EmbeddingProviderName | None = None,
    section: str | None = None,
    travel_type: str | None = None,
    season: str | None = None,
    retrieval_mode: str | RetrievalMode | None = None,
) -> AnswerResponse:
    """One-shot helper for answering from the current destination knowledge base."""

    service = create_rag_service(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
    )
    return service.answer(
        question,
        destination=destination,
        top_k=top_k,
        section=section,
        travel_type=travel_type,
        season=season,
        retrieval_mode=retrieval_mode,
    )


def get_destination_retriever(
    destination: str | None = None,
    top_k: int | None = None,
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
    embedding_provider: str | EmbeddingProviderName | None = None,
    section: str | None = None,
) -> RagRetriever:
    """Return a LangChain retriever for external retrieval pipelines."""

    service = create_rag_service(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        top_k=top_k,
    )
    return service.as_retriever(destination=destination, top_k=top_k, section=section)


def reset_knowledge_base(
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
    embedding_provider: str | EmbeddingProviderName | None = None,
) -> None:
    """Clear the local Chroma knowledge base."""

    service = create_rag_service(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
    )
    service.reset()


def get_knowledge_base_stats(
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
    embedding_provider: str | EmbeddingProviderName | None = None,
) -> dict[str, int | str]:
    """Return local knowledge-base stats."""

    service = create_rag_service(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
    )
    return service.stats()


def _embedding_provider(value: str | EmbeddingProviderName) -> EmbeddingProviderName:
    if isinstance(value, EmbeddingProviderName):
        return value
    return EmbeddingProviderName(value)


def _retrieval_mode(value: str | RetrievalMode) -> RetrievalMode:
    if isinstance(value, RetrievalMode):
        return value
    return RetrievalMode(value)


def _keyword_tokenizer(value: str | KeywordTokenizerName) -> KeywordTokenizerName:
    if isinstance(value, KeywordTokenizerName):
        return value
    return KeywordTokenizerName(value)


def _reranker(value: str | RerankerName) -> RerankerName:
    if isinstance(value, RerankerName):
        return value
    return RerankerName(value)


def _query_rewrite_mode(value: str | QueryRewriteMode) -> QueryRewriteMode:
    if isinstance(value, QueryRewriteMode):
        return value
    return QueryRewriteMode(value)
