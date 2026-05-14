"""旅行目的地 RAG 模块的公开 Python API。

外部应用应优先使用此模块，而非导入 CLI 内部模块。
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
    """供外部 Python 调用者使用的小型外观类。"""

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
        """创建一个可复用的 RAG 客户端。"""

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
        """将受支持的文档导入本地 Chroma 知识库。"""

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
        """检索相关文档块。"""

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
        """返回结构化的检索证据和追踪元数据。"""

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
        """检索相关文档块并附带原始查询。"""

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
        """使用当前知识库检索到的文档块回答问题。"""

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
        """返回一个 LangChain 检索器，供外部检索管道使用。"""

        return self.service.as_retriever(destination=destination, top_k=top_k, section=section)

    def stats(self) -> dict[str, int | str]:
        """返回本地知识库统计信息。"""

        return self.service.stats()

    def reset(self) -> None:
        """清空本地 Chroma 知识库。"""

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
    """为外部调用者创建一个配置好的 `RagService`。"""

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
    """导入目的地知识文档的一次性辅助函数。"""

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
    """检索目的地知识块的一次性辅助函数。"""

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
    """检索目的地知识并附带查询元数据的一次性辅助函数。"""

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
    """为 Agent 节点返回结构化检索证据的一次性辅助函数。"""

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
    """从当前目的地知识库回答问题的一次性辅助函数。"""

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
    """返回一个 LangChain 检索器，供外部检索管道使用。"""

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
    """清空本地 Chroma 知识库。"""

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
    """返回本地知识库统计信息。"""

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
