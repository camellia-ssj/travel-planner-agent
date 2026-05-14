"""旅行目的地知识检索的纯 RAG 模块。"""

from travel_agent.rag.api import (
    TravelRag,
    answer_destination_question,
    create_rag_service,
    get_destination_retriever,
    get_knowledge_base_stats,
    ingest_destination_documents,
    query_destination_knowledge,
    reset_knowledge_base,
    retrieve_destination_evidence,
    search_destination_knowledge,
)
from travel_agent.rag.config import (
    EmbeddingProviderName,
    KeywordTokenizerName,
    RagConfig,
    RerankerName,
    RetrievalMode,
)
from travel_agent.rag.evaluation import EvalReport, evaluate_rag
from travel_agent.rag.models import (
    AnswerResponse,
    EvidenceBundle,
    IngestReport,
    QueryResponse,
    QueryRewriteMode,
    QueryRewriteResult,
    RetrievalTrace,
    SearchResult,
)
from travel_agent.rag.query_rewrite import (
    LLMQueryRewriter,
    build_query_rewriter,
    search_with_query_rewrites,
)
from travel_agent.rag.service import RagRetriever, RagService

__all__ = [
    "IngestReport",
    "KeywordTokenizerName",
    "LLMQueryRewriter",
    "QueryResponse",
    "QueryRewriteMode",
    "QueryRewriteResult",
    "RagConfig",
    "RerankerName",
    "RagService",
    "RagRetriever",
    "RetrievalMode",
    "SearchResult",
    "TravelRag",
    "AnswerResponse",
    "EvidenceBundle",
    "EvalReport",
    "EmbeddingProviderName",
    "RetrievalTrace",
    "answer_destination_question",
    "build_query_rewriter",
    "create_rag_service",
    "evaluate_rag",
    "get_destination_retriever",
    "get_knowledge_base_stats",
    "ingest_destination_documents",
    "query_destination_knowledge",
    "retrieve_destination_evidence",
    "reset_knowledge_base",
    "search_destination_knowledge",
    "search_with_query_rewrites",
]
