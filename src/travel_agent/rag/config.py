"""Runtime configuration for the LangChain + Chroma RAG module."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingProviderName(StrEnum):
    """Supported embedding backend modes."""

    AUTO = "auto"
    QWEN = "qwen"
    DASHSCOPE = "dashscope"
    OPENAI = "openai"
    LOCAL = "local"
    SENTENCE_TRANSFORMERS = "sentence-transformers"


class RetrievalMode(StrEnum):
    """Supported retrieval strategies."""

    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class KeywordTokenizerName(StrEnum):
    """Supported keyword tokenizer strategies for BM25 retrieval."""

    AUTO = "auto"
    BUILTIN = "builtin"
    JIEBA = "jieba"


class RerankerName(StrEnum):
    """Supported reranker strategies."""

    KEYWORD = "keyword"
    CROSS_ENCODER = "cross-encoder"
    BGE_RERANKER = "bge-reranker"


class RagSettings(BaseSettings):
    """Settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="TRAVEL_RAG_",
        extra="ignore",
    )

    persist_dir: Path = Path("data/chroma")
    collection_name: str = "travel_destinations"
    embedding_provider: EmbeddingProviderName = EmbeddingProviderName.AUTO
    qwen_embedding_model: str = "text-embedding-v4"
    qwen_embedding_dimensions: int = Field(default=1024, gt=0)
    qwen_embedding_batch_size: int = Field(default=10, gt=0)
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    openai_embedding_model: str = "text-embedding-3-small"
    sentence_transformers_model: str = "BAAI/bge-m3"
    local_embedding_dimensions: int = Field(default=512, gt=0)
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)
    default_top_k: int = 5
    retrieval_candidate_multiplier: int = Field(default=5, gt=0)
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    rrf_k: int = Field(default=60, gt=0)
    vector_weight: float = Field(default=1.0, ge=0)
    keyword_weight: float = Field(default=1.0, ge=0)
    min_score: float = Field(default=0.0, ge=0)
    keyword_tokenizer: KeywordTokenizerName = KeywordTokenizerName.AUTO
    keyword_user_dict: Path | None = None
    reranker: RerankerName = RerankerName.KEYWORD
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_batch_size: int = Field(default=16, gt=0)
    reranker_device: str | None = None
    reranker_fallback: bool = True

    @field_validator("keyword_user_dict", "reranker_device", mode="before")
    @classmethod
    def _empty_optional_strings_are_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def ensure_parent_dirs(self) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)


RagConfig = RagSettings
