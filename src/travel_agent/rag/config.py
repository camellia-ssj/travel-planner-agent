"""Runtime configuration for the LangChain + Chroma RAG module."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field
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

    def ensure_parent_dirs(self) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)


RagConfig = RagSettings
