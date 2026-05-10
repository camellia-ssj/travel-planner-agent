"""LangChain embedding providers for the travel RAG module."""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings

from travel_agent.rag.config import EmbeddingProviderName, RagSettings


class LocalHashEmbeddings(Embeddings):
    """Deterministic test/demo embedding.

    This is intentionally not a production semantic embedding model. It keeps
    unit tests and offline demos deterministic when no external embedding
    provider is configured.
    """

    def __init__(self, dimensions: int = 512) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class SentenceTransformersEmbeddings(Embeddings):
    """Local multilingual embedding provider backed by sentence-transformers."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for "
                "TRAVEL_RAG_EMBEDDING_PROVIDER=sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode([text], normalize_embeddings=True)[0]
        return vector.tolist()


def build_embeddings(settings: RagSettings) -> Embeddings:
    """Build the configured embedding provider with deterministic local fallback."""

    provider = settings.embedding_provider
    if provider in {EmbeddingProviderName.QWEN, EmbeddingProviderName.DASHSCOPE}:
        return _qwen_embeddings(settings)

    if provider == EmbeddingProviderName.OPENAI:
        return _openai_embeddings(settings.openai_embedding_model)

    if provider == EmbeddingProviderName.SENTENCE_TRANSFORMERS:
        return SentenceTransformersEmbeddings(settings.sentence_transformers_model)

    if provider == EmbeddingProviderName.AUTO and os.getenv("DASHSCOPE_API_KEY"):
        return _qwen_embeddings(settings)

    if provider == EmbeddingProviderName.AUTO and os.getenv("OPENAI_API_KEY"):
        return _openai_embeddings(settings.openai_embedding_model)

    return LocalHashEmbeddings(dimensions=settings.local_embedding_dimensions)


def _qwen_embeddings(settings: RagSettings) -> Embeddings:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is required for "
            "TRAVEL_RAG_EMBEDDING_PROVIDER=qwen or dashscope"
        )

    return _openai_embeddings_class()(
        model=settings.qwen_embedding_model,
        api_key=api_key,
        base_url=settings.qwen_base_url,
        dimensions=settings.qwen_embedding_dimensions,
        chunk_size=settings.qwen_embedding_batch_size,
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"},
    )


def _openai_embeddings(model: str) -> Embeddings:
    return _openai_embeddings_class()(model=model)


def _openai_embeddings_class() -> type[Embeddings]:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings


def _tokenize(text: str) -> Sequence[str]:
    lowered = text.lower()
    latin_tokens = re.findall(r"[a-z0-9]+", lowered)
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]", lowered)
    bigrams = [lowered[index : index + 2] for index in range(max(len(lowered) - 1, 0))]
    return latin_tokens + cjk_tokens + bigrams
