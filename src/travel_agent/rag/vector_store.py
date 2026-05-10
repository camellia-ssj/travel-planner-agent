"""LangChain Chroma vector store factory and helpers."""

from __future__ import annotations

import importlib
import importlib.util
import shutil
import sys
import warnings
from pathlib import Path
from types import ModuleType

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever

from travel_agent.rag.config import RagSettings


def chroma_class() -> type[VectorStore]:
    """Return the LangChain Chroma integration used by this project.

    The community integration is preferred because ChromaDB 1.x's Rust backend
    crashes on upsert in this Windows workspace. It still exposes the standard
    LangChain VectorStore and Retriever interfaces.
    """

    _disable_chroma_default_embedding()
    warnings.filterwarnings("ignore", message="The class `Chroma` was deprecated.*")

    from langchain_community.vectorstores import Chroma

    return Chroma


def _disable_chroma_default_embedding() -> None:
    """Prevent ChromaDB 0.4.x from importing ONNX for its unused default embedding."""

    if "chromadb.utils.embedding_functions" in sys.modules:
        return
    utils_module = sys.modules.setdefault("chromadb.utils", ModuleType("chromadb.utils"))
    utils_module.__path__ = [_chromadb_utils_path()]  # type: ignore[attr-defined]
    utils_module.get_class = _get_class
    embedding_module = ModuleType("chromadb.utils.embedding_functions")
    embedding_module.DefaultEmbeddingFunction = lambda: None
    embedding_module.get_builtins = lambda: []
    utils_module.embedding_functions = embedding_module
    sys.modules["chromadb.utils.embedding_functions"] = embedding_module


def _get_class(fqn: str, type: object) -> object:
    module_name, class_name = fqn.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _chromadb_utils_path() -> str:
    spec = importlib.util.find_spec("chromadb")
    if spec is None or spec.origin is None:
        return ""
    return str(Path(spec.origin).parent / "utils")


def build_vector_store(settings: RagSettings, embeddings: Embeddings) -> VectorStore:
    """Create a persistent local Chroma vector store."""

    settings.ensure_parent_dirs()
    chroma = chroma_class()
    _disable_chroma_telemetry()
    from chromadb.config import Settings

    client_settings = Settings(
        is_persistent=True,
        anonymized_telemetry=False,
        persist_directory=str(settings.persist_dir),
    )
    return chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.persist_dir),
        client_settings=client_settings,
    )


def build_retriever(
    vector_store: VectorStore,
    top_k: int,
    destination: str | None = None,
    section: str | None = None,
) -> VectorStoreRetriever:
    """Build a LangChain retriever with optional metadata filtering."""

    search_kwargs: dict[str, object] = {"k": top_k}
    metadata_filter = build_chroma_filter(
        {
            "destination": destination,
            "section": section,
        }
    )
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter
    return vector_store.as_retriever(search_kwargs=search_kwargs)


def build_chroma_filter(filters: dict[str, object | None]) -> dict[str, object]:
    """Build a Chroma-compatible equality filter."""

    clean_filters = {
        key: value
        for key, value in filters.items()
        if value is not None and value != ""
    }
    if not clean_filters:
        return {}
    if len(clean_filters) == 1:
        key, value = next(iter(clean_filters.items()))
        return {key: value}
    return {"$and": [{key: value} for key, value in clean_filters.items()]}


def vector_store_count(vector_store: VectorStore) -> int:
    """Return Chroma collection size across supported LangChain integrations."""

    collection = getattr(vector_store, "_collection", None)
    if collection is not None and hasattr(collection, "count"):
        return int(collection.count())
    return 0


def vector_store_metadatas(vector_store: VectorStore) -> list[dict[str, object]]:
    """Return stored Chroma metadatas when the integration exposes the collection."""

    collection = getattr(vector_store, "_collection", None)
    if collection is None or not hasattr(collection, "get"):
        return []
    payload = collection.get(include=["metadatas"])
    metadatas = payload.get("metadatas", []) if isinstance(payload, dict) else []
    return [metadata for metadata in metadatas if isinstance(metadata, dict)]


def vector_store_documents(vector_store: VectorStore) -> list[Document]:
    """Return stored Chroma documents with metadata for keyword retrieval."""

    collection = getattr(vector_store, "_collection", None)
    if collection is None or not hasattr(collection, "get"):
        return []
    payload = collection.get(include=["documents", "metadatas"])
    if not isinstance(payload, dict):
        return []
    documents = payload.get("documents", [])
    metadatas = payload.get("metadatas", [])
    if not isinstance(documents, list) or not isinstance(metadatas, list):
        return []

    stored_documents: list[Document] = []
    for content, metadata in zip(documents, metadatas, strict=False):
        if not isinstance(content, str):
            continue
        stored_documents.append(
            Document(
                page_content=content,
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        )
    return stored_documents


def delete_documents_by_source(vector_store: VectorStore, sources: list[str]) -> int:
    """Delete stored chunks whose source metadata matches any scanned document."""

    return delete_documents_by_metadata(vector_store, "source", sources)


def delete_documents_by_document_id(vector_store: VectorStore, document_ids: list[str]) -> int:
    """Delete stored chunks whose document_id metadata matches any scanned document."""

    return delete_documents_by_metadata(vector_store, "document_id", document_ids)


def delete_documents_by_metadata(vector_store: VectorStore, key: str, values: list[str]) -> int:
    """Delete stored chunks whose metadata key equals one of the given values."""

    collection = getattr(vector_store, "_collection", None)
    if collection is None or not hasattr(collection, "get") or not hasattr(collection, "delete"):
        return 0

    deleted = 0
    for value in values:
        if not value:
            continue
        payload = collection.get(where={key: value})
        if not isinstance(payload, dict):
            continue
        ids = payload.get("ids", [])
        if not isinstance(ids, list) or not ids:
            continue
        collection.delete(ids=ids)
        deleted += len(ids)
    return deleted


def reset_chroma(settings: RagSettings) -> None:
    """Delete the persisted Chroma directory for a clean local index."""

    persist_dir = Path(settings.persist_dir).resolve()
    if persist_dir.exists():
        shutil.rmtree(persist_dir)


def _disable_chroma_telemetry() -> None:
    try:
        import posthog
    except ImportError:
        return
    posthog.disabled = True
    posthog.capture = lambda *args, **kwargs: None
