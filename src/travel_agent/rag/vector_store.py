"""LangChain Chroma 向量存储工厂和辅助工具。"""

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
    """返回本项目使用的 LangChain Chroma 集成。

    优先使用社区版集成，因为 ChromaDB 1.x 的 Rust 后端在此 Windows 工作区中
    执行 upsert 时会崩溃。它仍然暴露标准的 LangChain VectorStore 和 Retriever 接口。
    """

    _disable_chroma_default_embedding()
    warnings.filterwarnings("ignore", message="The class `Chroma` was deprecated.*")

    from langchain_community.vectorstores import Chroma

    return Chroma


def _disable_chroma_default_embedding() -> None:
    """阻止 ChromaDB 0.4.x 为其未使用的默认嵌入导入 ONNX。"""

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
    """创建一个持久化的本地 Chroma 向量存储。"""

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
    """构建带可选元数据过滤的 LangChain 检索器。"""

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
    """构建 Chroma 兼容的等值过滤器。"""

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
    """返回跨受支持 LangChain 集成的 Chroma 集合大小。"""

    collection = getattr(vector_store, "_collection", None)
    if collection is not None and hasattr(collection, "count"):
        return int(collection.count())
    return 0


def vector_store_metadatas(vector_store: VectorStore) -> list[dict[str, object]]:
    """当集成暴露集合时，返回存储的 Chroma 元数据。"""

    collection = getattr(vector_store, "_collection", None)
    if collection is None or not hasattr(collection, "get"):
        return []
    payload = collection.get(include=["metadatas"])
    metadatas = payload.get("metadatas", []) if isinstance(payload, dict) else []
    return [metadata for metadata in metadatas if isinstance(metadata, dict)]


def vector_store_documents(vector_store: VectorStore) -> list[Document]:
    """返回存储的带元数据的 Chroma 文档，用于关键词检索。"""

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
    """删除 source 元数据匹配任何已扫描文档的存储块。"""

    return delete_documents_by_metadata(vector_store, "source", sources)


def delete_documents_by_document_id(vector_store: VectorStore, document_ids: list[str]) -> int:
    """删除 document_id 元数据匹配任何已扫描文档的存储块。"""

    return delete_documents_by_metadata(vector_store, "document_id", document_ids)


def delete_documents_by_metadata(vector_store: VectorStore, key: str, values: list[str]) -> int:
    """删除元数据键等于给定值之一的存储块。"""

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
    """删除持久化的 Chroma 目录以获得干净的本地索引。"""

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
