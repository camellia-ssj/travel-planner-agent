"""LangChain document loading for destination knowledge files."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from pydantic import ValidationError

from travel_agent.rag.metadata import complete_document_metadata

MARKDOWN_EXTENSIONS = {".md", ".markdown"}
TEXT_EXTENSIONS = {".txt"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_DOCUMENT_EXTENSIONS = MARKDOWN_EXTENSIONS | TEXT_EXTENSIONS | PDF_EXTENSIONS


def discover_document_files(path: Path) -> list[Path]:
    """Return supported knowledge files under a file or directory path."""

    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS else []
    if not path.exists():
        raise FileNotFoundError(f"Document path does not exist: {path}")
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
    )


def discover_markdown_files(path: Path) -> list[Path]:
    """Return Markdown files under a file or directory path."""

    if path.is_file():
        return [path] if path.suffix.lower() in MARKDOWN_EXTENSIONS else []
    if not path.exists():
        raise FileNotFoundError(f"Document path does not exist: {path}")
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in MARKDOWN_EXTENSIONS
    )


def load_documents(path: Path, destination: str | None = None) -> list[Document]:
    """Load supported knowledge documents and enrich metadata."""

    files = discover_document_files(path)
    if not files:
        return []

    root = path.resolve() if path.is_dir() else path.resolve().parent
    documents = [_load_file(file) for file in files]
    return _enrich_documents(documents, path, root, destination)


def load_markdown_documents(path: Path, destination: str | None = None) -> list[Document]:
    """Load Markdown documents with LangChain loaders and enrich metadata."""

    files = discover_markdown_files(path)
    if not files:
        return []

    root = path.resolve() if path.is_dir() else path.resolve().parent
    documents = [_load_file(file) for file in files]
    return _enrich_documents(documents, path, root, destination)


def _enrich_documents(
    documents: list[Document],
    path: Path,
    root: Path,
    destination: str | None,
) -> list[Document]:
    """Populate normalized travel RAG metadata on loaded documents."""

    for document in documents:
        source = str(document.metadata.get("source", ""))
        source_path = Path(source) if source else path
        safe_source = _safe_source(source_path, root)
        raw_content = document.page_content
        document_hash = _content_hash(raw_content)
        try:
            front_matter = _parse_front_matter(document.page_content)
        except ValueError as exc:
            raise ValueError(f"Invalid front matter in {safe_source}: {exc}") from exc
        document.page_content = _strip_front_matter(document.page_content)
        resolved_destination = (
            destination
            or _string_value(front_matter.get("destination"))
            or _string_value(front_matter.get("city"))
            or source_path.stem
        )

        metadata: dict[str, object] = {
            "source": safe_source,
            "title": _title(document.page_content, source_path),
            "document_id": safe_source,
            "document_hash": document_hash,
            "file_type": _file_type(source_path),
        }
        metadata.update(_scalar_metadata(front_matter))
        try:
            document.metadata.update(complete_document_metadata(metadata, resolved_destination))
        except ValidationError as exc:
            raise ValueError(f"Invalid metadata in {safe_source}: {exc}") from exc

    return documents


def _load_file(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix in MARKDOWN_EXTENSIONS | TEXT_EXTENSIONS:
        return TextLoader(str(path), encoding="utf-8").load()[0]
    if suffix in PDF_EXTENSIONS:
        return Document(page_content=_load_pdf_text(path), metadata={"source": str(path)})
    raise ValueError(f"Unsupported document type: {path.suffix}")


def _load_pdf_text(path: Path) -> str:
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires langchain-community PDF support.") from exc

    pages = PyPDFLoader(str(path)).load()
    return "\n\n".join(page.page_content.strip() for page in pages if page.page_content.strip())


def _title(text: str, path: Path) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or path.stem
    return path.stem


def _parse_front_matter(text: str) -> dict[str, object]:
    if not text.startswith("---"):
        return {}
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, flags=re.DOTALL)
    if not match:
        return {}

    try:
        import yaml
    except ImportError:
        return _parse_simple_front_matter(match.group(1))

    payload = yaml.safe_load(match.group(1)) or {}
    if not isinstance(payload, dict):
        raise ValueError("front matter must be a mapping")
    return {str(key): value for key, value in payload.items()}


def _parse_simple_front_matter(text: str) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip("\"'")
    return metadata


def _strip_front_matter(text: str) -> str:
    if not text.startswith("---"):
        return text
    return re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.DOTALL)


def _string_value(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _safe_source(source_path: Path, root: Path) -> str:
    try:
        return source_path.resolve().relative_to(root).as_posix()
    except ValueError:
        return source_path.name


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _file_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in MARKDOWN_EXTENSIONS:
        return "markdown"
    if suffix in TEXT_EXTENSIONS:
        return "text"
    if suffix in PDF_EXTENSIONS:
        return "pdf"
    return suffix.lstrip(".")


def _scalar_metadata(metadata: dict[str, object]) -> dict[str, object]:
    return {key: _metadata_value(value) for key, value in metadata.items()}


def _metadata_value(value: object) -> str | int | float | bool:
    if value is None:
        return ""
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple | set):
        return ",".join(_metadata_text(item) for item in value)
    if isinstance(value, dict):
        return ",".join(f"{key}:{_metadata_text(item)}" for key, item in value.items())
    return str(value)


def _metadata_text(value: Any) -> str:
    return "" if value is None else str(value).strip()
