"""Ingest manifest for document version tracking."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from travel_agent.rag.config import RagSettings


@dataclass(frozen=True)
class ManifestDocument:
    """One indexed source document recorded in the manifest."""

    source: str
    document_id: str
    document_hash: str
    destination: str
    chunk_count: int
    updated_at: str
    indexed_at: str


class IngestManifest:
    """Small JSON manifest persisted next to the local Chroma index."""

    def __init__(self, path: Path, documents: dict[str, ManifestDocument] | None = None) -> None:
        self.path = path
        self.documents = documents or {}

    @property
    def collection_version(self) -> str:
        if not self.documents:
            return "empty"
        payload = "|".join(
            f"{item.source}:{item.document_hash}:{item.chunk_count}"
            for item in sorted(self.documents.values(), key=lambda document: document.source)
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def unchanged(self, source: str, document_hash: str) -> bool:
        existing = self.documents.get(source)
        return existing is not None and existing.document_hash == document_hash

    def update(self, document: ManifestDocument) -> None:
        self.documents[document.source] = document

    def remove(self, source: str) -> None:
        self.documents.pop(source, None)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "collection_version": self.collection_version,
            "documents": [document.__dict__ for document in self.documents.values()],
        }
        temp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.path)


def manifest_path(settings: RagSettings) -> Path:
    return Path(settings.persist_dir) / "ingest_manifest.json"


def load_manifest(settings: RagSettings) -> IngestManifest:
    path = manifest_path(settings)
    if not path.exists():
        return IngestManifest(path)

    try:
        raw_payload = path.read_text(encoding="utf-8").strip()
    except OSError:
        return IngestManifest(path)
    if not raw_payload:
        return IngestManifest(path)

    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError:
        return IngestManifest(path)
    if not isinstance(payload, dict):
        return IngestManifest(path)

    documents: dict[str, ManifestDocument] = {}
    for item in payload.get("documents", []):
        if not isinstance(item, dict):
            continue
        document = ManifestDocument(
            source=str(item.get("source", "")),
            document_id=str(item.get("document_id", "")),
            document_hash=str(item.get("document_hash", "")),
            destination=str(item.get("destination", "")),
            chunk_count=int(item.get("chunk_count", 0)),
            updated_at=str(item.get("updated_at", "")),
            indexed_at=str(item.get("indexed_at", "")),
        )
        if document.source:
            documents[document.source] = document
    return IngestManifest(path, documents)


def manifest_document(metadata: dict[str, Any], chunk_count: int) -> ManifestDocument:
    return ManifestDocument(
        source=str(metadata.get("source", "")),
        document_id=str(metadata.get("document_id", "")),
        document_hash=str(metadata.get("document_hash", "")),
        destination=str(metadata.get("destination", "")),
        chunk_count=chunk_count,
        updated_at=str(metadata.get("updated_at", "")),
        indexed_at=datetime.now(UTC).isoformat(),
    )
