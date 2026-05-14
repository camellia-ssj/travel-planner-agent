"""旅行 RAG 知识文档的元数据辅助工具。"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field, field_validator

TRAVEL_METADATA_FIELDS = (
    "destination",
    "city",
    "country",
    "travel_type",
    "season",
    "source_type",
    "updated_at",
    "language",
    "source_url",
    "license",
    "last_verified_at",
    "poi_names",
    "geo_area",
    "price_level",
    "suitable_for",
    "season_spring",
    "season_summer",
    "season_autumn",
    "season_winter",
    "document_id",
    "document_hash",
    "file_type",
)

CHUNK_METADATA_FIELDS = (
    *TRAVEL_METADATA_FIELDS,
    "section",
    "section_title",
    "title",
    "source",
    "chunk_index",
    "chunk_id",
    "start_index",
)

SECTION_ALIASES = {
    "概览": "overview",
    "适合人群": "audience",
    "交通": "traffic",
    "玩法": "itinerary",
    "预算": "budget",
    "住宿": "lodging",
    "餐饮": "dining",
    "拥挤风险": "crowd_risk",
    "天气风险": "weather_risk",
    "备选方案": "alternatives",
    "风险提醒": "risk",
}

DEFAULT_SECTION = "overview"

_H2_HEADING_RE = re.compile(r"^##(?!#)\s+(.+?)\s*$", flags=re.MULTILINE)


class TravelDocumentMetadata(BaseModel):
    """与每个目的地文档一起持久化的经过验证的元数据模式。"""

    model_config = ConfigDict(extra="ignore")

    destination: str = Field(min_length=1)
    city: str = ""
    country: str = ""
    travel_type: str = ""
    season: str = ""
    source_type: str = ""
    updated_at: str = ""
    language: str = "zh"
    source_url: str = ""
    license: str = ""
    last_verified_at: str = ""
    poi_names: str = ""
    geo_area: str = ""
    price_level: str = ""
    suitable_for: str = ""
    title: str = ""
    source: str = ""
    document_id: str = ""
    document_hash: str = ""
    file_type: str = ""

    @field_validator("*", mode="before")
    @classmethod
    def _primitive_to_string(cls, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, int | float | bool):
            return str(value)
        raise TypeError("metadata values must be primitive scalars")


def split_markdown_sections(document: Document) -> list[Document]:
    """将一个 Markdown 文档按二级标题拆分为章节。"""

    text = document.page_content
    matches = list(_H2_HEADING_RE.finditer(text))
    if not matches:
        return [_section_document(document, text, DEFAULT_SECTION, _document_title(document))]

    sections: list[Document] = []
    if matches[0].start() > 0:
        preface = text[: matches[0].start()].strip()
        if preface:
            sections.append(
                _section_document(
                    document,
                    preface,
                    DEFAULT_SECTION,
                    _document_title(document),
                )
            )

    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_title = match.group(1).strip()
        section_text = text[match.start() : next_start].strip()
        if not section_text:
            continue
        sections.append(
            _section_document(
                document,
                section_text,
                _section_name(section_title),
                section_title,
            )
        )

    return sections


def split_documents_by_markdown_section(documents: list[Document]) -> list[Document]:
    """在分块之前将加载的文档拆分为章节文档。"""

    section_documents: list[Document] = []
    for document in documents:
        if document.metadata.get("file_type") == "markdown":
            section_documents.extend(split_markdown_sections(document))
        else:
            section_documents.append(
                _section_document(
                    document,
                    document.page_content,
                    DEFAULT_SECTION,
                    _document_title(document),
                )
            )
    return section_documents


def complete_document_metadata(metadata: dict[str, object], destination: str) -> dict[str, object]:
    """返回经过验证的文档元数据，包含所有旅行模式字段。"""

    normalized = dict(metadata)
    normalized["destination"] = destination
    normalized["city"] = normalized.get("city") or destination
    normalized["document_id"] = normalized.get("document_id") or normalized.get("source", "")
    schema = TravelDocumentMetadata.model_validate(normalized)
    dumped = schema.model_dump()
    dumped.update(_season_flags(dumped.get("season", "")))
    return dumped


def ensure_chunk_metadata(metadata: dict[str, object]) -> None:
    """确保每个文档块都暴露第一阶段元数据模式的所有键。"""

    metadata.setdefault("destination", "")
    metadata.setdefault("city", metadata.get("destination", ""))
    metadata.setdefault("country", "")
    metadata.setdefault("travel_type", "")
    metadata.setdefault("season", "")
    metadata.setdefault("source_type", "")
    metadata.setdefault("updated_at", "")
    metadata.setdefault("language", "zh")
    metadata.setdefault("source_url", "")
    metadata.setdefault("license", "")
    metadata.setdefault("last_verified_at", "")
    metadata.setdefault("poi_names", "")
    metadata.setdefault("geo_area", "")
    metadata.setdefault("price_level", "")
    metadata.setdefault("suitable_for", "")
    for key, value in _season_flags(str(metadata.get("season", ""))).items():
        metadata.setdefault(key, value)
    metadata.setdefault("document_id", metadata.get("source", ""))
    metadata.setdefault("document_hash", "")
    metadata.setdefault("file_type", "")
    metadata.setdefault("section", DEFAULT_SECTION)
    metadata.setdefault("section_title", "")
    metadata.setdefault("title", "")
    metadata.setdefault("source", "")
    metadata.setdefault("chunk_index", 0)
    metadata.setdefault("chunk_id", "")
    metadata.setdefault("start_index", 0)


def _section_document(
    document: Document,
    page_content: str,
    section: str,
    section_title: str,
) -> Document:
    metadata = dict(document.metadata)
    metadata["section"] = section
    metadata["section_title"] = section_title
    return Document(page_content=page_content, metadata=metadata)


def _section_name(section_title: str) -> str:
    return SECTION_ALIASES.get(section_title.strip(), "other")


def _document_title(document: Document) -> str:
    title = document.metadata.get("title")
    return title if isinstance(title, str) and title else DEFAULT_SECTION


def _season_flags(season: str) -> dict[str, str]:
    tokens = {
        value.strip().lower()
        for value in re.split(r"[,;|/，、\s]+", season)
        if value.strip()
    }
    return {
        "season_spring": _flag("spring" in tokens or "春" in tokens or "春季" in tokens),
        "season_summer": _flag("summer" in tokens or "夏" in tokens or "夏季" in tokens),
        "season_autumn": _flag(
            "autumn" in tokens or "fall" in tokens or "秋" in tokens or "秋季" in tokens
        ),
        "season_winter": _flag("winter" in tokens or "冬" in tokens or "冬季" in tokens),
    }


def _flag(value: bool) -> str:
    return "true" if value else "false"
