"""LangChain 文本分割工具。"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from travel_agent.rag.config import RagSettings


def build_text_splitter(settings: RagSettings) -> RecursiveCharacterTextSplitter:
    """构建项目标准的 LangChain 文本分割器。"""

    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
    )
