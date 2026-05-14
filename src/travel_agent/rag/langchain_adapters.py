"""消费 LangChain 文档的小型辅助工具。"""

from __future__ import annotations

from langchain_core.documents import Document

from travel_agent.rag.models import SearchResult


def search_result_to_document(result: SearchResult) -> Document:
    """将面向用户的搜索结果转换回 LangChain Document。"""

    return Document(
        page_content=result.content,
        metadata={
            **result.metadata,
            "source": result.source,
            "destination": result.destination,
            "score": result.score,
        },
    )
