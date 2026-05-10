"""Small helpers for code that consumes LangChain documents."""

from __future__ import annotations

from langchain_core.documents import Document

from travel_agent.rag.models import SearchResult


def search_result_to_document(result: SearchResult) -> Document:
    """Convert a user-facing search result back into a LangChain Document."""

    return Document(
        page_content=result.content,
        metadata={
            **result.metadata,
            "source": result.source,
            "destination": result.destination,
            "score": result.score,
        },
    )
