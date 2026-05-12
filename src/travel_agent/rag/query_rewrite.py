"""LLM-driven query rewriting for travel-domain RAG retrieval.

When enabled, this module rewrites natural-language user questions into one or
more retrieval-friendly queries.  Multi-query mode fuses results from the
original plus rewritten queries via reciprocal rank fusion.
"""

from __future__ import annotations

import json
import os
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from travel_agent.rag.models import QueryRewriteMode, QueryRewriteResult

if TYPE_CHECKING:
    from travel_agent.rag.models import SearchResult
    from travel_agent.rag.service import RagService

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_REWRITE_SYSTEM_PROMPT = (
    "You are a travel-domain query rewriter. Your job is to convert a user's "
    "natural-language travel question into one or more concise, keyword-rich "
    "retrieval queries that will perform well against a destination knowledge base.\n\n"
    "Rules:\n"
    "- Normalise destination names (e.g. 「杭洲」「西湖那边」→「杭州西湖」).\n"
    "- Extract key constraints: days, budget, audience (family/elderly/solo/couple), "
    "season, preferences.\n"
    "- For multi-query mode generate 2-3 diverse queries covering different angles "
    "(e.g. itinerary, budget, crowd risk, weather, alternatives).\n"
    "- Never add facts the user didn't say.\n"
    "- Write queries in Chinese when the input is Chinese; English when English.\n"
    "- Output MUST be a single JSON object, no markdown fences, no commentary."
)

_REWRITE_USER_PROMPT = (
    "Rewrite the following travel question for knowledge-base retrieval.\n\n"
    "Original question: {question}\n\n"
    "Rewrite mode: {mode}\n\n"
    'Output JSON: {{"rewritten_query": "...", "search_queries": ["..."], "notes": ["..."]}}'
)

# ---------------------------------------------------------------------------
# Query rewriter
# ---------------------------------------------------------------------------


@dataclass
class LLMQueryRewriter:
    """Wraps a LangChain chat model to rewrite travel queries before retrieval.

    Parameters
    ----------
    model:
        A pre-built LangChain chat model.  When *None* the rewriter tries to
        build one lazily via *build_fn*.
    build_fn:
        Optional zero-argument callable that returns a chat model on demand.
        Only called on the first ``rewrite()`` invocation when *model* is None.
    """

    model: Any | None = None
    build_fn: Any | None = None
    _resolved: bool = False

    def _ensure_model(self) -> Any | None:
        if self._resolved:
            return self.model
        object.__setattr__(self, "_resolved", True)
        if self.model is not None:
            return self.model
        if self.build_fn is not None:
            with suppress(Exception):
                object.__setattr__(self, "model", self.build_fn())
        return self.model

    def rewrite(self, query: str, mode: QueryRewriteMode | str) -> QueryRewriteResult:
        """Rewrite *query* and return structured result.

        When the rewriter has no model, or the model call fails, the original
        query is returned as-is.
        """
        resolved_mode = _resolve_mode(mode)
        if resolved_mode is QueryRewriteMode.OFF:
            return QueryRewriteResult(
                original_query=query,
                rewritten_query=query,
                search_queries=[query],
                notes=["rewrite disabled"],
            )

        model = self._ensure_model()
        if model is None:
            return QueryRewriteResult(
                original_query=query,
                rewritten_query=query,
                search_queries=[query],
                notes=["no model available, using original query"],
            )

        try:
            result = _call_rewrite_model(model, query, resolved_mode.value)
        except Exception:
            return QueryRewriteResult(
                original_query=query,
                rewritten_query=query,
                search_queries=[query],
                notes=["rewrite model call failed, using original query"],
            )

        search_queries = _build_search_queries(query, result, resolved_mode)
        return QueryRewriteResult(
            original_query=query,
            rewritten_query=result.get("rewritten_query", query),
            search_queries=search_queries,
            notes=result.get("notes", []),
            raw_response=json.dumps(result, ensure_ascii=False),
        )


def _call_rewrite_model(
    model: Any,
    question: str,
    mode_str: str,
) -> dict[str, Any]:
    """Call the LLM and parse its JSON response."""
    from langchain_core.messages import HumanMessage, SystemMessage

    response = model.invoke([
        SystemMessage(content=_REWRITE_SYSTEM_PROMPT),
        HumanMessage(content=_REWRITE_USER_PROMPT.format(question=question, mode=mode_str)),
    ])
    text = _extract_message_text(response)
    return _parse_rewrite_json(text, question)


def _extract_message_text(response: Any) -> str:
    """Extract text content from various LangChain response shapes."""
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
    return str(response)


def _parse_rewrite_json(text: str, fallback_query: str) -> dict[str, Any]:
    """Robust JSON extraction from LLM output."""
    cleaned = text.strip()
    # Remove occasional markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find a JSON object by scanning for braces
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            try:
                result = json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                return {"rewritten_query": fallback_query}
        else:
            return {"rewritten_query": fallback_query}

    if not isinstance(result, dict):
        return {"rewritten_query": fallback_query}
    return result


def _build_search_queries(
    original_query: str,
    result: dict[str, Any],
    mode: QueryRewriteMode,
) -> list[str]:
    """Assemble the final list of search queries based on mode."""
    rewritten = result.get("rewritten_query", original_query) or original_query

    if mode is QueryRewriteMode.REWRITE_ONLY:
        return [rewritten]

    candidates = result.get("search_queries", [rewritten])
    if not isinstance(candidates, list):
        candidates = [rewritten]

    # Clean: drop empty / whitespace-only entries
    cleaned: list[str] = []
    seen: set[str] = set()
    for q in candidates:
        if not isinstance(q, str) or not q.strip():
            continue
        norm = q.strip()
        if norm not in seen:
            seen.add(norm)
            cleaned.append(norm)

    if not cleaned:
        return [original_query]

    # Ensure rewritten_query is first
    if rewritten.strip() not in seen:
        cleaned.insert(0, rewritten.strip())

    # Always include the original query for recall
    if original_query.strip() not in seen:
        cleaned.append(original_query.strip())

    return cleaned[:3]


def _resolve_mode(mode: QueryRewriteMode | str) -> QueryRewriteMode:
    if isinstance(mode, QueryRewriteMode):
        resolved = mode
    else:
        try:
            resolved = QueryRewriteMode(mode)
        except ValueError:
            resolved = QueryRewriteMode.OFF
    if resolved is QueryRewriteMode.ON:
        resolved = QueryRewriteMode.MULTI_QUERY
    return resolved


# ---------------------------------------------------------------------------
# Multi-query fusion search
# ---------------------------------------------------------------------------


def search_with_query_rewrites(
    rag: RagService,
    original_query: str,
    rewritten_queries: list[str],
    top_k: int = 5,
    destination: str | None = None,
    section: str | None = None,
    travel_type: str | None = None,
    season: str | None = None,
) -> list[SearchResult]:
    """Run retrieval for each rewritten query and fuse results via RRF.

    Each query is searched independently; results are deduplicated by chunk_id
    and merged with reciprocal rank fusion.
    """
    from travel_agent.rag.models import SearchResult

    all_results: list[tuple[str, int, SearchResult]] = []  # (query, rank, result)

    for q in rewritten_queries:
        evidence = rag.retrieve_evidence(
            q,
            top_k=top_k * 2,  # over-fetch for fusion
            destination=destination,
            section=section,
            travel_type=travel_type,
            season=season,
            query_rewrite_mode=QueryRewriteMode.OFF,
        )
        for rank, r in enumerate(evidence.results, start=1):
            all_results.append((q, rank, r))

    if not all_results:
        return []

    # RRF fusion
    rrf_k = 60
    scores: dict[str, float] = {}
    results_by_key: dict[str, SearchResult] = {}
    order: dict[str, int] = {}

    for idx, (_, rank, r) in enumerate(all_results):
        key = _chunk_dedup_key(r)
        results_by_key[key] = r
        if key not in order:
            order[key] = idx
        scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)

    ranked = sorted(scores, key=lambda k: (scores[k], -order[k]), reverse=True)
    return [
        SearchResult(
            content=results_by_key[key].content,
            source=results_by_key[key].source,
            destination=results_by_key[key].destination,
            score=scores[key],
            metadata=results_by_key[key].metadata,
        )
        for key in ranked[:top_k]
    ]


def _chunk_dedup_key(result: SearchResult) -> str:
    chunk_id = result.metadata.get("chunk_id")
    if isinstance(chunk_id, str) and chunk_id:
        return chunk_id
    content_prefix = result.content[:80]
    return f"{result.source}:{content_prefix}"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_query_rewriter(
    llm_provider: str = "qwen",
    model_name: str = "qwen3-max",
) -> LLMQueryRewriter:
    """Build a query rewriter with lazy model instantiation.

    The underlying chat model is only created on the first ``rewrite()`` call,
    not at construction time.  When no API key is configured the model stays
    *None* and every ``rewrite()`` call becomes a transparent pass-through.
    """
    return LLMQueryRewriter(
        model=None,
        build_fn=lambda: _build_chat_model(llm_provider, model_name),
    )


def _build_chat_model(provider: str, model_name: str) -> Any | None:
    provider = provider.strip().lower()
    if provider in {"qwen", "dashscope"}:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            return None
        return _chat_openai(
            model=model_name,
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        return _chat_openai(model=model_name, api_key=api_key)
    return None


def _chat_openai(model: str, api_key: str, base_url: str | None = None) -> Any:
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, object] = {
        "model": model,
        "api_key": api_key,
        "temperature": 0,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)
