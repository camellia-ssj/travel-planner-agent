"""面向旅行领域 RAG 检索的 LLM 驱动查询改写模块。

启用后，此模块将自然语言的用户问题改写为一个或多个更利于检索的查询。
多查询模式通过倒数排名融合（RRF）将原始查询与改写后的查询结果进行合并。
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
# 提示词模板
# ---------------------------------------------------------------------------

_REWRITE_SYSTEM_PROMPT = (
    "你是一个旅行领域的查询改写器。你的任务是将用户的自然语言旅行问题转换为一个或多个"
    "简洁、关键词丰富的检索查询，以便在目的地知识库上获得良好的检索效果。\n\n"
    "规则:\n"
    "- 规范化目的地名称（例如「杭洲」「西湖那边」→「杭州西湖」）。\n"
    "- 提取关键约束条件：天数、预算、出行人群（家庭/老人/独自/情侣）、季节、偏好。\n"
    "- 多查询模式下生成 2-3 个多样化的查询，覆盖不同角度"
    "（例如行程、预算、人流风险、天气、备选方案）。\n"
    "- 绝不添加用户未提及的事实。\n"
    "- 输入为中文时用中文撰写查询；英文时用英文。\n"
    "- 输出必须是一个单独的 JSON 对象，不要用 markdown 代码块，不要添加任何评论。"
)

_REWRITE_USER_PROMPT = (
    "请将以下旅行问题改写为适合知识库检索的查询。\n\n"
    "原始问题: {question}\n\n"
    "改写模式: {mode}\n\n"
    '输出 JSON: {{"rewritten_query": "...", "search_queries": ["..."], "notes": ["..."]}}'
)

# ---------------------------------------------------------------------------
# 查询改写器
# ---------------------------------------------------------------------------


@dataclass
class LLMQueryRewriter:
    """封装 LangChain 聊天模型，用于在检索前改写旅行查询。

    参数
    ----------
    model:
        预构建的 LangChain 聊天模型。当为 *None* 时，改写器会尝试通过
        *build_fn* 延迟构建一个。
    build_fn:
        可选的无参数可调用对象，按需返回聊天模型。
        仅在 *model* 为 None 且首次调用 ``rewrite()`` 时被调用。
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
        """改写 *query* 并返回结构化结果。

        当改写器没有模型，或模型调用失败时，原样返回原始查询。
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
    """调用 LLM 并解析其 JSON 响应。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    response = model.invoke([
        SystemMessage(content=_REWRITE_SYSTEM_PROMPT),
        HumanMessage(content=_REWRITE_USER_PROMPT.format(question=question, mode=mode_str)),
    ])
    text = _extract_message_text(response)
    return _parse_rewrite_json(text, question)


def _extract_message_text(response: Any) -> str:
    """从各种 LangChain 响应格式中提取文本内容。"""
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
    """从 LLM 输出中稳健地提取 JSON。"""
    cleaned = text.strip()
    # 去除偶尔出现的 markdown 代码块标记
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
        # 尝试通过扫描花括号来定位 JSON 对象
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
    """根据模式组装最终的搜索查询列表。"""
    rewritten = result.get("rewritten_query", original_query) or original_query

    if mode is QueryRewriteMode.REWRITE_ONLY:
        return [rewritten]

    candidates = result.get("search_queries", [rewritten])
    if not isinstance(candidates, list):
        candidates = [rewritten]

    # 清理：丢弃空条目或仅含空白字符的条目
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

    # 确保 rewritten_query 排在首位
    if rewritten.strip() not in seen:
        cleaned.insert(0, rewritten.strip())

    # 始终包含原始查询以保证召回率
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
# 多查询融合检索
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
    """对每个改写后的查询执行检索，并通过 RRF 融合结果。

    每个查询独立检索；结果按 chunk_id 去重，并通过倒数排名融合（RRF）进行合并。
    """
    from travel_agent.rag.models import SearchResult

    all_results: list[tuple[str, int, SearchResult]] = []  # (查询, 排名, 结果)

    for q in rewritten_queries:
        evidence = rag.retrieve_evidence(
            q,
            top_k=top_k * 2,  # 超量获取以便融合时有更多候选
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

    # RRF 融合
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
# 工厂函数
# ---------------------------------------------------------------------------


def build_query_rewriter(
    llm_provider: str = "qwen",
    model_name: str = "qwen3-max",
) -> LLMQueryRewriter:
    """构建一个带有延迟模型实例化的查询改写器。

    底层的聊天模型仅在首次调用 ``rewrite()`` 时创建，而非在构造时。
    当未配置 API key 时，模型保持 *None*，每次 ``rewrite()`` 调用将
    透明地原样返回。
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
