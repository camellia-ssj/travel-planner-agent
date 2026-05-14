"""基于 Typer + Rich 的 LangGraph 旅行规划 Agent CLI。"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from travel_agent.agent.display import display_plan_payload
from travel_agent.agent.evaluation import (
    AgentEvalMetrics,
    build_eval_report,
    evaluate_agent_plans,
    load_agent_eval_cases,
)
from travel_agent.agent.graph import build_travel_agent_graph, build_travel_agent_resume_graph
from travel_agent.agent.nodes import EvidenceService
from travel_agent.agent.planner import TravelPlanner, build_default_planner
from travel_agent.agent.reflection import ReflectionService, build_reflection_service
from travel_agent.agent.schemas import TravelPlan
from travel_agent.memory.models import UserProfile
from travel_agent.memory.store import MemoryStore
from travel_agent.observability.tracer import AgentTracer, get_tracer
from travel_agent.rag.api import create_rag_service
from travel_agent.rag.config import EmbeddingProviderName

app = typer.Typer(
    name="travel-agent",
    help="基于 LangGraph 的旅行规划 Agent，支持结构化 LLM 规划与规则兜底。",
    no_args_is_help=True,
)
console = Console()
EMBEDDING_PROVIDER_HELP = "auto, qwen, dashscope, openai, sentence-transformers 或 local。"
DEFAULT_CHECKPOINT_PATH = Path("data/agent_checkpoints.sqlite")
DEFAULT_MEMORY_PATH = Path("data/user_memory.sqlite")


def main() -> None:
    app()


@app.callback()
def _callback() -> None:
    """运行旅行规划 Agent 命令组。"""


@app.command()
def plan(
    request: Annotated[str, typer.Argument(help="自然语言旅行规划需求。")],
    destination: Annotated[
        str | None,
        typer.Option("--destination", "-d", help="手动指定目的地，覆盖解析结果。"),
    ] = None,
    days: Annotated[
        int | None,
        typer.Option("--days", help="手动指定游玩天数，覆盖解析结果。", min=1),
    ] = None,
    embedding_provider: Annotated[
        EmbeddingProviderName,
        typer.Option("--embedding-provider", help=EMBEDDING_PROVIDER_HELP),
    ] = EmbeddingProviderName.LOCAL,
    persist_dir: Annotated[
        Path,
        typer.Option("--persist-dir", help="Chroma 持久化目录。"),
    ] = Path("data/chroma"),
    collection: Annotated[
        str,
        typer.Option("--collection", help="Chroma 集合名称。"),
    ] = "travel_destinations",
    thread_id: Annotated[
        str | None,
        typer.Option("--thread-id", help="可选线程 ID，用于检查点恢复。"),
    ] = None,
    checkpoint_path: Annotated[
        Path,
        typer.Option("--checkpoint-path", help="SQLite 检查点数据库路径。"),
    ] = DEFAULT_CHECKPOINT_PATH,
    query_rewrite: Annotated[
        str,
        typer.Option("--query-rewrite", help="查询改写模式: off, rewrite_only, multi_query"),
    ] = "off",
    user_id: Annotated[
        str | None,
        typer.Option("--user-id", help="用户 ID，用于长期记忆画像。"),
    ] = None,
    memory_path: Annotated[
        Path,
        typer.Option("--memory-path", help="SQLite 记忆数据库路径。"),
    ] = DEFAULT_MEMORY_PATH,
    as_json: Annotated[bool, typer.Option("--json", help="输出机器可读的 JSON 格式。")] = False,
) -> None:
    """生成结构化旅行计划，基于 RAG 证据检索。"""

    rag_service = _build_rag_service(
        persist_dir=persist_dir,
        collection_name=collection,
        embedding_provider=embedding_provider,
        query_rewrite=query_rewrite,
    )
    memory_store = _build_memory_store(memory_path) if user_id else None
    result = run_plan(
        request,
        rag_service,
        planner=build_default_planner(),
        destination=destination,
        days=days,
        thread_id=thread_id,
        checkpoint_path=checkpoint_path,
        user_id=user_id,
        memory_store=memory_store,
    )
    if as_json:
        console.print_json(json.dumps(result, ensure_ascii=False))
        return

    _print_plan(result)


@app.command()
def resume(
    thread_id: Annotated[str, typer.Argument(help="之前规划运行的线程 ID。")],
    feedback: Annotated[str, typer.Argument(help="后续修改需求。")],
    embedding_provider: Annotated[
        EmbeddingProviderName,
        typer.Option("--embedding-provider", help=EMBEDDING_PROVIDER_HELP),
    ] = EmbeddingProviderName.LOCAL,
    persist_dir: Annotated[
        Path,
        typer.Option("--persist-dir", help="Chroma 持久化目录。"),
    ] = Path("data/chroma"),
    collection: Annotated[
        str,
        typer.Option("--collection", help="Chroma 集合名称。"),
    ] = "travel_destinations",
    checkpoint_path: Annotated[
        Path,
        typer.Option("--checkpoint-path", help="SQLite 检查点数据库路径。"),
    ] = DEFAULT_CHECKPOINT_PATH,
    query_rewrite: Annotated[
        str,
        typer.Option("--query-rewrite", help="查询改写模式: off, rewrite_only, multi_query"),
    ] = "off",
    user_id: Annotated[
        str | None,
        typer.Option("--user-id", help="用户 ID，用于长期记忆画像。"),
    ] = None,
    memory_path: Annotated[
        Path,
        typer.Option("--memory-path", help="SQLite 记忆数据库路径。"),
    ] = DEFAULT_MEMORY_PATH,
    as_json: Annotated[bool, typer.Option("--json", help="输出机器可读的 JSON 格式。")] = False,
) -> None:
    """恢复检查点中的计划并应用用户反馈。

    修改目的地/天数/预算的反馈将触发新的 RAG 证据检索，
    确保重新生成的计划使用最新数据。
    """

    rag_service = _build_rag_service(
        persist_dir=persist_dir,
        collection_name=collection,
        embedding_provider=embedding_provider,
        query_rewrite=query_rewrite,
    )
    memory_store = _build_memory_store(memory_path) if user_id else None
    result = resume_plan(
        thread_id=thread_id,
        feedback=feedback,
        rag_service=rag_service,
        planner=build_default_planner(),
        checkpoint_path=checkpoint_path,
        user_id=user_id,
        memory_store=memory_store,
    )
    if as_json:
        console.print_json(json.dumps(result, ensure_ascii=False))
        return

    _print_plan(result)


@app.command()
def chat(
    user_id: Annotated[
        str | None,
        typer.Option("--user-id", help="用户 ID，用于长期记忆画像。"),
    ] = None,
    thread_id: Annotated[
        str | None,
        typer.Option("--thread-id", help="通过线程 ID 恢复之前的对话。"),
    ] = None,
    embedding_provider: Annotated[
        EmbeddingProviderName,
        typer.Option("--embedding-provider", help=EMBEDDING_PROVIDER_HELP),
    ] = EmbeddingProviderName.LOCAL,
    persist_dir: Annotated[
        Path,
        typer.Option("--persist-dir", help="Chroma 持久化目录。"),
    ] = Path("data/chroma"),
    collection: Annotated[
        str,
        typer.Option("--collection", help="Chroma 集合名称。"),
    ] = "travel_destinations",
    checkpoint_path: Annotated[
        Path,
        typer.Option("--checkpoint-path", help="SQLite 检查点数据库路径。"),
    ] = DEFAULT_CHECKPOINT_PATH,
    query_rewrite: Annotated[
        str,
        typer.Option("--query-rewrite", help="查询改写模式: off, rewrite_only, multi_query"),
    ] = "off",
    memory_path: Annotated[
        Path,
        typer.Option("--memory-path", help="SQLite 记忆数据库路径。"),
    ] = DEFAULT_MEMORY_PATH,
    streaming: Annotated[
        bool,
        typer.Option("--streaming/--no-streaming", help="启用流式输出。"),
    ] = True,
) -> None:
    """启动交互式对话旅行规划会话。

    与 AI 旅行顾问自然对话。Agent 会先询问澄清问题，
    然后生成包含预算、风险提示和备选方案的完整旅行计划。
    你可以在同一会话中提出反馈并迭代优化计划。

    支持斜杠命令: /plan, /feedback, /profile, /history,
    /reset, /export, /help, /quit。
    """
    from travel_agent.agent.planner import _build_chat_model, AgentPlannerSettings
    from travel_agent.conversation.cli_repl import ConversationREPL
    from travel_agent.conversation.graph import build_conversation_graph

    import logging
    logging.root.handlers = [logging.NullHandler()]

    rag_service = _build_rag_service(
        persist_dir=persist_dir,
        collection_name=collection,
        embedding_provider=embedding_provider,
        query_rewrite=query_rewrite,
    )
    memory_store = _build_memory_store(memory_path) if user_id else None
    reflection_service = build_reflection_service()

    # 构建对话用的 LLM 聊天模型
    settings = AgentPlannerSettings.from_env()
    chat_model = _build_chat_model(settings)
    if chat_model is None:
        console.print(
            "[yellow]未配置 LLM API Key，将使用规则模式。"
            "对话体验会受限。[/yellow]\n"
            "[dim]设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY 以启用完整对话功能。[/dim]"
        )
        # 兜底：尝试构建一个简单的聊天模型
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
            base_url = None
            if os.getenv("DASHSCOPE_API_KEY"):
                base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            chat_model = ChatOpenAI(
                model=settings.model,
                api_key=api_key,
                base_url=base_url,
                temperature=0.7,
            )
        else:
            console.print(
                "[red]需要配置 API Key 才能使用对话模式。[/red]\n"
                "[dim]请设置 DASHSCOPE_API_KEY 或 OPENAI_API_KEY 环境变量。[/dim]"
            )
            raise typer.Exit(code=1)

    planner = build_default_planner(settings)

    with _sqlite_checkpointer(checkpoint_path) as checkpointer:
        conv_graph = build_conversation_graph(
            chat_model=chat_model,
            rag_service=rag_service,
            planner=planner,
            checkpointer=checkpointer,
            memory_service=memory_store,
            reflection_service=reflection_service,
        )
        repl = ConversationREPL(
            graph=conv_graph,
            console=console,
            user_id=user_id,
            thread_id=thread_id,
            streaming=streaming,
        )
        repl.run()


@app.command()
def eval(
    cases_path: Annotated[
        Path,
        typer.Option(
            "--cases", "-c",
            help="JSONL 评估用例文件路径（默认: tests/fixtures/agent_eval_cases.jsonl）",
        ),
    ] = Path("tests/fixtures/agent_eval_cases.jsonl"),
    as_json: Annotated[bool, typer.Option("--json", help="输出机器可读的 JSON 格式。")] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="显示每个用例的失败详情。")
    ] = False,
) -> None:
    """使用确定性规则规划器运行离线 Agent 评估。

    在 JSONL 测试集中的所有用例上评估计划质量，无需 LLM 调用。
    """
    if not cases_path.exists():
        console.print(f"[red]评估用例文件未找到: {cases_path}[/red]")
        raise typer.Exit(code=1)

    cases = load_agent_eval_cases(cases_path)
    if not cases:
        console.print("[red]未加载到评估用例。[/red]")
        raise typer.Exit(code=1)

    # 从测试集构建最小证据映射——每个声明了 expected_evidence_sources
    # 的用例都会获得每个来源的一条合成结果
    evidence_map: dict[str, list] = {}
    for case in cases:
        if case.expected_evidence_sources and not case.expected_empty:
            from travel_agent.rag.models import SearchResult

            evidence_map[case.query] = [
                SearchResult(
                    content=f"{case.destination or 'unknown'} itinerary content for day planning.",
                    source=src,
                    destination=case.destination or "",
                    score=0.9,
                    metadata={"section": "itinerary"},
                )
                for src in case.expected_evidence_sources
            ]

    metrics = evaluate_agent_plans(cases, evidence_map)
    report = build_eval_report(metrics, len(cases))

    if as_json:
        console.print_json(json.dumps(report, ensure_ascii=False))
        return

    _print_eval_report(report, metrics, verbose=verbose)


def run_plan(
    request: str,
    rag_service: EvidenceService,
    planner: TravelPlanner | None = None,
    reflection_service: ReflectionService | None = None,
    destination: str | None = None,
    days: int | None = None,
    thread_id: str | None = None,
    checkpoint_path: Path | None = None,
    tracer: AgentTracer | None = None,
    user_id: str | None = None,
    memory_store: MemoryStore | None = None,
) -> dict[str, Any]:
    """运行 Agent 图并返回可序列化的结果。"""

    if reflection_service is None:
        reflection_service = build_reflection_service()

    active_tracer = tracer or get_tracer()
    active_thread_id = thread_id or uuid.uuid4().hex
    trace_ctx = active_tracer.start_run(
        run_name="travel-agent-plan",
        user_request=request,
    )

    initial_state: dict[str, object] = _plan_initial_state(
        request,
        destination=destination,
        days=days,
        user_id=user_id,
        thread_id=active_thread_id,
    )
    if checkpoint_path is None:
        graph = build_travel_agent_graph(
            rag_service,
            planner=planner,
            memory_service=memory_store,
            reflection_service=reflection_service,
        )
        final_state = graph.invoke(initial_state)
    else:
        with _sqlite_checkpointer(checkpoint_path) as checkpointer:
            graph = build_travel_agent_graph(
                rag_service,
                planner=planner,
                checkpointer=checkpointer,
                memory_service=memory_store,
                reflection_service=reflection_service,
            )
            final_state = graph.invoke(
                initial_state,
                config=_thread_config(active_thread_id),
            )

    # 记录追踪指标
    request_obj = final_state.get("request")
    evidence = final_state.get("evidence")
    plan = final_state["plan"]

    if request_obj is not None:
        active_tracer.record_parse(trace_ctx, request_obj)
    if evidence is not None:
        active_tracer.record_retrieval(trace_ctx, evidence)
    active_tracer.record_planner(
        trace_ctx,
        model=os.getenv("TRAVEL_AGENT_MODEL", "qwen3-max"),
        fallback=getattr(plan, "fallback_used", False),
    )
    active_tracer.record_validation(
        trace_ctx,
        passed=final_state.get("is_valid", False),
        errors=final_state.get("validation_errors", []),
    )
    reflection_report = final_state.get("reflection_report")
    if reflection_report is not None:
        active_tracer.record_reflection(
            trace_ctx,
            report=reflection_report,
        )
    active_tracer.finish_run(trace_ctx, plan, thread_id=active_thread_id)

    return _state_payload(final_state, active_thread_id)


def resume_plan(
    thread_id: str,
    feedback: str,
    rag_service: EvidenceService | None = None,
    planner: TravelPlanner | None = None,
    reflection_service: ReflectionService | None = None,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    user_id: str | None = None,
    memory_store: MemoryStore | None = None,
) -> dict[str, Any]:
    """恢复检查点中的 Agent 线程并重新生成计划。

    提供 *rag_service* 时，反馈中的目的地/天数/预算变更
    会触发新的 RAG 证据检索。不提供时仍可恢复，
    但会复用检查点中原始的证据数据。
    """

    if reflection_service is None:
        reflection_service = build_reflection_service()

    with _sqlite_checkpointer(checkpoint_path) as checkpointer:
        graph = build_travel_agent_resume_graph(
            rag_service=rag_service,
            planner=planner,
            checkpointer=checkpointer,
            memory_service=memory_store,
            reflection_service=reflection_service,
        )
        config = _thread_config(thread_id)
        snapshot = graph.get_state(config)
        if not snapshot.values:
            raise typer.BadParameter(f"未找到 thread_id={thread_id!r} 的检查点。")
        invoke_state: dict[str, object] = {"latest_user_feedback": feedback}
        if user_id:
            invoke_state["user_id"] = user_id
        final_state = graph.invoke(invoke_state, config=config)
    return _state_payload(final_state, thread_id)


def _plan_initial_state(
    request: str,
    destination: str | None = None,
    days: int | None = None,
    user_id: str | None = None,
    thread_id: str | None = None,
) -> dict[str, object]:
    initial_state: dict[str, object] = {"question": request}
    if destination:
        initial_state["destination_override"] = destination
    if days is not None:
        if days < 1:
            raise ValueError(f"days must be >= 1, got {days}")
        initial_state["days_override"] = days
    if user_id:
        initial_state["user_id"] = user_id
    if thread_id:
        initial_state["thread_id"] = thread_id
    return initial_state


def _state_payload(final_state: dict[str, Any], thread_id: str) -> dict[str, Any]:
    plan = final_state["plan"]
    if not isinstance(plan, TravelPlan):
        raise typer.BadParameter("Agent 图未返回 TravelPlan。")

    tool_budget = final_state.get("tool_budget")
    tool_crowd = final_state.get("tool_crowd_risk")
    tool_alt = final_state.get("tool_alternatives")
    user_profile = final_state.get("user_profile")
    reflection_report = final_state.get("reflection_report")

    payload: dict[str, Any] = {
        "thread_id": thread_id,
        "original_user_request": final_state.get("original_user_request", ""),
        "user_feedback": final_state.get("user_feedback", []),
        "request": final_state["request"].model_dump(),
        "plan": plan.model_dump(),
        "evidence": final_state["evidence"].as_dict(),
        "validation": {
            "is_valid": final_state.get("is_valid", False),
            "errors": final_state.get("validation_errors", []),
        },
        "tool_budget": tool_budget.model_dump() if tool_budget is not None else None,
        "tool_crowd_risk": tool_crowd.model_dump() if tool_crowd is not None else None,
        "tool_alternatives": tool_alt.model_dump() if tool_alt is not None else None,
        "reflection": reflection_report.model_dump() if reflection_report is not None else None,
    }
    if user_profile is not None and isinstance(user_profile, UserProfile):
        payload["user_profile"] = user_profile.model_dump()
    return payload


@contextmanager
def _sqlite_checkpointer(path: Path) -> Iterator[Any]:
    """创建 SqliteSaver，显式指定序列化器以避免默认配置漂移。

    ``allowed_objects`` 警告在导入期间被抑制，因为它来自库内部
    的模块级 ``Reviver()`` 调用——与我们的配置无关。
    显式传入 ``JsonPlusSerializer`` 确保序列化约定不受
    未来 langgraph 默认值变更的影响。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*allowed_objects.*")
            from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
            from langgraph.checkpoint.sqlite import SqliteSaver

        checkpointer = SqliteSaver(conn, serde=JsonPlusSerializer())
        checkpointer.setup()
        yield checkpointer
    finally:
        conn.close()


def _thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def _build_rag_service(
    persist_dir: Path,
    collection_name: str,
    embedding_provider: EmbeddingProviderName,
    query_rewrite: str = "off",
) -> EvidenceService:
    return create_rag_service(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        query_rewrite=query_rewrite,
    )


def _build_memory_store(path: Path) -> MemoryStore:
    return MemoryStore(path)


def _print_plan(payload: dict[str, Any]) -> None:
    """美化打印旅行规划结果（委托给 display 模块）。"""
    display_plan_payload(payload)


def _print_eval_report(
    report: dict[str, Any],
    metrics: AgentEvalMetrics,
    verbose: bool = False,
) -> None:
    """美化打印 Agent 评估结果。"""
    from rich.table import Table as RichTable

    m = report["metrics"]
    r = report["run"]

    console.print(Panel.fit("Agent 评估结果", border_style="cyan"))
    console.print(
        f"[dim]规划器:[/dim] {r['planner']}  "
        f"[dim]用例数:[/dim] {r['total_cases']}  "
        f"[dim]模式:[/dim] {r['mode']}"
    )

    table = RichTable(title="质量指标")
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")
    table.add_column("状态", style="yellow")

    def _status(rate: float, threshold: float = 0.8) -> str:
        if rate >= threshold:
            return "[green]通过[/green]"
        return "[red]待检查[/red]"

    table.add_row("天数匹配率", f"{m['days_match_rate']:.2%}", _status(m["days_match_rate"]))
    table.add_row("预算覆盖率", f"{m['budget_present_rate']:.2%}", _status(m["budget_present_rate"]))
    table.add_row("风险提示率", f"{m['risk_notices_rate']:.2%}", _status(m["risk_notices_rate"]))
    table.add_row(
        "证据来源覆盖率",
        f"{m['evidence_source_coverage']:.2%}",
        _status(m["evidence_source_coverage"]),
    )
    table.add_row(
        "低置信度处理率",
        f"{m['low_confidence_handling_rate']:.2%}",
        _status(m["low_confidence_handling_rate"]),
    )
    table.add_row(
        "空结果处理率",
        f"{m['empty_result_handling_rate']:.2%}",
        _status(m["empty_result_handling_rate"]),
    )
    table.add_row(
        "校验通过率",
        f"{m['validation_pass_rate']:.2%}",
        _status(m["validation_pass_rate"]),
    )
    table.add_row("平均延迟 (ms)", f"{m['avg_latency_ms']:.2f}", "")
    console.print(table)

    if verbose and m.get("failures"):
        console.print("\n[bold]失败项:[/bold]")
        for failure in m["failures"]:
            console.print(f"  [red]- {failure}[/red]")

    if m.get("failures"):
        total_failures = len(m["failures"])
        console.print(
            f"\n[yellow]共 {total_failures} 项失败。使用 --verbose 查看详情。[/yellow]"
        )
        raise typer.Exit(code=1 if total_failures > 0 else 0)


if __name__ == "__main__":
    main()
