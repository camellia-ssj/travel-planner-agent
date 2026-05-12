"""Typer + Rich CLI for the LangGraph travel agent MVP."""

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
    help="LangGraph travel planning agent with structured LLM planning and rule fallback.",
    no_args_is_help=True,
)
console = Console()
EMBEDDING_PROVIDER_HELP = "auto, qwen, dashscope, openai, sentence-transformers or local."
DEFAULT_CHECKPOINT_PATH = Path("data/agent_checkpoints.sqlite")
DEFAULT_MEMORY_PATH = Path("data/user_memory.sqlite")


def main() -> None:
    app()


@app.callback()
def _callback() -> None:
    """Run the travel agent command group."""


@app.command()
def plan(
    request: Annotated[str, typer.Argument(help="Natural-language travel planning request.")],
    destination: Annotated[
        str | None,
        typer.Option("--destination", "-d", help="Override parsed destination."),
    ] = None,
    days: Annotated[
        int | None,
        typer.Option("--days", help="Override parsed trip length in days.", min=1),
    ] = None,
    embedding_provider: Annotated[
        EmbeddingProviderName,
        typer.Option("--embedding-provider", help=EMBEDDING_PROVIDER_HELP),
    ] = EmbeddingProviderName.LOCAL,
    persist_dir: Annotated[
        Path,
        typer.Option("--persist-dir", help="Chroma persistence directory."),
    ] = Path("data/chroma"),
    collection: Annotated[
        str,
        typer.Option("--collection", help="Chroma collection name."),
    ] = "travel_destinations",
    thread_id: Annotated[
        str | None,
        typer.Option("--thread-id", help="Optional thread id for checkpoint recovery."),
    ] = None,
    checkpoint_path: Annotated[
        Path,
        typer.Option("--checkpoint-path", help="SQLite checkpoint database path."),
    ] = DEFAULT_CHECKPOINT_PATH,
    query_rewrite: Annotated[
        str,
        typer.Option("--query-rewrite", help="Query rewrite mode: off, rewrite_only, multi_query"),
    ] = "off",
    user_id: Annotated[
        str | None,
        typer.Option("--user-id", help="User id for long-term memory profile."),
    ] = None,
    memory_path: Annotated[
        Path,
        typer.Option("--memory-path", help="SQLite memory database path."),
    ] = DEFAULT_MEMORY_PATH,
    as_json: Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")] = False,
) -> None:
    """Generate a structured travel plan with RAG evidence."""

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
    thread_id: Annotated[str, typer.Argument(help="Thread id of a previous planning run.")],
    feedback: Annotated[str, typer.Argument(help="Follow-up modification request.")],
    embedding_provider: Annotated[
        EmbeddingProviderName,
        typer.Option("--embedding-provider", help=EMBEDDING_PROVIDER_HELP),
    ] = EmbeddingProviderName.LOCAL,
    persist_dir: Annotated[
        Path,
        typer.Option("--persist-dir", help="Chroma persistence directory."),
    ] = Path("data/chroma"),
    collection: Annotated[
        str,
        typer.Option("--collection", help="Chroma collection name."),
    ] = "travel_destinations",
    checkpoint_path: Annotated[
        Path,
        typer.Option("--checkpoint-path", help="SQLite checkpoint database path."),
    ] = DEFAULT_CHECKPOINT_PATH,
    query_rewrite: Annotated[
        str,
        typer.Option("--query-rewrite", help="Query rewrite mode: off, rewrite_only, multi_query"),
    ] = "off",
    user_id: Annotated[
        str | None,
        typer.Option("--user-id", help="User id for long-term memory profile."),
    ] = None,
    memory_path: Annotated[
        Path,
        typer.Option("--memory-path", help="SQLite memory database path."),
    ] = DEFAULT_MEMORY_PATH,
    as_json: Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")] = False,
) -> None:
    """Resume a checkpointed plan and apply follow-up user feedback.

    Feedback that changes destination / days / budget will trigger fresh
    RAG evidence retrieval so the regenerated plan uses up-to-date data.
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
def eval(
    cases_path: Annotated[
        Path,
        typer.Option(
            "--cases", "-c",
            help="Path to JSONL eval cases (default: tests/fixtures/agent_eval_cases.jsonl)",
        ),
    ] = Path("tests/fixtures/agent_eval_cases.jsonl"),
    as_json: Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show per-case failures.")
    ] = False,
) -> None:
    """Run offline agent evaluation with deterministic rule-based planner.

    Evaluates plan quality across all cases in the JSONL fixture without any LLM calls.
    """
    if not cases_path.exists():
        console.print(f"[red]Eval cases file not found: {cases_path}[/red]")
        raise typer.Exit(code=1)

    cases = load_agent_eval_cases(cases_path)
    if not cases:
        console.print("[red]No eval cases loaded.[/red]")
        raise typer.Exit(code=1)

    # Build a minimal evidence map from the fixture — every case that
    # declares expected_evidence_sources gets one synthetic result per source.
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
    """Run the agent graph and return a serializable payload."""

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

    # Record trace metrics
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
    """Resume a checkpointed agent thread and regenerate the plan.

    When *rag_service* is provided, destination / day / budget changes in
    *feedback* trigger fresh RAG evidence retrieval.  Without it the
    resume still works but re-uses the original evidence from the
    checkpoint.
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
            raise typer.BadParameter(f"No checkpoint found for thread_id={thread_id!r}.")
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
        raise typer.BadParameter("Agent graph did not return a TravelPlan.")

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
    """Yield a SqliteSaver with an explicit serializer to avoid default-config drift.

    The ``allowed_objects`` warning is suppressed during import because it
    originates from a module-level ``Reviver()`` call in the library — not
    from our configuration.  Explicitly passing ``JsonPlusSerializer``
    ensures our serialization contract stays pinned regardless of future
    langgraph default changes.
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
    request = payload["request"]
    plan_payload = payload["plan"]
    validation = payload["validation"]

    console.print(
        Panel.fit(plan_payload["summary"], title="Travel Agent Plan", border_style="cyan")
    )
    console.print(f"[cyan]thread_id:[/cyan] {payload['thread_id']}")
    if payload.get("user_profile"):
        _print_user_profile(payload["user_profile"])
    if payload.get("user_feedback"):
        _print_list("User Feedback", payload["user_feedback"])
    _print_request(request)
    _print_day_plans(plan_payload["day_plans"])
    _print_budget(plan_payload["budget_items"])
    _print_risks(plan_payload["risk_notices"])
    _print_list("Alternatives", plan_payload["alternatives"])
    _print_list("Evidence Sources", plan_payload["evidence_sources"])

    if not validation["is_valid"]:
        _print_list("Validation Errors", validation["errors"])

    # Reflection report
    reflection = payload.get("reflection")
    if reflection:
        _print_reflection(reflection)


def _print_request(request: dict[str, Any]) -> None:
    table = Table(title="Parsed Travel Request")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("destination", str(request["destination"]))
    table.add_row("days", str(request["days"]))
    table.add_row("audience", ", ".join(request["audience"]))
    table.add_row("budget_preference", str(request["budget_preference"]))
    console.print(table)


def _print_day_plans(day_plans: list[dict[str, Any]]) -> None:
    table = Table(title="Daily Itinerary", show_lines=True)
    table.add_column("Day", justify="right", style="cyan", width=5)
    table.add_column("Title", style="green")
    table.add_column("Activities", overflow="fold")
    for day_plan in day_plans:
        table.add_row(
            str(day_plan["day"]),
            str(day_plan["title"]),
            "\n".join(f"- {activity}" for activity in day_plan["activities"]),
        )
    console.print(table)


def _print_budget(budget_items: list[dict[str, Any]]) -> None:
    table = Table(title="Budget Estimate")
    table.add_column("Category", style="cyan")
    table.add_column("Preference", style="green")
    table.add_column("Note", overflow="fold")
    for item in budget_items:
        table.add_row(str(item["category"]), str(item["preference"]), str(item["note"]))
    console.print(table)


def _print_risks(risk_notices: list[dict[str, Any]]) -> None:
    table = Table(title="Risk Notices")
    table.add_column("Type", style="cyan")
    table.add_column("Severity", style="yellow")
    table.add_column("Message", overflow="fold")
    for notice in risk_notices:
        table.add_row(
            str(notice["risk_type"]),
            str(notice["severity"]),
            str(notice["message"]),
        )
    console.print(table)


def _print_list(title: str, values: list[str]) -> None:
    table = Table(title=title)
    table.add_column("#", justify="right", style="cyan", width=4)
    table.add_column("Value", overflow="fold")
    for index, value in enumerate(values, start=1):
        table.add_row(str(index), value)
    console.print(table)


def _print_user_profile(profile: dict[str, Any]) -> None:
    table = Table(title="User Profile (Memory)")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("user_id", str(profile["user_id"]))
    table.add_row("total_trips", str(profile["total_trips"]))
    if profile.get("preferred_destinations"):
        table.add_row("preferred_destinations", ", ".join(profile["preferred_destinations"]))
    if profile.get("audience_types"):
        table.add_row("audience_types", ", ".join(profile["audience_types"]))
    table.add_row("budget_preference", str(profile["budget_preference"]))
    if profile.get("trip_length_avg"):
        table.add_row("avg_trip_length", f"{profile['trip_length_avg']:.1f} days")
    if profile.get("preferences_summary"):
        table.add_row("summary", profile["preferences_summary"])
    console.print(table)


def _print_reflection(reflection: dict[str, Any]) -> None:
    """Print the reflection (factuality review) report."""
    passed = reflection.get("passed", False)
    status_color = "green" if passed else "yellow"
    status_text = "PASSED" if passed else "FLAGGED"

    title = f"Reflection Report — {status_text}"
    console.print(Panel.fit(title, border_style=status_color))

    coverage = reflection.get("evidence_coverage", 0.0)
    confidence = reflection.get("confidence_score", 0.0)
    checked = reflection.get("checked_claims", 0)
    grounded = reflection.get("grounded_claims", 0)

    summary_text = (
        f"Evidence coverage: {coverage:.0%}  |  "
        f"Confidence: {confidence:.0%}  |  "
        f"Claims grounded: {grounded}/{checked}"
    )
    console.print(f"[dim]{summary_text}[/dim]")

    flags = reflection.get("hallucination_flags", [])
    if flags:
        flag_table = Table(title="Hallucination Flags")
        flag_table.add_column("Location", style="cyan")
        flag_table.add_column("Severity", style="red")
        flag_table.add_column("Claim", overflow="fold", max_width=60)
        flag_table.add_column("Issue", overflow="fold", max_width=40)
        for flag in flags:
            severity_style = "red" if flag.get("severity") == "high" else "yellow"
            flag_table.add_row(
                str(flag.get("location", "")),
                f"[{severity_style}]{flag.get('severity', '')}[/{severity_style}]",
                str(flag.get("claim", ""))[:200],
                str(flag.get("issue", "")),
            )
        console.print(flag_table)

    issues = reflection.get("issues", [])
    if issues:
        console.print("[bold yellow]Issues:[/bold yellow]")
        for issue in issues:
            console.print(f"  [yellow]- {issue}[/yellow]")

    suggestions = reflection.get("suggestions", [])
    if suggestions:
        console.print("[bold cyan]Suggestions:[/bold cyan]")
        for suggestion in suggestions:
            console.print(f"  [dim]- {suggestion}[/dim]")


def _print_eval_report(
    report: dict[str, Any],
    metrics: AgentEvalMetrics,
    verbose: bool = False,
) -> None:
    """Pretty-print agent eval results."""
    from rich.table import Table as RichTable

    m = report["metrics"]
    r = report["run"]

    console.print(Panel.fit("Agent Eval Results", border_style="cyan"))
    console.print(
        f"[dim]planner:[/dim] {r['planner']}  "
        f"[dim]cases:[/dim] {r['total_cases']}  "
        f"[dim]mode:[/dim] {r['mode']}"
    )

    table = RichTable(title="Quality Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Status", style="yellow")

    def _status(rate: float, threshold: float = 0.8) -> str:
        if rate >= threshold:
            return "[green]PASS[/green]"
        return "[red]CHECK[/red]"

    table.add_row("Days Match Rate", f"{m['days_match_rate']:.2%}", _status(m["days_match_rate"]))
    table.add_row(
        "Budget Present Rate",
        f"{m['budget_present_rate']:.2%}",
        _status(m["budget_present_rate"]),
    )
    table.add_row(
        "Risk Notices Rate",
        f"{m['risk_notices_rate']:.2%}",
        _status(m["risk_notices_rate"]),
    )
    table.add_row(
        "Evidence Source Coverage",
        f"{m['evidence_source_coverage']:.2%}",
        _status(m["evidence_source_coverage"]),
    )
    table.add_row(
        "Low Confidence Handling",
        f"{m['low_confidence_handling_rate']:.2%}",
        _status(m["low_confidence_handling_rate"]),
    )
    table.add_row(
        "Empty Result Handling",
        f"{m['empty_result_handling_rate']:.2%}",
        _status(m["empty_result_handling_rate"]),
    )
    table.add_row(
        "Validation Pass Rate",
        f"{m['validation_pass_rate']:.2%}",
        _status(m["validation_pass_rate"]),
    )
    table.add_row("Avg Latency (ms)", f"{m['avg_latency_ms']:.2f}", "")
    console.print(table)

    if verbose and m.get("failures"):
        console.print("\n[bold]Failures:[/bold]")
        for failure in m["failures"]:
            console.print(f"  [red]- {failure}[/red]")

    if m.get("failures"):
        total_failures = len(m["failures"])
        console.print(
            f"\n[yellow]{total_failures} failure(s) total. Use --verbose for details.[/yellow]"
        )
        raise typer.Exit(code=1 if total_failures > 0 else 0)


if __name__ == "__main__":
    main()
