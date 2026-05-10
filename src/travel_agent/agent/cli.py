"""Typer + Rich CLI for the LangGraph travel agent MVP."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

import typer
from langgraph.checkpoint.sqlite import SqliteSaver
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from travel_agent.agent.graph import build_travel_agent_graph, build_travel_agent_resume_graph
from travel_agent.agent.nodes import EvidenceService
from travel_agent.agent.planner import TravelPlanner, build_default_planner
from travel_agent.agent.schemas import TravelPlan
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
        typer.Option("--days", help="Override parsed trip length in days."),
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
    as_json: Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")] = False,
) -> None:
    """Generate a structured travel plan with RAG evidence."""

    rag_service = _build_rag_service(
        persist_dir=persist_dir,
        collection_name=collection,
        embedding_provider=embedding_provider,
    )
    result = run_plan(
        request,
        rag_service,
        planner=build_default_planner(),
        destination=destination,
        days=days,
        thread_id=thread_id,
        checkpoint_path=checkpoint_path,
    )
    if as_json:
        console.print_json(json.dumps(result, ensure_ascii=False))
        return

    _print_plan(result)


@app.command()
def resume(
    thread_id: Annotated[str, typer.Argument(help="Thread id of a previous planning run.")],
    feedback: Annotated[str, typer.Argument(help="Follow-up modification request.")],
    checkpoint_path: Annotated[
        Path,
        typer.Option("--checkpoint-path", help="SQLite checkpoint database path."),
    ] = DEFAULT_CHECKPOINT_PATH,
    as_json: Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")] = False,
) -> None:
    """Resume a checkpointed plan and apply follow-up user feedback."""

    result = resume_plan(
        thread_id=thread_id,
        feedback=feedback,
        planner=build_default_planner(),
        checkpoint_path=checkpoint_path,
    )
    if as_json:
        console.print_json(json.dumps(result, ensure_ascii=False))
        return

    _print_plan(result)


def run_plan(
    request: str,
    rag_service: EvidenceService,
    planner: TravelPlanner | None = None,
    destination: str | None = None,
    days: int | None = None,
    thread_id: str | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Run the agent graph and return a serializable payload."""

    active_thread_id = thread_id or uuid.uuid4().hex
    initial_state: dict[str, object] = _plan_initial_state(
        request,
        destination=destination,
        days=days,
    )
    if checkpoint_path is None:
        graph = build_travel_agent_graph(rag_service, planner=planner)
        final_state = graph.invoke(initial_state)
    else:
        with _sqlite_checkpointer(checkpoint_path) as checkpointer:
            graph = build_travel_agent_graph(
                rag_service,
                planner=planner,
                checkpointer=checkpointer,
            )
            final_state = graph.invoke(
                initial_state,
                config=_thread_config(active_thread_id),
            )
    return _state_payload(final_state, active_thread_id)


def resume_plan(
    thread_id: str,
    feedback: str,
    planner: TravelPlanner | None = None,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
) -> dict[str, Any]:
    """Resume a checkpointed agent thread and regenerate the plan."""

    with _sqlite_checkpointer(checkpoint_path) as checkpointer:
        graph = build_travel_agent_resume_graph(planner=planner, checkpointer=checkpointer)
        config = _thread_config(thread_id)
        snapshot = graph.get_state(config)
        if not snapshot.values:
            raise typer.BadParameter(f"No checkpoint found for thread_id={thread_id!r}.")
        final_state = graph.invoke(
            {"latest_user_feedback": feedback},
            config=config,
        )
    return _state_payload(final_state, thread_id)


def _plan_initial_state(
    request: str,
    destination: str | None = None,
    days: int | None = None,
) -> dict[str, object]:
    initial_state: dict[str, object] = {"question": request}
    if destination:
        initial_state["destination_override"] = destination
    if days is not None:
        initial_state["days_override"] = days
    return initial_state


def _state_payload(final_state: dict[str, Any], thread_id: str) -> dict[str, Any]:
    plan = final_state["plan"]
    if not isinstance(plan, TravelPlan):
        raise typer.BadParameter("Agent graph did not return a TravelPlan.")
    return {
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
    }


@contextmanager
def _sqlite_checkpointer(path: Path) -> Iterator[SqliteSaver]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as checkpointer:
        checkpointer.setup()
        yield checkpointer


def _thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def _build_rag_service(
    persist_dir: Path,
    collection_name: str,
    embedding_provider: EmbeddingProviderName,
) -> EvidenceService:
    return create_rag_service(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
    )


def _print_plan(payload: dict[str, Any]) -> None:
    request = payload["request"]
    plan_payload = payload["plan"]
    validation = payload["validation"]

    console.print(
        Panel.fit(plan_payload["summary"], title="Travel Agent Plan", border_style="cyan")
    )
    console.print(f"[cyan]thread_id:[/cyan] {payload['thread_id']}")
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


if __name__ == "__main__":
    main()
