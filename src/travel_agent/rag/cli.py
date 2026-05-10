"""Typer + Rich CLI for the local travel RAG knowledge base."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from travel_agent.rag.config import EmbeddingProviderName, RagSettings, RetrievalMode
from travel_agent.rag.embeddings import build_embeddings
from travel_agent.rag.evaluation import DEFAULT_QUALITY_THRESHOLDS, evaluate_rag
from travel_agent.rag.service import RagService
from travel_agent.rag.vector_store import reset_chroma

app = typer.Typer(
    name="travel-rag",
    help="Local LangChain + Chroma destination knowledge base.",
    no_args_is_help=True,
)
console = Console()
EMBEDDING_PROVIDER_HELP = "auto, qwen, dashscope, openai, sentence-transformers or local."
EMBEDDING_PROBE_TEXTS = [
    "杭州灵隐寺周末交通拥挤，建议使用地铁和公交。",
    "Tokyo family trip with rail transit and rainy-day alternatives.",
]


def main() -> None:
    app()


@app.command()
def ingest(
    path: Annotated[Path, typer.Argument(help="Knowledge file or directory to ingest.")],
    destination: Annotated[
        str | None,
        typer.Option("--destination", "-d", help="Destination metadata for all imported chunks."),
    ] = None,
    persist_dir: Annotated[
        Path,
        typer.Option("--persist-dir", help="Chroma persistence directory."),
    ] = Path("data/chroma"),
    collection: Annotated[
        str,
        typer.Option("--collection", help="Chroma collection name."),
    ] = "travel_destinations",
    embedding_provider: Annotated[
        EmbeddingProviderName,
        typer.Option("--embedding-provider", help=EMBEDDING_PROVIDER_HELP),
    ] = EmbeddingProviderName.AUTO,
    incremental: Annotated[
        bool,
        typer.Option("--incremental", help="Skip unchanged documents based on manifest hashes."),
    ] = False,
) -> None:
    """Import destination knowledge documents into Chroma."""

    service = _service(
        persist_dir=persist_dir,
        collection=collection,
        embedding_provider=embedding_provider,
    )
    report = service.ingest_documents(path, destination=destination, incremental=incremental)

    table = Table(title="Ingest Report")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for key, value in report.__dict__.items():
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Natural-language travel question.")],
    destination: Annotated[
        str | None,
        typer.Option("--destination", "-d", help="Filter by destination metadata."),
    ] = None,
    section: Annotated[
        str | None,
        typer.Option("--section", help="Filter by section metadata."),
    ] = None,
    travel_type: Annotated[
        str | None,
        typer.Option("--travel-type", help="Filter by travel_type metadata."),
    ] = None,
    season: Annotated[
        str | None,
        typer.Option("--season", help="Filter by season metadata."),
    ] = None,
    retrieval_mode: Annotated[
        RetrievalMode | None,
        typer.Option("--retrieval-mode", help="vector, keyword or hybrid."),
    ] = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Number of chunks to retrieve.")] = 5,
    persist_dir: Annotated[
        Path,
        typer.Option("--persist-dir", help="Chroma persistence directory."),
    ] = Path("data/chroma"),
    collection: Annotated[
        str,
        typer.Option("--collection", help="Chroma collection name."),
    ] = "travel_destinations",
    embedding_provider: Annotated[
        EmbeddingProviderName,
        typer.Option("--embedding-provider", help=EMBEDDING_PROVIDER_HELP),
    ] = EmbeddingProviderName.AUTO,
    as_json: Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")] = False,
) -> None:
    """Retrieve relevant chunks with source, destination and score."""

    service = _service(
        persist_dir=persist_dir,
        collection=collection,
        embedding_provider=embedding_provider,
        default_top_k=top_k,
    )
    response = service.query(
        question,
        top_k=top_k,
        destination=destination,
        section=section,
        travel_type=travel_type,
        season=season,
        retrieval_mode=retrieval_mode,
    )

    if as_json:
        console.print_json(json.dumps(response.as_dict(), ensure_ascii=False))
        return

    console.print(Panel.fit(question, title="Query", border_style="cyan"))
    if not response.results:
        console.print("[yellow]No relevant chunks found.[/yellow]")
        return

    table = Table(title="Retrieved Chunks", show_lines=True)
    table.add_column("#", justify="right", style="cyan", width=4)
    table.add_column("Score", justify="right", style="green", width=10)
    table.add_column("Destination", style="magenta", width=16)
    table.add_column("Section", style="yellow", width=14)
    table.add_column("Source", style="blue", overflow="fold")
    table.add_column("Chunk", overflow="fold")

    for index, result in enumerate(response.results, start=1):
        table.add_row(
            str(index),
            f"{result.score:.4f}",
            result.destination,
            str(result.metadata.get("section", "")),
            result.source,
            _preview(result.content),
        )
    console.print(table)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Natural-language travel question.")],
    destination: Annotated[
        str | None,
        typer.Option("--destination", "-d", help="Filter by destination metadata."),
    ] = None,
    section: Annotated[
        str | None,
        typer.Option("--section", help="Filter by section metadata."),
    ] = None,
    travel_type: Annotated[
        str | None,
        typer.Option("--travel-type", help="Filter by travel_type metadata."),
    ] = None,
    season: Annotated[
        str | None,
        typer.Option("--season", help="Filter by season metadata."),
    ] = None,
    retrieval_mode: Annotated[
        RetrievalMode | None,
        typer.Option("--retrieval-mode", help="vector, keyword or hybrid."),
    ] = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Number of chunks to retrieve.")] = 5,
    persist_dir: Annotated[
        Path,
        typer.Option("--persist-dir", help="Chroma persistence directory."),
    ] = Path("data/chroma"),
    collection: Annotated[
        str,
        typer.Option("--collection", help="Chroma collection name."),
    ] = "travel_destinations",
    embedding_provider: Annotated[
        EmbeddingProviderName,
        typer.Option("--embedding-provider", help=EMBEDDING_PROVIDER_HELP),
    ] = EmbeddingProviderName.AUTO,
    as_json: Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")] = False,
) -> None:
    """Answer a question from the current destination knowledge base."""

    service = _service(
        persist_dir=persist_dir,
        collection=collection,
        embedding_provider=embedding_provider,
        default_top_k=top_k,
    )
    response = service.answer(
        question,
        top_k=top_k,
        destination=destination,
        section=section,
        travel_type=travel_type,
        season=season,
        retrieval_mode=retrieval_mode,
    )

    if as_json:
        console.print_json(json.dumps(response.as_dict(), ensure_ascii=False))
        return

    console.print(Panel.fit(question, title="Question", border_style="cyan"))
    console.print(Panel(response.answer, title="Knowledge Base Answer", border_style="green"))


@app.command()
def interactive(
    destination: Annotated[
        str | None,
        typer.Option("--destination", "-d", help="Optional fixed destination metadata filter."),
    ] = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Number of chunks to retrieve.")] = 5,
    persist_dir: Annotated[
        Path,
        typer.Option("--persist-dir", help="Chroma persistence directory."),
    ] = Path("data/chroma"),
    collection: Annotated[
        str,
        typer.Option("--collection", help="Chroma collection name."),
    ] = "travel_destinations",
    embedding_provider: Annotated[
        EmbeddingProviderName,
        typer.Option("--embedding-provider", help=EMBEDDING_PROVIDER_HELP),
    ] = EmbeddingProviderName.AUTO,
) -> None:
    """Start an interactive pure-RAG question loop."""

    service = _service(
        persist_dir=persist_dir,
        collection=collection,
        embedding_provider=embedding_provider,
        default_top_k=top_k,
    )
    console.print("[cyan]纯 RAG 交互查询已启动。输入 q / quit / exit 退出。[/cyan]")

    while True:
        try:
            question = typer.prompt("请输入问题").strip()
        except (typer.Abort, EOFError, KeyboardInterrupt):
            console.print("\n[green]已退出。[/green]")
            return

        if question.lower() in {"q", "quit", "exit"}:
            console.print("[green]已退出。[/green]")
            return
        if not question:
            continue

        response = service.answer(question, top_k=top_k, destination=destination)
        console.print(Panel(response.answer, title="Knowledge Base Answer", border_style="green"))


@app.command()
def stats(
    persist_dir: Annotated[
        Path,
        typer.Option("--persist-dir", help="Chroma persistence directory."),
    ] = Path("data/chroma"),
    collection: Annotated[
        str,
        typer.Option("--collection", help="Chroma collection name."),
    ] = "travel_destinations",
    embedding_provider: Annotated[
        EmbeddingProviderName,
        typer.Option("--embedding-provider", help=EMBEDDING_PROVIDER_HELP),
    ] = EmbeddingProviderName.AUTO,
) -> None:
    """Show Chroma collection statistics."""

    service = _service(
        persist_dir=persist_dir,
        collection=collection,
        embedding_provider=embedding_provider,
    )
    console.print_json(json.dumps(service.stats(), ensure_ascii=False))


@app.command("verify-embedding")
def verify_embedding(
    embedding_provider: Annotated[
        EmbeddingProviderName,
        typer.Option("--embedding-provider", help=EMBEDDING_PROVIDER_HELP),
    ] = EmbeddingProviderName.AUTO,
    embedding_model: Annotated[
        str | None,
        typer.Option(
            "--embedding-model",
            help="Override model for qwen, openai or sentence-transformers.",
        ),
    ] = None,
    dimensions: Annotated[
        int | None,
        typer.Option("--dimensions", help="Override qwen embedding dimensions."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")] = False,
) -> None:
    """Verify that an embedding provider can embed probe texts."""

    settings = _embedding_verify_settings(
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        dimensions=dimensions,
    )
    embeddings = build_embeddings(settings)
    document_vectors = embeddings.embed_documents(EMBEDDING_PROBE_TEXTS)
    query_vector = embeddings.embed_query(EMBEDDING_PROBE_TEXTS[0])
    all_vectors = [*document_vectors, query_vector]
    finite = all(all(math.isfinite(value) for value in vector) for vector in all_vectors)
    vector_dimensions = [len(vector) for vector in all_vectors]
    consistent_dimensions = len(set(vector_dimensions)) == 1
    payload = {
        "embedding_provider": embedding_provider.value,
        "embedding_model": _resolved_embedding_model(settings),
        "dimensions": vector_dimensions[0] if vector_dimensions else 0,
        "document_vectors": len(document_vectors),
        "query_vector": bool(query_vector),
        "finite": finite,
        "consistent_dimensions": consistent_dimensions,
        "status": "ok" if finite and consistent_dimensions and query_vector else "failed",
    }

    if as_json:
        console.print_json(json.dumps(payload, ensure_ascii=False))
    else:
        table = Table(title="Embedding Verification")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        for key, value in payload.items():
            table.add_row(key, str(value))
        console.print(table)

    if payload["status"] != "ok":
        raise typer.Exit(1)


@app.command("eval")
def eval_command(
    cases: Annotated[
        Path,
        typer.Option("--cases", help="JSONL evaluation fixture path."),
    ] = Path("tests/fixtures/rag_eval_cases.jsonl"),
    docs: Annotated[
        Path,
        typer.Option("--docs", help="Knowledge docs to ingest before evaluation."),
    ] = Path("docs/destinations"),
    persist_dir: Annotated[
        Path,
        typer.Option("--persist-dir", help="Chroma persistence directory."),
    ] = Path("data/eval_chroma"),
    collection: Annotated[
        str,
        typer.Option("--collection", help="Chroma collection name."),
    ] = "travel_destinations_eval",
    embedding_provider: Annotated[
        EmbeddingProviderName,
        typer.Option("--embedding-provider", help=EMBEDDING_PROVIDER_HELP),
    ] = EmbeddingProviderName.LOCAL,
    retrieval_mode: Annotated[
        RetrievalMode,
        typer.Option("--retrieval-mode", help="vector, keyword or hybrid."),
    ] = RetrievalMode.HYBRID,
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Number of chunks to evaluate.")] = 3,
    min_recall: Annotated[
        float,
        typer.Option("--min-recall", help="Minimum recall@k required by the quality gate."),
    ] = DEFAULT_QUALITY_THRESHOLDS["recall_at_k"],
    min_mrr: Annotated[
        float,
        typer.Option("--min-mrr", help="Minimum MRR@k required by the quality gate."),
    ] = DEFAULT_QUALITY_THRESHOLDS["mrr_at_k"],
    min_keyword_hit_rate: Annotated[
        float,
        typer.Option(
            "--min-keyword-hit-rate",
            help="Minimum keyword hit rate@k required by the quality gate.",
        ),
    ] = DEFAULT_QUALITY_THRESHOLDS["keyword_hit_rate_at_k"],
    min_metadata_accuracy: Annotated[
        float,
        typer.Option(
            "--min-metadata-accuracy",
            help="Minimum metadata filter accuracy required by the quality gate.",
        ),
    ] = DEFAULT_QUALITY_THRESHOLDS["metadata_filter_accuracy"],
    min_expected_empty_accuracy: Annotated[
        float,
        typer.Option(
            "--min-expected-empty-accuracy",
            help="Minimum expected-empty accuracy required by the quality gate.",
        ),
    ] = DEFAULT_QUALITY_THRESHOLDS["expected_empty_accuracy"],
    quality_gate: Annotated[
        bool,
        typer.Option(
            "--quality-gate/--no-quality-gate",
            help="Exit with code 1 when metrics fall below thresholds.",
        ),
    ] = True,
    as_json: Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")] = False,
) -> None:
    """Run pure-RAG retrieval quality evaluation."""

    thresholds = {
        "recall_at_k": min_recall,
        "mrr_at_k": min_mrr,
        "keyword_hit_rate_at_k": min_keyword_hit_rate,
        "metadata_filter_accuracy": min_metadata_accuracy,
        "expected_empty_accuracy": min_expected_empty_accuracy,
    }
    report = evaluate_rag(
        docs_path=docs,
        cases_path=cases,
        persist_dir=persist_dir,
        collection_name=collection,
        embedding_provider=embedding_provider,
        retrieval_mode=retrieval_mode,
        top_k=top_k,
        thresholds=thresholds,
    )
    payload = report.as_dict()
    if as_json:
        console.print_json(json.dumps(payload, ensure_ascii=False))
        if quality_gate and not report.quality_gate["passed"]:
            raise typer.Exit(1)
        return

    table = Table(title="RAG Evaluation")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Threshold", style="yellow")
    for key, value in payload["metrics"].items():
        threshold = thresholds.get(key)
        table.add_row(
            key,
            f"{value:.4f}" if isinstance(value, float) else str(value),
            f"{threshold:.4f}" if threshold is not None else "",
        )
    console.print(table)
    console.print_json(json.dumps(payload["run"], ensure_ascii=False))
    if report.quality_gate["passed"]:
        console.print("[green]RAG quality gate passed.[/green]")
        return

    console.print("[red]RAG quality gate failed.[/red]")
    for failure in report.quality_gate["failures"]:
        console.print(f"[red]- {failure}[/red]")
    if quality_gate:
        raise typer.Exit(1)


@app.command()
def reset(
    persist_dir: Annotated[
        Path,
        typer.Option("--persist-dir", help="Chroma persistence directory."),
    ] = Path("data/chroma"),
    collection: Annotated[
        str,
        typer.Option("--collection", help="Chroma collection name."),
    ] = "travel_destinations",
    embedding_provider: Annotated[
        EmbeddingProviderName,
        typer.Option("--embedding-provider", help=EMBEDDING_PROVIDER_HELP),
    ] = EmbeddingProviderName.AUTO,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    """Delete the local Chroma index."""

    if not yes and not typer.confirm(f"Delete Chroma index at {persist_dir}?"):
        raise typer.Abort()

    settings = RagSettings(
        persist_dir=persist_dir,
        collection_name=collection,
        embedding_provider=embedding_provider,
    )
    reset_chroma(settings)
    console.print("[green]Chroma index reset.[/green]")


def _service(
    persist_dir: Path,
    collection: str,
    embedding_provider: EmbeddingProviderName,
    default_top_k: int = 5,
) -> RagService:
    settings = RagSettings(
        persist_dir=persist_dir,
        collection_name=collection,
        embedding_provider=embedding_provider,
        default_top_k=default_top_k,
    )
    return RagService(settings=settings)


def _embedding_verify_settings(
    embedding_provider: EmbeddingProviderName,
    embedding_model: str | None,
    dimensions: int | None,
) -> RagSettings:
    updates: dict[str, object] = {"embedding_provider": embedding_provider}
    if embedding_model:
        if embedding_provider in {EmbeddingProviderName.QWEN, EmbeddingProviderName.DASHSCOPE}:
            updates["qwen_embedding_model"] = embedding_model
        elif embedding_provider is EmbeddingProviderName.OPENAI:
            updates["openai_embedding_model"] = embedding_model
        elif embedding_provider is EmbeddingProviderName.SENTENCE_TRANSFORMERS:
            updates["sentence_transformers_model"] = embedding_model
    if dimensions is not None:
        updates["qwen_embedding_dimensions"] = dimensions
    return RagSettings().model_copy(update=updates)


def _resolved_embedding_model(settings: RagSettings) -> str:
    if settings.embedding_provider in {EmbeddingProviderName.QWEN, EmbeddingProviderName.DASHSCOPE}:
        return settings.qwen_embedding_model
    if settings.embedding_provider is EmbeddingProviderName.OPENAI:
        return settings.openai_embedding_model
    if settings.embedding_provider is EmbeddingProviderName.SENTENCE_TRANSFORMERS:
        return settings.sentence_transformers_model
    if settings.embedding_provider is EmbeddingProviderName.AUTO:
        return (
            settings.qwen_embedding_model
            if os.getenv("DASHSCOPE_API_KEY")
            else settings.openai_embedding_model
            if os.getenv("OPENAI_API_KEY")
            else "LocalHashEmbeddings"
        )
    return "LocalHashEmbeddings"


def _preview(text: str, limit: int = 420) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else normalized[: limit - 3] + "..."


if __name__ == "__main__":
    main()
