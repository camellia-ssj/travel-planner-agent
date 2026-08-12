"""DSPy 编译与 Prompt 管理的 CLI 子命令。

用法::

    # 从 JSONL 训练数据编译
    travel-agent dspy compile --train-examples tests/fixtures/dspy_train_examples.jsonl

    # 查看编译状态
    travel-agent dspy info --load-path data/dspy/compiled_planner.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

dspy_app = typer.Typer(
    name="dspy",
    help="DSPy 声明式优化：编译、查看和导出优化后的 Prompt。",
    no_args_is_help=True,
)
console = Console()


@dspy_app.command()
def compile(
    train_examples: Annotated[
        Path,
        typer.Option(
            "--train-examples",
            "-t",
            help="JSONL 训练示例文件路径。",
            exists=True,
        ),
    ],
    save_path: Annotated[
        Path,
        typer.Option(
            "--save-path",
            "-o",
            help="编译产物保存路径。",
        ),
    ] = Path("data/dspy/compiled_planner.json"),
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="DSPy 编译使用的模型名称。"),
    ] = "qwen3-max",
    optimizer: Annotated[
        str,
        typer.Option("--optimizer", help="DSPy 优化器: BootstrapFewShot, BootstrapFewShotWithRandomSearch。"),
    ] = "BootstrapFewShot",
    max_bootstrapped: Annotated[
        int,
        typer.Option("--max-bootstrapped", help="最大自举示例数。", min=0),
    ] = 4,
    max_labeled: Annotated[
        int,
        typer.Option("--max-labeled", help="最大标注示例数。", min=0),
    ] = 4,
    evidence_dir: Annotated[
        Path | None,
        typer.Option("--evidence-dir", help="可选—用于恢复证据引用的 Chroma 持久化目录。"),
    ] = None,
) -> None:
    """离线编译旅行规划 DSPy 模块，自动调优 Prompt。

    从 JSONL 训练示例中学习最优 few-shot 示例组合，
    以 Reflection 审校指标（证据覆盖率、置信度）为优化目标。

    编译产物保存为 JSON，可通过 ``--optimize`` 标志在运行
    ``travel-agent plan`` 时加载。
    """
    from travel_agent.agent.dspy_planner import (
        DSPyCompileSettings,
        compile_dspy_planner,
    )
    from travel_agent.agent.dspy_data import load_training_examples

    console.print(f"[cyan]加载训练示例:[/cyan] {train_examples}")
    examples = load_training_examples(train_examples)

    if not examples:
        console.print("[yellow]未找到训练示例 — 将生成未编译模块。[/yellow]")
    else:
        console.print(f"[green]已加载 {len(examples)} 条训练示例。[/green]")

    settings = DSPyCompileSettings(
        model=model,
        optimizer=optimizer,
        max_bootstrapped_demos=max_bootstrapped,
        max_labeled_demos=max_labeled,
        save_path=save_path,
    )

    console.print(f"[cyan]优化器:[/cyan] {settings.optimizer}")
    console.print(f"[cyan]模型:[/cyan] {settings.model}")
    console.print(f"[cyan]最大自举示例:[/cyan] {settings.max_bootstrapped_demos}")
    console.print(f"[cyan]最大标注示例:[/cyan] {settings.max_labeled_demos}")
    console.print()

    try:
        module = compile_dspy_planner(examples, settings)
    except ValueError as exc:
        console.print(f"[red]编译失败: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    if module.is_compiled:
        console.print(
            f"[green]✓ 编译完成！[/green] 产物已保存至 [bold]{save_path}[/bold]"
        )
        console.print(
            "[dim]运行 travel-agent plan --optimize 以使用编译模块。[/dim]"
        )
    else:
        console.print(
            "[yellow]未执行编译（无训练示例）。未编译模块已就绪。[/yellow]"
        )


@dspy_app.command()
def info(
    load_path: Annotated[
        Path,
        typer.Option(
            "--load-path",
            "-l",
            help="编译产物 JSON 文件路径。",
            exists=True,
        ),
    ] = Path("data/dspy/compiled_planner.json"),
) -> None:
    """查看已编译 DSPy 模块的信息。"""
    from travel_agent.agent.dspy_planner import load_compiled_planner

    module = load_compiled_planner(load_path)
    if module is None:
        console.print(f"[red]未找到编译产物: {load_path}[/red]")
        console.print("[dim]运行 'travel-agent dspy compile' 以创建编译产物。[/dim]")
        raise typer.Exit(code=1)

    console.print(f"[green]✓ 已加载编译模块:[/green] {load_path}")

    if module.is_compiled:
        console.print("[green]状态:[/green] 已编译（包含优化后的 few-shot 示例）")
        try:
            demos = getattr(module._module.generate, "demos", [])
            console.print(f"[cyan]Few-shot 示例数:[/cyan] {len(demos)}")
        except Exception:
            pass
    else:
        console.print("[yellow]状态:[/yellow] 未编译（无优化示例）")
        console.print(
            "[dim]运行 'travel-agent dspy compile' 并传入训练示例以编译。[/dim]"
        )
