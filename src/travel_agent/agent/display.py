"""旅行规划 Agent CLI 和 REPL 共享的 Rich 展示工具函数。"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def display_plan_payload(payload: dict[str, Any]) -> None:
    """美化打印完整的旅行规划结果。"""
    request = payload.get("request", {})
    plan_payload = payload.get("plan", {})
    validation = payload.get("validation", {})

    console.print(
        Panel.fit(
            plan_payload.get("summary", ""),
            title="旅行计划",
            border_style="cyan",
        )
    )
    if payload.get("thread_id"):
        console.print(f"[cyan]thread_id:[/cyan] {payload['thread_id']}")
    if payload.get("user_profile"):
        _print_user_profile(payload["user_profile"])
    if payload.get("user_feedback"):
        _print_list("用户反馈", payload["user_feedback"])
    _print_request(request)
    _print_day_plans(plan_payload.get("day_plans", []))
    _print_budget(plan_payload.get("budget_items", []))
    _print_risks(plan_payload.get("risk_notices", []))
    _print_list("备选方案", plan_payload.get("alternatives", []))
    _print_list("证据来源", plan_payload.get("evidence_sources", []))

    if not validation.get("is_valid", True):
        _print_list("校验错误", validation.get("errors", []))

    reflection = payload.get("reflection")
    if reflection:
        _print_reflection(reflection)


def _print_request(request: dict[str, Any]) -> None:
    table = Table(title="解析后的旅行需求")
    table.add_column("字段", style="cyan")
    table.add_column("值", style="green")
    table.add_row("目的地", str(request.get("destination", "")))
    table.add_row("天数", str(request.get("days", "")))
    table.add_row("出行人员", ", ".join(request.get("audience", [])))
    table.add_row("预算偏好", str(request.get("budget_preference", "")))
    console.print(table)


def _print_day_plans(day_plans: list[dict[str, Any]]) -> None:
    table = Table(title="每日行程", show_lines=True)
    table.add_column("天", justify="right", style="cyan", width=5)
    table.add_column("标题", style="green")
    table.add_column("活动", overflow="fold")
    for day_plan in day_plans:
        table.add_row(
            str(day_plan.get("day", "")),
            str(day_plan.get("title", "")),
            "\n".join(f"- {activity}" for activity in day_plan.get("activities", [])),
        )
    console.print(table)


def _print_budget(budget_items: list[dict[str, Any]]) -> None:
    table = Table(title="预算估算")
    table.add_column("类别", style="cyan")
    table.add_column("偏好", style="green")
    table.add_column("备注", overflow="fold")
    for item in budget_items:
        table.add_row(
            str(item.get("category", "")),
            str(item.get("preference", "")),
            str(item.get("note", "")),
        )
    console.print(table)


def _print_risks(risk_notices: list[dict[str, Any]]) -> None:
    table = Table(title="风险提示")
    table.add_column("类型", style="cyan")
    table.add_column("严重程度", style="yellow")
    table.add_column("信息", overflow="fold")
    for notice in risk_notices:
        table.add_row(
            str(notice.get("risk_type", "")),
            str(notice.get("severity", "")),
            str(notice.get("message", "")),
        )
    console.print(table)


def _print_list(title: str, values: list[str]) -> None:
    table = Table(title=title)
    table.add_column("#", justify="right", style="cyan", width=4)
    table.add_column("值", overflow="fold")
    for index, value in enumerate(values, start=1):
        table.add_row(str(index), value)
    console.print(table)


def _print_user_profile(profile: dict[str, Any]) -> None:
    table = Table(title="用户画像（记忆）")
    table.add_column("字段", style="cyan")
    table.add_column("值", style="green")
    table.add_row("用户ID", str(profile.get("user_id", "")))
    table.add_row("历史行程", str(profile.get("total_trips", "")))
    if profile.get("preferred_destinations"):
        table.add_row(
            "偏好目的地",
            ", ".join(profile["preferred_destinations"]),
        )
    if profile.get("audience_types"):
        table.add_row(
            "出行类型",
            ", ".join(profile.get("audience_types", [])),
        )
    table.add_row("预算偏好", str(profile.get("budget_preference", "")))
    if profile.get("trip_length_avg"):
        table.add_row(
            "平均行程长度",
            f"{profile['trip_length_avg']:.1f} 天",
        )
    if profile.get("preferences_summary"):
        table.add_row("摘要", profile.get("preferences_summary", ""))
    console.print(table)


def _print_reflection(reflection: dict[str, Any]) -> None:
    """打印事实性审核报告。"""
    passed = reflection.get("passed", False)
    status_color = "green" if passed else "yellow"
    status_text = "通过" if passed else "存疑"

    title = f"审核报告 — {status_text}"
    console.print(Panel.fit(title, border_style=status_color))

    coverage = reflection.get("evidence_coverage", 0.0)
    confidence = reflection.get("confidence_score", 0.0)
    checked = reflection.get("checked_claims", 0)
    grounded = reflection.get("grounded_claims", 0)

    summary_text = (
        f"证据覆盖率: {coverage:.0%}  |  "
        f"置信度: {confidence:.0%}  |  "
        f"已验证声明: {grounded}/{checked}"
    )
    console.print(f"[dim]{summary_text}[/dim]")

    flags = reflection.get("hallucination_flags", [])
    if flags:
        flag_table = Table(title="幻觉标记")
        flag_table.add_column("位置", style="cyan")
        flag_table.add_column("严重程度", style="red")
        flag_table.add_column("声明", overflow="fold", max_width=60)
        flag_table.add_column("问题", overflow="fold", max_width=40)
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
        console.print("[bold yellow]问题:[/bold yellow]")
        for issue in issues:
            console.print(f"  [yellow]- {issue}[/yellow]")

    suggestions = reflection.get("suggestions", [])
    if suggestions:
        console.print("[bold cyan]建议:[/bold cyan]")
        for suggestion in suggestions:
            console.print(f"  [dim]- {suggestion}[/dim]")
