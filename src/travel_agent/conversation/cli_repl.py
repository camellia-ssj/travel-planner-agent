"""对话式旅行规划 Agent 的交互式 REPL。

基于 Rich 的终端聊天界面，支持流式输出、命令历史、
斜杠命令和会话管理。
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, AIMessage
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from travel_agent.agent.display import display_plan_payload
from travel_agent.conversation.state import ConversationState

logger = logging.getLogger(__name__)

# ── 斜杠命令 ─────────────────────────────────────────────────

_SLASH_COMMANDS: dict[str, str] = {
    "/plan": "立即用当前收集到的信息生成旅行计划",
    "/feedback": "对当前计划提出修改意见，如 /feedback 改成北京",
    "/profile": "查看用户画像和历史行程",
    "/history": "查看当前对话摘要",
    "/reset": "重新开始一个新的旅行规划",
    "/export": "导出当前计划为 JSON",
    "/help": "显示帮助信息",
    "/quit": "退出对话（同 /exit）",
    "/exit": "退出对话",
}

# ── Readline 设置 ─────────────────────────────────────────────────

try:
    import readline
    _HAS_READLINE = True
except ImportError:
    try:
        import pyreadline3 as readline
        _HAS_READLINE = True
    except ImportError:
        _HAS_READLINE = False


def _setup_readline() -> None:
    """配置 readline 命令历史。"""
    if not _HAS_READLINE:
        return
    histfile = Path.home() / ".travel_agent_history"
    try:
        readline.read_history_file(str(histfile))
    except (FileNotFoundError, OSError):
        pass
    readline.set_history_length(1000)


def _save_readline() -> None:
    """持久化命令历史。"""
    if not _HAS_READLINE:
        return
    histfile = Path.home() / ".travel_agent_history"
    try:
        readline.write_history_file(str(histfile))
    except OSError:
        pass


def _add_history(line: str) -> None:
    """将一行添加到 readline 历史中。"""
    if _HAS_READLINE and line.strip():
        readline.add_history(line.strip())


# ── REPL 实现 ────────────────────────────────────────────


class ConversationREPL:
    """对话式旅行规划 Agent 的交互式 REPL。"""

    def __init__(
        self,
        graph: Any,
        console: Console | None = None,
        user_id: str | None = None,
        thread_id: str | None = None,
        streaming: bool = True,
    ) -> None:
        self.graph = graph
        self.console = console or Console()
        self.user_id = user_id or ""
        self.thread_id = thread_id or uuid.uuid4().hex
        self.streaming = streaming
        self._running = False
        self._config = {"configurable": {"thread_id": self.thread_id}}
        # 在多次调用之间缓存槽位状态以增强健壮性
        self._cached_state: dict[str, object] = {}

    # ── 公开 API ─────────────────────────────────────────────────

    def run(self) -> None:
        """启动交互式 REPL 循环。"""
        _setup_readline()
        self._running = True

        self._print_welcome()
        self._print_greeting()

        # 主输入循环——每回合从 START 调用一次图
        while self._running:
            try:
                user_input = self._prompt_user()
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[dim]再见！[/dim]")
                break

            if not user_input.strip():
                continue

            # 处理斜杠命令
            if user_input.startswith("/"):
                self._handle_slash_command(user_input.strip())
                continue

            _add_history(user_input)

            try:
                result = self._process_message(user_input)
                self._print_ai_messages(result)
            except Exception:
                logger.warning("Error processing message")
                self.console.print("[red]处理消息时出错，请重试[/red]")

        _save_readline()

    # ── 消息处理 ─────────────────────────────────────────────

    def _process_message(self, text: str) -> dict[str, Any]:
        """将用户消息送入图并返回结果。

        之前提取的槽位与新消息一起传入，
        使澄清节点能在各回合之间保留信息。
        """
        update: dict[str, object] = {
            "messages": [HumanMessage(content=text)],
            "user_message": text,
            **self._cached_state,  # 传入上一轮已知槽位
        }
        result = self.graph.invoke(update, self._config)

        # 缓存槽位值供下一轮使用
        for key in (
            "clarified_destination", "clarified_days",
            "clarified_budget", "clarified_audience",
            "original_request_text", "clarification_turn_count",
            "planning_output", "plan_generation_count",
        ):
            if key in result and result[key] is not None:
                self._cached_state[key] = result[key]

        return result

    def _print_ai_messages(self, result: dict[str, Any]) -> None:
        """仅打印图中的新 AI 消息。"""
        messages = result.get("messages", [])

        # 检查是否有计划数据需要展示
        planning_output = result.get("planning_output", {})
        if planning_output and planning_output.get("plan"):
            self._print_plan_panel(planning_output)

        # 打印最后一条 AI 消息（最新回复）
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]
        if ai_messages:
            latest = ai_messages[-1]
            content = latest.content if hasattr(latest, "content") else str(latest)
            self.console.print()
            if isinstance(content, str) and len(content) > 50:
                self.console.print(Markdown(str(content)))
            else:
                self.console.print(Panel.fit(
                    str(content),
                    border_style="cyan",
                    title="小旅",
                    title_align="left",
                ))

        # 如果对话自然结束，停止 REPL
        phase = result.get("phase", "")
        feedback_action = result.get("feedback_action", "")
        if phase == "feedback" and feedback_action == "approve":
            self.console.print()
            self.console.print("[green]感谢使用！如需修改计划，继续输入您的需求即可。[/green]")

    def _print_plan_panel(self, planning_output: dict[str, Any]) -> None:
        """使用 Rich 表格展示旅行计划。"""
        payload = {
            "thread_id": self.thread_id,
            "plan": planning_output.get("plan", {}),
            "request": planning_output.get("request", {}),
            "evidence": {
                "results": [],
                "evidence_count": planning_output.get("evidence_count", 0),
            },
            "validation": {
                "is_valid": planning_output.get("is_valid", False),
                "errors": planning_output.get("validation_errors", []),
            },
            "tool_budget": planning_output.get("tool_budget"),
            "tool_crowd_risk": planning_output.get("tool_crowd_risk"),
            "tool_alternatives": planning_output.get("tool_alternatives"),
            "reflection": planning_output.get("reflection_report"),
            "original_user_request": planning_output.get("plan_summary", ""),
            "user_feedback": [],
        }
        try:
            display_plan_payload(payload)
        except Exception:
            logger.warning("Failed to display plan panel")

    # ── 用户输入 ─────────────────────────────────────────────────

    def _prompt_user(self) -> str:
        """显示提示符并读取用户输入。"""
        self.console.print()
        try:
            return input(f"{Text('您', style='bold green')}: ")
        except (KeyboardInterrupt, EOFError):
            return "/quit"

    # ── 欢迎界面 ─────────────────────────────────────────────

    def _print_welcome(self) -> None:
        """打印欢迎横幅。"""
        self.console.print()
        self.console.print(
            Panel.fit(
                "[bold cyan]AI 旅行规划助手[/bold cyan]\n"
                "[dim]基于 LangChain + LangGraph 的智能旅行规划 Agent[/dim]\n\n"
                "[dim]输入 /help 查看可用命令，输入 /quit 退出对话[/dim]",
                border_style="cyan",
            )
        )

    def _print_greeting(self) -> None:
        """打印初始问候语。"""
        greeting = (
            "您好！👋 我是您的旅行规划顾问**小旅**。\n\n"
            "我可以帮您规划旅行路线，提供预算估算、拥挤风险提醒和备选方案。\n"
            "只需告诉我您的需求，比如：\n"
            "- 目的地（如杭州、成都、东京...）\n"
            "- 游玩天数\n"
            "- 预算偏好（经济/标准/高端）\n"
            "- 出行人员（亲子/情侣/朋友...）\n\n"
            "现在，告诉我您想去哪里吧！"
        )
        self.console.print(Panel.fit(greeting, border_style="cyan", title="小旅", title_align="left"))

    # ── 斜杠命令处理 ─────────────────────────────────────────────

    def _handle_slash_command(self, cmd: str) -> None:
        """将斜杠命令路由到对应的处理器。"""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        handlers = {
            "/plan": self._cmd_plan,
            "/feedback": self._cmd_feedback,
            "/profile": self._cmd_profile,
            "/history": self._cmd_history,
            "/reset": self._cmd_reset,
            "/export": self._cmd_export,
            "/help": self._cmd_help,
            "/quit": self._cmd_quit,
            "/exit": self._cmd_quit,
        }

        handler = handlers.get(command)
        if handler:
            handler(arg)
        else:
            self.console.print(f"[yellow]未知命令: {command}，输入 /help 查看可用命令[/yellow]")

    def _cmd_plan(self, arg: str) -> None:
        """用当前槽位强制触发规划。"""
        self.console.print("[cyan]正在为您生成旅行计划...[/cyan]")
        _add_history("/plan")
        try:
            update: dict[str, object] = {
                "messages": [HumanMessage(content="请为我规划行程")],
                "user_message": "请为我规划行程",
                "slot_filling_complete": True,
                "phase": "planning",
                **self._cached_state,
            }
            result = self.graph.invoke(update, self._config)
            # 从结果更新缓存
            for key in (
                "clarified_destination", "clarified_days",
                "clarified_budget", "clarified_audience",
                "original_request_text", "clarification_turn_count",
                "planning_output", "plan_generation_count",
            ):
                if key in result and result[key] is not None:
                    self._cached_state[key] = result[key]
            self._print_ai_messages(result)
        except Exception:
            self.console.print("[red]规划生成失败[/red]")

    def _cmd_feedback(self, arg: str) -> None:
        """对当前计划提供反馈。"""
        if not arg:
            self.console.print("[yellow]请在 /feedback 后面输入您的修改意见[/yellow]")
            return
        _add_history(f"/feedback {arg}")
        self.console.print(f"[cyan]收到反馈: {arg}[/cyan]")
        try:
            result = self._process_message(arg)
            self._print_ai_messages(result)
        except Exception:
            self.console.print("[red]反馈处理失败[/red]")

    def _cmd_profile(self, arg: str) -> None:
        """展示图中存储的用户画像。"""
        try:
            snapshot = self.graph.get_state(self._config)
            if snapshot and snapshot.values:
                profile = snapshot.values.get("user_profile")
                if profile:
                    self._print_profile_table(profile)
                else:
                    self.console.print("[dim]暂无用户画像。多使用几次后会自动生成。[/dim]")
            else:
                self.console.print("[dim]暂无会话状态[/dim]")
        except Exception:
            self.console.print("[dim]暂无用户画像[/dim]")

    def _cmd_history(self, arg: str) -> None:
        """展示对话摘要。"""
        try:
            snapshot = self.graph.get_state(self._config)
            if snapshot and snapshot.values:
                summary = snapshot.values.get("conversation_history_summary", "")
                phase = snapshot.values.get("phase", "")
                dest = snapshot.values.get("clarified_destination", "")
                days = snapshot.values.get("clarified_days", 0)
                budget = snapshot.values.get("clarified_budget", "")
                audience = snapshot.values.get("clarified_audience", [])

                table = Table(title="对话状态")
                table.add_column("项目", style="cyan")
                table.add_column("内容", style="green")
                table.add_row("阶段", phase)
                table.add_row("目的地", str(dest) if dest else "待定")
                table.add_row("天数", str(days) if days else "待定")
                table.add_row("预算", str(budget) if budget else "待定")
                table.add_row("出行人员", ", ".join(audience) if audience else "待定")
                self.console.print(table)

                if summary:
                    self.console.print(Panel(summary, title="对话摘要", border_style="dim"))
            else:
                self.console.print("[dim]暂无对话历史[/dim]")
        except Exception:
            self.console.print("[dim]暂无对话历史[/dim]")

    def _cmd_reset(self, arg: str) -> None:
        """重置对话。"""
        self.thread_id = uuid.uuid4().hex
        self._config = {"configurable": {"thread_id": self.thread_id}}
        self._cached_state = {}
        self.console.print("[cyan]会话已重置，开始新的旅行规划[/cyan]")
        self._print_greeting()

    def _cmd_export(self, arg: str) -> None:
        """导出当前计划为 JSON。"""
        import json

        try:
            snapshot = self.graph.get_state(self._config)
            if snapshot and snapshot.values:
                planning_output = snapshot.values.get("planning_output", {})
                if planning_output and planning_output.get("plan"):
                    output_path = Path(f"plan_{self.thread_id[:8]}.json")
                    output_path.write_text(
                        json.dumps(planning_output, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    self.console.print(f"[green]计划已导出到: {output_path}[/green]")
                else:
                    self.console.print("[yellow]暂无计划可导出，请先生成计划[/yellow]")
            else:
                self.console.print("[dim]暂无数据可导出[/dim]")
        except Exception:
            self.console.print("[red]导出失败[/red]")

    def _cmd_help(self, arg: str) -> None:
        """打印帮助。"""
        self.console.print()
        self.console.print("[bold]可用命令:[/bold]")
        for cmd, desc in _SLASH_COMMANDS.items():
            self.console.print(f"  [cyan]{cmd:12}[/cyan] {desc}")

        self.console.print()
        self.console.print("[bold]使用提示:[/bold]")
        self.console.print("  - 直接用自然语言描述你的旅行需求")
        self.console.print("  - 可以随时修改已提供的信息")
        self.console.print("  - 生成计划后可以要求调整（如\"第二天少走路\"）")

    def _cmd_quit(self, arg: str) -> None:
        """退出 REPL。"""
        self._running = False
        self.console.print("[dim]再见！[/dim]")

    def _print_profile_table(self, profile: Any) -> None:
        """打印用户画像 Rich 表格。"""
        table = Table(title="用户画像")
        table.add_column("项目", style="cyan")
        table.add_column("内容", style="green")

        user_id = getattr(profile, "user_id", "")
        table.add_row("用户ID", str(user_id))

        total = getattr(profile, "total_trips", 0)
        table.add_row("历史行程", str(total))

        preferred = getattr(profile, "preferred_destinations", [])
        if preferred:
            table.add_row("偏好目的地", ", ".join(preferred))

        audience = getattr(profile, "audience_types", [])
        if audience:
            table.add_row("出行类型", ", ".join(audience))

        budget = getattr(profile, "budget_preference", "")
        if budget:
            budget_labels = {"economy": "经济实惠", "premium": "高端舒适", "standard": "中等标准"}
            table.add_row("消费偏好", budget_labels.get(str(budget), str(budget)))

        avg_days = getattr(profile, "trip_length_avg", 0.0)
        if avg_days:
            table.add_row("平均天数", f"{avg_days:.1f}天")

        summary = getattr(profile, "preferences_summary", "")
        if summary:
            table.add_row("概要", str(summary))

        self.console.print(table)
