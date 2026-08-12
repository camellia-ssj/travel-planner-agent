"""DSPy 声明式优化模块 — 旅行规划任务的自动 Prompt 调优。

通过 DSPy Signature 定义输入/输出契约，使用 ChainOfThought 模块
封装推理过程，以 Reflection 指标作为优化目标，经由 BootstrapFewShot
等编译器自动发现最优 few-shot 示例和指令。

使用方式::

    # 离线编译
    travel-agent dspy compile --train-examples data/train.jsonl

    # 运行时加载编译产物
    from travel_agent.agent.dspy_planner import DSPyOptimizedPlanner, load_compiled_planner

    module = load_compiled_planner(Path("data/dspy/compiled_planner.json"))
    planner = DSPyOptimizedPlanner(dspy_module=module, fallback=base_planner)
    plan = planner.plan(request, evidence)  # 符合 TravelPlanner Protocol
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from travel_agent.agent.planner import TravelPlanner
from travel_agent.agent.schemas import TravelPlan, TravelRequest
from travel_agent.memory.models import UserProfile
from travel_agent.rag.models import EvidenceBundle, SearchResult
from travel_agent.skills.models import SkillSelection

# ---------------------------------------------------------------------------
# DSPy Signature
# ---------------------------------------------------------------------------


def _build_travel_planning_signature() -> type:
    """构建旅行规划 DSPy Signature 类（延迟导入 dspy）。"""
    import dspy

    class TravelPlanningSignature(dspy.Signature):
        """Generate a structured TravelPlan JSON from request, evidence, and tools.

        The output must be valid JSON matching the TravelPlan schema:
        - destination, days, summary, day_plans (with day/title/activities/evidence_sources)
        - budget_items (category/preference/note), risk_notices (risk_type/message/severity)
        - alternatives (list of strings), evidence_sources (list of strings)
        - fallback_used (boolean, set to false)

        Never fabricate POI names, prices, or crowd statistics beyond what
        the evidence provides. Every activity must cite its evidence source.
        """

        request: str = dspy.InputField(
            desc="User's travel request: destination, days, audience, budget preference"
        )
        evidence: str = dspy.InputField(
            desc="Retrieved RAG evidence corpus (source, content, section, score)"
        )
        tools: str = dspy.InputField(
            desc="Deterministic tool results: budget estimate, crowd risk, alternatives"
        )
        profile: str = dspy.InputField(
            desc="User profile context from long-term memory (may be empty)"
        )
        plan: str = dspy.OutputField(
            desc="Complete TravelPlan as JSON matching the schema exactly"
        )

    return TravelPlanningSignature


# ---------------------------------------------------------------------------
# DSPy Module
# ---------------------------------------------------------------------------


class TravelPlanningModule:
    """DSPy 模块 — 封装旅行规划 ChainOfThought 推理。

    此类延迟构造内部的 dspy.Module 实例，以避免在未安装
    dspy 或未配置 LM 时触发导入错误。调用 :meth:`ensure_module`
    完成初始化。

    用法::

        module = TravelPlanningModule()
        module.ensure_module()          # 触发 dspy.Module 构造
        pred = module.forward(req, ev, tools, profile)
    """

    def __init__(self) -> None:
        self._module: Any = None
        self._signature: type | None = None

    def ensure_module(self) -> None:
        """延迟构造内部 dspy.Module（首次调用时触发 dspy 导入）。"""
        if self._module is not None:
            return
        import dspy

        self._signature = _build_travel_planning_signature()

        class _Module(dspy.Module):
            def __init__(self_, sig: type) -> None:
                super().__init__()
                self_.generate = dspy.ChainOfThought(sig)

            def forward(self_, request: str, evidence: str, tools: str, profile: str) -> Any:
                return self_.generate(
                    request=request, evidence=evidence, tools=tools, profile=profile
                )

        self._module = _Module(self._signature)

    def forward(self, request: str, evidence: str, tools: str, profile: str) -> Any:
        """执行一次规划推理，返回 dspy.Prediction。"""
        self.ensure_module()
        return self._module.forward(
            request=request, evidence=evidence, tools=tools, profile=profile
        )

    def save(self, path: str | Path) -> None:
        """将编译后的模块持久化为 JSON 文件。"""
        self.ensure_module()
        self._module.save(str(path))

    def load(self, path: str | Path) -> None:
        """从 JSON 文件加载编译后的模块。"""
        self.ensure_module()
        self._module.load(str(path))

    @property
    def is_compiled(self) -> bool:
        """模块是否已完成编译（包含优化后的 few-shot 示例）。"""
        if self._module is None:
            return False
        try:
            demos = getattr(self._module.generate, "demos", None)
            return demos is not None and len(demos) > 0
        except Exception:
            return False


# ---------------------------------------------------------------------------
# 序列化辅助函数
# ---------------------------------------------------------------------------


def _serialize_request(request: TravelRequest) -> str:
    """将 TravelRequest 序列化为紧凑 JSON 字符串。"""
    return json.dumps(
        {
            "raw_query": request.raw_query,
            "destination": request.destination,
            "days": request.days,
            "audience": request.audience,
            "budget_preference": request.budget_preference,
        },
        ensure_ascii=False,
    )


def _serialize_evidence(evidence: EvidenceBundle) -> str:
    """将 EvidenceBundle 格式化为可读文本块。

    每个结果按 ``[N] source=... section=... score=...`` 格式输出，
    方便 DSPy 优化器理解证据结构。
    """
    lines: list[str] = []
    for idx, result in enumerate(evidence.results, start=1):
        meta_parts = [
            f"source={result.source}",
        ]
        dest = result.destination or result.metadata.get("destination", "")
        if dest:
            meta_parts.append(f"destination={dest}")
        section = result.metadata.get("section", "")
        if section:
            meta_parts.append(f"section={section}")
        score = result.score
        if score:
            meta_parts.append(f"score={score:.3f}")
        header = f"[{idx}] " + "; ".join(meta_parts)
        content = result.content.strip()
        lines.append(f"{header}\n{content}")
    return "\n\n".join(lines)


def _serialize_tool_results(tool_results: dict[str, object] | None) -> str:
    """将确定性工具结果序列化为紧凑 JSON。"""
    if not tool_results:
        return "{}"
    # 仅保留可 JSON 序列化的键
    clean: dict[str, object] = {}
    for key in ("tool_budget", "tool_crowd_risk", "tool_alternatives"):
        value = tool_results.get(key)
        if value is not None:
            if hasattr(value, "model_dump"):
                clean[key] = value.model_dump()
            elif hasattr(value, "__dict__"):
                clean[key] = {
                    k: v
                    for k, v in value.__dict__.items()
                    if not k.startswith("_")
                }
            else:
                clean[key] = value
    return json.dumps(clean, ensure_ascii=False, default=str)


def _serialize_profile(profile: UserProfile | None) -> str:
    """将用户画像序列化为自然语言文本。"""
    if profile is None:
        return "无用户历史记录。"
    context = profile.to_context_text()
    return context or "无用户历史记录。"


def _parse_plan_json(plan_json: str) -> TravelPlan | None:
    """将 DSPy 输出的 JSON 字符串解析为 TravelPlan。

    尝试多种解析策略以容错 LLM 输出格式：
    1. 直接 JSON 解析
    2. 提取 markdown 代码块中的 JSON
    3. 提取花括号之间的 JSON
    """
    cleaned = plan_json.strip()
    # 策略 1: 直接解析
    try:
        data = json.loads(cleaned)
        return TravelPlan.model_validate(data)
    except (json.JSONDecodeError, Exception):
        pass

    # 策略 2: 提取 ```json ... ``` 代码块
    import re

    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if m:
        try:
            data = json.loads(m.group(1).strip())
            return TravelPlan.model_validate(data)
        except (json.JSONDecodeError, Exception):
            pass

    # 策略 3: 提取最外层 {...}
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(cleaned[start : end + 1])
            return TravelPlan.model_validate(data)
        except (json.JSONDecodeError, Exception):
            pass

    return None


# ---------------------------------------------------------------------------
# Metric — 编译优化指标
# ---------------------------------------------------------------------------


def _format_evidence_for_reflection(evidence: EvidenceBundle) -> str:
    """将 EvidenceBundle 格式化为 reflection 可用的文本。"""
    parts: list[str] = []
    for r in evidence.results:
        parts.append(r.content.strip())
    return "\n\n".join(parts)


def build_reflection_metric(
    evidence: EvidenceBundle,
    tool_results: dict[str, object] | None = None,
) -> Any:
    """构建 DSPy 编译用的 metric 函数。

    返回的 metric 使用 :func:`deterministic_reflect` 评估预测计划质量，
    综合 evidence_coverage、confidence_score 和幻觉惩罚计算加权分数。

    参数
    ----------
    evidence:
        编译示例对应的 RAG 证据。
    tool_results:
        编译示例对应的确定性工具结果。
    """
    from travel_agent.agent.reflection import deterministic_reflect

    def metric(example: Any, pred: Any, trace: Any = None) -> float:
        """评估预测计划的质量分数，范围 [0.0, 1.0]。"""
        plan_json: str = ""
        if hasattr(pred, "plan"):
            plan_json = pred.plan
        elif isinstance(pred, dict):
            plan_json = pred.get("plan", "")

        plan = _parse_plan_json(plan_json)
        if plan is None:
            return 0.0

        report = deterministic_reflect(plan, evidence, tool_results)

        # 加权综合分数
        coverage_weight = 0.6
        confidence_weight = 0.3
        hallucination_weight = 0.1

        hallucination_penalty = min(
            1.0, len(report.hallucination_flags) * 0.15
        )
        score = (
            coverage_weight * report.evidence_coverage
            + confidence_weight * report.confidence_score
            + hallucination_weight * (1.0 - hallucination_penalty)
        )
        return max(0.0, min(1.0, score))

    return metric


# ---------------------------------------------------------------------------
# 编译入口
# ---------------------------------------------------------------------------


@dataclass
class DSPyCompileSettings:
    """DSPy 编译的运行时设置。"""

    model: str = "qwen3-max"
    api_key: str | None = None
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    optimizer: str = "BootstrapFewShot"
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 4
    save_path: Path = field(default_factory=lambda: Path("data/dspy/compiled_planner.json"))

    @classmethod
    def from_env(cls) -> DSPyCompileSettings:
        return cls(
            model=os.getenv("TRAVEL_AGENT_MODEL", "qwen3-max").strip() or "qwen3-max",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv(
                "TRAVEL_AGENT_DSPY_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            optimizer=os.getenv("TRAVEL_AGENT_DSPY_OPTIMIZER", "BootstrapFewShot"),
            max_bootstrapped_demos=int(
                os.getenv("TRAVEL_AGENT_DSPY_MAX_BOOTSTRAPPED", "4")
            ),
            max_labeled_demos=int(
                os.getenv("TRAVEL_AGENT_DSPY_MAX_LABELED", "4")
            ),
            save_path=Path(
                os.getenv(
                    "TRAVEL_AGENT_DSPY_SAVE_PATH",
                    "data/dspy/compiled_planner.json",
                )
            ),
        )


def _configure_dspy_lm(settings: DSPyCompileSettings) -> Any:
    """配置 DSPy 全局 LM。

    使用 openai/ 前缀 + dashscope base_url，
    因为 DashScope 兼容模式实现了 OpenAI chat completions 协议。
    """
    import dspy

    api_key = settings.api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "DSPy 编译需要 DASHSCOPE_API_KEY 或 OPENAI_API_KEY 环境变量"
        )

    lm = dspy.LM(
        model=f"openai/{settings.model}",
        api_key=api_key,
        api_base=settings.base_url,
        temperature=0,
    )
    dspy.configure(lm=lm)
    return lm


def compile_dspy_planner(
    train_examples: list[Any],
    settings: DSPyCompileSettings | None = None,
) -> TravelPlanningModule:
    """编译旅行规划 DSPy 模块。

    使用 BootstrapFewShot 优化器，以 reflection 指标为优化目标，
    自动发现最优 few-shot 示例组合。

    参数
    ----------
    train_examples:
        训练示例列表，每个示例需包含 request/evidence/tools/profile/plan 字段。
    settings:
        编译设置。为 None 时从环境变量读取。

    返回
    ----------
    编译后的 TravelPlanningModule（若训练集为空则返回未编译模块）。

    Raises
    ------
    ValueError:
        未配置 API 密钥时抛出。
    """
    active_settings = settings or DSPyCompileSettings.from_env()

    if not train_examples:
        module = TravelPlanningModule()
        module.ensure_module()
        return module

    _configure_dspy_lm(active_settings)

    module = TravelPlanningModule()
    module.ensure_module()

    import dspy

    # 为每个训练示例构建独立的 metric
    # DSPy BootstrapFewShot 逐条评估
    def _make_trainset_metric(example: Any) -> Any:
        # 从示例中重建证据
        evidence = None
        if hasattr(example, "evidence_bundle") and example.evidence_bundle is not None:
            evidence = example.evidence_bundle
        elif hasattr(example, "_evidence_bundle") and example._evidence_bundle is not None:
            evidence = example._evidence_bundle

        if evidence is None:
            raise ValueError(
                "训练示例必须包含 evidence_bundle 属性，"
                "请使用 dspy_data.build_training_example() 构建示例。"
            )

        tool_results = None
        if hasattr(example, "tool_results"):
            tool_results = example.tool_results

        return build_reflection_metric(evidence, tool_results)

    # 使用第一个示例构建 metric（DSPy 编译期间 metric 需全局一致，
    # 因此使用代表性示例的 evidence 构建 metric）
    if train_examples and hasattr(train_examples[0], "evidence_bundle"):
        evidence = train_examples[0].evidence_bundle
        tool_results = getattr(train_examples[0], "tool_results", None)
        metric_fn = build_reflection_metric(evidence, tool_results)
    else:
        raise ValueError("训练示例缺少 evidence_bundle 属性")

    optimizer_map: dict[str, type] = {}
    try:
        optimizer_map["BootstrapFewShot"] = dspy.BootstrapFewShot  # type: ignore[attr-defined]
    except AttributeError:
        pass
    try:
        optimizer_map["BootstrapFewShotWithRandomSearch"] = dspy.BootstrapFewShotWithRandomSearch  # type: ignore[attr-defined]
    except AttributeError:
        pass

    optimizer_cls = optimizer_map.get(
        active_settings.optimizer, dspy.BootstrapFewShot  # type: ignore[attr-defined]
    )

    optimizer = optimizer_cls(
        metric=metric_fn,
        max_bootstrapped_demos=active_settings.max_bootstrapped_demos,
        max_labeled_demos=active_settings.max_labeled_demos,
    )

    compiled = optimizer.compile(
        module._module,
        trainset=train_examples,
    )

    module._module = compiled

    if active_settings.save_path:
        active_settings.save_path.parent.mkdir(parents=True, exist_ok=True)
        module.save(active_settings.save_path)

    return module


def load_compiled_planner(path: str | Path) -> TravelPlanningModule | None:
    """从 JSON 文件加载已编译的 DSPy 模块。

    若文件不存在则返回 None（调用方应降级到未编译模块）。
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        module = TravelPlanningModule()
        module.load(path)
        return module
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Protocol 适配器 — DSPyOptimizedPlanner
# ---------------------------------------------------------------------------


@dataclass
class DSPyOptimizedPlanner:
    """DSPy 优化规划器 — 实现 TravelPlanner Protocol。

    封装编译/未编译的 DSPy 模块，在以下场景回退到 fallback：
    - DSPy 预测失败（JSON 解析错误、Pydantic 校验失败）
    - DSPy 模块未加载（无 API 密钥或编译产物不存在）

    用法::

        module = load_compiled_planner(Path("data/dspy/compiled_planner.json"))
        if module is None:
            module = TravelPlanningModule()
            module.ensure_module()
        planner = DSPyOptimizedPlanner(
            dspy_module=module,
            fallback=langchain_planner,
        )
        plan = planner.plan(request, evidence, tool_results=tools)
    """

    dspy_module: TravelPlanningModule
    fallback: TravelPlanner

    def plan(
        self,
        request: TravelRequest,
        evidence: EvidenceBundle,
        user_feedback: list[str] | None = None,
        tool_results: dict[str, object] | None = None,
        user_profile: UserProfile | None = None,
        active_skills: SkillSelection | None = None,
    ) -> TravelPlan:
        """通过 DSPy 模块生成计划，失败时回退到 fallback。"""
        try:
            self.dspy_module.ensure_module()

            req_str = _serialize_request(request)
            ev_str = _serialize_evidence(evidence)
            tools_str = _serialize_tool_results(tool_results)
            profile_str = _serialize_profile(user_profile)

            pred = self.dspy_module.forward(
                request=req_str,
                evidence=ev_str,
                tools=tools_str,
                profile=profile_str,
            )

            plan_json = getattr(pred, "plan", "")
            if not plan_json and isinstance(pred, dict):
                plan_json = pred.get("plan", "")

            plan = _parse_plan_json(plan_json)
            if plan is not None:
                plan.request = request
                return plan
        except Exception:
            pass

        return self.fallback.plan(
            request,
            evidence,
            user_feedback=user_feedback,
            tool_results=tool_results,
            user_profile=user_profile,
            active_skills=active_skills,
        )
