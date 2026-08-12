"""DSPy 训练数据构建与加载辅助模块。

提供训练示例的构建和 JSONL 序列化/反序列化工具。
示例格式::

    {
        "request": "<serialized TravelRequest JSON>",
        "evidence": "<formatted evidence text>",
        "tools": "<serialized tool results JSON>",
        "profile": "<user profile context string>",
        "plan": "<gold-standard TravelPlan JSON>"
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from travel_agent.agent.planner import TravelPlanner
from travel_agent.agent.schemas import TravelPlan, TravelRequest
from travel_agent.memory.models import UserProfile
from travel_agent.rag.models import EvidenceBundle
from travel_agent.skills.models import SkillSelection


def build_training_example(
    request: TravelRequest,
    evidence: EvidenceBundle,
    plan: TravelPlan,
    tool_results: dict[str, object] | None = None,
    user_profile: UserProfile | None = None,
) -> Any:
    """构建单个 DSPy 训练示例。

    示例包含 DSPy 模块所需的四个输入字段和一个 gold-standard 输出字段。
    额外通过私有属性附加 evidence_bundle 和 tool_results，
    供 metric 函数在编译期间使用。

    参数
    ----------
    request:
        结构化旅行请求。
    evidence:
        RAG 检索证据。
    plan:
        Gold-standard 行程计划（人工策划或确定性规划器输出）。
    tool_results:
        确定性工具结果。
    user_profile:
        用户画像。

    返回
    ----------
    可直接传入 :func:`compile_dspy_planner` 的 dspy.Example。
    """
    import dspy

    from travel_agent.agent.dspy_planner import (
        _serialize_evidence,
        _serialize_profile,
        _serialize_request,
        _serialize_tool_results,
    )

    example = dspy.Example(
        request=_serialize_request(request),
        evidence=_serialize_evidence(evidence),
        tools=_serialize_tool_results(tool_results),
        profile=_serialize_profile(user_profile),
        plan=plan.model_dump_json(exclude_none=True),
    )
    # 附加编译所需的原始对象（DSPy 不支持直接附加非基本类型，
    # 使用私有属性存储）
    example.evidence_bundle = evidence  # type: ignore[attr-defined]
    example.tool_results = tool_results  # type: ignore[attr-defined]
    return example


def build_training_examples_from_planner(
    requests: list[TravelRequest],
    evidence_map: dict[str, EvidenceBundle],
    planner: TravelPlanner,
    tool_results_map: dict[str, dict[str, object]] | None = None,
    user_profile: UserProfile | None = None,
) -> list[Any]:
    """使用规划器为多个请求批量生成训练示例。

    对每个请求，调用 *planner* 生成计划，并将结果作为 gold-standard。
    适用于使用确定性规划器（RuleBasedTravelPlanner）批量构建
    种子训练数据。

    参数
    ----------
    requests:
        旅行请求列表。
    evidence_map:
        按请求 raw_query 索引的证据映射。
    planner:
        用于生成 gold-standard 计划的规划器。
    tool_results_map:
        按请求 raw_query 索引的工具结果映射（可选）。
    user_profile:
        所有请求共享的用户画像（可选）。

    返回
    ----------
    dspy.Example 列表。
    """
    examples: list[Any] = []
    for req in requests:
        evidence = evidence_map.get(req.raw_query)
        if evidence is None:
            continue
        tool_results = (
            tool_results_map.get(req.raw_query) if tool_results_map else None
        )
        plan = planner.plan(
            req,
            evidence,
            tool_results=tool_results,
            user_profile=user_profile,
        )
        examples.append(
            build_training_example(req, evidence, plan, tool_results, user_profile)
        )
    return examples


def save_training_examples(examples: list[Any], path: str | Path) -> None:
    """将训练示例持久化为 JSONL 文件。

    每行为一个 JSON 对象，包含 request/evidence/tools/profile/plan 字段。
    不含 Python 私有属性（evidence_bundle、tool_results）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            record: dict[str, object] = {}
            for field in ("request", "evidence", "tools", "profile", "plan"):
                value = getattr(example, field, None)
                if value is not None:
                    record[field] = value
            if record:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_training_examples(
    path: str | Path,
    evidence_map: dict[str, EvidenceBundle] | None = None,
    tool_results_map: dict[str, dict[str, object]] | None = None,
) -> list[Any]:
    """从 JSONL 文件加载训练示例。

    若提供 *evidence_map* 和 *tool_results_map*，
    按请求文本匹配恢复原始对象引用，使示例可供 DSPy 编译使用。

    参数
    ----------
    path:
        JSONL 文件路径。
    evidence_map:
        按请求 raw_query 索引的证据映射（可选）。
    tool_results_map:
        按请求 raw_query 索引的工具结果映射（可选）。

    返回
    ----------
    dspy.Example 列表。
    """
    import dspy

    examples: list[Any] = []
    path = Path(path)
    if not path.exists():
        return examples

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"无法解析 {path}:{line_number}: {exc}"
            ) from exc

        example = dspy.Example(
            request=record.get("request", ""),
            evidence=record.get("evidence", ""),
            tools=record.get("tools", "{}"),
            profile=record.get("profile", "无用户历史记录。"),
            plan=record.get("plan", "{}"),
        )

        # 尝试恢复 evidence_bundle 引用
        if evidence_map:
            request_str = record.get("request", "")
            try:
                req_data = json.loads(request_str)
                key = req_data.get("raw_query", "")
            except (json.JSONDecodeError, TypeError):
                key = request_str
            if key and key in evidence_map:
                example.evidence_bundle = evidence_map[key]  # type: ignore[attr-defined]
                if tool_results_map and key in tool_results_map:
                    example.tool_results = tool_results_map[key]  # type: ignore[attr-defined]

        examples.append(example)

    return examples
