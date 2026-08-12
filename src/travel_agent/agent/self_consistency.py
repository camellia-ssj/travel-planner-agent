"""Self-Consistency 多路径投票 — 抑制 LLM 生成随机性。

通过生成 N 个候选计划（temperature > 0），使用 Reflection 审校
对每个候选评分，最终选出置信度最高的计划。所有类均实现
``TravelPlanner`` Protocol，可直接替换图中现有规划器，无需修改图结构。

使用方式::

    from travel_agent.agent.self_consistency import (
        SelfConsistencyPlanner,
        SelfConsistencySettings,
    )

    settings = SelfConsistencySettings(num_candidates=3)
    planner = SelfConsistencyPlanner(
        inner_planner=base_planner,
        reflection_service=reflection_service,
        settings=settings,
    )
    best_plan = planner.plan(request, evidence, tool_results=tools)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from travel_agent.agent.planner import TravelPlanner
from travel_agent.agent.reflection import ReflectionService, deterministic_reflect
from travel_agent.agent.schemas import TravelPlan, TravelRequest
from travel_agent.memory.models import UserProfile
from travel_agent.rag.models import EvidenceBundle
from travel_agent.skills.models import SkillSelection


@dataclass(frozen=True)
class SelfConsistencySettings:
    """多路径投票的运行时配置。

    通过环境变量 ``TRAVEL_AGENT_SC_*`` 控制所有参数。
    """

    num_candidates: int = 3
    """生成候选计划的数量（含）。设为 1 等价于禁用。"""

    selection_criterion: str = "confidence_score"
    """投票指标 — ``"confidence_score"`` 或 ``"evidence_coverage"``。"""

    enabled: bool = True
    """是否启用多路径投票。设为 False 时直接透传 inner_planner。"""

    @classmethod
    def from_env(cls) -> SelfConsistencySettings:
        return cls(
            num_candidates=int(
                os.getenv("TRAVEL_AGENT_SC_CANDIDATES", "3")
            ),
            selection_criterion=os.getenv(
                "TRAVEL_AGENT_SC_CRITERION", "confidence_score"
            )
            .strip()
            .lower(),
            enabled=os.getenv("TRAVEL_AGENT_SC_ENABLED", "true").strip().lower()
            != "false",
        )


@dataclass
class SelfConsistencyPlanner:
    """生成 N 个候选计划并通过 Reflection 评分选取最优。

    实现 ``TravelPlanner`` Protocol — 无需修改 LangGraph 图结构。
    内部使用 :func:`deterministic_reflect` 进行快速评分
    （不调用 LLM，避免评分环节引入随机性）。

    参数
    ----------
    inner_planner:
        底层规划器 — 应配置 temperature > 0 以保证候选多样性。
    reflection_service:
        用于评分的审校服务。默认使用确定性回退进行评分。
    settings:
        投票参数配置。
    """

    inner_planner: TravelPlanner
    reflection_service: ReflectionService | None = None
    settings: SelfConsistencySettings = field(
        default_factory=SelfConsistencySettings
    )

    def plan(
        self,
        request: TravelRequest,
        evidence: EvidenceBundle,
        user_feedback: list[str] | None = None,
        tool_results: dict[str, object] | None = None,
        user_profile: UserProfile | None = None,
        active_skills: SkillSelection | None = None,
    ) -> TravelPlan:
        """生成行程计划 — 若启用将执行 N 路生成 + 投票择优。

        当投票禁用或候选数 ≤ 1 时，直接透传 inner_planner。
        """
        if (
            not self.settings.enabled
            or self.settings.num_candidates <= 1
        ):
            return self.inner_planner.plan(
                request,
                evidence,
                user_feedback=user_feedback,
                tool_results=tool_results,
                user_profile=user_profile,
                active_skills=active_skills,
            )

        # ── 1. 生成 N 个候选 ──────────────────────────────────────
        candidates: list[TravelPlan] = []
        seen_hashes: set[str] = set()
        for _ in range(self.settings.num_candidates):
            try:
                plan = self.inner_planner.plan(
                    request,
                    evidence,
                    user_feedback=user_feedback,
                    tool_results=tool_results,
                    user_profile=user_profile,
                    active_skills=active_skills,
                )
            except Exception:
                continue

            # 去重：跳过完全相同的候选（temperature=0 或模型确定性行为）
            plan_hash = _plan_fingerprint(plan)
            if plan_hash in seen_hashes:
                continue
            seen_hashes.add(plan_hash)
            candidates.append(plan)

        if not candidates:
            # 所有尝试均失败 — 回退到单次调用
            return self.inner_planner.plan(
                request,
                evidence,
                user_feedback=user_feedback,
                tool_results=tool_results,
                user_profile=user_profile,
                active_skills=active_skills,
            )

        if len(candidates) == 1:
            # 所有候选均重复或仅生成一个唯一个体 — 仍然标记 SC
            unique_plan = candidates[0]
            if self.settings.num_candidates > 1:
                unique_plan = unique_plan.model_copy(
                    update={
                        "summary": (
                            f"[SC:{len(candidates)}/{self.settings.num_candidates}"
                            f" candidates, all duplicates] {unique_plan.summary}"
                        )
                    }
                )
            return unique_plan

        # ── 2. 对每个候选评分 ──────────────────────────────────────
        scored: list[tuple[TravelPlan, float]] = []
        for plan in candidates:
            score = self._score_candidate(
                plan, evidence, tool_results, active_skills
            )
            scored.append((plan, score))

        # ── 3. 按分数降序排列，选最高分 ────────────────────────────
        scored.sort(key=lambda item: item[1], reverse=True)
        best_plan, best_score = scored[0]

        # ── 4. 在摘要中标记投票信息 ────────────────────────────────
        return best_plan.model_copy(
            update={
                "summary": (
                    f"[SC:{len(candidates)}/{self.settings.num_candidates}"
                    f" candidates, best={self.settings.selection_criterion}"
                    f"={best_score:.3f}] {best_plan.summary}"
                )
            }
        )

    def _score_candidate(
        self,
        plan: TravelPlan,
        evidence: EvidenceBundle,
        tool_results: dict[str, object] | None,
        active_skills: SkillSelection | None,
    ) -> float:
        """对单个候选计划评分。

        优先使用注入的 ReflectionService，回退到确定性审校。
        """
        if self.reflection_service is not None:
            report = self.reflection_service.reflect(
                plan, evidence, tool_results, active_skills
            )
        else:
            report = deterministic_reflect(
                plan, evidence, tool_results, active_skills
            )

        criterion = self.settings.selection_criterion
        if criterion == "evidence_coverage":
            return report.evidence_coverage

        # 默认 confidence_score — 综合幻觉惩罚后的可信度
        return report.confidence_score


def _plan_fingerprint(plan: TravelPlan) -> str:
    """生成计划的轻量指纹用于去重。

    使用 destination + days + 所有活动文本拼接后哈希。
    """
    import hashlib

    parts = [plan.destination, str(plan.days)]
    for day_plan in plan.day_plans:
        parts.extend(day_plan.activities)
    return hashlib.sha256("|".join(parts).encode()).hexdigest()
