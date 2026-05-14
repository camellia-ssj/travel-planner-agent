"""LangGraph旅行智能体的结构化模式定义。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TravelRequest(BaseModel):
    """规则解析后的用户出行意图。"""

    raw_query: str
    destination: str = ""
    days: int = Field(default=1, ge=1)
    audience: list[str] = Field(default_factory=list)
    budget_preference: str = "standard"


class DayPlan(BaseModel):
    """生成行程中的某一天。"""

    day: int
    title: str
    activities: list[str]
    evidence_sources: list[str] = Field(default_factory=list)


class BudgetItem(BaseModel):
    """某一旅行消费类别的预算指导。"""

    category: str
    preference: str
    note: str


class RiskNotice(BaseModel):
    """基于检索证据生成的规则化风险提醒。"""

    risk_type: str
    message: str
    severity: str = "medium"


class BudgetEstimate(BaseModel):
    """由预算工具（budget_tool）生成的确定性各分类预算明细。"""

    accommodation: float = Field(description="预估住宿总费用（人民币）")
    dining: float = Field(description="预估餐饮总费用（人民币）")
    transport: float = Field(description="预估本地交通总费用（人民币）")
    tickets: float = Field(description="预估景点门票总费用（人民币）")
    total: float = Field(description="所有分类费用总和")
    daily_average: float = Field(description="日均支出")
    budget_level: str = Field(description="经济型 / 标准型 / 高端型")
    notes: list[str] = Field(default_factory=list)


class POICrowdRisk(BaseModel):
    """单个景点的拥挤风险评估。"""

    poi_name: str
    risk_level: str = Field(description="低 / 中 / 高")
    peak_times: str = Field(description="拥挤高峰时段")
    source_evidence: str = Field(description="来自RAG证据的摘要片段")


class CrowdRiskAssessment(BaseModel):
    """行程的确定性拥挤风险评估。"""

    destination: str
    is_weekend_holiday: bool
    poi_risks: list[POICrowdRisk] = Field(default_factory=list)
    overall_risk: str = Field(description="低 / 中 / 高")
    advice: str = Field(description="一行可操作的建议")


class AlternativeSuggestion(BaseModel):
    """单个备选推荐。"""

    original_scenario: str = Field(description="触发此备选方案的情境")
    suggested_alternative: str
    reason: str


class AlternativePlan(BaseModel):
    """基于风险和天气的确定性备选方案建议。"""

    destination: str
    alternatives: list[AlternativeSuggestion] = Field(default_factory=list)
    weather_note: str = Field(default="", description="基于天气的推荐摘要")


class TravelPlan(BaseModel):
    """旅行智能体图生成的结构化输出。"""

    request: TravelRequest
    destination: str
    days: int
    summary: str
    day_plans: list[DayPlan]
    budget_items: list[BudgetItem]
    risk_notices: list[RiskNotice]
    alternatives: list[str] = Field(default_factory=list)
    evidence_sources: list[str] = Field(default_factory=list)
    evidence_trace_id: str = ""
    fallback_used: bool = Field(
        default=False,
        description="当LLM规划器回退到基于规则的规划器时为True",
    )


class HallucinationFlag(BaseModel):
    """生成计划中被标记为无依据或可疑的内容声明。"""

    location: str = Field(
        description="声明在计划中出现的位置，例如 'day_plans[0].activities[1]'"
    )
    claim: str = Field(description="被标记为可疑的声明文本")
    issue: str = Field(description="此声明被标记的原因")
    severity: str = Field(default="medium", description="高 / 中 / 低")


class ReflectionReport(BaseModel):
    """生成后对TravelPlan进行的基于RAG证据的事实性审查。"""

    hallucination_flags: list[HallucinationFlag] = Field(default_factory=list)
    evidence_coverage: float = Field(
        default=0.0, ge=0.0, le=1.0, description="有证据支撑的声明比例"
    )
    confidence_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="对计划事实性的整体置信度"
    )
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    passed: bool = Field(default=True, description="计划是否通过了审查校验")
    checked_claims: int = Field(default=0, description="已检查的声明总数")
    grounded_claims: int = Field(default=0, description="有证据支撑的声明数量")
