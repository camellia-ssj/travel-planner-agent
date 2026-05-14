"""对话式旅行规划 Agent 的确定性槽位填充。

复用 ``travel_agent.agent.nodes`` 中的解析工具函数
进行目的地、天数、预算和出行人员的提取。
"""

from __future__ import annotations

from travel_agent.agent.nodes import (
    _parse_audience,
    _parse_budget_preference,
    _parse_days,
    _parse_destination,
    _has_change_intent,
    _days_explicitly_mentioned,
)
from travel_agent.conversation.state import ConversationState

# 知识库中已知的目的地
_KNOWN_DESTINATIONS = [
    "杭州", "北京", "成都", "长沙", "大理", "苏州",
    "东京", "巴黎", "上海", "西安", "重庆", "厦门",
    "三亚", "丽江", "桂林", "张家界", "黄山", "九寨沟",
    "拉萨", "哈尔滨", "青岛", "昆明", "南京", "武汉",
]

# 用户说"随便"/"推荐"时给出的推荐目的地
_RECOMMENDED_DESTINATIONS = [
    "杭州 — 西湖山水，悠闲惬意，适合全家出行",
    "成都 — 美食之都，慢生活，熊猫基地亲子首选",
    "大理 — 苍山洱海，风花雪月，情侣度假胜地",
    "北京 — 故宫长城，历史文化，老少皆宜",
    "长沙 — 烟火气十足，美食遍地，朋友结伴好去处",
]


def extract_slots(text: str, state: ConversationState) -> dict[str, object]:
    """使用现有正则解析器从用户消息中提取旅行槽位。

    返回包含新提取值的字典。未在 text 中提及的槽位不包含在返回结果中。
    """
    updates: dict[str, object] = {}

    destination = _parse_destination(text)
    if destination:
        updates["clarified_destination"] = destination

    if _days_explicitly_mentioned(text):
        days = _parse_days(text)
        updates["clarified_days"] = max(1, int(days))

    audience = _parse_audience(text)
    if audience and audience != ["general"]:
        updates["clarified_audience"] = audience

    budget = _parse_budget_preference(text)
    updates["clarified_budget"] = budget

    return updates


def _slot_extraction_function(
    text: str,
) -> dict[str, object]:
    """纯函数版本，用于测试——不依赖状态。"""
    result: dict[str, object] = {}

    destination = _parse_destination(text)
    if destination:
        result["clarified_destination"] = destination

    if _days_explicitly_mentioned(text):
        days = _parse_days(text)
        result["clarified_days"] = max(1, int(days))

    audience = _parse_audience(text)
    if audience and audience != ["general"]:
        result["clarified_audience"] = audience

    budget = _parse_budget_preference(text)
    result["clarified_budget"] = budget

    return result


def check_slots_complete(state: ConversationState) -> tuple[bool, list[str]]:
    """检查槽位是否足够触发规划。

    返回 (是否完整, 缺失槽位列表)。
    目的地是最低门槛。超过2轮澄清后，预算和出行人员使用默认值。
    """
    missing: list[str] = []

    destination = state.get("clarified_destination", "").strip()
    if not destination:
        missing.append("destination")

    days = state.get("clarified_days")
    if days is None or days < 1:
        missing.append("days")

    turn_count = state.get("clarification_turn_count", 0)

    # 超过2轮后不再阻塞预算/出行人员——使用默认值
    if turn_count < 2:
        budget = state.get("clarified_budget", "").strip()
        if not budget:
            missing.append("budget")

        audience = state.get("clarified_audience")
        if not audience:
            missing.append("audience")

    if destination:
        return len(missing) <= 1, missing

    return False, missing


def apply_defaults(state: ConversationState) -> dict[str, object]:
    """为缺失的槽位填充合理的默认值。"""
    defaults: dict[str, object] = {}

    if not state.get("clarified_days"):
        defaults["clarified_days"] = 3

    if not state.get("clarified_budget"):
        defaults["clarified_budget"] = "standard"

    if not state.get("clarified_audience"):
        defaults["clarified_audience"] = ["general"]

    return defaults


def get_recommendation_text() -> str:
    """为说'随便'的用户生成目的地推荐文案。"""
    lines = ["为您推荐几个热门目的地："]
    for i, dest in enumerate(_RECOMMENDED_DESTINATIONS, 1):
        lines.append(f"  {i}. {dest}")
    lines.append("您对哪个感兴趣呢？")
    return "\n".join(lines)


def is_vague_request(text: str) -> bool:
    """检查用户是否表达了模糊/无偏好需求。"""
    vague_keywords = {"随便", "推荐", "你来定", "都行", "都可以", "不知道", "没想好",
                      "建议", "有什么推荐", "哪里好玩", "去哪"}
    normalized = text.lower().strip()
    return any(kw in normalized for kw in vague_keywords)
