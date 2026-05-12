"""Centralized travel-domain knowledge shared across the RAG engine and Agent.

This module is the single source of truth for destination aliases, section
mappings, Chinese token dictionaries, holiday keywords, and related constants.
When adding a new destination, this is the only file that needs to be updated.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Destination aliases (Chinese / mixed-case → canonical English name)
# ---------------------------------------------------------------------------
DESTINATION_ALIASES: dict[str, str] = {
    "杭州": "Hangzhou",
    "hangzhou": "Hangzhou",
    "东京": "Tokyo",
    "東京": "Tokyo",
    "tokyo": "Tokyo",
    "苏州": "Suzhou",
    "suzhou": "Suzhou",
    "大理": "Dali",
    "dali": "Dali",
    "长沙": "Changsha",
    "changsha": "Changsha",
    "巴黎": "Paris",
    "paris": "Paris",
    "成都": "Chengdu",
    "chengdu": "Chengdu",
    "北京": "Beijing",
    "beijing": "Beijing",
}

DESTINATION_DISPLAY_NAMES: dict[str, str] = {
    "Beijing": "北京",
    "Changsha": "长沙",
    "Chengdu": "成都",
    "Dali": "大理",
    "Hangzhou": "杭州",
    "Paris": "巴黎",
    "Suzhou": "苏州",
    "Tokyo": "东京",
}

# ---------------------------------------------------------------------------
# Section aliases — map Chinese query terms → canonical section key
# ---------------------------------------------------------------------------
SECTION_QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "traffic": (
        "交通", "地铁", "公交", "机场", "高铁", "火车",
        "换乘", "停车", "打车", "怎么去", "如何去", "到达",
        "transit", "transport", "subway", "metro", "airport", "rail",
    ),
    "budget": ("预算", "费用", "花费", "价格", "门票", "多少钱", "budget", "cost", "price"),
    "lodging": ("住宿", "酒店", "住哪", "住在哪里", "民宿", "hotel", "lodging", "stay"),
    "dining": ("餐饮", "吃饭", "美食", "餐厅", "小吃", "吃什么", "dining", "food", "restaurant"),
    "crowd_risk": ("拥挤", "排队", "人多", "人流", "高峰", "crowd", "queue", "busy"),
    "weather_risk": (
        "天气", "下雨", "雨天", "高温", "寒冷", "台风",
        "weather", "rain", "hot", "cold",
    ),
    "itinerary": ("玩法", "怎么玩", "路线", "行程", "安排", "itinerary", "route", "plan"),
    "audience": ("适合人群", "亲子", "老人", "带孩子", "audience"),
    "alternatives": ("备选", "替代", "改去", "下雨去哪", "alternatives", "backup"),
}

# ---------------------------------------------------------------------------
# Chinese day numbers (used by both service.py and nodes.py regex parsing)
# ---------------------------------------------------------------------------
CHINESE_DAY_NUMBERS: dict[str, int] = {
    "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}

# ---------------------------------------------------------------------------
# Weekend / holiday detection keywords
# ---------------------------------------------------------------------------
WEEKEND_HOLIDAY_KEYWORDS: list[str] = [
    "周末", "双休", "周六", "周日", "礼拜六", "礼拜天", "星期六", "星期日",
    "周末游", "小长假", "长假", "黄金周", "国庆", "五一", "十一",
    "清明节", "劳动节", "端午节", "中秋节", "元旦", "春节", "端午",
    "清明", "中秋", "寒假", "暑假", "春节假期", "国定假日", "法定假日",
    "节假日", "假期", "节假", "休假", "放假",
    "weekend", "holiday", "vacation", "national day", "golden week",
    "spring festival", "christmas", "new year",
]

WEEKEND_HOLIDAY_PATTERN: re.Pattern[str] = re.compile(
    "|".join(re.escape(kw) for kw in WEEKEND_HOLIDAY_KEYWORDS),
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Built-in Chinese travel terms (BM25 tokenizer dictionary)
# ---------------------------------------------------------------------------
BUILTIN_CHINESE_TERMS: set[str] = {
    "八达岭", "白堤", "巴黎", "北京", "备选", "博物馆", "餐厅", "茶馆",
    "成都", "长城", "长沙", "出租车", "春季", "打车", "大理", "地铁",
    "东京", "断桥", "儿童", "费用", "高峰", "高铁", "公交", "故宫",
    "杭州", "酒店", "机场", "家庭", "交通", "景区", "老人", "雷峰塔",
    "灵隐寺", "龙井", "门票", "民宿", "亲子", "亲子游", "秋季", "人多",
    "人流", "三天", "商圈", "上海", "苏堤", "苏州", "天气", "铁路",
    "西湖", "下雨", "夏季", "行程", "预算", "雨天", "游客", "拥挤",
    "住宿", "周末",
}

# ---------------------------------------------------------------------------
# Keyword tokens for extractive answer ranking (service._keyword_tokens)
# ---------------------------------------------------------------------------
KEYWORD_ANSWER_TOKENS: tuple[str, ...] = (
    "周末", "拥挤", "亲子", "酒店", "灵隐寺", "东京", "杭州", "迪士尼",
)

# ---------------------------------------------------------------------------
# People-count implicit patterns (used by nodes._parse_people_count)
# ---------------------------------------------------------------------------
_PEOPLE_COUNT_PATTERN_DEFS: list[tuple[str, str]] = [
    (r"([一-鿿]+)(\d+)\s*个?\s*人", "m.group(2)"),
    (r"(\d+)\s*个?\s*人", "m.group(1)"),
    (r"我们\s*(\d+)\s*个", "m.group(1)"),
    (r"([\d]+)\s*(?:位|名|adults?|people|persons?)", "m.group(1)"),
    (r"一家\s*([\d一二两三])\s*口", "m.group(1)"),  # special handling needed
    (r"([一二两三四五六七八九十])\s*个?\s*人", "m.group(1)"),  # special handling needed
]

PEOPLE_IMPLICIT: dict[str, int] = {
    "一个人": 1, "独自": 1, "一个人去": 1, "单独": 1, "solo": 1,
    "我和父母": 3, "我和爸妈": 3, "带父母": 3, "带爸妈": 3,
    "我们俩": 2, "两个人": 2, "两人": 2, "二人": 2, "两口子": 2,
    "一家三口": 3, "一家四口": 4, "一家五口": 5,
    "三口之家": 3, "四口之家": 4,
    "亲子": 3, "一家": 3,
}
