"""Memory 长期记忆模块 — 旅行智能体的用户画像与历史行程。

此模块通过跨会话持久化行程记录并将其聚合为偏好画像，
为智能体提供"用户画像"能力。
"""

from travel_agent.memory.models import TripRecord, UserProfile
from travel_agent.memory.store import MemoryStore

__all__ = [
    "MemoryStore",
    "TripRecord",
    "UserProfile",
]
