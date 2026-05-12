"""Memory long-term memory module — user profiles and trip history for the Travel Agent.

This module gives the Agent a "用户画像" (user portrait) by persisting trip
records across sessions and aggregating them into preference profiles.
"""

from travel_agent.memory.models import TripRecord, UserProfile
from travel_agent.memory.store import MemoryStore

__all__ = [
    "MemoryStore",
    "TripRecord",
    "UserProfile",
]
