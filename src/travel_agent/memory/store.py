"""SQLite-backed persistent storage for user profiles and trip memories."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from travel_agent.memory.models import TripRecord, UserProfile


class MemoryStore:
    """Persistent long-term memory backed by SQLite.

    Stores trip records and aggregated user profiles so the Agent can
    build a "user portrait" across sessions. Thread-safe via WAL mode.

    Usage::

        store = MemoryStore(Path("data/user_memory.sqlite"))
        store.save_trip(TripRecord(...))
        profile = store.get_profile("user_123")
        print(profile.to_context_text())
    """

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Trip records
    # ------------------------------------------------------------------

    def save_trip(self, record: TripRecord) -> None:
        """Persist a trip record and update the user profile."""
        self._conn.execute(
            """INSERT INTO trip_memories
               (memory_id, user_id, thread_id, destination, days, audience,
                budget_preference, plan_summary, user_feedback, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.memory_id,
                record.user_id,
                record.thread_id,
                record.destination,
                record.days,
                json.dumps(record.audience, ensure_ascii=False),
                record.budget_preference,
                record.plan_summary,
                json.dumps(record.user_feedback, ensure_ascii=False),
                record.created_at,
            ),
        )
        self._conn.commit()
        self._rebuild_profile(record.user_id)

    def list_user_trips(self, user_id: str, limit: int = 20) -> list[TripRecord]:
        """Return recent trips for a user, newest first."""
        rows = self._conn.execute(
            """SELECT * FROM trip_memories
               WHERE user_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (user_id, limit),
        ).fetchall()
        return [_row_to_trip(row) for row in rows]

    # ------------------------------------------------------------------
    # User profile
    # ------------------------------------------------------------------

    def get_profile(self, user_id: str) -> UserProfile:
        """Return the user profile, creating a blank one if not found."""
        row = self._conn.execute(
            "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row is None:
            return UserProfile(user_id=user_id)
        return _row_to_profile(row)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _migrate(self) -> None:
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS trip_memories (
                memory_id   TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL,
                thread_id   TEXT NOT NULL DEFAULT '',
                destination TEXT NOT NULL DEFAULT '',
                days        INTEGER NOT NULL DEFAULT 1,
                audience    TEXT NOT NULL DEFAULT '[]',
                budget_preference TEXT NOT NULL DEFAULT 'standard',
                plan_summary TEXT NOT NULL DEFAULT '',
                user_feedback TEXT NOT NULL DEFAULT '[]',
                created_at  TEXT NOT NULL DEFAULT ''
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS user_profiles (
                user_id              TEXT PRIMARY KEY,
                preferred_destinations TEXT NOT NULL DEFAULT '[]',
                budget_preference    TEXT NOT NULL DEFAULT 'standard',
                audience_types       TEXT NOT NULL DEFAULT '[]',
                trip_length_avg      REAL NOT NULL DEFAULT 0.0,
                total_trips          INTEGER NOT NULL DEFAULT 0,
                last_destination     TEXT NOT NULL DEFAULT '',
                preferences_summary  TEXT NOT NULL DEFAULT '',
                created_at           TEXT NOT NULL DEFAULT '',
                updated_at           TEXT NOT NULL DEFAULT ''
            )"""
        )
        self._conn.commit()

    def _rebuild_profile(self, user_id: str) -> None:
        """Aggregate all trip records into an up-to-date user profile."""
        rows = self._conn.execute(
            """SELECT destination, days, audience, budget_preference
               FROM trip_memories
               WHERE user_id = ?
               ORDER BY created_at DESC""",
            (user_id,),
        ).fetchall()

        if not rows:
            return

        total = len(rows)
        destinations: list[str] = []
        dest_count: dict[str, int] = {}
        audience_all: list[str] = []
        budget_votes: dict[str, int] = {}
        total_days = 0

        for row in rows:
            dest = row["destination"]
            if dest:
                destinations.append(dest)
                dest_count[dest] = dest_count.get(dest, 0) + 1
            total_days += row["days"]
            audience_all.extend(json.loads(row["audience"]))
            bp = row["budget_preference"]
            budget_votes[bp] = budget_votes.get(bp, 0) + 1

        # Top destinations by frequency
        ranked_dests = sorted(dest_count.items(), key=lambda x: (-x[1], x[0]))
        preferred = [d for d, _ in ranked_dests[:5]]

        # Most common budget
        top_budget = max(budget_votes, key=budget_votes.get) if budget_votes else "standard"

        # Top audience types
        aud_count: dict[str, int] = {}
        for a in audience_all:
            aud_count[a] = aud_count.get(a, 0) + 1
        top_audiences = sorted(aud_count, key=aud_count.get, reverse=True)[:3]  # type: ignore[arg-type]

        now = datetime.now(UTC).isoformat()
        last_dest = destinations[0] if destinations else ""

        # Build a simple Chinese summary for the planner
        summary_parts: list[str] = []
        if preferred:
            summary_parts.append(f"偏好目的地: {', '.join(preferred)}")
        if top_audiences:
            audience_labels: dict[str, str] = {
                "family_with_children": "亲子",
                "elderly": "带老人",
                "couple": "情侣",
                "friends": "朋友结伴",
                "solo": "独自出行",
                "general": "通用",
            }
            aud_str = ", ".join(audience_labels.get(a, a) for a in top_audiences)
            summary_parts.append(f"出行类型: {aud_str}")
        budget_labels = {"economy": "经济实惠型", "premium": "高端舒适型", "standard": "中等标准型"}
        summary_parts.append(f"消费偏好: {budget_labels.get(top_budget, top_budget)}")
        summary_parts.append(f"平均行程: {total_days / total:.1f} 天")

        profile = UserProfile(
            user_id=user_id,
            preferred_destinations=preferred,
            budget_preference=top_budget,
            audience_types=top_audiences,
            trip_length_avg=round(total_days / total, 1),
            total_trips=total,
            last_destination=last_dest,
            preferences_summary="; ".join(summary_parts),
            updated_at=now,
        )

        # Upsert
        existing = self._conn.execute(
            "SELECT user_id FROM user_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        if existing:
            self._conn.execute(
                """UPDATE user_profiles
                   SET preferred_destinations = ?, budget_preference = ?,
                       audience_types = ?, trip_length_avg = ?, total_trips = ?,
                       last_destination = ?, preferences_summary = ?,
                       updated_at = ?
                   WHERE user_id = ?""",
                (
                    json.dumps(profile.preferred_destinations, ensure_ascii=False),
                    profile.budget_preference,
                    json.dumps(profile.audience_types, ensure_ascii=False),
                    profile.trip_length_avg,
                    profile.total_trips,
                    profile.last_destination,
                    profile.preferences_summary,
                    profile.updated_at,
                    user_id,
                ),
            )
        else:
            self._conn.execute(
                """INSERT INTO user_profiles
                   (user_id, preferred_destinations, budget_preference,
                    audience_types, trip_length_avg, total_trips,
                    last_destination, preferences_summary, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id,
                    json.dumps(profile.preferred_destinations, ensure_ascii=False),
                    profile.budget_preference,
                    json.dumps(profile.audience_types, ensure_ascii=False),
                    profile.trip_length_avg,
                    profile.total_trips,
                    profile.last_destination,
                    profile.preferences_summary,
                    profile.created_at,
                    profile.updated_at,
                ),
            )
        self._conn.commit()


# ------------------------------------------------------------------
# Row deserialization helpers
# ------------------------------------------------------------------


def _row_to_trip(row: sqlite3.Row) -> TripRecord:
    return TripRecord(
        memory_id=row["memory_id"],
        user_id=row["user_id"],
        thread_id=row["thread_id"],
        destination=row["destination"],
        days=row["days"],
        audience=json.loads(row["audience"]),
        budget_preference=row["budget_preference"],
        plan_summary=row["plan_summary"],
        user_feedback=json.loads(row["user_feedback"]),
        created_at=row["created_at"],
    )


def _row_to_profile(row: sqlite3.Row) -> UserProfile:
    return UserProfile(
        user_id=row["user_id"],
        preferred_destinations=json.loads(row["preferred_destinations"]),
        budget_preference=row["budget_preference"],
        audience_types=json.loads(row["audience_types"]),
        trip_length_avg=row["trip_length_avg"],
        total_trips=row["total_trips"],
        last_destination=row["last_destination"],
        preferences_summary=row["preferences_summary"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
