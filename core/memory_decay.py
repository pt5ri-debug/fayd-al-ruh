from __future__ import annotations

from datetime import UTC, datetime
from math import exp

from core.episodic_memory import Episode


class MemoryDecay:
    """Forgetting curve that preserves frequently recalled episodes."""
    def retention(self, episode: Episode, rate: float = .01, now: datetime | None = None) -> float:
        if rate < 0:
            raise ValueError("Decay rate cannot be negative.")
        current = now or datetime.now(UTC)
        age_hours = max(0.0, (current - episode.created_at).total_seconds() / 3600)
        protection = 1 + episode.access_count
        return round(episode.importance * exp(-(rate * age_hours) / protection), 6)

    def apply(self, episode: Episode, rate: float = .01, now: datetime | None = None) -> float:
        episode.importance = self.retention(episode, rate, now)
        return episode.importance
