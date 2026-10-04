from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class Episode:
    id: str
    content: Any
    importance: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0


class EpisodicMemory:
    """Stores time-stamped experiences rather than abstract facts."""
    def __init__(self) -> None:
        self.episodes: dict[str, Episode] = {}

    def store(self, episode_id: str, content: Any, importance: float = .5) -> Episode:
        if episode_id in self.episodes:
            raise ValueError(f"Episode '{episode_id}' already exists.")
        if not 0.0 <= importance <= 1.0:
            raise ValueError("Episode importance must be between 0 and 1.")
        episode = Episode(episode_id, content, importance)
        self.episodes[episode_id] = episode
        return episode

    def recall(self, episode_id: str) -> Episode | None:
        episode = self.episodes.get(episode_id)
        if episode is not None:
            episode.access_count += 1
        return episode
