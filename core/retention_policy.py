from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from math import exp

from core.memory import CognitiveMemory, MemoryItem


@dataclass(frozen=True)
class RetentionPolicy:
    """Decides which long-term memory items survive, age out, or are protected.

    Items are evicted only when they are both old enough (older than
    ``max_age_days``) and weak enough (retained importance below
    ``minimum_importance``). Frequently recalled items are protected because
    ``MemoryDecay`` slows their decay by ``1 + access_count``.
    """

    max_age_days: float = 30.0
    minimum_importance: float = 0.1
    decay_rate: float = 0.01
    protect_item_types: frozenset[str] = field(default_factory=frozenset)
    max_items: int | None = None

    def __post_init__(self) -> None:
        if self.max_age_days < 0:
            raise ValueError("max_age_days cannot be negative.")
        if not 0.0 <= self.minimum_importance <= 1.0:
            raise ValueError("minimum_importance must be between 0 and 1.")
        if self.decay_rate < 0:
            raise ValueError("decay_rate cannot be negative.")
        if self.max_items is not None and self.max_items < 1:
            raise ValueError("max_items must be at least 1 when provided.")

    def retained_importance(
        self,
        item: MemoryItem,
        now: datetime | None = None,
    ) -> float:
        current = now or datetime.now(timezone.utc)
        age_hours = max(0.0, (current - item.created_at).total_seconds() / 3600)
        protection = 1 + item.access_count
        return round(item.importance * exp(-(self.decay_rate * age_hours) / protection), 6)

    def expired(self, item: MemoryItem, now: datetime | None = None) -> bool:
        if item.item_type in self.protect_item_types:
            return False
        current = now or datetime.now(timezone.utc)
        age = current - item.created_at
        if age < timedelta(days=self.max_age_days):
            return False
        return self.retained_importance(item, current) < self.minimum_importance

    def select_evictions(
        self,
        memory: CognitiveMemory,
        now: datetime | None = None,
    ) -> list[MemoryItem]:
        """Return the items a prune should remove, deterministically ordered.

        First the expired (old and weak) items, then, if ``max_items`` is set,
        the weakest survivors needed to get back under capacity.
        """
        current = now or datetime.now(timezone.utc)

        def weakest_first(item: MemoryItem) -> tuple[float, datetime, str]:
            return (
                self.retained_importance(item, current),
                item.created_at,
                item.item_id,
            )

        evictions = [item for item in memory._items.values() if self.expired(item, current)]
        evictions.sort(key=weakest_first)

        if self.max_items is not None:
            remaining = [
                item for item in memory._items.values() if item not in evictions
            ]
            remaining.sort(key=weakest_first)
            if len(remaining) > self.max_items:
                evictions.extend(remaining[: len(remaining) - self.max_items])

        return evictions


@dataclass
class RetentionReport:
    """Outcome of one prune: what was removed and what remains."""

    removed: list[str] = field(default_factory=list)
    kept: int = 0
    policy: RetentionPolicy | None = None

    @property
    def removed_count(self) -> int:
        return len(self.removed)


def prune(memory: CognitiveMemory, policy: RetentionPolicy, now: datetime | None = None) -> RetentionReport:
    """Apply ``policy`` to ``memory`` and return what was removed."""
    evictions = policy.select_evictions(memory, now)
    removed: list[str] = []
    for item in evictions:
        if item.item_id in memory._items:
            memory.remove(item.item_id)
            removed.append(item.item_id)
    return RetentionReport(removed=removed, kept=memory.size, policy=policy)
