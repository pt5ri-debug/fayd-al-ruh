from __future__ import annotations

from core.temporal_event import TemporalEvent


class Timeline:
    """Chronological event store that accepts events with overlapping intervals."""
    def __init__(self) -> None:
        self.events: dict[str, TemporalEvent] = {}

    def add(self, event: TemporalEvent) -> None:
        if event.id in self.events:
            raise ValueError(f"Temporal event '{event.id}' already exists.")
        self.events[event.id] = event

    def ordered(self) -> list[TemporalEvent]:
        return sorted(self.events.values(), key=lambda event: (event.start, event.finish, event.id))

    def between(self, start, end) -> list[TemporalEvent]:
        return [event for event in self.ordered() if event.start >= start and event.finish <= end]
