from __future__ import annotations

from datetime import timedelta

from core.temporal_event import TemporalEvent
from core.temporal_pattern import TemporalPattern
from core.temporal_relation import TemporalRelation
from core.time_prediction import TimePrediction


class TemporalReasoner:
    def relate(self, first: TemporalEvent, second: TemporalEvent) -> TemporalRelation:
        if first.start == second.start and first.finish == second.finish:
            return TemporalRelation.SIMULTANEOUS
        if first.finish < second.start:
            return TemporalRelation.BEFORE
        if first.start > second.finish:
            return TemporalRelation.AFTER
        if first.start >= second.start and first.finish <= second.finish:
            return TemporalRelation.DURING
        if second.start >= first.start and second.finish <= first.finish:
            return TemporalRelation.CONTAINS
        return TemporalRelation.OVERLAPS

    def find_pattern(self, events: list[TemporalEvent], description: str) -> TemporalPattern | None:
        matches = sorted((event for event in events if event.description == description), key=lambda event: event.start)
        if not matches:
            return None
        durations = [event.duration_seconds for event in matches]
        average_duration = timedelta(seconds=sum(durations) / len(durations))
        intervals = [(later.start - earlier.start) for earlier, later in zip(matches, matches[1:])]
        average_interval = timedelta(seconds=sum(interval.total_seconds() for interval in intervals) / len(intervals)) if intervals else None
        values = [event.value for event in matches if event.value is not None]
        direction = "stable"
        if len(values) >= 2:
            direction = "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[0] else "stable"
        confidence = min(1.0, len(matches) / 5)
        return TemporalPattern(description, len(matches), average_interval, average_duration, direction, round(confidence, 6))

    def predict_next(self, events: list[TemporalEvent], description: str) -> TimePrediction | None:
        pattern = self.find_pattern(events, description)
        if pattern is None or pattern.average_interval is None:
            return None
        latest = max((event for event in events if event.description == description), key=lambda event: event.start)
        return TimePrediction(description, latest.start + pattern.average_interval, pattern.confidence, pattern.occurrences)
