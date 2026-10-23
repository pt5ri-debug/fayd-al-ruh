from datetime import UTC, datetime, timedelta

from core.temporal_event import TemporalEvent
from core.temporal_reasoner import TemporalReasoner


def test_pattern_finds_repetition_and_direction():
    now = datetime.now(UTC)
    events = [TemporalEvent("rain", now, value=1), TemporalEvent("rain", now + timedelta(days=1), value=2)]
    pattern = TemporalReasoner().find_pattern(events, "rain")
    assert pattern.occurrences == 2 and pattern.direction == "increasing"
