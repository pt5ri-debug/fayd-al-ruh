from datetime import UTC, datetime, timedelta

from core.temporal_event import TemporalEvent


def test_event_exposes_duration():
    now = datetime.now(UTC)
    assert TemporalEvent("rain", now, now + timedelta(minutes=5)).duration_seconds == 300
