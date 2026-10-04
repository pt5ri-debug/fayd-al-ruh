from datetime import datetime, timedelta, timezone
from core.temporal_event import TemporalEvent


def test_event_exposes_duration():
    now = datetime.now(timezone.utc)
    assert TemporalEvent("rain", now, now + timedelta(minutes=5)).duration_seconds == 300
