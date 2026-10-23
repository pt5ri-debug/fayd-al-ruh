from datetime import UTC, datetime, timedelta

from core.temporal_event import TemporalEvent
from core.timeline import Timeline


def test_timeline_orders_events():
    now = datetime.now(UTC)
    timeline = Timeline()
    timeline.add(TemporalEvent("later", now + timedelta(hours=1)))
    timeline.add(TemporalEvent("now", now))
    assert [event.description for event in timeline.ordered()] == ["now", "later"]
