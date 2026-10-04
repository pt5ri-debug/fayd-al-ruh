from datetime import datetime, timedelta, timezone
from core.temporal_event import TemporalEvent
from core.timeline import Timeline


def test_timeline_orders_events():
    now = datetime.now(timezone.utc); timeline = Timeline()
    timeline.add(TemporalEvent("later", now + timedelta(hours=1))); timeline.add(TemporalEvent("now", now))
    assert [event.description for event in timeline.ordered()] == ["now", "later"]
