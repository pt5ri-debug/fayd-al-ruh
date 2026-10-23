from datetime import UTC, datetime, timedelta

from core.temporal_event import TemporalEvent
from core.temporal_reasoner import TemporalReasoner
from core.temporal_relation import TemporalRelation


def test_reasoner_understands_before_and_during():
    now = datetime.now(UTC)
    reasoner = TemporalReasoner()
    first = TemporalEvent("rain", now, now + timedelta(minutes=2))
    later = TemporalEvent("sun", now + timedelta(minutes=3))
    assert reasoner.relate(first, later) is TemporalRelation.BEFORE
