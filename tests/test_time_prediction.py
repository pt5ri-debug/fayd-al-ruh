from datetime import datetime, timedelta, timezone
from core.temporal_event import TemporalEvent
from core.temporal_reasoner import TemporalReasoner


def test_prediction_uses_repeated_interval():
    now = datetime.now(timezone.utc)
    events = [TemporalEvent("rain", now), TemporalEvent("rain", now + timedelta(hours=2))]
    prediction = TemporalReasoner().predict_next(events, "rain")
    assert prediction.predicted_time == now + timedelta(hours=4)
