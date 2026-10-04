from datetime import datetime, timedelta, timezone
from core.temporal_event import TemporalEvent
from core.temporal_reasoner import TemporalReasoner


def test_temporal_contract_predicts_repetition():
    now = datetime.now(timezone.utc)
    assert TemporalReasoner().predict_next([TemporalEvent("rain", now), TemporalEvent("rain", now + timedelta(hours=1))], "rain") is not None
