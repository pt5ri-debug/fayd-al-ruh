from core.error_monitor import ErrorMonitor
from core.prediction_error import PredictionError


def test_error_monitor_reports_highest_prediction_error():
    result = ErrorMonitor().inspect([PredictionError("a", .2, 1., .8), PredictionError("b", .8, 1., .2)])
    assert result.highest_error_prediction_id == "a"
