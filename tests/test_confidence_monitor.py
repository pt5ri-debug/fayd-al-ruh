from core.belief import Belief
from core.confidence_monitor import ConfidenceMonitor


def test_monitor_reports_uncertain_belief():
    record = ConfidenceMonitor().inspect([Belief("rain", .73)])[0]
    assert record.uncertainty.uncertainty == .27
