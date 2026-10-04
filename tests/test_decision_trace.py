from core.decision_trace import DecisionTrace


def test_empty_decision_trace_explains_no_choice():
    assert DecisionTrace((), None, "none").selected is None
