from core.cognitive_trace import CognitiveTrace


def test_create_trace():

    trace = CognitiveTrace(
        observation_id="OBS-001",
    )

    assert trace.observation_id == "OBS-001"
    assert trace.pattern_keys == []
    assert trace.hypothesis_ids == []
    assert trace.prediction_ids == []
    assert trace.prediction_error_ids == []
    assert trace.belief_id is None


def test_trace_records_causal_chain():

    trace = CognitiveTrace(
        observation_id="OBS-001",
    )

    trace.add_pattern(
        "the ball moved."
    )

    trace.add_hypothesis(
        "HYP-001"
    )

    trace.add_prediction(
        "PRD-001"
    )

    trace.add_prediction_error(
        "ERR-001"
    )

    trace.set_belief(
        "BEL-001"
    )

    assert trace.pattern_keys == [
        "the ball moved."
    ]

    assert trace.hypothesis_ids == [
        "HYP-001"
    ]

    assert trace.prediction_ids == [
        "PRD-001"
    ]

    assert trace.prediction_error_ids == [
        "ERR-001"
    ]

    assert trace.belief_id == "BEL-001"


def test_trace_does_not_duplicate_ids():

    trace = CognitiveTrace(
        observation_id="OBS-001",
    )

    trace.add_prediction("PRD-001")
    trace.add_prediction("PRD-001")

    trace.add_pattern("pattern")
    trace.add_pattern("pattern")

    assert trace.prediction_ids == [
        "PRD-001"
    ]

    assert trace.pattern_keys == [
        "pattern"
    ]