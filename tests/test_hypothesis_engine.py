from core.hypothesis_engine import HypothesisEngine
from core.pattern_detector import Pattern


def test_pattern_becomes_hypothesis():

    pattern = Pattern(
        key="the ball moved.",
        occurrences=3,
        confidence=0.9,
        memory_ids=[
            "OBS-1",
            "OBS-2",
            "OBS-3",
        ],
    )

    engine = HypothesisEngine()

    hypothesis = engine.create(pattern)

    assert hypothesis.source_pattern == \
        "the ball moved."

    assert hypothesis.confidence == 0.8

    assert "the ball moved." in \
        hypothesis.statement