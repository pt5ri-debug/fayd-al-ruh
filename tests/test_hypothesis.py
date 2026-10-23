import pytest
from core.hypothesis import Hypothesis


def test_create_hypothesis():

    hypothesis = Hypothesis(
        statement="The ball moves repeatedly.",
        confidence=0.7,
        source_pattern="the ball moved.",
    )

    assert hypothesis.statement == \
        "The ball moves repeatedly."

    assert hypothesis.confidence == 0.7

    assert hypothesis.source_pattern == \
        "the ball moved."


def test_empty_statement_is_rejected():

    with pytest.raises(ValueError):

        Hypothesis(
            statement="",
            confidence=0.5,
            source_pattern="pattern",
        )


def test_invalid_confidence_is_rejected():

    with pytest.raises(ValueError):

        Hypothesis(
            statement="Test",
            confidence=1.5,
            source_pattern="pattern",
        )
