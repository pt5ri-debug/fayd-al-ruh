import pytest

from core.observation import Observation


def test_create_observation():

    observation = Observation(
        content="The ball is red.",
        source="vision_sensor",
        confidence=0.95,
    )

    assert observation.content == "The ball is red."
    assert observation.source == "vision_sensor"
    assert observation.confidence == 0.95


def test_empty_content_is_rejected():

    with pytest.raises(ValueError):

        Observation(
            content="",
            source="vision_sensor",
        )


def test_empty_source_is_rejected():

    with pytest.raises(ValueError):

        Observation(
            content="The ball is red.",
            source="",
        )


def test_invalid_confidence_is_rejected():

    with pytest.raises(ValueError):

        Observation(
            content="The ball is red.",
            source="vision_sensor",
            confidence=1.5,
        )


def test_observation_becomes_evidence():

    observation = Observation(
        content="The ball is red.",
        source="vision_sensor",
        confidence=0.9,
    )

    evidence = observation.to_evidence()

    assert evidence.description == \
        "The ball is red."

    assert evidence.source == \
        "vision_sensor"

    assert evidence.strength == 0.9

    assert evidence.supports is True