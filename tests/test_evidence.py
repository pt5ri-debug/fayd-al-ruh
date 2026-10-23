import pytest
from core.evidence import Evidence


def test_create_evidence():

    evidence = Evidence(
        description="Ball moved after being pushed.",
        source="sensory_observation",
        strength=0.9,
    )

    assert evidence.description == \
        "Ball moved after being pushed."

    assert evidence.source == \
        "sensory_observation"

    assert evidence.strength == 0.9

    assert evidence.supports is True


def test_invalid_strength():

    with pytest.raises(ValueError):

        Evidence(
            description="Invalid evidence",
            source="test",
            strength=1.5,
        )


def test_empty_description():

    with pytest.raises(ValueError):

        Evidence(
            description="",
            source="test",
        )
