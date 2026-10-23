from core.evidence_processor import (
    EvidencePolarity,
    EvidenceProcessor,
)
from core.observation import Observation


def test_supporting_observation():

    observation = Observation(
        content="The ball moved.",
        source="motion_sensor",
        confidence=0.9,
    )

    processor = EvidenceProcessor()

    evidence = processor.process(
        observation,
        EvidencePolarity.SUPPORTS,
    )

    assert evidence.supports is True
    assert evidence.strength == 0.9
    assert evidence.data["polarity"] == "supports"
    assert evidence.data["observation_id"] == observation.id


def test_contradicting_observation():

    observation = Observation(
        content="The ball did not move.",
        source="motion_sensor",
        confidence=0.8,
    )

    processor = EvidenceProcessor()

    evidence = processor.process(
        observation,
        EvidencePolarity.CONTRADICTS,
    )

    assert evidence.supports is False
    assert evidence.strength == 0.8
    assert evidence.data["polarity"] == "contradicts"


def test_neutral_observation():

    observation = Observation(
        content="The ball is visible.",
        source="vision_sensor",
        confidence=0.95,
    )

    processor = EvidenceProcessor()

    evidence = processor.process(
        observation,
        EvidencePolarity.NEUTRAL,
    )

    assert evidence.strength == 0.0
    assert evidence.data["polarity"] == "neutral"
