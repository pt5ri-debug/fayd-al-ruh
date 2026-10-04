from core.belief import Belief
from core.evidence import Evidence
from core.learning_engine import LearningEngine


def test_learning_from_supporting_evidence():

    belief = Belief(
        statement="The ball moves.",
        confidence=0.5,
    )

    evidence = [
        Evidence(
            description="The ball moved.",
            source="sensor",
            supports=True,
            strength=1.0,
        )
    ]

    engine = LearningEngine()

    result = engine.learn(
        belief,
        evidence,
    )

    assert result.belief_id == belief.id
    assert result.evidence_count == 1
    assert result.learned is True
    assert result.new_confidence > 0.5


def test_learning_from_contradicting_evidence():

    belief = Belief(
        statement="The ball moves.",
        confidence=0.9,
    )

    evidence = [
        Evidence(
            description="The ball did not move.",
            source="sensor",
            supports=False,
            strength=1.0,
        )
    ]

    engine = LearningEngine()

    result = engine.learn(
        belief,
        evidence,
    )

    assert result.learned is True
    assert result.new_confidence < 0.9


def test_no_evidence_means_no_learning():

    belief = Belief(
        statement="The ball moves.",
        confidence=0.7,
    )

    engine = LearningEngine()

    result = engine.learn(
        belief,
        [],
    )

    assert result.learned is False
    assert result.new_confidence == 0.7


def test_learning_tracks_previous_confidence():

    belief = Belief(
        statement="The ball moves.",
        confidence=0.4,
    )

    evidence = [
        Evidence(
            description="The ball moved.",
            source="sensor",
            supports=True,
            strength=0.8,
        )
    ]

    engine = LearningEngine()

    result = engine.learn(
        belief,
        evidence,
    )

    assert result.previous_confidence == 0.4
    assert result.new_confidence == belief.confidence