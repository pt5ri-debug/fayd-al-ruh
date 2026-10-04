from core.belief import Belief
from core.evidence import Evidence
from core.evidence_evaluator import EvidenceEvaluator


def test_supporting_evidence_increases_confidence():

    belief = Belief(
        statement="The ball moves.",
        confidence=0.5,
    )

    evidence = [
        Evidence(
            description="Ball moved.",
            source="sensor",
            supports=True,
            strength=1.0,
        )
    ]

    evaluator = EvidenceEvaluator()

    result = evaluator.apply(
        belief,
        evidence,
    )

    assert result.new_confidence > 0.5
    assert belief.confidence == result.new_confidence


def test_contradicting_evidence_reduces_confidence():

    belief = Belief(
        statement="The ball moves.",
        confidence=0.8,
    )

    evidence = [
        Evidence(
            description="Ball did not move.",
            source="sensor",
            supports=False,
            strength=1.0,
        )
    ]

    evaluator = EvidenceEvaluator()

    result = evaluator.apply(
        belief,
        evidence,
    )

    assert result.new_confidence < 0.8


def test_no_evidence_keeps_confidence():

    belief = Belief(
        statement="The ball moves.",
        confidence=0.7,
    )

    evaluator = EvidenceEvaluator()

    result = evaluator.evaluate(
        belief,
        [],
    )

    assert result.new_confidence == 0.7