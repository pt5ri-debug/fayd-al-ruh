import pytest

from core.belief import Belief, BeliefStatus
from core.evidence import Evidence


def test_create_belief():

    belief = Belief(
        statement="Pushing the ball makes it move.",
        confidence=0.5,
    )

    assert belief.statement == \
        "Pushing the ball makes it move."

    assert belief.confidence == 0.5

    assert belief.status == \
        BeliefStatus.HYPOTHESIS


def test_empty_statement():

    with pytest.raises(ValueError):

        Belief(
            statement=""
        )


def test_invalid_confidence():

    with pytest.raises(ValueError):

        Belief(
            statement="Test belief",
            confidence=1.5,
        )


def test_add_supporting_evidence():

    belief = Belief(
        statement="The ball moves when pushed."
    )

    evidence = Evidence(
        description="The ball moved after pushing.",
        source="sensory_observation",
        supports=True,
        strength=0.9,
    )

    belief.add_evidence(evidence)

    assert evidence.id in \
        belief.supporting_evidence

    assert belief.evidence_count() == 1


def test_add_contradicting_evidence():

    belief = Belief(
        statement="The ball always moves when pushed."
    )

    evidence = Evidence(
        description="The ball did not move.",
        source="sensory_observation",
        supports=False,
        strength=0.8,
    )

    belief.add_evidence(evidence)

    assert evidence.id in \
        belief.contradicting_evidence

    assert belief.evidence_count() == 1


def test_change_confidence():

    belief = Belief(
        statement="The ball moves."
    )

    belief.set_confidence(0.85)

    assert belief.confidence == 0.85


def test_change_status():

    belief = Belief(
        statement="The ball moves."
    )

    belief.set_status(
        BeliefStatus.PROVISIONAL
    )

    assert belief.status == \
        BeliefStatus.PROVISIONAL


def test_to_dict():

    belief = Belief(
        statement="The ball moves.",
        confidence=0.8,
    )

    data = belief.to_dict()

    assert data["statement"] == \
        "The ball moves."

    assert data["confidence"] == 0.8

    assert data["status"] == "hypothesis"


def test_belief_can_reference_source_hypothesis():

    belief = Belief(
        statement="The ball is moving repeatedly.",
        confidence=0.85,
        source_hypothesis_id="HYP-test",
    )

    assert belief.source_hypothesis_id == "HYP-test"


def test_belief_without_source_hypothesis_is_allowed():

    belief = Belief(
        statement="The ball exists.",
        confidence=0.9,
    )

    assert belief.source_hypothesis_id is None


def test_belief_to_dict_contains_source_hypothesis():

    belief = Belief(
        statement="The ball is moving repeatedly.",
        confidence=0.85,
        source_hypothesis_id="HYP-test",
    )

    data = belief.to_dict()

    assert data["source_hypothesis_id"] == "HYP-test"


def test_empty_source_hypothesis_id_is_rejected():

    with pytest.raises(ValueError):
        Belief(
            statement="The ball is moving.",
            confidence=0.85,
            source_hypothesis_id="",
        )    