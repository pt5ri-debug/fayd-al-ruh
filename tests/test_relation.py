import pytest
from core.relation import Relation


def test_create_relation():

    relation = Relation(
        source="E-robot",
        target="E-ball",
        relation_type="SEES",
        confidence=0.9,
    )

    assert relation.source == "E-robot"
    assert relation.target == "E-ball"
    assert relation.relation_type == "SEES"
    assert relation.confidence == 0.9
    assert relation.active is True


def test_empty_source():

    with pytest.raises(ValueError):

        Relation(
            source="",
            target="E-ball",
            relation_type="SEES",
        )


def test_empty_target():

    with pytest.raises(ValueError):

        Relation(
            source="E-robot",
            target="",
            relation_type="SEES",
        )


def test_empty_relation_type():

    with pytest.raises(ValueError):

        Relation(
            source="E-robot",
            target="E-ball",
            relation_type="",
        )


def test_invalid_confidence():

    with pytest.raises(ValueError):

        Relation(
            source="E-robot",
            target="E-ball",
            relation_type="SEES",
            confidence=1.5,
        )


def test_self_relation_is_rejected():

    with pytest.raises(ValueError):

        Relation(
            source="E-robot",
            target="E-robot",
            relation_type="SEES",
        )


def test_attach_belief():

    relation = Relation(
        source="E-robot",
        target="E-ball",
        relation_type="SEES",
    )

    relation.attach_belief("BL-1234")

    assert relation.belief_id == "BL-1234"


def test_add_evidence():

    relation = Relation(
        source="E-robot",
        target="E-ball",
        relation_type="SEES",
    )

    relation.add_evidence("EV-1234")

    assert "EV-1234" in relation.evidence_ids


def test_duplicate_evidence_is_not_added():

    relation = Relation(
        source="E-robot",
        target="E-ball",
        relation_type="SEES",
    )

    relation.add_evidence("EV-1234")
    relation.add_evidence("EV-1234")

    assert relation.evidence_ids == ["EV-1234"]


def test_deactivate_relation():

    relation = Relation(
        source="E-robot",
        target="E-ball",
        relation_type="SEES",
    )

    relation.deactivate()

    assert relation.active is False


def test_activate_relation():

    relation = Relation(
        source="E-robot",
        target="E-ball",
        relation_type="SEES",
    )

    relation.deactivate()
    relation.activate()

    assert relation.active is True


def test_to_dict():

    relation = Relation(
        source="E-robot",
        target="E-ball",
        relation_type="SEES",
        confidence=0.85,
    )

    data = relation.to_dict()

    assert data["source"] == "E-robot"
    assert data["target"] == "E-ball"
    assert data["relation_type"] == "SEES"
    assert data["confidence"] == 0.85
    assert data["active"] is True
