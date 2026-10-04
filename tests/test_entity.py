from datetime import datetime, timezone
from time import sleep

import pytest

from core.entity import Entity, EntityType


def test_create_entity():

    entity = Entity(
        entity_type=EntityType.PERSON,
        name="Robot",
    )

    assert entity.id.startswith("E-")
    assert entity.entity_type == EntityType.PERSON
    assert entity.name == "Robot"

    assert entity.confidence == 1.0
    assert entity.activation == 0.0

    assert entity.properties == {}
    assert entity.relations == []
    assert entity.metadata == {}


def test_entity_timestamps_are_utc_aware():

    entity = Entity(
        entity_type=EntityType.OBJECT,
        name="Ball",
    )

    assert entity.created_at.tzinfo is not None
    assert entity.updated_at.tzinfo is not None

    assert (
        entity.created_at.utcoffset()
        == timezone.utc.utcoffset(entity.created_at)
    )

    assert (
        entity.updated_at.utcoffset()
        == timezone.utc.utcoffset(entity.updated_at)
    )


def test_created_at_is_before_or_equal_to_updated_at():

    entity = Entity(
        entity_type=EntityType.OBJECT,
        name="Ball",
    )

    assert entity.created_at <= entity.updated_at


def test_touch_updates_timestamp():

    entity = Entity(
        entity_type=EntityType.OBJECT,
        name="Ball",
    )

    original_updated_at = entity.updated_at

    sleep(0.001)

    entity.touch()

    assert entity.updated_at > original_updated_at

    assert entity.created_at <= entity.updated_at


def test_empty_name_is_rejected():

    with pytest.raises(ValueError):
        Entity(
            entity_type=EntityType.OBJECT,
            name="",
        )


def test_whitespace_name_is_rejected():

    with pytest.raises(ValueError):
        Entity(
            entity_type=EntityType.OBJECT,
            name="   ",
        )


def test_invalid_confidence_is_rejected():

    with pytest.raises(ValueError):

        Entity(
            entity_type=EntityType.OBJECT,
            name="Ball",
            confidence=1.1,
        )


def test_negative_confidence_is_rejected():

    with pytest.raises(ValueError):

        Entity(
            entity_type=EntityType.OBJECT,
            name="Ball",
            confidence=-0.1,
        )


def test_invalid_activation_is_rejected():

    with pytest.raises(ValueError):

        Entity(
            entity_type=EntityType.OBJECT,
            name="Ball",
            activation=1.1,
        )


def test_negative_activation_is_rejected():

    with pytest.raises(ValueError):

        Entity(
            entity_type=EntityType.OBJECT,
            name="Ball",
            activation=-0.1,
        )


def test_set_property():

    entity = Entity(
        entity_type=EntityType.OBJECT,
        name="Ball",
    )

    original_updated_at = entity.updated_at

    sleep(0.001)

    entity.set_property(
        "color",
        "red",
    )

    assert entity.get_property("color") == "red"
    assert entity.updated_at > original_updated_at


def test_get_missing_property_returns_none():

    entity = Entity(
        entity_type=EntityType.OBJECT,
        name="Ball",
    )

    assert entity.get_property(
        "missing"
    ) is None


def test_remove_property():

    entity = Entity(
        entity_type=EntityType.OBJECT,
        name="Ball",
        properties={
            "color": "red",
        },
    )

    entity.remove_property("color")

    assert "color" not in entity.properties


def test_remove_missing_property_does_nothing():

    entity = Entity(
        entity_type=EntityType.OBJECT,
        name="Ball",
    )

    entity.remove_property("missing")

    assert entity.properties == {}


def test_add_relation():

    entity = Entity(
        entity_type=EntityType.PERSON,
        name="Robot",
    )

    entity.add_relation("R-001")

    assert entity.relations == ["R-001"]


def test_duplicate_relation_is_not_added():

    entity = Entity(
        entity_type=EntityType.PERSON,
        name="Robot",
    )

    entity.add_relation("R-001")
    entity.add_relation("R-001")

    assert entity.relations == ["R-001"]


def test_remove_relation():

    entity = Entity(
        entity_type=EntityType.PERSON,
        name="Robot",
        relations=["R-001"],
    )

    entity.remove_relation("R-001")

    assert entity.relations == []


def test_remove_missing_relation_does_nothing():

    entity = Entity(
        entity_type=EntityType.PERSON,
        name="Robot",
    )

    entity.remove_relation("R-001")

    assert entity.relations == []


def test_set_confidence():

    entity = Entity(
        entity_type=EntityType.OBJECT,
        name="Ball",
    )

    entity.set_confidence(0.75)

    assert entity.confidence == 0.75


def test_set_confidence_boundary_values():

    entity = Entity(
        entity_type=EntityType.OBJECT,
        name="Ball",
    )

    entity.set_confidence(0.0)

    assert entity.confidence == 0.0

    entity.set_confidence(1.0)

    assert entity.confidence == 1.0


def test_set_confidence_invalid_value():

    entity = Entity(
        entity_type=EntityType.OBJECT,
        name="Ball",
    )

    with pytest.raises(ValueError):
        entity.set_confidence(1.1)

    with pytest.raises(ValueError):
        entity.set_confidence(-0.1)


def test_activate():

    entity = Entity(
        entity_type=EntityType.OBJECT,
        name="Ball",
    )

    entity.activate(0.7)

    assert entity.activation == 0.7


def test_activate_caps_at_one():

    entity = Entity(
        entity_type=EntityType.OBJECT,
        name="Ball",
    )

    entity.activate(5.0)

    assert entity.activation == 1.0


def test_deactivate():

    entity = Entity(
        entity_type=EntityType.OBJECT,
        name="Ball",
        activation=0.8,
    )

    entity.deactivate()

    assert entity.activation == 0.0


def test_to_dict():

    entity = Entity(
        entity_type=EntityType.OBJECT,
        name="Ball",
        confidence=0.9,
        activation=0.4,
        properties={
            "color": "red",
        },
        metadata={
            "source": "sensor",
        },
    )

    data = entity.to_dict()

    assert data["id"] == entity.id
    assert data["type"] == "Object"
    assert data["name"] == "Ball"
    assert data["confidence"] == 0.9
    assert data["activation"] == 0.4
    assert data["properties"] == {
        "color": "red",
    }
    assert data["metadata"] == {
        "source": "sensor",
    }

    assert isinstance(
        data["created_at"],
        str,
    )

    assert isinstance(
        data["updated_at"],
        str,
    )


def test_to_dict_timestamps_are_iso_format():

    entity = Entity(
        entity_type=EntityType.EVENT,
        name="Movement",
    )

    data = entity.to_dict()

    created = datetime.fromisoformat(
        data["created_at"]
    )

    updated = datetime.fromisoformat(
        data["updated_at"]
    )

    assert created.tzinfo is not None
    assert updated.tzinfo is not None


def test_all_entity_types_are_supported():

    for entity_type in EntityType:

        entity = Entity(
            entity_type=entity_type,
            name="Test",
        )

        assert entity.entity_type == entity_type
