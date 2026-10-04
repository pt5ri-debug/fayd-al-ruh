import pytest

from core.memory import CognitiveMemory


def test_create_empty_memory():

    memory = CognitiveMemory()

    assert memory.size == 0


def test_store_memory():

    memory = CognitiveMemory()

    item = memory.store(
        item_id="OBS-001",
        item_type="observation",
        content="The ball moved.",
        importance=0.8,
    )

    assert memory.size == 1
    assert item.item_id == "OBS-001"
    assert item.item_type == "observation"
    assert item.importance == 0.8


def test_get_memory():

    memory = CognitiveMemory()

    memory.store(
        item_id="OBS-001",
        item_type="observation",
        content="The ball moved.",
    )

    item = memory.get("OBS-001")

    assert item is not None
    assert item.content == "The ball moved."
    assert item.access_count == 1


def test_get_missing_memory():

    memory = CognitiveMemory()

    assert memory.get("UNKNOWN") is None


def test_duplicate_memory_is_rejected():

    memory = CognitiveMemory()

    memory.store(
        item_id="OBS-001",
        item_type="observation",
        content="Test",
    )

    with pytest.raises(ValueError):

        memory.store(
            item_id="OBS-001",
            item_type="observation",
            content="Duplicate",
        )


def test_invalid_importance():

    memory = CognitiveMemory()

    with pytest.raises(ValueError):

        memory.store(
            item_id="OBS-001",
            item_type="observation",
            content="Test",
            importance=1.5,
        )


def test_find_by_type():

    memory = CognitiveMemory()

    memory.store(
        item_id="OBS-001",
        item_type="observation",
        content="Observation 1",
    )

    memory.store(
        item_id="OBS-002",
        item_type="observation",
        content="Observation 2",
    )

    memory.store(
        item_id="BL-001",
        item_type="belief",
        content="Belief 1",
    )

    observations = memory.find_by_type(
        "observation"
    )

    assert len(observations) == 2


def test_important_items():

    memory = CognitiveMemory()

    memory.store(
        item_id="LOW",
        item_type="observation",
        content="Low importance",
        importance=0.2,
    )

    memory.store(
        item_id="HIGH",
        item_type="observation",
        content="High importance",
        importance=0.9,
    )

    important = memory.important_items(
        minimum_importance=0.7
    )

    assert len(important) == 1
    assert important[0].item_id == "HIGH"


def test_remove_memory():

    memory = CognitiveMemory()

    memory.store(
        item_id="OBS-001",
        item_type="observation",
        content="Test",
    )

    memory.remove("OBS-001")

    assert memory.size == 0
    assert memory.get("OBS-001") is None


def test_clear_memory():

    memory = CognitiveMemory()

    memory.store(
        item_id="OBS-001",
        item_type="observation",
        content="Test 1",
    )

    memory.store(
        item_id="OBS-002",
        item_type="observation",
        content="Test 2",
    )

    memory.clear()

    assert memory.size == 0