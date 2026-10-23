from core.memory_associator import MemoryAssociator


def test_associator_returns_related_memories():
    associator = MemoryAssociator()
    associator.associate("A", "B", .8)
    assert associator.related_to("A")[0].target_id == "B"
