from core.working_memory import WorkingMemory


def test_working_memory_evicts_oldest_item_at_capacity():
    memory = WorkingMemory(capacity=2)
    memory.put("a", 1)
    memory.put("b", 2)
    memory.put("c", 3)
    assert [item.key for item in memory.items()] == ["b", "c"]
