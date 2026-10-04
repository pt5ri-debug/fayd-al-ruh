from core.semantic_memory import SemanticMemory


def test_semantic_memory_stores_retrievable_fact():
    memory = SemanticMemory(); memory.learn("F1", "rain makes ground wet")
    assert memory.retrieve("rain wet") == ["F1"]
