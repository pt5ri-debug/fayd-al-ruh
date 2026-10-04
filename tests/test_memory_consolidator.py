from core.episodic_memory import EpisodicMemory
from core.memory_consolidator import MemoryConsolidator
from core.working_memory import WorkingMemory


def test_consolidator_promotes_activated_context():
    working = WorkingMemory(); working.put("E1", "rain", .8)
    assert MemoryConsolidator().consolidate(working, EpisodicMemory()) == ["E1"]
