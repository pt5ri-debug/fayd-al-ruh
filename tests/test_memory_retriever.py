from core.episodic_memory import EpisodicMemory
from core.memory_associator import MemoryAssociator
from core.memory_retriever import MemoryRetriever
from core.semantic_memory import SemanticMemoryIndex


def test_retriever_uses_semantic_index():
    episodic = EpisodicMemory(); episodic.store("E1", "rain makes ground wet")
    index = SemanticMemoryIndex(); index.add("E1", "rain makes ground wet")
    result = MemoryRetriever().retrieve("rain wet", episodic, index, MemoryAssociator())
    assert result[0].id == "E1"
