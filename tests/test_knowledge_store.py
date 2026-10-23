from core.belief import Belief
from core.cognitive_context import ContextRegistry
from core.cognitive_state import CognitiveState
from core.evidence import Evidence
from core.knowledge_graph import KnowledgeGraph
from core.knowledge_store import KnowledgeStore
from core.memory import CognitiveMemory
from core.observation import Observation
from core.snapshot_loader import SnapshotLoader
from storage.json_store import JsonStore


def _populated_components():
    graph = KnowledgeGraph()
    evidence = Evidence("Rain fell on the garden.", "demo_sensor", supports=True, strength=0.9)
    graph.add_evidence(evidence)
    belief = Belief("Rain makes the garden wet.", confidence=0.7)
    graph.add_belief(belief)

    memory = CognitiveMemory()
    memory.store("OBS-1", "observation", Observation("Rain falls.", "demo_sensor"), 0.8)

    state = CognitiveState()
    state.add_belief(belief)

    contexts = ContextRegistry()
    contexts.start(Observation("Rain falls.", "demo_sensor"), evidence, belief.id)

    return graph, memory, state, contexts


def test_knowledge_store_round_trips_graph_and_memory(tmp_path):
    graph, memory, state, contexts = _populated_components()
    store = KnowledgeStore(JsonStore(tmp_path))

    snapshot = store.snapshot("episode", graph, memory, state, contexts)
    assert snapshot.verify()

    restored = store.restore_snapshot("episode")
    assert restored is not None
    assert set(restored.graph.beliefs) == set(graph.beliefs)
    assert restored.memory.size == memory.size
    assert restored.state.cycle_count == state.cycle_count
    assert len(restored.contexts.contexts) == len(contexts.contexts)


def test_snapshot_loader_rejects_tampered_checksum():
    _graph, _memory, _state, _contexts = _populated_components()
    record = {
        "name": "episode",
        "payload": {"graph": {}, "memory": {}, "state": {}, "contexts": {}},
        "checksum": "0" * 64,
    }
    try:
        SnapshotLoader().restore(record)
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("Tampered snapshot must be rejected.")
