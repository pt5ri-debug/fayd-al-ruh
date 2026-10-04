from __future__ import annotations

from core.causal_graph import CausalGraph
from core.graph_store import GraphStore
from core.cognitive_context import ContextRegistry
from core.cognitive_state import CognitiveState
from core.knowledge_graph import KnowledgeGraph
from core.memory import CognitiveMemory
from core.memory_store import MemoryStore
from core.persistence import Persistence
from core.snapshot import Snapshot
from core.snapshot_loader import RestoredSnapshot, SnapshotLoader
from core.serializer import Serializer


class KnowledgeStore:
    """Coordinates durable snapshots of the knowledge graph and memory."""
    def __init__(self, persistence: Persistence, graph_store: GraphStore | None = None, memory_store: MemoryStore | None = None) -> None:
        self.persistence = persistence
        self.graph_store = graph_store or GraphStore()
        self.memory_store = memory_store or MemoryStore()
        self.serializer = Serializer()

    def snapshot(
        self,
        name: str,
        graph: KnowledgeGraph,
        memory: CognitiveMemory,
        state: CognitiveState | None = None,
        contexts: ContextRegistry | None = None,
        causal_graph: CausalGraph | None = None,
    ) -> Snapshot:
        payload = {
            "graph": self.graph_store.dump(graph),
            "memory": self.memory_store.dump(memory),
            "state": self.serializer.to_primitive(state) if state is not None else {},
            "contexts": contexts.dump() if contexts is not None else {},
            "causal_graph": self.serializer.to_primitive({"links": causal_graph.links}) if causal_graph is not None else {},
        }
        snapshot = Snapshot.create(name, payload)
        self.persistence.save("snapshots", name, {
            "name": name,
            "payload": snapshot.payload,
            "checksum": snapshot.checksum,
            "created_at": snapshot.created_at.isoformat(),
        })
        return snapshot

    def load_snapshot(self, name: str) -> dict | None:
        return self.persistence.load("snapshots", name)

    def restore_snapshot(self, name: str) -> RestoredSnapshot | None:
        record = self.load_snapshot(name)
        return SnapshotLoader().restore(record) if record is not None else None
