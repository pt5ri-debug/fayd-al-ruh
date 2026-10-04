from __future__ import annotations

from core.knowledge_graph import KnowledgeGraph
from core.serializer import Serializer


class GraphStore:
    """Exports graph data with stable IDs for persistence or external graph stores."""
    def __init__(self, serializer: Serializer | None = None) -> None:
        self.serializer = serializer or Serializer()

    def dump(self, graph: KnowledgeGraph) -> dict:
        return self.serializer.to_primitive({
            "entities": graph.entities,
            "evidence": graph.evidence,
            "beliefs": graph.beliefs,
            "relations": graph.relations,
        })
