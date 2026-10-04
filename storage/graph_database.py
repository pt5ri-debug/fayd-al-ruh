from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StoredEdge:
    source: str
    target: str
    relation: str


class GraphDatabase:
    """Portable in-memory graph-store adapter; can later be replaced by Neo4j or another provider."""
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.edges: set[StoredEdge] = set()

    def upsert_node(self, node_id: str, properties: dict) -> None:
        self.nodes[node_id] = dict(properties)

    def add_edge(self, source: str, target: str, relation: str) -> None:
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Graph edge endpoints must exist.")
        self.edges.add(StoredEdge(source, target, relation))
