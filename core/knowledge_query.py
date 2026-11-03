"""One read interface over the knowledge graph for memory and reasoning.

The graph (:mod:`core.knowledge_graph`) already knows how to find relations,
neighbours, and a shortest path. What was missing is a **query surface** that
answers the questions a later component (memory, reasoning, the UI) actually
asks -- and that answers them *consistently*, so no caller re-derives a traversal
and drifts from the others.

``KnowledgeQueryEngine`` wraps one graph and offers: entity lookup, relationship
lookup, multi-hop traversal, evidence retrieval, context-aware retrieval,
contradiction retrieval, temporal queries, and source-based filtering. Every
method is **read-only** (it never mutates the graph) and returns plain dataclasses
so a caller can render or serialise them.

Two honesty rules:

* an unknown entity, an empty result, and a broken path are all **``()``/``None``**
  -- a query never invents a node or an edge to fill its answer;
* a *contradiction* is reported by the graph's own relation polarity
  (:mod:`core.relation_polarity`), never by a word guess here.

Dependency-free; it imports no model and does not touch the network.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from core.entity import Entity
from core.evidence import Evidence
from core.knowledge_graph import KnowledgeGraph
from core.relation import Relation
from core.relation_polarity import opposes

#: The default bound on a multi-hop traversal, so a query can never walk a whole
#: graph by accident.
DEFAULT_MAX_HOPS = 3

#: The default cap on rows returned, so a broad query stays readable.
DEFAULT_LIMIT = 50


@dataclass(frozen=True)
class GraphStep:
    """One hop of a traversed path, with the direction it was walked."""

    source: str
    relation_type: str
    target: str
    confidence: float
    forward: bool = True
    relation_id: str = ""

    def describe(self) -> str:
        arrow = "→" if self.forward else "←"
        return f"{self.source} {arrow}{self.relation_type}{arrow if self.forward else '—'} {self.target}"


@dataclass(frozen=True)
class GraphPath:
    """A multi-hop path between two entities, with its weakest-link confidence."""

    start: str
    end: str
    steps: tuple[GraphStep, ...] = ()

    @property
    def hops(self) -> int:
        return len(self.steps)

    @property
    def confidence(self) -> float:
        """The product of the steps' confidences -- a longer chain is honestly weaker."""
        value = 1.0
        for step in self.steps:
            value *= step.confidence
        return round(value, 6)

    def describe(self) -> str:
        if not self.steps:
            return f"{self.start} (no path)"
        # Render the endpoints by name (the steps carry them), so a reader sees
        # concepts rather than internal ids; the ids stay for identity.
        origin = self.steps[0].source
        destination = self.steps[-1].target
        return f"{origin} → {destination} ({self.hops} hop(s), confidence {self.confidence:.3f})"


@dataclass(frozen=True)
class Contradiction:
    """Two relations that assert opposing things on the same ordered pair."""

    first: Relation
    second: Relation
    severity: float

    def describe(self) -> str:
        return (
            f"{self.first.source} —{self.first.relation_type}→ {self.first.target} "
            f"vs —{self.second.relation_type}→ (severity {self.severity:.2f})"
        )


@dataclass(frozen=True)
class EntitySummary:
    """A readable view of one entity and what it participates in."""

    id: str
    name: str
    entity_type: str
    confidence: float
    relation_count: int
    evidence_count: int
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        return f"{self.name} [{self.entity_type}] ({self.relation_count} relation(s))"


class KnowledgeQueryEngine:
    """Read-only queries over one :class:`KnowledgeGraph`.

    The engine holds no state of its own, so several components can share one and
    always see the same graph.
    """

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph

    # -- entity ------------------------------------------------------------

    def entity(self, entity_id: str) -> Entity | None:
        """One entity by id, or ``None``."""
        return self.graph.get_entity(entity_id)

    def find_entity(self, name: str) -> Entity | None:
        """One entity by name (case-insensitive), or ``None``."""
        wanted = name.strip().casefold()
        if not wanted:
            return None
        for entity in self.graph.entities.values():
            if entity.name.strip().casefold() == wanted:
                return entity
        return None

    def entities(self, *, entity_type: str | None = None, name_contains: str = "", limit: int = DEFAULT_LIMIT) -> tuple[Entity, ...]:
        """Entities, optionally filtered by type or a name substring."""
        needle = name_contains.casefold()
        found = [
            entity
            for entity in self.graph.entities.values()
            if (entity_type is None or entity.entity_type.value == entity_type)
            and (not needle or needle in entity.name.casefold())
        ]
        found.sort(key=lambda item: item.name.casefold())
        return tuple(found[:limit])

    def summarize(self, entity_id: str) -> EntitySummary | None:
        """A readable summary of one entity, or ``None`` for an unknown id."""
        entity = self.graph.get_entity(entity_id)
        if entity is None:
            return None
        relations = self.relations_of(entity_id)
        evidence: set[str] = set()
        for relation in relations:
            evidence.update(relation.evidence_ids)
        return EntitySummary(
            id=entity.id,
            name=entity.name,
            entity_type=entity.entity_type.value,
            confidence=entity.confidence,
            relation_count=len(relations),
            evidence_count=len(evidence),
            created_at=entity.created_at.isoformat(),
            metadata=dict(entity.metadata),
        )

    # -- relationships -----------------------------------------------------

    def relations_of(self, entity_id: str, *, relation_type: str | None = None, active_only: bool = True) -> tuple[Relation, ...]:
        """Every relation touching an entity (either endpoint)."""
        found = [
            relation
            for relation in self.graph.relations.values()
            if (not active_only or relation.active)
            and (relation.source == entity_id or relation.target == entity_id)
            and (relation_type is None or relation.relation_type == relation_type)
        ]
        found.sort(key=lambda item: (item.relation_type, item.source, item.target))
        return tuple(found)

    def between(self, source: str, target: str, *, relation_type: str | None = None) -> tuple[Relation, ...]:
        """Direct relations from one entity to another (by id), ordered."""
        found = [
            relation
            for relation in self.graph.relations.values()
            if relation.active
            and relation.source == source
            and relation.target == target
            and (relation_type is None or relation.relation_type == relation_type)
        ]
        found.sort(key=lambda item: item.relation_type)
        return tuple(found)

    def neighbors(self, entity_id: str, *, depth: int = 1) -> tuple[Entity, ...]:
        """Entities reachable within ``depth`` hops (breadth-first, de-duplicated)."""
        if depth < 1:
            return ()
        seen: set[str] = {entity_id}
        frontier = [entity_id]
        found: list[Entity] = []
        for _ in range(depth):
            next_frontier: list[str] = []
            for current in frontier:
                for relation in self.relations_of(current):
                    other = relation.target if relation.source == current else relation.source
                    if other in seen:
                        continue
                    seen.add(other)
                    entity = self.graph.get_entity(other)
                    if entity is not None:
                        found.append(entity)
                        next_frontier.append(other)
            frontier = next_frontier
        return tuple(found)

    # -- traversal ---------------------------------------------------------

    def path(self, start: str, end: str, *, max_hops: int = DEFAULT_MAX_HOPS) -> GraphPath | None:
        """The shortest typed path between two entities, or ``None``.

        Breadth-first over every active relation in **both** directions, so a
        chain that reaches an endpoint through an incoming edge is found too. A
        step records whether it was walked forward (with the stored edge) or
        backward (against it), so a reader sees which claim each step rests on.
        """
        if max_hops < 1:
            return None
        if start == end:
            return GraphPath(start=start, end=end, steps=())
        if start not in self.graph.entities or end not in self.graph.entities:
            return None

        # queue of (node, path steps)
        queue: deque[tuple[str, tuple[GraphStep, ...]]] = deque([(start, ())])
        visited: set[str] = {start}
        while queue:
            node, trail = queue.popleft()
            if len(trail) >= max_hops:
                continue
            for relation in self.relations_of(node):
                if relation.source == node:
                    other, forward = relation.target, True
                else:
                    other, forward = relation.source, False
                if other in visited:
                    continue
                step = GraphStep(
                    source=self._name(relation.source),
                    relation_type=relation.relation_type,
                    target=self._name(relation.target),
                    confidence=relation.confidence,
                    forward=forward,
                    relation_id=relation.id,
                )
                new_trail = (*trail, step)
                if other == end:
                    return GraphPath(start=start, end=end, steps=new_trail)
                visited.add(other)
                queue.append((other, new_trail))
        return None

    def _name(self, entity_id: str) -> str:
        entity = self.graph.get_entity(entity_id)
        return entity.name if entity is not None else entity_id

    # -- evidence ----------------------------------------------------------

    def evidence_for_relation(self, relation_id: str) -> tuple[Evidence, ...]:
        """Every evidence item attached to a relation."""
        return tuple(self.graph.get_relation_evidence(relation_id))

    def evidence_for_entity(self, entity_id: str) -> tuple[Evidence, ...]:
        """Every evidence item attached to any relation touching an entity."""
        found: list[Evidence] = []
        seen: set[str] = set()
        for relation in self.relations_of(entity_id):
            for item in self.graph.get_relation_evidence(relation.id):
                if item.id not in seen:
                    seen.add(item.id)
                    found.append(item)
        return tuple(found)

    def by_source(self, source: str, *, limit: int = DEFAULT_LIMIT) -> tuple[Evidence, ...]:
        """Evidence from one source (a host, a document, a personality)."""
        wanted = source.strip().casefold()
        found = [item for item in self.graph.evidence.values() if item.source.strip().casefold() == wanted]
        found.sort(key=lambda item: item.created_at)
        return tuple(found[:limit])

    # -- context -----------------------------------------------------------

    def context(self, entity_id: str, *, depth: int = 1, limit: int = DEFAULT_LIMIT) -> tuple[Relation, ...]:
        """The relations around an entity within ``depth`` hops -- its context."""
        ids = {entity_id} | {entity.id for entity in self.neighbors(entity_id, depth=depth)}
        found = [
            relation
            for relation in self.graph.relations.values()
            if relation.active and relation.source in ids and relation.target in ids
        ]
        found.sort(key=lambda item: (item.relation_type, item.source, item.target))
        return tuple(found[:limit])

    # -- contradictions ----------------------------------------------------

    def contradictions(self, *, entity_id: str | None = None, limit: int = DEFAULT_LIMIT) -> tuple[Contradiction, ...]:
        """Opposing relations on the same ordered pair (optionally scoped to one entity)."""
        found: list[Contradiction] = []
        relations = list(self.graph.relations.values())
        for index, first in enumerate(relations):
            if not first.active:
                continue
            if entity_id is not None and entity_id not in (first.source, first.target):
                continue
            for second in relations[index + 1 :]:
                if not second.active:
                    continue
                if (first.source, first.target) != (second.source, second.target):
                    continue
                if opposes(first.relation_type, second.relation_type):
                    severity = round(min(first.confidence, second.confidence), 6)
                    found.append(Contradiction(first=first, second=second, severity=severity))
        found.sort(key=lambda item: (-item.severity, item.first.relation_type, item.second.relation_type))
        return tuple(found[:limit])

    # -- temporal ----------------------------------------------------------

    def created_between(
        self,
        start: datetime,
        end: datetime,
        *,
        entity_id: str | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> tuple[Relation, ...]:
        """Relations created within a time window, optionally scoped to an entity.

        ``end`` may be a naive datetime; it is read as UTC. A reversed window
        yields ``()`` rather than raising -- a query is not a mutation.
        """
        start_aware = start if start.tzinfo is not None else start.replace(tzinfo=UTC)
        end_aware = end if end.tzinfo is not None else end.replace(tzinfo=UTC)
        if end_aware < start_aware:
            return ()
        found = [
            relation
            for relation in self.graph.relations.values()
            if start_aware <= _aware(relation.created_at) <= end_aware
            and (entity_id is None or entity_id in (relation.source, relation.target))
        ]
        found.sort(key=lambda item: item.created_at)
        return tuple(found[:limit])

    def latest(self, *, entity_id: str | None = None, limit: int = DEFAULT_LIMIT) -> tuple[Relation, ...]:
        """The most recently created relations, newest first."""
        found = [
            relation
            for relation in self.graph.relations.values()
            if entity_id is None or entity_id in (relation.source, relation.target)
        ]
        found.sort(key=lambda item: item.created_at, reverse=True)
        return tuple(found[:limit])

    def statistics(self) -> dict[str, int]:
        """Counts over the graph, for a status line."""
        return {
            "entities": len(self.graph.entities),
            "relations": len(self.graph.relations),
            "evidence": len(self.graph.evidence),
            "beliefs": len(self.graph.beliefs),
            "contradictions": len(self.contradictions(limit=10_000)),
        }

    def describe(self) -> str:
        stats = self.statistics()
        return (
            "KnowledgeQueryEngine("
            + ", ".join(f"{name}={value}" for name, value in stats.items())
            + ")"
        )


def _aware(moment: datetime) -> datetime:
    """A datetime read as UTC when it carries no timezone."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


__all__ = [
    "DEFAULT_LIMIT",
    "DEFAULT_MAX_HOPS",
    "Contradiction",
    "EntitySummary",
    "GraphPath",
    "GraphStep",
    "KnowledgeQueryEngine",
]
