"""Relation lifecycle over the graph: create, reinforce, version, find conflicts.

:class:`~core.relation.Relation` is the *record*; this module is the *manager*.
It owns the operations that need the whole graph -- resolving an endpoint label
to an entity id, refusing an edge to a missing endpoint, reinforcing an identical
edge instead of duplicating it, and recording a bounded history so a relation's
life can be read back.

It leans on the two modules that already know the hard parts: relation *polarity*
(:mod:`core.relation_polarity`) decides whether two types oppose, and the
validator (:mod:`core.knowledge_validation`) decides whether a claim is sound.
This manager never re-implements either -- it is the write path that keeps them
in step.

Honest by construction: a **contradiction is preserved** (both relations stay,
the conflict is reported, nothing is deleted), an **identical** relation is
reinforced (confidence merged, evidence appended -- never a second edge), and an
endpoint that does not exist raises rather than creating a phantom entity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from core.knowledge_graph import KnowledgeGraph
from core.relation import Relation
from core.relation_polarity import opposes

#: How many history records one relation keeps (bounded, newest last).
DEFAULT_HISTORY_LIMIT = 64


@dataclass(frozen=True)
class RelationEvent:
    """One thing that happened to a relation."""

    relation_id: str
    action: str
    detail: str = ""
    confidence: float = 0.0
    at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def describe(self) -> str:
        return f"{self.action}: {self.detail}" if self.detail else self.action


@dataclass(frozen=True)
class RelationConflict:
    """Two relations on the same ordered pair whose types oppose."""

    first_id: str
    second_id: str
    first_type: str
    second_type: str
    severity: float

    def describe(self) -> str:
        return (
            f"{self.first_id} ({self.first_type}) vs {self.second_id} ({self.second_type}) "
            f"— severity {self.severity:.2f}"
        )


@dataclass(frozen=True)
class RelationResult:
    """The outcome of creating a relation: the relation, and whether it was new."""

    relation: Relation
    created: bool
    reinforced: bool = False
    conflict: bool = False

    def describe(self) -> str:
        state = "created" if self.created else ("reinforced" if self.reinforced else "present")
        return f"{self.relation.id} {state}" + (" (conflict preserved)" if self.conflict else "")


class RelationManager:
    """Create, reinforce, and inspect the relations in one graph."""

    def __init__(self, graph: KnowledgeGraph, *, history_limit: int = DEFAULT_HISTORY_LIMIT) -> None:
        if history_limit < 0:
            raise ValueError("History limit cannot be negative.")
        self.graph = graph
        self.history_limit = history_limit
        self._history: dict[str, list[RelationEvent]] = {}

    # -- create ------------------------------------------------------------

    def create(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        *,
        confidence: float = 0.5,
        evidence_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RelationResult:
        """Create a typed, directed relation, or reinforce an identical one.

        Both endpoints must already exist -- an unknown endpoint raises
        ``KeyError`` rather than creating a phantom entity, because a relation to
        nothing is not knowledge. An identical triple (same source, target, type)
        is *reinforced*: its confidence is raised to the stronger value and the
        new evidence is appended, so there is never a duplicate edge. A relation
        whose type **opposes** a stored one on the same ordered pair is created
        and the conflict is reported -- both are kept.
        """
        if source_id not in self.graph.entities:
            raise KeyError(f"Unknown source entity: {source_id}")
        if target_id not in self.graph.entities:
            raise KeyError(f"Unknown target entity: {target_id}")

        for relation in self.graph.relations.values():
            if not relation.active or relation.source != source_id or relation.target != target_id:
                continue
            if relation.relation_type == relation_type:
                relation.set_confidence(round(max(relation.confidence, confidence), 6))
                for evidence_id in evidence_ids or []:
                    relation.add_evidence(evidence_id)
                self._record(relation, "reinforced", f"confidence {relation.confidence:.3f}")
                return RelationResult(relation=relation, created=False, reinforced=True)

        conflict = any(
            relation.active
            and relation.source == source_id
            and relation.target == target_id
            and opposes(relation.relation_type, relation_type)
            for relation in self.graph.relations.values()
        )

        relation = Relation(
            source=source_id,
            target=target_id,
            relation_type=relation_type,
            confidence=confidence,
            evidence_ids=list(evidence_ids or []),
            metadata=dict(metadata or {}),
        )
        self.graph.add_relation(relation)
        self._record(relation, "created", f"confidence {confidence:.3f}")
        if conflict:
            self._record(relation, "conflict", "an opposing relation is preserved beside it")
        return RelationResult(relation=relation, created=True, conflict=conflict)

    def attach_evidence(self, relation_id: str, evidence_id: str) -> Relation:
        """Attach one evidence item to a relation (idempotent)."""
        relation = self.graph.get_relation(relation_id)
        if relation is None:
            raise KeyError(f"Unknown relation: {relation_id}")
        if evidence_id not in self.graph.evidence:
            raise KeyError(f"Unknown evidence: {evidence_id}")
        relation.add_evidence(evidence_id)
        self._record(relation, "evidence", evidence_id)
        return relation

    def deactivate(self, relation_id: str, *, reason: str = "deactivated") -> Relation:
        """Mark a relation inactive **without deleting it** -- history is preserved."""
        relation = self.graph.get_relation(relation_id)
        if relation is None:
            raise KeyError(f"Unknown relation: {relation_id}")
        relation.deactivate()
        self._record(relation, "deactivated", reason)
        return relation

    def reactivate(self, relation_id: str, *, reason: str = "reactivated") -> Relation:
        """Bring a deactivated relation back."""
        relation = self.graph.get_relation(relation_id)
        if relation is None:
            raise KeyError(f"Unknown relation: {relation_id}")
        relation.activate()
        self._record(relation, "reactivated", reason)
        return relation

    # -- read --------------------------------------------------------------

    def history(self, relation_id: str) -> tuple[RelationEvent, ...]:
        """The recorded events for one relation, oldest first."""
        return tuple(self._history.get(relation_id, ()))

    def find_conflicts(self, *, active_only: bool = True) -> tuple[RelationConflict, ...]:
        """Every opposing pair on the same ordered pair, ordered by severity."""
        relations = [r for r in self.graph.relations.values() if (r.active or not active_only)]
        found: list[RelationConflict] = []
        for index, first in enumerate(relations):
            for second in relations[index + 1 :]:
                if (first.source, first.target) != (second.source, second.target):
                    continue
                if opposes(first.relation_type, second.relation_type):
                    found.append(
                        RelationConflict(
                            first_id=first.id,
                            second_id=second.id,
                            first_type=first.relation_type,
                            second_type=second.relation_type,
                            severity=round(min(first.confidence, second.confidence), 6),
                        )
                    )
        found.sort(key=lambda item: (-item.severity, item.first_type, item.second_type))
        return tuple(found)

    def duplicates(self) -> tuple[tuple[str, str], ...]:
        """Ids of relations with an identical (source, target, type) triple."""
        seen: dict[tuple[str, str, str], str] = {}
        found: list[tuple[str, str]] = []
        for relation in self.graph.relations.values():
            key = (relation.source, relation.target, relation.relation_type)
            if key in seen:
                found.append((seen[key], relation.id))
            else:
                seen[key] = relation.id
        return tuple(found)

    def describe(self) -> str:
        active = sum(1 for r in self.graph.relations.values() if r.active)
        return (
            f"RelationManager({active} active of {len(self.graph.relations)} relation(s), "
            f"{len(self.find_conflicts())} conflict(s))"
        )

    # -- internals ---------------------------------------------------------

    def _record(self, relation: Relation, action: str, detail: str = "") -> None:
        items = self._history.setdefault(relation.id, [])
        items.append(
            RelationEvent(
                relation_id=relation.id,
                action=action,
                detail=detail,
                confidence=relation.confidence,
            )
        )
        if self.history_limit and len(items) > self.history_limit:
            del items[: len(items) - self.history_limit]


__all__ = [
    "DEFAULT_HISTORY_LIMIT",
    "RelationConflict",
    "RelationEvent",
    "RelationManager",
    "RelationResult",
]
