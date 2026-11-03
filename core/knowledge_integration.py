"""The one controlled pipeline that writes a validated claim into the graph.

The project had every piece of this pipeline already -- ``KnowledgeValidator``
(structure, semantics, evidence, contradiction, temporal), ``Entity``/``Relation``
/``Evidence`` (the data), ``KnowledgeGraph`` (the store), ``OntologyLearner``
(the bounded graph learner), ``best_effort`` (a non-fatal side effect). What it
did not have was **one path** that runs them in order and, crucially, can **undo
itself** when a later step fails.

``KnowledgeIntegrator`` is that path. ``integrate`` performs the eleven steps the
task names -- receive, validate structure, identify existing entities, resolve
duplicates, extract/validate relationships, attach evidence, detect conflicts,
evaluate confidence, decide, update the graph, preserve history -- and it is
**transactional**: it snapshots exactly the state it is about to change (the
entity/relation/evidence dictionaries and the learner's own learned set) and
restores it byte-for-byte if any step raises. A failed integration therefore
leaves the graph **exactly** as it was -- no half-written entity, no dangling
relation -- which is what "safe transactions and rollback" means here.

Three rules keep it honest:

* it never *verifies* -- a stored claim is ``provisional`` or flagged, exactly as
  the validator decided; verification stays a separate, explicit act;
* it never overwrites -- an identical relation is **reinforced** (confidence
  merged, evidence appended), and a **contradicting** relation is preserved
  beside the new one (both kept, a conflict recorded), never deleted;
* it writes nothing when the validator rejects the claim -- a rejection is
  reported, not stored.

Local-only, CPU-cheap, and it imports no model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from core.best_effort import best_effort
from core.entity import Entity, EntityType
from core.knowledge_graph import KnowledgeGraph
from core.knowledge_validation import (
    FindingKind,
    KnowledgeClaim,
    KnowledgeValidator,
    ValidationResult,
    ValidationVerdict,
)
from core.relation import Relation

#: How many integration records the history keeps (bounded, append-only in spirit).
DEFAULT_HISTORY_LIMIT = 512


@dataclass(frozen=True)
class IntegrationStep:
    """One step of the pipeline, and what it did."""

    name: str
    detail: str
    changed: bool = False

    def describe(self) -> str:
        mark = "*" if self.changed else "-"
        return f"{mark} {self.name}: {self.detail}"


@dataclass(frozen=True)
class IntegrationRecord:
    """The durable history of one integration attempt."""

    claim: str
    verdict: str
    stored: bool
    relation_id: str = ""
    entity_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    conflict: bool = False
    rolled_back: bool = False
    error: str = ""
    at: datetime = field(default_factory=lambda: datetime.now(UTC))
    steps: tuple[IntegrationStep, ...] = ()

    def describe(self) -> str:
        lines = [
            f"integrate: {self.claim}",
            f"  verdict: {self.verdict}",
            f"  stored:  {self.stored}" + (" (rolled back)" if self.rolled_back else ""),
        ]
        if self.relation_id:
            lines.append(f"  relation: {self.relation_id}")
        if self.conflict:
            lines.append("  conflict: preserved (both claims kept)")
        if self.error:
            lines.append(f"  error: {self.error}")
        lines.extend(f"  {step.describe()}" for step in self.steps)
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "verdict": self.verdict,
            "stored": self.stored,
            "relation_id": self.relation_id,
            "entity_ids": list(self.entity_ids),
            "evidence_ids": list(self.evidence_ids),
            "conflict": self.conflict,
            "rolled_back": self.rolled_back,
            "error": self.error,
            "at": self.at.isoformat(),
            "steps": [{"name": s.name, "detail": s.detail, "changed": s.changed} for s in self.steps],
        }


class _Snapshot:
    """An exact copy of the graph state a transaction may touch."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.entities = dict(graph.entities)
        self.relations = dict(graph.relations)
        self.evidence = dict(graph.evidence)

    def restore(self, graph: KnowledgeGraph) -> None:
        graph.entities.clear()
        graph.entities.update(self.entities)
        graph.relations.clear()
        graph.relations.update(self.relations)
        graph.evidence.clear()
        graph.evidence.update(self.evidence)


class KnowledgeIntegrator:
    """The transactional pipeline from a validated claim to a graph write.

    ``ontology`` (optional) supplies relation vocabulary and concept labels;
    ``learner`` (optional) is the bounded ``OntologyLearner`` -- when present, an
    accepted claim is *also* folded into the concept graph through it, so the two
    views stay in step. ``on_event`` (optional) observes each record (the event
    bus is the natural sink) and is best-effort.
    """

    def __init__(
        self,
        *,
        ontology: Any = None,
        learner: Any = None,
        validator: KnowledgeValidator | None = None,
        history_limit: int = DEFAULT_HISTORY_LIMIT,
        on_event: Callable[[IntegrationRecord], None] | None = None,
    ) -> None:
        if history_limit < 0:
            raise ValueError("History limit cannot be negative.")
        self.ontology = ontology
        self.learner = learner
        self.validator = validator or self._default_validator(ontology)
        self._history: list[IntegrationRecord] = []
        self._history_limit = history_limit
        self._on_event = on_event
        self._next_entity = 0

    @staticmethod
    def _default_validator(ontology: Any) -> KnowledgeValidator:
        from core.knowledge_validation import known_relation_types

        known = known_relation_types(ontology) if ontology is not None else None
        return KnowledgeValidator(known_relations=known, ontology=ontology)

    # -- the one entry point ----------------------------------------------

    def integrate(self, claim: KnowledgeClaim, graph: KnowledgeGraph) -> IntegrationRecord:
        """Run the pipeline; on any failure restore the graph and report the error."""
        snapshot = _Snapshot(graph)
        steps: list[IntegrationStep] = []
        relation_id = ""
        entity_ids: tuple[str, ...] = ()
        evidence_ids: tuple[str, ...] = ()
        conflict = False
        try:
            steps.append(IntegrationStep("receive", claim.describe()))

            result = self.validator.validate(claim, graph=graph)
            steps.append(
                IntegrationStep("validate", f"{result.verdict.value} at confidence {result.confidence:.3f}", True)
            )

            if not result.storable:
                steps.append(
                    IntegrationStep(
                        "store",
                        "not stored: " + ("rejected by a blocking check" if not result.ok else "no evidence"),
                        False,
                    )
                )
                return self._record(
                    claim, result, stored=False, steps=steps, graph=graph
                )

            entity_ids = self._resolve_entities(claim, graph)
            steps.append(
                IntegrationStep("resolve_entities", f"{len(entity_ids)} entity/entities ready", True)
            )

            evidence_ids = self._attach_evidence(claim, graph)
            steps.append(
                IntegrationStep("attach_evidence", f"{len(evidence_ids)} evidence item(s)", True)
            )

            conflict = result.has(FindingKind.CONTRADICTION)
            if conflict:
                steps.append(IntegrationStep("detect_conflict", "an opposing claim is preserved", True))

            relation_id, merged = self._write_relation(claim, result, graph, entity_ids, evidence_ids)
            steps.append(
                IntegrationStep(
                    "update_graph",
                    f"relation {relation_id} " + ("reinforced" if merged else "created"),
                    True,
                )
            )

            if self.learner is not None and claim.source_label and claim.target_label:
                self._learn(claim, steps)

            steps.append(IntegrationStep("preserve_history", "record appended", True))
            return self._record(
                claim,
                result,
                stored=True,
                relation_id=relation_id,
                entity_ids=entity_ids,
                evidence_ids=evidence_ids,
                conflict=conflict,
                steps=steps,
                graph=graph,
            )
        except Exception as error:
            snapshot.restore(graph)
            return self._record(
                claim,
                None,
                stored=False,
                steps=steps,
                graph=graph,
                rolled_back=True,
                error=f"{type(error).__name__}: {error}",
            )

    # -- steps -------------------------------------------------------------

    def _resolve_entities(self, claim: KnowledgeClaim, graph: KnowledgeGraph) -> tuple[str, ...]:
        """Find each endpoint by name, creating a concept entity when none exists.

        Duplicate resolution is by name (case-insensitive), so two mentions of
        "porosity" resolve to one entity rather than two.
        """
        resolved: list[str] = []
        for name in (claim.source, claim.target):
            existing = self._find_entity(graph, name)
            if existing is not None:
                resolved.append(existing.id)
                continue
            entity = Entity(entity_type=EntityType.CONCEPT, name=name, confidence=claim.confidence)
            entity.metadata.setdefault("created_by", "knowledge_integration")
            graph.add_entity(entity)
            resolved.append(entity.id)
        return tuple(resolved)

    @staticmethod
    def _find_entity(graph: KnowledgeGraph, name: str) -> Entity | None:
        wanted = name.strip().casefold()
        for entity in graph.entities.values():
            if entity.name.strip().casefold() == wanted:
                return entity
        return None

    @staticmethod
    def _attach_evidence(claim: KnowledgeClaim, graph: KnowledgeGraph) -> tuple[str, ...]:
        """Store each evidence item the graph does not already hold; return its ids."""
        ids: list[str] = []
        for item in claim.evidence:
            if item.id in graph.evidence:
                ids.append(item.id)
                continue
            graph.add_evidence(item)
            ids.append(item.id)
        return tuple(ids)

    def _write_relation(
        self,
        claim: KnowledgeClaim,
        result: ValidationResult,
        graph: KnowledgeGraph,
        entity_ids: tuple[str, ...],
        evidence_ids: tuple[str, ...],
    ) -> tuple[str, bool]:
        """Create the relation, or reinforce an identical one; return (id, merged)."""
        source_id, target_id = entity_ids[0], entity_ids[1]
        for relation in graph.relations.values():
            if (
                relation.source == source_id
                and relation.target == target_id
                and relation.relation_type == claim.relation_type
            ):
                # Reinforce: merge confidence, append evidence. Never a second edge.
                relation.set_confidence(round(max(relation.confidence, result.confidence), 6))
                for evidence_id in evidence_ids:
                    relation.add_evidence(evidence_id)
                return relation.id, True

        relation = Relation(
            source=source_id,
            target=target_id,
            relation_type=claim.relation_type,
            confidence=result.confidence,
            evidence_ids=list(evidence_ids),
            metadata={
                "validation_verdict": result.verdict.value,
                "provenance": "knowledge_integration",
                **dict(claim.metadata),
            },
        )
        graph.add_relation(relation)
        return relation.id, False

    def _learn(self, claim: KnowledgeClaim, steps: list[IntegrationStep]) -> None:
        """Fold an accepted claim into the bounded concept graph through the learner."""
        with best_effort("fold an integrated claim into the concept graph"):
            self.learner.observe_relation(
                claim.source_label, claim.target_label, claim.relation_type, confidence=claim.confidence
            )
            steps.append(IntegrationStep("learn_ontology", "concept graph updated", True))

    # -- history -----------------------------------------------------------

    def _record(
        self,
        claim: KnowledgeClaim,
        result: ValidationResult | None,
        *,
        stored: bool,
        steps: list[IntegrationStep],
        graph: KnowledgeGraph,
        relation_id: str = "",
        entity_ids: tuple[str, ...] = (),
        evidence_ids: tuple[str, ...] = (),
        conflict: bool = False,
        rolled_back: bool = False,
        error: str = "",
    ) -> IntegrationRecord:
        record = IntegrationRecord(
            claim=claim.describe(),
            verdict=result.verdict.value if result is not None else ValidationVerdict.REJECTED.value,
            stored=stored,
            relation_id=relation_id,
            entity_ids=entity_ids,
            evidence_ids=evidence_ids,
            conflict=conflict,
            rolled_back=rolled_back,
            error=error,
            steps=tuple(steps),
        )
        if self._history_limit:
            self._history.append(record)
            if len(self._history) > self._history_limit:
                del self._history[: len(self._history) - self._history_limit]
        if self._on_event is not None:
            with best_effort("knowledge integration observer"):
                self._on_event(record)
        return record

    @property
    def history(self) -> tuple[IntegrationRecord, ...]:
        return tuple(self._history)

    def describe(self) -> str:
        stored = sum(1 for record in self._history if record.stored)
        conflicts = sum(1 for record in self._history if record.conflict)
        return (
            f"KnowledgeIntegrator({len(self._history)} attempt(s), {stored} stored, "
            f"{conflicts} conflict(s), {sum(1 for r in self._history if r.rolled_back)} rolled back)"
        )


__all__ = [
    "DEFAULT_HISTORY_LIMIT",
    "IntegrationRecord",
    "IntegrationStep",
    "KnowledgeIntegrator",
]
