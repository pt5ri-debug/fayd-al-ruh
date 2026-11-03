"""Cognitive knowledge learning: a new information item, from source to graph.

Phase 3 requires a *controlled* process for acquiring and integrating knowledge:
receive an item, identify its provenance, normalise it, extract entities and
claims, find duplicates, weigh the evidence, detect contradictions, integrate
what is accepted into the Phase-1 Cognitive Core, preserve what is disputed, and
record the learning event.

This module is that process. It **reuses** the Phase-1 services rather than
re-implementing them:

* :class:`~core.knowledge_validation.KnowledgeValidator` decides a claim's
  verdict (structure, semantics, evidence, contradiction, temporal);
* :class:`~core.knowledge_integration.KnowledgeIntegrator` writes an accepted
  claim into the graph **transactionally**, and preserves a conflicting claim
  beside the one it opposes;
* :class:`~core.relation_extraction.RelationExtractor` reads the typed relations a
  sentence states, so the claims come from the text rather than from a caller's
  guess.

**Honest by construction.** Generated text is never treated as verified fact: a
claim with no evidence is *needs-review* and never integrated, a fresh learning
event is never marked trusted, an unsupported claim is reported rather than
stored, and a contradiction is **preserved** (both claims kept) rather than
silently resolved. Duplicate entities are resolved by name through the
integrator, and duplicate claims reinforce the existing edge instead of adding a
second one. Nothing here loads a model or reaches the network.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from core.evidence import Evidence
from core.knowledge_graph import KnowledgeGraph
from core.knowledge_integration import IntegrationRecord, KnowledgeIntegrator
from core.knowledge_validation import KnowledgeClaim, KnowledgeValidator, ValidationResult
from core.learning_events import (
    KIND_KNOWLEDGE,
    OUTCOME_ACCEPTED,
    OUTCOME_CONFLICTED,
    OUTCOME_REJECTED,
    SOURCE_SYSTEM,
    LearningEvent,
)
from core.learning_store import LearningStore

#: A fixed, documented reliability weight per provenance kind. A user's own
#: statement and a document outrank a model's guess; generated text is the
#: weakest, which is what keeps "the model said so" from becoming fact.
SOURCE_RELIABILITY: dict[str, float] = {
    "user": 0.9,
    "document": 0.85,
    "import": 0.7,
    "internet": 0.6,
    "character": 0.6,
    "system": 0.5,
    "reasoning": 0.4,
    "model": 0.3,
}

#: The evidence strength a source contributes (independent of the claim's own
#: stated confidence, so the two are never conflated).
SOURCE_STRENGTH: dict[str, float] = {
    "user": 0.9,
    "document": 0.85,
    "import": 0.7,
    "internet": 0.6,
    "character": 0.6,
    "system": 0.5,
    "reasoning": 0.4,
    "model": 0.3,
}

#: A claim below this confidence is stored as a memory only, never integrated.
DEFAULT_MIN_INTEGRATION_CONFIDENCE = 0.4


@dataclass
class KnowledgeLearningResult:
    """What one information item taught, with every claim's outcome."""

    target: str
    source: str
    source_reference: str = ""
    claims: tuple[KnowledgeClaim, ...] = ()
    records: tuple[IntegrationRecord, ...] = ()
    events: tuple[LearningEvent, ...] = ()
    duplicates: int = 0
    conflicts: int = 0
    rejected: int = 0
    reason: str = ""

    @property
    def integrated(self) -> int:
        """How many claims were actually written to the graph."""
        return sum(1 for record in self.records if record.stored)

    @property
    def changed(self) -> bool:
        return self.integrated > 0

    def describe(self) -> str:
        return (
            f"knowledge from {self.source}"
            f"{' (' + self.source_reference + ')' if self.source_reference else ''}: "
            f"{len(self.claims)} claim(s), {self.integrated} integrated, "
            f"{self.duplicates} duplicate(s), {self.conflicts} conflict(s), "
            f"{self.rejected} rejected"
            + (f"; {self.reason}" if self.reason else "")
        )


def source_reliability(source: str) -> float:
    """The documented reliability weight of a provenance kind (0.3 default)."""
    return SOURCE_RELIABILITY.get(source, 0.3)


def source_strength(source: str) -> float:
    """The evidence strength a provenance kind contributes."""
    return SOURCE_STRENGTH.get(source, 0.3)


def _evidence_for(source: str, reference: str, statement: str) -> Evidence:
    """Build the evidence record one information item contributes.

    The evidence carries the **source** and the **reference** so the claim's
    provenance survives into the graph, and its strength is the source's own
    documented weight -- never the claim's stated confidence.
    """
    description = statement.strip() or "learning source"
    return Evidence(
        description=description[:200],
        source=reference or source,
        supports=True,
        strength=source_strength(source),
        data={"provenance": source},
    )


class KnowledgeLearningPipeline:
    """The one controlled path from an information item to the knowledge graph."""

    def __init__(
        self,
        graph: KnowledgeGraph,
        *,
        validator: KnowledgeValidator | None = None,
        integrator: KnowledgeIntegrator | None = None,
        ontology: Any = None,
        learner: Any = None,
        store: LearningStore | None = None,
        owner: str = "fayd_al_ruh",
        min_confidence: float = DEFAULT_MIN_INTEGRATION_CONFIDENCE,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("Minimum integration confidence must be in [0, 1].")
        self.graph = graph
        self.ontology = ontology
        self.learner = learner
        self.owner = owner
        self.min_confidence = min_confidence
        self.validator = validator or KnowledgeValidator(ontology=ontology)
        self.integrator = integrator or KnowledgeIntegrator(
            ontology=ontology, learner=learner, validator=self.validator
        )
        self.store = store

    # -- the one entry point ----------------------------------------------
    def learn(
        self,
        text: str,
        *,
        source: str = SOURCE_SYSTEM,
        source_reference: str = "",
        confidence: float = 0.5,
        claims: Iterable[KnowledgeClaim] | None = None,
        record: bool = True,
    ) -> KnowledgeLearningResult:
        """Learn one information item: extract, validate, integrate, record.

        ``claims`` may be supplied directly (a caller that already parsed the
        item); otherwise the relations the text states are read with the project's
        own :class:`~core.relation_extraction.RelationExtractor`, so the claims
        come from the item's own words.
        """
        statement = text.strip()
        if not statement:
            raise ValueError("A knowledge item cannot be empty.")
        extracted = tuple(claims) if claims is not None else self._claims_from_text(
            statement, source=source, reference=source_reference, confidence=confidence
        )

        records: list[IntegrationRecord] = []
        events: list[LearningEvent] = []
        duplicates = 0
        conflicts = 0
        rejected = 0

        for claim in extracted:
            duplicate = self._is_duplicate(claim)
            validation = self.validator.validate(claim, graph=self.graph)
            outcome, applied = self._integrate(claim, validation, records)
            if outcome == OUTCOME_ACCEPTED and applied and duplicate:
                duplicates += 1
            if outcome == OUTCOME_CONFLICTED:
                conflicts += 1
            if outcome == OUTCOME_REJECTED:
                rejected += 1
            if record:
                events.append(self._event(claim, validation, outcome, applied, source, source_reference))

        if record and self.store is not None and events:
            version = self.store.next_version() if any(event.applied for event in events) else self.store.version
            stamped = tuple(
                event if not event.applied else _with_version(event, str(version)) for event in events
            )
            self.store.record_many(stamped)
            events = list(stamped)

        return KnowledgeLearningResult(
            target=statement[:120],
            source=source,
            source_reference=source_reference,
            claims=extracted,
            records=tuple(records),
            events=tuple(events),
            duplicates=duplicates,
            conflicts=conflicts,
            rejected=rejected,
        )

    def learn_claim(
        self,
        claim: KnowledgeClaim,
        *,
        source: str | None = None,
        source_reference: str = "",
        record: bool = True,
    ) -> KnowledgeLearningResult:
        """Learn one already-structured claim through the same controlled path."""
        source_kind = source or str(claim.metadata.get("source_kind", SOURCE_SYSTEM))
        return self.learn(
            claim.describe(),
            source=source_kind,
            source_reference=source_reference,
            confidence=claim.confidence,
            claims=(claim,),
            record=record,
        )

    # -- internals ---------------------------------------------------------
    def _claims_from_text(
        self, text: str, *, source: str, reference: str, confidence: float
    ) -> tuple[KnowledgeClaim, ...]:
        """Read the typed relations the text states, as claims to validate."""
        if self.ontology is None:
            return ()
        from core.relation_extraction import RelationExtractor

        extraction = RelationExtractor(self.ontology).extract(text)
        evidence = _evidence_for(source, reference, text)
        return tuple(
            KnowledgeClaim(
                source=relation.source,
                relation_type=relation.relation_type,
                target=relation.target,
                evidence=(evidence,),
                confidence=confidence,
                source_label=relation.source,
                target_label=relation.target,
                metadata={"pattern": relation.pattern, "sentence": relation.sentence, "source_kind": source},
            )
            for relation in extraction.relations
        )

    def _integrate(
        self,
        claim: KnowledgeClaim,
        validation: ValidationResult,
        records: list[IntegrationRecord],
    ) -> tuple[str, bool]:
        """Integrate one claim, mapping the validation verdict to an outcome.

        The rule the project already enforces is reused rather than re-stated: a
        claim is written when ``validation.storable`` holds -- an *unsupported*
        claim is never storable, while a genuine **conflict** (both claims carry
        evidence) is stored flagged and reported as conflicted.
        """
        if not validation.storable or validation.confidence < self.min_confidence:
            return OUTCOME_REJECTED, False
        record = self.integrator.integrate(claim, self.graph)
        records.append(record)
        if record.conflict:
            return OUTCOME_CONFLICTED, record.stored
        if record.stored:
            return OUTCOME_ACCEPTED, True
        return OUTCOME_REJECTED, False

    def _is_duplicate(self, claim: KnowledgeClaim) -> bool:
        """True when the graph already held this exact typed relation."""
        wanted = claim.triple
        for relation in self.graph.relations.values():
            source = self.graph.entities.get(relation.source)
            target = self.graph.entities.get(relation.target)
            if source is None or target is None:
                continue
            if (
                source.name.strip().casefold() == claim.source.strip().casefold()
                and target.name.strip().casefold() == claim.target.strip().casefold()
                and relation.relation_type == wanted[1]
            ):
                return True
        return False

    def _event(
        self,
        claim: KnowledgeClaim,
        validation: ValidationResult,
        outcome: str,
        applied: bool,
        source: str,
        reference: str,
    ) -> LearningEvent:
        return LearningEvent(
            kind=KIND_KNOWLEDGE,
            target=claim.describe(),
            owner=self.owner,
            source=source,
            source_reference=reference,
            summary=f"{validation.verdict.value} at confidence {validation.confidence:.3f}",
            after=claim.describe() if applied else "",
            outcome=outcome,
            applied=applied,
            confidence=validation.confidence,
        )

    # -- reporting ---------------------------------------------------------
    def history(self) -> tuple[LearningEvent, ...]:
        """The knowledge-learning events this pipeline has recorded."""
        if self.store is None:
            return ()
        return tuple(self.store.iter_events(kind=KIND_KNOWLEDGE))

    def describe(self) -> str:
        stored = len(self.history())
        return (
            f"knowledge learning for {self.owner}: {len(self.graph.relations)} relation(s) "
            f"in the graph, {stored} learning event(s) recorded"
        )


def _with_version(event: LearningEvent, version: str) -> LearningEvent:
    """A copy of ``event`` stamped with a version (frozen records need a copy)."""
    from dataclasses import replace

    return replace(event, version=version)


def claim_from_relation(
    source: str,
    relation_type: str,
    target: str,
    *,
    evidence: Iterable[Evidence] = (),
    confidence: float = 0.5,
    source_kind: str = SOURCE_SYSTEM,
    reference: str = "",
) -> KnowledgeClaim:
    """Build a claim from a triple, attaching the source's own evidence."""
    items = tuple(evidence)
    if not items:
        items = (_evidence_for(source_kind, reference, f"{source} {relation_type} {target}"),)
    return KnowledgeClaim(
        source=source,
        relation_type=relation_type,
        target=target,
        evidence=items,
        confidence=confidence,
        metadata={"source_kind": source_kind},
    )


__all__ = [
    "DEFAULT_MIN_INTEGRATION_CONFIDENCE",
    "SOURCE_RELIABILITY",
    "SOURCE_STRENGTH",
    "KnowledgeLearningPipeline",
    "KnowledgeLearningResult",
    "claim_from_relation",
    "source_reliability",
    "source_strength",
]
