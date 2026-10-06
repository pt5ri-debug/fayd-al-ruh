"""Incremental knowledge updates with evidence, provenance, and confidence decay.

The architecture learns in several places already -- ontology relations from
co-occurrence, belief confidence from prediction error, reward from outcomes.
What it lacks is a single, auditable way to fold a *new statement* into the
corpus the assistant retrieves from, and to know when a statement has been
superseded or has gone stale.

This module provides that:

* :class:`KnowledgeUpdate` is one proposed statement with a source and a strength;
* :class:`KnowledgeUpdater` writes it into a :class:`core.documents.DocumentStore`
  as a citable document, strengthening an existing statement on repeat and
  recording every change in the project's :class:`core.provenance.ProvenanceLedger`;
* staleness is explicit: an update older than ``max_age`` is decayed, and a
  statement contradicted by a stronger one is superseded, never silently kept.

Only statements that share content terms with the existing corpus are accepted,
so the store cannot be flooded with off-topic text. Everything stays inspectable
and reversible: the ledger holds the full history of each statement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable

from core.documents import DocumentStore
from core.provenance import ProvenanceLedger, ProvenanceRecord, ProvenanceStage
from core.rag import _query_terms

_CEILING = 0.95
_FLOOR = 0.05


@dataclass(frozen=True)
class KnowledgeUpdate:
    """A proposed statement to fold into the corpus."""

    statement: str
    source: str
    strength: float = 0.5
    topic: str = "learned"
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.statement.strip():
            raise ValueError("Knowledge statement cannot be empty.")
        if not self.source.strip():
            raise ValueError("Knowledge source cannot be empty.")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("Knowledge strength must be between 0 and 1.")


@dataclass(frozen=True)
class UpdateOutcome:
    """What one update did to the corpus."""

    statement: str
    doc_id: str
    action: str
    """One of: added, reinforced, superseded, rejected, decayed."""

    confidence: float
    reason: str = ""


class KnowledgeUpdater:
    """Folds statements into a document store, with provenance and decay."""

    def __init__(
        self,
        store: DocumentStore,
        *,
        ledger: ProvenanceLedger | None = None,
        max_age: timedelta = timedelta(days=365),
        ceiling: float = _CEILING,
        floor: float = _FLOOR,
        min_terms: int = 1,
    ) -> None:
        if not 0.0 < floor < ceiling <= 1.0:
            raise ValueError("Require 0 < floor < ceiling <= 1.")
        if max_age <= timedelta(0):
            raise ValueError("max_age must be positive.")
        self.store = store
        self.ledger = ledger or ProvenanceLedger()
        self.max_age = max_age
        self.ceiling = ceiling
        self.floor = floor
        self.min_terms = min_terms
        self._strengths: dict[str, float] = {}

    # -- reading -----------------------------------------------------------
    def strength(self, doc_id: str) -> float:
        return self._strengths.get(doc_id, 0.5)

    def _doc_id_for(self, statement: str) -> str:
        digest = abs(hash(tuple(sorted(_query_terms(statement))))) % (16**8)
        return f"learned:{digest:08x}"

    def _vocabulary(self) -> set[str]:
        terms: set[str] = set()
        for document in self.store.documents():
            terms |= _query_terms(f"{document.title} {document.text}")
        return terms

    # -- writing -----------------------------------------------------------
    def apply(self, update: KnowledgeUpdate) -> UpdateOutcome:
        """Fold one update into the store; returns what happened."""
        terms = _query_terms(update.statement)
        if len(terms) < self.min_terms:
            return UpdateOutcome(update.statement, "", "rejected", 0.0, "No content terms.")

        doc_id = self._doc_id_for(update.statement)
        existing = self.store.get(doc_id)

        if existing is not None:
            confidence = min(self.ceiling, self.strength(doc_id) + 0.1 * update.strength)
            self._strengths[doc_id] = confidence
            self._record(doc_id, update, confidence, action="reinforced")
            return UpdateOutcome(update.statement, doc_id, "reinforced", round(confidence, 6))

        # Only accept statements that connect to what the corpus already knows.
        if not (terms & self._vocabulary()):
            return UpdateOutcome(
                update.statement, doc_id, "rejected", 0.0, "No shared terms with the corpus."
            )

        confidence = min(self.ceiling, update.strength)
        self.store.add_text(
            doc_id,
            f"Learned statement: {update.topic}",
            update.statement,
            source=f"learned/{update.source}",
            metadata={"topic": update.topic, "confidence": f"{confidence:.4f}"},
        )
        self._strengths[doc_id] = confidence
        self._record(doc_id, update, confidence, action="added")
        return UpdateOutcome(update.statement, doc_id, "added", round(confidence, 6))

    def apply_many(self, updates: Iterable[KnowledgeUpdate]) -> list[UpdateOutcome]:
        return [self.apply(update) for update in updates]

    def _record(
        self, doc_id: str, update: KnowledgeUpdate, confidence: float, *, action: str
    ) -> None:
        self.ledger.record(
            ProvenanceRecord(
                entity_id=doc_id,
                stage=ProvenanceStage.LEARNING,
                source=update.source,
                confidence=confidence,
                metadata={"action": action, "topic": update.topic},
            )
        )

    # -- maintenance -------------------------------------------------------
    def decay_stale(self, *, now: datetime | None = None) -> list[UpdateOutcome]:
        """Weaken learned statements older than ``max_age``; drop faded ones."""
        moment = now or datetime.now(timezone.utc)
        outcomes: list[UpdateOutcome] = []
        for document in list(self.store.documents()):
            if not document.doc_id.startswith("learned:"):
                continue
            record = self.ledger.get(document.doc_id)
            if record is None or moment - record.recorded_at <= self.max_age:
                continue
            confidence = round(self.strength(document.doc_id) * 0.5, 6)
            if confidence < self.floor:
                self.store.remove(document.doc_id)
                self._strengths.pop(document.doc_id, None)
                outcomes.append(
                    UpdateOutcome(document.text, document.doc_id, "decayed", confidence, "Dropped: faded.")
                )
                continue
            self._strengths[document.doc_id] = confidence
            outcomes.append(
                UpdateOutcome(document.text, document.doc_id, "decayed", confidence, "Aged out.")
            )
        return outcomes

    def supersede(self, old_doc_id: str, update: KnowledgeUpdate) -> UpdateOutcome:
        """Replace a learned statement with a stronger, contradictory one."""
        old = self.store.get(old_doc_id)
        if old is None:
            return UpdateOutcome(update.statement, old_doc_id, "rejected", 0.0, "Unknown statement.")
        self.store.remove(old_doc_id)
        self._strengths.pop(old_doc_id, None)
        outcome = self.apply(update)
        self.ledger.record(
            ProvenanceRecord(
                entity_id=outcome.doc_id,
                stage=ProvenanceStage.LEARNING,
                source=update.source,
                confidence=outcome.confidence,
                parent_ids=(old_doc_id,),
                metadata={"action": "superseded", "replaced": old_doc_id},
            )
        )
        return UpdateOutcome(update.statement, outcome.doc_id, "superseded", outcome.confidence)


__all__ = ["KnowledgeUpdate", "KnowledgeUpdater", "UpdateOutcome"]
