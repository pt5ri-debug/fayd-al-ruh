"""One learning record: what a character (or the main model) learned, and how.

A *learning record* is the unit the whole independent-learning system is built
from. It is deliberately explicit about provenance, because the honesty rule the
task states is that **no language-model answer and no internet result is ever
treated as verified truth**: a record carries where it came from, how sure the
system is, and what has been done to it.

Five source types are distinguished (:data:`SOURCE_TYPES`):

* ``language_model`` -- an answer the character's own language model produced;
* ``internet`` -- information obtained from the web (a governed search);
* ``user`` -- something the user stated;
* ``character`` -- knowledge acquired from another character through the
  controlled sharing interface (only the main model does this);
* ``system`` -- a conclusion the system itself drew (an inference, a summary).

Six learning statuses (:data:`LEARNING_STATUSES`) describe the record's life:
``pending`` (not yet checked), ``provisional`` (accepted but not verified),
``verified``, ``rejected``, ``contradicted`` (it disagrees with something already
held), and ``superseded`` (a later record replaced it -- the old one is kept, so
a revised conclusion keeps its reason and its predecessor).

Nothing here reads or writes a file: this module is the *shape*, so it is
dependency-free, testable on its own, and safe to import from anywhere.
:mod:`core.character_learning_store` is the store that persists these records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

#: The five kinds of source a record can come from, in the order a report shows
#: them. ``character`` is only ever produced by the main model's sharing path.
SOURCE_LANGUAGE_MODEL = "language_model"
SOURCE_INTERNET = "internet"
SOURCE_USER = "user"
SOURCE_CHARACTER = "character"
SOURCE_SYSTEM = "system"

SOURCE_TYPES: tuple[str, ...] = (
    SOURCE_LANGUAGE_MODEL,
    SOURCE_INTERNET,
    SOURCE_USER,
    SOURCE_CHARACTER,
    SOURCE_SYSTEM,
)

#: Readable labels, so a report never shows a bare identifier.
SOURCE_LABELS: dict[str, str] = {
    SOURCE_LANGUAGE_MODEL: "the language model",
    SOURCE_INTERNET: "the internet",
    SOURCE_USER: "the user",
    SOURCE_CHARACTER: "another character",
    SOURCE_SYSTEM: "the system's own reasoning",
}

#: The six learning statuses, from least to most settled.
STATUS_PENDING = "pending"
STATUS_PROVISIONAL = "provisional"
STATUS_VERIFIED = "verified"
STATUS_REJECTED = "rejected"
STATUS_CONTRADICTED = "contradicted"
STATUS_SUPERSEDED = "superseded"

LEARNING_STATUSES: tuple[str, ...] = (
    STATUS_PENDING,
    STATUS_PROVISIONAL,
    STATUS_VERIFIED,
    STATUS_REJECTED,
    STATUS_CONTRADICTED,
    STATUS_SUPERSEDED,
)

#: Readable labels for the statuses.
STATUS_LABELS: dict[str, str] = {
    STATUS_PENDING: "pending verification",
    STATUS_PROVISIONAL: "accepted provisionally",
    STATUS_VERIFIED: "verified",
    STATUS_REJECTED: "rejected",
    STATUS_CONTRADICTED: "contradicted",
    STATUS_SUPERSEDED: "superseded",
}

#: The statuses that count as "this is knowledge we may act on". A ``pending``,
#: ``rejected``, ``contradicted``, or ``superseded`` record is *kept* -- history is
#: never deleted -- but it is not treated as knowledge.
TRUSTED_STATUSES: frozenset[str] = frozenset({STATUS_PROVISIONAL, STATUS_VERIFIED})

#: How many statements one record keeps, and how long its free text may be. A
#: bounded record keeps a store's size predictable without dropping provenance.
MAX_STATEMENTS = 24
MAX_TEXT_CHARS = 4000
MAX_EVIDENCE = 32


def _now() -> str:
    """An ISO-8601 UTC timestamp, so a record's time is unambiguous."""
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class LearningRecord:
    """One thing learned, with everything needed to judge and to revise it.

    Every field the task requires is here: a unique id, the owner (a character key
    or the main model), the source type and reference, the question/context, the
    original answer, the extracted knowledge, a timestamp, a confidence, a
    verification status, evidence references, and a learning status. The record is
    **frozen**: a revision is a *new* record, so the original is never mutated and
    a superseded record keeps pointing at what replaced it.
    """

    record_id: str
    owner: str
    source_type: str
    source_reference: str
    question: str
    answer: str
    knowledge: str
    created_at: str = field(default_factory=_now)
    confidence: float = 0.0
    verification: str = STATUS_PENDING
    learning_status: str = STATUS_PENDING
    statements: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    #: Provenance for a record acquired from another character (the main model's
    #: sharing path). ``source_character`` is the character's key, ``source_date``
    #: the source record's own timestamp, and ``source_record_id`` its id -- so the
    #: main model's copy can always be traced back to exactly what it came from.
    source_character: str = ""
    source_record_id: str = ""
    source_date: str = ""
    #: The id of the record this one supersedes, when it is a revision.
    supersedes: str = ""
    #: Why the record is in its current status (a contradiction, a rejection
    #: reason, a revision note). Kept so a revised conclusion is explained.
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise ValueError("A learning record needs a record id.")
        if not self.owner.strip():
            raise ValueError("A learning record needs an owner.")
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"Unknown source type {self.source_type!r}.")
        if self.verification not in LEARNING_STATUSES:
            raise ValueError(f"Unknown verification status {self.verification!r}.")
        if self.learning_status not in LEARNING_STATUSES:
            raise ValueError(f"Unknown learning status {self.learning_status!r}.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be in [0, 1].")

    # ---------------------------------------------------------------- honesty

    @property
    def trusted(self) -> bool:
        """True when the record counts as knowledge (provisional or verified).

        A ``pending``, ``rejected``, ``contradicted``, or ``superseded`` record is
        still stored and still readable, but it is **not** knowledge.
        """
        return self.learning_status in TRUSTED_STATUSES

    def describe(self) -> str:
        """One honest line naming what was learned and where it came from."""
        origin = SOURCE_LABELS.get(self.source_type, self.source_type)
        status = STATUS_LABELS.get(self.learning_status, self.learning_status)
        if self.source_type == SOURCE_CHARACTER and self.source_character:
            origin = f"the character {self.source_character}"
        return f"[{status}] {self.knowledge or self.answer} (from {origin}, {self.confidence:.2f})"

    # ------------------------------------------------------------ serialisation

    def to_primitive(self) -> dict[str, Any]:
        """The record as plain JSON-able data, for the append-only store."""
        return {
            "record_id": self.record_id,
            "owner": self.owner,
            "source_type": self.source_type,
            "source_reference": self.source_reference,
            "question": self.question,
            "answer": self.answer,
            "knowledge": self.knowledge,
            "created_at": self.created_at,
            "confidence": self.confidence,
            "verification": self.verification,
            "learning_status": self.learning_status,
            "statements": list(self.statements),
            "evidence": list(self.evidence),
            "source_character": self.source_character,
            "source_record_id": self.source_record_id,
            "source_date": self.source_date,
            "supersedes": self.supersedes,
            "note": self.note,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_primitive(cls, data: dict[str, Any]) -> LearningRecord:
        """Rebuild a record from stored data, tolerating a missing optional field.

        A field a future (or older) version did not write falls back to its
        default rather than raising, so a store written by one build is readable by
        the next.
        """
        return cls(
            record_id=str(data.get("record_id") or f"LR-{uuid4().hex[:12]}"),
            owner=str(data.get("owner") or ""),
            source_type=str(data.get("source_type") or SOURCE_SYSTEM),
            source_reference=str(data.get("source_reference") or ""),
            question=str(data.get("question") or ""),
            answer=str(data.get("answer") or ""),
            knowledge=str(data.get("knowledge") or ""),
            created_at=str(data.get("created_at") or _now()),
            confidence=float(data.get("confidence") or 0.0),
            verification=str(data.get("verification") or STATUS_PENDING),
            learning_status=str(data.get("learning_status") or STATUS_PENDING),
            statements=tuple(str(item) for item in data.get("statements", ()) or ()),
            evidence=tuple(str(item) for item in data.get("evidence", ()) or ()),
            source_character=str(data.get("source_character") or ""),
            source_record_id=str(data.get("source_record_id") or ""),
            source_date=str(data.get("source_date") or ""),
            supersedes=str(data.get("supersedes") or ""),
            note=str(data.get("note") or ""),
            metadata=dict(data.get("metadata") or {}),
        )

    def with_status(
        self,
        status: str,
        *,
        note: str = "",
        confidence: float | None = None,
    ) -> LearningRecord:
        """A **new** record with a different status (the original is untouched).

        Revising a record never edits it in place: the store appends the new
        version, so the history of a conclusion -- what it was, why it changed --
        is preserved exactly as the task requires.
        """
        from dataclasses import replace

        changes: dict[str, Any] = {"learning_status": status, "note": note}
        if status == STATUS_VERIFIED:
            changes["verification"] = STATUS_VERIFIED
        if confidence is not None:
            changes["confidence"] = confidence
        return replace(self, **changes)


def new_record(
    owner: str,
    *,
    source_type: str,
    source_reference: str = "",
    question: str = "",
    answer: str = "",
    knowledge: str = "",
    confidence: float = 0.0,
    verification: str = STATUS_PENDING,
    learning_status: str = STATUS_PENDING,
    statements: tuple[str, ...] = (),
    evidence: tuple[str, ...] = (),
    source_character: str = "",
    source_record_id: str = "",
    source_date: str = "",
    supersedes: str = "",
    note: str = "",
    metadata: dict[str, Any] | None = None,
) -> LearningRecord:
    """Build a new record with a fresh id and bounded text.

    The id is ``LR-<hex>`` so it is stable, unique, and obviously a learning
    record; the free-text fields are capped so one runaway answer cannot bloat the
    store.
    """
    return LearningRecord(
        record_id=f"LR-{uuid4().hex[:12]}",
        owner=owner,
        source_type=source_type,
        source_reference=source_reference[:MAX_TEXT_CHARS],
        question=question[:MAX_TEXT_CHARS],
        answer=answer[:MAX_TEXT_CHARS],
        knowledge=knowledge[:MAX_TEXT_CHARS],
        confidence=max(0.0, min(1.0, float(confidence))),
        verification=verification,
        learning_status=learning_status,
        statements=tuple(item[:MAX_TEXT_CHARS] for item in statements[:MAX_STATEMENTS]),
        evidence=tuple(str(item) for item in evidence[:MAX_EVIDENCE]),
        source_character=source_character,
        source_record_id=source_record_id,
        source_date=source_date,
        supersedes=supersedes,
        note=note,
        metadata=dict(metadata or {}),
    )


__all__ = [
    "LEARNING_STATUSES",
    "MAX_EVIDENCE",
    "MAX_STATEMENTS",
    "MAX_TEXT_CHARS",
    "SOURCE_CHARACTER",
    "SOURCE_INTERNET",
    "SOURCE_LABELS",
    "SOURCE_LANGUAGE_MODEL",
    "SOURCE_SYSTEM",
    "SOURCE_TYPES",
    "SOURCE_USER",
    "STATUS_CONTRADICTED",
    "STATUS_LABELS",
    "STATUS_PENDING",
    "STATUS_PROVISIONAL",
    "STATUS_REJECTED",
    "STATUS_SUPERSEDED",
    "STATUS_VERIFIED",
    "TRUSTED_STATUSES",
    "LearningRecord",
    "new_record",
]
