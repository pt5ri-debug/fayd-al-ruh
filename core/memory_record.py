"""The one record contract every long-term memory category shares.

Phase 2 gives the project four memory categories -- working, episodic, semantic,
and procedural -- and a single durable store behind them. This module is the
*shape* of a memory, so the four stores, the retrieval layer, the management
layer, and the gate all agree on what a memory is without each inventing its own
dictionary.

**Honest by construction.** A memory always carries its **provenance** (where it
came from), its **confidence** (how sure the system is), its **verification**
status (whether a human or a process confirmed it -- a fresh record is never
verified), and its **temporal validity** (from when, until when). A memory is
never deleted to resolve a conflict: a correction writes a *new* record that
`supersedes` the old one, and the old one is kept and marked `superseded_by`.
Nothing here opens a file or loads a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from hashlib import blake2b
from typing import Any

from core.tokenize import normalize_phrase, tokenize


class MemoryKind(str, Enum):
    """The four memory categories the task requires.

    ``WORKING`` is the short-lived context of the current task; ``EPISODIC`` a
    timestamped experience; ``SEMANTIC`` a durable fact bound to the knowledge
    core; ``PROCEDURAL`` a reusable procedure. A genuine string enum, so
    ``str(MemoryKind.EPISODIC)`` is ``"MemoryKind.EPISODIC"`` and a persisted
    value round-trips.
    """

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


#: The durable categories -- the ones the store keeps across a session. Working
#: memory is runtime state and is deliberately **not** persisted by default.
DURABLE_KINDS: tuple[MemoryKind, ...] = (
    MemoryKind.EPISODIC,
    MemoryKind.SEMANTIC,
    MemoryKind.PROCEDURAL,
)

#: Verification statuses, in increasing strength. A record is born
#: ``unverified`` and only an explicit act reaches a stronger status.
VERIFICATION_UNVERIFIED = "unverified"
VERIFICATION_PROVISIONAL = "provisional"
VERIFICATION_VERIFIED = "verified"
VERIFICATION_CONTRADICTED = "contradicted"
VERIFICATION_SUPERSEDED = "superseded"

#: The statuses that count as *trusted enough to reason with*.
TRUSTED_VERIFICATIONS: frozenset[str] = frozenset(
    {VERIFICATION_PROVISIONAL, VERIFICATION_VERIFIED}
)

VERIFICATIONS: tuple[str, ...] = (
    VERIFICATION_UNVERIFIED,
    VERIFICATION_PROVISIONAL,
    VERIFICATION_VERIFIED,
    VERIFICATION_CONTRADICTED,
    VERIFICATION_SUPERSEDED,
)

#: Where a memory came from. A closed set, so provenance is comparable.
SOURCE_USER = "user"
SOURCE_MODEL = "model"
SOURCE_SYSTEM = "system"
SOURCE_INTERNET = "internet"
SOURCE_DOCUMENT = "document"
SOURCE_CHARACTER = "character"
SOURCE_REASONING = "reasoning"
SOURCE_IMPORT = "import"

SOURCES: tuple[str, ...] = (
    SOURCE_USER,
    SOURCE_MODEL,
    SOURCE_SYSTEM,
    SOURCE_INTERNET,
    SOURCE_DOCUMENT,
    SOURCE_CHARACTER,
    SOURCE_REASONING,
    SOURCE_IMPORT,
)


def now_utc() -> datetime:
    """The current time, timezone-aware, so every timestamp is comparable."""
    return datetime.now(UTC)


def _aware(moment: datetime | None) -> datetime | None:
    """Return ``moment`` as timezone-aware UTC, or ``None``.

    A naive timestamp is read as UTC rather than silently compared against an
    aware one (which raises). This keeps a hand-written or legacy record usable.
    """
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def make_memory_id(kind: MemoryKind | str, content: str, *, owner: str = "") -> str:
    """A stable, content-addressed memory id.

    The id is a ``blake2b`` digest of the category, the owner, and the normalised
    content, so the *same* memory captured twice yields the same id -- which is
    what lets duplicate detection find it without a second index. It is a name,
    never a claim about truth.
    """
    kind_value = kind.value if isinstance(kind, MemoryKind) else str(kind)
    seed = f"{kind_value}\x1f{normalize_phrase(owner)}\x1f{normalize_phrase(content)}"
    digest = blake2b(seed.encode("utf-8"), digest_size=12).hexdigest()
    return f"mem:{kind_value}:{digest}"


@dataclass(frozen=True)
class MemoryRecord:
    """One memory, of one category, with its full provenance and validity.

    Frozen on purpose: a "correction" is a **new** record (see
    :func:`supersede`), never a mutation of this one, so the original and its
    history are always preserved.
    """

    record_id: str
    kind: MemoryKind
    content: str
    summary: str = ""
    owner: str = "fayd_al_ruh"
    source: str = SOURCE_SYSTEM
    source_reference: str = ""
    confidence: float = 0.5
    importance: float = 0.5
    verification: str = VERIFICATION_UNVERIFIED
    created_at: datetime = field(default_factory=now_utc)
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    access_count: int = 0
    last_accessed: datetime | None = None
    superseded_by: str = ""
    supersedes: str = ""
    archived: bool = False
    entities: tuple[str, ...] = ()
    relations: tuple[tuple[str, str, str], ...] = ()
    evidence_ids: tuple[str, ...] = ()
    task_refs: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise ValueError("A memory record needs a non-empty id.")
        if not isinstance(self.kind, MemoryKind):
            object.__setattr__(self, "kind", MemoryKind(self.kind))
        if not self.content.strip():
            raise ValueError("A memory record needs non-empty content.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Memory confidence must be between 0 and 1.")
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("Memory importance must be between 0 and 1.")
        if self.verification not in VERIFICATIONS:
            raise ValueError(f"Unknown verification status {self.verification!r}.")
        if self.source not in SOURCES:
            raise ValueError(f"Unknown memory source {self.source!r}.")
        # Normalise the timestamps so a comparison never raises.
        object.__setattr__(self, "created_at", _aware(self.created_at) or now_utc())
        object.__setattr__(self, "valid_from", _aware(self.valid_from))
        object.__setattr__(self, "valid_until", _aware(self.valid_until))
        object.__setattr__(self, "last_accessed", _aware(self.last_accessed))

    # -- derived facts ------------------------------------------------------
    @property
    def trusted(self) -> bool:
        """True only when the record's verification status is a trusted one."""
        return self.verification in TRUSTED_VERIFICATIONS

    @property
    def active(self) -> bool:
        """True when the record is neither archived nor superseded."""
        return not self.archived and self.verification != VERIFICATION_SUPERSEDED

    def is_valid_at(self, moment: datetime | None = None) -> bool:
        """Whether the record's temporal window covers ``moment``.

        An open end (``valid_until`` unset) means "still valid"; a record with no
        window at all is valid at every moment. A moment before ``valid_from`` is
        *not* covered -- a fact is not valid before it began.
        """
        when = _aware(moment) or now_utc()
        start = _aware(self.valid_from)
        end = _aware(self.valid_until)
        if start is not None and when < start:
            return False
        if end is not None and when > end:
            return False
        return True

    @property
    def terms(self) -> set[str]:
        """The content terms of the record, for lexical retrieval."""
        return tokenize(f"{self.summary} {self.content}")

    def describe(self) -> str:
        """A one-line, honest description: kind, owner, source, status."""
        return (
            f"[{self.kind.value}] {self.summary or self.content[:60]} "
            f"(owner={self.owner}, source={self.source}, "
            f"status={self.verification}, confidence={self.confidence:.2f})"
        )

    # -- serialisation ------------------------------------------------------
    def to_primitive(self) -> dict[str, Any]:
        """A JSON-safe dictionary (used by the durable store and export)."""
        return {
            "record_id": self.record_id,
            "kind": self.kind.value,
            "content": self.content,
            "summary": self.summary,
            "owner": self.owner,
            "source": self.source,
            "source_reference": self.source_reference,
            "confidence": self.confidence,
            "importance": self.importance,
            "verification": self.verification,
            "created_at": self.created_at.isoformat(),
            "valid_from": self.valid_from.isoformat() if self.valid_from else "",
            "valid_until": self.valid_until.isoformat() if self.valid_until else "",
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else "",
            "superseded_by": self.superseded_by,
            "supersedes": self.supersedes,
            "archived": self.archived,
            "entities": list(self.entities),
            "relations": [list(relation) for relation in self.relations],
            "evidence_ids": list(self.evidence_ids),
            "task_refs": list(self.task_refs),
            "tags": list(self.tags),
            "metadata": self.metadata,
        }

    @classmethod
    def from_primitive(cls, data: dict[str, Any]) -> MemoryRecord:
        """Rebuild a record from :meth:`to_primitive` output.

        Raises ``KeyError``/``ValueError`` on a malformed row so the caller (the
        store) can skip and count it rather than crash.
        """
        relations_raw = data.get("relations") or []
        relations = tuple(
            (str(relation[0]), str(relation[1]), str(relation[2]))
            for relation in relations_raw
            if isinstance(relation, (list, tuple)) and len(relation) == 3
        )
        return cls(
            record_id=str(data["record_id"]),
            kind=MemoryKind(str(data["kind"])),
            content=str(data["content"]),
            summary=str(data.get("summary", "")),
            owner=str(data.get("owner", "fayd_al_ruh")),
            source=str(data.get("source", SOURCE_SYSTEM)),
            source_reference=str(data.get("source_reference", "")),
            confidence=float(data.get("confidence", 0.5)),
            importance=float(data.get("importance", 0.5)),
            verification=str(data.get("verification", VERIFICATION_UNVERIFIED)),
            created_at=_parse_time(data.get("created_at")) or now_utc(),
            valid_from=_parse_time(data.get("valid_from")),
            valid_until=_parse_time(data.get("valid_until")),
            access_count=int(data.get("access_count", 0)),
            last_accessed=_parse_time(data.get("last_accessed")),
            superseded_by=str(data.get("superseded_by", "")),
            supersedes=str(data.get("supersedes", "")),
            archived=bool(data.get("archived", False)),
            entities=tuple(str(item) for item in (data.get("entities") or [])),
            relations=relations,
            evidence_ids=tuple(str(item) for item in (data.get("evidence_ids") or [])),
            task_refs=tuple(str(item) for item in (data.get("task_refs") or [])),
            tags=tuple(str(item) for item in (data.get("tags") or [])),
            metadata=dict(data.get("metadata") or {}),
        )


def _parse_time(value: Any) -> datetime | None:
    """Parse an ISO timestamp, returning ``None`` for an empty or bad value."""
    if not value:
        return None
    if isinstance(value, datetime):
        return _aware(value)
    try:
        return _aware(datetime.fromisoformat(str(value)))
    except ValueError:
        return None


def supersede(
    old: MemoryRecord,
    new_content: str,
    *,
    new_id: str | None = None,
    confidence: float | None = None,
    verification: str = VERIFICATION_PROVISIONAL,
    source: str | None = None,
    reason: str = "",
) -> tuple[MemoryRecord, MemoryRecord]:
    """Return ``(superseded_old, new_record)`` for a correction.

    A correction is never an overwrite: the old record is returned with
    ``verification='superseded'`` and ``superseded_by`` pointing at the new one,
    and the new record carries ``supersedes`` back to it. Both are kept, so the
    history is preserved and the correction is auditable.
    """
    record_id = new_id or make_memory_id(old.kind, new_content, owner=old.owner)
    if record_id == old.record_id:
        raise ValueError("A correction must produce a new record id.")
    metadata = dict(old.metadata)
    if reason:
        metadata["correction_reason"] = reason
    new_record = replace(
        old,
        record_id=record_id,
        content=new_content,
        summary="",
        confidence=old.confidence if confidence is None else confidence,
        verification=verification,
        source=old.source if source is None else source,
        created_at=now_utc(),
        access_count=0,
        last_accessed=None,
        superseded_by="",
        supersedes=old.record_id,
        archived=False,
        metadata=metadata,
    )
    superseded_old = replace(
        old,
        verification=VERIFICATION_SUPERSEDED,
        superseded_by=record_id,
        archived=True,
    )
    return superseded_old, new_record


def content_tokens(record: MemoryRecord) -> set[str]:
    """The record's searchable terms (kept as a function for a stable import)."""
    return record.terms


_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def term_list(text: str) -> list[str]:
    """A deterministic, ordered term list -- used to build a search index key."""
    return _TOKEN_RE.findall(normalize_phrase(text))


__all__ = [
    "DURABLE_KINDS",
    "SOURCES",
    "SOURCE_CHARACTER",
    "SOURCE_DOCUMENT",
    "SOURCE_IMPORT",
    "SOURCE_INTERNET",
    "SOURCE_MODEL",
    "SOURCE_REASONING",
    "SOURCE_SYSTEM",
    "SOURCE_USER",
    "TRUSTED_VERIFICATIONS",
    "VERIFICATIONS",
    "VERIFICATION_CONTRADICTED",
    "VERIFICATION_PROVISIONAL",
    "VERIFICATION_SUPERSEDED",
    "VERIFICATION_UNVERIFIED",
    "VERIFICATION_VERIFIED",
    "MemoryKind",
    "MemoryRecord",
    "content_tokens",
    "make_memory_id",
    "now_utc",
    "supersede",
    "term_list",
]
