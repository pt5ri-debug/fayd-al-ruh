"""Natasha's durable memory: what she remembers, and how it is kept honest.

Memory is the program's job, not the model's. The companion *stores and retrieves*
memories; the Qwen3:4B model is never trusted to remember or to decide what is
true. Three distinctions are enforced here:

* **Kind** -- a memory is a ``preference``, a ``fact`` the user asked to keep, a
  ``summary`` of a conversation, or something about the ``companion`` itself.
* **Source and confidence** -- every memory records where it came from and how
  sure it is, so an uncertain inference is never silently promoted to fact.
* **Fiction vs reality** -- a memory can be marked ``fictional`` (an event that
  happened *inside* the role-play), and a fictional memory is never mixed into
  factual recall.

Nothing is stored automatically as a permanent truth: :meth:`MemoryManager.propose`
only ever *suggests*, and a memory becomes active only when it is explicitly added
(with the user's intent) or a stored one is corrected. Correcting a memory keeps
the old row as a superseded record, so the change is auditable. Retrieval scores
memories against the current message and returns only the most relevant few --
never the whole store -- and sensitive personal data is refused unless the user
clearly asks to keep it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable
from uuid import uuid4

from companion.database import CompanionDatabase, _utc_now

#: The memory kinds. Each states what it holds and how much it should be trusted.
KIND_PREFERENCE = "preference"   # the user's stated likes and dislikes
KIND_FACT = "fact"               # something the user explicitly asked to remember
KIND_SUMMARY = "summary"         # a digest of a conversation
KIND_COMPANION = "companion"     # a detail about Natasha herself
KINDS: tuple[str, ...] = (KIND_PREFERENCE, KIND_FACT, KIND_SUMMARY, KIND_COMPANION)

#: How confident a memory is by default, per kind. A summary is a paraphrase (less
#: certain); a fact the user asked to keep is certain; a preference is strong; a
#: detail about the companion is definitionally true.
DEFAULT_CONFIDENCE: dict[str, float] = {
    KIND_PREFERENCE: 0.9,
    KIND_FACT: 1.0,
    KIND_SUMMARY: 0.6,
    KIND_COMPANION: 1.0,
}

#: Marker phrases that mean "please remember this". Only these make a *fact*
#: candidate; everything else is at most a weak proposal.
_SAVE_CUES = (
    "remember", "don't forget", "do not forget", "note that", "keep in mind",
    "احفظ", "تذكر", "تذكري", "افتكر", "لا تنسى", "خلي في بالك", "خليك فاكرة",
)

#: Patterns for clearly sensitive personal data. These are *refused* as a
#: permanent fact unless the caller passes ``allow_sensitive=True``, so a
#: password, a card number, or an ID is never stored by accident.
_SENSITIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),                      # card-like digit runs
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                        # US SSN
    re.compile(r"\b\d{10,}\b"),                                  # long ID/number
    re.compile(r"\b(?:password|passcode|pin|otp|cvv|secret)\b\s*[:=]?\s*\S+", re.IGNORECASE),
    re.compile(r"\b(?:api[_ -]?key|token)\b\s*[:=]?\s*\S+", re.IGNORECASE),
    re.compile(r"(كلمة\s*السر|الرقم\s*السري|رمز\s*التحقق|رقم\s*البطاقة)\s*[:=]?\s*\S+"),
)


def looks_sensitive(text: str) -> bool:
    """True when ``text`` looks like a secret or an identifying number."""
    return any(pattern.search(text) for pattern in _SENSITIVE_PATTERNS)


def is_save_request(text: str) -> bool:
    """True when the user explicitly asked for something to be remembered."""
    lowered = text.casefold()
    return any(cue.casefold() in lowered for cue in _SAVE_CUES)


_TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+")
_DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670\u0640]")
_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "of", "to", "in", "on", "and", "or", "for",
        "with", "that", "this", "it", "you", "i", "me", "my", "your",
        "في", "من", "على", "إلى", "عن", "هو", "هي", "التي", "الذي", "و", "أو", "مع",
        "هذا", "هذه", "ذلك", "انا", "أنا", "انت", "أنت", "لي", "لك", "ما", "هل",
    }
)


def _normalize_token(token: str) -> str:
    value = _DIACRITICS_RE.sub("", token).translate(_ALEF).casefold()
    if len(value) > 4 and value.startswith("ال"):
        value = value[2:]
    return value


def _terms(text: str) -> set[str]:
    return {_normalize_token(t) for t in _TOKEN_RE.findall(text)} - _STOPWORDS


@dataclass(frozen=True)
class Memory:
    """One stored memory."""

    id: str
    kind: str
    content: str
    source: str
    confidence: float
    created_at: str
    updated_at: str
    superseded_by: str | None = None
    active: bool = True
    fictional: bool = False

    @property
    def citation(self) -> str:
        tag = "fiction" if self.fictional else "fact"
        return f"[{self.kind}/{tag}:{self.id}] {self.content}"

    def to_primitive(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "content": self.content,
            "source": self.source,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "superseded_by": self.superseded_by,
            "active": self.active,
            "fictional": self.fictional,
        }


@dataclass(frozen=True)
class MemoryCandidate:
    """A *proposed* memory: suggested, not yet stored, so it can be reviewed."""

    kind: str
    content: str
    confidence: float
    reason: str


def _row_to_memory(row: Any) -> Memory:
    return Memory(
        id=row["id"],
        kind=row["kind"],
        content=row["content"],
        source=row["source"],
        confidence=float(row["confidence"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        superseded_by=row["superseded_by"],
        active=bool(row["active"]),
        fictional=bool(row["fictional"]),
    )


class MemoryManager:
    """Adds, retrieves, corrects, and deletes Natasha's memories."""

    def __init__(self, database: CompanionDatabase) -> None:
        self._db = database

    # -- writing ----------------------------------------------------------
    def add(
        self,
        content: str,
        *,
        kind: str = KIND_FACT,
        source: str = "user",
        confidence: float | None = None,
        fictional: bool = False,
        allow_sensitive: bool = False,
    ) -> Memory:
        """Store a memory.

        A sensitive-looking value is refused unless ``allow_sensitive=True``, so a
        password or an ID is never kept by accident. ``confidence`` defaults to the
        value for the kind, and is clamped to ``[0, 1]``.
        """
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}.")
        text = (content or "").strip()
        if not text:
            raise ValueError("A memory cannot be empty.")
        if looks_sensitive(text) and not allow_sensitive:
            raise ValueError(
                "This looks like sensitive personal data (a secret or an ID number). "
                "It is not stored unless you clearly ask to keep it."
            )
        value = DEFAULT_CONFIDENCE[kind] if confidence is None else max(0.0, min(1.0, float(confidence)))
        identifier = _new_id("MEM")
        now = _utc_now()
        self._db.execute(
            "INSERT INTO memories(id, kind, content, source, confidence, created_at, updated_at, "
            "superseded_by, active, fictional) VALUES(?, ?, ?, ?, ?, ?, ?, NULL, 1, ?)",
            (identifier, kind, text, source, value, now, now, 1 if fictional else 0),
        )
        return Memory(
            id=identifier,
            kind=kind,
            content=text,
            source=source,
            confidence=value,
            created_at=now,
            updated_at=now,
            fictional=fictional,
        )

    def correct(
        self,
        memory_id: str,
        new_content: str,
        *,
        source: str = "user-correction",
        confidence: float | None = None,
        allow_sensitive: bool = False,
    ) -> Memory:
        """Correct a memory: the newest correction wins, the old row is kept.

        The superseded row is deactivated (not deleted) and points at its
        replacement, so the change is recorded rather than erased.
        """
        original = self.get(memory_id)
        if original is None:
            raise KeyError(f"Unknown memory: {memory_id}")
        text = (new_content or "").strip()
        if not text:
            raise ValueError("A correction cannot be empty.")
        if looks_sensitive(text) and not allow_sensitive:
            raise ValueError(
                "This correction looks like sensitive personal data; it is not stored "
                "unless you clearly ask to keep it."
            )
        value = original.confidence if confidence is None else max(0.0, min(1.0, float(confidence)))
        identifier = _new_id("MEM")
        now = _utc_now()
        with self._db.transaction() as connection:
            connection.execute(
                "INSERT INTO memories(id, kind, content, source, confidence, created_at, updated_at, "
                "superseded_by, active, fictional) VALUES(?, ?, ?, ?, ?, ?, ?, NULL, 1, ?)",
                (identifier, original.kind, text, source, value, now, now, 1 if original.fictional else 0),
            )
            connection.execute(
                "UPDATE memories SET active = 0, superseded_by = ?, updated_at = ? WHERE id = ?",
                (identifier, now, memory_id),
            )
        return Memory(
            id=identifier,
            kind=original.kind,
            content=text,
            source=source,
            confidence=value,
            created_at=now,
            updated_at=now,
            fictional=original.fictional,
        )

    def update(self, memory_id: str, *, content: str | None = None, confidence: float | None = None) -> Memory:
        """Edit a memory in place (its own record, not a new correction)."""
        original = self.get(memory_id)
        if original is None:
            raise KeyError(f"Unknown memory: {memory_id}")
        new_text = original.content if content is None else (content or "").strip()
        if not new_text:
            raise ValueError("A memory cannot be emptied.")
        new_confidence = (
            original.confidence if confidence is None else max(0.0, min(1.0, float(confidence)))
        )
        self._db.execute(
            "UPDATE memories SET content = ?, confidence = ?, updated_at = ? WHERE id = ?",
            (new_text, new_confidence, _utc_now(), memory_id),
        )
        return Memory(
            id=original.id,
            kind=original.kind,
            content=new_text,
            source=original.source,
            confidence=new_confidence,
            created_at=original.created_at,
            updated_at=_utc_now(),
            superseded_by=original.superseded_by,
            active=original.active,
            fictional=original.fictional,
        )

    def delete(self, memory_id: str) -> bool:
        """Delete a memory (and any records that superseded it)."""
        with self._db.transaction() as connection:
            connection.execute("UPDATE memories SET superseded_by = NULL WHERE superseded_by = ?", (memory_id,))
            cursor = connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return cursor.rowcount > 0

    # -- reading ----------------------------------------------------------
    def get(self, memory_id: str) -> Memory | None:
        row = self._db.query_one("SELECT * FROM memories WHERE id = ?", (memory_id,))
        return _row_to_memory(row) if row is not None else None

    def all(self, *, include_inactive: bool = False) -> list[Memory]:
        """Every memory, newest first; inactive (superseded) rows are hidden by default."""
        sql = "SELECT * FROM memories"
        if not include_inactive:
            sql += " WHERE active = 1"
        sql += " ORDER BY created_at DESC, id DESC"
        return [_row_to_memory(row) for row in self._db.query(sql)]

    def by_kind(self, kind: str, *, include_inactive: bool = False) -> list[Memory]:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}.")
        sql = "SELECT * FROM memories WHERE kind = ?"
        params: list[object] = [kind]
        if not include_inactive:
            sql += " AND active = 1"
        sql += " ORDER BY created_at DESC, id DESC"
        return [_row_to_memory(row) for row in self._db.query(sql, params)]

    def retrieve(
        self,
        query: str,
        *,
        limit: int = 6,
        fictional: bool = False,
        kinds: Iterable[str] | None = None,
    ) -> list[Memory]:
        """The memories most relevant to ``query``, newest and most relevant first.

        Relevance is lexical overlap (Arabic-normalised) plus the memory's own
        confidence; only ``limit`` are returned, so the whole store is never loaded
        into a prompt. Fictional memories are returned only when ``fictional=True``,
        so in-dialogue events are never mixed into factual recall.
        """
        if limit <= 0:
            return []
        wanted_kinds = set(kinds) if kinds else None
        rows = self._db.query("SELECT * FROM memories WHERE active = 1")
        query_terms = _terms(query)
        scored: list[tuple[float, Memory]] = []
        for row in rows:
            memory = _row_to_memory(row)
            if memory.fictional != fictional:
                continue
            if wanted_kinds is not None and memory.kind not in wanted_kinds:
                continue
            memory_terms = _terms(memory.content)
            overlap = len(query_terms & memory_terms) / len(query_terms) if query_terms else 0.0
            score = overlap * 0.8 + memory.confidence * 0.2
            # A memory that shares nothing with the question is still allowed at a
            # low score, so a fresh session can recall the most confident facts
            # even before the user's wording overlaps them.
            scored.append((score, memory))
        scored.sort(key=lambda pair: (pair[0], pair[1].created_at), reverse=True)
        return [memory for _score, memory in scored[:limit]]

    # -- proposals --------------------------------------------------------
    def propose(self, text: str) -> list[MemoryCandidate]:
        """Suggest memories from a user message -- never store them.

        Only an explicit "remember this" yields a strong ``fact`` candidate; a
        self-description ("I like ...", "أنا أحب ...") yields a weak ``preference``
        candidate. The caller (or the user) decides what to keep.
        """
        message = (text or "").strip()
        if not message:
            return []
        candidates: list[MemoryCandidate] = []
        if is_save_request(message):
            candidates.append(
                MemoryCandidate(
                    kind=KIND_FACT,
                    content=message,
                    confidence=DEFAULT_CONFIDENCE[KIND_FACT],
                    reason="the user explicitly asked to remember this",
                )
            )
        lowered = message.casefold()
        for marker in ("i like", "i love", "i prefer", "أنا أحب", "بحب", "أفضل", "مفضل"):
            if marker in lowered:
                candidates.append(
                    MemoryCandidate(
                        kind=KIND_PREFERENCE,
                        content=message,
                        confidence=0.6,
                        reason="the user described a preference",
                    )
                )
                break
        return candidates

    # -- maintenance ------------------------------------------------------
    def count(self, *, include_inactive: bool = False) -> int:
        sql = "SELECT COUNT(*) AS n FROM memories"
        if not include_inactive:
            sql += " WHERE active = 1"
        row = self._db.query_one(sql)
        return int(row["n"]) if row is not None else 0

    def export(self) -> list[dict[str, Any]]:
        """Every memory (including superseded ones) as plain data, for a backup."""
        return [memory.to_primitive() for memory in self.all(include_inactive=True)]

    def delete_all(self) -> int:
        """Remove every memory. Used by the explicit "reset memories" action."""
        cursor = self._db.execute("DELETE FROM memories")
        return cursor.rowcount


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:10]}"


__all__ = [
    "DEFAULT_CONFIDENCE",
    "KIND_COMPANION",
    "KIND_FACT",
    "KIND_PREFERENCE",
    "KIND_SUMMARY",
    "KINDS",
    "Memory",
    "MemoryCandidate",
    "MemoryManager",
    "is_save_request",
    "looks_sensitive",
]
