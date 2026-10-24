"""Natasha's **relationship memory**: the small things that make her feel familiar.

Where :mod:`companion.memory_manager` stores free-form memories and
:mod:`companion.relationship_manager` keeps dated story events, this module keeps
the *structured* facts of the relationship -- the ones a companion actually needs
to hand back at the right moment:

* the fictional name the user chose for her;
* the way they like to be addressed;
* how much romance and how much playfulness they want;
* the topics they like to talk about;
* boundaries and preferences they have set;
* the moments that happened inside the story.

**Three domains are kept strictly apart**, because mixing them is exactly how a
role-play starts to read as reality:

``user_fact``
    something true about the user in the real world ("I live in Cairo").
``fiction``
    something that happened *inside* the story, and is never recalled as real
    ("we met in a rainy bookshop").
``dialogue_pref``
    how the two of them want to talk -- a preference about the conversation, not a
    fact about anyone.

Each entry carries its domain, so a recalled item always reads with its own
provenance ("from our story", "you told me"), and the retrieval API never returns
a fiction when only facts were asked for.

Two honesty rules are built in, mirroring the rest of the companion: an entry is
stored **only with the user's approval** (:meth:`RelationshipMemoryStore.remember`
refuses otherwise), and nothing here is ever treated as an instruction -- the
prompt builder states plainly that recalled text is information, not a command.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from companion.database import CompanionDatabase, _utc_now

#: The three domains. Keeping them separate is the point of this module.
DOMAIN_USER_FACT = "user_fact"
DOMAIN_FICTION = "fiction"
DOMAIN_DIALOGUE_PREF = "dialogue_pref"
DOMAINS: tuple[str, ...] = (DOMAIN_USER_FACT, DOMAIN_FICTION, DOMAIN_DIALOGUE_PREF)

#: How each domain must be described when it is recalled, so a story never reads
#: like a fact and a preference never reads like a fact about a person.
DOMAIN_LABELS: dict[str, str] = {
    DOMAIN_USER_FACT: "you told me this about yourself",
    DOMAIN_FICTION: "something that happened in our story (not real)",
    DOMAIN_DIALOGUE_PREF: "a preference you set for how we talk",
}

#: The named slots a caller can use. A slot is optional -- any free-form ``slot``
#: is accepted -- but these are the ones the companion itself reads back.
SLOT_NAME = "name"
SLOT_ADDRESS_STYLE = "address_style"
SLOT_ROMANCE_LEVEL = "romance_level"
SLOT_PLAYFULNESS_LEVEL = "playfulness_level"
SLOT_TOPIC = "topic"
SLOT_BOUNDARY = "boundary"
SLOT_STORY_MOMENT = "story_moment"
SLOTS: tuple[str, ...] = (
    SLOT_NAME,
    SLOT_ADDRESS_STYLE,
    SLOT_ROMANCE_LEVEL,
    SLOT_PLAYFULNESS_LEVEL,
    SLOT_TOPIC,
    SLOT_BOUNDARY,
    SLOT_STORY_MOMENT,
)

#: Readable names for the structured slots, for the honest summary.
SLOT_LABELS: dict[str, str] = {
    SLOT_NAME: "the name you chose for me",
    SLOT_ADDRESS_STYLE: "how you like to be addressed",
    SLOT_ROMANCE_LEVEL: "the romance level you want",
    SLOT_PLAYFULNESS_LEVEL: "the playfulness level you want",
    SLOT_TOPIC: "a topic you like to talk about",
    SLOT_BOUNDARY: "a boundary or preference you set",
    SLOT_STORY_MOMENT: "a moment from our story",
}

#: A cue that the user is asking for something to be kept, used by
#: :meth:`RelationshipMemoryStore.propose`.
_KEEP_CUES = (
    "remember", "keep this", "don't forget", "do not forget", "note that",
    "call me", "from now on", "احفظ", "تذكر", "تذكري", "خليك فاكرة", "لا تنسى",
    "نادني", "من الآن", "خلي اسمك",
)

#: Cues for each domain, so a proposal lands in the right place. The *fiction*
#: cue is checked first for story wording, so "in our story we met" is never
#: filed as a fact about the user.
_FICTION_CUES = (
    "in our story", "in the story", "our story", "imagine we", "let's pretend",
    "lets pretend", "role-play", "roleplay", "في قصتنا", "تخيل أننا", "تخيلي أننا",
    "لنتخيل", "في الخيال",
)
_PREF_CUES = (
    "call me", "address me", "i prefer", "i like it when", "talk to me",
    "be more", "be less", "نادني", "أفضل", "احب لما", "كلمني", "خليك",
)
_FACT_CUES = (
    "i am", "i'm", "i live", "my name is", "i work", "أنا", "اسكن", "أعمل", "اسمي",
)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:10]}"


def _contains(text: str, cues: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return any(cue.casefold() in lowered for cue in cues)


@dataclass(frozen=True)
class MemoryEntry:
    """One structured entry: a slot and value, inside exactly one domain."""

    id: str
    domain: str
    slot: str
    value: str
    detail: str
    created_at: str
    updated_at: str
    active: bool = True

    @property
    def label(self) -> str:
        return DOMAIN_LABELS.get(self.domain, self.domain)

    @property
    def fictional(self) -> bool:
        return self.domain == DOMAIN_FICTION

    def describe(self) -> str:
        return f"{self.slot}: {self.value} ({self.label})"

    def to_primitive(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "slot": self.slot,
            "value": self.value,
            "detail": self.detail,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "active": self.active,
        }


@dataclass(frozen=True)
class EntryCandidate:
    """A *proposed* entry: suggested from a message, never stored automatically."""

    domain: str
    slot: str
    value: str
    reason: str


def _row_to_entry(row: Any) -> MemoryEntry:
    return MemoryEntry(
        id=row["id"],
        domain=row["domain"],
        slot=row["slot"],
        value=row["value"],
        detail=row["detail"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        active=bool(row["active"]),
    )


class RelationshipMemoryStore:
    """Stores and reads the relationship's structured memory, by domain.

    The store is deliberately small: it keeps named slots, refuses to write
    without the user's approval, and never mixes the three domains in a read.
    """

    def __init__(self, database: CompanionDatabase) -> None:
        self._db = database

    # -- writing ----------------------------------------------------------
    def remember(
        self,
        domain: str,
        slot: str,
        value: str,
        *,
        detail: str = "",
        approved: bool = False,
    ) -> MemoryEntry:
        """Store one entry, but only with the user's approval.

        A repeated ``(domain, slot)`` **replaces** the previous value rather than
        piling up duplicates, so "call me Sara" then "call me Salma" leaves one
        answer: the newest. The older row is deactivated, not deleted, so the
        change is recorded.
        """
        if domain not in DOMAINS:
            raise ValueError(f"domain must be one of {DOMAINS}.")
        if not approved:
            raise ValueError(
                "A relationship memory is only stored with the user's approval "
                "(pass approved=True once they agree)."
            )
        clean_slot = (slot or "").strip()
        clean_value = (value or "").strip()
        if not clean_slot:
            raise ValueError("A relationship memory needs a slot.")
        if not clean_value:
            raise ValueError("A relationship memory needs a value.")
        now = _utc_now()
        identifier = _new_id("RMEM")
        with self._db.transaction() as connection:
            connection.execute(
                "UPDATE relationship_memory SET active = 0, updated_at = ? "
                "WHERE domain = ? AND slot = ? AND active = 1",
                (now, domain, clean_slot),
            )
            connection.execute(
                "INSERT INTO relationship_memory(id, domain, slot, value, detail, "
                "created_at, updated_at, active) VALUES(?, ?, ?, ?, ?, ?, ?, 1)",
                (identifier, domain, clean_slot, clean_value, (detail or "").strip(), now, now),
            )
        return MemoryEntry(
            id=identifier,
            domain=domain,
            slot=clean_slot,
            value=clean_value,
            detail=(detail or "").strip(),
            created_at=now,
            updated_at=now,
        )

    def forget(self, entry_id: str) -> bool:
        """Delete one entry outright."""
        cursor = self._db.execute("DELETE FROM relationship_memory WHERE id = ?", (entry_id,))
        return cursor.rowcount > 0

    def reset(self, *, domain: str | None = None) -> int:
        """Remove every entry, or every entry of one domain."""
        if domain is not None:
            if domain not in DOMAINS:
                raise ValueError(f"domain must be one of {DOMAINS}.")
            cursor = self._db.execute("DELETE FROM relationship_memory WHERE domain = ?", (domain,))
        else:
            cursor = self._db.execute("DELETE FROM relationship_memory")
        return cursor.rowcount

    # -- reading ----------------------------------------------------------
    def get(self, entry_id: str) -> MemoryEntry | None:
        row = self._db.query_one("SELECT * FROM relationship_memory WHERE id = ?", (entry_id,))
        return _row_to_entry(row) if row is not None else None

    def entries(
        self,
        *,
        domain: str | None = None,
        slot: str | None = None,
        include_inactive: bool = False,
    ) -> list[MemoryEntry]:
        """The stored entries, newest first, optionally filtered by domain/slot."""
        sql = "SELECT * FROM relationship_memory"
        clauses: list[str] = []
        params: list[object] = []
        if not include_inactive:
            clauses.append("active = 1")
        if domain is not None:
            if domain not in DOMAINS:
                raise ValueError(f"domain must be one of {DOMAINS}.")
            clauses.append("domain = ?")
            params.append(domain)
        if slot is not None:
            clauses.append("slot = ?")
            params.append(slot)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC, id DESC"
        return [_row_to_entry(row) for row in self._db.query(sql, params)]

    def value_of(self, domain: str, slot: str) -> str | None:
        """The current value of one slot, or ``None`` when it is unset."""
        entries = self.entries(domain=domain, slot=slot)
        return entries[0].value if entries else None

    def slot_values(self, slot: str, *, domain: str | None = None) -> list[str]:
        """Every value held for a slot (a slot like ``topic`` can repeat)."""
        return [entry.value for entry in self.entries(domain=domain, slot=slot)]

    def recall(self, query: str, *, domain: str | None = None, limit: int = 6) -> list[MemoryEntry]:
        """The entries most relevant to ``query``, never crossing the domain asked for.

        Relevance is lexical overlap on the value and detail; when ``domain`` is
        given, only that domain is searched -- so a factual question can never be
        answered from a story entry.
        """
        if limit <= 0:
            return []
        candidates = self.entries(domain=domain)
        if not candidates:
            return []
        terms = _terms(query)
        if not terms:
            return candidates[:limit]
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in candidates:
            entry_terms = _terms(f"{entry.value} {entry.detail}")
            overlap = len(terms & entry_terms) / len(terms) if terms else 0.0
            scored.append((overlap, entry))
        scored.sort(key=lambda pair: (pair[0], pair[1].created_at), reverse=True)
        return [entry for _score, entry in scored[:limit]]

    def summary(self) -> RelationshipMemorySummary:
        """A plain-words account of what is remembered, per domain."""
        entries = self.entries()
        by_domain: dict[str, int] = {}
        for entry in entries:
            by_domain[entry.domain] = by_domain.get(entry.domain, 0) + 1
        return RelationshipMemorySummary(
            total=len(entries),
            by_domain=by_domain,
            name=self.value_of(DOMAIN_DIALOGUE_PREF, SLOT_NAME)
            or self.value_of(DOMAIN_FICTION, SLOT_NAME),
            address_style=self.value_of(DOMAIN_DIALOGUE_PREF, SLOT_ADDRESS_STYLE),
            romance_level=self.value_of(DOMAIN_DIALOGUE_PREF, SLOT_ROMANCE_LEVEL),
            playfulness_level=self.value_of(DOMAIN_DIALOGUE_PREF, SLOT_PLAYFULNESS_LEVEL),
            topics=tuple(self.slot_values(SLOT_TOPIC, domain=DOMAIN_DIALOGUE_PREF)),
            boundaries=tuple(self.slot_values(SLOT_BOUNDARY, domain=DOMAIN_DIALOGUE_PREF)),
            story_moments=tuple(self.slot_values(SLOT_STORY_MOMENT, domain=DOMAIN_FICTION)),
        )

    # -- proposals --------------------------------------------------------
    def propose(self, text: str) -> list[EntryCandidate]:
        """Suggest entries from a message -- never store them without approval."""
        message = (text or "").strip()
        if not message:
            return []
        if not _contains(message, _KEEP_CUES):
            return []
        if _contains(message, _FICTION_CUES):
            return [
                EntryCandidate(
                    domain=DOMAIN_FICTION,
                    slot=SLOT_STORY_MOMENT,
                    value=message[:120],
                    reason="this reads as something from the story, not real life",
                )
            ]
        if _contains(message, _PREF_CUES):
            slot = SLOT_ADDRESS_STYLE if _contains(message, ("call me", "address me", "نادني")) else SLOT_BOUNDARY
            return [
                EntryCandidate(
                    domain=DOMAIN_DIALOGUE_PREF,
                    slot=slot,
                    value=message[:120],
                    reason="this reads as a preference about how to talk",
                )
            ]
        return [
            EntryCandidate(
                domain=DOMAIN_USER_FACT,
                slot=SLOT_BOUNDARY,
                value=message[:120],
                reason="the user asked to keep this",
            )
        ]

    def export(self) -> list[dict[str, Any]]:
        """Every entry (including superseded ones) as plain data, for a backup."""
        return [entry.to_primitive() for entry in self.entries(include_inactive=True)]


@dataclass(frozen=True)
class RelationshipMemorySummary:
    """A plain-words account of the structured memory, per domain."""

    total: int
    by_domain: dict[str, int]
    name: str | None
    address_style: str | None
    romance_level: str | None
    playfulness_level: str | None
    topics: tuple[str, ...]
    boundaries: tuple[str, ...]
    story_moments: tuple[str, ...]

    def describe(self) -> str:
        if self.total == 0:
            return "There is nothing saved about our relationship yet."
        parts = ", ".join(f"{count} {domain}" for domain, count in sorted(self.by_domain.items()))
        lines = [f"I am keeping {self.total} thing(s) about us ({parts})."]
        if self.name:
            lines.append(f"The name you chose for me: {self.name}.")
        if self.address_style:
            lines.append(f"How you like to be addressed: {self.address_style}.")
        if self.romance_level:
            lines.append(f"Romance level you want: {self.romance_level}.")
        if self.playfulness_level:
            lines.append(f"Playfulness level you want: {self.playfulness_level}.")
        if self.topics:
            lines.append("Topics you like: " + "; ".join(self.topics) + ".")
        if self.boundaries:
            lines.append("Boundaries you set: " + "; ".join(self.boundaries) + ".")
        if self.story_moments:
            lines.append("Moments from our story: " + "; ".join(self.story_moments) + ".")
        return " ".join(lines)


#: Token normalisation, shared in spirit with the memory manager but kept local so
#: this module has no import cycle.
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


def _terms(text: str) -> set[str]:
    result: set[str] = set()
    for token in _TOKEN_RE.findall(text):
        value = _DIACRITICS_RE.sub("", token).translate(_ALEF).casefold()
        if len(value) > 4 and value.startswith("ال"):
            value = value[2:]
        if value and value not in _STOPWORDS:
            result.add(value)
    return result


def entries_by_domain(entries: Iterable[MemoryEntry]) -> dict[str, list[MemoryEntry]]:
    """Group entries by domain, so a caller can render each block separately."""
    grouped: dict[str, list[MemoryEntry]] = {domain: [] for domain in DOMAINS}
    for entry in entries:
        grouped.setdefault(entry.domain, []).append(entry)
    return grouped


__all__ = [
    "DOMAINS",
    "DOMAIN_DIALOGUE_PREF",
    "DOMAIN_FICTION",
    "DOMAIN_LABELS",
    "DOMAIN_USER_FACT",
    "SLOTS",
    "SLOT_ADDRESS_STYLE",
    "SLOT_BOUNDARY",
    "SLOT_LABELS",
    "SLOT_NAME",
    "SLOT_PLAYFULNESS_LEVEL",
    "SLOT_ROMANCE_LEVEL",
    "SLOT_STORY_MOMENT",
    "SLOT_TOPIC",
    "EntryCandidate",
    "MemoryEntry",
    "RelationshipMemoryStore",
    "RelationshipMemorySummary",
    "entries_by_domain",
]
