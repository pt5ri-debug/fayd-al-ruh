"""One unified, read-only view over a character's whole memory.

A character's memory is spread across **three** stores, each written by a
different subsystem:

* :class:`core.character_learning.CharacterLearning` -- the durable facts it was
  told and the exchanges it took part in (its personality memory);
* :class:`core.character_learning_store.CharacterLearningStore` -- the
  provenance-carrying learning records the pipeline wrote;
* the character's isolated learning directory (the same store, resolved through
  the isolation policy).

Each is bounded and correct on its own, but a *reader* -- the evaluation window,
the CLI, the per-mouth chat -- has to open three things and know which one holds
what. This module is the **one** place that answers "what does this character
remember?" with a single object: :class:`CharacterMemoryView`.

Honest by construction:

* it is **read-only** -- it never writes a fact, a record, or a decay;
* it resolves every path through :mod:`core.learning_isolation`, so it can never
  be pointed at another character's store, and asking for a private character's
  records is refused exactly as every other path is;
* an unreadable store contributes nothing and is reported, never raised;
* the **semantic search** over the view uses the project's own learned
  :mod:`core.embeddings` space when it is available and falls back to a lexical
  match when it is not, so a query reaches a fact by meaning where the space can
  and by word where it cannot -- and it never invents a match.

The **decay** half (:func:`apply_fact_decay`) uses the project's own
:class:`core.memory_decay.MemoryDecay` forgetting curve to order a character's
facts by retention (frequently recalled facts survive, stale ones fade), so a
reader can rank or prune without a second formula.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from core.character_learning import (
    MAX_EXCHANGES,
    MAX_FACTS,
    CharacterLearning,
)
from core.learning_isolation import (
    assert_read,
    is_private,
    owner_for,
)

#: The number of entries one view keeps per store, so a huge character memory
#: cannot blow up a caller's transcript. Bounded like every other reader here.
DEFAULT_LIMIT = 200


@dataclass(frozen=True)
class MemoryEntry:
    """One remembered thing, from any of the three stores.

    ``kind`` names which store it came from (``fact``, ``exchange``, ``record``),
    ``text`` is the readable content, and ``provenance`` says where the knowledge
    originated (the source reference, the citations, or ``""``). ``at`` is the
    timestamp when one is known, so a decay reading has something to age.
    """

    kind: str
    text: str
    provenance: str = ""
    at: str = ""

    def describe(self) -> str:
        suffix = f" ({self.provenance})" if self.provenance else ""
        return f"[{self.kind}] {self.text}{suffix}"

    def to_primitive(self) -> dict[str, str]:
        return {"kind": self.kind, "text": self.text, "provenance": self.provenance, "at": self.at}


@dataclass(frozen=True)
class MemorySearchHit:
    """One entry matched by a query, with the signal that found it."""

    entry: MemoryEntry
    score: float
    signal: str
    """``semantic`` when the learned space matched it, ``lexical`` otherwise."""

    def describe(self) -> str:
        return f"{self.entry.describe()}  [{self.signal} {self.score:.2f}]"


@dataclass(frozen=True)
class MemoryStats:
    """An honest count of what one character's memory holds."""

    owner: str
    facts: int
    exchanges: int
    records: int
    private: bool

    @property
    def total(self) -> int:
        return self.facts + self.exchanges + self.records

    def describe(self) -> str:
        flag = " (private)" if self.private else ""
        return (
            f"{self.owner}{flag}: {self.facts}/{MAX_FACTS} facts, "
            f"{self.exchanges}/{MAX_EXCHANGES} exchanges, {self.records} records"
        )


@dataclass
class CharacterMemoryView:
    """The one read-only view over a character's whole memory.

    Built for exactly one owner: the owner is resolved from the key through the
    isolation policy and never taken from a caller, so the view has no method
    that could be pointed at another character's records. ``records`` is refused
    for a private character, exactly as the sharing policy requires.
    """

    key: str
    root: str | Path | None = None
    _entries: tuple[MemoryEntry, ...] = field(default=(), repr=False)
    _stats: MemoryStats | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.owner = owner_for(self.key)

    # -- reading ------------------------------------------------------------

    def entries(self, *, limit: int = DEFAULT_LIMIT) -> tuple[MemoryEntry, ...]:
        """Every remembered entry, facts and exchanges and records, bounded.

        Records are included only when the owner's records may be read (a private
        character's records are omitted rather than exposed), so the view is
        isolation-safe by construction.
        """
        collected: list[MemoryEntry] = []
        learning = self._learning()
        for fact in learning.facts():
            collected.append(MemoryEntry(kind="fact", text=fact, provenance="told"))
        for exchange in learning.exchanges():
            collected.append(
                MemoryEntry(
                    kind="exchange",
                    text=f"Q: {exchange.question} → A: {exchange.answer}",
                    provenance=", ".join(exchange.citations) or ("web" if exchange.used_web else ""),
                    at=exchange.at,
                )
            )
        collected.extend(self._record_entries())
        return tuple(collected[:limit])

    def stats(self) -> MemoryStats:
        """The counts of the three stores, read honestly."""
        learning = self._learning()
        return MemoryStats(
            owner=self.owner,
            facts=len(learning.facts()),
            exchanges=len(learning.exchanges()),
            records=self._record_count(),
            private=is_private(self.owner),
        )

    def describe(self) -> str:
        return self.stats().describe()

    # -- semantic search ----------------------------------------------------

    def search(self, query: str, *, limit: int = 5) -> tuple[MemorySearchHit, ...]:
        """Find the entries closest to ``query``, by meaning then by word.

        The learned :mod:`core.embeddings` space backs the semantic signal when
        it is available; a lexical (whole-word) match backs the rest, so a fact
        that shares a word is still found when the space cannot reach it. The two
        signals are combined, de-duplicated by entry, and ordered by score -- so
        the same query always yields the same hits and nothing is invented.
        """
        if not query.strip():
            raise ValueError("A memory search needs a non-empty query.")
        entries = self.entries()
        if not entries:
            return ()
        space = self._semantic_space()
        tokens = {t for t in query.lower().split() if len(t) > 2}
        scored: list[MemorySearchHit] = []
        for entry in entries:
            lexical = self._lexical_score(entry.text, tokens)
            semantic = 0.0
            if space is not None:
                try:
                    semantic = float(space.similarity(query, entry.text))
                except (KeyError, ValueError):
                    semantic = 0.0
            if semantic <= 0.0 and lexical <= 0.0:
                continue
            if semantic >= lexical and semantic > 0.0:
                signal, score = "semantic", semantic
            else:
                signal, score = "lexical", lexical
            scored.append(MemorySearchHit(entry=entry, score=round(score, 6), signal=signal))
        scored.sort(key=lambda hit: (-hit.score, hit.entry.kind, hit.entry.text))
        return tuple(scored[:limit])

    def fact_decay(self, *, rate: float = 0.01, now: datetime | None = None) -> tuple[tuple[str, float], ...]:
        """The character's facts ordered by retention, weakest first.

        Uses the project's own :class:`core.memory_decay.MemoryDecay` forgetting
        curve over a synthetic episode per fact (a fact has no access count of its
        own, so each is aged from when it was stored when a timestamp is known and
        from "now" otherwise). The result is a *ranking*, never a deletion: this
        module never drops a fact.
        """
        return apply_fact_decay(self._learning().facts(), rate=rate, now=now)

    # -- internals ----------------------------------------------------------

    def _learning(self) -> CharacterLearning:
        return CharacterLearning(self.owner, root=self.root)

    def _record_count(self) -> int:
        store = self._record_store()
        if store is None:
            return 0
        try:
            return int(store.stats().total)
        except Exception:
            return 0

    def _record_entries(self) -> list[MemoryEntry]:
        store = self._record_store()
        if store is None:
            return []
        out: list[MemoryEntry] = []
        try:
            for record in store.recent(limit=DEFAULT_LIMIT):
                text = getattr(record, "knowledge", "") or getattr(record, "answer", "")
                provenance = getattr(record, "source_reference", "") or getattr(
                    record, "source_type", ""
                )
                out.append(
                    MemoryEntry(
                        kind="record",
                        text=str(text),
                        provenance=str(provenance),
                        at=str(getattr(record, "created_at", "")),
                    )
                )
        except Exception:
            return []
        return out

    def _record_store(self):
        """The character's own record store, or ``None`` when it may not be read.

        The isolation guard runs **before** any path is resolved. The reader is
        the character itself -- a character may always read its *own* records --
        so the view shows a character its own memory while the cross-character
        and main-model rules are untouched (this module never reads another
        character's store and never lets the main model reach Natasha's).
        """
        try:
            assert_read(self.owner, self.owner)
        except Exception:
            return None
        try:
            from core.character_learning_store import CharacterLearningStore

            return CharacterLearningStore(self.owner, root=self.root)
        except Exception:
            return None

    def _semantic_space(self):
        try:
            from core.embeddings import semantic_space

            return semantic_space(build=False)
        except Exception:
            return None

    @staticmethod
    def _lexical_score(text: str, tokens: set[str]) -> float:
        if not tokens:
            return 0.0
        lowered = text.lower()
        hits = sum(1 for token in tokens if token in lowered)
        return hits / len(tokens)


def apply_fact_decay(
    facts: tuple[str, ...],
    *,
    rate: float = 0.01,
    now: datetime | None = None,
) -> tuple[tuple[str, float], ...]:
    """Order facts by the project's own forgetting curve, weakest first.

    Each fact becomes a synthetic :class:`core.episodic_memory.Episode` with unit
    importance and zero accesses, and :class:`core.memory_decay.MemoryDecay`
    scores its retention. Facts are aged from the same reference point, so the
    ranking is a *stable, deterministic* order (equal facts keep their input
    order) rather than a random shuffle. Returns ``(fact, retention)`` pairs.
    """
    from core.episodic_memory import Episode
    from core.memory_decay import MemoryDecay

    reference = now or datetime.now(UTC)
    decay = MemoryDecay()
    scored: list[tuple[str, float, int]] = []
    for index, fact in enumerate(facts):
        episode = Episode(id=f"fact-{index}", content=fact, importance=1.0, created_at=reference)
        scored.append((fact, decay.retention(episode, rate=rate, now=reference), index))
    # A stable sort by retention, then by the fact's original position, so two
    # facts with equal retention never swap places between calls.
    scored.sort(key=lambda item: (item[1], item[2]))
    return tuple((fact, score) for fact, score, _index in scored)


def memory_view(key: str, *, root: str | Path | None = None) -> CharacterMemoryView:
    """Build the unified memory view for one character (a convenience)."""
    return CharacterMemoryView(key, root=root)


def memory_view_stats(key: str, *, root: str | Path | None = None) -> MemoryStats:
    """The counts for one character, without building the full view's entries."""
    return CharacterMemoryView(key, root=root).stats()


__all__ = [
    "DEFAULT_LIMIT",
    "CharacterMemoryView",
    "MemoryEntry",
    "MemorySearchHit",
    "MemoryStats",
    "apply_fact_decay",
    "memory_view",
    "memory_view_stats",
]
