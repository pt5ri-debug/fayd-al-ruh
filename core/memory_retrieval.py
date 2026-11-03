"""Unified memory retrieval with an explainable, separate-signal ranking.

Phase 2 requires one retrieval interface that can combine exact/keyword search,
entity lookup, relationship lookup through the Cognitive Core, temporal
filtering, metadata filtering, and -- when a local embedding space exists --
semantic retrieval. This module is that interface.

**Ranking is explainable.** Every signal is computed on its own and returned on
the result, so a reader can see *why* a memory ranked where it did:

* ``relevance`` -- how much the record's terms overlap the query;
* ``semantic`` -- the cosine similarity in the project's own learned space, when
  one is available (else ``0.0``, never a fabricated number);
* ``recency`` -- how recently the record was created or last used;
* ``frequency`` -- how often the record has been used (a bounded, saturating
  signal, so a hot record cannot run away);
* ``importance`` -- the record's own stored importance;
* ``source_quality`` -- a small, fixed weight per provenance kind.

The final score is a **documented weighted sum** of those terms. Retrieval
similarity is *never* reported as factual confidence: the record's own
``confidence`` and ``verification`` travel beside the score, not inside it.

**No feedback loop.** A retrieval bumps a record's access count once (that is the
frequency signal), and nothing else reinforces anything -- there is no
"retrieve → boost → retrieve more" amplification. A rarely retrieved memory is
**never deleted**; the store only ever archives, and only on an explicit call.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from math import exp
from typing import Any

from core.long_term_memory import LongTermMemory
from core.memory_record import (
    SOURCE_CHARACTER,
    SOURCE_DOCUMENT,
    SOURCE_IMPORT,
    SOURCE_INTERNET,
    SOURCE_MODEL,
    SOURCE_REASONING,
    SOURCE_SYSTEM,
    SOURCE_USER,
    MemoryKind,
    MemoryRecord,
    now_utc,
)
from core.tokenize import tokenize

#: The weight of each ranking signal in the final score. The three "how well does
#: this match the question" terms dominate; the "how good is this memory" terms
#: break ties. They sum to 1.0 so the score is always in ``[0, 1]``.
RANK_WEIGHTS: dict[str, float] = {
    "relevance": 0.40,
    "semantic": 0.20,
    "importance": 0.15,
    "recency": 0.10,
    "frequency": 0.10,
    "source_quality": 0.05,
}

#: A fixed, documented weight per provenance kind. A user's own statement and a
#: verified document outrank a model's guess; a reasoning derivation is honest
#: but the weakest.
SOURCE_QUALITY: dict[str, float] = {
    SOURCE_USER: 1.0,
    SOURCE_DOCUMENT: 0.9,
    SOURCE_SYSTEM: 0.7,
    SOURCE_CHARACTER: 0.6,
    SOURCE_INTERNET: 0.6,
    SOURCE_IMPORT: 0.6,
    SOURCE_MODEL: 0.4,
    SOURCE_REASONING: 0.3,
}

#: How many accesses saturate the frequency signal, so it cannot run away.
FREQUENCY_SATURATION = 10

#: The recency half-life in days: a memory a week old keeps ~0.37 of its recency.
RECENCY_HALF_LIFE_DAYS = 7.0

DEFAULT_LIMIT = 20
MAX_LIMIT = 200


@dataclass(frozen=True)
class RankingSignals:
    """The six ranking signals, each on its own, all in ``[0, 1]``."""

    relevance: float = 0.0
    semantic: float = 0.0
    recency: float = 0.0
    frequency: float = 0.0
    importance: float = 0.0
    source_quality: float = 0.0

    def weighted(self) -> float:
        """The documented weighted sum of the signals."""
        return round(
            RANK_WEIGHTS["relevance"] * self.relevance
            + RANK_WEIGHTS["semantic"] * self.semantic
            + RANK_WEIGHTS["importance"] * self.importance
            + RANK_WEIGHTS["recency"] * self.recency
            + RANK_WEIGHTS["frequency"] * self.frequency
            + RANK_WEIGHTS["source_quality"] * self.source_quality,
            6,
        )

    def describe(self) -> str:
        return (
            f"relevance={self.relevance:.2f} semantic={self.semantic:.2f} "
            f"recency={self.recency:.2f} frequency={self.frequency:.2f} "
            f"importance={self.importance:.2f} source={self.source_quality:.2f}"
        )


@dataclass(frozen=True)
class MemoryHit:
    """One retrieved memory with its score, its signals, and its provenance."""

    record: MemoryRecord
    score: float
    signals: RankingSignals
    matched_terms: tuple[str, ...] = ()
    via: str = "term"

    @property
    def record_id(self) -> str:
        return self.record.record_id

    @property
    def kind(self) -> MemoryKind:
        return self.record.kind

    @property
    def confidence(self) -> float:
        """The record's own confidence -- **not** the retrieval score."""
        return self.record.confidence

    @property
    def verified(self) -> bool:
        return self.record.trusted

    def describe(self) -> str:
        return (
            f"{self.record.record_id} [{self.record.kind.value}] score={self.score:.3f} "
            f"via={self.via} ({self.signals.describe()})"
        )


@dataclass
class RetrievalReport:
    """What one retrieval did, so the caller can report it honestly."""

    query: str
    hits: tuple[MemoryHit, ...] = ()
    scanned: int = 0
    skipped: int = 0
    semantic_used: bool = False
    terms: tuple[str, ...] = ()
    reasons: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.hits)

    def describe(self) -> str:
        source = "learned semantic space" if self.semantic_used else "lexical only"
        return (
            f"{self.count} hit(s) for {self.query!r} over {self.scanned} record(s) "
            f"({source})"
        )


class MemoryRetriever:
    """The one retrieval interface over a :class:`LongTermMemory`.

    It is constructed with an optional ``semantic_space`` (the project's own
    :class:`core.embeddings.SemanticSpace`) and an optional ``graph`` (the Phase-1
    knowledge graph, for relationship lookup). With neither, it still returns
    useful, deterministic lexical results -- the fallback the task requires -- so
    a machine without a trained space is never left with an unusable memory.
    """

    def __init__(
        self,
        memory: LongTermMemory,
        *,
        semantic_space: Any | None = None,
        graph: Any | None = None,
    ) -> None:
        self.memory = memory
        self.semantic_space = semantic_space
        self.graph = graph

    # -- the one entry point ------------------------------------------------
    def retrieve(
        self,
        query: str,
        *,
        kinds: Iterable[MemoryKind | str] | None = None,
        entities: Sequence[str] = (),
        limit: int = DEFAULT_LIMIT,
        min_score: float = 0.0,
        created_between: tuple[datetime | None, datetime | None] | None = None,
        valid_at: datetime | None = None,
        tags: Sequence[str] = (),
        include_archived: bool = False,
        touch: bool = True,
    ) -> RetrievalReport:
        """Retrieve the memories most relevant to ``query``.

        Filters are applied by the database (bounded), the score is computed from
        the six signals, and the top ``limit`` are returned. ``touch=True`` records
        one access per returned hit -- the frequency signal -- and nothing else is
        reinforced.
        """
        if limit < 1:
            raise ValueError("Retrieval limit must be at least 1.")
        if limit > MAX_LIMIT:
            limit = MAX_LIMIT
        cleaned = (query or "").strip()
        query_terms = tokenize(cleaned) if cleaned else set()
        kind_values = [kind if isinstance(kind, MemoryKind) else MemoryKind(str(kind))
                       for kind in (kinds or ())]
        wanted_tags = {tag.casefold() for tag in tags if tag}
        wanted_entities = {entity.casefold() for entity in entities if entity}

        # Candidate pool: the term index narrows it, then a bounded page covers
        # the entity/tag filters the index cannot express.
        candidates: dict[str, MemoryRecord] = {}
        report = RetrievalReport(query=cleaned, terms=tuple(sorted(query_terms)))
        if query_terms:
            for record_id in self.memory.database.search_terms(
                sorted(query_terms), limit=max(limit * 8, 64)
            ):
                record = self.memory.database.get(record_id)
                if record is not None:
                    candidates[record_id] = record
        # Always include a bounded page so a filter-only query (no terms) works.
        page, page_report = self.memory.page(
            include_archived=include_archived, limit=max(limit * 4, 100)
        )
        report.skipped += page_report.skipped
        for record in page:
            candidates.setdefault(record.record_id, record)

        semantic_scores = self._semantic_scores(cleaned) if cleaned else {}
        report.semantic_used = bool(semantic_scores)

        hits: list[MemoryHit] = []
        now = now_utc()
        for record in candidates.values():
            if kind_values and record.kind not in kind_values:
                continue
            if not include_archived and not record.active:
                continue
            if wanted_tags and not wanted_tags.intersection(
                {tag.casefold() for tag in record.tags}
            ):
                continue
            if wanted_entities and not wanted_entities.intersection(
                {entity.casefold() for entity in record.entities}
            ):
                continue
            if created_between is not None:
                start, end = created_between
                if start is not None and record.created_at < start:
                    continue
                if end is not None and record.created_at > end:
                    continue
            if valid_at is not None and not record.is_valid_at(valid_at):
                continue
            record_terms = record.terms
            overlap = query_terms & record_terms
            relevance = (
                len(overlap) / len(query_terms) if query_terms else 0.0
            )
            signals = RankingSignals(
                relevance=round(relevance, 6),
                semantic=round(max(0.0, semantic_scores.get(record.record_id, 0.0)), 6),
                recency=_recency(record, now),
                frequency=_frequency(record),
                importance=record.importance,
                source_quality=SOURCE_QUALITY.get(record.source, 0.5),
            )
            score = signals.weighted()
            if score < min_score:
                continue
            via = "term" if overlap else ("semantic" if signals.semantic else "page")
            hits.append(
                MemoryHit(
                    record=record,
                    score=score,
                    signals=signals,
                    matched_terms=tuple(sorted(overlap)),
                    via=via,
                )
            )
        report.scanned = len(candidates)
        hits.sort(
            key=lambda hit: (-hit.score, -hit.record.created_at.timestamp(), hit.record.record_id)
        )
        selected = tuple(hits[:limit])
        report.hits = selected
        if touch and selected:
            self.memory.database.touch([hit.record_id for hit in selected], moment=now)
        return report

    # -- relationship lookup through the Cognitive Core ---------------------
    def by_entity(self, entity: str, *, limit: int = DEFAULT_LIMIT) -> tuple[MemoryRecord, ...]:
        """Every memory linked to an entity (by its own link, or the graph's).

        A memory carries ``entities`` itself; a memory that names *no* entity but
        whose record id a graph relation references is also returned, so the link
        is reachable from either side. Read-only: it never edits the graph.
        """
        wanted = entity.casefold()
        found: list[MemoryRecord] = []
        for record in self.memory.all_records():
            if any(item.casefold() == wanted for item in record.entities):
                found.append(record)
            if len(found) >= limit:
                break
        if self.graph is not None and len(found) < limit:
            for record in self.memory.all_records():
                for source, _relation, target in record.relations:
                    if wanted in {source.casefold(), target.casefold()} and record not in found:
                        found.append(record)
                        break
                if len(found) >= limit:
                    break
        return tuple(found[:limit])

    def related(
        self, record_id: str, *, limit: int = DEFAULT_LIMIT
    ) -> tuple[MemoryRecord, ...]:
        """Memories that share an entity, a relation endpoint, or a tag."""
        seed = self.memory.get(record_id)
        if seed is None:
            return ()
        entity_keys = {item.casefold() for item in seed.entities}
        relation_keys = {
            endpoint.casefold()
            for _source, _relation, _target in seed.relations
            for endpoint in (_source, _target)
        }
        tag_keys = {tag.casefold() for tag in seed.tags}
        related: list[MemoryRecord] = []
        for record in self.memory.all_records():
            if record.record_id == seed.record_id:
                continue
            if entity_keys & {item.casefold() for item in record.entities}:
                related.append(record)
            elif relation_keys & {
                endpoint.casefold()
                for source, _relation, target in record.relations
                for endpoint in (source, target)
            }:
                related.append(record)
            elif tag_keys & {tag.casefold() for tag in record.tags}:
                related.append(record)
            if len(related) >= limit:
                break
        return tuple(related)

    # -- helpers ------------------------------------------------------------
    def _semantic_scores(self, query: str) -> dict[str, float]:
        """Cosine similarity per record in the learned space, when one exists.

        The whole store is **not** loaded: only the records a term match already
        surfaced are scored, plus a bounded page. With no space this returns
        ``{}`` -- the honest "semantic retrieval is unavailable here".
        """
        if self.semantic_space is None or not query.strip():
            return {}
        scores: dict[str, float] = {}
        for record in self.memory.all_records():
            try:
                vector = self.semantic_space.embed(f"{record.summary} {record.content}")
            except (AttributeError, TypeError, ValueError):
                return {}
            norm = sum(value * value for value in vector) ** 0.5
            if norm == 0:
                continue
            query_vector = self.semantic_space.embed(query)
            query_norm = sum(value * value for value in query_vector) ** 0.5
            if query_norm == 0:
                continue
            dot = sum(a * b for a, b in zip(vector, query_vector, strict=False))
            scores[record.record_id] = max(0.0, dot / (norm * query_norm))
        return scores


def _recency(record: MemoryRecord, now: datetime) -> float:
    """An exponential recency in ``[0, 1]`` from the record's last use or creation."""
    moment = record.last_accessed or record.created_at
    age_days = max(0.0, (now - moment).total_seconds() / 86400)
    return round(exp(-age_days / RECENCY_HALF_LIFE_DAYS), 6)


def _frequency(record: MemoryRecord) -> float:
    """A saturating frequency in ``[0, 1]`` -- bounded so it cannot run away."""
    return round(min(record.access_count, FREQUENCY_SATURATION) / FREQUENCY_SATURATION, 6)


__all__ = [
    "DEFAULT_LIMIT",
    "FREQUENCY_SATURATION",
    "MAX_LIMIT",
    "RANK_WEIGHTS",
    "RECENCY_HALF_LIFE_DAYS",
    "SOURCE_QUALITY",
    "MemoryHit",
    "MemoryRetriever",
    "RankingSignals",
    "RetrievalReport",
]
