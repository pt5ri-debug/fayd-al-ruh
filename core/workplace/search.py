"""Cross-entity search for the Workplace (the brief's §14 "advanced search").

The Workplace holds twenty-odd record kinds across one SQLite file. Phase 10 adds
**one** search entry point over them, so a reader can ask *"where does API 1104
appear?"* and reach the standard, the procedure that cites it, the project that
applies it, and the inspection that witnessed it -- rather than opening twenty
lists in turn.

It is deliberately a **view**, never a second index: it reads the same repository
every other Workplace surface reads (the brief's §11 rule -- no parallel knowledge
base), and it adds no table and no migration. The scoring is the project's own
tokenizer (:mod:`core.tokenize`), so an Arabic query and an English one are
compared in the same normalised space the retriever uses.

**Honest by construction.** A hit is a *real* row with the field that matched named
(``matched``), so nothing is fabricated; an empty query returns nothing rather than
every row; a record kind the caller cannot read is skipped (the clearance filter is
applied by the caller); and the score is a coverage figure (how much of the query a
row's own text carries), never a claim that the row *is* the answer.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from core.tokenize import content_terms
from core.workplace.records import DOMAIN_RECORDS
from core.workplace.repository import WorkplaceRepository

#: The text fields searched, per record kind, in the order they are shown.
#: A kind not listed here is searched over its string fields generically, so a
#: newly added record is searchable without editing this table.
_SEARCH_FIELDS: dict[str, tuple[str, ...]] = {
    "people": ("name", "name_ar", "role", "discipline", "summary"),
    "qualifications": ("title", "issuer", "reference"),
    "clients": ("name", "name_ar", "contact", "summary"),
    "projects": ("number", "title", "title_ar", "scope", "location"),
    "contracts": ("number", "title", "terms"),
    "standards": ("body", "number", "title", "edition", "revision", "applicability"),
    "procedures": ("number", "title", "revision", "summary", "body"),
    "inspections": ("title", "itp_reference", "result"),
    "tests": ("kind", "title", "standard_reference", "value", "unit"),
    "workplace_documents": ("title", "title_ar", "path"),
    "reports": ("number", "title", "summary", "body"),
    "correspondence": ("subject", "body"),
    "assignments": ("role", "note"),
    "findings": ("number", "title", "description"),
    "knowledge_records": ("title", "summary"),
    "workplace_users": ("name",),
}

#: The readable name of each searchable kind (falls back to the table name).
_KIND_LABELS: dict[str, str] = {
    "people": "Personnel",
    "qualifications": "Qualification",
    "clients": "Client / contractor",
    "projects": "Project",
    "contracts": "Contract",
    "standards": "Standard",
    "procedures": "Procedure",
    "inspections": "Inspection",
    "tests": "Test",
    "workplace_documents": "Document",
    "reports": "Report",
    "correspondence": "Correspondence",
    "assignments": "Assignment",
    "findings": "Finding / NCR",
    "knowledge_records": "Knowledge record",
    "workplace_users": "User",
}


@dataclass(frozen=True)
class SearchHit:
    """One record a query matched, with the field that matched named.

    ``score`` is the share of the query's content terms the row's own text carries
    (``0 < score <= 1``), so a row that carries the whole query ranks above one that
    carries a single word -- and the number is a *coverage* figure, never a claim
    that the row answers the question.
    """

    kind: str
    record_id: str
    title: str
    matched: tuple[str, ...] = ()
    score: float = 0.0

    @property
    def kind_label(self) -> str:
        return _KIND_LABELS.get(self.kind, self.kind)

    def describe(self, *, language: str = "en") -> str:
        label = self.kind_label
        matched = f" — matched: {', '.join(self.matched)}" if self.matched else ""
        return f"[{label}] {self.title}{matched}"


@dataclass
class SearchReport:
    """The result of one Workplace search: the hits, and what was searched."""

    query: str = ""
    hits: list[SearchHit] = field(default_factory=list)
    searched_kinds: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return not self.hits

    @property
    def count(self) -> int:
        return len(self.hits)

    def by_kind(self) -> dict[str, list[SearchHit]]:
        grouped: dict[str, list[SearchHit]] = {}
        for hit in self.hits:
            grouped.setdefault(hit.kind, []).append(hit)
        return grouped

    def describe(self, *, language: str = "en") -> str:
        if not self.query.strip():
            return "Search: enter a word or phrase."
        if self.empty:
            return f"Search: no record matches “{self.query}”."
        kinds = len(self.by_kind())
        return (
            f"Search: {self.count} record(s) across {kinds} kind(s) match "
            f"“{self.query}”."
        )


def _fields_for(kind: str) -> tuple[str, ...]:
    """The text fields searched for ``kind`` (its declared set, else its strings)."""
    declared = _SEARCH_FIELDS.get(kind)
    if declared is not None:
        return declared
    record = DOMAIN_RECORDS.get(kind)
    if record is None:
        return ()
    names: list[str] = []
    for spec in dataclasses.fields(record):
        if spec.name in {"record_id", "created_at", "updated_at", "metadata", "active"}:
            continue
        if isinstance(spec.default, str):
            names.append(spec.name)
    return tuple(names)


def _record_title(kind: str, record: Any) -> str:
    """A readable one-line title for a hit, from whatever the record holds."""
    for name in ("title", "name", "subject", "number"):
        value = getattr(record, name, "")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return getattr(record, "record_id", "") or kind


def search_workplace(
    repository: WorkplaceRepository,
    query: str,
    *,
    kinds: Sequence[str] | None = None,
    limit: int = 20,
    min_score: float = 0.0,
) -> SearchReport:
    """Search the Workplace's records for ``query`` and return ranked hits.

    ``kinds`` narrows the search to a subset of the domain tables (a surface can
    offer a "search projects only" mode); with no ``kinds`` every searchable table
    is read. A blank query returns an empty report rather than every row -- the
    honest answer to "search for nothing". A non-positive ``limit`` returns no
    hits, and ``min_score`` drops a row that carries too little of the query.
    """
    terms = content_terms(query)
    report = SearchReport(query=query, terms=tuple(sorted(terms)))
    if not terms or limit <= 0:
        return report

    searchable = [
        kind for kind in (kinds if kinds is not None else _SEARCH_FIELDS)
        if kind in DOMAIN_RECORDS
    ]
    report.searched_kinds = tuple(searchable)

    hits: list[SearchHit] = []
    for kind in searchable:
        fields = _fields_for(kind)
        if not fields:
            continue
        for record in repository.list_domain(kind):
            row_terms = set()
            matched_fields: list[str] = []
            for name in fields:
                value = getattr(record, name, "")
                if not isinstance(value, str) or not value:
                    continue
                field_terms = content_terms(value)
                shared = terms & field_terms
                if shared:
                    matched_fields.append(name)
                    row_terms |= shared
            if not row_terms:
                continue
            score = len(row_terms) / len(terms)
            if score < min_score:
                continue
            hits.append(
                SearchHit(
                    kind=kind,
                    record_id=getattr(record, "record_id", ""),
                    title=_record_title(kind, record),
                    matched=tuple(matched_fields),
                    score=round(score, 4),
                )
            )

    # Strongest first, then a stable tie-break on the readable title and kind, so
    # the same query always returns the same order.
    hits.sort(key=lambda hit: (-hit.score, hit.kind, hit.title, hit.record_id))
    report.hits = hits[:limit]
    return report


def searchable_kinds() -> list[tuple[str, str]]:
    """``(kind, readable label)`` for every table the search reads, in order."""
    return [(kind, _KIND_LABELS.get(kind, kind)) for kind in _SEARCH_FIELDS]


__all__ = [
    "SearchHit",
    "SearchReport",
    "search_workplace",
    "searchable_kinds",
]
