"""The Workplace technical library: the standards a project works to (the brief's §7).

The library is a **view** over the Workplace's ``standards`` rows, not a second
database. It answers the questions a technical office actually asks:

* *what do we hold, and which body issued it?* -- :meth:`TechnicalLibrary.standards`
  filters by issuing body, by reference kind, or by a free-text query;
* *which revision is current?* -- a standard's revisions are ordered by their
  publication date, and :meth:`TechnicalLibrary.compare_revisions` reports what
  changed between two of them;
* *where does a body sit in the catalogue?* -- :data:`STANDARD_BODIES` names the
  bodies the brief lists, so a surface can offer them and say honestly when the
  library holds none of a body's documents.

Three honesty rules run through the module:

* the four **reference kinds** the brief distinguishes -- an *official standard*,
  an *internal procedure*, a *project specification*, and *technical guidance* --
  are never presented as the same thing;
* the library stores a standard's **metadata** (number, title, edition, revision,
  date, source, applicability), never the standard's text. It does **not**
  reproduce a copyrighted standard, and it never fabricates a requirement. A
  ``source`` is recorded so a reader knows where the copy came from;
* a revision is a **new row** (keyed by body + number), so the original is never
  mutated and a change is visible.

Writes go through the audited :class:`~core.workplace.domain.WorkplaceDomain`, so
adding a standard is recorded in the audit trail like every other Workplace change.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from core.workplace.domain import WorkplaceDomain
from core.workplace.records import REFERENCE_KINDS, Standard
from core.workplace.repository import WorkplaceRepository

#: The issuing bodies the brief's §7 names, with their readable titles. This is a
#: *catalogue of names*, not of requirements: the library records which body issued
#: a standard, and never reproduces what the standard says.
STANDARD_BODIES: tuple[tuple[str, str], ...] = (
    ("API", "American Petroleum Institute"),
    ("ASME", "American Society of Mechanical Engineers"),
    ("ASTM", "ASTM International"),
    ("ISO", "International Organization for Standardization"),
    ("AWS", "American Welding Society"),
    ("AMPP", "Association for Materials Protection and Performance (NACE)"),
    ("NACE", "National Association of Corrosion Engineers (now AMPP)"),
    ("DNV", "Det Norske Veritas"),
    ("EN", "European Norm"),
    ("IEC", "International Electrotechnical Commission"),
    ("BS", "British Standard"),
    ("EOS", "Egyptian Organization for Standardization"),
)

#: The readable label of each reference kind (English + Arabic).
REFERENCE_KIND_LABELS: dict[str, tuple[str, str]] = {
    "standard": ("Official standard", "معيار رسمي"),
    "internal_procedure": ("Internal procedure", "إجراء داخلي"),
    "project_specification": ("Project specification", "مواصفة مشروع"),
    "guidance": ("Technical guidance", "إرشاد فني"),
}


def reference_kind_label(kind: str, *, language: str = "en") -> str:
    """The readable label of a reference kind (its own key when unknown)."""
    labels = REFERENCE_KIND_LABELS.get(kind)
    if labels is None:
        return kind
    return labels[1] if language == "ar" else labels[0]


def body_title(body: str) -> str:
    """The readable title of an issuing body (its own code when unknown)."""
    for code, title in STANDARD_BODIES:
        if code == body:
            return title
    return body


def _normalise(text: str) -> str:
    return " ".join(text.casefold().split())


def _matches(needle: str, *haystacks: str) -> bool:
    """A whole-phrase, case-insensitive match against any of ``haystacks``."""
    if not needle:
        return True
    return any(needle in _normalise(value) for value in haystacks if value)


@dataclass(frozen=True)
class StandardRevision:
    """One revision of a standard, as the library orders it."""

    record_id: str
    body: str
    number: str
    title: str
    edition: str
    revision: str
    published_on: str
    reference_kind: str
    applicability: str
    status: str
    source: str

    @property
    def label(self) -> str:
        """A readable ``BODY NUMBER (edition/revision)`` label."""
        parts = [f"{self.body} {self.number}".strip()]
        edition = self.edition or self.revision
        if edition:
            parts.append(f"({edition})")
        return " ".join(parts)

    def describe(self) -> str:
        head = f"{self.label} — {self.title}" if self.title else self.label
        extra = [f"[{self.reference_kind}]"]
        if self.published_on:
            extra.append(f"published {self.published_on}")
        if self.status and self.status != "active":
            extra.append(f"({self.status})")
        return f"{head} {' '.join(extra)}"

    @classmethod
    def from_record(cls, record: Standard) -> StandardRevision:
        return cls(
            record_id=record.record_id,
            body=record.body,
            number=record.number,
            title=record.title,
            edition=record.edition,
            revision=record.revision,
            published_on=record.published_on,
            reference_kind=record.reference_kind,
            applicability=record.applicability,
            status=record.status,
            source=record.source,
        )


@dataclass(frozen=True)
class RevisionComparison:
    """What changed between two revisions of the same standard (the brief's §10)."""

    body: str
    number: str
    from_revision: StandardRevision
    to_revision: StandardRevision

    def changes(self) -> list[tuple[str, str, str]]:
        """The fields that differ, as ``(field, before, after)`` rows."""
        rows: list[tuple[str, str, str]] = []
        for field in ("title", "edition", "revision", "published_on", "applicability", "status"):
            before = getattr(self.from_revision, field)
            after = getattr(self.to_revision, field)
            if before != after:
                rows.append((field, before, after))
        return rows

    @property
    def changed(self) -> bool:
        return bool(self.changes())

    def describe(self) -> str:
        changes = self.changes()
        if not changes:
            return (
                f"{self.from_revision.label} and {self.to_revision.label} record the "
                "same metadata (no change to compare)."
            )
        lines = [f"{self.from_revision.label} → {self.to_revision.label}:"]
        for field, before, after in changes:
            lines.append(f"  {field}: {before or '(none)'} → {after or '(none)'}")
        return "\n".join(lines)


class TechnicalLibrary:
    """The standards library: list, filter, add, and compare revisions.

    Read paths use the repository directly; writes go through ``domain`` (the
    audited service), and a library built without one is read-only.
    """

    def __init__(
        self, repository: WorkplaceRepository, domain: WorkplaceDomain | None = None
    ) -> None:
        self.repository = repository
        self.domain = domain

    # -- read -------------------------------------------------------------

    def standards(
        self,
        *,
        body: str | None = None,
        reference_kind: str | None = None,
        query: str | None = None,
        include_inactive: bool = False,
    ) -> list[StandardRevision]:
        """The library, optionally narrowed by body, kind, or a free-text query."""
        rows = self.repository.list_domain(
            "standards", include_inactive=include_inactive
        )
        standards = [row for row in rows if isinstance(row, Standard)]
        if body:
            standards = [s for s in standards if s.body == body]
        if reference_kind:
            standards = [s for s in standards if s.reference_kind == reference_kind]
        if query:
            needle = _normalise(query)
            standards = [
                s for s in standards
                if _matches(needle, s.number, s.title, s.applicability, s.body)
            ]
        standards.sort(key=lambda s: (s.body, s.number, s.published_on))
        return [StandardRevision.from_record(s) for s in standards]

    def standard(self, record_id: str) -> StandardRevision | None:
        record = self.repository.get_domain("standards", record_id)
        return StandardRevision.from_record(record) if isinstance(record, Standard) else None

    def revisions(self, *, body: str, number: str) -> list[StandardRevision]:
        """Every revision of one standard, oldest first (by publication date)."""
        rows = self.repository.list_domain("standards", include_inactive=True)
        matching = [
            row for row in rows
            if isinstance(row, Standard) and row.body == body and row.number == number
        ]
        matching.sort(key=lambda s: (s.published_on, s.created_at))
        return [StandardRevision.from_record(s) for s in matching]

    def current(self, *, body: str, number: str) -> StandardRevision | None:
        """The most recently published revision of a standard, or ``None``."""
        chain = self.revisions(body=body, number=number)
        return chain[-1] if chain else None

    def compare_revisions(
        self, *, body: str, number: str, from_revision: str = "", to_revision: str = ""
    ) -> RevisionComparison | None:
        """Compare two revisions of one standard.

        With no revision given, the last two are compared; with one, it is compared
        against the current revision. Returns ``None`` when there is nothing to
        compare (fewer than two revisions, or a named revision that does not exist)
        -- never a fabricated comparison.
        """
        chain = self.revisions(body=body, number=number)
        if len(chain) < 2:
            return None
        if not from_revision and not to_revision:
            return RevisionComparison(body=body, number=number, from_revision=chain[-2], to_revision=chain[-1])
        lookup = {r.revision: r for r in chain if r.revision}
        lookup.update({r.record_id: r for r in chain})
        target = lookup.get(to_revision) if to_revision else chain[-1]
        source = lookup.get(from_revision) if from_revision else chain[-2]
        if target is None or source is None or source.record_id == target.record_id:
            return None
        return RevisionComparison(body=body, number=number, from_revision=source, to_revision=target)

    def bodies(self) -> list[tuple[str, str, int]]:
        """Every issuing body the library holds, with its title and how many it has.

        The shipped catalogue is included so a surface can offer a body it holds
        nothing of yet; a body with no document is honestly reported with ``0``.
        """
        rows = self.repository.list_domain("standards")
        counts: dict[str, int] = {}
        for row in rows:
            if isinstance(row, Standard) and row.body:
                counts[row.body] = counts.get(row.body, 0) + 1
        known = dict(STANDARD_BODIES)
        for code in counts:
            known.setdefault(code, code)
        return sorted(
            ((code, title, counts.get(code, 0)) for code, title in known.items()),
            key=lambda row: (-row[2], row[0]),
        )

    def counts_by_kind(self) -> dict[str, int]:
        """How many references the library holds of each reference kind."""
        rows = self.repository.list_domain("standards")
        counts = dict.fromkeys(REFERENCE_KINDS, 0)
        for row in rows:
            if isinstance(row, Standard):
                counts[row.reference_kind] = counts.get(row.reference_kind, 0) + 1
        return counts

    # -- write (through the audited domain) -------------------------------

    def add_standard(
        self,
        *,
        body: str,
        number: str,
        title: str = "",
        edition: str = "",
        revision: str = "",
        published_on: str = "",
        reference_kind: str = "standard",
        applicability: str = "",
        source: str = "",
        status: str = "active",
    ) -> StandardRevision:
        """Add a reference to the library (a *metadata row*, never the standard's text)."""
        if self.domain is None:
            raise RuntimeError("This library is read-only (no domain service supplied).")
        if not (body.strip() or number.strip()):
            raise ValueError("A reference needs an issuing body or a number.")
        if reference_kind not in REFERENCE_KINDS:
            raise ValueError(
                f"Unknown reference kind {reference_kind!r}; expected one of "
                f"{', '.join(REFERENCE_KINDS)}."
            )
        record = self.domain.create_standard(
            body=body.strip(), number=number.strip(), title=title.strip(),
            edition=edition.strip(), revision=revision.strip(),
            published_on=published_on.strip(), reference_kind=reference_kind,
            applicability=applicability.strip(), source=source.strip(), status=status,
        )
        return StandardRevision.from_record(record)

    def record_new_revision(
        self, record_id: str, **changes: Any
    ) -> StandardRevision:
        """Record a *new* revision of a standard (the original row is untouched).

        The new row carries the previous revision's fields, overridden by
        ``changes``, so a revision is a complete record and the original is
        preserved -- the same "a revision is a new record" discipline the learning
        records use.
        """
        if self.domain is None:
            raise RuntimeError("This library is read-only (no domain service supplied).")
        current = self.repository.get_domain("standards", record_id)
        if not isinstance(current, Standard):
            raise ValueError(f"No standard with id {record_id!r}.")
        fields = {
            field.name: getattr(current, field.name)
            for field in dataclasses.fields(current)
            if field.name not in {"record_id", "created_at", "updated_at", "active"}
        }
        fields.update(changes)
        record = self.domain.create("standards", **fields)
        return StandardRevision.from_record(record)

    def deactivate(self, record_id: str) -> StandardRevision:
        """Withdraw a reference (the soft delete); history survives."""
        if self.domain is None:
            raise RuntimeError("This library is read-only (no domain service supplied).")
        updated = self.domain.deactivate("standards", record_id)
        return StandardRevision.from_record(updated)


__all__ = [
    "REFERENCE_KIND_LABELS",
    "STANDARD_BODIES",
    "RevisionComparison",
    "StandardRevision",
    "TechnicalLibrary",
    "body_title",
    "reference_kind_label",
]
