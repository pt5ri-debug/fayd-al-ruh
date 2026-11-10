"""Cumulative institutional knowledge: capture the *outcome* of the work.

The Workplace already keeps a *registry* of institutional knowledge
(:class:`core.workplace.records.KnowledgeRecord`) and the assistant already reads
it -- but nothing turned a *finished* piece of work into a knowledge record. A
completed project, an approved report, an issued document, a closed NCR, a
completed inspection, and a finished test are exactly the experiences the
institution wants to remember, and re-typing each one by hand is the step that
never happens.

This module is that step, and only that step: it **captures** a finished source
into the *existing* knowledge registry (never a second store), carrying the
source's own provenance and classification. It is deliberately:

* **A view, not a second store.** The captured row is an ordinary
  :class:`~core.workplace.records.KnowledgeRecord` in the ``knowledge_records``
  table the assistant already reads (the brief's §11 rule). No new table, no
  parallel corpus.
* **Idempotent.** A source is captured once; a second capture of the same source
  is reported *skipped*, never duplicated (the ``source``/``reference`` pair is
  the dedup key). So re-running a capture can never double the registry.
* **Faithful about confidentiality.** The captured record inherits the source's
  own classification (a restricted document yields a restricted knowledge
  record), so the assistant's clearance filter keeps withholding it from a reader
  who may not see the source -- the brief's "preserve confidentiality".
* **Honest about eligibility.** Only a *finished* source is captured -- a project
  in progress, a draft report, an open finding yield **nothing** (the negative
  control the gate pins), and an unknown source id is **refused**, never invented.
* **Provenance-carrying.** The record names its origin (the source kind in
  ``source``, the source id in ``reference``) and is marked ``verified`` only when
  the source itself carries a human approval or a closed/issued state.

It is Tk-free, model-free, and network-free, so the service, the CLI, and the
tests all drive the *same* capture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.workplace.records import (
    Finding,
    Inspection,
    KnowledgeRecord,
    Project,
    Report,
    Test,
    WorkplaceDocument,
)
from core.workplace.repository import WorkplaceRepository

#: The knowledge-record kinds the registry accepts (its own closed set).
KNOWLEDGE_KINDS = ("lesson", "best_practice", "procedure_note", "technical_note")

#: The source kinds a capture may read, in the order a report lists them.
SOURCE_KINDS = ("project", "report", "document", "finding", "inspection", "test")

#: The project lifecycle states that mean the work is *finished*.
COMPLETED_PROJECT_STATUSES = frozenset({"completed", "closed"})

#: The report states that carry a human approval (only these are knowledge).
APPROVED_REPORT_STATUSES = frozenset({"approved", "issued"})

#: The finding states that mean the nonconformity is closed.
CLOSED_FINDING_STATUSES = frozenset({"closed"})

#: The inspection states that mean the visit is finished.
COMPLETED_INSPECTION_STATUSES = frozenset({"completed"})

#: The prefix every captured record's ``source`` carries (its dedup namespace).
SOURCE_PREFIX = "workplace"

_SOURCE_LABELS: dict[str, tuple[str, str]] = {
    "project": ("Completed project", "مشروع مكتمل"),
    "report": ("Approved report", "تقرير معتمد"),
    "document": ("Issued document", "وثيقة صادرة"),
    "finding": ("Closed finding", "ملاحظة مغلقة"),
    "inspection": ("Completed inspection", "فحص مكتمل"),
    "test": ("Finished test", "اختبار منتهٍ"),
}

#: The knowledge-record kind each source kind captures *as*.
_SOURCE_KNOWLEDGE_KIND: dict[str, str] = {
    "project": "technical_note",
    "report": "technical_note",
    "document": "technical_note",
    "finding": "lesson",
    "inspection": "best_practice",
    "test": "technical_note",
}


class WorkplaceKnowledgeError(RuntimeError):
    """A knowledge capture could not be completed; the message says why."""


def source_label(source_kind: str, *, language: str = "en") -> str:
    """The readable label of a capture source kind (its own key when unknown)."""
    labels = _SOURCE_LABELS.get(source_kind)
    if labels is None:
        return source_kind
    return labels[1] if language == "ar" else labels[0]


@dataclass(frozen=True)
class KnowledgeCandidate:
    """A finished source as a *proposed* knowledge record (not yet written)."""

    source_kind: str
    source_id: str
    title: str
    kind: str
    summary: str
    confidentiality: str = "internal"
    project_id: str = ""
    division_id: str = ""
    verified: bool = False

    def describe(self, *, language: str = "en") -> str:
        origin = source_label(self.source_kind, language=language)
        return f"[{origin}] {self.title} ({self.kind}, {self.confidentiality})"


@dataclass
class CaptureReport:
    """What one capture run did: what it wrote, skipped, refused, or failed on."""

    captured: list[KnowledgeRecord] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    refused: list[tuple[str, str]] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def count(self) -> int:
        """How many knowledge records this run wrote."""
        return len(self.captured)

    @property
    def empty(self) -> bool:
        return not (self.captured or self.skipped or self.refused or self.errors)

    def describe(self, *, language: str = "en") -> str:
        if self.empty:
            return "Nothing to capture."
        parts = [f"Captured {self.count} knowledge record(s)"]
        if self.skipped:
            parts.append(f"skipped {len(self.skipped)} already captured")
        if self.refused:
            parts.append(f"refused {len(self.refused)} not finished")
        if self.errors:
            parts.append(f"{len(self.errors)} could not be read")
        return "; ".join(parts) + "."

    def as_dict(self) -> dict[str, Any]:
        return {
            "captured": [record.record_id for record in self.captured],
            "skipped": list(self.skipped),
            "refused": [{"source": source, "reason": reason} for source, reason in self.refused],
            "errors": [{"source": source, "reason": reason} for source, reason in self.errors],
            "count": self.count,
        }


class KnowledgeCapture:
    """Turn the Workplace's *finished* work into cumulative institutional knowledge.

    Built over the repository the service already holds (no second database) and
    over the audited domain service when one is supplied (so every write is on
    the audit trail). The engine reads a source record, decides whether it is
    finished, and -- if it is -- writes an ordinary ``knowledge_records`` row that
    the Workplace assistant already cites.
    """

    def __init__(
        self,
        repository: WorkplaceRepository,
        *,
        domain: Any = None,
        projects: Any = None,
    ) -> None:
        self.repository = repository
        self.domain = domain
        self.projects = projects

    # -- the dedup key ----------------------------------------------------

    def existing_references(self) -> set[tuple[str, str]]:
        """The ``(source, reference)`` pairs already captured (idempotency key).

        A captured record carries ``source = "workplace:<kind>"`` and
        ``reference = <source id>``, so this set says exactly which sources are
        already in the registry -- including any that were deactivated, so a
        soft-deleted capture is not silently re-created.
        """
        pairs: set[tuple[str, str]] = set()
        for row in self.repository.list_domain("knowledge_records", include_inactive=True):
            if isinstance(row, KnowledgeRecord):
                pairs.add((row.source, row.reference))
        return pairs

    @staticmethod
    def _dedup_key(candidate: KnowledgeCandidate) -> tuple[str, str]:
        return (f"{SOURCE_PREFIX}:{candidate.source_kind}", candidate.source_id)

    # -- eligibility + building ------------------------------------------

    def _source_record(self, source_kind: str, source_id: str) -> Any:
        if source_kind not in SOURCE_KINDS:
            raise WorkplaceKnowledgeError(
                f"Unknown capture source {source_kind!r}; expected one of "
                f"{', '.join(SOURCE_KINDS)}."
            )
        return self.repository.get_domain(self._table(source_kind), source_id)

    @staticmethod
    def _table(source_kind: str) -> str:
        return {
            "project": "projects",
            "report": "reports",
            "document": "workplace_documents",
            "finding": "findings",
            "inspection": "inspections",
            "test": "tests",
        }[source_kind]

    def candidate(self, source_kind: str, source_id: str) -> KnowledgeCandidate | None:
        """The knowledge a *finished* source yields, or ``None`` if it is not.

        A source that does not exist, or is not finished, returns ``None`` -- a
        capture never invents knowledge for work that is still in progress.
        """
        record = self._source_record(source_kind, source_id)
        if record is None:
            return None
        if source_kind == "project" and isinstance(record, Project):
            return self._from_project(record)
        if source_kind == "report" and isinstance(record, Report):
            return self._from_report(record)
        if source_kind == "document" and isinstance(record, WorkplaceDocument):
            return self._from_document(record)
        if source_kind == "finding" and isinstance(record, Finding):
            return self._from_finding(record)
        if source_kind == "inspection" and isinstance(record, Inspection):
            return self._from_inspection(record)
        if source_kind == "test" and isinstance(record, Test):
            return self._from_test(record)
        return None

    def _from_project(self, project: Project) -> KnowledgeCandidate | None:
        if project.status not in COMPLETED_PROJECT_STATUSES:
            return None
        summary = self._project_summary(project)
        return KnowledgeCandidate(
            source_kind="project",
            source_id=project.record_id,
            title=project.title or project.number or project.record_id,
            kind="technical_note",
            summary=summary,
            confidentiality="internal",
            project_id=project.record_id,
            division_id=project.division_id,
            verified=True,
        )

    def _project_summary(self, project: Project) -> str:
        lines = [f"A completed project ({project.status})."]
        if project.number:
            lines.append(f"Number: {project.number}.")
        if project.scope:
            lines.append(f"Scope: {project.scope}")
        if self.projects is not None:
            try:
                resolved = self.projects.resolve(project.record_id)
            except Exception:  # a missing reference never blocks a capture
                resolved = None
            if resolved is not None and getattr(resolved, "client", None) is not None:
                client_name = getattr(resolved.client, "name", "")
                if client_name:
                    lines.append(f"Client: {client_name}.")
        return " ".join(lines)

    def _from_report(self, report: Report) -> KnowledgeCandidate | None:
        if report.status not in APPROVED_REPORT_STATUSES:
            return None
        lines = [f"An {report.status} {report.kind.replace('_', ' ')}."]
        if report.summary:
            lines.append(report.summary)
        elif report.body:
            lines.append(report.body.strip().split("\n", 1)[0][:400])
        return KnowledgeCandidate(
            source_kind="report",
            source_id=report.record_id,
            title=report.title or report.number or report.record_id,
            kind="technical_note",
            summary=" ".join(lines),
            confidentiality="internal",
            project_id=report.project_id,
            division_id=report.division_id,
            verified=True,
        )

    def _from_document(self, document: WorkplaceDocument) -> KnowledgeCandidate | None:
        if not (document.approved_by or document.issued_on):
            return None
        lines = [f"An issued {document.doc_type.replace('_', ' ')} document."]
        if document.title_ar:
            lines.append(document.title_ar)
        if document.revision:
            lines.append(f"Revision: {document.revision}.")
        return KnowledgeCandidate(
            source_kind="document",
            source_id=document.record_id,
            title=document.title or document.record_id,
            kind="technical_note",
            summary=" ".join(lines),
            confidentiality=document.confidentiality or "internal",
            project_id=document.project_id,
            division_id=document.division_id,
            verified=True,
        )

    def _from_finding(self, finding: Finding) -> KnowledgeCandidate | None:
        if finding.status not in CLOSED_FINDING_STATUSES:
            return None
        lines = [f"A closed {finding.kind.upper()} ({finding.severity})."]
        if finding.description:
            lines.append(finding.description)
        return KnowledgeCandidate(
            source_kind="finding",
            source_id=finding.record_id,
            title=finding.title or finding.number or finding.record_id,
            kind="lesson",
            summary=" ".join(lines),
            confidentiality="internal",
            project_id=finding.project_id,
            division_id=finding.division_id,
            verified=True,
        )

    def _from_inspection(self, inspection: Inspection) -> KnowledgeCandidate | None:
        if inspection.status not in COMPLETED_INSPECTION_STATUSES:
            return None
        lines = [f"A completed {inspection.kind}."]
        if inspection.result:
            lines.append(f"Result: {inspection.result}")
        if inspection.itp_reference:
            lines.append(f"ITP: {inspection.itp_reference}.")
        return KnowledgeCandidate(
            source_kind="inspection",
            source_id=inspection.record_id,
            title=inspection.title or inspection.record_id,
            kind="best_practice",
            summary=" ".join(lines),
            confidentiality="internal",
            project_id=inspection.project_id,
            division_id=inspection.division_id,
            verified=bool(inspection.result),
        )

    def _from_test(self, test: Test) -> KnowledgeCandidate | None:
        if not test.result:
            return None
        lines = [f"A finished {test.kind or 'test'}."]
        measurement = " ".join(part for part in (test.value, test.unit) if part)
        if measurement:
            lines.append(f"Measured: {measurement}.")
        if test.standard_reference:
            lines.append(f"Standard: {test.standard_reference}.")
        lines.append(f"Result: {test.result}.")
        return KnowledgeCandidate(
            source_kind="test",
            source_id=test.record_id,
            title=test.title or test.record_id,
            kind="technical_note",
            summary=" ".join(lines),
            confidentiality="internal",
            project_id=test.project_id,
            division_id="",
            verified=True,
        )

    # -- candidate listing ------------------------------------------------

    def candidates(self, *, source_kind: str | None = None) -> list[KnowledgeCandidate]:
        """Every finished source that yields knowledge (already-captured included)."""
        kinds = (source_kind,) if source_kind else SOURCE_KINDS
        out: list[KnowledgeCandidate] = []
        for kind in kinds:
            if kind not in SOURCE_KINDS:
                raise WorkplaceKnowledgeError(f"Unknown capture source {kind!r}.")
            for row in self.repository.list_domain(self._table(kind), include_inactive=True):
                candidate = self.candidate(kind, getattr(row, "record_id", ""))
                if candidate is not None:
                    out.append(candidate)
        out.sort(key=lambda c: (c.source_kind, c.title.casefold()))
        return out

    # -- the write --------------------------------------------------------

    def capture(self, source_kind: str, source_id: str) -> KnowledgeRecord:
        """Capture one finished source into the knowledge registry.

        Raises :class:`WorkplaceKnowledgeError` when the source does not exist or
        is not finished, so an ineligible capture is a named refusal, never a
        silent no-op or an invented record.
        """
        record = self._source_record(source_kind, source_id)
        if record is None:
            raise WorkplaceKnowledgeError(
                f"No {source_kind} with id {source_id!r}."
            )
        candidate = self.candidate(source_kind, source_id)
        if candidate is None:
            raise WorkplaceKnowledgeError(
                f"The {source_kind} {source_id!r} is not finished, so it carries no "
                "knowledge to capture yet."
            )
        return self._write(candidate)

    def _write(self, candidate: KnowledgeCandidate) -> KnowledgeRecord:
        source = f"{SOURCE_PREFIX}:{candidate.source_kind}"
        fields: dict[str, Any] = {
            "title": candidate.title,
            "summary": candidate.summary,
            "kind": candidate.kind if candidate.kind in KNOWLEDGE_KINDS else "technical_note",
            "source": source,
            "reference": candidate.source_id,
            "confidentiality": candidate.confidentiality or "internal",
            "verified": candidate.verified,
        }
        if candidate.project_id:
            fields["project_id"] = candidate.project_id
        if candidate.division_id:
            fields["division_id"] = candidate.division_id
        if self.domain is not None:
            return self.domain.add_knowledge(title=candidate.title, **{
                key: value for key, value in fields.items() if key != "title"
            })
        created = self.repository.create_domain("knowledge_records", **fields)
        assert isinstance(created, KnowledgeRecord)
        return created

    # -- bulk -------------------------------------------------------------

    def capture_all(self, *, source_kind: str | None = None) -> CaptureReport:
        """Capture every finished source of ``source_kind`` (or every kind).

        Idempotent: a source already captured is reported *skipped* rather than
        written again. A source that cannot be read is recorded in ``errors`` and
        never aborts the run.
        """
        report = CaptureReport()
        existing = self.existing_references()
        for candidate in self.candidates(source_kind=source_kind):
            key = self._dedup_key(candidate)
            if key in existing:
                report.skipped.append(candidate.source_id)
                continue
            try:
                record = self._write(candidate)
            except Exception as error:  # one bad source never aborts the batch
                report.errors.append((candidate.source_id, str(error)))
                continue
            report.captured.append(record)
            existing.add(key)
        return report

    def capture_project(self, project_id: str) -> CaptureReport:
        """Capture a project's own finished work: the project and its outcomes.

        Captures the project itself (when finished) and, for that project only,
        its approved reports, issued documents, closed findings, completed
        inspections, and finished tests -- so "remember what this project taught"
        is one call. Idempotent, exactly like :meth:`capture_all`.
        """
        report = CaptureReport()
        existing = self.existing_references()
        for source_kind in SOURCE_KINDS:
            table = self._table(source_kind)
            # The project row itself is keyed by its own id; every other source
            # carries a ``project_id`` foreign key, so it is scoped that way.
            if source_kind == "project":
                rows = self.repository.list_domain(
                    table, where="record_id = ?", params=(project_id,), include_inactive=True
                )
            else:
                rows = self.repository.list_domain(
                    table, where="project_id = ?", params=(project_id,), include_inactive=True
                )
            for row in rows:
                candidate = self.candidate(source_kind, getattr(row, "record_id", ""))
                if candidate is None:
                    continue
                key = self._dedup_key(candidate)
                if key in existing:
                    report.skipped.append(candidate.source_id)
                    continue
                try:
                    record = self._write(candidate)
                except Exception as error:
                    report.errors.append((candidate.source_id, str(error)))
                    continue
                report.captured.append(record)
                existing.add(key)
        return report

    # -- read -------------------------------------------------------------

    def knowledge(self, *, project_id: str | None = None) -> list[KnowledgeRecord]:
        """The captured knowledge records, newest first (optionally per project)."""
        if project_id:
            rows = self.repository.list_domain(
                "knowledge_records", where="project_id = ?", params=(project_id,)
            )
        else:
            rows = self.repository.list_domain("knowledge_records")
        records = [row for row in rows if isinstance(row, KnowledgeRecord)]
        records.sort(key=lambda row: (row.created_at, row.title), reverse=True)
        return records

    def summary(self) -> dict[str, int]:
        """How many knowledge records the registry holds, by source kind."""
        counts: dict[str, int] = dict.fromkeys(SOURCE_KINDS, 0)
        counts["manual"] = 0
        for row in self.repository.list_domain("knowledge_records"):
            if not isinstance(row, KnowledgeRecord):
                continue
            kind = row.source.split(":", 1)[1] if row.source.startswith(f"{SOURCE_PREFIX}:") else ""
            counts[kind if kind in counts else "manual"] += 1
        return counts

    def describe(self, *, language: str = "en") -> str:
        """An honest one-line status of what could be captured."""
        pending = [
            candidate
            for candidate in self.candidates()
            if self._dedup_key(candidate) not in self.existing_references()
        ]
        held = self.summary()
        captured_total = sum(held[kind] for kind in SOURCE_KINDS)
        return (
            f"{captured_total} knowledge record(s) captured from finished work; "
            f"{len(pending)} finished source(s) not captured yet."
        )


__all__ = [
    "APPROVED_REPORT_STATUSES",
    "CLOSED_FINDING_STATUSES",
    "COMPLETED_INSPECTION_STATUSES",
    "COMPLETED_PROJECT_STATUSES",
    "KNOWLEDGE_KINDS",
    "SOURCE_KINDS",
    "SOURCE_PREFIX",
    "CaptureReport",
    "KnowledgeCandidate",
    "KnowledgeCapture",
    "WorkplaceKnowledgeError",
    "source_label",
]
