"""Reporting and proposal workflows for the Workplace (the brief's §9, §10, §13).

Phase 4 gave every report, document, and approval its own row; Phase 5 assembled a
project's records into a :class:`~core.workplace.projects.ProjectSummary`; Phase 7
gave the Workplace an assistant over its own records. This module is the *workflow*
that ties them together: it **drafts** a report, a proposal, a management summary,
or a formal correspondence from a project's own records; it moves a report through
its **lifecycle** with a validated state machine; and it records every **approval**
as a new row, so a report can only be *issued* through the user's own decision --
never by the AI (the brief's §13).

Two honesty rules run through the whole module:

* **Nothing is invented.** A draft is assembled from the records the Workplace
  already holds (the client, the scope, the standards, the inspections, the tests,
  the findings, the documents, the people assigned). A section with nothing to say
  says so rather than padding itself.
* **The AI advises; it never decides.** When a model is wired, the assistant's
  cited answer fills a section and is marked ``ai=True``; the answer still carries
  :data:`~core.workplace.assistant.DISCLAIMER`, and an *issue* is refused until a
  named human has approved the report. A recommendation is never an approved
  engineering decision.

It is deliberately read-mostly: drafting reads; creating, submitting, approving,
rejecting, and issuing write through the *same* audited
:class:`~core.workplace.domain.WorkplaceDomain` every other Workplace write uses,
so the audit trail stays complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.workplace.assistant import DISCLAIMER, WorkplaceAssistant
from core.workplace.projects import ProjectManager, ProjectSummary
from core.workplace.records import Report
from core.workplace.repository import WorkplaceRepository

#: The kinds of report the Workplace drafts (the brief's §9, §10). Each maps onto
#: the ``kind`` field of a ``Report`` row, so a new kind needs no migration.
REPORT_KINDS = (
    "technical_report",
    "inspection_report",
    "progress_report",
    "management_summary",
    "method_statement",
    "proposal",
    "correspondence",
)

#: The report lifecycle states (a small, closed set).
REPORT_STATUSES = (
    "draft",
    "submitted",
    "approved",
    "rejected",
    "issued",
    "superseded",
)

#: The legal transitions of the lifecycle. A report reaches ``issued`` only through
#: ``approved`` -- so an *issued* deliverable always carries a human approval, and
#: the AI can never issue one on its own (the brief's §13).
REPORT_TRANSITIONS: dict[str, frozenset[str]] = {
    "draft": frozenset({"submitted", "rejected"}),
    "submitted": frozenset({"approved", "rejected", "draft"}),
    "approved": frozenset({"issued", "superseded", "draft"}),
    "rejected": frozenset({"draft"}),
    "issued": frozenset({"superseded"}),
    "superseded": frozenset(),
}

#: The report kinds that are a *deliverable* (issued to the client).
DELIVERABLE_KINDS = frozenset({"technical_report", "inspection_report", "proposal", "method_statement"})

#: The kind of a report that has been issued (its document is the deliverable).
REPORT_DOC_TYPES: dict[str, str] = {
    "technical_report": "technical_report",
    "inspection_report": "inspection_report",
    "progress_report": "technical_report",
    "management_summary": "technical_report",
    "method_statement": "method_statement",
    "proposal": "proposal",
    "correspondence": "correspondence",
}


class ReportingError(Exception):
    """A reporting or proposal workflow was asked to do something it cannot."""


def report_kind_label(kind: str, *, language: str = "en") -> str:
    """The readable label of a report kind (its own key when unknown)."""
    labels = _REPORT_KIND_LABELS.get(kind)
    if labels is None:
        return kind
    return labels[1] if language == "ar" else labels[0]


_REPORT_KIND_LABELS: dict[str, tuple[str, str]] = {
    "technical_report": ("Technical report", "تقرير فني"),
    "inspection_report": ("Inspection report", "تقرير فحص"),
    "progress_report": ("Progress report", "تقرير تقدم"),
    "management_summary": ("Management summary", "ملخص إداري"),
    "method_statement": ("Method statement", "بيان طريقة العمل"),
    "proposal": ("Technical / commercial proposal", "عرض فني / تجاري"),
    "correspondence": ("Formal correspondence", "مراسلات رسمية"),
}


def status_label(status: str, *, language: str = "en") -> str:
    """The readable label of a report status (its own key when unknown)."""
    labels = _STATUS_LABELS.get(status)
    if labels is None:
        return status
    return labels[1] if language == "ar" else labels[0]


_STATUS_LABELS: dict[str, tuple[str, str]] = {
    "draft": ("Draft", "مسودة"),
    "submitted": ("Submitted for approval", "مُقدَّم للاعتماد"),
    "approved": ("Approved", "معتمد"),
    "rejected": ("Rejected", "مرفوض"),
    "issued": ("Issued", "صادر"),
    "superseded": ("Superseded", "مُستبدل"),
}


def can_transition(current: str, target: str) -> bool:
    """Whether ``current`` may move to ``target`` (the one lifecycle rule)."""
    return target in REPORT_TRANSITIONS.get(current, frozenset())


@dataclass(frozen=True)
class ReportSection:
    """One section of a draft: a heading and the lines that belong to it."""

    title: str
    lines: tuple[str, ...] = ()
    ai: bool = False
    """True when a language model wrote this section (its advice, not a decision)."""

    @property
    def empty(self) -> bool:
        return not self.lines

    def describe(self) -> str:
        if not self.lines:
            return f"{self.title}\n  (nothing on file)"
        body = "\n".join(f"  - {line}" for line in self.lines)
        marker = "  [AI advice -- not an approved decision]" if self.ai else ""
        return f"{self.title}{marker}\n{body}"


@dataclass(frozen=True)
class ReportDraft:
    """A drafted report: its kind, its title, and its ordered sections."""

    kind: str
    title: str
    project_id: str
    sections: tuple[ReportSection, ...] = ()
    citations: tuple[str, ...] = ()
    used_assistant: bool = False

    def section(self, title: str) -> ReportSection | None:
        for section in self.sections:
            if section.title == title:
                return section
        return None

    def to_text(self) -> str:
        """The draft as plain text, section by section (AI sections marked)."""
        head = f"{report_kind_label(self.kind)}: {self.title}"
        body = "\n\n".join(section.describe() for section in self.sections)
        tail = ""
        if self.used_assistant:
            tail = f"\n\n{DISCLAIMER}"
        if self.citations:
            sources = "\n".join(f"  - {citation}" for citation in self.citations)
            tail = f"{tail}\n\nSources:\n{sources}"
        return f"{head}\n\n{body}{tail}".strip()

    def describe(self) -> str:
        return f"{report_kind_label(self.kind)} draft: {self.title} ({len(self.sections)} sections)"


@dataclass(frozen=True)
class Deliverable:
    """A report that has been issued, with its registered document (if any)."""

    report: Report
    document_id: str = ""
    document_title: str = ""

    def describe(self, *, language: str = "en") -> str:
        number = self.report.number or self.report.record_id
        title = self.report.title
        doc = f" -> {self.document_title}" if self.document_title else ""
        return f"{number} — {title} [{status_label(self.report.status, language=language)}]{doc}"


@dataclass
class _DraftInputs:
    """The project's own records, gathered once for every section builder."""

    summary: ProjectSummary
    standards: list[str] = field(default_factory=list)
    standards_meta: list[str] = field(default_factory=list)


class ReportingWorkflow:
    """Draft, submit, approve, and issue a project's reports and proposals.

    Built over the repository the service already holds, so it never opens a
    second database, and over the audited domain service, so every write is
    recorded. An assistant may be supplied: when it is, a draft's sections are
    filled with the assistant's cited answers (marked as AI advice); with no
    assistant the draft is assembled deterministically from the records alone.
    """

    def __init__(
        self,
        repository: WorkplaceRepository,
        *,
        domain: Any = None,
        projects: ProjectManager | None = None,
        assistant: WorkplaceAssistant | None = None,
        documents: Any = None,
    ) -> None:
        self.repository = repository
        self.domain = domain
        self.projects = projects or ProjectManager(repository)
        self.assistant = assistant
        self.documents = documents

    # -- gather the project's own records ---------------------------------

    def _inputs(self, project_id: str) -> _DraftInputs:
        summary = self.projects.resolve(project_id)
        if summary is None:
            raise ReportingError(f"No project with id {project_id!r}.")
        standards: list[str] = []
        for inspection in summary.inspections:
            if inspection.itp_reference:
                standards.append(inspection.itp_reference)
        for test in summary.tests:
            if test.standard_reference:
                standards.append(test.standard_reference)
        # A division's own profile names the standards it works to.
        if summary.division is not None:
            for detail in self.repository.list_domain(
                "division_details",
                where="division_id = ? AND kind = ?",
                params=(summary.division.record_id, "standard"),
            ):
                label = getattr(detail, "title", "") or getattr(detail, "value", "")
                if label:
                    standards.append(label)
        deduped = list(dict.fromkeys(s for s in standards if s))
        return _DraftInputs(summary=summary, standards=deduped)

    # -- the sections -----------------------------------------------------

    def _overview(self, inputs: _DraftInputs) -> ReportSection:
        summary = inputs.summary
        project = summary.project
        lines: list[str] = []
        if project.number or project.title:
            lines.append(f"Project {project.number or project.record_id}: {project.title}")
        if summary.client is not None:
            lines.append(f"Client: {summary.client.name}")
        if summary.contractor is not None:
            lines.append(f"Contractor: {summary.contractor.name}")
        if summary.consultant is not None:
            lines.append(f"Consultant: {summary.consultant.name}")
        if summary.division is not None:
            lines.append(f"Division: {summary.division.name}")
        if summary.manager is not None:
            lines.append(f"Project manager: {summary.manager.name}")
        if project.location:
            lines.append(f"Location: {project.location}")
        lines.append(f"Status: {project.status}")
        return ReportSection("Overview", tuple(lines))

    def _scope(self, inputs: _DraftInputs) -> ReportSection:
        project = inputs.summary.project
        lines: list[str] = []
        if project.scope:
            lines.append(project.scope)
        if inputs.summary.division is not None and inputs.summary.division.scope:
            lines.append(inputs.summary.division.scope)
        return ReportSection("Scope of work", tuple(lines))

    def _standards(self, inputs: _DraftInputs) -> ReportSection:
        return ReportSection("Applicable standards", tuple(inputs.standards))

    def _inspections(self, inputs: _DraftInputs) -> ReportSection:
        lines: list[str] = []
        for inspection in inputs.summary.inspections:
            detail = f"{inspection.title or inspection.kind or 'inspection'}"
            if inspection.performed_on:
                detail += f" ({inspection.performed_on})"
            if inspection.result:
                detail += f": {inspection.result}"
            elif inspection.status:
                detail += f" [{inspection.status}]"
            lines.append(detail)
        return ReportSection("Inspections", tuple(lines))

    def _tests(self, inputs: _DraftInputs) -> ReportSection:
        lines: list[str] = []
        for test in inputs.summary.tests:
            detail = f"{test.title or test.kind or 'test'}"
            if test.result:
                detail += f": {test.result}"
            if test.value:
                detail += f" ({test.value}{(' ' + test.unit) if test.unit else ''})"
            lines.append(detail)
        return ReportSection("Tests and measurements", tuple(lines))

    def _findings(self, inputs: _DraftInputs) -> ReportSection:
        lines: list[str] = []
        for finding in inputs.summary.findings:
            detail = f"{finding.number or finding.record_id}: {finding.title}"
            if finding.severity:
                detail += f" [severity {finding.severity}]"
            detail += f" ({finding.status})"
            lines.append(detail)
        return ReportSection("Findings and nonconformities", tuple(lines))

    def _personnel(self, inputs: _DraftInputs) -> ReportSection:
        lines: list[str] = []
        for assignment in inputs.summary.assignments:
            person = self.repository.get_domain("people", assignment.person_id)
            name = getattr(person, "name", "") or assignment.person_id
            role = assignment.role or getattr(person, "role", "")
            lines.append(f"{name}" + (f" — {role}" if role else ""))
        return ReportSection("Assigned personnel", tuple(lines))

    def _documents(self, inputs: _DraftInputs) -> ReportSection:
        lines: list[str] = []
        for document in inputs.summary.documents:
            revision = f" rev {document.revision}" if document.revision else ""
            lines.append(f"{document.title}{revision}")
        return ReportSection("Reference documents", tuple(lines))

    def _deliverables(self, inputs: _DraftInputs) -> ReportSection:
        lines = [
            "The technical findings above, with the applicable standards and the "
            "inspection and test evidence that supports them.",
        ]
        if inputs.summary.open_findings:
            lines.append(
                f"{len(inputs.summary.open_findings)} finding(s) remain open and are "
                "tracked to closure."
            )
        return ReportSection("Deliverables", tuple(lines))

    def _commercial(self, inputs: _DraftInputs) -> ReportSection:
        lines: list[str] = []
        for contract in inputs.summary.contracts:
            detail = f"Contract {contract.number or contract.record_id}"
            if contract.value:
                detail += f": {contract.value} {contract.currency}".rstrip()
            if contract.starts_on or contract.ends_on:
                detail += f" ({contract.starts_on} -> {contract.ends_on})"
            lines.append(detail)
        return ReportSection("Contractual basis", tuple(lines))

    def _management_summary(self, inputs: _DraftInputs) -> ReportSection:
        counts = inputs.summary.counts()
        lines = [
            f"{counts.get('documents', 0)} document(s), {counts.get('assignments', 0)} assignment(s).",
            f"{counts.get('open_findings', 0)} open finding(s); "
            f"{counts.get('reports_awaiting_approval', 0)} report(s) awaiting approval.",
        ]
        return ReportSection("Management figures", tuple(lines))

    def _sections_for(self, kind: str, inputs: _DraftInputs) -> list[ReportSection]:
        """The ordered sections of a draft of ``kind`` (nothing invented)."""
        if kind == "proposal":
            return [
                self._overview(inputs),
                self._scope(inputs),
                ReportSection("Technical approach", self._approach_lines(inputs)),
                self._standards(inputs),
                self._personnel(inputs),
                self._deliverables(inputs),
                self._commercial(inputs),
            ]
        if kind == "management_summary":
            return [
                self._overview(inputs),
                self._management_summary(inputs),
                self._findings(inputs),
                self._deliverables(inputs),
            ]
        if kind == "correspondence":
            return [self._overview(inputs), self._scope(inputs)]
        if kind == "progress_report":
            return [
                self._overview(inputs),
                self._scope(inputs),
                self._inspections(inputs),
                self._tests(inputs),
                self._findings(inputs),
                self._management_summary(inputs),
            ]
        if kind == "method_statement":
            return [
                self._overview(inputs),
                self._scope(inputs),
                ReportSection("Procedure", self._approach_lines(inputs)),
                self._standards(inputs),
                self._personnel(inputs),
            ]
        # technical_report / inspection_report (the default).
        return [
            self._overview(inputs),
            self._scope(inputs),
            self._standards(inputs),
            self._inspections(inputs),
            self._tests(inputs),
            self._findings(inputs),
            self._deliverables(inputs),
        ]

    def _approach_lines(self, inputs: _DraftInputs) -> tuple[str, ...]:
        lines: list[str] = []
        if inputs.summary.division is not None and inputs.summary.division.scope:
            lines.append(inputs.summary.division.scope)
        for procedure in self.repository.list_domain(
            "procedures",
            where="project_id = ?",
            params=(inputs.summary.project.record_id,),
        ):
            title = getattr(procedure, "title", "")
            if title:
                lines.append(title)
        if inputs.standards:
            lines.append("Work is carried out to: " + ", ".join(inputs.standards) + ".")
        return tuple(lines)

    # -- the draft --------------------------------------------------------

    def draft(
        self,
        project_id: str,
        *,
        kind: str = "technical_report",
        title: str = "",
        use_assistant: bool = False,
        profile: str = "",
    ) -> ReportDraft:
        """Draft a report from the project's own records.

        With ``use_assistant`` the sections are filled with the Workplace
        assistant's *cited* answers (marked as AI advice); with no assistant or no
        model the draft is assembled deterministically from the records.
        """
        if kind not in REPORT_KINDS:
            raise ReportingError(
                f"Unknown report kind {kind!r}; expected one of {', '.join(REPORT_KINDS)}."
            )
        inputs = self._inputs(project_id)
        sections = self._sections_for(kind, inputs)
        draft_title = title.strip() or self._default_title(kind, inputs)
        citations: list[str] = []
        used_assistant = False
        if use_assistant and self.assistant is not None:
            sections, citations, used_assistant = self._fill_with_assistant(
                kind, sections, profile=profile
            )
        return ReportDraft(
            kind=kind,
            title=draft_title,
            project_id=project_id,
            sections=tuple(sections),
            citations=tuple(dict.fromkeys(citations)),
            used_assistant=used_assistant,
        )

    def _default_title(self, kind: str, inputs: _DraftInputs) -> str:
        project = inputs.summary.project
        return f"{report_kind_label(kind)} — {project.title or project.number or project.record_id}"

    def _fill_with_assistant(
        self, kind: str, sections: list[ReportSection], *, profile: str
    ) -> tuple[list[ReportSection], list[str], bool]:
        """Replace each section's lines with the assistant's cited answer.

        A section the assistant cannot answer keeps its deterministic lines, so a
        draft is never left blank because a model was unavailable.
        """
        from core.workplace.assistant import DEFAULT_PROFILE

        chosen = profile or DEFAULT_PROFILE
        filled: list[ReportSection] = []
        citations: list[str] = []
        used = False
        for section in sections:
            question = f"{report_kind_label(kind)}: {section.title} — what should it state?"
            try:
                answer = self.assistant.ask(question, profile=chosen)  # type: ignore[union-attr]
            except Exception:  # a model failure never breaks a draft
                filled.append(section)
                continue
            text = (answer.text or "").strip()
            if not text:
                filled.append(section)
                continue
            lines = tuple(line.strip() for line in text.splitlines() if line.strip())
            filled.append(ReportSection(section.title, lines, ai=True))
            citations.extend(answer.citations)
            used = True
        return filled, citations, used

    # -- the lifecycle ----------------------------------------------------

    def create(
        self,
        project_id: str,
        *,
        kind: str = "technical_report",
        title: str = "",
        body: str = "",
        author_id: str = "",
        division_id: str = "",
        status: str = "draft",
        number: str = "",
    ) -> Report:
        """Create a report row (a *new* record; the audit records the creation)."""
        if self.domain is None:
            raise ReportingError("This workflow is read-only (no domain service supplied).")
        if kind not in REPORT_KINDS:
            raise ReportingError(f"Unknown report kind {kind!r}.")
        if status not in REPORT_STATUSES:
            raise ReportingError(f"Unknown report status {status!r}.")
        report = self.domain.create_report(
            project_id=project_id,
            title=title.strip() or report_kind_label(kind),
            kind=kind,
            body=body,
            author_id=author_id,
            division_id=division_id,
            status=status,
            number=number,
        )
        return report

    def save_draft(
        self, project_id: str, draft: ReportDraft, *, author_id: str = "", number: str = ""
    ) -> Report:
        """Create a report row from a drafted report (its text becomes the body)."""
        return self.create(
            project_id,
            kind=draft.kind,
            title=draft.title,
            body=draft.to_text(),
            author_id=author_id,
            number=number,
            status="draft",
        )

    def _transition(self, report_id: str, target: str, **changes: Any) -> Report:
        if self.domain is None:
            raise ReportingError("This workflow is read-only (no domain service supplied).")
        report = self.repository.get_domain("reports", report_id)
        if not isinstance(report, Report):
            raise ReportingError(f"No report with id {report_id!r}.")
        if not can_transition(report.status, target):
            raise ReportingError(
                f"A report cannot move from {report.status!r} to {target!r}."
            )
        updated = self.domain.update("reports", report_id, status=target, **changes)
        assert isinstance(updated, Report)
        return updated

    def submit(self, report_id: str) -> Report:
        """Submit a draft for approval."""
        return self._transition(report_id, "submitted")

    def reject(self, report_id: str, *, note: str = "") -> Report:
        """Reject a submitted report (it returns to the draft state to be reworked)."""
        report = self._transition(report_id, "rejected")
        return report

    def approve(
        self, report_id: str, *, approved_by: str, role: str = "", note: str = ""
    ) -> Report:
        """Record a **human** approval of a submitted report and mark it approved.

        The approval is a *new* :class:`~core.workplace.records.Approval` row (a
        re-approval never overwrites the first), and the report moves to
        ``approved``. ``approved_by`` must name a person -- the AI cannot approve.
        """
        if not approved_by.strip():
            raise ReportingError("An approval must name the person who approved it.")
        from core.workplace.security import is_ai_actor

        if is_ai_actor(approved_by):
            raise ReportingError(
                "An approval must name a person; the AI can never approve a deliverable."
            )
        report = self._transition(report_id, "approved")
        self.domain.approve(  # type: ignore[union-attr]
            subject_id=report_id,
            subject_kind="reports",
            decision="approved",
            approved_by=approved_by.strip(),
            role=role,
            note=note,
        )
        return report

    def issue(
        self,
        report_id: str,
        *,
        register_document: bool = True,
        issued_on: str = "",
    ) -> Deliverable:
        """Issue an *approved* report and (optionally) register its deliverable.

        Issuing is refused until the report is approved, so a deliverable always
        carries a human approval behind it. The deliverable is registered through
        the Workplace's *own* document registry (Phase 6), never a second store.
        """
        if self.domain is None:
            raise ReportingError("This workflow is read-only (no domain service supplied).")
        report = self.repository.get_domain("reports", report_id)
        if not isinstance(report, Report):
            raise ReportingError(f"No report with id {report_id!r}.")
        if report.status != "approved":
            raise ReportingError(
                "A report can only be issued once it has been approved by a person."
            )
        document_id = ""
        document_title = ""
        if register_document and self.documents is not None:
            record = self.documents.register_text(
                report.body or report.title,
                title=report.title,
                doc_type=REPORT_DOC_TYPES.get(report.kind, "other"),
                project_id=report.project_id,
                division_id=report.division_id,
                author_id=report.author_id,
                approved_by=report.author_id,
                issued_on=issued_on,
            )
            document_id = record.record_id
            document_title = record.title
        updated = self._transition(report_id, "issued", issued_on=issued_on, document_id=document_id)
        return Deliverable(report=updated, document_id=document_id, document_title=document_title)

    def supersede(self, report_id: str) -> Report:
        """Mark a report superseded (its document stays, its row is never deleted)."""
        return self._transition(report_id, "superseded")

    # -- read -------------------------------------------------------------

    def reports(self, project_id: str, *, kind: str = "") -> list[Report]:
        rows = self.repository.list_domain(
            "reports", where="project_id = ?", params=(project_id,)
        )
        reports = [row for row in rows if isinstance(row, Report)]
        if kind:
            reports = [row for row in reports if row.kind == kind]
        reports.sort(key=lambda row: (row.issued_on, row.title), reverse=True)
        return reports

    def report(self, report_id: str) -> Report | None:
        row = self.repository.get_domain("reports", report_id)
        return row if isinstance(row, Report) else None

    def approvals(self, report_id: str) -> list[Any]:
        return self.repository.list_domain(
            "approvals", where="subject_id = ?", params=(report_id,)
        )

    def deliverables(self, project_id: str) -> list[Deliverable]:
        """Every *issued* deliverable of a project, newest first."""
        rows = [
            report
            for report in self.reports(project_id)
            if report.status == "issued" and report.kind in DELIVERABLE_KINDS
        ]
        out: list[Deliverable] = []
        for report in rows:
            document = (
                self.repository.get_domain("workplace_documents", report.document_id)
                if report.document_id
                else None
            )
            out.append(
                Deliverable(
                    report=report,
                    document_id=report.document_id,
                    document_title=getattr(document, "title", "") if document else "",
                )
            )
        return out


__all__ = [
    "DELIVERABLE_KINDS",
    "REPORT_DOC_TYPES",
    "REPORT_KINDS",
    "REPORT_STATUSES",
    "REPORT_TRANSITIONS",
    "Deliverable",
    "ReportDraft",
    "ReportSection",
    "ReportingError",
    "ReportingWorkflow",
    "can_transition",
    "report_kind_label",
    "status_label",
]
