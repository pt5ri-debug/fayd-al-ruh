"""Project management for the Workplace (the brief's §6).

Phase 4 gave every project its own row and every supporting record its own table.
This module assembles those rows into the *view* a project manager needs: one
project with its client/contractor/consultant resolved, its contract, its
inspections and tests, its documents and reports, its correspondence, the people
assigned to it, its findings and approvals, and its knowledge records.

It is deliberately **read-only**: it reads the Workplace's own tables and resolves
each reference to the record it points at. Nothing here invents a record, copies a
document, or re-implements retrieval -- the brief's §11 rule. A reference that does
not resolve (a client the Workplace does not hold) is reported as ``None`` rather
than fabricated, and an unknown project id yields ``None`` rather than a guess.

The counts on :class:`ProjectSummary` (open findings, reports awaiting approval,
...) are the same honest figures the dashboard shows, computed from the project's
own rows, so a surface never displays a made-up number.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.workplace.model import Division
from core.workplace.records import (
    Approval,
    Assignment,
    Client,
    Contract,
    Correspondence,
    Finding,
    Inspection,
    KnowledgeRecord,
    Person,
    Project,
    Report,
    Test,
    WorkplaceDocument,
)
from core.workplace.repository import WorkplaceRepository

#: The project statuses that count as "open work" for the dashboard and a view.
ACTIVE_STATUSES = frozenset({"proposed", "awarded", "active", "on_hold"})

#: The report statuses that mean "not yet approved".
AWAITING_APPROVAL_STATUSES = frozenset({"draft", "submitted"})

#: The finding statuses that mean "still open".
OPEN_FINDING_STATUSES = frozenset({"open", "answered"})


@dataclass(frozen=True)
class ProjectSummary:
    """One project and every record that belongs to it, each reference resolved."""

    project: Project
    client: Client | None = None
    contractor: Client | None = None
    consultant: Client | None = None
    division: Division | None = None
    manager: Person | None = None
    contracts: tuple[Contract, ...] = ()
    inspections: tuple[Inspection, ...] = ()
    tests: tuple[Test, ...] = ()
    documents: tuple[WorkplaceDocument, ...] = ()
    reports: tuple[Report, ...] = ()
    correspondence: tuple[Correspondence, ...] = ()
    assignments: tuple[Assignment, ...] = ()
    findings: tuple[Finding, ...] = ()
    approvals: tuple[Approval, ...] = ()
    knowledge: tuple[KnowledgeRecord, ...] = ()

    # -- honest sub-views ------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self.project.status in ACTIVE_STATUSES

    @property
    def open_findings(self) -> tuple[Finding, ...]:
        """The findings that are not yet closed (an answered NCR is still open)."""
        return tuple(f for f in self.findings if f.status in OPEN_FINDING_STATUSES)

    @property
    def reports_awaiting_approval(self) -> tuple[Report, ...]:
        return tuple(r for r in self.reports if r.status in AWAITING_APPROVAL_STATUSES)

    @property
    def planned_inspections(self) -> tuple[Inspection, ...]:
        return tuple(i for i in self.inspections if i.status == "planned")

    def counts(self) -> dict[str, int]:
        """The project's own figures, every one a real count."""
        return {
            "contracts": len(self.contracts),
            "inspections": len(self.inspections),
            "tests": len(self.tests),
            "documents": len(self.documents),
            "reports": len(self.reports),
            "correspondence": len(self.correspondence),
            "assignments": len(self.assignments),
            "findings": len(self.findings),
            "open_findings": len(self.open_findings),
            "approvals": len(self.approvals),
            "knowledge": len(self.knowledge),
        }

    def describe(self, *, language: str = "en") -> str:
        """A readable one-block account of the project."""
        project = self.project
        if language == "ar":
            head = f"{project.number} — {project.title_ar or project.title}"
            lines = [
                head,
                f"الحالة: {project.status}",
                f"العميل: {self.client.name if self.client else '—'}",
                f"القسم: {self.division.name if self.division else '—'}",
            ]
            counts = self.counts()
            lines.append(
                " · ".join(f"{key}: {value}" for key, value in counts.items() if value)
            )
            return "\n".join(lines)
        head = f"{project.number} — {project.title}"
        lines = [
            head,
            f"Status: {project.status}",
            f"Client: {self.client.name if self.client else '—'}",
            f"Division: {self.division.name if self.division else '—'}",
        ]
        counts = self.counts()
        detail = " · ".join(f"{key}: {value}" for key, value in counts.items() if value)
        if detail:
            lines.append(detail)
        return "\n".join(lines)


@dataclass(frozen=True)
class ProjectRow:
    """A light project row for a list: the project plus a few real counts."""

    project: Project
    client_name: str = ""
    division_name: str = ""
    open_findings: int = 0
    reports_awaiting_approval: int = 0
    documents: int = 0
    assignments: int = 0

    def describe(self, *, language: str = "en") -> str:
        if language == "ar":
            return f"{self.project.number} — {self.project.title_ar or self.project.title} ({self.project.status})"
        return f"{self.project.number} — {self.project.title} ({self.project.status})"


class ProjectManager:
    """Read-only project views over the Workplace repository.

    Built around the repository the service already holds, so it never opens a
    second database. Every method is a *read*: it resolves references and counts
    rows, and it returns ``None`` or an empty tuple rather than inventing data.
    """

    def __init__(self, repository: WorkplaceRepository) -> None:
        self.repository = repository

    # -- resolution helpers ----------------------------------------------

    def _get(self, table: str, record_id: str) -> object | None:
        if not record_id:
            return None
        return self.repository.get_domain(table, record_id)

    def _list(self, table: str, column: str, value: str) -> list:
        return self.repository.list_domain(table, where=f"{column} = ?", params=(value,))

    def project(self, record_id: str) -> Project | None:
        record = self.repository.get_domain("projects", record_id)
        return record if isinstance(record, Project) else None

    def project_by_number(self, number: str) -> Project | None:
        rows = self.repository.list_domain(
            "projects", where="number = ?", params=(number,), limit=1
        )
        return rows[0] if rows else None

    def resolve(self, project_id: str) -> ProjectSummary | None:
        """The full view of one project, or ``None`` for an unknown id."""
        project = self.project(project_id)
        if project is None:
            return None
        client = self._get("clients", project.client_id)
        contractor = self._get("clients", project.contractor_id)
        consultant = self._get("clients", project.consultant_id)
        division = self._get("divisions", project.division_id)
        manager = self._get("people", project.manager_id)
        findings = self._list("findings", "project_id", project.record_id)
        return ProjectSummary(
            project=project,
            client=client if isinstance(client, Client) else None,
            contractor=contractor if isinstance(contractor, Client) else None,
            consultant=consultant if isinstance(consultant, Client) else None,
            division=division if isinstance(division, Division) else None,
            manager=manager if isinstance(manager, Person) else None,
            contracts=tuple(self._list("contracts", "project_id", project.record_id)),
            inspections=tuple(self._list("inspections", "project_id", project.record_id)),
            tests=tuple(self._list("tests", "project_id", project.record_id)),
            documents=tuple(
                self._list("workplace_documents", "project_id", project.record_id)
            ),
            reports=tuple(self._list("reports", "project_id", project.record_id)),
            correspondence=tuple(
                self._list("correspondence", "project_id", project.record_id)
            ),
            assignments=tuple(self._list("assignments", "project_id", project.record_id)),
            findings=tuple(findings),
            approvals=tuple(self._approvals_for(project.record_id)),
            knowledge=tuple(
                self._list("knowledge_records", "project_id", project.record_id)
            ),
        )

    def _approvals_for(self, project_id: str) -> list[Approval]:
        """Approvals of the project itself *and* of every record that belongs to it.

        An approval names the record it approved (``subject_id``/``subject_kind``),
        so a project's approval set is the project's own approvals plus the ones
        recorded against its reports, documents, and findings.
        """
        subject_ids = {project_id}
        for table in ("reports", "workplace_documents", "findings"):
            subject_ids.update(
                row.record_id for row in self._list(table, "project_id", project_id)
            )
        approvals: list[Approval] = []
        for subject_id in sorted(subject_ids):
            approvals.extend(self._list("approvals", "subject_id", subject_id))
        return approvals

    # -- listings ---------------------------------------------------------

    def rows(
        self, *, status: str | None = None, division_id: str | None = None
    ) -> list[ProjectRow]:
        """A light listing of projects with their own real counts.

        The per-project counts are gathered from **one** read of each supporting
        table rather than one read per project, so listing a hundred projects does
        not issue four hundred queries (the N+1 the first version had).
        """
        where = []
        params: list[str] = []
        if status:
            where.append("status = ?")
            params.append(status)
        if division_id:
            where.append("division_id = ?")
            params.append(division_id)
        clause = " AND ".join(where)
        projects = self.repository.list_domain("projects", where=clause, params=tuple(params))
        projects.sort(key=lambda p: (p.number, p.title))

        # One pass over each supporting table, bucketed by project id.
        clients = {c.record_id: c for c in self.repository.list_domain("clients")}
        divisions = {d.record_id: d for d in self.repository.list_domain("divisions")}
        findings_by_project = self._count_by("findings", "project_id")
        reports_by_project = self._count_by("reports", "project_id")
        documents_by_project = self._count_by("workplace_documents", "project_id")
        assignments_by_project = self._count_by("assignments", "project_id")

        rows: list[ProjectRow] = []
        for project in projects:
            client = clients.get(project.client_id)
            division = divisions.get(project.division_id)
            rows.append(
                ProjectRow(
                    project=project,
                    client_name=client.name if isinstance(client, Client) else "",
                    division_name=division.name if isinstance(division, Division) else "",
                    open_findings=findings_by_project.get(project.record_id, {}).get(
                        "open", 0
                    ),
                    reports_awaiting_approval=reports_by_project.get(
                        project.record_id, {}
                    ).get("awaiting", 0),
                    documents=documents_by_project.get(project.record_id, {}).get(
                        "total", 0
                    ),
                    assignments=assignments_by_project.get(project.record_id, {}).get(
                        "total", 0
                    ),
                )
            )
        return rows

    def _count_by(self, table: str, column: str) -> dict[str, dict[str, int]]:
        """Bucket a supporting table's rows by ``column`` in **one** read.

        ``findings`` and ``reports`` count a status subset, so the bucket carries
        the two figures the listing shows (``open``/``awaiting``) beside the total;
        a table with no status subset just carries ``total``. This is the fix for
        the listing's N+1 reads.
        """
        counts: dict[str, dict[str, int]] = {}
        for record in self.repository.list_domain(table):
            key = getattr(record, column, "")
            if not key:
                continue
            bucket = counts.setdefault(key, {"total": 0, "open": 0, "awaiting": 0})
            bucket["total"] += 1
            status = getattr(record, "status", "")
            if status in OPEN_FINDING_STATUSES:
                bucket["open"] += 1
            if status in AWAITING_APPROVAL_STATUSES:
                bucket["awaiting"] += 1
        return counts

    # -- the project's own document repository ---------------------------

    def documents(self, project_id: str, *, doc_type: str | None = None) -> list[WorkplaceDocument]:
        """The project's document registry, optionally narrowed to one type."""
        rows = self._list("workplace_documents", "project_id", project_id)
        if doc_type:
            rows = [row for row in rows if row.doc_type == doc_type]
        rows.sort(key=lambda row: (row.issued_on, row.title), reverse=True)
        return rows


__all__ = [
    "ACTIVE_STATUSES",
    "AWAITING_APPROVAL_STATUSES",
    "OPEN_FINDING_STATUSES",
    "ProjectManager",
    "ProjectRow",
    "ProjectSummary",
]
