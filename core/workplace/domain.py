"""The Workplace's domain service: typed, audited access to the work records.

:class:`~core.workplace.repository.WorkplaceRepository` reads and writes rows;
this module is the *service* over it. It gives every domain entity a small,
typed API (create / read / list / update / deactivate) that appends an **audit
row** on every write, so the trail the security phase needs is complete from the
first record, and it never mutates a record in place -- an edit produces a new
record value (a bumped ``updated_at``) and a new audit row, exactly the discipline
the rest of the project uses for revisions.

It is deliberately a thin layer and it is Tk-free, model-free, and network-free,
so the window, the CLI, and the tests all drive the *same* service.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, TypeVar

from core.workplace.model import RecordBase, utc_now
from core.workplace.records import (
    DOMAIN_RECORDS,
    Approval,
    Assignment,
    AuditEntry,
    Client,
    Contract,
    Correspondence,
    Finding,
    Inspection,
    KnowledgeRecord,
    Person,
    Procedure,
    Project,
    Qualification,
    Report,
    Standard,
    Test,
    WorkplaceDocument,
)
from core.workplace.repository import WorkplaceRepository

R = TypeVar("R", bound=RecordBase)

#: The entity names a caller may pass to :meth:`WorkplaceDomain.create`. They map
#: one-to-one onto :data:`core.workplace.records.DOMAIN_RECORDS`, so the friendly
#: name and the table name are the same thing.
ENTITY_NAMES: tuple[str, ...] = tuple(DOMAIN_RECORDS)


class WorkplaceDomainError(RuntimeError):
    """A domain operation could not be completed; the message says why."""


class WorkplaceDomain:
    """Typed, audited CRUD over the Workplace's work records.

    ``actor`` is the name recorded on every audit row (the security phase will
    supply a real user; a headless run uses ``"system"``). Nothing here is
    destructive: :meth:`deactivate` sets the soft-delete flag and records it, so
    history that references the record survives.
    """

    def __init__(self, repository: WorkplaceRepository, *, actor: str = "system") -> None:
        self.repository = repository
        self.actor = actor or "system"

    # -- generic ----------------------------------------------------------

    def _record_type(self, entity: str) -> type[RecordBase]:
        record_type = DOMAIN_RECORDS.get(entity)
        if record_type is None:
            raise WorkplaceDomainError(f"Unknown Workplace entity: {entity!r}")
        return record_type

    def create(self, entity: str, **fields: Any) -> Any:
        """Create a record of ``entity`` and record the creation in the audit."""
        try:
            record = self.repository.create_domain(entity, **fields)
        except KeyError as error:
            raise WorkplaceDomainError(f"Unknown Workplace entity: {entity!r}") from error
        self._audit("create", entity, record)
        return record

    def get(self, entity: str, record_id: str) -> Any:
        self._record_type(entity)  # validate the entity name
        return self.repository.get_domain(entity, record_id)

    def list_records(
        self,
        entity: str,
        *,
        where: str = "",
        params: Sequence[Any] = (),
        include_inactive: bool = False,
        limit: int | None = None,
    ) -> list[Any]:
        self._record_type(entity)
        return self.repository.list_domain(
            entity,
            where=where,
            params=params,
            include_inactive=include_inactive,
            limit=limit,
        )

    def update(self, entity: str, record_id: str, **changes: Any) -> Any:
        """Apply ``changes`` to a record as a *new* value and audit the edit.

        A frozen record is never mutated: the change is applied with
        :func:`dataclasses.replace`, the ``updated_at`` stamp is bumped, and the
        result is written. The ``before``/``after`` summaries on the audit row
        name only the fields that changed, so the trail is readable.
        """
        current = self.repository.get_domain(entity, record_id)
        if current is None:
            raise WorkplaceDomainError(f"No {entity} with id {record_id!r}")
        unknown = set(changes) - {f.name for f in dataclasses.fields(current)}
        if unknown:
            raise WorkplaceDomainError(
                f"{entity} has no field(s): {', '.join(sorted(unknown))}"
            )
        updated = dataclasses.replace(current, updated_at=utc_now(), **changes)
        self.repository.save_domain(entity, updated)
        changed = ", ".join(sorted(changes)) or "(no field)"
        self._audit("update", entity, updated, detail=changed,
                    before=_summary(current), after=_summary(updated))
        return updated

    def deactivate(self, entity: str, record_id: str) -> Any:
        updated = self.repository.deactivate_domain(entity, record_id)
        if updated is None:
            raise WorkplaceDomainError(f"No {entity} with id {record_id!r}")
        self._audit("deactivate", entity, updated)
        return updated

    # -- the audit trail --------------------------------------------------

    def _audit(
        self,
        action: str,
        entity: str,
        record: Any,
        *,
        detail: str = "",
        before: str = "",
        after: str = "",
    ) -> AuditEntry:
        return self.repository.audit(
            actor=self.actor,
            action=action,
            entity=entity,
            entity_id=getattr(record, "record_id", ""),
            detail=detail,
            before=before,
            after=after,
        )

    def record_audit(
        self, action: str, entity: str, *, entity_id: str = "", detail: str = ""
    ) -> AuditEntry:
        """Record an action that is not a plain create/update (an approval, ...)."""
        return self.repository.audit(
            actor=self.actor, action=action, entity=entity, entity_id=entity_id, detail=detail
        )

    def audit_tail(self, *, limit: int = 50) -> list[AuditEntry]:
        return self.repository.audit_tail(limit=limit)

    def audit_for(self, entity_id: str, *, limit: int = 50) -> list[AuditEntry]:
        return self.repository.audit_for(entity_id, limit=limit)

    # -- personnel --------------------------------------------------------

    def create_person(self, *, name: str, **fields: Any) -> Person:
        return self.create("people", name=name, **fields)

    def people(self, *, division_id: str | None = None) -> list[Person]:
        where, params = _filter("division_id", division_id)
        rows = self.list_records("people", where=where, params=params)
        rows.sort(key=lambda row: row.name.casefold())
        return rows

    def add_qualification(self, *, person_id: str, title: str, **fields: Any) -> Qualification:
        self._require("people", person_id)
        return self.create("qualifications", person_id=person_id, title=title, **fields)

    def qualifications(self, person_id: str) -> list[Qualification]:
        return self.list_records("qualifications", where="person_id = ?", params=(person_id,))

    # -- clients ----------------------------------------------------------

    def create_client(self, *, name: str, **fields: Any) -> Client:
        return self.create("clients", name=name, **fields)

    def clients(self) -> list[Client]:
        rows = self.list_records("clients")
        rows.sort(key=lambda row: row.name.casefold())
        return rows

    # -- projects & contracts ---------------------------------------------

    def create_project(self, *, number: str, title: str, **fields: Any) -> Project:
        for field_name in ("client_id", "contractor_id", "consultant_id"):
            if fields.get(field_name):
                self._require("clients", fields[field_name])
        if fields.get("division_id"):
            self._require("divisions", fields["division_id"])
        if fields.get("manager_id"):
            self._require("people", fields["manager_id"])
        return self.create("projects", number=number, title=title, **fields)

    def projects(self, *, status: str | None = None, division_id: str | None = None) -> list[Project]:
        where, params = _filter("status", status, "division_id", division_id)
        rows = self.list_records("projects", where=where, params=params)
        rows.sort(key=lambda row: (row.number, row.title))
        return rows

    def project_by_number(self, number: str) -> Project | None:
        rows = self.list_records("projects", where="number = ?", params=(number,), limit=1)
        return rows[0] if rows else None

    def create_contract(self, *, project_id: str, number: str = "", **fields: Any) -> Contract:
        self._require("projects", project_id)
        return self.create("contracts", project_id=project_id, number=number, **fields)

    def contracts(self, project_id: str) -> list[Contract]:
        return self.list_records("contracts", where="project_id = ?", params=(project_id,))

    # -- standards & procedures -------------------------------------------

    def create_standard(self, *, number: str, title: str = "", **fields: Any) -> Standard:
        return self.create("standards", number=number, title=title, **fields)

    def standards(self, *, body: str | None = None) -> list[Standard]:
        where, params = _filter("body", body)
        rows = self.list_records("standards", where=where, params=params)
        rows.sort(key=lambda row: (row.body, row.number))
        return rows

    def create_procedure(self, *, title: str, **fields: Any) -> Procedure:
        if fields.get("division_id"):
            self._require("divisions", fields["division_id"])
        if fields.get("project_id"):
            self._require("projects", fields["project_id"])
        return self.create("procedures", title=title, **fields)

    # -- inspections & tests ----------------------------------------------

    def create_inspection(self, *, project_id: str, title: str, **fields: Any) -> Inspection:
        self._require("projects", project_id)
        return self.create("inspections", project_id=project_id, title=title, **fields)

    def inspections(self, *, project_id: str | None = None, status: str | None = None) -> list[Inspection]:
        where, params = _filter("project_id", project_id, "status", status)
        rows = self.list_records("inspections", where=where, params=params)
        rows.sort(key=lambda row: (row.scheduled_on, row.title))
        return rows

    def create_test(self, *, project_id: str, title: str = "", **fields: Any) -> Test:
        self._require("projects", project_id)
        return self.create("tests", project_id=project_id, title=title, **fields)

    def tests(self, *, project_id: str | None = None) -> list[Test]:
        where, params = _filter("project_id", project_id)
        return self.list_records("tests", where=where, params=params)

    # -- documents & reports ----------------------------------------------

    def register_document(self, *, title: str, **fields: Any) -> WorkplaceDocument:
        """Register a document (a *metadata row*, never a copy of the file)."""
        if fields.get("project_id"):
            self._require("projects", fields["project_id"])
        return self.create("workplace_documents", title=title, **fields)

    def documents(self, *, project_id: str | None = None, doc_type: str | None = None) -> list[WorkplaceDocument]:
        where, params = _filter("project_id", project_id, "doc_type", doc_type)
        rows = self.list_records("workplace_documents", where=where, params=params)
        rows.sort(key=lambda row: (row.issued_on, row.title), reverse=True)
        return rows

    def document_history(self, record_id: str) -> list[WorkplaceDocument]:
        """A document's revision chain: the record and everything it superseded."""
        chain: list[WorkplaceDocument] = []
        seen: set[str] = set()
        current = self.get("workplace_documents", record_id)
        while isinstance(current, WorkplaceDocument) and current.record_id not in seen:
            seen.add(current.record_id)
            chain.append(current)
            if not current.supersedes_id:
                break
            current = self.get("workplace_documents", current.supersedes_id)
        return chain

    def create_report(self, *, project_id: str, title: str, **fields: Any) -> Report:
        self._require("projects", project_id)
        return self.create("reports", project_id=project_id, title=title, **fields)

    def reports(self, *, project_id: str | None = None, status: str | None = None) -> list[Report]:
        where, params = _filter("project_id", project_id, "status", status)
        rows = self.list_records("reports", where=where, params=params)
        rows.sort(key=lambda row: (row.issued_on, row.title), reverse=True)
        return rows

    # -- correspondence, assignments, findings, approvals, knowledge ------

    def create_correspondence(self, *, subject: str, **fields: Any) -> Correspondence:
        if fields.get("project_id"):
            self._require("projects", fields["project_id"])
        return self.create("correspondence", subject=subject, **fields)

    def correspondence(self, project_id: str) -> list[Correspondence]:
        rows = self.list_records("correspondence", where="project_id = ?", params=(project_id,))
        rows.sort(key=lambda row: (row.sent_on, row.subject))
        return rows

    def assign(self, *, project_id: str, person_id: str, role: str = "", **fields: Any) -> Assignment:
        self._require("projects", project_id)
        self._require("people", person_id)
        return self.create("assignments", project_id=project_id, person_id=person_id, role=role, **fields)

    def assignments(self, *, project_id: str | None = None, person_id: str | None = None) -> list[Assignment]:
        where, params = _filter("project_id", project_id, "person_id", person_id)
        return self.list_records("assignments", where=where, params=params)

    def create_finding(self, *, project_id: str, title: str, **fields: Any) -> Finding:
        self._require("projects", project_id)
        return self.create("findings", project_id=project_id, title=title, **fields)

    def findings(self, *, project_id: str | None = None, status: str | None = None) -> list[Finding]:
        where, params = _filter("project_id", project_id, "status", status)
        rows = self.list_records("findings", where=where, params=params)
        rows.sort(key=lambda row: (row.raised_on, row.title), reverse=True)
        return rows

    def approve(
        self, *, subject_id: str, subject_kind: str, decision: str = "approved", **fields: Any
    ) -> Approval:
        """Record an approval (a *new* record, so a re-approval never overwrites)."""
        self._require(subject_kind, subject_id)
        approval = self.create(
            "approvals", subject_id=subject_id, subject_kind=subject_kind, decision=decision, **fields
        )
        self._audit("approve", subject_kind, approval, detail=decision)
        return approval

    def approvals(self, subject_id: str) -> list[Approval]:
        return self.list_records("approvals", where="subject_id = ?", params=(subject_id,))

    def add_knowledge(self, *, title: str, **fields: Any) -> KnowledgeRecord:
        """Register a knowledge record (a *registry row*, never a second corpus)."""
        if fields.get("project_id"):
            self._require("projects", fields["project_id"])
        return self.create("knowledge_records", title=title, **fields)

    def knowledge(self, *, project_id: str | None = None) -> list[KnowledgeRecord]:
        where, params = _filter("project_id", project_id)
        rows = self.list_records("knowledge_records", where=where, params=params)
        rows.sort(key=lambda row: row.title.casefold())
        return rows

    # -- helpers ----------------------------------------------------------

    def _require(self, entity: str, record_id: str) -> None:
        if not record_id:
            return
        if self.repository.get_domain(entity, record_id) is None:
            raise WorkplaceDomainError(f"No {entity} with id {record_id!r}")


def _filter(*pairs: Any) -> tuple[str, tuple[Any, ...]]:
    """Build a ``WHERE`` clause and params from ``(column, value)`` pairs."""
    clauses: list[str] = []
    params: list[Any] = []
    for column, value in zip(pairs[::2], pairs[1::2], strict=True):
        if value is not None:
            clauses.append(f"{column} = ?")
            params.append(value)
    return " AND ".join(clauses), tuple(params)


def _summary(record: RecordBase) -> str:
    """A short, readable one-line summary of a record for the audit trail."""
    parts: list[str] = []
    for name in ("title", "name", "subject", "number"):
        value = getattr(record, name, "")
        if value:
            parts.append(str(value))
            break
    parts.append(record.record_id)
    return " · ".join(parts)


__all__ = [
    "ENTITY_NAMES",
    "WorkplaceDomain",
    "WorkplaceDomainError",
]
