"""The Workplace's domain records: projects, personnel, standards, documents, ...

Phase 2/3 modelled the institutional hierarchy (institutions, units, divisions,
their profiles, and the taxonomy). This module adds the records the *work* needs
(the brief's §12 relational model): personnel and their qualifications, clients,
projects and contracts, standards and procedures, inspections and tests,
documents and reports, correspondence, assignments, technical findings,
approvals, and knowledge records -- plus the append-only **audit trail**.

Every record is a frozen dataclass built on :class:`~core.workplace.model.RecordBase`,
so it carries the same stable id, the ``created_at``/``updated_at`` audit pair,
and the ``active`` **soft-delete** flag. Deleting (deactivating) an organisational
unit, a division, or a person therefore never destroys the historical project
records that reference it: the reference is a foreign key to a row that still
exists.

**Documents and knowledge are registries, never a second store.** A
:class:`WorkplaceDocument` records a document's *metadata* and points at a file
the project's own ingestion/document system already owns (``path`` +
``content_hash``); a :class:`KnowledgeRecord` records a piece of institutional
knowledge and points at the project's own knowledge store (``source`` +
``reference``). Nothing here copies a corpus, re-implements chunking, or adds a
second RAG -- exactly the rule the package docstring sets out.

Keeping the records free of I/O is what makes the whole Workplace testable with no
database file and no display.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from core.workplace.model import RecordBase

#: The project lifecycle states (a small, closed set).
PROJECT_STATUSES = (
    "proposed",
    "awarded",
    "active",
    "on_hold",
    "completed",
    "closed",
    "cancelled",
)

#: The confidentiality classifications a record may carry (the brief's §13).
CONFIDENTIALITY_LEVELS = ("public", "internal", "confidential", "restricted")

#: The kinds of document the Workplace tracks (the brief's §9).
DOCUMENT_TYPES = (
    "proposal",          # technical / commercial proposal
    "inspection_report",
    "technical_report",
    "method_statement",
    "itp",               # inspection and test plan
    "welding_document",
    "coating_procedure",
    "pqt",               # procedure qualification test / record
    "ncr",               # nonconformance report
    "car",               # corrective action request
    "correspondence",
    "minutes",           # meeting minutes
    "contract",
    "other",
)

#: The kinds of standard / reference the technical library distinguishes (the
#: brief's §7): official standards, internal procedures, project specifications,
#: and technical guidance must never be confused.
REFERENCE_KINDS = ("standard", "internal_procedure", "project_specification", "guidance")


@dataclass(frozen=True)
class Person(RecordBase):
    """A member of staff: an engineer, chemist, geologist, inspector, ..."""

    name: str = ""
    name_ar: str = ""
    role: str = ""
    discipline: str = ""
    division_id: str = ""
    email: str = ""
    phone: str = ""
    available: bool = True
    summary: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Qualification(RecordBase):
    """A person's qualification, certification, or training record."""

    person_id: str = ""
    kind: str = "certification"  # degree | certification | training | membership
    title: str = ""
    issuer: str = ""
    reference: str = ""
    issued_on: str = ""
    expires_on: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Client(RecordBase):
    """A client, contractor, or consultant the Workplace works with."""

    name: str = ""
    name_ar: str = ""
    kind: str = "client"  # client | contractor | consultant | authority
    country: str = ""
    contact: str = ""
    email: str = ""
    phone: str = ""
    summary: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Project(RecordBase):
    """A project (the brief's §6): number, title, client, scope, status, dates."""

    number: str = ""
    title: str = ""
    title_ar: str = ""
    client_id: str = ""
    contractor_id: str = ""
    consultant_id: str = ""
    division_id: str = ""
    status: str = "proposed"
    scope: str = ""
    location: str = ""
    start_date: str = ""
    end_date: str = ""
    manager_id: str = ""
    confidentiality: str = "internal"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Contract(RecordBase):
    """A project's contract: number, value, dates, and key terms."""

    project_id: str = ""
    number: str = ""
    title: str = ""
    client_id: str = ""
    value: str = ""
    currency: str = ""
    signed_on: str = ""
    starts_on: str = ""
    ends_on: str = ""
    terms: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Standard(RecordBase):
    """A technical reference in the library (the brief's §7).

    ``reference_kind`` keeps the four families apart -- an *official standard*
    (API, ASME, ISO, ...), an *internal procedure*, a *project specification*, and
    *technical guidance* are never presented as the same thing. ``body`` is the
    issuing body (API, ASME, ASTM, ISO, AWS, AMPP/NACE, DNV, EN, IEC, ...).
    """

    body: str = ""
    number: str = ""
    title: str = ""
    edition: str = ""
    revision: str = ""
    published_on: str = ""
    reference_kind: str = "standard"
    applicability: str = ""
    status: str = "active"
    source: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Procedure(RecordBase):
    """A procedure or work instruction, optionally a division's own."""

    division_id: str = ""
    project_id: str = ""
    number: str = ""
    title: str = ""
    revision: str = ""
    summary: str = ""
    body: str = ""
    approved_by: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Inspection(RecordBase):
    """An inspection, witness point, or surveillance visit (the brief's §8)."""

    project_id: str = ""
    division_id: str = ""
    kind: str = "inspection"  # inspection | witness | surveillance | audit
    title: str = ""
    itp_reference: str = ""
    scheduled_on: str = ""
    performed_on: str = ""
    inspector_id: str = ""
    status: str = "planned"
    result: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Test(RecordBase):
    """A test or examination performed (mechanical, NDT, chemical, coating, ...)."""

    project_id: str = ""
    inspection_id: str = ""
    kind: str = ""
    title: str = ""
    standard_reference: str = ""
    performed_on: str = ""
    result: str = ""
    value: str = ""
    unit: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkplaceDocument(RecordBase):
    """A **registry row** for a document the project already owns (the brief's §9).

    This is deliberately *not* a document store: it records the document's
    metadata and points at a file the project's own reader/ingestion system owns
    (``path`` + ``content_hash``). Revision history is a chain of these rows
    (``revision`` + ``supersedes_id``), so a new revision is a new record and the
    original is never mutated -- the same "a revision is a new record" discipline
    the learning records use. ``confidentiality`` carries the classification.
    """

    project_id: str = ""
    division_id: str = ""
    doc_type: str = "other"
    title: str = ""
    title_ar: str = ""
    path: str = ""
    content_hash: str = ""
    revision: str = ""
    supersedes_id: str = ""
    confidentiality: str = "internal"
    author_id: str = ""
    approved_by: str = ""
    issued_on: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Report(RecordBase):
    """A technical report or progress report, with its approval state."""

    project_id: str = ""
    division_id: str = ""
    kind: str = "technical"  # technical | progress | inspection | final | management
    number: str = ""
    title: str = ""
    summary: str = ""
    body: str = ""
    document_id: str = ""
    author_id: str = ""
    status: str = "draft"
    issued_on: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Correspondence(RecordBase):
    """A letter, memo, or technical query exchanged about a project."""

    project_id: str = ""
    kind: str = "letter"  # letter | memo | query | response | transmittal
    reference: str = ""
    subject: str = ""
    direction: str = "outgoing"  # incoming | outgoing
    correspondent: str = ""
    body: str = ""
    sent_on: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Assignment(RecordBase):
    """A person assigned to a project (with the role and the period)."""

    project_id: str = ""
    person_id: str = ""
    role: str = ""
    from_date: str = ""
    to_date: str = ""
    allocation: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Finding(RecordBase):
    """A technical finding, nonconformity, or observation (NCR/CAR, the brief's §8)."""

    project_id: str = ""
    division_id: str = ""
    kind: str = "observation"  # observation | ncr | car | comment
    number: str = ""
    title: str = ""
    description: str = ""
    severity: str = "minor"  # minor | major | critical
    status: str = "open"  # open | answered | closed
    raised_on: str = ""
    closed_on: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Approval(RecordBase):
    """An approval of a record (a report, a document, a finding's response).

    ``subject_id`` is the id of the record being approved, ``subject_kind`` its
    kind, and ``decision`` one of ``approved``/``rejected``/``for_review``. An
    approval is a *new* record, so a re-approval never overwrites the first.
    """

    subject_id: str = ""
    subject_kind: str = ""
    decision: str = "for_review"
    approved_by: str = ""
    role: str = ""
    note: str = ""
    decided_on: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeRecord(RecordBase):
    """A piece of institutional knowledge, pointing at the project's own store.

    Like :class:`WorkplaceDocument`, this is a **registry**: it records that a
    piece of knowledge exists, its provenance (``source`` + ``reference``), its
    confidentiality, and how far it has been verified -- but the knowledge itself
    lives in the project's own knowledge base, never a second copy here.
    """

    project_id: str = ""
    division_id: str = ""
    title: str = ""
    summary: str = ""
    kind: str = "lesson"  # lesson | best_practice | procedure_note | technical_note
    source: str = ""
    reference: str = ""
    confidentiality: str = "internal"
    verified: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditEntry(RecordBase):
    """One append-only audit-trail row (the brief's §13).

    Every create/edit/deactivate/approval writes one row here: *who* did it
    (``actor``), *what* was acted on (``entity`` + ``entity_id``), *which*
    operation, the before/after summary, and when. The trail is append-only --
    nothing rewrites or deletes a row -- so it is a genuine record of what
    happened, which is what the security phase builds on.
    """

    actor: str = ""
    action: str = ""
    entity: str = ""
    entity_id: str = ""
    detail: str = ""
    before: str = ""
    after: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


#: Every domain record this module defines, by table name. The schema and the
#: repository both read this, so adding a record is one entry here plus its table.
DOMAIN_RECORDS: dict[str, type[RecordBase]] = {
    "people": Person,
    "qualifications": Qualification,
    "clients": Client,
    "projects": Project,
    "contracts": Contract,
    "standards": Standard,
    "procedures": Procedure,
    "inspections": Inspection,
    "tests": Test,
    "workplace_documents": WorkplaceDocument,
    "reports": Report,
    "correspondence": Correspondence,
    "assignments": Assignment,
    "findings": Finding,
    "approvals": Approval,
    "knowledge_records": KnowledgeRecord,
    "audit_entries": AuditEntry,
}

#: The readable id prefix each domain table's records carry (``people`` ->
#: ``PER``), so a bare id names its own kind. The table name is plural while the
#: prefix is singular, which is why this is a map and not a rule.
DOMAIN_PREFIXES: dict[str, str] = {
    "people": "PER",
    "qualifications": "QUAL",
    "clients": "CLI",
    "projects": "PRJ",
    "contracts": "CON",
    "standards": "STD",
    "procedures": "PROC",
    "inspections": "INSP",
    "tests": "TEST",
    "workplace_documents": "DOC",
    "reports": "REP",
    "correspondence": "COR",
    "assignments": "ASG",
    "findings": "FND",
    "approvals": "APR",
    "knowledge_records": "KNW",
    "audit_entries": "AUD",
}


__all__ = [
    "CONFIDENTIALITY_LEVELS",
    "DOCUMENT_TYPES",
    "DOMAIN_RECORDS",
    "PROJECT_STATUSES",
    "REFERENCE_KINDS",
    "Approval",
    "Assignment",
    "AuditEntry",
    "Client",
    "Contract",
    "Correspondence",
    "Finding",
    "Inspection",
    "KnowledgeRecord",
    "Person",
    "Procedure",
    "Project",
    "Qualification",
    "Report",
    "Standard",
    "Test",
    "WorkplaceDocument",
]
