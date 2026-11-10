"""The Workplace: the institute's digital office inside Fayd al-Ruh.

This package is a **domain layer** for the Egyptian Petroleum Research Institute
(EPRI) and its Technical Support & Technology Center (TSTC) -- *معهد بحوث البترول
المصري – مركز الدعم الفني والتكنولوجي (بترول - صناعة)*. It lets the user manage
technical work, projects, staff, documents, standards, inspections, and client
relations, and it provides a specialised assistant that knows how the institute
works.

Two rules govern the whole package, and they are the reason it is shaped the way
it is:

* **No parallel knowledge system.** Documents, ingestion, retrieval, and model
  management already exist in this project (``core.documents``,
  ``core.knowledge_base``, ``core.knowledge_index``, ``core.rag``, ``ingest``,
  ``core.language_models``). The Workplace *reuses* them. Its own database stores
  the institutional *records* (projects, personnel, standards, inspections) and a
  **registry** of documents -- never a second copy of the corpus or a second RAG.
* **Nothing invented about the institution.** Only publicly documented EPRI
  information is seeded; the TSTC divisions ship as **proposed, configurable**
  defaults and are marked as such in the data and the UI.

Nothing here loads a model, imports Tk, or reaches the network at import time, so
the CLI never pulls in the window and a headless run stays cheap.

This module (Phase 2) establishes the foundation: the record model, the versioned
SQLite database, the typed repository, and the paths. The institutional hierarchy,
taxonomy, projects, personnel, standards, documents, inspections, reports, the
assistant, and security arrive in later phases, each additive.
"""

from __future__ import annotations

from core.workplace.assistant import (
    DEFAULT_CLEARANCE,
    DEFAULT_PROFILE,
    DISCLAIMER,
    PROFILE_KEYS,
    PROFILES,
    CorpusReport,
    WorkplaceAssistant,
    WorkplaceProfile,
    build_workplace_documents,
    profile_for,
)
from core.workplace.database import (
    SCHEMA_VERSION,
    WorkplaceDatabase,
    WorkplaceDatabaseError,
    WorkplaceLoadReport,
)
from core.workplace.documents import (
    DOCUMENT_TYPE_LABELS,
    DocumentManager,
    DocumentRecord,
    DocumentVerification,
    content_hash,
    document_type_label,
)
from core.workplace.domain import (
    ENTITY_NAMES,
    WorkplaceDomain,
    WorkplaceDomainError,
)
from core.workplace.knowledge_capture import (
    SOURCE_KINDS,
    CaptureReport,
    KnowledgeCandidate,
    KnowledgeCapture,
    WorkplaceKnowledgeError,
    source_label,
)
from core.workplace.library import (
    REFERENCE_KIND_LABELS,
    STANDARD_BODIES,
    RevisionComparison,
    StandardRevision,
    TechnicalLibrary,
    body_title,
    reference_kind_label,
)
from core.workplace.model import (
    DIVISION_DETAIL_KINDS,
    Division,
    DivisionDetail,
    Institution,
    OrganisationalUnit,
    RecordBase,
    TaxonomyNode,
    WorkplaceRecord,
    make_id,
)
from core.workplace.personnel import (
    Candidate,
    CompetencyMatrix,
    PersonnelDirectory,
    PersonProfile,
)
from core.workplace.projects import (
    ProjectManager,
    ProjectRow,
    ProjectSummary,
)
from core.workplace.records import (
    CONFIDENTIALITY_LEVELS,
    DOCUMENT_TYPES,
    DOMAIN_RECORDS,
    PROJECT_STATUSES,
    REFERENCE_KINDS,
    ProjectMember,
    WorkplaceUser,
)
from core.workplace.reporting import (
    REPORT_KINDS,
    REPORT_STATUSES,
    REPORT_TRANSITIONS,
    Deliverable,
    ReportDraft,
    ReportingError,
    ReportingWorkflow,
    ReportSection,
    can_transition,
    report_kind_label,
    status_label,
)
from core.workplace.repository import WorkplaceRepository
from core.workplace.search import (
    SearchHit,
    SearchReport,
    search_workplace,
    searchable_kinds,
)
from core.workplace.security import (
    AI_ACTORS,
    DEFAULT_ROLE,
    PERMISSIONS,
    ROLES,
    AccessDecision,
    ReadReport,
    SecurityManager,
    WorkplaceSecurityError,
    is_ai_actor,
    permissions_for,
    role_ceiling,
)
from core.workplace.seed import SeedReport, seed_workplace
from core.workplace.settings import (
    WorkplaceSettings,
    load_settings,
    save_settings,
)
from core.workplace.taxonomy import TAXONOMY, TaxonomyArea, area_for, discipline_count

__all__ = [
    "AI_ACTORS",
    "CONFIDENTIALITY_LEVELS",
    "DEFAULT_CLEARANCE",
    "DEFAULT_PROFILE",
    "DEFAULT_ROLE",
    "DISCLAIMER",
    "DIVISION_DETAIL_KINDS",
    "DOCUMENT_TYPES",
    "DOCUMENT_TYPE_LABELS",
    "DOMAIN_RECORDS",
    "ENTITY_NAMES",
    "PERMISSIONS",
    "PROFILES",
    "PROFILE_KEYS",
    "PROJECT_STATUSES",
    "REFERENCE_KINDS",
    "REFERENCE_KIND_LABELS",
    "REPORT_KINDS",
    "REPORT_STATUSES",
    "REPORT_TRANSITIONS",
    "ROLES",
    "SCHEMA_VERSION",
    "SOURCE_KINDS",
    "STANDARD_BODIES",
    "TAXONOMY",
    "AccessDecision",
    "Candidate",
    "CaptureReport",
    "CompetencyMatrix",
    "CorpusReport",
    "Deliverable",
    "Division",
    "DivisionDetail",
    "DocumentManager",
    "DocumentRecord",
    "DocumentVerification",
    "Institution",
    "KnowledgeCandidate",
    "KnowledgeCapture",
    "OrganisationalUnit",
    "PersonProfile",
    "PersonnelDirectory",
    "ProjectManager",
    "ProjectMember",
    "ProjectRow",
    "ProjectSummary",
    "ReadReport",
    "RecordBase",
    "ReportDraft",
    "ReportSection",
    "ReportingError",
    "ReportingWorkflow",
    "RevisionComparison",
    "SearchHit",
    "SearchReport",
    "SecurityManager",
    "SeedReport",
    "StandardRevision",
    "TaxonomyArea",
    "TaxonomyNode",
    "TechnicalLibrary",
    "WorkplaceAssistant",
    "WorkplaceDatabase",
    "WorkplaceDatabaseError",
    "WorkplaceDomain",
    "WorkplaceDomainError",
    "WorkplaceKnowledgeError",
    "WorkplaceLoadReport",
    "WorkplaceProfile",
    "WorkplaceRecord",
    "WorkplaceRepository",
    "WorkplaceSecurityError",
    "WorkplaceSettings",
    "WorkplaceUser",
    "area_for",
    "body_title",
    "build_workplace_documents",
    "can_transition",
    "content_hash",
    "discipline_count",
    "document_type_label",
    "is_ai_actor",
    "load_settings",
    "make_id",
    "permissions_for",
    "profile_for",
    "reference_kind_label",
    "report_kind_label",
    "role_ceiling",
    "save_settings",
    "search_workplace",
    "searchable_kinds",
    "seed_workplace",
    "source_label",
    "status_label",
]
