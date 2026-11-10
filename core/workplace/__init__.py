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

from core.workplace.database import (
    SCHEMA_VERSION,
    WorkplaceDatabase,
    WorkplaceDatabaseError,
    WorkplaceLoadReport,
)
from core.workplace.domain import (
    ENTITY_NAMES,
    WorkplaceDomain,
    WorkplaceDomainError,
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
)
from core.workplace.repository import WorkplaceRepository
from core.workplace.seed import SeedReport, seed_workplace
from core.workplace.settings import (
    WorkplaceSettings,
    load_settings,
    save_settings,
)
from core.workplace.taxonomy import TAXONOMY, TaxonomyArea, area_for, discipline_count

__all__ = [
    "CONFIDENTIALITY_LEVELS",
    "DIVISION_DETAIL_KINDS",
    "DOCUMENT_TYPES",
    "DOMAIN_RECORDS",
    "ENTITY_NAMES",
    "PROJECT_STATUSES",
    "REFERENCE_KINDS",
    "SCHEMA_VERSION",
    "TAXONOMY",
    "Candidate",
    "CompetencyMatrix",
    "Division",
    "DivisionDetail",
    "Institution",
    "OrganisationalUnit",
    "PersonProfile",
    "PersonnelDirectory",
    "ProjectManager",
    "ProjectRow",
    "ProjectSummary",
    "RecordBase",
    "SeedReport",
    "TaxonomyArea",
    "TaxonomyNode",
    "WorkplaceDatabase",
    "WorkplaceDatabaseError",
    "WorkplaceDomain",
    "WorkplaceDomainError",
    "WorkplaceLoadReport",
    "WorkplaceRecord",
    "WorkplaceRepository",
    "WorkplaceSettings",
    "area_for",
    "discipline_count",
    "load_settings",
    "make_id",
    "save_settings",
    "seed_workplace",
]
