"""The Workplace service: the Tk-free half of the Workplace window.

This is where the Workplace's work actually happens -- resolve the database,
seed the defaults, read and edit the institutional hierarchy, report a status --
with no display, so the whole workflow is testable on a fresh clone. The window
and the CLI both drive *this* service, so they can never disagree about what the
Workplace holds.

It is deliberately a thin, honest layer over the domain package:

* the relational store is :class:`core.workplace.database.WorkplaceDatabase`;
* typed access is :class:`core.workplace.repository.WorkplaceRepository`;
* the defaults are :func:`core.workplace.seed.seed_workplace`;
* the taxonomy is :mod:`core.workplace.taxonomy`.

Everything is local: nothing here opens a network connection or loads a model. A
missing database or a refused edit is reported as a :class:`WorkplaceError`
carrying the real reason, never a silent failure. A read-only data folder is
handled honestly: the service still reports status, and an edit that cannot be
written says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.paths import workplace_db_path
from core.workplace.assistant import CorpusReport, WorkplaceAssistant
from core.workplace.database import WorkplaceDatabase
from core.workplace.documents import (
    DocumentManager,
    DocumentRecord,
    DocumentVerification,
    document_type_label,
)
from core.workplace.domain import WorkplaceDomain, WorkplaceDomainError
from core.workplace.library import (
    RevisionComparison,
    StandardRevision,
    TechnicalLibrary,
    reference_kind_label,
)
from core.workplace.model import (
    DIVISION_DETAIL_KINDS,
    Division,
    DivisionDetail,
    Institution,
    OrganisationalUnit,
    TaxonomyNode,
)
from core.workplace.personnel import Candidate, CompetencyMatrix, PersonnelDirectory, PersonProfile
from core.workplace.projects import ProjectManager, ProjectRow, ProjectSummary
from core.workplace.records import Assignment, Person
from core.workplace.repository import WorkplaceRepository
from core.workplace.seed import SeedReport, seed_workplace
from core.workplace.settings import (
    WorkplaceSettings,
    load_settings,
    save_settings,
)
from core.workplace.taxonomy import TAXONOMY, TaxonomyArea


class WorkplaceError(RuntimeError):
    """A Workplace operation could not be completed; the message says why."""


@dataclass
class DivisionSummary:
    """One division as a surface shows it: the readable fields plus its place."""

    record_id: str
    key: str
    name: str
    name_ar: str
    summary: str
    scope: str
    proposed: bool
    order: int
    active: bool
    unit_id: str
    unit_name: str = ""

    def label(self, *, language: str = "en") -> str:
        if language == "ar" and self.name_ar:
            return self.name_ar
        return self.name

    def describe(self, *, language: str = "en") -> str:
        parts = [self.label(language=language)]
        if self.proposed:
            parts.append("(proposed)" if language == "en" else "(مقترح)")
        if not self.active:
            parts.append("(inactive)" if language == "en" else "(غير نشط)")
        if self.summary:
            parts.append(f"— {self.summary}")
        return " ".join(parts)


@dataclass
class WorkplaceStatus:
    """An honest snapshot of the Workplace for the window's dashboard header."""

    db_path: str
    schema_version: int
    institutions: int
    units: int
    divisions: int
    active_divisions: int
    taxonomy_areas: int
    taxonomy_disciplines: int
    ready: bool
    message: str = ""

    def describe(self, *, language: str = "en") -> str:
        if language == "ar":
            return (
                f"قاعدة بيانات المكتب: {self.db_path} (الإصدار {self.schema_version}) — "
                f"{self.institutions} مؤسسة، {self.units} وحدة، {self.divisions} قسم "
                f"({self.active_divisions} نشط)."
            )
        return (
            f"Workplace database: {self.db_path} (schema {self.schema_version}) — "
            f"{self.institutions} institution(s), {self.units} unit(s), "
            f"{self.divisions} division(s) ({self.active_divisions} active)."
        )


@dataclass
class HierarchyView:
    """The institution tree as the sidebar draws it: units and their divisions."""

    institution: Institution | None
    units: list[OrganisationalUnit] = field(default_factory=list)
    divisions_by_unit: dict[str, list[DivisionSummary]] = field(default_factory=dict)
    unassigned: list[DivisionSummary] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return self.institution is None and not self.units and not self.unassigned


@dataclass
class DivisionDetailSummary:
    """One item of a division's technical profile, as a surface shows it."""

    record_id: str
    kind: str
    title: str
    detail: str
    reference: str
    order: int
    active: bool

    @property
    def kind_label(self) -> str:
        """The readable name of the kind (``division_detail_kind_label``)."""
        return division_detail_kind_label(self.kind)

    def describe(self) -> str:
        head = self.title or self.kind_label
        parts = [f"{self.kind_label}: {head}"]
        if self.detail:
            parts.append(f"— {self.detail}")
        if self.reference:
            parts.append(f"[{self.reference}]")
        if not self.active:
            parts.append("(inactive)")
        return " ".join(parts)


@dataclass
class DivisionProfile:
    """A division's full technical profile, grouped by the brief's §3 kinds."""

    division: DivisionSummary
    details: list[DivisionDetailSummary] = field(default_factory=list)

    def by_kind(self) -> dict[str, list[DivisionDetailSummary]]:
        grouped: dict[str, list[DivisionDetailSummary]] = {}
        for detail in self.details:
            grouped.setdefault(detail.kind, []).append(detail)
        return grouped

    def count(self, kind: str) -> int:
        return sum(1 for detail in self.details if detail.kind == kind)

    @property
    def empty(self) -> bool:
        return not self.details


@dataclass
class WorkplaceDashboard:
    """The dashboard's figures (the brief's §5) -- **real records**, never demo data.

    Every figure is a count read from the Workplace database at the moment the
    dashboard is built, so a dashboard never shows a made-up number. A figure that
    has no meaning yet (no projects, no inspections) is honestly ``0``.
    """

    active_projects: int = 0
    completed_projects: int = 0
    total_projects: int = 0
    pending_assignments: int = 0
    upcoming_inspections: int = 0
    outstanding_findings: int = 0
    reports_awaiting_approval: int = 0
    documents: int = 0
    personnel: int = 0
    clients: int = 0
    standards: int = 0
    knowledge_records: int = 0
    divisions: int = 0

    def as_rows(self, *, language: str = "en") -> list[tuple[str, str]]:
        """The figures as ``(label, value)`` rows a surface draws."""
        labels_en = {
            "active_projects": "Active projects",
            "completed_projects": "Completed projects",
            "total_projects": "Total projects",
            "pending_assignments": "Assignments",
            "upcoming_inspections": "Inspections (planned)",
            "outstanding_findings": "Outstanding findings",
            "reports_awaiting_approval": "Reports awaiting approval",
            "documents": "Documents registered",
            "personnel": "Personnel",
            "clients": "Clients / contractors",
            "standards": "Standards in the library",
            "knowledge_records": "Knowledge records",
            "divisions": "Divisions",
        }
        labels_ar = {
            "active_projects": "المشروعات النشطة",
            "completed_projects": "المشروعات المكتملة",
            "total_projects": "إجمالي المشروعات",
            "pending_assignments": "التكليفات",
            "upcoming_inspections": "الفحوص (مخططة)",
            "outstanding_findings": "الملاحظات القائمة",
            "reports_awaiting_approval": "تقارير بانتظار الاعتماد",
            "documents": "الوثائق المسجلة",
            "personnel": "الكوادر",
            "clients": "العملاء / المقاولون",
            "standards": "المعايير في المكتبة",
            "knowledge_records": "سجلات المعرفة",
            "divisions": "الأقسام",
        }
        labels = labels_ar if language == "ar" else labels_en
        order = [
            "active_projects", "completed_projects", "total_projects",
            "upcoming_inspections", "outstanding_findings", "reports_awaiting_approval",
            "pending_assignments", "documents", "personnel", "clients", "standards",
            "knowledge_records", "divisions",
        ]
        return [(labels[key], str(getattr(self, key))) for key in order]

    def describe(self, *, language: str = "en") -> str:
        rows = self.as_rows(language=language)
        return " · ".join(f"{label}: {value}" for label, value in rows)


#: The readable labels for the divisional detail kinds (English + Arabic).
DIVISION_DETAIL_LABELS: dict[str, tuple[str, str]] = {
    "discipline": ("Technical disciplines", "التخصصات الفنية"),
    "personnel": ("Personnel and qualifications", "الكوادر والمؤهلات"),
    "standard": ("Applicable standards", "المعايير المطبقة"),
    "capability": ("Inspection and testing capabilities", "قدرات الفحص والاختبار"),
    "procedure": ("Procedures and work instructions", "الإجراءات وتعليمات العمل"),
    "equipment": ("Equipment and resources", "المعدات والموارد"),
    "project": ("Project assignments", "التكليفات بالمشروعات"),
    "report": ("Technical reports", "التقارير الفنية"),
    "history": ("Historical records", "السجلات التاريخية"),
    "knowledge": ("Knowledge documents", "الوثائق المعرفية"),
}


def division_detail_kind_label(kind: str, *, language: str = "en") -> str:
    """The readable label of a divisional detail kind (its own key if unknown)."""
    labels = DIVISION_DETAIL_LABELS.get(kind)
    if labels is None:
        return kind
    return labels[1] if language == "ar" else labels[0]


class WorkplaceService:
    """The one engine the Workplace window and the CLI both drive."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        db_path: Path | None = None,
        auto_seed: bool = True,
    ) -> None:
        # ``root`` is the data root the caller uses (``data_dir()`` by default);
        # ``db_path`` overrides the file itself, which is what a test injects.
        self.root = root
        self._db_path = Path(db_path) if db_path is not None else (
            Path(root) / "workplace" / "workplace.db" if root is not None else workplace_db_path()
        )
        self._db: WorkplaceDatabase | None = None
        self._repo: WorkplaceRepository | None = None
        self._domain: WorkplaceDomain | None = None
        self._projects: ProjectManager | None = None
        self._personnel: PersonnelDirectory | None = None
        self._library: TechnicalLibrary | None = None
        self._documents: DocumentManager | None = None
        self._assistant: WorkplaceAssistant | None = None
        self._documents_dir = (
            Path(root) / "workplace" / "documents" if root is not None else None
        )
        self._actor = "system"
        self._seed_report: SeedReport | None = None
        self._auto_seed = auto_seed

    # -- lifecycle --------------------------------------------------------

    @property
    def db_path(self) -> Path:
        return self._db_path

    def open(self) -> WorkplaceDatabase:
        """Open (and migrate) the database, seeding the defaults on first use."""
        if self._db is not None:
            return self._db
        try:
            self._db = WorkplaceDatabase(self._db_path)
        except Exception as error:  # a broken file is reported, never swallowed
            raise WorkplaceError(f"Could not open the Workplace database: {error}") from error
        self._repo = WorkplaceRepository(self._db)
        if self._auto_seed:
            self._seed_report = seed_workplace(self._repo)
        return self._db

    @property
    def repository(self) -> WorkplaceRepository:
        if self._repo is None:
            self.open()
        assert self._repo is not None
        return self._repo

    @property
    def domain(self) -> WorkplaceDomain:
        """The audited domain service over the repository (created on first use)."""
        if self._domain is None:
            self._domain = WorkplaceDomain(self.repository, actor=self._actor)
        return self._domain

    def set_actor(self, actor: str) -> None:
        """Set the name recorded on the audit trail (the security phase supplies one)."""
        self._actor = actor or "system"
        if self._domain is not None:
            self._domain.actor = self._actor

    @property
    def projects(self) -> ProjectManager:
        """The read-only project views (the brief's §6)."""
        if self._projects is None:
            self._projects = ProjectManager(self.repository)
        return self._projects

    @property
    def personnel(self) -> PersonnelDirectory:
        """The personnel directory, matrix, and selection (the brief's §8)."""
        if self._personnel is None:
            self._personnel = PersonnelDirectory(self.repository, self.domain)
        return self._personnel

    @property
    def library(self) -> TechnicalLibrary:
        """The technical library of standards and references (the brief's §7)."""
        if self._library is None:
            self._library = TechnicalLibrary(self.repository, self.domain)
        return self._library

    @property
    def document_manager(self) -> DocumentManager:
        """The project document registry (the brief's §9)."""
        if self._documents is None:
            self._documents = DocumentManager(
                self.repository, self.domain, documents_dir=self._documents_dir
            )
        return self._documents

    def seed(self, *, force: bool = False) -> SeedReport:
        """Install (or reset) the Workplace defaults."""
        report = seed_workplace(self.repository, force=force)
        self._seed_report = report
        return report

    # -- status -----------------------------------------------------------

    def status(self, *, language: str = "en") -> WorkplaceStatus:
        """An honest snapshot: real counts, the schema, and whether it is ready."""
        try:
            self.open()
        except WorkplaceError as error:
            return WorkplaceStatus(
                db_path=str(self._db_path), schema_version=0, institutions=0, units=0,
                divisions=0, active_divisions=0, taxonomy_areas=len(TAXONOMY),
                taxonomy_disciplines=sum(a.discipline_count for a in TAXONOMY),
                ready=False, message=str(error),
            )
        repo = self.repository
        divisions = repo.list_divisions(include_inactive=True)
        return WorkplaceStatus(
            db_path=str(self._db_path),
            schema_version=self.repository.db.schema_version(),
            institutions=len(repo.list_institutions(include_inactive=True)),
            units=len(repo.list_units(include_inactive=True)),
            divisions=len(divisions),
            active_divisions=sum(1 for d in divisions if d.active),
            taxonomy_areas=len(TAXONOMY),
            taxonomy_disciplines=sum(a.discipline_count for a in TAXONOMY),
            ready=True,
            message="",
        )

    # -- dashboard (the brief's §5) --------------------------------------

    def dashboard(self, *, language: str = "en") -> WorkplaceDashboard:
        """The dashboard's figures, every one a **real** count from the database."""
        self.open()
        repo = self.repository
        projects = repo.list_domain("projects")
        inspections = repo.list_domain("inspections")
        findings = repo.list_domain("findings")
        reports = repo.list_domain("reports")
        return WorkplaceDashboard(
            active_projects=sum(1 for p in projects if p.status == "active"),
            completed_projects=sum(1 for p in projects if p.status in {"completed", "closed"}),
            total_projects=len(projects),
            pending_assignments=len(repo.list_domain("assignments")),
            upcoming_inspections=sum(1 for i in inspections if i.status == "planned"),
            outstanding_findings=sum(1 for f in findings if f.status != "closed"),
            reports_awaiting_approval=sum(
                1 for r in reports if r.status in {"draft", "submitted"}
            ),
            documents=len(repo.list_domain("workplace_documents")),
            personnel=len(repo.list_domain("people")),
            clients=len(repo.list_domain("clients")),
            standards=len(repo.list_domain("standards")),
            knowledge_records=len(repo.list_domain("knowledge_records")),
            divisions=len(repo.list_divisions()),
        )

    # -- projects & personnel (the brief's §6 and §8) ---------------------

    def project_rows(
        self, *, status: str | None = None, division_id: str | None = None
    ) -> list[ProjectRow]:
        """A light project listing with each project's own real counts."""
        self.open()
        return self.projects.rows(status=status, division_id=division_id)

    def project_view(self, project_id: str) -> ProjectSummary | None:
        """The full view of one project (every reference resolved), or ``None``."""
        self.open()
        return self.projects.resolve(project_id)

    def project_documents(self, project_id: str, *, doc_type: str | None = None):
        """The project's own document registry, optionally narrowed to one type."""
        self.open()
        return self.projects.documents(project_id, doc_type=doc_type)

    def project_assignments(self, project_id: str) -> list[Assignment]:
        """The people assigned to a project, with their role resolved."""
        self.open()
        return self.repository.list_domain(
            "assignments", where="project_id = ?", params=(project_id,)
        )

    def assign_person(
        self, *, project_id: str, person_id: str, role: str = "", **fields: object
    ) -> Assignment:
        """Assign a person to a project (audited through the domain service)."""
        self.open()
        return self.personnel.assign(
            project_id=project_id, person_id=person_id, role=role, **fields
        )

    def personnel_people(
        self,
        *,
        division_id: str | None = None,
        discipline: str | None = None,
        available_only: bool = False,
    ) -> list[Person]:
        self.open()
        return self.personnel.people(
            division_id=division_id, discipline=discipline, available_only=available_only
        )

    def person_view(self, person_id: str) -> PersonProfile | None:
        self.open()
        return self.personnel.profile(person_id)

    def competency_matrix(self, *, division_id: str | None = None) -> CompetencyMatrix:
        self.open()
        return self.personnel.competency_matrix(division_id=division_id)

    def select_personnel(
        self,
        *,
        skills: tuple[str, ...] = (),
        qualifications: tuple[str, ...] = (),
        division_id: str | None = None,
        limit: int = 5,
    ) -> list[Candidate]:
        """Rank the people who can cover the required skills/qualifications."""
        self.open()
        return self.personnel.select(
            skills=skills,
            qualifications=qualifications,
            division_id=division_id,
            limit=limit,
        )

    # -- technical library (the brief's §7) -------------------------------

    def library_standards(
        self,
        *,
        body: str | None = None,
        reference_kind: str | None = None,
        query: str | None = None,
    ) -> list[StandardRevision]:
        """The standards library, optionally narrowed by body, kind, or query."""
        self.open()
        return self.library.standards(body=body, reference_kind=reference_kind, query=query)

    def library_bodies(self) -> list[tuple[str, str, int]]:
        """Every issuing body with its title and how many references it has."""
        self.open()
        return self.library.bodies()

    def library_counts(self) -> dict[str, int]:
        """How many references the library holds of each reference kind."""
        self.open()
        return self.library.counts_by_kind()

    def standard_revisions(self, *, body: str, number: str) -> list[StandardRevision]:
        """Every revision of one standard, oldest first."""
        self.open()
        return self.library.revisions(body=body, number=number)

    def compare_standard_revisions(
        self, *, body: str, number: str, from_revision: str = "", to_revision: str = ""
    ) -> RevisionComparison | None:
        """Compare two revisions of one standard (``None`` when there is nothing)."""
        self.open()
        return self.library.compare_revisions(
            body=body, number=number, from_revision=from_revision, to_revision=to_revision
        )

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
        """Add a reference to the library (audited)."""
        self.open()
        return self.library.add_standard(
            body=body, number=number, title=title, edition=edition, revision=revision,
            published_on=published_on, reference_kind=reference_kind,
            applicability=applicability, source=source, status=status,
        )

    def record_standard_revision(self, record_id: str, **changes: object) -> StandardRevision:
        """Record a *new* revision of a standard (the original row is untouched)."""
        self.open()
        return self.library.record_new_revision(record_id, **changes)

    # -- document registry (the brief's §9) -------------------------------

    def register_document_file(
        self,
        source: Path | str,
        *,
        title: str = "",
        title_ar: str = "",
        project_id: str = "",
        division_id: str = "",
        doc_type: str = "other",
        revision: str = "",
        supersedes_id: str = "",
        confidentiality: str = "internal",
        author_id: str = "",
        issued_on: str = "",
    ) -> DocumentRecord:
        """Copy a file in and register it in the project document registry."""
        self.open()
        return self.document_manager.register_file(
            source, title=title, title_ar=title_ar, project_id=project_id,
            division_id=division_id, doc_type=doc_type, revision=revision,
            supersedes_id=supersedes_id, confidentiality=confidentiality,
            author_id=author_id, issued_on=issued_on,
        )

    def register_document_text(
        self,
        text: str,
        *,
        title: str,
        doc_type: str = "other",
        project_id: str = "",
        division_id: str = "",
        revision: str = "",
        confidentiality: str = "internal",
    ) -> DocumentRecord:
        """Register a document whose text is supplied directly."""
        self.open()
        return self.document_manager.register_text(
            text, title=title, doc_type=doc_type, project_id=project_id,
            division_id=division_id, revision=revision, confidentiality=confidentiality,
        )

    def document_rows(
        self,
        *,
        project_id: str | None = None,
        doc_type: str | None = None,
        query: str | None = None,
    ) -> list[DocumentRecord]:
        """The document registry, optionally narrowed."""
        self.open()
        return self.document_manager.documents(
            project_id=project_id, doc_type=doc_type, query=query
        )

    def document_record(self, record_id: str) -> DocumentRecord | None:
        self.open()
        return self.document_manager.document(record_id)

    def document_history(self, record_id: str) -> list[DocumentRecord]:
        """A document's revision chain."""
        self.open()
        return self.document_manager.history(record_id)

    def verify_document(self, record_id: str) -> DocumentVerification:
        """Re-hash a stored document and report whether it still matches."""
        self.open()
        return self.document_manager.verify(record_id)

    def document_text(self, record_id: str):
        """Read a registered document's text through the project's own reader."""
        self.open()
        return self.document_manager.read_text(record_id)

    def document_chunks(
        self, record_id: str, *, chunk_chars: int = 900, overlap: int = 120
    ) -> list[str]:
        """The document's text as retrievable chunks (the project's own chunker)."""
        self.open()
        return self.document_manager.chunks(record_id, chunk_chars=chunk_chars, overlap=overlap)

    def document_types(self, *, language: str = "en") -> list[tuple[str, str]]:
        """Every document type with its readable label, for a picker."""
        return self.document_manager.types(language=language)

    # -- the Workplace assistant (the brief's §10) ------------------------

    @property
    def assistant(self) -> WorkplaceAssistant:
        """The Workplace assistant over the records (the brief's §10)."""
        if self._assistant is None:
            self._assistant = WorkplaceAssistant(
                self.repository,
                domain=self.domain,
                documents=self.document_manager,
            )
        return self._assistant

    def assistant_profiles(self, *, language: str = "en") -> list[tuple[str, str, str]]:
        """``(key, name, specialization)`` for every selectable assistant profile."""
        return self.assistant.profile_summaries(language=language)

    def ask_workplace(
        self, question: str, *, profile: str = "petroleum", limit: int = 4
    ) -> str:
        """Answer a Workplace question in the chosen profile, with the disclaimer."""
        return self.assistant.answer_text(question, profile=profile, limit=limit)

    def refresh_assistant(self) -> CorpusReport:
        """Rebuild the assistant's corpus from the records (cumulative knowledge)."""
        return self.assistant.refresh()

    def assistant_status(self, *, language: str = "en") -> str:
        """An honest one-line status of the Workplace assistant."""
        return self.assistant.status(language=language)

    # -- institution hierarchy -------------------------------------------

    def hierarchy(self, *, include_inactive: bool = False) -> HierarchyView:
        """The institution tree: units and the divisions that belong to each."""
        self.open()
        repo = self.repository
        institutions = repo.list_institutions(include_inactive=include_inactive)
        institution = institutions[0] if institutions else None
        units = (
            repo.list_units(institution_id=institution.record_id, include_inactive=include_inactive)
            if institution is not None
            else []
        )
        by_unit: dict[str, list[DivisionSummary]] = {unit.record_id: [] for unit in units}
        unassigned: list[DivisionSummary] = []
        for division in repo.list_divisions(include_inactive=include_inactive):
            summary = self._summarise(division, units)
            if division.unit_id in by_unit:
                by_unit[division.unit_id].append(summary)
            else:
                unassigned.append(summary)
        for items in by_unit.values():
            items.sort(key=lambda d: (d.order, d.name))
        unassigned.sort(key=lambda d: (d.order, d.name))
        return HierarchyView(
            institution=institution, units=units, divisions_by_unit=by_unit, unassigned=unassigned
        )

    def divisions(self, *, include_inactive: bool = False) -> list[DivisionSummary]:
        """Every division, in display order, with its unit's name resolved."""
        self.open()
        repo = self.repository
        units = repo.list_units(include_inactive=True)
        summaries = [
            self._summarise(division, units)
            for division in repo.list_divisions(include_inactive=include_inactive)
        ]
        summaries.sort(key=lambda d: (d.order, d.name))
        return summaries

    def division(self, record_id: str) -> DivisionSummary | None:
        """One division by id, or ``None``."""
        self.open()
        division = self.repository.get_division(record_id)
        if division is None:
            return None
        return self._summarise(division, self.repository.list_units(include_inactive=True))

    # -- division editing (no source change) ------------------------------

    def add_division(
        self,
        *,
        name: str,
        name_ar: str = "",
        key: str = "",
        unit_id: str = "",
        summary: str = "",
        scope: str = "",
        proposed: bool = True,
    ) -> DivisionSummary:
        """Add a division. ``name`` is required; the rest are optional."""
        if not name.strip():
            raise WorkplaceError("A division needs a name.")
        self.open()
        repo = self.repository
        if not unit_id:
            institutions = repo.list_institutions()
            units = repo.list_units(institution_id=institutions[0].record_id) if institutions else []
            unit_id = units[0].record_id if units else ""
        if key and repo.division_by_key(key) is not None:
            raise WorkplaceError(f"A division with key {key!r} already exists.")
        order = len(repo.list_divisions(include_inactive=True)) + 1
        record = repo.create_division(
            unit_id=unit_id,
            key=key or _slugify(name),
            name=name.strip(),
            name_ar=name_ar.strip(),
            summary=summary.strip(),
            scope=scope.strip(),
            proposed=proposed,
            order=order,
            metadata={"source": "user", "note": "Added by the user in the Workplace."},
        )
        return self._summarise(record, repo.list_units(include_inactive=True))

    def rename_division(self, record_id: str, *, name: str, name_ar: str = "") -> DivisionSummary:
        """Rename a division without touching source code."""
        if not name.strip():
            raise WorkplaceError("A division needs a name.")
        self.open()
        updated = self.repository.rename_division(record_id, name=name.strip(), name_ar=name_ar)
        if updated is None:
            raise WorkplaceError(f"No division with id {record_id!r}.")
        return self._summarise(updated, self.repository.list_units(include_inactive=True))

    def set_division_unit(self, record_id: str, *, unit_id: str) -> DivisionSummary:
        """Move a division to another unit (a reorganisation, never a delete)."""
        self.open()
        division = self.repository.get_division(record_id)
        if division is None:
            raise WorkplaceError(f"No division with id {record_id!r}.")
        updated = self.repository.save_division(division.touch(unit_id=unit_id))
        return self._summarise(updated, self.repository.list_units(include_inactive=True))

    def set_division_order(self, record_id: str, *, order: int) -> DivisionSummary:
        """Reorder a division within its unit."""
        self.open()
        updated = self.repository.reorder_division(record_id, order=order)
        if updated is None:
            raise WorkplaceError(f"No division with id {record_id!r}.")
        return self._summarise(updated, self.repository.list_units(include_inactive=True))

    def deactivate_division(self, record_id: str) -> DivisionSummary:
        """Deactivate a division (the soft delete); historical records survive."""
        self.open()
        updated = self.repository.deactivate_division(record_id)
        if updated is None:
            raise WorkplaceError(f"No division with id {record_id!r}.")
        return self._summarise(updated, self.repository.list_units(include_inactive=True))

    def activate_division(self, record_id: str) -> DivisionSummary:
        """Reactivate a deactivated division."""
        self.open()
        division = self.repository.get_division(record_id)
        if division is None:
            raise WorkplaceError(f"No division with id {record_id!r}.")
        updated = self.repository.save_division(division.activate())
        return self._summarise(updated, self.repository.list_units(include_inactive=True))

    # -- taxonomy ---------------------------------------------------------

    def taxonomy(self) -> tuple[TaxonomyArea, ...]:
        """The professional knowledge taxonomy (declarative; no database needed)."""
        return TAXONOMY

    def taxonomy_summary(self) -> str:
        areas = len(TAXONOMY)
        disciplines = sum(area.discipline_count for area in TAXONOMY)
        return f"{areas} areas, {disciplines} disciplines"

    # -- divisional profile (the brief's §3) ------------------------------

    def division_profile(self, record_id: str) -> DivisionProfile:
        """A division's full technical profile, grouped by kind."""
        self.open()
        summary = self.division(record_id)
        if summary is None:
            raise WorkplaceError(f"No division with id {record_id!r}.")
        rows = self.repository.list_details(division_id=record_id)
        details = [self._summarise_detail(row) for row in rows]
        return DivisionProfile(division=summary, details=details)

    def add_division_detail(
        self,
        *,
        division_id: str,
        kind: str,
        title: str = "",
        detail: str = "",
        reference: str = "",
    ) -> DivisionDetailSummary:
        """Add one item to a division's profile (no source change)."""
        if kind not in DIVISION_DETAIL_KINDS:
            raise WorkplaceError(
                f"Unknown detail kind {kind!r}; expected one of "
                f"{', '.join(DIVISION_DETAIL_KINDS)}."
            )
        if not (title.strip() or detail.strip()):
            raise WorkplaceError("A profile item needs a title or a detail.")
        self.open()
        if self.repository.get_division(division_id) is None:
            raise WorkplaceError(f"No division with id {division_id!r}.")
        order = len(self.repository.list_details(division_id=division_id, include_inactive=True)) + 1
        record = self.repository.create_detail(
            division_id=division_id, kind=kind, title=title.strip(),
            detail=detail.strip(), reference=reference.strip(), order=order,
            metadata={"source": "user", "note": "Added by the user in the Workplace."},
        )
        return self._summarise_detail(record)

    def remove_division_detail(self, record_id: str) -> DivisionDetailSummary:
        """Deactivate a profile item (the soft delete); history survives."""
        self.open()
        updated = self.repository.deactivate_detail(record_id)
        if updated is None:
            raise WorkplaceError(f"No profile item with id {record_id!r}.")
        return self._summarise_detail(updated)

    def detail_kind_labels(self, *, language: str = "en") -> list[tuple[str, str]]:
        """Every kind with its readable label, for a picker."""
        return [
            (kind, division_detail_kind_label(kind, language=language))
            for kind in DIVISION_DETAIL_KINDS
        ]

    # -- taxonomy editing (the brief's §4) --------------------------------

    def taxonomy_tree(self, *, include_inactive: bool = False) -> list[TaxonomyNode]:
        """The stored taxonomy's areas, each with its disciplines in ``metadata``.

        The stored nodes are returned as-is (a flat list of areas, each carrying a
        ``disciplines`` list in its metadata), so a surface can draw the tree and
        an administrator can edit it. When nothing is stored yet the shipped
        :data:`~core.workplace.taxonomy.TAXONOMY` is used, so the tree is never
        empty.
        """
        self.open()
        areas = self.repository.list_taxonomy_nodes(include_inactive=include_inactive)
        areas = [node for node in areas if node.kind == "area"]
        if not areas:
            return []
        out: list[TaxonomyNode] = []
        for area in areas:
            disciplines = self.repository.list_taxonomy_nodes(
                parent_id=area.record_id, include_inactive=include_inactive
            )
            metadata = dict(area.metadata)
            metadata["disciplines"] = [node.name for node in disciplines]
            metadata["discipline_ids"] = [node.record_id for node in disciplines]
            out.append(
                TaxonomyNode(
                    record_id=area.record_id, created_at=area.created_at,
                    updated_at=area.updated_at, active=area.active, parent_id="",
                    kind="area", key=area.key, name=area.name, name_ar=area.name_ar,
                    summary=area.summary, order=area.order, proposed=area.proposed,
                    metadata=metadata,
                )
            )
        return out

    def add_taxonomy_area(
        self, *, name: str, name_ar: str = "", key: str = "", summary: str = ""
    ) -> TaxonomyNode:
        """Add a taxonomy area (no source change)."""
        if not name.strip():
            raise WorkplaceError("A taxonomy area needs a name.")
        self.open()
        chosen_key = key.strip() or _slugify(name)
        if self.repository.taxonomy_node_by_key(chosen_key) is not None:
            raise WorkplaceError(f"A taxonomy node with key {chosen_key!r} already exists.")
        order = len(self.repository.list_taxonomy_nodes()) + 1
        return self.repository.create_taxonomy_node(
            kind="area", key=chosen_key, name=name.strip(), name_ar=name_ar.strip(),
            summary=summary.strip(), order=order, proposed=False,
            metadata={"source": "user", "note": "Added by the user in the Workplace."},
        )

    def add_taxonomy_discipline(
        self, *, area_id: str, name: str, name_ar: str = ""
    ) -> TaxonomyNode:
        """Add a sub-discipline under a taxonomy area (no source change)."""
        if not name.strip():
            raise WorkplaceError("A discipline needs a name.")
        self.open()
        area = self.repository.get_taxonomy_node(area_id)
        if area is None or area.kind != "area":
            raise WorkplaceError(f"No taxonomy area with id {area_id!r}.")
        siblings = self.repository.list_taxonomy_nodes(parent_id=area_id)
        return self.repository.create_taxonomy_node(
            kind="discipline", parent_id=area_id,
            key=f"{area.key}:{_slugify(name)}", name=name.strip(), name_ar=name_ar.strip(),
            order=len(siblings) + 1, proposed=False,
            metadata={"source": "user", "note": "Added by the user in the Workplace."},
        )

    def rename_taxonomy_node(self, record_id: str, *, name: str, name_ar: str = "") -> TaxonomyNode:
        """Rename a taxonomy area or discipline (no source change)."""
        if not name.strip():
            raise WorkplaceError("A taxonomy node needs a name.")
        self.open()
        node = self.repository.get_taxonomy_node(record_id)
        if node is None:
            raise WorkplaceError(f"No taxonomy node with id {record_id!r}.")
        changes: dict[str, object] = {"name": name.strip()}
        if name_ar:
            changes["name_ar"] = name_ar.strip()
        return self.repository.save_taxonomy_node(node.touch(**changes))

    def remove_taxonomy_node(self, record_id: str) -> TaxonomyNode:
        """Deactivate a taxonomy node (the soft delete)."""
        self.open()
        updated = self.repository.deactivate_taxonomy_node(record_id)
        if updated is None:
            raise WorkplaceError(f"No taxonomy node with id {record_id!r}.")
        return updated

    # -- settings ---------------------------------------------------------

    def settings(self) -> WorkplaceSettings:
        """The saved Workplace settings (defaults when none are stored)."""
        return load_settings(self.root)

    def update_settings(self, **changes: object) -> WorkplaceSettings:
        """Save a settings change and return the settings in force."""
        current = self.settings()
        if "language" in changes:
            current = current.with_language(str(changes["language"]))
        if "rtl" in changes:
            current = current.with_rtl(bool(changes["rtl"]))
        if "default_institution" in changes:
            current = current.with_default_institution(str(changes["default_institution"]))
        if "show_proposed" in changes:
            current = current.with_show_proposed(bool(changes["show_proposed"]))
        if not save_settings(current, self.root):
            raise WorkplaceError("Could not save the Workplace settings (read-only folder?).")
        return current

    # -- backup -----------------------------------------------------------

    def backup(self, destination: Path | None = None) -> str:
        """Write a consistent copy of the database and return its path."""
        self.open()
        from core.paths import workplace_backups_dir

        if destination is None:
            from core.workplace.model import utc_now

            stamp = utc_now().replace(":", "").replace("-", "")[:15]
            destination = workplace_backups_dir() / f"workplace-{stamp}.db"
        report = self.repository.db.backup(destination)
        if not report.ok:
            raise WorkplaceError(report.message or "backup failed")
        return report.path

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _summarise(
        division: Division, units: list[OrganisationalUnit]
    ) -> DivisionSummary:
        unit_name = next((u.name for u in units if u.record_id == division.unit_id), "")
        return DivisionSummary(
            record_id=division.record_id,
            key=division.key,
            name=division.name,
            name_ar=division.name_ar,
            summary=division.summary,
            scope=division.scope,
            proposed=division.proposed,
            order=division.order,
            active=division.active,
            unit_id=division.unit_id,
            unit_name=unit_name,
        )

    @staticmethod
    def _summarise_detail(detail: DivisionDetail) -> DivisionDetailSummary:
        return DivisionDetailSummary(
            record_id=detail.record_id,
            kind=detail.kind,
            title=detail.title,
            detail=detail.detail,
            reference=detail.reference,
            order=detail.order,
            active=detail.active,
        )


def _slugify(text: str) -> str:
    """A stable key from a division name (lower-case, hyphenated)."""
    cleaned = [character if character.isalnum() else " " for character in text.strip().casefold()]
    parts = "".join(cleaned).split()
    return "-".join(parts) or "division"


__all__ = [
    "DIVISION_DETAIL_LABELS",
    "Candidate",
    "CompetencyMatrix",
    "DivisionDetailSummary",
    "DivisionProfile",
    "DivisionSummary",
    "DocumentRecord",
    "DocumentVerification",
    "HierarchyView",
    "PersonProfile",
    "ProjectRow",
    "ProjectSummary",
    "RevisionComparison",
    "StandardRevision",
    "WorkplaceDashboard",
    "WorkplaceDomainError",
    "WorkplaceError",
    "WorkplaceService",
    "WorkplaceStatus",
    "division_detail_kind_label",
    "document_type_label",
    "reference_kind_label",
]
