"""The Workplace assistant: institutional knowledge, retrieved and cited (§10).

This is the EPRI/TSTC Workplace's own assistant. It is deliberately a **thin
layer over the systems the project already has** -- the brief's §11 -- and it
adds no second RAG, no second document engine, and no second knowledge database:

* the corpus is built as :class:`core.documents.Document`s **from the Workplace's
  own records** (the institutions, units, divisions and their profiles, projects,
  people and their qualifications, clients, the technical library, the registered
  documents -- whose text is read through the project's own reader -- procedures,
  inspections, findings, reports, correspondence, and knowledge records). Every
  document is a *view* derived from a row, so the Workplace keeps exactly one copy
  of everything;
* retrieval is the project's own :class:`core.rag.RagRetriever` (lexical + semantic
  + ontology-guided + optional bulk), and the answer is the project's own
  :class:`core.assistant.GroundedAssistant` -- so every answer carries the
  citations it was built from, falls back to the extractive answer with no model,
  and passes the project's groundedness gate.

**Cumulative by construction.** A new project, an approved document, a recorded
finding, or a knowledge record becomes a new citable document the next time the
corpus is built, so the institutional knowledge grows with the work -- while each
document keeps its own provenance (which record it came from) and its
confidentiality classification.

**Honest by construction.** The assistant never presents its output as an approved
engineering decision or an official EPRI commitment (the brief's §13): every
answer carries :data:`DISCLAIMER`. A document whose confidentiality the caller is
not cleared for is **withheld and counted**, never silently included; a document
with no readable text is skipped, never stored empty.

Profiles (the brief's §10): :data:`PROFILES` names the eleven assistants, each a
:class:`WorkplaceProfile` carrying its specialization, the extra system
instruction, and the disciplines/reference bodies it focuses on, so a profile
retrieves a narrower, more relevant slice of the same corpus.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.documents import Document, DocumentStore
from core.workplace.repository import WorkplaceRepository

if TYPE_CHECKING:
    from core.assistant import GroundedAssistant
    from core.workplace.domain import WorkplaceDomain

#: The line every Workplace answer carries: the assistant advises, it does not
#: decide. The brief's §13 forbids treating an AI recommendation as an approved
#: engineering decision or an official EPRI commitment.
DISCLAIMER = (
    "This is AI-generated technical guidance for the Workplace. It is not an "
    "approved engineering decision, a signed inspection result, or an official "
    "EPRI/TSTC commitment; a qualified engineer must review and approve it."
)

#: The confidentiality levels a caller is cleared for by default. A document
#: classified above the clearance is withheld and counted, never included.
DEFAULT_CLEARANCE = ("public", "internal")

#: The most text taken from one registered document, so a huge file cannot blow
#: the corpus up; a bounded read, never the whole of a 1000-page PDF.
MAX_DOCUMENT_CHARS = 20000

#: The most registered documents read into the corpus in one build.
MAX_DOCUMENTS = 200


@dataclass(frozen=True)
class WorkplaceProfile:
    """One selectable assistant (the brief's §10).

    ``divisions`` narrows the profile to the Workplace divisions it focuses on
    (matched against a document's ``division_id``/``division_key``); an empty tuple
    means the whole institution. ``references`` names the issuing bodies the
    profile cares about (API, ASME, ...), used to prioritise the library. Both are
    honest filters over the one corpus -- a profile never invents knowledge.
    """

    key: str
    name: str
    name_ar: str
    specialization: str
    instruction: str
    divisions: tuple[str, ...] = ()
    references: tuple[str, ...] = ()

    def system_instruction(self, base: str) -> str:
        """The profile's own instruction laid over the project's base instruction."""
        parts = [base, "", f"Role: {self.name} ({self.specialization}).", self.instruction]
        if self.divisions:
            parts.append(
                "Focus on the Workplace divisions: " + ", ".join(self.divisions) + "."
            )
        if self.references:
            parts.append(
                "Prioritise the applicable references from: "
                + ", ".join(self.references) + "."
            )
        return "\n".join(parts)


#: The eleven selectable assistants the brief's §10 asks for.
PROFILES: tuple[WorkplaceProfile, ...] = (
    WorkplaceProfile(
        key="petroleum",
        name="Petroleum Engineering Assistant",
        name_ar="مساعد هندسة البترول",
        specialization="drilling, production, reservoir, and well integrity",
        instruction=(
            "Answer as a petroleum engineer: reason about drilling, production, "
            "reservoir behaviour, petroleum geology, well integrity, oil and gas "
            "processing, and pipelines and flowlines. State the governing relation "
            "and the key quantity with its units, and name the applicable standard."
        ),
        references=("API", "ISO", "DNV"),
    ),
    WorkplaceProfile(
        key="welding",
        name="Welding Engineering Assistant",
        name_ar="مساعد هندسة اللحام",
        specialization="welding processes, procedures, and qualification",
        instruction=(
            "Answer as a welding engineer: reason about welding processes, WPS / "
            "PQR / WPQ, welder qualification, welding inspection, metallurgy and "
            "heat treatment, and weld defects with their acceptance criteria."
        ),
        divisions=("welding-engineering",),
        references=("ASME", "AWS", "API", "ISO"),
    ),
    WorkplaceProfile(
        key="coating",
        name="Coating and Corrosion Assistant",
        name_ar="مساعد الدهانات والتآكل",
        specialization="surface preparation, coatings, and corrosion control",
        instruction=(
            "Answer as a coating and corrosion engineer: reason about surface "
            "preparation, protective coatings, field joint coating, internal and "
            "external coatings, concrete weight coating, insulation, cathodic "
            "protection, corrosion monitoring, and coating inspection and testing."
        ),
        divisions=("painting-coating", "concrete-weight-coating"),
        references=("AMPP", "ISO", "ASTM", "DNV"),
    ),
    WorkplaceProfile(
        key="civil",
        name="Civil and Concrete Engineering Assistant",
        name_ar="مساعد الهندسة المدنية والخرسانة",
        specialization="structural, concrete, and construction engineering",
        instruction=(
            "Answer as a civil engineer: reason about structural engineering, "
            "concrete technology, construction inspection, material testing, and "
            "offshore and onshore civil works."
        ),
        divisions=("civil-engineering",),
        references=("ASTM", "ISO", "EN"),
    ),
    WorkplaceProfile(
        key="electrical",
        name="Electrical Engineering Assistant",
        name_ar="مساعد الهندسة الكهربائية",
        specialization="electrical installations, instrumentation, and commissioning",
        instruction=(
            "Answer as an electrical engineer: reason about electrical "
            "installations, electrical inspection, instrumentation, and testing "
            "and commissioning."
        ),
        divisions=("electrical-engineering",),
        references=("IEC", "ISO", "EN"),
    ),
    WorkplaceProfile(
        key="qa-qc",
        name="QA/QC and Inspection Assistant",
        name_ar="مساعد ضبط الجودة والفحص",
        specialization="quality assurance, NDT, and inspection planning",
        instruction=(
            "Answer as a QA/QC engineer: reason about quality assurance and "
            "control, NDT, inspection planning, quality auditing, NCR / CAR, and "
            "testing and evaluation. State the acceptance criterion and its source."
        ),
        divisions=("inspection-quality-control",),
        references=("ISO", "ASME", "ASTM", "API"),
    ),
    WorkplaceProfile(
        key="surveillance",
        name="Technical Surveillance Assistant",
        name_ar="مساعد الإشراف الفني",
        specialization="technical surveillance and witness points",
        instruction=(
            "Answer as a technical surveillance engineer: reason about witness "
            "points, hold points, surveillance visits, and the evidence a "
            "surveillance report must carry."
        ),
        divisions=("technical-surveillance",),
    ),
    WorkplaceProfile(
        key="project-management",
        name="Project Management Assistant",
        name_ar="مساعد إدارة المشروعات",
        specialization="project planning, scheduling, and delivery",
        instruction=(
            "Answer as a project manager: reason about scope, schedule, "
            "milestones, resourcing, risk, progress reporting, and the technical "
            "deliverables a project owes."
        ),
        divisions=("project-technical-support",),
    ),
    WorkplaceProfile(
        key="tender",
        name="Tender and Proposal Assistant",
        name_ar="مساعد العطاءات والعروض",
        specialization="technical and commercial proposals",
        instruction=(
            "Answer as a tender engineer: draft and review technical and "
            "commercial proposals, describe the scope of services, and state the "
            "assumptions, exclusions, and the standards the offer rests on."
        ),
    ),
    WorkplaceProfile(
        key="contract",
        name="Contract and Correspondence Assistant",
        name_ar="مساعد العقود والمراسلات",
        specialization="contractual requirements and formal correspondence",
        instruction=(
            "Answer as a contracts engineer: analyse contractual requirements, "
            "identify obligations and risks, and draft formal correspondence and "
            "minutes. Quote the clause you rely on and say when the record is silent."
        ),
    ),
    WorkplaceProfile(
        key="research",
        name="Research and Laboratory Assistant",
        name_ar="مساعد البحث والمعامل",
        specialization="analytical, industrial, and materials science",
        instruction=(
            "Answer as a research scientist: reason about analytical, industrial, "
            "and physical chemistry, geochemistry, geology, materials and polymer "
            "science, and environmental sciences, and state the method and its "
            "uncertainty."
        ),
        references=("ASTM", "ISO"),
    ),
)

PROFILE_KEYS: tuple[str, ...] = tuple(profile.key for profile in PROFILES)
DEFAULT_PROFILE = "petroleum"


def profile_for(key: str) -> WorkplaceProfile:
    """The profile with this key, or the default for an unknown key (never a guess)."""
    for profile in PROFILES:
        if profile.key == key:
            return profile
    for profile in PROFILES:
        if profile.key == DEFAULT_PROFILE:
            return profile
    return PROFILES[0]


@dataclass
class CorpusReport:
    """What one corpus build actually produced (read off the records, not assumed)."""

    documents: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    withheld: int = 0  # classified above the caller's clearance
    unreadable: int = 0  # a registered document with no readable text

    def describe(self) -> str:
        parts = [f"{self.documents} citable documents"]
        if self.by_kind:
            parts.append(
                "(" + ", ".join(f"{kind}: {count}" for kind, count in sorted(self.by_kind.items())) + ")"
            )
        if self.withheld:
            parts.append(f"{self.withheld} withheld by confidentiality")
        if self.unreadable:
            parts.append(f"{self.unreadable} with no readable text")
        return "; ".join(parts)


def _cleared(confidentiality: str, clearance: Sequence[str]) -> bool:
    """Whether a record's classification is within the caller's clearance."""
    if not confidentiality:
        return True
    return confidentiality in clearance


def build_workplace_documents(
    repository: WorkplaceRepository,
    *,
    documents=None,
    clearance: Sequence[str] = DEFAULT_CLEARANCE,
    include_document_text: bool = True,
    report: CorpusReport | None = None,
) -> list[Document]:
    """The Workplace's own records as citable documents (the brief's §10).

    Every document is a *view* derived from a row -- one copy of the facts lives
    in the Workplace database, and this is how the project's retriever reaches it.
    ``documents`` is the optional :class:`core.workplace.documents.DocumentManager`
    used to read a registered document's text through the project's own reader; a
    document with no readable text is skipped (and counted), never stored empty.
    ``clearance`` is the set of confidentiality levels the caller may read; a
    record classified above it is withheld (and counted), never silently included.
    ``report`` (when supplied) is filled with the real counts.
    """
    from core.workplace.records import (
        Assignment,
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
    )

    out: list[Document] = []
    counts: dict[str, int] = {}
    withheld = 0
    unreadable = 0

    def add(kind: str, doc_id: str, title: str, text: str, **metadata: object) -> None:
        body = text.strip()
        if not body:
            return
        out.append(
            Document(
                doc_id=doc_id,
                title=title,
                text=body,
                source=f"workplace:{kind}",
                metadata={"kind": kind, **metadata},
            )
        )
        counts[kind] = counts.get(kind, 0) + 1

    # -- the institution and its structure --------------------------------
    for institution in repository.list_institutions(include_inactive=True):
        lines = [f"Institution {institution.name}."]
        if institution.name_ar:
            lines.append(f"Arabic name: {institution.name_ar}.")
        if institution.kind:
            lines.append(f"Kind: {institution.kind}.")
        if institution.country:
            lines.append(f"Country: {institution.country}.")
        if institution.summary:
            lines.append(institution.summary)
        add("institution", f"workplace:institution:{institution.record_id}", f"Institution: {institution.name}", "\n".join(lines))

    for unit in repository.list_units(include_inactive=True):
        lines = [f"Organisational unit {unit.name} ({unit.kind})."]
        if unit.name_ar:
            lines.append(f"Arabic name: {unit.name_ar}.")
        if unit.summary:
            lines.append(unit.summary)
        add("unit", f"workplace:unit:{unit.record_id}", f"Unit: {unit.name}", "\n".join(lines))

    for division in repository.list_divisions(include_inactive=True):
        lines = [f"Division {division.name}."]
        if division.name_ar:
            lines.append(f"Arabic name: {division.name_ar}.")
        if division.summary:
            lines.append(division.summary)
        if division.scope:
            lines.append(f"Scope of services: {division.scope}")
        if division.proposed:
            lines.append("This division is a proposed, configurable default (not an approved structure).")
        details = repository.list_details(division_id=division.record_id, include_inactive=True)
        for detail in details:
            piece = f"{detail.kind}: {detail.title}" if detail.title else detail.kind
            if detail.detail:
                piece += f" -- {detail.detail}"
            if detail.reference:
                piece += f" (ref: {detail.reference})"
            lines.append(piece)
        add(
            "division", f"workplace:division:{division.record_id}",
            f"Division: {division.name}", "\n".join(lines),
            division_id=division.record_id, division_key=division.key,
        )

    # -- the technical library --------------------------------------------
    for standard in repository.list_domain("standards", include_inactive=True):
        if not isinstance(standard, Standard):
            continue
        lines = [
            f"Reference {standard.body} {standard.number}"
            + (f" -- {standard.title}" if standard.title else "")
            + f" ({standard.reference_kind})."
        ]
        if standard.edition:
            lines.append(f"Edition: {standard.edition}.")
        if standard.revision:
            lines.append(f"Revision: {standard.revision}.")
        if standard.published_on:
            lines.append(f"Published: {standard.published_on}.")
        if standard.applicability:
            lines.append(f"Applicability: {standard.applicability}.")
        if standard.source:
            lines.append(f"Source: {standard.source}.")
        add(
            "standard", f"workplace:standard:{standard.record_id}",
            f"Reference: {standard.body} {standard.number}", "\n".join(lines),
            body=standard.body, reference_kind=standard.reference_kind,
        )

    # -- clients, projects, contracts -------------------------------------
    for client in repository.list_domain("clients", include_inactive=True):
        if not isinstance(client, Client):
            continue
        lines = [f"{client.kind.title()} {client.name}."]
        if client.country:
            lines.append(f"Country: {client.country}.")
        if client.contact:
            lines.append(f"Contact: {client.contact}.")
        if client.summary:
            lines.append(client.summary)
        add("client", f"workplace:client:{client.record_id}", f"Client: {client.name}", "\n".join(lines))

    projects = [p for p in repository.list_domain("projects", include_inactive=True) if isinstance(p, Project)]
    for project in projects:
        if not _cleared(project.confidentiality, clearance):
            withheld += 1
            continue
        lines = [f"Project {project.number} -- {project.title}."]
        if project.title_ar:
            lines.append(f"Arabic title: {project.title_ar}.")
        if project.scope:
            lines.append(f"Scope of work: {project.scope}")
        if project.location:
            lines.append(f"Location: {project.location}.")
        if project.status:
            lines.append(f"Status: {project.status}.")
        if project.start_date or project.end_date:
            lines.append(f"Schedule: {project.start_date or '?'} to {project.end_date or '?'}.")
        add(
            "project", f"workplace:project:{project.record_id}",
            f"Project: {project.number}", "\n".join(lines),
            project_id=project.record_id, division_id=project.division_id,
        )

    for contract in repository.list_domain("contracts", include_inactive=True):
        if not isinstance(contract, Contract):
            continue
        lines = [f"Contract {contract.number} for project {contract.project_id}."]
        if contract.title:
            lines.append(f"Title: {contract.title}.")
        for name in ("value", "currency", "signed_on", "starts_on", "ends_on"):
            value = getattr(contract, name, "")
            if value:
                lines.append(f"{name.replace('_', ' ').title()}: {value}.")
        if contract.terms:
            lines.append(f"Terms: {contract.terms}")
        add(
            "contract", f"workplace:contract:{contract.record_id}",
            f"Contract: {contract.number}", "\n".join(lines),
            project_id=contract.project_id,
        )

    # -- people and their qualifications ----------------------------------
    people = [p for p in repository.list_domain("people", include_inactive=True) if isinstance(p, Person)]
    qualifications = [q for q in repository.list_domain("qualifications", include_inactive=True) if isinstance(q, Qualification)]
    by_person: dict[str, list[Qualification]] = {}
    for qualification in qualifications:
        by_person.setdefault(qualification.person_id, []).append(qualification)
    for person in people:
        lines = [f"{person.name}" + (f" -- {person.role}" if person.role else "") + "."]
        if person.discipline:
            lines.append(f"Discipline: {person.discipline}.")
        for qualification in by_person.get(person.record_id, ()):
            piece = f"{qualification.kind}: {qualification.title}"
            if qualification.issuer:
                piece += f" ({qualification.issuer})"
            if qualification.reference:
                piece += f" ref {qualification.reference}"
            lines.append(piece)
        if person.summary:
            lines.append(person.summary)
        add(
            "person", f"workplace:person:{person.record_id}",
            f"Personnel: {person.name}", "\n".join(lines),
            division_id=person.division_id,
        )

    # -- procedures, inspections, tests, findings, reports ---------------
    for procedure in repository.list_domain("procedures", include_inactive=True):
        if not isinstance(procedure, Procedure):
            continue
        lines = [f"Procedure {procedure.number} -- {procedure.title}."]
        if procedure.revision:
            lines.append(f"Revision: {procedure.revision}.")
        if procedure.summary:
            lines.append(procedure.summary)
        if procedure.body:
            lines.append(procedure.body)
        add(
            "procedure", f"workplace:procedure:{procedure.record_id}",
            f"Procedure: {procedure.title or procedure.number}", "\n".join(lines),
            project_id=procedure.project_id, division_id=procedure.division_id,
        )

    for inspection in repository.list_domain("inspections", include_inactive=True):
        if not isinstance(inspection, Inspection):
            continue
        lines = [f"Inspection {inspection.kind} on project {inspection.project_id}."]
        for name in ("title", "itp_reference", "scheduled_on", "performed_on", "result", "status"):
            value = getattr(inspection, name, "")
            if value:
                lines.append(f"{name.replace('_', ' ').title()}: {value}.")
        add(
            "inspection", f"workplace:inspection:{inspection.record_id}",
            f"Inspection: {inspection.title or inspection.kind}", "\n".join(lines),
            project_id=inspection.project_id,
        )

    for test in repository.list_domain("tests", include_inactive=True):
        if not isinstance(test, Test):
            continue
        lines = [f"Test {test.kind} on project {test.project_id}."]
        for name in ("title", "standard_reference", "result", "value", "unit", "performed_on"):
            value = getattr(test, name, "")
            if value:
                lines.append(f"{name.replace('_', ' ').title()}: {value}.")
        add(
            "test", f"workplace:test:{test.record_id}",
            f"Test: {test.title or test.kind}", "\n".join(lines),
            project_id=test.project_id,
        )

    for finding in repository.list_domain("findings", include_inactive=True):
        if not isinstance(finding, Finding):
            continue
        lines = [f"Finding ({finding.kind}) on project {finding.project_id}."]
        for name in ("number", "title", "description", "severity", "status", "raised_on", "closed_on"):
            value = getattr(finding, name, "")
            if value:
                lines.append(f"{name.replace('_', ' ').title()}: {value}.")
        add(
            "finding", f"workplace:finding:{finding.record_id}",
            f"Finding: {finding.title or finding.kind}", "\n".join(lines),
            project_id=finding.project_id,
        )

    for report in repository.list_domain("reports", include_inactive=True):
        if not isinstance(report, Report):
            continue
        lines = [f"Report {report.kind} on project {report.project_id}."]
        for name in ("number", "title", "summary", "body", "status", "issued_on"):
            value = getattr(report, name, "")
            if value:
                lines.append(f"{name.replace('_', ' ').title()}: {value}.")
        add(
            "report", f"workplace:report:{report.record_id}",
            f"Report: {report.title or report.kind}", "\n".join(lines),
            project_id=report.project_id,
        )

    for letter in repository.list_domain("correspondence", include_inactive=True):
        if not isinstance(letter, Correspondence):
            continue
        lines = [f"Correspondence {letter.reference} on project {letter.project_id}."]
        for name in ("kind", "subject", "body", "direction", "correspondent", "sent_on"):
            value = getattr(letter, name, "")
            if value:
                lines.append(f"{name.replace('_', ' ').title()}: {value}.")
        add(
            "correspondence", f"workplace:correspondence:{letter.record_id}",
            f"Correspondence: {letter.subject or letter.reference}", "\n".join(lines),
            project_id=letter.project_id,
        )

    for record in repository.list_domain("knowledge_records", include_inactive=True):
        if not isinstance(record, KnowledgeRecord):
            continue
        if not _cleared(record.confidentiality, clearance):
            withheld += 1
            continue
        lines = [f"Knowledge ({record.kind}): {record.title}."]
        if record.summary:
            lines.append(record.summary)
        if record.source:
            lines.append(f"Source: {record.source}.")
        if record.reference:
            lines.append(f"Reference: {record.reference}.")
        if record.verified:
            lines.append("This record is marked verified.")
        add(
            "knowledge", f"workplace:knowledge:{record.record_id}",
            f"Knowledge: {record.title}", "\n".join(lines),
            project_id=record.project_id, division_id=record.division_id,
        )

    # -- the registered documents' own text (cumulative knowledge) --------
    if documents is not None and include_document_text:
        rows = documents.documents()
        for record in rows[:MAX_DOCUMENTS]:
            if not _cleared(record.confidentiality, clearance):
                withheld += 1
                continue
            try:
                text = documents.read_text(record.record_id).text
            except Exception:
                unreadable += 1
                continue
            if not text.strip():
                unreadable += 1
                continue
            body = text[:MAX_DOCUMENT_CHARS]
            add(
                "document", f"workplace:document:{record.record_id}",
                f"Document: {record.title or record.filename}",
                body, project_id=record.project_id, division_id=record.division_id,
                doc_type=record.doc_type,
            )

    # -- assignments (who works on what) ----------------------------------
    for assignment in repository.list_domain("assignments", include_inactive=True):
        if not isinstance(assignment, Assignment):
            continue
        lines = [f"Assignment: {assignment.person_id} on project {assignment.project_id}."]
        if assignment.role:
            lines.append(f"Role: {assignment.role}.")
        if assignment.from_date or assignment.to_date:
            lines.append(f"Period: {assignment.from_date or '?'} to {assignment.to_date or '?'}.")
        if assignment.allocation:
            lines.append(f"Allocation: {assignment.allocation}.")
        add(
            "assignment", f"workplace:assignment:{assignment.record_id}",
            "Assignment", "\n".join(lines),
            project_id=assignment.project_id,
        )

    if report is not None:
        report.documents = len(out)
        report.by_kind = dict(counts)
        report.withheld = withheld
        report.unreadable = unreadable
    return out


class WorkplaceAssistant:
    """The Workplace assistant: the project's grounded assistant over the records.

    One :class:`core.assistant.GroundedAssistant` is built per profile, all over
    the same corpus (the Workplace's own records as citable documents), so a
    profile answers in its own specialization while retrieval stays the project's
    own RAG. The corpus is rebuilt on :meth:`refresh`, so a project or document
    added since the last build becomes citable -- the knowledge is cumulative.

    With no model wired the assistant still answers: extractively, from the cited
    Workplace records. Every answer carries :data:`DISCLAIMER`.
    """

    def __init__(
        self,
        repository: WorkplaceRepository,
        *,
        domain: WorkplaceDomain | None = None,
        documents=None,
        clearance: Sequence[str] = DEFAULT_CLEARANCE,
        model=None,
        include_document_text: bool = True,
    ) -> None:
        self.repository = repository
        self.domain = domain
        self.documents = documents
        self.clearance = tuple(clearance)
        self.model = model
        self.include_document_text = include_document_text
        self.report = CorpusReport()
        self._store: DocumentStore | None = None
        self._assistants: dict[str, GroundedAssistant] = {}

    # -- the corpus -------------------------------------------------------

    def _build_store(self) -> DocumentStore:
        """The Workplace's records plus the project's own knowledge (controlled reuse).

        The project's own corpus (disciplines, courses, the petroleum corpus) is
        reused alongside the Workplace's records -- the brief's §11 "controlled
        reuse of shared knowledge" -- so a Workplace question can also reach the
        project's petroleum knowledge, and the Workplace's own records rank beside
        it. The Workplace records are *views* over its database rows, so nothing is
        duplicated.
        """
        from core.documents import build_document_store

        workplace = build_workplace_documents(
            self.repository,
            documents=self.documents,
            clearance=self.clearance,
            include_document_text=self.include_document_text,
            report=self.report,
        )
        return build_document_store(extra=workplace)

    @property
    def store(self) -> DocumentStore:
        if self._store is None:
            self._store = self._build_store()
        return self._store

    def refresh(self) -> CorpusReport:
        """Rebuild the corpus from the records and drop the cached assistants."""
        self.report = CorpusReport()
        self._store = self._build_store()
        self._assistants.clear()
        return self.report

    def corpus_report(self) -> CorpusReport:
        return self.report

    # -- the assistants ---------------------------------------------------

    def _assistant_for(self, profile: WorkplaceProfile) -> GroundedAssistant:
        cached = self._assistants.get(profile.key)
        if cached is not None:
            return cached
        from core.assistant import build_default_assistant

        assistant = build_default_assistant(
            model=self.model,
            documents=self.store,
            memory=None,
        )
        assistant.system_instruction = profile.system_instruction(assistant.system_instruction)
        self._assistants[profile.key] = assistant
        return assistant

    def ask(self, question: str, *, profile: str = DEFAULT_PROFILE, limit: int = 4):
        """Answer a Workplace question in the chosen profile's specialization."""
        chosen = profile_for(profile)
        assistant = self._assistant_for(chosen)
        return assistant.ask(question, limit=limit)

    def answer_text(self, question: str, *, profile: str = DEFAULT_PROFILE, limit: int = 4) -> str:
        """The answer text with the honesty disclaimer appended."""
        answer = self.ask(question, profile=profile, limit=limit)
        return f"{answer.text}\n\n{DISCLAIMER}"

    def profile_summaries(self, *, language: str = "en") -> list[tuple[str, str, str]]:
        """``(key, name, specialization)`` for every selectable profile."""
        rows = []
        for profile in PROFILES:
            name = profile.name_ar if language == "ar" else profile.name
            rows.append((profile.key, name, profile.specialization))
        return rows

    def status(self, *, language: str = "en") -> str:
        """An honest one-line status of the Workplace assistant (never forces a build)."""
        report = self.report
        total = len(self._store.documents()) if self._store is not None else report.documents
        model = "a language model" if self.model is not None else "the records (no model)"
        return (
            f"Workplace assistant: {len(PROFILES)} profiles; {report.documents} Workplace "
            f"records among {total} citable sources, answered with {model}."
        )


__all__ = [
    "DEFAULT_CLEARANCE",
    "DEFAULT_PROFILE",
    "DISCLAIMER",
    "MAX_DOCUMENTS",
    "MAX_DOCUMENT_CHARS",
    "PROFILES",
    "PROFILE_KEYS",
    "CorpusReport",
    "WorkplaceAssistant",
    "WorkplaceProfile",
    "build_workplace_documents",
    "profile_for",
]
