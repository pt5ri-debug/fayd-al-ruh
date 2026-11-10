"""The held-out gate for the Workplace Phase 11 (cumulative institutional knowledge).

The user's most-emphasised requirement is that the institutional knowledge base
be **cumulative**: every completed project, approved report, and documented
technical experience should add value, while the source and the confidentiality
of each piece are preserved. This module is the measurement of that requirement:
a catalogue of hand-written expectations run against the **real** capture engine
and the **real** repository, each with its own negative control, so a capture that
invented knowledge, duplicated it, dropped its provenance, leaked a classified
source, or captured work that is not finished fails the evaluation instead of
passing silently.

The judgements are:

* a completed project is captured as a knowledge record;
* an **in-progress** project yields nothing (negative control);
* an approved report is captured; a **draft** report yields nothing (negative control);
* a closed NCR is captured; an **open** NCR yields nothing (negative control);
* a second capture of the same source is **skipped**, never duplicated (idempotent);
* the captured record **names its source** (provenance is preserved);
* the captured record **inherits the source's confidentiality** (a restricted
  source yields a restricted record, so the clearance filter still withholds it);
* an **unknown** source id is **refused**, never invented;
* the capture writes to the **existing** ``knowledge_records`` table (no second
  store, the brief's §11 rule);
* ``capture_project`` captures a project's own finished work in one call.

``workplace_phase11_accuracy`` is the share the real behaviour satisfies; with no
repository it is ``0.0`` rather than a faked pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from core.workplace.database import WorkplaceDatabase
from core.workplace.domain import WorkplaceDomain
from core.workplace.knowledge_capture import KnowledgeCapture, WorkplaceKnowledgeError
from core.workplace.repository import WorkplaceRepository


@dataclass(frozen=True)
class Phase11Judgement:
    """One expectation the real capture must satisfy, and why it exists."""

    key: str
    description: str


#: The held-out catalogue. Each key is checked by :func:`_run_judgement`; a key
#: with no check is a failure, so the catalogue and the checks cannot drift.
PHASE11_JUDGEMENTS: tuple[Phase11Judgement, ...] = (
    Phase11Judgement("captures-completed-project", "a completed project becomes knowledge"),
    Phase11Judgement("in-progress-yields-nothing", "an unfinished project yields nothing (negative control)"),
    Phase11Judgement("captures-approved-report", "an approved report becomes knowledge"),
    Phase11Judgement("draft-report-yields-nothing", "a draft report yields nothing (negative control)"),
    Phase11Judgement("captures-closed-finding", "a closed NCR becomes knowledge"),
    Phase11Judgement("open-finding-yields-nothing", "an open NCR yields nothing (negative control)"),
    Phase11Judgement("idempotent", "a second capture of the same source is skipped, never duplicated"),
    Phase11Judgement("names-the-source", "the captured record preserves its provenance"),
    Phase11Judgement("inherits-confidentiality", "a restricted source yields a restricted record"),
    Phase11Judgement("unknown-refused", "an unknown source id is refused, never invented"),
    Phase11Judgement("no-second-store", "the capture writes to the existing knowledge table"),
    Phase11Judgement("captures-a-whole-project", "capture_project gathers the project's finished work"),
)


def _build(tmp: Path) -> tuple[WorkplaceRepository, WorkplaceDomain, KnowledgeCapture]:
    repository = WorkplaceRepository(WorkplaceDatabase(tmp / "workplace.db"))
    domain = WorkplaceDomain(repository)
    capture = KnowledgeCapture(repository, domain=domain)
    return repository, domain, capture


def _run_judgement(
    repository: WorkplaceRepository, domain: WorkplaceDomain, capture: KnowledgeCapture, key: str
) -> bool:
    """Run one judgement against the real capture engine; return whether it held."""
    if key == "captures-completed-project":
        project = domain.create("projects", number="P1", title="Done", status="completed")
        record = capture.capture("project", project.record_id)
        return record.reference == project.record_id and record.title == "Done"
    if key == "in-progress-yields-nothing":
        project = domain.create("projects", number="P2", title="WIP", status="active")
        return capture.candidate("project", project.record_id) is None
    if key == "captures-approved-report":
        project = domain.create("projects", number="P3", title="P", status="active")
        report = domain.create_report(project_id=project.record_id, title="Approved R", status="approved")
        record = capture.capture("report", report.record_id)
        return record.reference == report.record_id
    if key == "draft-report-yields-nothing":
        project = domain.create("projects", number="P4", title="P", status="active")
        report = domain.create_report(project_id=project.record_id, title="Draft R", status="draft")
        return capture.candidate("report", report.record_id) is None
    if key == "captures-closed-finding":
        project = domain.create("projects", number="P5", title="P", status="active")
        finding = domain.create_finding(project_id=project.record_id, title="NCR", status="closed")
        record = capture.capture("finding", finding.record_id)
        return record.kind == "lesson" and record.reference == finding.record_id
    if key == "open-finding-yields-nothing":
        project = domain.create("projects", number="P6", title="P", status="active")
        finding = domain.create_finding(project_id=project.record_id, title="NCR", status="open")
        return capture.candidate("finding", finding.record_id) is None
    if key == "idempotent":
        project = domain.create("projects", number="P7", title="Once", status="completed")
        first = capture.capture("project", project.record_id)
        before = len(capture.knowledge())
        capture_report = capture.capture_all()
        after = len(capture.knowledge())
        return first.record_id in [r.record_id for r in capture.knowledge()] and (
            after == before and project.record_id in capture_report.skipped
        )
    if key == "names-the-source":
        project = domain.create("projects", number="P8", title="Named", status="completed")
        record = capture.capture("project", project.record_id)
        return record.source == "workplace:project" and record.reference == project.record_id
    if key == "inherits-confidentiality":
        project = domain.create("projects", number="P9", title="P", status="active")
        document = domain.register_document(
            title="Secret", doc_type="technical_report", project_id=project.record_id,
            confidentiality="restricted", approved_by="PER-x", issued_on="2026-01-01",
        )
        record = capture.capture("document", document.record_id)
        return record.confidentiality == "restricted"
    if key == "unknown-refused":
        try:
            capture.capture("report", "REP-does-not-exist")
        except WorkplaceKnowledgeError:
            return True
        return False
    if key == "no-second-store":
        from core.workplace.schema import TABLES

        return "knowledge_records" in TABLES and "knowledge_capture" not in TABLES
    if key == "captures-a-whole-project":
        project = domain.create("projects", number="P10", title="Whole", status="completed")
        domain.create_report(project_id=project.record_id, title="R", status="approved")
        domain.create_finding(project_id=project.record_id, title="NCR", status="closed")
        domain.create_report(project_id=project.record_id, title="Draft", status="draft")
        capture_report = capture.capture_project(project.record_id)
        titles = {record.title for record in capture_report.captured}
        return {"Whole", "R", "NCR"} <= titles and "Draft" not in titles
    raise KeyError(key)


def workplace_phase11_accuracy(
    judgements: tuple[Phase11Judgement, ...] = PHASE11_JUDGEMENTS,
) -> float:
    """The share of the held-out catalogue the real capture satisfies."""
    if not judgements:
        return 0.0
    passed = 0
    with TemporaryDirectory() as tmp:
        repository, domain, capture = _build(Path(tmp))
        for judgement in judgements:
            try:
                if _run_judgement(repository, domain, capture, judgement.key):
                    passed += 1
            except Exception:  # a check that raises is a failure, never a pass
                pass
    return passed / len(judgements)


__all__ = [
    "PHASE11_JUDGEMENTS",
    "Phase11Judgement",
    "workplace_phase11_accuracy",
]
