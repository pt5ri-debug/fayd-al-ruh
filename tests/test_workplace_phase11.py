"""Tests for the Workplace Phase 11: cumulative institutional knowledge.

The user's most-emphasised requirement is that the institutional knowledge base be
**cumulative** -- every completed project, approved report, and documented
technical experience adds value, while the source and the confidentiality of each
piece are preserved. Phase 11 makes that a *workflow*: the finished work the
Workplace already holds is captured into the *existing* knowledge registry (never
a second store), idempotently, carrying its provenance and classification.

Everything here runs on the real capture engine, the real service, and the real
repository -- no model, no network, no display.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cli.workplace import main as workplace_main
from core.workplace.knowledge_capture import (
    SOURCE_KINDS,
    CaptureReport,
    KnowledgeCapture,
    WorkplaceKnowledgeError,
)
from core.workplace.phase11_gate import PHASE11_JUDGEMENTS, workplace_phase11_accuracy
from core.workplace.records import KnowledgeRecord
from gui.workplace import WorkplaceService


@pytest.fixture()
def service(tmp_path: Path) -> WorkplaceService:
    return WorkplaceService(root=tmp_path)


def _completed_project(service: WorkplaceService, number: str = "PRJ-1", title: str = "Pipeline"):
    return service.domain.create(
        "projects", number=number, title=title, scope="Welding inspection", status="completed"
    )


# -- the capture engine ---------------------------------------------------


def test_a_completed_project_is_captured(service: WorkplaceService):
    project = _completed_project(service)
    record = service.capture_knowledge("project", project.record_id)
    assert isinstance(record, KnowledgeRecord)
    assert record.title == "Pipeline"
    assert record.reference == project.record_id
    assert record.source == "workplace:project"
    assert record.kind == "technical_note"


def test_an_in_progress_project_yields_nothing(service: WorkplaceService):
    project = service.domain.create("projects", number="P", title="WIP", status="active")
    assert service.knowledge_capture.candidate("project", project.record_id) is None
    with pytest.raises(WorkplaceKnowledgeError):
        service.capture_knowledge("project", project.record_id)


def test_an_approved_report_is_captured_but_a_draft_is_not(service: WorkplaceService):
    project = _completed_project(service)
    approved = service.domain.create_report(
        project_id=project.record_id, title="Approved", status="approved"
    )
    draft = service.domain.create_report(
        project_id=project.record_id, title="Draft", status="draft"
    )
    assert service.capture_knowledge("report", approved.record_id).reference == approved.record_id
    assert service.knowledge_capture.candidate("report", draft.record_id) is None


def test_a_closed_finding_is_captured_but_an_open_one_is_not(service: WorkplaceService):
    project = _completed_project(service)
    closed = service.domain.create_finding(
        project_id=project.record_id, title="NCR closed", status="closed"
    )
    opened = service.domain.create_finding(
        project_id=project.record_id, title="NCR open", status="open"
    )
    record = service.capture_knowledge("finding", closed.record_id)
    assert record.kind == "lesson"
    assert service.knowledge_capture.candidate("finding", opened.record_id) is None


def test_a_completed_inspection_is_captured(service: WorkplaceService):
    project = _completed_project(service)
    inspection = service.domain.create_inspection(
        project_id=project.record_id, title="Inspection", kind="welding", status="completed",
        result="accepted",
    )
    record = service.capture_knowledge("inspection", inspection.record_id)
    assert record.kind == "best_practice"
    assert record.verified is True


def test_a_finished_test_is_captured(service: WorkplaceService):
    project = _completed_project(service)
    test = service.domain.create_test(
        project_id=project.record_id, title="Tensile", kind="tensile", value="520", unit="MPa",
        result="pass",
    )
    record = service.capture_knowledge("test", test.record_id)
    assert record.reference == test.record_id
    assert "520" in record.summary


def test_an_issued_document_is_captured_but_a_draft_is_not(service: WorkplaceService):
    project = _completed_project(service)
    issued = service.domain.register_document(
        title="Spec", doc_type="technical_report", project_id=project.record_id,
        approved_by="PER-x", issued_on="2026-01-01",
    )
    draft = service.domain.register_document(
        title="Draft spec", doc_type="technical_report", project_id=project.record_id
    )
    assert service.capture_knowledge("document", issued.record_id).reference == issued.record_id
    assert service.knowledge_capture.candidate("document", draft.record_id) is None


# -- provenance, confidentiality, idempotence -----------------------------


def test_the_captured_record_preserves_its_provenance(service: WorkplaceService):
    project = _completed_project(service)
    record = service.capture_knowledge("project", project.record_id)
    assert record.source == "workplace:project"
    assert record.reference == project.record_id


def test_the_captured_record_inherits_the_source_confidentiality(service: WorkplaceService):
    project = _completed_project(service)
    secret = service.domain.register_document(
        title="Secret", doc_type="technical_report", project_id=project.record_id,
        confidentiality="restricted", approved_by="PER-x", issued_on="2026-01-01",
    )
    record = service.capture_knowledge("document", secret.record_id)
    assert record.confidentiality == "restricted"


def test_a_second_capture_is_skipped_not_duplicated(service: WorkplaceService):
    project = _completed_project(service)
    service.capture_knowledge("project", project.record_id)
    before = len(service.captured_knowledge())
    report = service.capture_all_knowledge()
    assert len(service.captured_knowledge()) == before
    assert project.record_id in report.skipped
    assert report.count == 0


def test_an_unknown_source_id_is_refused(service: WorkplaceService):
    with pytest.raises(WorkplaceKnowledgeError):
        service.capture_knowledge("report", "REP-does-not-exist")


def test_an_unknown_source_kind_is_refused(service: WorkplaceService):
    with pytest.raises(WorkplaceKnowledgeError):
        service.capture_knowledge("nonsense", "X-1")


# -- bulk capture ---------------------------------------------------------


def test_capture_all_gathers_every_finished_source(service: WorkplaceService):
    project = _completed_project(service)
    service.domain.create_report(project_id=project.record_id, title="R", status="approved")
    service.domain.create_finding(project_id=project.record_id, title="NCR", status="closed")
    service.domain.create_report(project_id=project.record_id, title="Draft", status="draft")
    report = service.capture_all_knowledge()
    titles = {record.title for record in report.captured}
    assert {"Pipeline", "R", "NCR"} <= titles
    assert "Draft" not in titles


def test_capture_project_gathers_a_projects_own_finished_work(service: WorkplaceService):
    first = _completed_project(service, number="PRJ-1", title="First")
    second = _completed_project(service, number="PRJ-2", title="Second")
    service.domain.create_report(project_id=first.record_id, title="First report", status="approved")
    service.domain.create_report(project_id=second.record_id, title="Second report", status="approved")
    report = service.capture_project_knowledge(first.record_id)
    titles = {record.title for record in report.captured}
    assert "First" in titles and "First report" in titles
    assert "Second" not in titles and "Second report" not in titles


def test_the_capture_report_describes_itself(service: WorkplaceService):
    project = _completed_project(service)
    service.capture_knowledge("project", project.record_id)
    report = service.capture_all_knowledge()
    assert isinstance(report, CaptureReport)
    assert "skipped" in report.describe()
    assert report.as_dict()["skipped"]


def test_the_summary_counts_by_source_kind(service: WorkplaceService):
    project = _completed_project(service)
    service.domain.create_finding(project_id=project.record_id, title="NCR", status="closed")
    service.capture_all_knowledge()
    summary = service.knowledge_capture.summary()
    assert summary["project"] == 1
    assert summary["finding"] == 1
    assert set(summary) == set(SOURCE_KINDS) | {"manual"}


def test_the_status_names_what_is_left(service: WorkplaceService):
    _completed_project(service)
    assert "1 finished source" in service.knowledge_capture_status()
    service.capture_all_knowledge()
    assert "0 finished source" in service.knowledge_capture_status()


def test_the_source_kinds_are_labelled(service: WorkplaceService):
    kinds = dict(service.knowledge_source_kinds())
    assert kinds["project"] == "Completed project"
    assert kinds["finding"] == "Closed finding"


# -- cumulative: a captured record becomes citable -------------------------


def test_a_captured_record_enters_the_assistant_corpus(service: WorkplaceService):
    project = _completed_project(service, title="Deepwater pipeline")
    service.capture_knowledge("project", project.record_id)
    service.refresh_assistant()
    # The knowledge record is now part of the corpus the assistant retrieves from.
    documents = service.assistant.store.documents()
    assert any("Deepwater pipeline" in document.text for document in documents)


def test_the_capture_adds_no_second_store(service: WorkplaceService):
    from core.workplace.schema import TABLES

    assert "knowledge_records" in TABLES
    assert "knowledge_capture" not in TABLES


# -- the held-out gate ----------------------------------------------------


def test_the_gate_is_green_on_the_real_capture():
    assert workplace_phase11_accuracy() == 1.0


def test_the_gate_is_zero_with_no_judgements():
    assert workplace_phase11_accuracy(()) == 0.0


def test_the_catalogue_has_negative_controls():
    keys = {judgement.key for judgement in PHASE11_JUDGEMENTS}
    assert "in-progress-yields-nothing" in keys
    assert "draft-report-yields-nothing" in keys
    assert "open-finding-yields-nothing" in keys
    assert "unknown-refused" in keys
    assert "no-second-store" in keys


# -- the CLI --------------------------------------------------------------


def test_the_cli_lists_source_kinds(tmp_path: Path, capsys):
    assert workplace_main(["--data-root", str(tmp_path), "knowledge-sources"]) == 0
    assert "Completed project" in capsys.readouterr().out


def test_the_cli_reports_the_capture_status(tmp_path: Path, capsys):
    service = WorkplaceService(root=tmp_path)
    _completed_project(service)
    assert workplace_main(["--data-root", str(tmp_path), "knowledge-status"]) == 0
    assert "1 finished source" in capsys.readouterr().out


def test_the_cli_lists_candidates(tmp_path: Path, capsys):
    service = WorkplaceService(root=tmp_path)
    _completed_project(service)
    assert workplace_main(["--data-root", str(tmp_path), "knowledge-candidates", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload and payload[0]["source_kind"] == "project"


def test_the_cli_captures_and_lists(tmp_path: Path, capsys):
    service = WorkplaceService(root=tmp_path)
    _completed_project(service)
    assert workplace_main(["--data-root", str(tmp_path), "capture-all-knowledge"]) == 0
    capsys.readouterr()
    assert workplace_main(["--data-root", str(tmp_path), "knowledge"]) == 0
    assert "captured knowledge record(s)" in capsys.readouterr().out


def test_the_cli_capture_is_idempotent(tmp_path: Path, capsys):
    service = WorkplaceService(root=tmp_path)
    _completed_project(service)
    assert workplace_main(["--data-root", str(tmp_path), "capture-all-knowledge"]) == 0
    capsys.readouterr()
    assert workplace_main(["--data-root", str(tmp_path), "capture-all-knowledge", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 0 and payload["skipped"]


def test_the_cli_captures_a_named_source(tmp_path: Path, capsys):
    service = WorkplaceService(root=tmp_path)
    project = _completed_project(service)
    assert workplace_main([
        "--data-root", str(tmp_path), "capture-knowledge", "project", project.record_id, "--json",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["reference"] == project.record_id


def test_the_cli_captures_a_project(tmp_path: Path, capsys):
    service = WorkplaceService(root=tmp_path)
    project = _completed_project(service)
    service.domain.create_finding(project_id=project.record_id, title="NCR", status="closed")
    assert workplace_main([
        "--data-root", str(tmp_path), "capture-project-knowledge", project.record_id,
    ]) == 0
    assert "Captured 2 knowledge record" in capsys.readouterr().out


def test_the_cli_refuses_an_unfinished_source(tmp_path: Path, capsys):
    service = WorkplaceService(root=tmp_path)
    project = service.domain.create("projects", number="P", title="WIP", status="active")
    code = workplace_main([
        "--data-root", str(tmp_path), "capture-knowledge", "project", project.record_id,
    ])
    assert code == 4  # the refusal exit code
    assert "not finished" in capsys.readouterr().out


def test_the_engine_can_be_built_directly(tmp_path: Path):
    service = WorkplaceService(root=tmp_path)
    capture = KnowledgeCapture(service.repository, domain=service.domain)
    assert isinstance(capture, KnowledgeCapture)
    assert capture.existing_references() == set()
