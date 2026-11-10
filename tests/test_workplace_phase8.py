"""Tests for the Workplace Phase 8: reporting and proposal workflows (§9, §10, §13).

Phase 8 adds the workflow that ties a project's records to its deliverables: a
**draft** assembled from the records the Workplace already holds, a **lifecycle**
that a report moves through, and an **approval** recorded as a new row -- so a
deliverable is only ever *issued* through a person's decision, never the AI's
(the brief's §13). Everything here runs on the real service, the real database,
and the real repository: no model, no network, no display.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cli.workplace import main as workplace_main
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
from gui.workplace import WorkplaceService


@pytest.fixture()
def service(tmp_path: Path) -> WorkplaceService:
    return WorkplaceService(root=tmp_path)


@pytest.fixture()
def domain(service: WorkplaceService):
    return service.domain


def _project(service: WorkplaceService, number: str = "TSTC-2026-100"):
    return service.domain.create_project(
        number=number,
        title="Flowline coating and inspection",
        status="active",
        scope="External coating of the 12-inch flowline to project specification.",
        location="Alexandria",
    )


def _run(tmp_path: Path, *argv: str) -> int:
    return workplace_main(["--data-root", str(tmp_path), *argv])


# -- the vocabulary (the brief's §9) --------------------------------------


def test_report_kinds_are_the_briefs_set():
    for kind in ("technical_report", "inspection_report", "proposal", "method_statement"):
        assert kind in REPORT_KINDS
    for kind in REPORT_KINDS:
        assert report_kind_label(kind)
        assert report_kind_label(kind, language="ar")
    assert report_kind_label("nonsense") == "nonsense"


def test_report_statuses_are_a_closed_set_with_labels():
    for status in REPORT_STATUSES:
        assert status_label(status)
        assert status_label(status, language="ar")
    assert status_label("nonsense") == "nonsense"


def test_the_lifecycle_only_reaches_issued_through_approved():
    assert can_transition("draft", "submitted")
    assert can_transition("submitted", "approved")
    assert can_transition("approved", "issued")
    # A draft can never jump straight to issued.
    assert not can_transition("draft", "issued")
    assert not can_transition("draft", "approved")
    # An issued report is terminal except for supersession.
    assert can_transition("issued", "superseded")
    assert not can_transition("issued", "draft")
    assert REPORT_TRANSITIONS["superseded"] == frozenset()


def test_the_transition_table_matches_the_can_transition_helper():
    for current, targets in REPORT_TRANSITIONS.items():
        for target in REPORT_STATUSES:
            assert can_transition(current, target) == (target in targets)


# -- drafting from the project's own records (§10) ------------------------


def test_a_draft_is_assembled_from_the_records(service):
    service.open()
    project = _project(service)
    service.domain.create_inspection(
        project_id=project.record_id,
        title="Coating DFT check",
        itp_reference="API RP 5L2",
        result="pass",
        status="performed",
    )
    service.domain.create_test(
        project_id=project.record_id,
        title="Pull-off adhesion",
        standard_reference="ASTM D4541",
        result="pass",
        value="8.5",
        unit="MPa",
    )
    service.domain.create_finding(
        project_id=project.record_id,
        title="Holiday in field joint",
        severity="major",
        status="open",
    )
    draft = service.draft_report(project.record_id, kind="technical_report")
    assert isinstance(draft, ReportDraft)
    assert draft.kind == "technical_report"
    text = draft.to_text()
    assert "Coating DFT check" in text
    assert "Pull-off adhesion" in text
    assert "Holiday in field joint" in text
    assert "API RP 5L2" in text
    assert "ASTM D4541" in text


def test_a_draft_never_invents_an_empty_section(service):
    service.open()
    project = _project(service)
    draft = service.draft_report(project.record_id, kind="technical_report")
    inspections = draft.section("Inspections")
    assert inspections is not None
    assert inspections.empty  # nothing on file, honestly reported
    assert "nothing on file" in draft.to_text()


def test_a_draft_with_no_assistant_is_deterministic(service):
    service.open()
    project = _project(service)
    first = service.draft_report(project.record_id, kind="proposal").to_text()
    second = service.draft_report(project.record_id, kind="proposal").to_text()
    assert first == second


def test_an_unknown_project_is_refused(service):
    service.open()
    with pytest.raises(ReportingError):
        service.draft_report("PRJ-does-not-exist")


def test_an_unknown_report_kind_is_refused(service):
    service.open()
    project = _project(service)
    with pytest.raises(ReportingError):
        service.draft_report(project.record_id, kind="nonsense")


def test_the_management_summary_reports_real_counts(service):
    service.open()
    project = _project(service)
    service.domain.create_finding(project_id=project.record_id, title="NCR-1", status="open")
    draft = service.draft_report(project.record_id, kind="management_summary")
    figures = draft.section("Management figures")
    assert figures is not None
    assert any("open finding" in line for line in figures.lines)


def test_a_proposal_draft_carries_the_contractual_basis(service):
    service.open()
    project = _project(service)
    service.domain.create(
        "contracts",
        project_id=project.record_id,
        number="C-2026-77",
        value="1500000",
        currency="EGP",
        starts_on="2026-01-01",
        ends_on="2026-06-30",
    )
    draft = service.draft_report(project.record_id, kind="proposal")
    commercial = draft.section("Contractual basis")
    assert commercial is not None
    assert any("C-2026-77" in line for line in commercial.lines)


# -- the lifecycle (the brief's §13) --------------------------------------


def test_a_report_moves_draft_to_submitted(service):
    service.open()
    project = _project(service)
    report = service.create_report(
        project.record_id, kind="technical_report", title="Draft A"
    )
    assert report.status == "draft"
    submitted = service.submit_report(report.record_id)
    assert submitted.status == "submitted"


def test_a_report_cannot_be_issued_without_approval(service):
    service.open()
    project = _project(service)
    report = service.create_report(project.record_id, title="Draft B")
    with pytest.raises(ReportingError):
        service.issue_report(report.record_id)


def test_an_approval_records_a_new_row_and_the_approver(service):
    service.open()
    project = _project(service)
    report = service.create_report(project.record_id, title="Draft C")
    service.submit_report(report.record_id)
    approved = service.approve_report(report.record_id, approved_by="Dr. Ahmed")
    assert approved.status == "approved"
    approvals = service.report_approvals(report.record_id)
    assert len(approvals) == 1
    assert approvals[0].approved_by == "Dr. Ahmed"
    assert approvals[0].decision == "approved"


def test_an_approval_must_name_a_person(service):
    service.open()
    project = _project(service)
    report = service.create_report(project.record_id, title="Draft D")
    service.submit_report(report.record_id)
    with pytest.raises(ReportingError):
        service.approve_report(report.record_id, approved_by="   ")


def test_a_second_approval_is_a_new_row_never_an_overwrite(service):
    service.open()
    project = _project(service)
    report = service.create_report(project.record_id, title="Draft E")
    service.submit_report(report.record_id)
    service.approve_report(report.record_id, approved_by="Dr. Ahmed")
    # Re-approval requires moving back to draft first (the lifecycle forbids a
    # double approval from the approved state), so the row count stays honest.
    with pytest.raises(ReportingError):
        service.approve_report(report.record_id, approved_by="Dr. Mona")
    assert len(service.report_approvals(report.record_id)) == 1


def test_a_rejected_report_returns_to_draft(service):
    service.open()
    project = _project(service)
    report = service.create_report(project.record_id, title="Draft F")
    service.submit_report(report.record_id)
    rejected = service.reject_report(report.record_id, note="Needs more evidence")
    assert rejected.status == "rejected"
    # A rejected report may be reworked back to draft.
    reworked = service.reporting._transition(report.record_id, "draft")
    assert reworked.status == "draft"


def test_a_draft_cannot_be_approved_directly(service):
    service.open()
    project = _project(service)
    report = service.create_report(project.record_id, title="Draft G")
    with pytest.raises(ReportingError):
        service.approve_report(report.record_id, approved_by="Dr. Ahmed")


# -- issuing a deliverable (§9) -------------------------------------------


def test_issuing_registers_the_deliverable_as_a_document(service):
    service.open()
    project = _project(service)
    report = service.create_report(
        project.record_id, kind="technical_report", title="Final report", body="Body text."
    )
    service.submit_report(report.record_id)
    service.approve_report(report.record_id, approved_by="Dr. Ahmed")
    deliverable = service.issue_report(report.record_id)
    assert isinstance(deliverable, Deliverable)
    assert deliverable.report.status == "issued"
    assert deliverable.document_id
    assert deliverable.document_title == "Final report"
    # The deliverable is in the Workplace's own document registry (Phase 6).
    assert service.document_record(deliverable.document_id) is not None


def test_issuing_can_skip_the_document(service):
    service.open()
    project = _project(service)
    report = service.create_report(project.record_id, title="No file")
    service.submit_report(report.record_id)
    service.approve_report(report.record_id, approved_by="Dr. Ahmed")
    deliverable = service.issue_report(report.record_id, register_document=False)
    assert deliverable.report.status == "issued"
    assert deliverable.document_id == ""


def test_the_deliverables_list_holds_only_issued_deliverables(service):
    service.open()
    project = _project(service)
    draft = service.create_report(project.record_id, kind="technical_report", title="Open draft")
    issued = service.create_report(project.record_id, kind="technical_report", title="Issued one")
    service.submit_report(issued.record_id)
    service.approve_report(issued.record_id, approved_by="Dr. Ahmed")
    service.issue_report(issued.record_id)
    deliverables = service.project_deliverables(project.record_id)
    ids = {d.report.record_id for d in deliverables}
    assert issued.record_id in ids
    assert draft.record_id not in ids


def test_superseding_keeps_the_report_row(service):
    service.open()
    project = _project(service)
    report = service.create_report(project.record_id, title="Rev A")
    service.submit_report(report.record_id)
    service.approve_report(report.record_id, approved_by="Dr. Ahmed")
    service.issue_report(report.record_id, register_document=False)
    superseded = service.supersede_report(report.record_id)
    assert superseded.status == "superseded"
    assert service.report_record(report.record_id) is not None


def test_project_reports_can_be_filtered_by_kind(service):
    service.open()
    project = _project(service)
    service.create_report(project.record_id, kind="technical_report", title="Tech")
    service.create_report(project.record_id, kind="proposal", title="Prop")
    proposals = service.project_reports(project.record_id, kind="proposal")
    assert [r.title for r in proposals] == ["Prop"]


# -- saving a draft -------------------------------------------------------


def test_saving_a_draft_creates_a_report_row(service):
    service.open()
    project = _project(service)
    person = service.domain.create("people", name="Dr. Ahmed", role="QA Manager")
    draft = service.draft_report(project.record_id, kind="proposal")
    report = service.save_report_draft(
        project.record_id, draft, author_id=person.record_id, number="R-1"
    )
    assert report.status == "draft"
    assert report.number == "R-1"
    assert report.body == draft.to_text()


# -- the assistant fills a section as advice (§10, §13) --------------------


class _StubAssistant:
    """A stand-in assistant: a cited answer, marked as AI advice."""

    def ask(self, question: str, *, profile: str = "", limit: int = 4):
        from core.assistant import AssistantAnswer

        return AssistantAnswer(
            text="A cited recommendation about the coating system.",
            citations=("[workplace:DOC-1] Coating spec",),
            passages=(),
            confidence=0.5,
            used_model=True,
            conversation_id="stub",
        )


def test_a_draft_filled_by_the_assistant_is_marked_as_advice(service):
    service.open()
    project = _project(service)
    workflow = ReportingWorkflow(
        service.repository,
        domain=service.domain,
        projects=service.projects,
        assistant=_StubAssistant(),
        documents=service.document_manager,
    )
    draft = workflow.draft(project.record_id, kind="proposal", use_assistant=True)
    assert draft.used_assistant
    assert draft.citations
    overview = draft.section("Overview")
    assert overview is not None and overview.ai
    # The AI disclaimer is carried in the draft text.
    assert "AI advice" in draft.to_text()


def test_an_assistant_failure_never_breaks_a_draft(service):
    service.open()
    project = _project(service)

    class _Broken:
        def ask(self, question: str, *, profile: str = "", limit: int = 4):
            raise RuntimeError("model down")

    workflow = ReportingWorkflow(
        service.repository,
        domain=service.domain,
        projects=service.projects,
        assistant=_Broken(),
        documents=service.document_manager,
    )
    draft = workflow.draft(project.record_id, kind="proposal", use_assistant=True)
    # Falls back to the deterministic sections rather than failing.
    assert not draft.used_assistant
    assert "Flowline coating" in draft.to_text()


# -- the read-only workflow refuses a write -------------------------------


def test_a_read_only_workflow_refuses_a_write(service):
    service.open()
    project = _project(service)
    workflow = ReportingWorkflow(service.repository)  # no domain
    with pytest.raises(ReportingError):
        workflow.create(project.record_id, title="No write")


# -- the CLI (the headless twin) ------------------------------------------


def test_cli_report_kinds_and_statuses(tmp_path, capsys):
    assert _run(tmp_path, "report-kinds", "--json") == 0
    kinds = json.loads(capsys.readouterr().out)
    assert {k["key"] for k in kinds} == set(REPORT_KINDS)
    assert _run(tmp_path, "report-statuses") == 0
    assert "draft" in capsys.readouterr().out


def test_cli_drafts_saves_and_issues(tmp_path, capsys):
    service = WorkplaceService(root=tmp_path)
    service.open()
    project = _project(service, number="TSTC-2026-200")
    # Draft and save.
    assert _run(tmp_path, "draft-report", project.record_id, "--kind", "proposal", "--save") == 0
    out = capsys.readouterr().out
    assert "Saved as report" in out
    reports = service.project_reports(project.record_id)
    assert reports
    report_id = reports[0].record_id
    # Submit, approve, issue.
    assert _run(tmp_path, "submit-report", report_id) == 0
    assert _run(tmp_path, "approve-report", report_id, "--by", "Dr. Ahmed") == 0
    capsys.readouterr()
    assert _run(tmp_path, "issue-report", report_id, "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "issued"
    assert payload["document_id"]
    assert _run(tmp_path, "deliverables", project.record_id) == 0
    assert "issued" in capsys.readouterr().out.lower()


def test_cli_refuses_to_issue_an_unapproved_report(tmp_path, capsys):
    service = WorkplaceService(root=tmp_path)
    service.open()
    project = _project(service, number="TSTC-2026-300")
    report = service.create_report(project.record_id, title="Not approved")
    # The refusal is reported, and the exit code is the "refused" one.
    code = _run(tmp_path, "issue-report", report.record_id)
    assert code != 0
    assert "approved" in capsys.readouterr().out.lower()


def test_cli_report_approvals(tmp_path, capsys):
    service = WorkplaceService(root=tmp_path)
    service.open()
    project = _project(service, number="TSTC-2026-400")
    report = service.create_report(project.record_id, title="Approved report")
    service.submit_report(report.record_id)
    service.approve_report(report.record_id, approved_by="Dr. Mona", role="QA Manager")
    assert _run(tmp_path, "report-approvals", report.record_id, "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["approved_by"] == "Dr. Mona"
    assert payload[0]["role"] == "QA Manager"


def test_cli_report_show(tmp_path, capsys):
    service = WorkplaceService(root=tmp_path)
    service.open()
    project = _project(service, number="TSTC-2026-500")
    report = service.create_report(project.record_id, title="Shown report", body="The body.")
    assert _run(tmp_path, "report", report.record_id) == 0
    out = capsys.readouterr().out
    assert "Shown report" in out
    assert "The body." in out


# -- the section object ---------------------------------------------------


def test_a_report_section_describes_itself():
    empty = ReportSection("Scope")
    assert empty.empty
    assert "nothing on file" in empty.describe()
    filled = ReportSection("Scope", ("Line one",), ai=True)
    assert not filled.empty
    assert "Line one" in filled.describe()
    assert "AI advice" in filled.describe()
