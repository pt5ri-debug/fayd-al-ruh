"""Tests for the Workplace Phase 5: project and personnel management.

Phase 5 adds the two *views* the brief's §6 and §8 ask for -- a project
assembled with every reference resolved, and a personnel directory with a
competency matrix and an explainable project-specific selection -- over the
Phase-4 domain records. Everything is exercised on the real service, the real
database, and the real repository: no model, no network, no display.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cli.workplace import main as workplace_main
from core.workplace.personnel import PersonnelDirectory
from core.workplace.projects import ProjectManager
from gui.workplace import WorkplaceService


@pytest.fixture()
def service(tmp_path: Path) -> WorkplaceService:
    return WorkplaceService(root=tmp_path)


@pytest.fixture()
def domain(service: WorkplaceService):
    return service.domain


def _make_client(domain, name: str, kind: str = "client"):
    return domain.create_client(name=name, kind=kind)


def _make_project(domain, *, number="TSTC-2026-001", title="Coating inspection", **fields):
    return domain.create_project(number=number, title=title, **fields)


# -- the project view (the brief's §6) ------------------------------------


def test_project_view_resolves_every_reference(service, domain):
    client = _make_client(domain, "GUPCO", "client")
    contractor = _make_client(domain, "ENPPI", "contractor")
    consultant = _make_client(domain, "EPRI", "consultant")
    person = domain.create_person(name="Ali Hassan", role="Welding Engineer")
    project = _make_project(
        domain,
        client_id=client.record_id,
        contractor_id=contractor.record_id,
        consultant_id=consultant.record_id,
        manager_id=person.record_id,
    )
    view = service.project_view(project.record_id)
    assert view is not None
    assert view.client is not None and view.client.name == "GUPCO"
    assert view.contractor is not None and view.contractor.name == "ENPPI"
    assert view.consultant is not None and view.consultant.name == "EPRI"
    assert view.manager is not None and view.manager.name == "Ali Hassan"
    assert view.project.number == "TSTC-2026-001"


def test_project_view_gathers_its_own_records(service, domain):
    project = _make_project(domain)
    domain.create_contract(project_id=project.record_id, number="C-1", value="1000")
    domain.create_inspection(project_id=project.record_id, title="Witness the weld")
    domain.create_test(project_id=project.record_id, title="RT", kind="ndt")
    domain.register_document(title="Method statement", project_id=project.record_id)
    domain.create_report(project_id=project.record_id, title="Progress 1")
    domain.create_correspondence(subject="Transmittal", project_id=project.record_id)
    person = domain.create_person(name="Mona")
    domain.assign(project_id=project.record_id, person_id=person.record_id, role="Inspector")
    domain.create_finding(project_id=project.record_id, title="Undercut")
    domain.add_knowledge(title="Lesson on coating", project_id=project.record_id)

    view = service.project_view(project.record_id)
    assert view is not None
    counts = view.counts()
    assert counts["contracts"] == 1
    assert counts["inspections"] == 1
    assert counts["tests"] == 1
    assert counts["documents"] == 1
    assert counts["reports"] == 1
    assert counts["correspondence"] == 1
    assert counts["assignments"] == 1
    assert counts["findings"] == 1
    assert counts["knowledge"] == 1


def test_project_view_unknown_id_is_none(service):
    assert service.project_view("PRJ-does-not-exist") is None


def test_open_findings_and_reports_awaiting_approval_are_counted(service, domain):
    project = _make_project(domain)
    domain.create_finding(project_id=project.record_id, title="A", status="open")
    domain.create_finding(project_id=project.record_id, title="B", status="answered")
    domain.create_finding(project_id=project.record_id, title="C", status="closed")
    domain.create_report(project_id=project.record_id, title="Draft", status="draft")
    domain.create_report(project_id=project.record_id, title="Done", status="approved")
    view = service.project_view(project.record_id)
    assert view is not None
    assert len(view.open_findings) == 2  # an answered NCR is still open
    assert len(view.reports_awaiting_approval) == 1


def test_approvals_include_the_projects_records(service, domain):
    project = _make_project(domain)
    report = domain.create_report(project_id=project.record_id, title="Final report")
    domain.approve(subject_id=report.record_id, subject_kind="reports", approved_by="Manager")
    view = service.project_view(project.record_id)
    assert view is not None
    assert len(view.approvals) == 1
    assert view.approvals[0].subject_id == report.record_id


def test_project_rows_carry_real_counts(service, domain):
    project = _make_project(domain)
    domain.create_finding(project_id=project.record_id, title="A", status="open")
    domain.register_document(title="D", project_id=project.record_id)
    rows = service.project_rows()
    assert len(rows) == 1
    row = rows[0]
    assert row.project.record_id == project.record_id
    assert row.open_findings == 1
    assert row.documents == 1


def test_project_rows_filter_by_status(service, domain):
    _make_project(domain, number="A-1", title="One", status="active")
    _make_project(domain, number="A-2", title="Two", status="completed")
    active = service.project_rows(status="active")
    assert [row.project.number for row in active] == ["A-1"]


def test_project_manager_finds_by_number(service, domain):
    _make_project(domain, number="TSTC-2026-777", title="Pipeline")
    manager = ProjectManager(service.repository)
    found = manager.project_by_number("TSTC-2026-777")
    assert found is not None and found.title == "Pipeline"


def test_project_documents_filter_by_type(service, domain):
    project = _make_project(domain)
    domain.register_document(title="Method", project_id=project.record_id, doc_type="method_statement")
    domain.register_document(title="Report", project_id=project.record_id, doc_type="report")
    documents = service.project_documents(project.record_id, doc_type="method_statement")
    assert [d.title for d in documents] == ["Method"]


def test_assign_person_is_audited(service, domain):
    project = _make_project(domain)
    person = domain.create_person(name="Sara")
    service.assign_person(project_id=project.record_id, person_id=person.record_id, role="Lead")
    assignments = service.project_assignments(project.record_id)
    assert len(assignments) == 1 and assignments[0].role == "Lead"
    assert any(row.action == "create" and row.entity == "assignments" for row in domain.audit_tail())


# -- the personnel directory (the brief's §8) -----------------------------


def test_personnel_directory_lists_and_filters(service, domain):
    division = service.divisions()[0]
    domain.create_person(name="Ali", role="Welding Engineer", division_id=division.record_id)
    domain.create_person(name="Mona", role="Chemist", discipline="Analytical Chemistry")
    everyone = service.personnel_people()
    assert {p.name for p in everyone} == {"Ali", "Mona"}
    chemists = service.personnel_people(discipline="Analytical Chemistry")
    assert [p.name for p in chemists] == ["Mona"]
    by_division = service.personnel_people(division_id=division.record_id)
    assert [p.name for p in by_division] == ["Ali"]


def test_person_profile_gathers_qualifications_and_projects(service, domain):
    person = domain.create_person(name="Ali", role="Welding Engineer")
    domain.add_qualification(person_id=person.record_id, title="CSWIP 3.1", kind="certification")
    project = _make_project(domain)
    domain.assign(project_id=project.record_id, person_id=person.record_id, role="Lead")
    profile = service.person_view(person.record_id)
    assert profile is not None
    assert "CSWIP 3.1" in profile.certification_titles
    assert [p.number for p in profile.projects] == [project.number]


def test_person_profile_unknown_is_none(service):
    assert service.person_view("PER-missing") is None


def test_competency_matrix_marks_holdings(service, domain):
    ali = domain.create_person(name="Ali")
    mona = domain.create_person(name="Mona")
    domain.add_qualification(person_id=ali.record_id, title="CSWIP 3.1")
    domain.add_qualification(person_id=ali.record_id, title="ASNT Level II")
    domain.add_qualification(person_id=mona.record_id, title="CSWIP 3.1")
    matrix = service.competency_matrix()
    assert "CSWIP 3.1" in matrix.columns
    coverage = matrix.coverage()
    assert coverage["CSWIP 3.1"] == 2
    assert coverage["ASNT Level II"] == 1


def test_competency_matrix_is_empty_with_no_data(service):
    matrix = service.competency_matrix()
    assert matrix.empty


def test_selection_ranks_by_coverage(service, domain):
    ali = domain.create_person(name="Ali", role="Welding Engineer")
    mona = domain.create_person(name="Mona", role="Chemist")
    domain.add_qualification(person_id=ali.record_id, title="CSWIP 3.1")
    candidates = service.select_personnel(
        skills=("Welding",), qualifications=("CSWIP 3.1",)
    )
    assert candidates[0].person.record_id == ali.record_id
    assert candidates[0].score == 1.0
    assert set(candidates[0].matched) == {"Welding", "CSWIP 3.1"}
    # Mona matched nothing, so she is ranked below Ali and her gap is named.
    mona_row = next(c for c in candidates if c.person.record_id == mona.record_id)
    assert mona_row.score == 0.0
    assert set(mona_row.missing) == {"Welding", "CSWIP 3.1"}


def test_selection_is_deterministic(service, domain):
    domain.create_person(name="Bilal", role="Inspector")
    domain.create_person(name="Amr", role="Inspector")
    first = [c.person.name for c in service.select_personnel(skills=("Inspector",))]
    second = [c.person.name for c in service.select_personnel(skills=("Inspector",))]
    assert first == second == ["Amr", "Bilal"]


def test_selection_can_exclude_unavailable_people(service, domain):
    domain.create_person(name="Busy", role="Inspector", available=False)
    domain.create_person(name="Free", role="Inspector", available=True)
    candidates = service.select_personnel(skills=("Inspector",))
    assert [c.person.name for c in candidates] == ["Free"]


def test_read_only_directory_refuses_writes(service):
    directory = PersonnelDirectory(service.repository)
    with pytest.raises(RuntimeError):
        directory.add_person(name="Nobody")


# -- the CLI (the headless twin) ------------------------------------------


def _run(tmp_path: Path, *argv: str) -> int:
    return workplace_main(["--data-root", str(tmp_path), *argv])


def test_cli_projects_lists_real_records(tmp_path: Path, capsys):
    service = WorkplaceService(root=tmp_path)
    service.domain.create_project(number="TSTC-2026-002", title="Weld map", status="active")
    assert _run(tmp_path, "projects", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["number"] == "TSTC-2026-002"
    assert payload[0]["status"] == "active"


def test_cli_project_shows_the_full_view(tmp_path: Path, capsys):
    service = WorkplaceService(root=tmp_path)
    client = service.domain.create_client(name="GUPCO")
    project = service.domain.create_project(
        number="TSTC-2026-003", title="Pipeline", client_id=client.record_id
    )
    assert _run(tmp_path, "project", project.record_id, "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["client"] == "GUPCO"
    assert payload["number"] == "TSTC-2026-003"


def test_cli_project_accepts_a_number(tmp_path: Path, capsys):
    service = WorkplaceService(root=tmp_path)
    service.domain.create_project(number="TSTC-2026-004", title="Flowlines")
    assert _run(tmp_path, "project", "TSTC-2026-004") == 0
    assert "Flowlines" in capsys.readouterr().out


def test_cli_project_unknown_is_unavailable(tmp_path: Path, capsys):
    assert _run(tmp_path, "project", "PRJ-nope") == 3
    assert "No project" in capsys.readouterr().out


def test_cli_people_and_person(tmp_path: Path, capsys):
    service = WorkplaceService(root=tmp_path)
    person = service.domain.create_person(name="Ali", role="Welding Engineer")
    assert _run(tmp_path, "people", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["name"] == "Ali"
    assert _run(tmp_path, "person", person.record_id) == 0
    assert "Ali" in capsys.readouterr().out


def test_cli_matrix_and_select(tmp_path: Path, capsys):
    service = WorkplaceService(root=tmp_path)
    ali = service.domain.create_person(name="Ali", role="Welding Engineer")
    service.domain.add_qualification(person_id=ali.record_id, title="CSWIP 3.1")
    assert _run(tmp_path, "matrix", "--json") == 0
    payload = json.loads(capsys.readouterr().out)
    assert "CSWIP 3.1" in payload["columns"]
    assert _run(
        tmp_path, "select", "--skill", "Welding", "--qualification", "CSWIP 3.1", "--json"
    ) == 0
    ranked = json.loads(capsys.readouterr().out)
    assert ranked[0]["name"] == "Ali"
    assert ranked[0]["score"] == 1.0
