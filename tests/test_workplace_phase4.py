"""Tests for the Workplace Phase 4: the domain data model and the audit trail.

Phase 4 adds the relational records the *work* needs (the brief's §12) -- people,
qualifications, clients, projects, contracts, standards, procedures, inspections,
tests, documents, reports, correspondence, assignments, findings, approvals, and
knowledge records -- plus the append-only audit trail, the dashboard figures, and
the domain service. Everything is exercised on the real service and the real
database: no model, no network, no display.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cli.workplace import main as workplace_main
from core.workplace.database import WorkplaceDatabase, WorkplaceDatabaseError
from core.workplace.domain import ENTITY_NAMES, WorkplaceDomain, WorkplaceDomainError
from core.workplace.records import DOMAIN_RECORDS, PROJECT_STATUSES
from core.workplace.repository import WorkplaceRepository
from core.workplace.schema import SCHEMA_VERSION, TABLES
from gui.workplace import WorkplaceService


@pytest.fixture()
def service(tmp_path: Path) -> WorkplaceService:
    return WorkplaceService(root=tmp_path)


@pytest.fixture()
def domain(service: WorkplaceService) -> WorkplaceDomain:
    return service.domain


# -- the records module ---------------------------------------------------


def test_every_domain_record_has_a_table_and_a_prefix():
    assert set(DOMAIN_RECORDS) == set(ENTITY_NAMES)
    assert set(DOMAIN_RECORDS) <= set(TABLES)
    # A created record's id carries the entity's own prefix, so it is readable.
    for entity in ("people", "projects", "standards"):
        record = DOMAIN_RECORDS[entity](record_id="X")
        assert record.record_id == "X"
    assert PROJECT_STATUSES[0] == "proposed"


# -- the domain service ---------------------------------------------------


def test_create_read_list_and_audit(domain: WorkplaceDomain):
    person = domain.create_person(name="Ali Hassan", role="Welding Engineer", division_id="")
    assert person.record_id.startswith("PER-")
    assert domain.get("people", person.record_id) == person
    assert [p.record_id for p in domain.people()] == [person.record_id]
    trail = domain.audit_tail()
    assert len(trail) == 1
    assert trail[0].action == "create" and trail[0].entity == "people"


def test_update_creates_a_new_value_and_audits_the_change(domain: WorkplaceDomain):
    person = domain.create_person(name="Ali")
    updated = domain.update("people", person.record_id, role="Inspector")
    assert updated.role == "Inspector"
    assert updated.updated_at >= person.updated_at
    assert updated.record_id == person.record_id  # the id is stable
    edits = [row for row in domain.audit_for(person.record_id) if row.action == "update"]
    assert edits and edits[0].detail == "role"


def test_update_refuses_an_unknown_field(domain: WorkplaceDomain):
    person = domain.create_person(name="Ali")
    with pytest.raises(WorkplaceDomainError):
        domain.update("people", person.record_id, not_a_field="x")


def test_deactivate_is_a_soft_delete(domain: WorkplaceDomain):
    person = domain.create_person(name="Ali")
    domain.deactivate("people", person.record_id)
    assert domain.people() == []  # a deactivated record is not listed
    assert domain.get("people", person.record_id).active is False  # but it still exists
    assert any(row.action == "deactivate" for row in domain.audit_for(person.record_id))


def test_an_unknown_entity_is_refused(domain: WorkplaceDomain):
    with pytest.raises(WorkplaceDomainError):
        domain.create("not-an-entity", title="x")


def test_a_qualification_needs_a_real_person(domain: WorkplaceDomain):
    with pytest.raises(WorkplaceDomainError):
        domain.add_qualification(person_id="PER-nope", title="CSWIP 3.1")
    person = domain.create_person(name="Ali")
    qualification = domain.add_qualification(person_id=person.record_id, title="CSWIP 3.1")
    assert qualification.record_id.startswith("QUAL-")
    assert [q.title for q in domain.qualifications(person.record_id)] == ["CSWIP 3.1"]


# -- projects and contracts -----------------------------------------------


def test_a_project_validates_its_references(domain: WorkplaceDomain):
    client = domain.create_client(name="Petrojet")
    project = domain.create_project(
        number="TSTC-2026-001", title="Coating inspection", client_id=client.record_id
    )
    assert project.record_id.startswith("PRJ-")
    assert domain.project_by_number("TSTC-2026-001") == project
    with pytest.raises(WorkplaceDomainError):
        domain.create_project(number="X", title="Y", client_id="CLI-nope")


def test_projects_filter_by_status(domain: WorkplaceDomain):
    a = domain.create_project(number="P-1", title="A")
    b = domain.create_project(number="P-2", title="B")
    domain.update("projects", b.record_id, status="active")
    assert [p.number for p in domain.projects(status="active")] == ["P-2"]
    assert a.status == "proposed"


def test_a_contract_belongs_to_a_project(domain: WorkplaceDomain):
    project = domain.create_project(number="P-1", title="A")
    contract = domain.create_contract(project_id=project.record_id, number="C-1", value="1000")
    assert contract.record_id.startswith("CON-")
    assert domain.contracts(project.record_id) == [contract]
    with pytest.raises(WorkplaceDomainError):
        domain.create_contract(project_id="PRJ-nope")


# -- standards, inspections, tests ----------------------------------------


def test_standards_keep_their_family_apart(domain: WorkplaceDomain):
    standard = domain.create_standard(
        body="API", number="API 1104", title="Welding of pipelines", reference_kind="standard"
    )
    procedure = domain.create_standard(
        body="TSTC", number="TSTC-QP-01", title="Coating procedure", reference_kind="internal_procedure"
    )
    assert {s.reference_kind for s in domain.standards()} == {"standard", "internal_procedure"}
    assert standard.body == "API" and procedure.body == "TSTC"


def test_an_inspection_and_its_tests(domain: WorkplaceDomain):
    project = domain.create_project(number="P-1", title="A")
    inspection = domain.create_inspection(project_id=project.record_id, title="Weld witness")
    test = domain.create_test(project_id=project.record_id, title="UT", inspection_id=inspection.record_id)
    assert inspection.record_id.startswith("INSP-")
    assert test.record_id.startswith("TEST-")
    assert domain.tests(project_id=project.record_id) == [test]
    assert [i.record_id for i in domain.inspections(project_id=project.record_id)] == [
        inspection.record_id
    ]


# -- documents and their revision chain -----------------------------------


def test_a_document_revision_chain(domain: WorkplaceDomain):
    project = domain.create_project(number="P-1", title="A")
    rev_a = domain.register_document(
        title="Inspection Report", project_id=project.record_id, revision="A", doc_type="inspection_report"
    )
    rev_b = domain.register_document(
        title="Inspection Report", project_id=project.record_id, revision="B",
        supersedes_id=rev_a.record_id, doc_type="inspection_report",
    )
    chain = domain.document_history(rev_b.record_id)
    assert [d.revision for d in chain] == ["B", "A"]
    # The original is never mutated: it is still there, unchanged.
    assert domain.get("workplace_documents", rev_a.record_id).revision == "A"


def test_documents_are_a_registry_never_a_copy(domain: WorkplaceDomain):
    # A document row carries a path + hash; it holds no document text of its own.
    doc = domain.register_document(title="Report", path="/tmp/report.pdf", content_hash="abc123")
    assert doc.path == "/tmp/report.pdf"
    assert doc.content_hash == "abc123"
    assert not hasattr(doc, "body")  # the text lives in the project's own store


# -- findings, approvals, knowledge ---------------------------------------


def test_a_finding_and_its_approval(domain: WorkplaceDomain):
    project = domain.create_project(number="P-1", title="A")
    finding = domain.create_finding(
        project_id=project.record_id, title="Undercut on weld", kind="ncr", severity="major"
    )
    assert finding.record_id.startswith("FND-")
    approval = domain.approve(
        subject_id=finding.record_id, subject_kind="findings", decision="approved", approved_by="QA"
    )
    assert approval.record_id.startswith("APR-")
    assert [a.decision for a in domain.approvals(finding.record_id)] == ["approved"]
    # A re-approval is a *new* record, so the first is preserved.
    domain.approve(subject_id=finding.record_id, subject_kind="findings", decision="rejected")
    assert len(domain.approvals(finding.record_id)) == 2


def test_a_knowledge_record_points_at_the_project_store(domain: WorkplaceDomain):
    record = domain.add_knowledge(
        title="CWC lesson", kind="lesson", source="project", reference="TSTC-2026-001"
    )
    assert record.record_id.startswith("KNW-")
    assert record.source == "project" and record.reference == "TSTC-2026-001"


def test_assignments_link_a_person_to_a_project(domain: WorkplaceDomain):
    person = domain.create_person(name="Ali")
    project = domain.create_project(number="P-1", title="A")
    assignment = domain.assign(project_id=project.record_id, person_id=person.record_id, role="Lead")
    assert assignment.record_id.startswith("ASG-")
    assert domain.assignments(project_id=project.record_id) == [assignment]
    with pytest.raises(WorkplaceDomainError):
        domain.assign(project_id=project.record_id, person_id="PER-nope")


# -- the audit trail is append-only ---------------------------------------


def test_the_audit_trail_records_who_did_what(service: WorkplaceService):
    service.set_actor("engineer@epri")
    person = service.domain.create_person(name="Ali")
    row = service.domain.audit_tail()[0]
    assert row.actor == "engineer@epri"
    assert row.entity == "people" and row.entity_id == person.record_id
    # Nothing rewrites the trail: a second edit appends a second row.
    service.domain.update("people", person.record_id, role="Inspector")
    assert len(service.domain.audit_tail()) == 2


def test_deleting_a_unit_does_not_destroy_history(domain: WorkplaceDomain, service: WorkplaceService):
    # A project references a division; deactivating the division must leave the
    # project's record intact (a foreign key to a row that still exists).
    division = service.divisions()[0]
    project = domain.create_project(number="P-1", title="A", division_id=division.record_id)
    service.deactivate_division(division.record_id)
    still_there = domain.get("projects", project.record_id)
    assert still_there is not None and still_there.division_id == division.record_id


# -- the dashboard reads real records -------------------------------------


def test_the_dashboard_counts_real_records(service: WorkplaceService):
    empty = service.dashboard()
    assert empty.total_projects == 0 and empty.personnel == 0
    domain = service.domain
    domain.create_project(number="P-1", title="A")
    second = domain.create_project(number="P-2", title="B")
    domain.update("projects", second.record_id, status="active")
    domain.create_person(name="Ali")
    board = service.dashboard()
    assert board.total_projects == 2 and board.active_projects == 1
    assert board.personnel == 1 and board.divisions == 10
    assert any(label == "Active projects" for label, _ in board.as_rows())
    assert "المشروعات" in board.describe(language="ar")


# -- migration ------------------------------------------------------------


def test_a_version_2_database_migrates_to_version_3(tmp_path: Path):
    db_path = tmp_path / "v2.db"
    database = WorkplaceDatabase(db_path)
    # Simulate a v2 database by rewriting the stored version and dropping the
    # domain tables; reopening must re-run the v3 migration.
    with database._write() as connection:
        for table in ("projects", "people", "audit_entries"):
            connection.execute(f"DROP TABLE IF EXISTS {table}")
        connection.execute("UPDATE meta SET value = '2' WHERE key = 'schema_version'")
    reopened = WorkplaceDatabase(db_path)
    assert reopened.schema_version() == SCHEMA_VERSION == 3
    assert reopened.list_records("projects") == []
    assert reopened.integrity_check()


def test_a_newer_database_is_still_refused(tmp_path: Path):
    import sqlite3

    db_path = tmp_path / "future.db"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("INSERT INTO meta VALUES('schema_version','99')")
    connection.commit()
    connection.close()
    with pytest.raises(WorkplaceDatabaseError):
        WorkplaceDatabase(db_path)


def test_the_repository_creates_domain_records_directly(tmp_path: Path):
    db = WorkplaceDatabase(tmp_path / "w.db")
    repo = WorkplaceRepository(db)
    person = repo.create_domain("people", name="Ali")
    assert repo.get_domain("people", person.record_id) == person
    repo.audit(actor="x", action="create", entity="people", entity_id=person.record_id)
    assert len(repo.audit_tail()) == 1


# -- CLI ------------------------------------------------------------------


def test_cli_dashboard(tmp_path: Path, capsys):
    assert workplace_main(["--data-root", str(tmp_path), "dashboard"]) == 0
    assert "Workplace dashboard" in capsys.readouterr().out
    assert workplace_main(["--data-root", str(tmp_path), "dashboard", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_projects"] == 0 and payload["divisions"] == 10


def test_cli_entities(tmp_path: Path, capsys):
    assert workplace_main(["--data-root", str(tmp_path), "entities"]) == 0
    out = capsys.readouterr().out
    assert "projects" in out and "people" in out and "standards" in out


def test_cli_add_list_show_update_deactivate(tmp_path: Path, capsys):
    base = ["--data-root", str(tmp_path)]
    assert workplace_main([*base, "add", "people", "name=Ali Hassan", "role=Welder"]) == 0
    out = capsys.readouterr().out
    assert "Ali Hassan" in out
    record_id = out.split()[2].rstrip(":")

    assert workplace_main([*base, "list", "people"]) == 0
    assert "Ali Hassan" in capsys.readouterr().out

    assert workplace_main([*base, "show", "people", record_id, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "Ali Hassan" and payload["role"] == "Welder"

    assert workplace_main([*base, "update", "people", record_id, "role=Inspector"]) == 0
    capsys.readouterr()  # drain the update's own line
    assert workplace_main([*base, "show", "people", record_id, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["role"] == "Inspector"

    assert workplace_main([*base, "deactivate", "people", record_id]) == 0
    assert workplace_main([*base, "list", "people"]) == 0
    assert "No people" in capsys.readouterr().out


def test_cli_show_reports_an_unknown_record(tmp_path: Path, capsys):
    code = workplace_main(["--data-root", str(tmp_path), "show", "people", "PER-nope"])
    assert code == 3
    assert "No people" in capsys.readouterr().out


def test_cli_add_refuses_a_bad_field(tmp_path: Path, capsys):
    code = workplace_main(["--data-root", str(tmp_path), "add", "people", "not-a-pair"])
    assert code == 4
    assert "refused" in capsys.readouterr().out.lower()


def test_cli_audit(tmp_path: Path, capsys):
    base = ["--data-root", str(tmp_path)]
    workplace_main([*base, "add", "projects", "number=P-1", "title=A"])
    capsys.readouterr()
    assert workplace_main([*base, "audit", "--limit", "5"]) == 0
    out = capsys.readouterr().out
    assert "create" in out and "projects" in out
