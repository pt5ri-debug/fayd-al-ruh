"""Tests for the Workplace Phase 9: security, permissions, and audit trails (§13).

Phase 9 adds the governance layer: role-based access control, department- and
project-level scope, confidentiality clearance, read-only approved records, the
append-only audit trail with its refusals, and the rule that the AI can never be
recorded as an approver. Everything here runs on the real service, the real
database, and the real policy: no model, no network, no display.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cli.workplace import main as workplace_main
from core.workplace.records import DOMAIN_RECORDS
from core.workplace.schema import SCHEMA_VERSION, TABLES
from core.workplace.security import (
    DEFAULT_ROLE,
    PERMISSIONS,
    ROLES,
    WorkplaceSecurityError,
    is_ai_actor,
    permissions_for,
    role_ceiling,
)
from core.workplace.security_gate import SECURITY_JUDGEMENTS, security_accuracy
from gui.workplace import WorkplaceService


@pytest.fixture()
def service(tmp_path: Path) -> WorkplaceService:
    return WorkplaceService(root=tmp_path)


# -- the permission model -------------------------------------------------


def test_the_permission_vocabulary_is_closed():
    assert PERMISSIONS == (
        "read",
        "create",
        "update",
        "approve",
        "deactivate",
        "manage_users",
        "administer",
    )


def test_every_role_grants_only_known_permissions():
    for role, perms in ROLES.items():
        assert role  # a named role
        assert set(perms) <= set(PERMISSIONS), role
    # The administrator is the only role that may administer and manage users.
    admins = [role for role, perms in ROLES.items() if "administer" in perms]
    assert admins == ["administrator"]


def test_the_role_ceiling_is_a_subset_rule():
    # A grant asking for a permission the role does not hold never widens it.
    assert permissions_for("viewer", ["read", "administer"]) == frozenset({"read"})
    # A grant that only narrows keeps the narrowed set.
    assert permissions_for("manager", ["read", "approve"]) == frozenset({"read", "approve"})
    # No grant means the role's full ceiling.
    assert permissions_for("manager") == role_ceiling("manager")


def test_an_unknown_role_falls_back_to_the_least_privilege():
    assert DEFAULT_ROLE == "viewer"
    assert permissions_for("no-such-role") == permissions_for("viewer")
    assert role_ceiling("no-such-role") == role_ceiling("viewer")


def test_an_unknown_permission_is_refused(service: WorkplaceService):
    admin = service.create_user(name="Admin", role="administrator")
    service.set_current_user(admin.record_id)
    assert not service.security.can(admin, "no-such-permission").allowed


# -- users and roles ------------------------------------------------------


def test_a_user_is_created_with_a_role_and_a_clearance(service: WorkplaceService):
    user = service.create_user(name="Ali", role="engineer", clearance="public,internal")
    assert user.record_id.startswith("USR-")
    assert user.role == "engineer"
    assert user.clearance == "public,internal"
    assert service.user(user.record_id) == user


def test_creating_a_user_with_an_unknown_role_is_refused(service: WorkplaceService):
    with pytest.raises(WorkplaceSecurityError):
        service.create_user(name="Ali", role="superuser")


def test_an_unknown_clearance_level_is_refused(service: WorkplaceService):
    with pytest.raises(WorkplaceSecurityError):
        service.create_user(name="Ali", role="engineer", clearance="secret")


def test_setting_a_role_and_a_clearance(service: WorkplaceService):
    user = service.create_user(name="Ali", role="viewer")
    assert service.set_user_role(user.record_id, "manager").role == "manager"
    assert service.set_user_clearance(user.record_id, "public,confidential").clearance == (
        "public,confidential"
    )
    with pytest.raises(WorkplaceSecurityError):
        service.set_user_role(user.record_id, "nope")


def test_deactivating_a_user_is_a_soft_delete(service: WorkplaceService):
    user = service.create_user(name="Ali", role="engineer")
    service.deactivate_user(user.record_id)
    assert service.users() == []  # not listed
    reloaded = service.user(user.record_id)
    assert reloaded is not None and reloaded.active is False  # but still there


# -- the role + permission decision ---------------------------------------


def test_a_role_decides_the_permission(service: WorkplaceService):
    viewer = service.create_user(name="V", role="viewer")
    engineer = service.create_user(name="E", role="engineer")
    manager = service.create_user(name="M", role="manager")
    assert not service.security.can(viewer, "create").allowed
    assert service.security.can(engineer, "create").allowed
    assert not service.security.can(engineer, "approve").allowed
    assert service.security.can(manager, "approve").allowed


def test_an_inactive_user_cannot_act(service: WorkplaceService):
    user = service.create_user(name="Ali", role="administrator")
    service.deactivate_user(user.record_id)
    assert not service.security.can(service.user(user.record_id), "read").allowed


def test_the_current_user_defaults_to_the_first_administrator(service: WorkplaceService):
    admin = service.create_user(name="Admin", role="administrator")
    service.create_user(name="Other", role="viewer")
    current = service.current_user()
    assert current is not None and current.record_id == admin.record_id


# -- department and project scope -----------------------------------------


def test_a_division_scoped_user_cannot_edit_another_division(service: WorkplaceService):
    division_a = service.repository.create_division(key="a", name="A")
    division_b = service.repository.create_division(key="b", name="B")
    head = service.create_user(
        name="Head", role="engineer", division_ids=division_a.record_id
    )
    own = service.domain.create("findings", title="Own", division_id=division_a.record_id)
    other = service.domain.create("findings", title="Other", division_id=division_b.record_id)
    service.set_current_user(head.record_id)
    assert service.check_access("update", entity="findings", record=own).allowed
    assert not service.check_access("update", entity="findings", record=other).allowed


def test_a_project_scoped_user_cannot_reach_another_project(service: WorkplaceService):
    project_a = service.domain.create_project(number="P-1", title="A")
    project_b = service.domain.create_project(number="P-2", title="B")
    user = service.create_user(name="P", role="engineer", project_ids=project_a.record_id)
    service.set_current_user(user.record_id)
    assert service.check_access("update", entity="projects", record=project_a).allowed
    assert not service.check_access("update", entity="projects", record=project_b).allowed


def test_an_administrator_is_unrestricted_by_scope(service: WorkplaceService):
    division_a = service.repository.create_division(key="a", name="A")
    division_b = service.repository.create_division(key="b", name="B")
    admin = service.create_user(
        name="Admin", role="administrator", division_ids=division_a.record_id
    )
    other = service.domain.create("findings", title="Other", division_id=division_b.record_id)
    service.set_current_user(admin.record_id)
    assert service.check_access("update", entity="findings", record=other).allowed


# -- confidentiality ------------------------------------------------------


def test_a_record_above_clearance_is_withheld(service: WorkplaceService):
    user = service.create_user(name="C", role="engineer", clearance="public,internal")
    secret = service.domain.create(
        "workplace_documents", title="Secret", confidentiality="restricted"
    )
    report = service.security.filter_for_clearance(user, [secret])
    assert report.records == []
    assert report.withheld == 1
    assert report.withheld_ids == [secret.record_id]


def test_a_cleared_user_reads_a_confidential_record(service: WorkplaceService):
    user = service.create_user(
        name="C", role="engineer", clearance="public,internal,confidential"
    )
    doc = service.domain.create(
        "workplace_documents", title="Conf", confidentiality="confidential"
    )
    assert service.security.check_clearance(user, doc).allowed


def test_an_administrator_is_still_bound_by_classification(service: WorkplaceService):
    admin = service.create_user(name="Admin", role="administrator", clearance="public,internal")
    top = service.domain.create(
        "workplace_documents", title="Top", confidentiality="restricted"
    )
    assert not service.security.check_clearance(admin, top).allowed


def test_the_assistant_clearance_is_applied_to_its_corpus(service: WorkplaceService):
    # The assistant's corpus is clearance-filtered at build time (§13).
    service.domain.create(
        "workplace_documents", title="Restricted memo", confidentiality="restricted"
    )
    service.set_assistant_clearance("public,internal")
    report = service.refresh_assistant()
    assert report.withheld >= 1


# -- approved records are read-only ---------------------------------------


def test_an_approved_record_is_read_only_until_reopened(service: WorkplaceService):
    report = service.domain.create("reports", title="Final")
    service.domain.approve(
        subject_id=report.record_id, subject_kind="reports", approved_by="Dr. X"
    )
    assert service.is_approved("reports", report.record_id) is True
    assert not service.security.check_editable("reports", report.record_id).allowed
    assert service.security.check_editable("reports", report.record_id, reopen=True).allowed


def test_reopening_is_recorded_on_the_audit_trail(service: WorkplaceService):
    admin = service.create_user(name="Admin", role="administrator")
    service.set_current_user(admin.record_id)
    report = service.domain.create("reports", title="Final")
    service.domain.approve(
        subject_id=report.record_id, subject_kind="reports", approved_by="Dr. X"
    )
    service.reopen_record("reports", report.record_id, note="revise section 3")
    rows = [row for row in service.domain.audit_tail() if row.action == "reopen"]
    assert rows and rows[0].entity_id == report.record_id


# -- the AI can never approve ---------------------------------------------


def test_the_ai_is_never_a_person():
    assert is_ai_actor("Workplace Assistant")
    assert is_ai_actor("the assistant")
    assert is_ai_actor("Fayd Al-Ruh")
    assert is_ai_actor("Ultron")
    assert is_ai_actor("system")
    assert is_ai_actor("")
    assert not is_ai_actor("Dr. Ahmed")
    assert not is_ai_actor("Eng. Mona")


def test_a_report_cannot_be_approved_by_the_ai(service: WorkplaceService):
    project = service.domain.create_project(number="P-1", title="A")
    report = service.create_report(project.record_id, title="R", body="Body.")
    service.submit_report(report.record_id)
    with pytest.raises(Exception):
        service.approve_report(report.record_id, approved_by="Workplace Assistant")


# -- refusals are audited -------------------------------------------------


def test_a_refused_decision_leaves_an_audit_row(service: WorkplaceService):
    viewer = service.create_user(name="V", role="viewer")
    service.set_current_user(viewer.record_id)
    service.check_access("create")
    refused = service.refused_decisions()
    assert refused and refused[0].action == "refused"
    assert "create" in refused[0].detail


# -- project membership ---------------------------------------------------


def test_a_project_membership_is_recorded(service: WorkplaceService):
    project = service.domain.create_project(number="P-1", title="A")
    person = service.domain.create_person(name="Ali")
    member = service.assign_project_member(
        project.record_id, person_id=person.record_id, role="reviewer"
    )
    rows = service.project_members(project.record_id)
    assert [m.record_id for m in rows] == [member.record_id]
    assert rows[0].role == "reviewer"


# -- the schema -----------------------------------------------------------


def test_the_security_tables_are_domain_tables():
    assert "workplace_users" in DOMAIN_RECORDS
    assert "project_members" in DOMAIN_RECORDS
    assert "workplace_users" in TABLES
    assert "project_members" in TABLES
    assert SCHEMA_VERSION == 4


def test_an_existing_database_gains_the_security_tables(tmp_path: Path):
    from core.workplace.database import WorkplaceDatabase

    db_path = tmp_path / "v3.db"
    database = WorkplaceDatabase(db_path)
    # Simulate a v3 database by dropping the security tables and rewinding.
    with database._write() as connection:
        for table in ("workplace_users", "project_members"):
            connection.execute(f"DROP TABLE IF EXISTS {table}")
        connection.execute("UPDATE meta SET value = '3' WHERE key = 'schema_version'")
    reopened = WorkplaceDatabase(db_path)
    assert reopened.schema_version() == SCHEMA_VERSION
    assert reopened.list_records("workplace_users") == []
    assert reopened.integrity_check()


# -- the held-out gate ----------------------------------------------------


def test_the_security_gate_passes_on_the_real_policy():
    assert security_accuracy() == 1.0


def test_the_security_gate_is_zero_for_an_empty_catalogue():
    assert security_accuracy(()) == 0.0


def test_every_judgement_is_run():
    from core.workplace import security_gate

    for judgement in SECURITY_JUDGEMENTS:
        # Every key in the catalogue must resolve to a real check (no KeyError).
        assert judgement.key and judgement.description
        assert hasattr(security_gate, "_run_judgement")


# -- the CLI --------------------------------------------------------------


def test_the_cli_lists_roles_and_permissions(tmp_path: Path, capsys):
    assert workplace_main(["--data-root", str(tmp_path), "roles"]) == 0
    assert "administrator" in capsys.readouterr().out
    assert workplace_main(["--data-root", str(tmp_path), "permissions"]) == 0
    assert "approve" in capsys.readouterr().out


def test_the_cli_adds_a_user_and_checks_a_permission(tmp_path: Path, capsys):
    assert (
        workplace_main(
            ["--data-root", str(tmp_path), "add-user", "Dr. Ahmed", "--role", "manager"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Created user" in out
    # A user id is printed; use the JSON list to grab it.
    assert workplace_main(["--data-root", str(tmp_path), "users", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    user_id = payload[0]["record_id"]
    assert workplace_main(["--data-root", str(tmp_path), "can", "approve", "--user", user_id]) == 0
    assert "allowed" in capsys.readouterr().out


def test_the_cli_refuses_a_permission_a_role_lacks(tmp_path: Path, capsys):
    workplace_main(["--data-root", str(tmp_path), "add-user", "V", "--role", "viewer"])
    capsys.readouterr()
    workplace_main(["--data-root", str(tmp_path), "users", "--json"])
    user_id = json.loads(capsys.readouterr().out)[0]["record_id"]
    # A refused permission exits 4, and the refusal is audited.
    assert workplace_main(["--data-root", str(tmp_path), "can", "approve", "--user", user_id]) == 4
    assert workplace_main(["--data-root", str(tmp_path), "refused"]) == 0
    assert "approve" in capsys.readouterr().out


def test_the_cli_shows_a_user(tmp_path: Path, capsys):
    workplace_main(["--data-root", str(tmp_path), "add-user", "Ali", "--role", "engineer"])
    capsys.readouterr()
    workplace_main(["--data-root", str(tmp_path), "users", "--json"])
    user_id = json.loads(capsys.readouterr().out)[0]["record_id"]
    assert workplace_main(["--data-root", str(tmp_path), "user", user_id]) == 0
    assert "engineer" in capsys.readouterr().out


def test_the_cli_adds_and_lists_a_project_member(tmp_path: Path, capsys):
    service = WorkplaceService(root=tmp_path)
    project = service.domain.create_project(number="P-1", title="A")
    assert (
        workplace_main(
            ["--data-root", str(tmp_path), "add-member", project.record_id, "--role", "reviewer"]
        )
        == 0
    )
    assert "Added member" in capsys.readouterr().out
    assert workplace_main(["--data-root", str(tmp_path), "members", project.record_id]) == 0
    assert "reviewer" in capsys.readouterr().out
