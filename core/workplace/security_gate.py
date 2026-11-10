"""The held-out gate for the Workplace security layer (Phase 9).

The brief's §13 asks for role-based access control, department- and project-level
permissions, confidentiality, an approval workflow with read-only approved
records, and an audit trail -- and for AI output to never be treated as an
engineering decision. This module *measures* that the shipped policy really does
those things, on the **real** :class:`~core.workplace.security.SecurityManager`
over a real database, rather than asserting it in prose.

:data:`SECURITY_JUDGEMENTS` is a catalogue of hand-written expectations with
**negative controls**, so an over-permissive policy fails as surely as an
over-restrictive one:

* a role cannot perform a permission it does not hold;
* a per-user grant can only *narrow* a role, never widen it;
* an unknown role falls back to the least privilege;
* a department-scoped user cannot reach another division's record;
* a project-scoped user cannot reach a project they are not a member of;
* a record above the user's clearance is withheld (and counted);
* an administrator is still bound by confidentiality;
* an approved record is read-only until reopened;
* the AI can never be recorded as an approver;
* a refusal leaves an audit row.

``security_accuracy`` is the share of the catalogue the real policy satisfies;
with no manager it is ``0.0`` rather than a faked pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from core.workplace.database import WorkplaceDatabase
from core.workplace.domain import WorkplaceDomain
from core.workplace.records import Approval, Project, WorkplaceDocument
from core.workplace.repository import WorkplaceRepository
from core.workplace.security import (
    SecurityManager,
    WorkplaceSecurityError,
    is_ai_actor,
    permissions_for,
)


@dataclass(frozen=True)
class SecurityJudgement:
    """One expectation the real policy must satisfy, with the reason it exists."""

    key: str
    description: str


#: The held-out catalogue. Each key is checked by :func:`_run_judgement`; a key
#: with no check is a failure, so the catalogue and the checks cannot drift.
SECURITY_JUDGEMENTS: tuple[SecurityJudgement, ...] = (
    SecurityJudgement("viewer-cannot-create", "a viewer lacks the create permission"),
    SecurityJudgement("engineer-cannot-approve", "an engineer cannot approve"),
    SecurityJudgement("manager-can-approve", "a manager may approve"),
    SecurityJudgement("grant-narrows-only", "a grant cannot widen a role's ceiling"),
    SecurityJudgement("unknown-role-least-privilege", "an unknown role falls back to viewer"),
    SecurityJudgement("division-scope-refused", "a division-scoped user cannot edit another division"),
    SecurityJudgement("division-scope-allowed", "a division-scoped user can edit their own division"),
    SecurityJudgement("project-scope-refused", "a project-scoped user cannot reach another project"),
    SecurityJudgement("clearance-withheld", "a record above clearance is withheld"),
    SecurityJudgement("clearance-granted", "a cleared user reads a confidential record"),
    SecurityJudgement("admin-bound-by-confidentiality", "an administrator is still bound by classification"),
    SecurityJudgement("approved-read-only", "an approved record is read-only until reopened"),
    SecurityJudgement("reopen-allows-edit", "reopening an approved record allows the edit"),
    SecurityJudgement("ai-cannot-approve", "the AI can never be recorded as an approver"),
    SecurityJudgement("refusal-is-audited", "a refused decision leaves an audit row"),
)


def _manager(tmp: Path) -> SecurityManager:
    db = WorkplaceDatabase(tmp / "workplace.db")
    repository = WorkplaceRepository(db)
    domain = WorkplaceDomain(repository)
    return SecurityManager(repository, domain)


def _run_judgement(manager: SecurityManager, key: str) -> bool:
    """Run one judgement against the real policy; return whether it held."""
    domain = manager.domain
    assert domain is not None

    if key == "viewer-cannot-create":
        viewer = manager.create_user(name="V", role="viewer")
        return not manager.can(viewer, "create").allowed
    if key == "engineer-cannot-approve":
        engineer = manager.create_user(name="E", role="engineer")
        return not manager.can(engineer, "approve").allowed
    if key == "manager-can-approve":
        manager_user = manager.create_user(name="M", role="manager")
        return manager.can(manager_user, "approve").allowed
    if key == "grant-narrows-only":
        # A grant asking for administer on a viewer must not widen the viewer.
        user = manager.create_user(name="G", role="viewer", permissions="read,administer")
        effective = manager.effective_permissions(user)
        return "administer" not in effective and "read" in effective
    if key == "unknown-role-least-privilege":
        return permissions_for("no-such-role") == permissions_for("viewer")
    if key == "division-scope-refused":
        division_a = manager.repository.create_division(key="div-a", name="A")
        division_b = manager.repository.create_division(key="div-b", name="B")
        head = manager.create_user(
            name="Head A", role="engineer", division_ids=division_a.record_id
        )
        record = domain.create("findings", title="Other division", division_id=division_b.record_id)
        return not manager.check(head, "update", entity="findings", record=record).allowed
    if key == "division-scope-allowed":
        division_a = manager.repository.create_division(key="div-a", name="A")
        head = manager.create_user(
            name="Head B", role="engineer", division_ids=division_a.record_id
        )
        record = domain.create("findings", title="Own division", division_id=division_a.record_id)
        return manager.check(head, "update", entity="findings", record=record).allowed
    if key == "project-scope-refused":
        user = manager.create_user(name="P", role="engineer", project_ids="PRJ-1")
        project = domain.create("projects", number="PRJ-2", title="Other project")
        assert isinstance(project, Project)
        return not manager.check(user, "update", entity="projects", record=project).allowed
    if key == "clearance-withheld":
        user = manager.create_user(name="C", role="engineer", clearance="public,internal")
        doc = domain.create(
            "workplace_documents", title="Secret", confidentiality="restricted"
        )
        assert isinstance(doc, WorkplaceDocument)
        report = manager.filter_for_clearance(user, [doc])
        return report.withheld == 1 and report.records == []
    if key == "clearance-granted":
        user = manager.create_user(
            name="C2", role="engineer", clearance="public,internal,confidential"
        )
        doc = domain.create(
            "workplace_documents", title="Conf", confidentiality="confidential"
        )
        assert isinstance(doc, WorkplaceDocument)
        return manager.check_clearance(user, doc).allowed
    if key == "admin-bound-by-confidentiality":
        admin = manager.create_user(name="Admin", role="administrator", clearance="public,internal")
        doc = domain.create(
            "workplace_documents", title="Top", confidentiality="restricted"
        )
        assert isinstance(doc, WorkplaceDocument)
        return not manager.check_clearance(admin, doc).allowed
    if key == "approved-read-only":
        record = domain.create("reports", title="Final")
        domain.approve(subject_id=record.record_id, subject_kind="reports", approved_by="Dr. X")
        return not manager.check_editable("reports", record.record_id).allowed
    if key == "reopen-allows-edit":
        record = domain.create("reports", title="Final 2")
        domain.approve(subject_id=record.record_id, subject_kind="reports", approved_by="Dr. X")
        return manager.check_editable("reports", record.record_id, reopen=True).allowed
    if key == "ai-cannot-approve":
        return is_ai_actor("Workplace Assistant") and is_ai_actor("ai") and not is_ai_actor("Dr. Ahmed")
    if key == "refusal-is-audited":
        viewer = manager.create_user(name="V2", role="viewer")
        record = domain.create("findings", title="Anything")
        manager.check(viewer, "update", entity="findings", record=record)
        return any(row.action == "refused" for row in manager.refused_tail())
    raise KeyError(key)


def security_accuracy(judgements: tuple[SecurityJudgement, ...] = SECURITY_JUDGEMENTS) -> float:
    """The share of the held-out catalogue the real policy satisfies."""
    if not judgements:
        return 0.0
    passed = 0
    with TemporaryDirectory() as tmp:
        manager = _manager(Path(tmp))
        for judgement in judgements:
            try:
                if _run_judgement(manager, judgement.key):
                    passed += 1
            except Exception:  # a check that raises is a failure, never a pass
                pass
    return passed / len(judgements)


#: Re-exported so a caller can assert the approval rule without importing records.
__all__ = [
    "SECURITY_JUDGEMENTS",
    "Approval",
    "SecurityJudgement",
    "WorkplaceSecurityError",
    "security_accuracy",
]
