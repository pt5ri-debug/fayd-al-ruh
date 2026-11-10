"""The Workplace's security and governance layer (the brief's §13).

The Workplace already *records* who did what (every write appends an audit row)
and already *classifies* a record's confidentiality. This module is the policy
that sits on top of those two facts:

* **Role-based access control.** A person is a *user* with a **role**; a role
  carries a set of **permissions**. A permission is one closed verb
  (:data:`PERMISSIONS`) -- ``read``, ``create``, ``update``, ``approve``,
  ``deactivate``, ``manage_users``, ``administer`` -- and the role's set is a
  **hard ceiling**: a user's effective permissions are
  ``role permissions ∩ per-user grants``, so a hand-edited row can never widen
  them. An unknown role falls back to the least-privileged one, never a made-up
  set of permissions.
* **Department-level scope.** A user may be confined to their own division(s)
  (and, for a project-scoped user, to named projects), so a division head reads
  and edits their division's work and not another's. An administrator is
  unrestricted, but the scope is checked *before* a record is even read.
* **Confidentiality clearance.** A record carries one of
  :data:`~core.workplace.records.CONFIDENTIALITY_LEVELS`; a user is *cleared* for
  a set of them. A read of a record classified above the user's clearance is
  **withheld and counted**, never silently returned -- the same rule the assistant
  already applies to its corpus, applied to the domain records.
* **Read-only approved records.** Once a record has been approved (it has an
  ``approvals`` row with a non-rejected decision), an ordinary edit is refused
  until the record is deliberately **reopened** for revision. The approval is what
  protects an approved engineering record from a silent change.
* **The audit trail.** Every decision here -- granted, refused, withheld -- is
  appended to the same append-only trail the domain service writes, so a refusal
  leaves a trace rather than vanishing.
* **AI is never a decision.** An approval or a signature must name a **person**;
  the assistant's own name is refused. :data:`AI_ACTORS` names the actors that are
  never allowed to approve, and the reporting workflow's ``approved_by`` check is
  routed through :func:`is_ai_actor` here so the rule lives in one place.

Everything in this module is dependency-free, Tk-free, model-free, and
network-free, so the window, the CLI, and the tests all drive the *same* policy.
The records it manages (:class:`~core.workplace.records.WorkplaceUser`,
:class:`~core.workplace.records.ProjectMember`) live in the same database as the
rest of the Workplace, so no second store is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.workplace.domain import WorkplaceDomain
from core.workplace.records import (
    CONFIDENTIALITY_LEVELS,
    Approval,
    WorkplaceUser,
)
from core.workplace.repository import WorkplaceRepository

#: The closed set of permission verbs. A role grants a subset of these and a
#: user's effective permissions are always a subset of the role's.
PERMISSIONS: tuple[str, ...] = (
    "read",
    "create",
    "update",
    "approve",
    "deactivate",
    "manage_users",
    "administer",
)

#: The roles, most-privileged first, each with its hard permission ceiling. A
#: role is a *ceiling*, not a grant: a per-user grant may narrow it and can never
#: widen it. ``viewer`` is the fallback for an unknown role, so an unset or
#: mistyped role reads as the least privilege rather than the most.
ROLES: dict[str, tuple[str, ...]] = {
    "administrator": (
        "read",
        "create",
        "update",
        "approve",
        "deactivate",
        "manage_users",
        "administer",
    ),
    "manager": ("read", "create", "update", "approve", "deactivate"),
    "engineer": ("read", "create", "update"),
    "inspector": ("read", "create", "update", "approve"),
    "reviewer": ("read", "approve"),
    "viewer": ("read",),
}

#: The role an unknown or empty role falls back to (least privilege).
DEFAULT_ROLE = "viewer"

#: The roles whose users are *not* confined to a division or a project set --
#: they may read and edit across the institute. Every other role is scoped.
UNRESTRICTED_ROLES: frozenset[str] = frozenset({"administrator"})

#: The actors that are never allowed to approve or sign. The Workplace assistant
#: and the project's other mouths are named here, and any name that reads as a
#: machine ("ai", "assistant", "system", ...) is refused by :func:`is_ai_actor`,
#: so an AI recommendation can never be recorded as an engineering decision.
AI_ACTORS: frozenset[str] = frozenset(
    {
        "ai",
        "assistant",
        "workplace assistant",
        "epri assistant",
        "fayd al-ruh",
        "fayd-al-ruh",
        "fayd",
        "ultron",
        "vision",
        "banner",
        "natasha",
        "hakeem",
        "cloud",
        "jarvis",
        "system",
        "model",
        "the assistant",
    }
)

#: The tokens that make a free-text actor name read as a machine, so a new mouth
#: is refused without editing this list.
_AI_TOKENS: tuple[str, ...] = ("assistant", "ai agent", "the model", "language model", "llm")


def is_ai_actor(actor: str) -> bool:
    """Whether ``actor`` names the AI (or a machine), so it may never approve.

    The check is deliberately generous: a name that *is* a known mouth, or that
    reads as an assistant/model, is refused. A real person's name ("Dr. Ahmed")
    is not.
    """
    key = " ".join(actor.strip().casefold().split())
    if not key:
        return True
    if key in AI_ACTORS:
        return True
    return any(token in key for token in _AI_TOKENS)


def permissions_for(role: str, grants: Any = ()) -> frozenset[str]:
    """The effective permissions for ``role`` narrowed by per-user ``grants``.

    The role's set is the ceiling: a grant the role does not carry is ignored, so
    a configuration can never widen a user's authority. An unknown role is
    :data:`DEFAULT_ROLE`.
    """
    ceiling = set(ROLES.get(role, ROLES[DEFAULT_ROLE]))
    granted = {str(item) for item in (grants or ()) if str(item) in PERMISSIONS}
    return frozenset(ceiling & granted if granted else ceiling)


def role_ceiling(role: str) -> frozenset[str]:
    """The full permission set a role may ever hold (its ceiling)."""
    return frozenset(ROLES.get(role, ROLES[DEFAULT_ROLE]))


class WorkplaceSecurityError(RuntimeError):
    """A security decision refused an operation; the message says why."""


@dataclass(frozen=True)
class AccessDecision:
    """The outcome of one access check, so a caller can see *why* it was refused."""

    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:  # a decision is truthy only when it is allowed
        return self.allowed


@dataclass(frozen=True)
class ReadReport:
    """What one clearance-filtered read returned, and what it withheld.

    ``records`` are the rows the caller may read; ``withheld`` is how many were
    classified above the caller's clearance; ``withheld_ids`` names them (so an
    administrator can be told *what* was hidden, without the content).
    """

    records: list[Any]
    withheld: int = 0
    withheld_ids: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.withheld_ids is None:
            object.__setattr__(self, "withheld_ids", [])


class SecurityManager:
    """The role/permission/clearance/approval policy over the Workplace records.

    It is a thin layer over :class:`WorkplaceDomain` (which writes and audits) and
    :class:`WorkplaceRepository` (which reads). Nothing here bypasses the audit
    trail: a refusal is recorded as a ``security`` audit row, so the trail shows
    who was told no and why.
    """

    def __init__(
        self,
        repository: WorkplaceRepository,
        domain: WorkplaceDomain | None = None,
    ) -> None:
        self.repository = repository
        self.domain = domain

    # -- users ------------------------------------------------------------

    def create_user(self, *, name: str, **fields: Any) -> WorkplaceUser:
        """Create a Workplace user (a role, a clearance, a scope, and grants)."""
        if self.domain is None:
            raise WorkplaceSecurityError("This security layer is read-only.")
        role = str(fields.get("role", DEFAULT_ROLE))
        if role not in ROLES:
            raise WorkplaceSecurityError(
                f"Unknown role {role!r}; choose one of: {', '.join(ROLES)}."
            )
        clearance = fields.get("clearance")
        if clearance is not None:
            fields["clearance"] = self._normalise_clearance(clearance)
        return self.domain.create("workplace_users", name=name, **fields)

    @staticmethod
    def _normalise_clearance(clearance: Any) -> str:
        """A clearance is stored as a comma-joined, ordered, known-level string."""
        if isinstance(clearance, str):
            items = [item.strip() for item in clearance.split(",") if item.strip()]
        else:
            items = [str(item).strip() for item in (clearance or ()) if str(item).strip()]
        known = [level for level in CONFIDENTIALITY_LEVELS if level in items]
        unknown = [item for item in items if item not in CONFIDENTIALITY_LEVELS]
        if unknown:
            raise WorkplaceSecurityError(
                f"Unknown confidentiality level(s): {', '.join(sorted(unknown))}."
            )
        return ",".join(known)

    def users(self, *, include_inactive: bool = False) -> list[WorkplaceUser]:
        return self.repository.list_domain("workplace_users", include_inactive=include_inactive)

    def user(self, user_id: str) -> WorkplaceUser | None:
        row = self.repository.get_domain("workplace_users", user_id)
        return row if isinstance(row, WorkplaceUser) else None

    def user_by_name(self, name: str) -> WorkplaceUser | None:
        wanted = name.strip().casefold()
        for user in self.users():
            if user.name.strip().casefold() == wanted:
                return user
        return None

    def set_role(self, user_id: str, role: str) -> WorkplaceUser:
        if role not in ROLES:
            raise WorkplaceSecurityError(
                f"Unknown role {role!r}; choose one of: {', '.join(ROLES)}."
            )
        if self.domain is None:
            raise WorkplaceSecurityError("This security layer is read-only.")
        updated = self.domain.update("workplace_users", user_id, role=role)
        assert isinstance(updated, WorkplaceUser)
        return updated

    def set_clearance(self, user_id: str, clearance: Any) -> WorkplaceUser:
        if self.domain is None:
            raise WorkplaceSecurityError("This security layer is read-only.")
        updated = self.domain.update(
            "workplace_users", user_id, clearance=self._normalise_clearance(clearance)
        )
        assert isinstance(updated, WorkplaceUser)
        return updated

    def set_scope(self, user_id: str, *, division_ids: Any = None, project_ids: Any = None) -> WorkplaceUser:
        if self.domain is None:
            raise WorkplaceSecurityError("This security layer is read-only.")
        changes: dict[str, Any] = {}
        if division_ids is not None:
            changes["division_ids"] = self._join_ids(division_ids)
        if project_ids is not None:
            changes["project_ids"] = self._join_ids(project_ids)
        updated = self.domain.update("workplace_users", user_id, **changes)
        assert isinstance(updated, WorkplaceUser)
        return updated

    def grant(self, user_id: str, permissions: Any) -> WorkplaceUser:
        """Set the per-user grant (always narrowed by the role's ceiling)."""
        if self.domain is None:
            raise WorkplaceSecurityError("This security layer is read-only.")
        items = [str(item).strip() for item in self._as_sequence(permissions)]
        unknown = [item for item in items if item not in PERMISSIONS]
        if unknown:
            raise WorkplaceSecurityError(
                f"Unknown permission(s): {', '.join(sorted(unknown))}."
            )
        updated = self.domain.update("workplace_users", user_id, permissions=",".join(items))
        assert isinstance(updated, WorkplaceUser)
        return updated

    @staticmethod
    def _as_sequence(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return list(value)

    @staticmethod
    def _join_ids(value: Any) -> str:
        if isinstance(value, str):
            items = [item.strip() for item in value.split(",") if item.strip()]
        else:
            items = [str(item).strip() for item in (value or ()) if str(item).strip()]
        return ",".join(items)

    # -- the decision -----------------------------------------------------

    def effective_permissions(self, user: WorkplaceUser) -> frozenset[str]:
        return permissions_for(user.role, self._as_sequence(user.permissions))

    def can(self, user: WorkplaceUser | None, permission: str) -> AccessDecision:
        """Whether ``user`` may perform ``permission`` (a pure role/permission check)."""
        if permission not in PERMISSIONS:
            return AccessDecision(False, f"Unknown permission {permission!r}.")
        if user is None or not user.active:
            return AccessDecision(False, "No active user was identified.")
        if permission in self.effective_permissions(user):
            return AccessDecision(True, "")
        return AccessDecision(
            False,
            f"Role {user.role!r} does not grant {permission!r}.",
        )

    def check(
        self,
        user: WorkplaceUser | None,
        permission: str,
        *,
        entity: str = "",
        record: Any = None,
    ) -> AccessDecision:
        """The full check: permission, then scope, then confidentiality.

        A refusal is appended to the audit trail (as a ``security`` row) so the
        trail shows the refusal, not only the successes.
        """
        decision = self.can(user, permission)
        if decision.allowed and entity and record is not None:
            decision = self._check_scope(user, entity, record)
        if decision.allowed and permission == "read" and record is not None:
            decision = self.check_clearance(user, record)
        if not decision.allowed:
            self._audit_refusal(user, permission, entity, record, decision.reason)
        return decision

    def require(
        self,
        user: WorkplaceUser | None,
        permission: str,
        *,
        entity: str = "",
        record: Any = None,
    ) -> None:
        """Raise :class:`WorkplaceSecurityError` when the check refuses."""
        decision = self.check(user, permission, entity=entity, record=record)
        if not decision.allowed:
            raise WorkplaceSecurityError(decision.reason)

    def _check_scope(self, user: WorkplaceUser | None, entity: str, record: Any) -> AccessDecision:
        if user is None or user.role in UNRESTRICTED_ROLES:
            return AccessDecision(True, "")
        division_id = str(getattr(record, "division_id", "") or "")
        project_id = str(getattr(record, "project_id", "") or "")
        if entity == "projects":
            project_id = str(getattr(record, "record_id", "") or "")
        if project_id and user.project_ids:
            if project_id not in self._as_sequence(user.project_ids):
                return AccessDecision(
                    False, f"{user.name} is not assigned to project {project_id}."
                )
            return AccessDecision(True, "")
        if division_id and user.division_ids:
            if division_id not in self._as_sequence(user.division_ids):
                return AccessDecision(
                    False, f"{user.name} is not a member of division {division_id}."
                )
        return AccessDecision(True, "")

    # -- confidentiality --------------------------------------------------

    @staticmethod
    def clearance_levels(user: WorkplaceUser | None) -> tuple[str, ...]:
        if user is None:
            return ()
        return tuple(
            level
            for level in CONFIDENTIALITY_LEVELS
            if level in {item.strip() for item in (user.clearance or "").split(",")}
        )

    def check_clearance(self, user: WorkplaceUser | None, record: Any) -> AccessDecision:
        """Whether ``user`` is cleared to read ``record``.

        A record with no classification is readable by any active user. A
        classified record is readable only when the user's clearance names the
        level; an administrator is still bound by it (classification protects the
        *content*, not the role), so a ``restricted`` document is withheld even
        from an administrator unless it was granted.
        """
        classification = str(getattr(record, "confidentiality", "") or "")
        if not classification or classification == "public":
            return AccessDecision(True, "")
        if user is None:
            return AccessDecision(False, "No user was identified.")
        if classification in self.clearance_levels(user):
            return AccessDecision(True, "")
        return AccessDecision(
            False,
            f"Record is classified {classification!r}; {user.name} is not cleared for it.",
        )

    def filter_for_clearance(self, user: WorkplaceUser | None, records: list[Any]) -> ReadReport:
        """Split ``records`` into what ``user`` may read and what is withheld."""
        allowed: list[Any] = []
        withheld_ids: list[str] = []
        for record in records:
            if self.check_clearance(user, record).allowed:
                allowed.append(record)
            else:
                withheld_ids.append(str(getattr(record, "record_id", "")))
        return ReadReport(records=allowed, withheld=len(withheld_ids), withheld_ids=withheld_ids)

    # -- read-only approved records ---------------------------------------

    def is_approved(self, entity: str, record_id: str) -> bool:
        """Whether ``record_id`` carries a live (non-rejected) approval."""
        for approval in self.approvals_for(record_id):
            if approval.decision in ("approved", "for_review"):
                return True
        return False

    def approvals_for(self, record_id: str) -> list[Approval]:
        rows = self.repository.list_domain(
            "approvals", where="subject_id = ?", params=(record_id,)
        )
        return [row for row in rows if isinstance(row, Approval)]

    def check_editable(self, entity: str, record_id: str, *, reopen: bool = False) -> AccessDecision:
        """Whether an approved record may be edited (refused unless reopened).

        An ordinary edit of an approved record is refused; a caller that means to
        revise it passes ``reopen=True``, which is the deliberate act the approval
        protects against being silent. The reopen is recorded on the audit trail
        by :meth:`reopen`.
        """
        if reopen or not self.is_approved(entity, record_id):
            return AccessDecision(True, "")
        return AccessDecision(
            False,
            f"{record_id} is approved and read-only; reopen it to revise it.",
        )

    def reopen(self, entity: str, record_id: str, *, actor: str, note: str = "") -> None:
        """Record a deliberate reopen of an approved record on the audit trail."""
        if self.domain is None:
            raise WorkplaceSecurityError("This security layer is read-only.")
        self.domain.record_audit(
            "reopen", entity, entity_id=record_id, detail=note or "approved record reopened"
        )

    # -- audit ------------------------------------------------------------

    def _audit_refusal(
        self,
        user: WorkplaceUser | None,
        permission: str,
        entity: str,
        record: Any,
        reason: str,
    ) -> None:
        if self.domain is None:
            return
        actor = user.name if user is not None else "unknown"
        try:
            self.repository.audit(
                actor=actor,
                action="refused",
                entity=entity or "security",
                entity_id=str(getattr(record, "record_id", "") or ""),
                detail=f"{permission}: {reason}",
            )
        except Exception:  # a refusal must never crash on an audit-write failure
            pass

    def refused_tail(self, *, limit: int = 50) -> list[Any]:
        """The most recent refused decisions (the audit rows with action ``refused``)."""
        rows = [
            row
            for row in self.repository.audit_tail(limit=limit * 4)
            if getattr(row, "action", "") == "refused"
        ]
        return rows[:limit]


__all__ = [
    "AI_ACTORS",
    "DEFAULT_ROLE",
    "PERMISSIONS",
    "ROLES",
    "UNRESTRICTED_ROLES",
    "AccessDecision",
    "ReadReport",
    "SecurityManager",
    "WorkplaceSecurityError",
    "is_ai_actor",
    "permissions_for",
    "role_ceiling",
]
