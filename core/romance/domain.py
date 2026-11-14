"""The Studio's domain service: typed, audited access to the companion records.

:class:`~core.romance.repository.RomanceRepository` reads and writes rows; this
module is the *service* over it. It gives every entity a small, typed API
(create / read / list / update / deactivate) that appends an **audit row** on
every write, and it never mutates a record in place -- an edit produces a new
record value (a bumped ``updated_at``) and a new audit row, exactly the
discipline :mod:`core.workplace.domain` uses for revisions.

It is deliberately a thin layer and it is Tk-free, model-free, and network-free,
so the window, the CLI, and the tests all drive the *same* service.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

from core.romance.model import RECORDS, RecordBase, utc_now
from core.romance.repository import RomanceRepository

#: The entity names a caller may pass. They map one-to-one onto the table names in
#: :data:`core.romance.model.RECORDS`.
ENTITY_NAMES: tuple[str, ...] = tuple(RECORDS)


class RomanceDomainError(RuntimeError):
    """A domain operation could not be completed; the message says why."""


def _summary(record: Any) -> str:
    """A short, non-sensitive summary of a record for an audit row."""
    label = getattr(record, "name", "") or getattr(record, "title", "") or ""
    kind = getattr(record, "kind", "") or getattr(record, "role", "") or ""
    return f"{record.record_id} {label} {kind}".strip()


class RomanceDomain:
    """Typed, audited CRUD over the Studio records.

    ``actor`` is the name recorded on every audit row (the safety phase supplies a
    real user; a headless run uses ``"system"``). Nothing here is destructive:
    :meth:`deactivate` sets the soft-delete flag and records it, so history that
    references the record survives.
    """

    def __init__(self, repository: RomanceRepository, *, actor: str = "system") -> None:
        self.repository = repository
        self.actor = actor or "system"

    # -- generic ----------------------------------------------------------

    def _record_type(self, entity: str) -> type[RecordBase]:
        record_type = RECORDS.get(entity)
        if record_type is None:
            raise RomanceDomainError(f"Unknown Romance entity: {entity!r}")
        return record_type

    def create(self, entity: str, **fields: Any) -> Any:
        """Create a record of ``entity`` and record the creation in the audit."""
        try:
            record = self.repository.create_record(entity, **fields)
        except KeyError as error:
            raise RomanceDomainError(f"Unknown Romance entity: {entity!r}") from error
        self._audit("create", entity, record)
        return record

    def get(self, entity: str, record_id: str) -> Any:
        self._record_type(entity)
        return self.repository.get_record(entity, record_id)

    def list_records(
        self,
        entity: str,
        *,
        where: str = "",
        params: Sequence[Any] = (),
        include_inactive: bool = False,
        order_by: str = "created_at, record_id",
        limit: int | None = None,
    ) -> list[Any]:
        self._record_type(entity)
        return self.repository.list_records(
            entity,
            where=where,
            params=params,
            include_inactive=include_inactive,
            order_by=order_by,
            limit=limit,
        )

    def update(self, entity: str, record_id: str, **changes: Any) -> Any:
        """Apply ``changes`` to a record as a *new* value and audit the edit."""
        current = self.repository.get_record(entity, record_id)
        if current is None:
            raise RomanceDomainError(f"No {entity} with id {record_id!r}")
        unknown = set(changes) - {f.name for f in dataclasses.fields(current)}
        if unknown:
            raise RomanceDomainError(f"{entity} has no field(s): {', '.join(sorted(unknown))}")
        updated = dataclasses.replace(current, updated_at=utc_now(), **changes)
        self.repository.save_record(entity, updated)
        changed = ", ".join(sorted(changes)) or "(no field)"
        self._audit(
            "update", entity, updated, detail=changed,
            before=_summary(current), after=_summary(updated),
        )
        return updated

    def deactivate(self, entity: str, record_id: str) -> Any:
        """Soft-delete a record (mark it inactive) and audit the change."""
        updated = self.repository.deactivate_record(entity, record_id)
        if updated is None:
            raise RomanceDomainError(f"No {entity} with id {record_id!r}")
        self._audit("deactivate", entity, updated)
        return updated

    def activate(self, entity: str, record_id: str) -> Any:
        """Restore a soft-deleted record and audit the change."""
        updated = self.repository.activate_record(entity, record_id)
        if updated is None:
            raise RomanceDomainError(f"No {entity} with id {record_id!r}")
        self._audit("activate", entity, updated)
        return updated

    def count(self, entity: str, *, include_inactive: bool = False) -> int:
        """How many records of ``entity`` the table holds."""
        self._record_type(entity)
        return self.repository.count(entity, include_inactive=include_inactive)

    def delete(self, entity: str, record_id: str) -> bool:
        """Hard-delete a record. Used only where a retention policy asks for it."""
        self._record_type(entity)
        removed = self.repository.delete_record(entity, record_id)
        if removed:
            self.repository.audit(
                actor=self.actor, action="delete", entity=entity, entity_id=record_id
            )
        return removed

    # -- the audit trail --------------------------------------------------

    def _audit(
        self,
        action: str,
        entity: str,
        record: Any,
        *,
        detail: str = "",
        before: str = "",
        after: str = "",
    ) -> Any:
        return self.repository.audit(
            actor=self.actor,
            action=action,
            entity=entity,
            entity_id=getattr(record, "record_id", ""),
            detail=detail,
            before=before,
            after=after,
        )

    def record_audit(
        self, action: str, entity: str, *, entity_id: str = "", detail: str = ""
    ) -> Any:
        """Record an action that is not a plain create/update (an approval, ...)."""
        return self.repository.audit(
            actor=self.actor, action=action, entity=entity, entity_id=entity_id, detail=detail
        )

    def audit_tail(self, *, limit: int = 50) -> list[Any]:
        return self.repository.audit_tail(limit=limit)

    def audit_for(self, entity_id: str, *, limit: int = 50) -> list[Any]:
        return self.repository.audit_for(entity_id, limit=limit)


__all__ = [
    "ENTITY_NAMES",
    "RomanceDomain",
    "RomanceDomainError",
]
