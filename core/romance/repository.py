"""Typed CRUD over the Romance & Companion Studio database.

This is the thin data-access layer the domain service sits on, mirroring
:mod:`core.workplace.repository`. Every method is typed, and the generic
``*_record`` methods work for any table in :data:`core.romance.model.RECORDS`, so
a later phase adds a record type without adding a repository method here.

It is Tk-free, model-free, and network-free: the window, the CLI, and the tests
all drive the same layer through the domain service.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from core.romance.database import RomanceDatabase, RomanceLoadReport
from core.romance.model import RECORDS, RecordBase, make_id, utc_now

#: The id prefix each table mints (so an id reads as its kind).
PREFIXES: dict[str, str] = {
    "companion_characters": "CHR",
    "character_personalities": "PTY",
    "character_profiles": "PRF",
    "character_boundaries": "BND",
    "relationship_states": "RLS",
    "relationship_events": "RLE",
    "companion_memories": "MEM",
    "companion_conversations": "CNV",
    "companion_messages": "MSG",
    "companion_scenarios": "SCN",
    "scenario_participants": "SPT",
    "scenario_sessions": "SES",
    "companion_model_profiles": "MDP",
    "character_voice_profiles": "VOP",
    "companion_settings": "SET",
    "companion_safety_events": "SAF",
    "companion_audit_entries": "AUD",
}


class RomanceRepository:
    """Typed CRUD over one Studio database."""

    def __init__(self, database: RomanceDatabase) -> None:
        self.db = database

    # -- generic ----------------------------------------------------------

    def record_type(self, table: str) -> type[RecordBase]:
        record_type = RECORDS.get(table)
        if record_type is None:
            raise KeyError(f"Unknown Romance table: {table!r}")
        return record_type

    def create_record(self, table: str, *, prefix: str | None = None, **fields: Any) -> Any:
        """Create a record in ``table`` (the class is looked up by name)."""
        record_type = self.record_type(table)
        record = record_type(
            record_id=make_id(table, prefix=prefix or PREFIXES.get(table)), **fields
        )
        self.db.upsert(table, record)
        return record

    def get_record(self, table: str, record_id: str) -> Any:
        self.record_type(table)
        return self.db.get(table, record_id)

    def list_records(
        self,
        table: str,
        *,
        where: str = "",
        params: Sequence[Any] = (),
        include_inactive: bool = False,
        order_by: str = "created_at, record_id",
        limit: int | None = None,
        report: RomanceLoadReport | None = None,
    ) -> list[Any]:
        self.record_type(table)
        return self.db.list_records(
            table,
            where=where,
            params=params,
            include_inactive=include_inactive,
            order_by=order_by,
            limit=limit,
            report=report,
        )

    def save_record(self, table: str, record: RecordBase) -> RecordBase:
        self.record_type(table)
        self.db.upsert(table, record)
        return record

    def deactivate_record(self, table: str, record_id: str) -> RecordBase | None:
        current = self.db.get(table, record_id)
        if current is None:
            return None
        updated = current.touch(active=False)
        self.db.upsert(table, updated)
        return updated

    def activate_record(self, table: str, record_id: str) -> RecordBase | None:
        current = self.db.get(table, record_id)
        if current is None:
            return None
        updated = current.touch(active=True)
        self.db.upsert(table, updated)
        return updated

    def delete_record(self, table: str, record_id: str) -> bool:
        self.record_type(table)
        return self.db.delete(table, record_id)

    def count(self, table: str, *, include_inactive: bool = False) -> int:
        self.record_type(table)
        return self.db.count(table, include_inactive=include_inactive)

    # -- audit ------------------------------------------------------------

    def audit(
        self,
        *,
        actor: str,
        action: str,
        entity: str,
        entity_id: str = "",
        detail: str = "",
        before: str = "",
        after: str = "",
    ) -> Any:
        """Append one audit row (never updated, never deleted)."""
        return self.create_record(
            "companion_audit_entries",
            actor=actor,
            action=action,
            entity=entity,
            entity_id=entity_id,
            detail=detail,
            before=before,
            after=after,
            created_at=utc_now(),
        )

    def audit_tail(self, *, limit: int = 50) -> list[Any]:
        """The most recent audit rows, newest first (insertion order, not clock)."""
        return list(
            reversed(
                self.list_records(
                    "companion_audit_entries", order_by="rowid", limit=limit
                )
            )
        )

    def audit_for(self, entity_id: str, *, limit: int = 50) -> list[Any]:
        rows = self.list_records(
            "companion_audit_entries",
            where="entity_id = ?",
            params=(entity_id,),
            order_by="rowid",
            limit=limit,
        )
        return list(reversed(rows))


__all__ = [
    "PREFIXES",
    "RomanceRepository",
]
