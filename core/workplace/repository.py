"""The Workplace repository: typed CRUD over the database.

This is the only module that turns a caller's intent into SQL. It wraps
:class:`core.workplace.database.WorkplaceDatabase` with named operations so the
service, the window, and the CLI never write a query by hand, and so the
"deactivate, never delete" rule is enforced in one place.

The repository is deliberately small in this phase: it manages the institutional
hierarchy (institutions, organisational units, divisions). Later phases add their
own repositories (projects, personnel, standards, ...) beside it, each over the
same database.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

from core.workplace.database import WorkplaceDatabase, WorkplaceLoadReport
from core.workplace.model import (
    DIVISION_KIND,
    INSTITUTION_KIND,
    UNIT_KIND,
    Division,
    Institution,
    OrganisationalUnit,
    RecordBase,
    make_id,
)

T = TypeVar("T", bound=RecordBase)

#: The table names, named once so a typo is a single fix, not a hunt.
INSTITUTIONS = "institutions"
UNITS = "organisational_units"
DIVISIONS = "divisions"


class WorkplaceRepository:
    """Typed access to the institutional hierarchy."""

    def __init__(self, database: WorkplaceDatabase) -> None:
        self.db = database

    # -- institutions -----------------------------------------------------

    def create_institution(self, **fields: Any) -> Institution:
        """Create an institution; ``key``/``name`` are the required fields."""
        record = Institution(record_id=make_id(INSTITUTION_KIND), **fields)
        self.db.upsert(INSTITUTIONS, record)
        return record

    def get_institution(self, record_id: str) -> Institution | None:
        return self.db.get(INSTITUTIONS, record_id)  # type: ignore[return-value]

    def institution_by_key(self, key: str) -> Institution | None:
        rows = self.db.list_records(INSTITUTIONS, where="key = ?", params=(key,), limit=1)
        return rows[0] if rows else None

    def list_institutions(self, *, include_inactive: bool = False) -> list[Institution]:
        return self.db.list_records(INSTITUTIONS, include_inactive=include_inactive)

    def save_institution(self, record: Institution) -> Institution:
        self.db.upsert(INSTITUTIONS, record)
        return record

    # -- organisational units --------------------------------------------

    def create_unit(self, *, institution_id: str, **fields: Any) -> OrganisationalUnit:
        """Create a unit under an institution (or another unit via ``parent_id``)."""
        record = OrganisationalUnit(
            record_id=make_id(UNIT_KIND), institution_id=institution_id, **fields
        )
        self.db.upsert(UNITS, record)
        return record

    def get_unit(self, record_id: str) -> OrganisationalUnit | None:
        return self.db.get(UNITS, record_id)  # type: ignore[return-value]

    def list_units(
        self, *, institution_id: str | None = None, include_inactive: bool = False
    ) -> list[OrganisationalUnit]:
        if institution_id is None:
            return self.db.list_records(UNITS, include_inactive=include_inactive)
        return self.db.list_records(
            UNITS,
            where="institution_id = ?",
            params=(institution_id,),
            include_inactive=include_inactive,
        )

    def children_of(self, parent_id: str, *, include_inactive: bool = False) -> list[OrganisationalUnit]:
        return self.db.list_records(
            UNITS,
            where="parent_id = ?",
            params=(parent_id,),
            include_inactive=include_inactive,
        )

    def save_unit(self, record: OrganisationalUnit) -> OrganisationalUnit:
        self.db.upsert(UNITS, record)
        return record

    def deactivate_unit(self, record_id: str) -> OrganisationalUnit | None:
        """Deactivate a unit (the soft delete) -- history is never destroyed.

        A unit's children and the divisions/projects that reference it keep their
        rows and their foreign keys; only ``active`` changes, so a reorganisation
        never orphans a project record.
        """
        unit = self.get_unit(record_id)
        if unit is None:
            return None
        updated = unit.deactivate()
        self.db.upsert(UNITS, updated)
        return updated  # type: ignore[return-value]

    # -- divisions --------------------------------------------------------

    def create_division(self, *, unit_id: str = "", **fields: Any) -> Division:
        record = Division(record_id=make_id(DIVISION_KIND), unit_id=unit_id, **fields)
        self.db.upsert(DIVISIONS, record)
        return record

    def get_division(self, record_id: str) -> Division | None:
        return self.db.get(DIVISIONS, record_id)  # type: ignore[return-value]

    def division_by_key(self, key: str) -> Division | None:
        rows = self.db.list_records(DIVISIONS, where="key = ?", params=(key,), limit=1)
        return rows[0] if rows else None

    def list_divisions(
        self, *, unit_id: str | None = None, include_inactive: bool = False
    ) -> list[Division]:
        if unit_id is None:
            return self.db.list_records(DIVISIONS, include_inactive=include_inactive)
        return self.db.list_records(
            DIVISIONS,
            where="unit_id = ?",
            params=(unit_id,),
            include_inactive=include_inactive,
        )

    def save_division(self, record: Division) -> Division:
        self.db.upsert(DIVISIONS, record)
        return record

    def rename_division(self, record_id: str, *, name: str, name_ar: str = "") -> Division | None:
        """Rename a division without touching source code (the brief's requirement)."""
        division = self.get_division(record_id)
        if division is None:
            return None
        changes: dict[str, Any] = {"name": name}
        if name_ar:
            changes["name_ar"] = name_ar
        updated = division.touch(**changes)
        self.db.upsert(DIVISIONS, updated)
        return updated  # type: ignore[return-value]

    def deactivate_division(self, record_id: str) -> Division | None:
        """Deactivate a division (the soft delete) -- historical records survive."""
        division = self.get_division(record_id)
        if division is None:
            return None
        updated = division.deactivate()
        self.db.upsert(DIVISIONS, updated)
        return updated  # type: ignore[return-value]

    def reorder_division(self, record_id: str, *, order: int) -> Division | None:
        division = self.get_division(record_id)
        if division is None:
            return None
        updated = division.touch(order=order)
        self.db.upsert(DIVISIONS, updated)
        return updated  # type: ignore[return-value]

    # -- bulk -------------------------------------------------------------

    def bulk_insert(self, table: str, records: Sequence[RecordBase]) -> int:
        """Insert several records at once (used by the seed loader)."""
        return self.db.upsert_many(table, records)

    def load_report(self) -> WorkplaceLoadReport:
        """A fresh load report the caller can pass to a read for corruption counts."""
        return WorkplaceLoadReport()


__all__ = [
    "DIVISIONS",
    "INSTITUTIONS",
    "UNITS",
    "WorkplaceRepository",
]
