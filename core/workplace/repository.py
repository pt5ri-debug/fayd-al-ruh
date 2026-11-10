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
    DivisionDetail,
    Institution,
    OrganisationalUnit,
    RecordBase,
    TaxonomyNode,
    make_id,
)

T = TypeVar("T", bound=RecordBase)

#: The table names, named once so a typo is a single fix, not a hunt.
INSTITUTIONS = "institutions"
UNITS = "organisational_units"
DIVISIONS = "divisions"
DIVISION_DETAILS = "division_details"
TAXONOMY_NODES = "taxonomy_nodes"


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

    # -- divisional detail (the brief's §3) -------------------------------

    def create_detail(
        self, *, division_id: str, kind: str, title: str = "", **fields: Any
    ) -> DivisionDetail:
        """Add one item to a division's technical profile."""
        record = DivisionDetail(
            record_id=make_id("division_detail"),
            division_id=division_id,
            kind=kind,
            title=title,
            **fields,
        )
        self.db.upsert(DIVISION_DETAILS, record)
        return record

    def get_detail(self, record_id: str) -> DivisionDetail | None:
        return self.db.get(DIVISION_DETAILS, record_id)  # type: ignore[return-value]

    def list_details(
        self,
        *,
        division_id: str | None = None,
        kind: str | None = None,
        include_inactive: bool = False,
    ) -> list[DivisionDetail]:
        """A division's profile items, optionally filtered to one ``kind``."""
        clauses = []
        params: list[Any] = []
        if division_id is not None:
            clauses.append("division_id = ?")
            params.append(division_id)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        rows = self.db.list_records(
            DIVISION_DETAILS,
            where=" AND ".join(clauses),
            params=tuple(params),
            include_inactive=include_inactive,
        )
        rows.sort(key=lambda row: (row.kind, row.order, row.title))
        return rows

    def save_detail(self, record: DivisionDetail) -> DivisionDetail:
        self.db.upsert(DIVISION_DETAILS, record)
        return record

    def deactivate_detail(self, record_id: str) -> DivisionDetail | None:
        detail = self.get_detail(record_id)
        if detail is None:
            return None
        updated = detail.deactivate()
        self.db.upsert(DIVISION_DETAILS, updated)
        return updated  # type: ignore[return-value]

    def detail_kinds(self) -> list[str]:
        """The kinds of detail this Workplace actually holds, in use."""
        seen = {row.kind for row in self.db.list_records(DIVISION_DETAILS, include_inactive=True)}
        return sorted(seen)

    # -- stored taxonomy (the brief's §4) ---------------------------------

    def create_taxonomy_node(
        self, *, kind: str = "area", parent_id: str = "", **fields: Any
    ) -> TaxonomyNode:
        record = TaxonomyNode(
            record_id=make_id("taxonomy"), kind=kind, parent_id=parent_id, **fields
        )
        self.db.upsert(TAXONOMY_NODES, record)
        return record

    def get_taxonomy_node(self, record_id: str) -> TaxonomyNode | None:
        return self.db.get(TAXONOMY_NODES, record_id)  # type: ignore[return-value]

    def taxonomy_node_by_key(self, key: str) -> TaxonomyNode | None:
        rows = self.db.list_records(
            TAXONOMY_NODES, where="key = ?", params=(key,), limit=1
        )
        return rows[0] if rows else None

    def list_taxonomy_nodes(
        self, *, parent_id: str | None = None, include_inactive: bool = False
    ) -> list[TaxonomyNode]:
        if parent_id is None:
            rows = self.db.list_records(TAXONOMY_NODES, include_inactive=include_inactive)
        else:
            rows = self.db.list_records(
                TAXONOMY_NODES,
                where="parent_id = ?",
                params=(parent_id,),
                include_inactive=include_inactive,
            )
        rows.sort(key=lambda row: (row.order, row.name))
        return rows

    def save_taxonomy_node(self, record: TaxonomyNode) -> TaxonomyNode:
        self.db.upsert(TAXONOMY_NODES, record)
        return record

    def deactivate_taxonomy_node(self, record_id: str) -> TaxonomyNode | None:
        node = self.get_taxonomy_node(record_id)
        if node is None:
            return None
        updated = node.deactivate()
        self.db.upsert(TAXONOMY_NODES, updated)
        return updated  # type: ignore[return-value]

    # -- bulk -------------------------------------------------------------

    def load_report(self) -> WorkplaceLoadReport:
        """A fresh load report the caller can pass to a read for corruption counts."""
        return WorkplaceLoadReport()


__all__ = [
    "DIVISIONS",
    "DIVISION_DETAILS",
    "INSTITUTIONS",
    "TAXONOMY_NODES",
    "UNITS",
    "WorkplaceRepository",
]
