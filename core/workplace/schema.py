"""The Workplace SQL schema: the DDL, the numbered migrations, and the column map.

This module is the one place the database's *shape* is written down. It has three
parts:

* :data:`SCHEMA_VERSION` and :data:`MIGRATIONS` -- the numbered steps that bring a
  file up to the current version, each run in its own transaction. A database from
  a *newer* build is refused rather than read with a schema this build does not
  understand, exactly like :mod:`core.memory_db`.
* :data:`TABLES` -- one :class:`TableSpec` per table, mapping each dataclass field
  to a real SQL column (so foreign keys and indexes are genuine, not emulated),
  with a JSON column for a free-form ``metadata`` map.
* The foreign keys, with ``ON DELETE RESTRICT``: because a record is *deactivated*
  rather than deleted, history that references it can never be orphaned.

The tables here are the institutional hierarchy (this phase). Later phases append
their own tables and a new numbered migration; nothing already written changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.workplace.model import (
    Division,
    DivisionDetail,
    Institution,
    OrganisationalUnit,
    TaxonomyNode,
)

#: The schema version this build writes and understands.
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class Column:
    """One SQL column and the record field it carries."""

    name: str
    field: str
    sql_type: str = "TEXT"
    json: bool = False
    boolean: bool = False


@dataclass(frozen=True)
class TableSpec:
    """One table: its record class, its columns, and its foreign keys."""

    name: str
    record: type
    columns: tuple[Column, ...]
    foreign_keys: tuple[tuple[str, str, str], ...] = ()
    indexes: tuple[tuple[str, ...], ...] = ()

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def foreign_key_columns(self) -> frozenset[str]:
        """The columns that reference another table (an empty value means NULL)."""
        return frozenset(column for column, _table, _ref in self.foreign_keys)


#: The base columns every record table carries.
_BASE_COLUMNS = (
    Column("record_id", "record_id"),
    Column("created_at", "created_at"),
    Column("updated_at", "updated_at"),
    Column("active", "active", "INTEGER", boolean=True),
)

_METADATA = Column("metadata", "metadata", json=True)

#: ``order`` is a SQL keyword, so the column is named ``sort_order``.
INSTITUTIONS = TableSpec(
    name="institutions",
    record=Institution,
    columns=(
        *_BASE_COLUMNS,
        Column("key", "key"),
        Column("name", "name"),
        Column("name_ar", "name_ar"),
        Column("kind", "kind"),
        Column("country", "country"),
        Column("website", "website"),
        Column("summary", "summary"),
        Column("official", "official", "INTEGER", boolean=True),
        _METADATA,
    ),
    indexes=(("key",),),
)

ORGANISATIONAL_UNITS = TableSpec(
    name="organisational_units",
    record=OrganisationalUnit,
    columns=(
        *_BASE_COLUMNS,
        Column("institution_id", "institution_id"),
        Column("parent_id", "parent_id"),
        Column("key", "key"),
        Column("name", "name"),
        Column("name_ar", "name_ar"),
        Column("kind", "kind"),
        Column("summary", "summary"),
        Column("official", "official", "INTEGER", boolean=True),
        Column("sort_order", "order", "INTEGER"),
        _METADATA,
    ),
    foreign_keys=(
        ("institution_id", "institutions", "record_id"),
        ("parent_id", "organisational_units", "record_id"),
    ),
    indexes=(("institution_id",), ("parent_id",)),
)

DIVISIONS = TableSpec(
    name="divisions",
    record=Division,
    columns=(
        *_BASE_COLUMNS,
        Column("unit_id", "unit_id"),
        Column("key", "key"),
        Column("name", "name"),
        Column("name_ar", "name_ar"),
        Column("summary", "summary"),
        Column("scope", "scope"),
        Column("proposed", "proposed", "INTEGER", boolean=True),
        Column("sort_order", "order", "INTEGER"),
        _METADATA,
    ),
    foreign_keys=(("unit_id", "organisational_units", "record_id"),),
    indexes=(("unit_id",), ("key",)),
)

DIVISION_DETAILS = TableSpec(
    name="division_details",
    record=DivisionDetail,
    columns=(
        *_BASE_COLUMNS,
        Column("division_id", "division_id"),
        Column("kind", "kind"),
        Column("title", "title"),
        Column("detail", "detail"),
        Column("reference", "reference"),
        Column("sort_order", "order", "INTEGER"),
        _METADATA,
    ),
    foreign_keys=(("division_id", "divisions", "record_id"),),
    indexes=(("division_id",), ("kind",)),
)

TAXONOMY_NODES = TableSpec(
    name="taxonomy_nodes",
    record=TaxonomyNode,
    columns=(
        *_BASE_COLUMNS,
        Column("parent_id", "parent_id"),
        Column("kind", "kind"),
        Column("key", "key"),
        Column("name", "name"),
        Column("name_ar", "name_ar"),
        Column("summary", "summary"),
        Column("sort_order", "order", "INTEGER"),
        Column("proposed", "proposed", "INTEGER", boolean=True),
        _METADATA,
    ),
    foreign_keys=(("parent_id", "taxonomy_nodes", "record_id"),),
    indexes=(("parent_id",), ("key",)),
)

#: Every table this build knows, by name.
TABLES: dict[str, TableSpec] = {
    INSTITUTIONS.name: INSTITUTIONS,
    ORGANISATIONAL_UNITS.name: ORGANISATIONAL_UNITS,
    DIVISIONS.name: DIVISIONS,
    DIVISION_DETAILS.name: DIVISION_DETAILS,
    TAXONOMY_NODES.name: TAXONOMY_NODES,
}

#: The table that records the schema version on disk.
_META_DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _create_table(spec: TableSpec) -> str:
    """Render the ``CREATE TABLE`` statement for one spec (with its foreign keys)."""
    lines = [f"    {column.name} {column.sql_type}" for column in spec.columns]
    primary = "    PRIMARY KEY (record_id)"
    constraints = []
    for column, ref_table, ref_column in spec.foreign_keys:
        constraints.append(
            f"    FOREIGN KEY ({column}) REFERENCES {ref_table}({ref_column}) ON DELETE RESTRICT"
        )
    body = ",\n".join([*lines, primary, *constraints])
    return f"CREATE TABLE IF NOT EXISTS {spec.name} (\n{body}\n);"


def _create_indexes(spec: TableSpec) -> list[str]:
    """Render the ``CREATE INDEX`` statements for one spec."""
    statements = []
    for columns in spec.indexes:
        suffix = "_".join(columns)
        statements.append(
            f"CREATE INDEX IF NOT EXISTS idx_{spec.name}_{suffix} "
            f"ON {spec.name} ({', '.join(columns)});"
        )
    return statements


def _migration_1(connection: object) -> None:
    """Version 1: the institutional hierarchy tables."""
    execute = connection.execute  # type: ignore[attr-defined]
    execute(_META_DDL)
    for spec in (INSTITUTIONS, ORGANISATIONAL_UNITS, DIVISIONS):
        execute(_create_table(spec))
        for statement in _create_indexes(spec):
            execute(statement)


def _migration_2(connection: object) -> None:
    """Version 2: divisional detail and the stored taxonomy (Phase 3)."""
    execute = connection.execute  # type: ignore[attr-defined]
    for spec in (DIVISION_DETAILS, TAXONOMY_NODES):
        execute(_create_table(spec))
        for statement in _create_indexes(spec):
            execute(statement)


#: The numbered migration steps. Each entry is ``(version, callable)``; a callable
#: runs inside its own transaction and only adds what its version introduces.
MIGRATIONS: tuple[tuple[int, object], ...] = (
    (1, _migration_1),
    (2, _migration_2),
)


__all__ = [
    "DIVISIONS",
    "DIVISION_DETAILS",
    "INSTITUTIONS",
    "MIGRATIONS",
    "ORGANISATIONAL_UNITS",
    "SCHEMA_VERSION",
    "TABLES",
    "TAXONOMY_NODES",
    "Column",
    "TableSpec",
]
