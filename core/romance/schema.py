"""The Romance & Companion Studio SQL schema: the DDL and the numbered migrations.

This module is the one place the Studio database's *shape* is written down. It
follows :mod:`core.workplace.schema` -- the project's precedent for a versioned
domain store -- with three parts:

* :data:`SCHEMA_VERSION` and :data:`MIGRATIONS` -- the numbered steps that bring a
  file up to the current version, each run in its own transaction. A database from
  a *newer* build is refused rather than read with a schema this build does not
  understand.
* :data:`TABLES` -- one :class:`TableSpec` per table. The spec is **derived from
  the record's own dataclass** (:mod:`core.romance.model`), so a field's SQL type
  is read from its annotation rather than duplicated here: ``str`` -> ``TEXT``,
  ``int`` -> ``INTEGER``, ``float`` -> ``REAL``, ``bool`` -> ``INTEGER`` (0/1),
  a ``tuple`` -> a JSON array, a ``dict`` -> a JSON object, and ``X | None`` ->
  a nullable column.
* The foreign keys, with ``ON DELETE RESTRICT`` (a record is *deactivated*, never
  deleted, so history can never be orphaned) and ``ON DELETE CASCADE`` only where
  a child has no meaning without its parent (a message without its conversation).

Later phases append their own tables and a new numbered migration; nothing
already written changes.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, fields
from typing import Any, Union, get_args, get_origin, get_type_hints

from core.romance.model import RECORDS

#: The schema version this build writes and understands.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Column:
    """One SQL column and the record field it carries.

    ``json`` marks a column that stores a JSON-encoded value; ``json_kind`` says
    whether that value is a ``list`` (a tuple field) or a ``dict``, so the reader
    can restore the exact type. ``boolean`` marks a 0/1 integer column.
    """

    name: str
    field: str
    sql_type: str = "TEXT"
    json: bool = False
    boolean: bool = False
    json_kind: str = ""
    nullable: bool = False


@dataclass(frozen=True)
class TableSpec:
    """One table: its record class, its columns, its keys, and its indexes."""

    name: str
    record: type
    columns: tuple[Column, ...]
    foreign_keys: tuple[tuple[str, str, str], ...] = ()
    uniques: tuple[tuple[str, ...], ...] = ()
    indexes: tuple[tuple[str, ...], ...] = ()

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def foreign_key_columns(self) -> frozenset[str]:
        """The columns that reference another table (an empty value means NULL)."""
        return frozenset(column for column, _table, _ref in self.foreign_keys)


#: The base columns every record table carries, in the fixed order the model
#: declares them.
_BASE_FIELDS = ("record_id", "created_at", "updated_at", "active")

#: Field names that are SQL keywords and must be renamed as columns. The
#: ``Column.field`` keeps the Python name, so the record attribute is unaffected.
_RENAMED_COLUMNS: dict[str, str] = {"order": "sort_order"}


def _sql_type_for(hint: Any) -> tuple[str, str, bool, bool]:
    """Classify a field annotation into ``(sql_type, json_kind, json, nullable)``."""
    origin = get_origin(hint)
    args = get_args(hint)
    nullable = False
    if origin is Union or origin is types.UnionType:
        nullable = type(None) in args
        non_none = [arg for arg in args if arg is not type(None)]
        if len(non_none) == 1:
            inner, inner_json_kind, inner_json, _ = _sql_type_for(non_none[0])
            return inner, inner_json_kind, inner_json, nullable
        return "TEXT", "", False, nullable
    if origin in (tuple, list):
        return "TEXT", "list", True, nullable
    if origin in (dict,):
        return "TEXT", "dict", True, nullable
    if hint is bool:
        return "INTEGER", "", False, nullable
    if hint is int:
        return "INTEGER", "", False, nullable
    if hint is float:
        return "REAL", "", False, nullable
    return "TEXT", "", False, nullable


def _build_table(name: str, record: type) -> TableSpec:
    """Derive a :class:`TableSpec` from a record's own dataclass fields."""
    hints = get_type_hints(record)
    columns: list[Column] = []
    for spec in fields(record):
        if spec.name not in _BASE_FIELDS and spec.name not in hints:
            continue
        sql_type, json_kind, is_json, nullable = _sql_type_for(hints[spec.name])
        columns.append(
            Column(
                name=_RENAMED_COLUMNS.get(spec.name, spec.name),
                field=spec.name,
                sql_type=sql_type,
                json=is_json,
                boolean=hints[spec.name] is bool,
                json_kind=json_kind,
                nullable=nullable,
            )
        )
    return TableSpec(name=name, record=record, columns=tuple(columns))


#: Every table this build knows, by name, derived from the record model.
TABLES: dict[str, TableSpec] = {
    name: _build_table(name, record) for name, record in RECORDS.items()
}


#: The foreign keys: ``table -> ((column, ref_table), ...)``. Every reference is
#: ``ON DELETE RESTRICT`` (a record is deactivated, never deleted) except a
#: message, which has no meaning without its conversation.
_FOREIGN_KEYS: dict[str, tuple[tuple[str, str], ...]] = {
    "character_personalities": (("character_id", "companion_characters"),),
    "character_profiles": (("character_id", "companion_characters"),),
    "character_boundaries": (("character_id", "companion_characters"),),
    "relationship_states": (("character_id", "companion_characters"),),
    "relationship_events": (("character_id", "companion_characters"),),
    "companion_memories": (("character_id", "companion_characters"),),
    "companion_conversations": (("character_id", "companion_characters"),),
    "companion_messages": (("conversation_id", "companion_conversations"),),
    "scenario_participants": (
        ("scenario_id", "companion_scenarios"),
        ("character_id", "companion_characters"),
    ),
    "scenario_sessions": (
        ("scenario_id", "companion_scenarios"),
        ("conversation_id", "companion_conversations"),
    ),
    "character_voice_profiles": (("character_id", "companion_characters"),),
}

#: The tables whose child rows are deleted with their parent (a message without
#: its conversation, a participant without its scenario, a session without its
#: conversation). Every other reference is ``ON DELETE RESTRICT``.
_CASCADE_TABLES: frozenset[str] = frozenset(
    {"companion_messages", "scenario_participants", "scenario_sessions"}
)

#: Unique constraints, ``table -> (columns, ...)``.
_UNIQUES: dict[str, tuple[tuple[str, ...], ...]] = {
    "relationship_states": (("character_id", "user_id"),),
    "companion_messages": (("conversation_id", "seq"),),
}

#: Indexed columns, ``table -> (columns, ...)``.
_INDEXES: dict[str, tuple[tuple[str, ...], ...]] = {
    "companion_characters": (("owner",), ("active",), ("archived",), ("name",)),
    "character_personalities": (("character_id",),),
    "character_profiles": (("character_id",), ("kind",)),
    "character_boundaries": (("character_id",),),
    "relationship_states": (("character_id",),),
    "relationship_events": (("character_id",), ("occurred_at",)),
    "companion_memories": (("character_id",), ("kind",), ("created_at",)),
    "companion_conversations": (("character_id",), ("last_active",)),
    "companion_messages": (("conversation_id",),),
    "companion_scenarios": (("category",), ("title",)),
    "scenario_participants": (("scenario_id",), ("character_id",)),
    "scenario_sessions": (("scenario_id",), ("conversation_id",)),
    "companion_model_profiles": (("scope",), ("scope_id",)),
    "character_voice_profiles": (("character_id",),),
    "companion_safety_events": (("character_id",), ("kind",)),
    "companion_audit_entries": (("entity",), ("entity_id",), ("actor",)),
}


def _with_keys(spec: TableSpec) -> TableSpec:
    """Attach the declared foreign keys, unique constraints, and indexes."""
    foreign_keys = tuple(
        (column, ref_table, "record_id")
        for column, ref_table in _FOREIGN_KEYS.get(spec.name, ())
    )
    return TableSpec(
        name=spec.name,
        record=spec.record,
        columns=spec.columns,
        foreign_keys=foreign_keys,
        uniques=_UNIQUES.get(spec.name, ()),
        indexes=_INDEXES.get(spec.name, ()),
    )


TABLES = {name: _with_keys(spec) for name, spec in TABLES.items()}


#: The table that records the schema version on disk.
_META_DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _create_table(spec: TableSpec) -> str:
    """Render the ``CREATE TABLE`` statement for one spec."""
    lines = []
    for column in spec.columns:
        null = "" if (column.name == "record_id" or not column.nullable) else " NULL"
        lines.append(f"    {column.name} {column.sql_type}{null}")
    constraints = ["    PRIMARY KEY (record_id)"]
    for fk_column, ref_table, ref_column in spec.foreign_keys:
        action = "CASCADE" if spec.name in _CASCADE_TABLES else "RESTRICT"
        constraints.append(
            f"    FOREIGN KEY ({fk_column}) REFERENCES {ref_table}({ref_column}) "
            f"ON DELETE {action}"
        )
    for unique_columns in spec.uniques:
        constraints.append(f"    UNIQUE ({', '.join(unique_columns)})")
    body = ",\n".join([*lines, *constraints])
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


def _migration_1(connection: Any) -> None:
    """Version 1: every Studio table."""
    execute = connection.execute
    execute(_META_DDL)
    for spec in TABLES.values():
        execute(_create_table(spec))
        for statement in _create_indexes(spec):
            execute(statement)


#: The numbered migration steps. Each entry is ``(version, callable)``; a callable
#: runs inside its own transaction and only adds what its version introduces.
MIGRATIONS: tuple[tuple[int, Any], ...] = ((1, _migration_1),)


__all__ = [
    "MIGRATIONS",
    "SCHEMA_VERSION",
    "TABLES",
    "Column",
    "TableSpec",
]
