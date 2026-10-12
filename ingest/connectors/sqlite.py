"""Read a SQLite database, read-only, as a streaming source of records.

This is the *standard-library* SQLite connector: it uses :mod:`sqlite3` only, so
the base install stays dependency-free. It supersedes the SQLite branch of
:mod:`ingest.connectors.sql`, which remains for the optional DB-API drivers
(PostgreSQL, MySQL). ``sqlite://`` sources resolve here.

The posture matches the rest of the project:

* **Read-only, and never creating the file.** The connection is opened through a
  ``file:...?mode=ro`` URI, so SQLite cannot create a missing database, and
  ``PRAGMA query_only=ON`` makes any write fail even if one were attempted. A
  missing path is a clear error, not an empty database.
* **Schema from SQLite's own metadata.** Tables, columns, declared types,
  primary keys (including composite), foreign keys (including composite), and
  WITHOUT ROWID tables come from ``PRAGMA table_list`` / ``PRAGMA table_xinfo``
  / ``PRAGMA foreign_key_list``, with a ``sqlite_master`` fallback.
* **Bounded streaming.** Rows are read with ``fetchmany(page)`` and grouped into
  a record every ``row_group`` rows; neither the batch size nor the group size
  is ever the whole table, so a large table does not land in RAM.
* **Internally generated SQL only.** The statement is a ``SELECT`` the connector
  builds from validated, quoted identifiers (identifiers cannot be bound as
  query parameters, so they are quoted, never interpolated from a value).
  Natural-language, operator-supplied, or otherwise external SQL is never
  executed here.
* **Honest about identity.** A table with a primary key yields a stable key from
  its primary-key values; a table *without* one is read explicitly but no stable
  row identity is invented (offsets are positional, not identities).
* **No incremental synchronisation.** The connector reports no fingerprint, so a
  re-run reads the source again; duplicate *chunks* are still dropped by the
  pipeline's content-addressed ledger. A reliable change-detection strategy for
  a table is not assumed.

Value rendering (documented, stable): each cell is ``column: value``. ``None``
is the token ``NULL``; a text value is written verbatim except that an empty
string is ``''`` (so NULL and empty text never collapse); a BLOB is ``X'<hex>'``;
integers and reals use their canonical decimal text; bytes are never decoded as
text. Date/time values are kept exactly as stored (TEXT/INTEGER/REAL) and are
never converted.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ingest.errors import ConnectorError
from ingest.hashing import source_key
from ingest.records import ConnectorInfo, IngestItem, SourceRecord, SourceSpec

#: Rows fetched per round-trip (``fetchmany``); bounds memory regardless of size.
DEFAULT_PAGE = 500
#: Rows grouped into one :class:`SourceRecord` so a table becomes useful passages.
DEFAULT_ROW_GROUP = 40
#: Hard ceilings so a bad option cannot defeat the bounded streaming.
MAX_PAGE = 100_000
MAX_ROW_GROUP = 100_000
#: Seconds to wait for a lock before giving up; 0 fails immediately.
DEFAULT_TIMEOUT = 5.0

SQLITE_SCHEMES = frozenset({"sqlite", "sqlite3"})


class SqliteConnectorError(ConnectorError):
    """Base class for the SQLite connector's failures (a :class:`ConnectorError`)."""


class SqliteOptionError(SqliteConnectorError):
    """An option (batch size, table/column rule) is invalid."""


class SqlitePathError(SqliteConnectorError):
    """The path is missing, a directory, or escapes the allowed root."""


class SqliteAccessError(SqliteConnectorError):
    """The file is inaccessible: permission denied, or the database is locked."""


class SqliteSchemaError(SqliteConnectorError):
    """The file is not a database, or its schema cannot be understood."""


class SqliteValueError(SqliteConnectorError):
    """A stored value is malformed (e.g. text that is not valid UTF-8)."""


class SqliteReadError(SqliteConnectorError):
    """A read failed part-way (interrupted or otherwise aborted)."""


# ---------------------------------------------------------------------------
# schema model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnInfo:
    """One column, as SQLite's metadata declares it (never a guessed type)."""

    name: str
    declared_type: str = ""
    nullable: bool = True
    primary_key: int = 0
    """1-based position within the primary key; 0 when the column is not part of it."""
    default: str | None = None
    hidden: int = 0
    """``table_xinfo`` hidden flag: 0 normal, 1 generated, 2/3 virtual-table hidden."""


@dataclass(frozen=True)
class ForeignKey:
    """One foreign key, grouping a composite key's columns back together."""

    columns: tuple[str, ...]
    referenced_table: str
    referenced_columns: tuple[str, ...]
    on_update: str = "NO ACTION"
    on_delete: str = "NO ACTION"
    match: str = "NONE"
    id: int = 0

    def as_primitive(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "referenced_table": self.referenced_table,
            "referenced_columns": list(self.referenced_columns),
            "on_update": self.on_update,
            "on_delete": self.on_delete,
            "match": self.match,
            "id": self.id,
        }


@dataclass(frozen=True)
class TableSchema:
    """A table's structure: columns, primary key, foreign keys, and rowid status."""

    name: str
    columns: tuple[ColumnInfo, ...]
    primary_key: tuple[str, ...] = ()
    foreign_keys: tuple[ForeignKey, ...] = ()
    without_rowid: bool = False

    @property
    def has_primary_key(self) -> bool:
        return bool(self.primary_key)

    @property
    def composite_key(self) -> bool:
        return len(self.primary_key) > 1


@dataclass(frozen=True)
class _RowPolicy:
    """Which columns and tables to read, honouring exclusion before any read."""

    include_tables: tuple[str, ...] = ()
    exclude_tables: tuple[str, ...] = ()
    include_columns: tuple[str, ...] = ()
    exclude_columns: tuple[str, ...] = ()

    def keeps_table(self, name: str) -> bool:
        key = name.casefold()
        if key in {t.casefold() for t in self.exclude_tables}:
            return False
        if self.include_tables:
            return key in {t.casefold() for t in self.include_tables}
        return True

    def included_columns(self, columns: Iterable[ColumnInfo]) -> tuple[ColumnInfo, ...]:
        """Columns to read: exclusions win, then the include list, then hidden ones."""
        include = {c.casefold() for c in self.include_columns}
        exclude = {c.casefold() for c in self.exclude_columns}
        kept = []
        for column in columns:
            key = column.name.casefold()
            if key in exclude:
                continue
            if include and key not in include:
                continue
            if column.hidden:
                continue
            kept.append(column)
        return tuple(kept)


@dataclass(frozen=True)
class SqliteOptions:
    """Validated connector options, resolved from a :class:`SourceSpec`."""

    page: int = DEFAULT_PAGE
    row_group: int = DEFAULT_ROW_GROUP
    timeout: float = DEFAULT_TIMEOUT
    root: str = ""
    policy: _RowPolicy = _RowPolicy()

    @classmethod
    def from_spec(cls, spec: SourceSpec) -> "SqliteOptions":
        return cls(
            page=_as_int(spec.option("page", DEFAULT_PAGE), "page", 1, MAX_PAGE),
            row_group=_as_int(
                spec.option("row_group", DEFAULT_ROW_GROUP), "row_group", 1, MAX_ROW_GROUP
            ),
            timeout=_as_float(spec.option("timeout", DEFAULT_TIMEOUT), "timeout", 0.0),
            root=str(spec.option("root", "") or ""),
            policy=_policy_from_spec(spec),
        )

    def to_metadata(self) -> dict[str, Any]:
        """The serialisable half of the options, carried on an :class:`IngestItem`."""
        return {
            "page": self.page,
            "row_group": self.row_group,
            "timeout": self.timeout,
            "root": self.root,
            "include_tables": list(self.policy.include_tables),
            "exclude_tables": list(self.policy.exclude_tables),
            "include_columns": list(self.policy.include_columns),
            "exclude_columns": list(self.policy.exclude_columns),
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "SqliteOptions":
        """Rebuild the options (including the column policy) from item metadata.

        The exclusion policy must survive from ``plan`` to ``stream_item``, so a
        sensitive column is never selected even when only the item is available.
        """
        return cls(
            page=int(metadata.get("page", DEFAULT_PAGE)),
            row_group=int(metadata.get("row_group", DEFAULT_ROW_GROUP)),
            timeout=float(metadata.get("timeout", DEFAULT_TIMEOUT)),
            root=str(metadata.get("root", "")),
            policy=_RowPolicy(
                include_tables=_as_names(metadata.get("include_tables")),
                exclude_tables=_as_names(metadata.get("exclude_tables")),
                include_columns=_as_names(metadata.get("include_columns")),
                exclude_columns=_as_names(metadata.get("exclude_columns")),
            ),
        )


# ---------------------------------------------------------------------------
# connector
# ---------------------------------------------------------------------------


class SqliteConnector:
    """A read-only, streaming connector for SQLite databases (standard library)."""

    info = ConnectorInfo(
        name="sqlite",
        description="Read-only, streaming SQLite databases (standard library).",
        schemes=SQLITE_SCHEMES,
        suffixes=frozenset({".db", ".sqlite", ".sqlite3"}),
    )

    # -- planning ---------------------------------------------------------

    def plan(self, spec: SourceSpec) -> Iterator[IngestItem]:
        options = SqliteOptions.from_spec(spec)
        path = _resolve_path(spec.uri, options.root)
        with self._connect(path, options) as connection:
            for schema in self._inspect(connection, options):
                kept = options.policy.included_columns(schema.columns)
                if not kept:
                    # Every column was excluded; read nothing (never touch them).
                    continue
                yield IngestItem(
                    source_key=source_key(str(path), extra=schema.name),
                    uri=f"{_as_uri(path)}#{schema.name}",
                    title=schema.name,
                    fingerprint="",
                    metadata={
                        **options.to_metadata(),
                        "kind": "sqlite",
                        "path": str(path),
                        "table": schema.name,
                        "columns": [column.name for column in kept],
                        "primary_key": list(schema.primary_key),
                        "composite_key": schema.composite_key,
                        "foreign_keys": [fk.as_primitive() for fk in schema.foreign_keys],
                        "without_rowid": schema.without_rowid,
                    },
                )

    def _inspect(self, connection: sqlite3.Connection, options: SqliteOptions) -> list[TableSchema]:
        """Every table the policy keeps, in name order (schema read cheaply)."""
        schemas = []
        for name, without_rowid in _list_tables(connection):
            if not options.policy.keeps_table(name):
                continue
            schemas.append(_inspect_table(connection, name, without_rowid=without_rowid))
        return schemas

    # -- reading ----------------------------------------------------------

    def stream_item(self, item: IngestItem) -> Iterator[SourceRecord]:
        metadata = item.metadata
        path = Path(str(metadata["path"]))
        table = str(metadata["table"])
        options = SqliteOptions.from_metadata(metadata)
        with self._connect(path, options) as connection:
            schema = _inspect_table(
                connection, table, without_rowid=_without_rowid(connection, table)
            )
            yield from self._stream_table(connection, schema, item, options)

    def stream(self, spec: SourceSpec) -> Iterator[SourceRecord]:
        for item in self.plan(spec):
            yield from self.stream_item(item)

    def inspect(self, spec: SourceSpec) -> tuple[TableSchema, ...]:
        """The kept tables' schemas, without reading any row (for callers/tests)."""
        options = SqliteOptions.from_spec(spec)
        path = _resolve_path(spec.uri, options.root)
        with self._connect(path, options) as connection:
            return tuple(self._inspect(connection, options))

    # -- internals --------------------------------------------------------

    def _stream_table(
        self,
        connection: sqlite3.Connection,
        schema: TableSchema,
        item: IngestItem,
        options: SqliteOptions,
    ) -> Iterator[SourceRecord]:
        kept = options.policy.included_columns(schema.columns)
        if not kept:
            return
        columns = [column.name for column in kept]
        kept_names = {name.casefold() for name in columns}
        pk_columns = tuple(
            name for name in schema.primary_key if name.casefold() in kept_names
        )
        # A stable key needs the *whole* primary key; a partial one is not stable.
        has_stable_key = schema.has_primary_key and len(pk_columns) == len(schema.primary_key)
        statement = _select_statement(schema, columns)
        cursor = connection.cursor()
        try:
            cursor.execute(statement)
            index_of = {column.name.casefold(): i for i, column in enumerate(kept)}
            described = [description[0] for description in cursor.description or []]
            if len(described) != len(columns):
                raise SqliteSchemaError(
                    f"Unexpected column count reading {schema.name!r}."
                )
            buffer: list[str] = []
            first_row = 0
            position = 0
            while True:
                try:
                    rows = cursor.fetchmany(options.page)
                except sqlite3.Error as error:
                    raise _translate(error, schema.name) from error
                if not rows:
                    break
                for row in rows:
                    position += 1
                    try:
                        buffer.append(
                            _render_row(
                                kept,
                                row,
                                index_of,
                                pk_columns=pk_columns,
                                stable=has_stable_key,
                            )
                        )
                    except SqliteValueError:
                        raise
                    except (UnicodeDecodeError, ValueError) as error:
                        raise SqliteValueError(
                            f"Malformed value in table {schema.name!r}: {error}"
                        ) from error
                    if len(buffer) >= options.row_group:
                        end = position
                        yield _record(
                            item=item,
                            schema=schema,
                            keep=kept,
                            lines=buffer,
                            start=first_row + 1,
                            end=end,
                            pk_columns=pk_columns,
                            stable=has_stable_key,
                        )
                        first_row = end
                        buffer = []
            if buffer:
                yield _record(
                    item=item,
                    schema=schema,
                    keep=kept,
                    lines=buffer,
                    start=first_row + 1,
                    end=position,
                    pk_columns=pk_columns,
                    stable=has_stable_key,
                )
        except SqliteConnectorError:
            raise
        except sqlite3.Error as error:
            raise _translate(error, schema.name) from error
        finally:
            cursor.close()

    @contextmanager
    def _connect(self, path: Path, options: SqliteOptions) -> Iterator[sqlite3.Connection]:
        """Open ``path`` read-only; never creates it, never writes to it."""
        connection = _open_read_only(path, options.timeout)
        try:
            yield connection
        finally:
            connection.close()


# ---------------------------------------------------------------------------
# path handling
# ---------------------------------------------------------------------------


def _resolve_path(uri: str, root: str) -> Path:
    """Turn a ``sqlite://`` URI (or a bare path) into a validated, existing file."""
    raw = uri
    if "://" in uri:
        _scheme, raw = uri.split("://", 1)
    candidate = Path(raw)
    if not candidate.is_absolute() and root:
        candidate = Path(root) / candidate
    candidate = candidate.expanduser()
    if not candidate.exists():
        raise SqlitePathError(f"No such SQLite database: {candidate}")
    if not candidate.is_file():
        raise SqlitePathError(f"Not a database file: {candidate}")
    if root:
        base = Path(root).expanduser().resolve()
        target = candidate.resolve()
        if target != base and not target.is_relative_to(base):
            raise SqlitePathError(f"Database {target} is outside the allowed root {base}.")
        candidate = target
    else:
        candidate = candidate.resolve()
    if not os.access(candidate, os.R_OK):
        raise SqliteAccessError(f"Cannot read the database file: {candidate}")
    return candidate


def _as_uri(path: Path) -> str:
    return f"sqlite://{path}"


def _open_read_only(path: Path, timeout: float) -> sqlite3.Connection:
    """Open ``path`` with a read-only URI; refuse to create a missing file."""
    if not path.exists():
        raise SqlitePathError(f"No such SQLite database: {path}")
    uri = f"{path.as_uri()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=timeout)
    except sqlite3.OperationalError as error:
        raise _translate(error, str(path)) from error
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.text_factory = _strict_text
    except sqlite3.Error as error:
        connection.close()
        raise _translate(error, str(path)) from error
    return connection


def _strict_text(raw: bytes) -> str:
    """Decode TEXT strictly, so malformed bytes raise instead of mojibake."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SqliteValueError(f"Malformed UTF-8 text value: {error}") from error


# ---------------------------------------------------------------------------
# schema inspection
# ---------------------------------------------------------------------------


def _list_tables(connection: sqlite3.Connection) -> list[tuple[str, bool]]:
    """``(name, without_rowid)`` for every real table, name-ordered."""
    try:
        rows = connection.execute("PRAGMA table_list").fetchall()
        table = [
            (str(name), bool(wr))
            for schema, name, kind, _ncol, wr, _strict in rows
            if schema == "main" and kind == "table" and not str(name).startswith("sqlite_")
        ]
        return sorted(table, key=lambda item: item[0])
    except sqlite3.Error as error:
        raise _translate(error, "schema") from error


def _without_rowid(connection: sqlite3.Connection, table: str) -> bool:
    try:
        rows = connection.execute("PRAGMA table_list").fetchall()
        for schema, name, _kind, _ncol, wr, _strict in rows:
            if schema == "main" and str(name) == table:
                return bool(wr)
    except sqlite3.Error:
        pass
    return False


def _inspect_table(connection: sqlite3.Connection, table: str, *, without_rowid: bool) -> TableSchema:
    columns = _columns(connection, table)
    primary_key = tuple(
        column.name
        for column in sorted(
            (c for c in columns if c.primary_key > 0), key=lambda c: c.primary_key
        )
    )
    return TableSchema(
        name=table,
        columns=columns,
        primary_key=primary_key,
        foreign_keys=_foreign_keys(connection, table),
        without_rowid=without_rowid,
    )


def _columns(connection: sqlite3.Connection, table: str) -> tuple[ColumnInfo, ...]:
    try:
        rows = connection.execute(f"PRAGMA table_xinfo({_quote(table)})").fetchall()
    except sqlite3.Error:
        try:
            rows = connection.execute(f"PRAGMA table_info({_quote(table)})").fetchall()
        except sqlite3.Error as error:
            raise _translate(error, table) from error
    columns = []
    for row in rows:
        hidden = int(row[6]) if len(row) > 6 else 0
        columns.append(
            ColumnInfo(
                name=str(row[1]),
                declared_type=str(row[2] or ""),
                nullable=not bool(row[3]),
                primary_key=int(row[5] or 0),
                default=None if row[4] is None else str(row[4]),
                hidden=hidden,
            )
        )
    return tuple(columns)


def _foreign_keys(connection: sqlite3.Connection, table: str) -> tuple[ForeignKey, ...]:
    try:
        rows = connection.execute(f"PRAGMA foreign_key_list({_quote(table)})").fetchall()
    except sqlite3.Error as error:
        raise _translate(error, table) from error
    grouped: dict[int, list[tuple]] = {}
    for row in rows:
        grouped.setdefault(int(row[0]), []).append(row)
    foreign_keys = []
    for key_id in sorted(grouped):
        group = sorted(grouped[key_id], key=lambda r: int(r[1]))
        foreign_keys.append(
            ForeignKey(
                columns=tuple(str(r[3]) for r in group),
                referenced_table=str(group[0][2]),
                referenced_columns=tuple(str(r[4]) for r in group),
                on_update=str(group[0][5]),
                on_delete=str(group[0][6]),
                match=str(group[0][7]),
                id=key_id,
            )
        )
    return tuple(foreign_keys)


# ---------------------------------------------------------------------------
# SQL construction (validated identifiers, no bound values, no external SQL)
# ---------------------------------------------------------------------------


def _quote(identifier: str) -> str:
    """A double-quoted SQL identifier; embedded quotes are doubled, not escaped."""
    if not identifier or "\x00" in identifier:
        raise SqliteOptionError("Invalid SQL identifier.")
    return '"' + identifier.replace('"', '""') + '"'


def _select_statement(schema: TableSchema, columns: list[str]) -> str:
    """An internally generated read-only SELECT over the kept columns, in order."""
    projection = ", ".join(_quote(column) for column in columns)
    order = _order_clause(schema)
    return f"SELECT {projection} FROM {_quote(schema.name)}{order}"


def _order_clause(schema: TableSchema) -> str:
    """Deterministic ordering when a rowid or primary key exists; else none."""
    kept = {column.name.casefold() for column in schema.columns}
    if not schema.without_rowid and "rowid" not in kept:
        return " ORDER BY rowid"
    if schema.primary_key:
        return " ORDER BY " + ", ".join(_quote(name) for name in schema.primary_key)
    return ""


def _render_row(
    kept: tuple[ColumnInfo, ...],
    row: sqlite3.Row | tuple,
    index_of: Mapping[str, int],
    *,
    pk_columns: tuple[str, ...],
    stable: bool,
) -> str:
    parts = [f"{column.name}: {_render_cell(row[index_of[column.name.casefold()]])}" for column in kept]
    line = "; ".join(parts)
    if stable and pk_columns:
        key = "; ".join(
            f"{name}={_render_cell(row[index_of[name.casefold()]])}" for name in pk_columns
        )
        line = f"[{key}] {line}"
    return line


def _render_cell(value: Any) -> str:
    """A documented, lossless-for-meaning rendering of one SQLite value."""
    if value is None:
        return "NULL"
    if isinstance(value, bytes):
        return "X'" + value.hex() + "'"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    text = str(value)
    return "''" if text == "" else text


def _record(
    *,
    item: IngestItem,
    schema: TableSchema,
    keep: tuple[ColumnInfo, ...],
    lines: list[str],
    start: int,
    end: int,
    pk_columns: tuple[str, ...],
    stable: bool,
) -> SourceRecord:
    return SourceRecord(
        title=f"{schema.name} — rows {start}-{end}",
        text="\n".join(lines),
        source="sqlite",
        source_key=item.source_key,
        metadata={
            "kind": "sqlite",
            "table": schema.name,
            "path": str(item.metadata.get("path", "")),
            "columns": [column.name for column in keep],
            "primary_key": list(schema.primary_key),
            "pk_columns": list(pk_columns),
            "composite_key": schema.composite_key,
            "stable_identity": stable,
            "without_rowid": schema.without_rowid,
            "row_start": start,
            "row_end": end,
            "foreign_keys": [fk.as_primitive() for fk in schema.foreign_keys],
        },
    )


# ---------------------------------------------------------------------------
# option parsing and error translation
# ---------------------------------------------------------------------------


def _policy_from_spec(spec: SourceSpec) -> _RowPolicy:
    """The row policy from a spec's ``include_*/exclude_*`` options."""
    return _RowPolicy(
        include_tables=_as_names(spec.option("include_tables")),
        exclude_tables=_as_names(spec.option("exclude_tables")),
        include_columns=_as_names(spec.option("include_columns")),
        exclude_columns=_as_names(spec.option("exclude_columns")),
    )


def _as_names(value: Any) -> tuple[str, ...]:
    """A name list from a string (comma/space separated) or an iterable."""
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part for part in _split(value) if part)
    if isinstance(value, Iterable):
        names: list[str] = []
        for item in value:
            names.extend(part for part in _split(str(item)) if part)
        return tuple(names)
    return (str(value),)


def _split(text: str) -> list[str]:
    return [part.strip() for chunk in text.split(",") for part in chunk.split() if part.strip()]


def _as_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise SqliteOptionError(f"{name} must be an integer, got {value!r}.") from error
    if not minimum <= number <= maximum:
        raise SqliteOptionError(f"{name} must be between {minimum} and {maximum}, got {number}.")
    return number


def _as_float(value: Any, name: str, minimum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise SqliteOptionError(f"{name} must be a number, got {value!r}.") from error
    if number < minimum:
        raise SqliteOptionError(f"{name} must be at least {minimum}, got {number}.")
    return number


def _translate(error: sqlite3.Error, context: str) -> SqliteConnectorError:
    """Map a raw sqlite error onto the connector's clear, typed error."""
    message = str(error)
    lowered = message.casefold()
    if "locked" in lowered or "busy" in lowered:
        return SqliteAccessError(f"The database is locked ({context}): {message}")
    if "unable to open database file" in lowered:
        return SqliteAccessError(f"Cannot open the database ({context}): {message}")
    if "not a database" in lowered or "malformed" in lowered or "encrypted" in lowered:
        return SqliteSchemaError(f"Not a readable SQLite database ({context}): {message}")
    if isinstance(error, sqlite3.IntegrityError):  # pragma: no cover - defensive
        return SqliteReadError(f"Integrity error reading {context}: {message}")
    return SqliteReadError(f"Read failed for {context}: {message}")


__all__ = [
    "DEFAULT_PAGE",
    "DEFAULT_ROW_GROUP",
    "MAX_PAGE",
    "MAX_ROW_GROUP",
    "SQLITE_SCHEMES",
    "ColumnInfo",
    "ForeignKey",
    "SqliteAccessError",
    "SqliteConnector",
    "SqliteConnectorError",
    "SqliteOptionError",
    "SqliteOptions",
    "SqlitePathError",
    "SqliteReadError",
    "SqliteSchemaError",
    "SqliteValueError",
    "TableSchema",
]
