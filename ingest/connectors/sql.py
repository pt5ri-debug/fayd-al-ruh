"""Read SQL databases as a streaming source of records.

SQLite is supported with the standard library (no new dependency), and other
engines (PostgreSQL, MySQL, ...) go through the DB-API when their driver is
installed -- an *optional* extra, imported lazily so the base install stays
dependency-free, exactly as :mod:`core.document_reader` treats PDF.

Safety is the same posture the rest of the project uses:

* **Read-only.** The SQLite connection is opened with a ``file:...?mode=ro``
  URI, and other engines are never issued a mutating statement. A statement that
  is not a ``SELECT``/``WITH`` is refused before it runs.
* **Streaming.** Rows are fetched with a server-side (or chunked) cursor and
  yielded a page at a time, so a table with hundreds of millions of rows never
  lands in RAM. Only ``page`` rows are held at once.
* **No SQL injection.** The statement is a read-only ``SELECT`` the operator
  supplies; values are never interpolated, and the connection string is read
  from the source URI/options, never logged.

Each table (or the single query given in options) becomes one source item, whose
fingerprint is the table's row count and max rowid when available, so an
unchanged database is skipped on a re-run.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ingest.errors import ConnectorError
from ingest.hashing import source_key
from ingest.records import (
    ConnectorInfo,
    IngestItem,
    SourceRecord,
    SourceSpec,
)

#: Rows fetched per round-trip; bounds memory regardless of table size.
DEFAULT_PAGE = 500
#: Rows grouped into one SourceRecord so a table becomes useful passages.
DEFAULT_ROW_GROUP = 40

_READ_ONLY_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_SQLITE_SCHEMES = frozenset({"sqlite", "sqlite3"})
_DRIVER_SCHEMES = {
    "postgresql": "psycopg",
    "postgres": "psycopg",
    "mysql": "pymysql",
}


class SqlConnector:
    """A read-only, streaming connector for DB-API SQL databases.

    SQLite is read by the dedicated standard-library :class:`ingest.connectors.sqlite.SqliteConnector`
    in :mod:`ingest.connectors.sqlite`; this connector keeps the optional driver
    schemes (PostgreSQL, MySQL) and its SQLite helpers for direct use.
    """

    info = ConnectorInfo(
        name="sql",
        description="Read-only, streaming SQL databases (PostgreSQL/MySQL via DB-API).",
        schemes=frozenset(_DRIVER_SCHEMES),
        optional_dependency="psycopg",
    )

    # -- planning ---------------------------------------------------------

    def plan(self, spec: SourceSpec) -> Iterator[IngestItem]:
        scheme = spec.uri.split(":", 1)[0].lower()
        explicit_query = spec.option("query")
        if explicit_query:
            tables = [str(spec.option("table", "query"))]
        else:
            tables = self._tables(spec, scheme)
        page = int(spec.option("page", DEFAULT_PAGE) or DEFAULT_PAGE)
        row_group = int(spec.option("row_group", DEFAULT_ROW_GROUP) or DEFAULT_ROW_GROUP)
        db_path = _sqlite_uri_path(spec.uri) if scheme in _SQLITE_SCHEMES else ""
        for name in tables:
            yield IngestItem(
                source_key=source_key(spec.uri, extra=name),
                uri=f"{spec.uri}#{name}",
                title=name,
                fingerprint=self._fingerprint(spec, scheme, name),
                metadata={
                    "table": name,
                    "scheme": scheme,
                    "page": page,
                    "row_group": row_group,
                    "query": explicit_query or "",
                    "dsn": str(spec.option("dsn", "") or ""),
                    "db_path": db_path,
                },
            )

    def _tables(self, spec: SourceSpec, scheme: str) -> list[str]:
        configured = spec.option("tables")
        if configured:
            return [str(name) for name in configured]
        if scheme in _SQLITE_SCHEMES:
            return self._sqlite_tables(self._sqlite_path(spec))
        raise ConnectorError(
            f"Listing tables needs an explicit tables= option for {scheme!r} databases."
        )

    def _fingerprint(self, spec: SourceSpec, scheme: str, table: str) -> str:
        if scheme not in _SQLITE_SCHEMES:
            return ""
        try:
            connection = self._sqlite_connect(self._sqlite_path(spec))
        except ConnectorError:
            return ""
        try:
            count = connection.execute(
                f'SELECT count(*) FROM "{_safe_identifier(table)}"'
            ).fetchone()[0]
            try:
                rowid = connection.execute(
                    f'SELECT max(rowid) FROM "{_safe_identifier(table)}"'
                ).fetchone()[0]
            except sqlite3.Error:
                rowid = ""
        except sqlite3.Error:
            return ""
        finally:
            connection.close()
        return f"{count}:{rowid}"

    # -- reading ----------------------------------------------------------

    def stream_item(self, item: IngestItem) -> Iterator[SourceRecord]:
        scheme = str(item.metadata.get("scheme", ""))
        table = str(item.metadata.get("table", item.title))
        page = int(item.metadata.get("page", DEFAULT_PAGE) or DEFAULT_PAGE)
        row_group = int(item.metadata.get("row_group", DEFAULT_ROW_GROUP) or DEFAULT_ROW_GROUP)
        query = item.metadata.get("query") or None
        if scheme in _SQLITE_SCHEMES:
            connection = self._sqlite_connect(Path(str(item.metadata.get("db_path", ""))))
            try:
                statement = query or f'SELECT * FROM "{_safe_identifier(table)}"'
                yield from self._stream(connection, statement, table, item, page, row_group)
            finally:
                connection.close()
            return
        yield from self._stream_driver(scheme, item, table, query, page, row_group)

    def _stream(
        self,
        connection: Any,
        statement: str,
        title: str,
        item: IngestItem,
        page: int,
        row_group: int,
    ) -> Iterator[SourceRecord]:
        _require_read_only(statement)
        cursor = connection.cursor()
        try:
            cursor.execute(statement)
            columns = [description[0] for description in cursor.description or []]
            buffer: list[str] = []
            part = 0
            while True:
                rows = cursor.fetchmany(page)
                if not rows:
                    break
                for row in rows:
                    buffer.append(_render_row(columns, row))
                    if len(buffer) >= row_group:
                        part += 1
                        yield _record(item, title, buffer, part)
                        buffer = []
            if buffer:
                part += 1
                yield _record(item, title, buffer, part)
        except ConnectorError:
            raise
        except Exception as error:  # driver-specific error type
            raise ConnectorError(f"Query failed: {error}") from error
        finally:
            cursor.close()

    def _stream_driver(
        self, scheme: str, item: IngestItem, table: str, query: Any, page: int, row_group: int
    ) -> Iterator[SourceRecord]:
        module_name = _DRIVER_SCHEMES.get(scheme)
        if module_name is None:
            raise ConnectorError(f"No SQL driver is configured for {scheme!r}.")
        # The connection string is supplied through the source options, never
        # built from user input and never logged.
        dsn = str(item.metadata.get("dsn", "") or "")
        if not dsn:
            raise ConnectorError(
                f"{scheme!r} sources need a dsn= option (a libpq/MySQL connection string)."
            )
        try:
            module = __import__(module_name)
        except ImportError as error:
            raise ConnectorError(
                f"The {scheme!r} connector needs the optional '{module_name}' package. "
                f"Install it with `pip install {module_name}` and try again."
            ) from error
        try:
            connection = module.connect(dsn)
        except Exception as error:  # driver-specific; keep the message, redact anyway
            raise ConnectorError(f"Could not connect to the {scheme!r} database: {error}") from error
        try:
            statement = query or f'SELECT * FROM "{_safe_identifier(table)}"'
            yield from self._stream(connection, statement, table, item, page, row_group)
        finally:
            connection.close()

    # -- SQLite helpers ---------------------------------------------------

    def _sqlite_path(self, spec: SourceSpec) -> Path:
        return Path(_sqlite_uri_path(spec.uri)).expanduser()

    def _sqlite_connect(self, path: Path) -> sqlite3.Connection:
        if not path.exists():
            raise ConnectorError(f"No such SQLite database: {path}")
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
        return connection

    def _sqlite_tables(self, path: Path) -> list[str]:
        connection = self._sqlite_connect(path)
        try:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        except sqlite3.Error as error:
            raise ConnectorError(f"Could not list tables: {error}") from error
        finally:
            connection.close()
        return [row[0] for row in rows]

    def stream(self, spec: SourceSpec) -> Iterator[SourceRecord]:
        for item in self.plan(spec):
            yield from self.stream_item(item)


def _sqlite_uri_path(uri: str) -> str:
    """The filesystem path inside a ``sqlite://`` URI (or a bare path).

    ``sqlite:///data/db.sqlite`` is an absolute path ``/data/db.sqlite``;
    ``sqlite://data/db.sqlite`` and a plain ``data/db.sqlite`` are relative.
    """
    if "://" not in uri:
        return uri
    _scheme, rest = uri.split("://", 1)
    return rest


def _require_read_only(statement: str) -> None:
    if not _READ_ONLY_RE.match(statement):
        raise ConnectorError(
            "Only read-only SELECT/WITH statements are allowed; refusing the query."
        )


def _safe_identifier(name: str) -> str:
    """A bare SQL identifier, quoted; anything odd is refused, never interpolated."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ConnectorError(f"Unsafe SQL identifier: {name!r}")
    return name


def _render_row(columns: list[str], row: Any) -> str:
    pairs = [f"{column}: {value}" for column, value in zip(columns, row) if value is not None]
    return "; ".join(pairs) or " ".join(str(item) for item in row)


def _record(item: IngestItem, title: str, rows: list[str], part: int) -> SourceRecord:
    return SourceRecord(
        title=f"{title} — part {part}",
        text="\n".join(rows),
        source="sql",
        source_key=item.source_key,
        metadata={"table": title, "part": part, "kind": "sql"},
    )


__all__ = ["DEFAULT_PAGE", "DEFAULT_ROW_GROUP", "SqlConnector"]
