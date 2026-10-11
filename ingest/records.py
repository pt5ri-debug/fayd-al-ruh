"""The small contracts every ingestion connector speaks.

Keeping the connector surface to a single ``stream`` method that yields
:class:`SourceRecord` objects is what lets the pipeline treat a folder of CSV
files, a SQL table, and an authorised web page the same way -- and what keeps
any one connector from holding a whole dataset in memory: it *yields*, it does
not return a list.

Nothing here imports the training pipeline or the store; a record is a plain
value that the pipeline later turns into a citable chunk.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class SourceSpec:
    """What the caller asked to ingest.

    ``uri`` is the only required field: a filesystem path, an ``sqlite://`` URL,
    a ``https://`` URL, and so on. ``options`` carries connector-specific
    settings (a SQL query, a column mapping, a glob) without leaking them into
    the pipeline.
    """

    uri: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.uri.strip():
            raise ValueError("Source uri cannot be empty.")

    def option(self, name: str, default: Any = None) -> Any:
        return self.options.get(name, default)


@dataclass(frozen=True)
class SourceRecord:
    """One logical unit of text on its way into the store.

    A record is deliberately close to :class:`core.documents.Document` but keeps
    the connector-specific bits (a stable ``source_key`` for incremental skips,
    an optional per-record ``cursor`` for resumable SQL/file paging) that the
    pipeline needs before it chunks and stores.
    """

    title: str
    text: str
    source: str
    source_key: str = ""
    url: str = ""
    license: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    cursor: str = ""

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("SourceRecord title cannot be empty.")
        if not self.text.strip():
            raise ValueError("SourceRecord text cannot be empty.")
        if not self.source.strip():
            raise ValueError("SourceRecord source cannot be empty.")


@dataclass(frozen=True)
class IngestItem:
    """A source *item* the connector can read, described without reading it.

    A connector's :meth:`Connector.plan` yields these cheaply (a file path and
    its fingerprint, a table name, a URL). The pipeline checks each against the
    dedup ledger and only calls :meth:`Connector.stream_item` for the items that
    actually need reading, so an incremental re-run of a 100 GB folder touches
    no file contents.
    """

    source_key: str
    uri: str
    title: str
    fingerprint: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_key.strip():
            raise ValueError("IngestItem source_key cannot be empty.")
        if not self.uri.strip():
            raise ValueError("IngestItem uri cannot be empty.")
        if not self.title.strip():
            raise ValueError("IngestItem title cannot be empty.")


@dataclass(frozen=True)
class ConnectorInfo:
    """Static description of a connector, used by the registry and the CLI."""

    name: str
    description: str
    schemes: frozenset[str] = frozenset()
    suffixes: frozenset[str] = frozenset()
    remote: bool = False
    """Remote connectors touch the network and must pass the permission gates."""

    optional_dependency: str = ""
    """When set, the connector needs an optional install to run."""

    default: bool = False
    """The fallback connector, used when no scheme or suffix matches (files)."""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ConnectorInfo name cannot be empty.")
        if not self.description.strip():
            raise ValueError(f"Connector {self.name!r} needs a description.")


@dataclass
class IngestionReport:
    """A running summary of one ingestion run."""

    sources: int = 0
    records: int = 0
    chunks: int = 0
    bytes_written: int = 0
    skipped_duplicates: int = 0
    skipped_unchanged: int = 0
    failed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "sources": self.sources,
            "records": self.records,
            "chunks": self.chunks,
            "bytes_written": self.bytes_written,
            "skipped_duplicates": self.skipped_duplicates,
            "skipped_unchanged": self.skipped_unchanged,
            "failed": self.failed,
        }


@runtime_checkable
class Connector(Protocol):
    """Reads one kind of source as a lazy stream of :class:`SourceRecord`.

    ``plan`` yields the source's *items* cheaply (without reading their text) so
    the pipeline can decide which ones are new; ``stream_item`` reads one item
    into records; ``stream`` is the convenience that walks every item, for
    callers that do not need incremental skipping.
    """

    info: ConnectorInfo

    def plan(self, spec: SourceSpec) -> Iterator[IngestItem]:
        """Yield the source's items, cheaply and lazily."""
        ...

    def stream_item(self, item: IngestItem) -> Iterator[SourceRecord]:
        """Read one item into records (called only for items that need reading)."""
        ...

    def stream(self, spec: SourceSpec) -> Iterator[SourceRecord]:
        """Yield records for ``spec`` without materialising the whole source."""
        ...


def records_from_texts(
    texts: Iterable[tuple[str, str]],
    *,
    source: str,
    url: str = "",
    license: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[SourceRecord]:
    """Adapt ``(title, text)`` pairs into records, a convenience for connectors."""
    base = dict(metadata or {})
    for title, text in texts:
        if not text.strip():
            continue
        yield SourceRecord(
            title=title,
            text=text,
            source=source,
            url=url,
            license=license,
            metadata=base,
        )


__all__ = [
    "Connector",
    "ConnectorInfo",
    "IngestItem",
    "IngestionReport",
    "SourceRecord",
    "SourceSpec",
    "records_from_texts",
]
