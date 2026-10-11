"""Explicit interfaces for the ingestion subsystem: connectors and jobs.

The concrete pieces already exist -- :class:`ingest.records.Connector` is the
connector protocol, :class:`ingest.jobs.JobState`/:class:`ingest.jobs.JobStore`
are the job records and store. This module gives those contracts *names* a caller
can import and check against, so an extension point is explicit rather than
implied:

* :class:`SourceConnector` -- what every source connector must provide;
* :class:`IngestionJob` -- the shape of a job's progress record;
* :class:`JobStateStore` -- what a job-state store must provide.

They are :func:`typing.runtime_checkable`, so a caller can ``isinstance``-check a
candidate before using it (e.g. a third-party connector, or a test double),
without importing the concrete classes.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from ingest.records import (
    Connector,
    ConnectorInfo,
    IngestItem,
    SourceRecord,
    SourceSpec,
)


@runtime_checkable
class SourceConnector(Connector, Protocol):
    """The explicit interface every source connector implements.

    Structurally identical to :class:`ingest.records.Connector`; declared here so
    the connector extension point has a stable, named type that a third-party
    connector (or a test double) can be checked against.
    """

    info: ConnectorInfo

    def plan(self, spec: SourceSpec) -> Iterator[IngestItem]:
        """Yield the source's items cheaply and lazily (no text read)."""
        ...

    def stream_item(self, item: IngestItem) -> Iterator[SourceRecord]:
        """Read one planned item into records; called only for new items."""
        ...

    def stream(self, spec: SourceSpec) -> Iterator[SourceRecord]:
        """Yield every record for ``spec`` without materialising the source."""
        ...


@runtime_checkable
class IngestionJob(Protocol):
    """The shape of a job's progress record (:class:`ingest.jobs.JobState`).

    A job is identified by ``job_id``, names the ``source`` it works on, and
    carries the ``position`` reached, the running totals, and a ``status``.
    """

    job_id: str
    source: str
    position: int
    records: int
    chunks: int
    bytes_written: int
    status: str


@runtime_checkable
class JobStateStore(Protocol):
    """The explicit interface for persisting job state (:class:`ingest.jobs.JobStore`)."""

    def save(self, state: IngestionJob) -> object:
        """Persist ``state`` (add or replace)."""
        ...

    def load(self, job_id: str) -> object | None:
        """Return the stored state for ``job_id``, or ``None``."""
        ...

    def list(self) -> list[object]:
        """Return every stored state."""
        ...

    def remove(self, job_id: str) -> bool:
        """Forget ``job_id``; return whether it existed."""
        ...


__all__ = [
    "IngestionJob",
    "JobStateStore",
    "SourceConnector",
]
