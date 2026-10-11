"""Persisted registration of the sources a user has added to the ingest system.

The dedup ledger (:mod:`ingest.dedup`) answers *what has already been read*; it
is a cache that can be rebuilt from the store. This module answers a different
question the subsystem needs for its CLI and UI: *what sources has the user
actually registered*, whether or not they have been ingested yet, and what
happened the last time each was run.

A registration is a small, text-free record -- the dedup ledger already knows the
chunk count, so this one holds identity, provenance, and promotion state:

* the :class:`SourceSpec` (uri + options) that can rebuild a connector for it;
* where it came from (``local`` or an authorised ``remote`` source) and who
  provided it;
* the conservative promotion flags ``trusted`` and ``enabled`` -- an external
  source is **not** trusted just because it was registered;
* the outcome of its last run (status, counts, timestamp).

Storage is the project's existing :class:`storage.sqlite_store.SqliteStore`
(namespaced JSON values in a local SQLite file), the same lightweight precedent
:mod:`ingest.dedup` follows; no new dependency and no server.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from collections.abc import Mapping

from storage.sqlite_store import SqliteStore

_NAMESPACE = "sources"

#: Registration status values, mirroring :mod:`ingest.jobs`.
PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

_STATUSES = frozenset({PENDING, RUNNING, DONE, FAILED})

#: Where a source came from. ``remote`` is the only value that touches a network.
LOCAL = "local"
REMOTE = "remote"


@dataclass(frozen=True)
class SourceRegistration:
    """One registered source, with provenance and last-run state."""

    source_id: str
    uri: str
    origin: str = LOCAL
    connector: str = ""
    provider: str = ""
    title: str = ""
    license: str = ""
    tags: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)
    trusted: bool = False
    enabled: bool = True
    status: str = PENDING
    last_run: str = ""
    chunks: int = 0
    bytes: int = 0
    documents: int = 0
    error: str = ""

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("SourceRegistration source_id cannot be empty.")
        if not self.uri.strip():
            raise ValueError("SourceRegistration uri cannot be empty.")
        if self.origin not in {LOCAL, REMOTE}:
            raise ValueError(f"Unknown origin: {self.origin!r}")
        if self.status not in _STATUSES:
            raise ValueError(f"Unknown status: {self.status!r}")

    # -- serialisation ----------------------------------------------------

    def to_primitive(self) -> dict[str, Any]:
        data = asdict(self)
        data["tags"] = list(self.tags)
        data["options"] = dict(self.options)
        return data

    @classmethod
    def from_primitive(cls, data: Mapping[str, Any]) -> "SourceRegistration":
        return cls(
            source_id=str(data["source_id"]),
            uri=str(data["uri"]),
            origin=str(data.get("origin", LOCAL)),
            connector=str(data.get("connector", "")),
            provider=str(data.get("provider", "")),
            title=str(data.get("title", "")),
            license=str(data.get("license", "")),
            tags=tuple(str(tag) for tag in data.get("tags", ())),
            options=dict(data.get("options", {})),
            trusted=bool(data.get("trusted", False)),
            enabled=bool(data.get("enabled", True)),
            status=str(data.get("status", PENDING)),
            last_run=str(data.get("last_run", "")),
            chunks=int(data.get("chunks", 0)),
            bytes=int(data.get("bytes", 0)),
            documents=int(data.get("documents", 0)),
            error=str(data.get("error", "")),
        )

    def mark_run(
        self,
        status: str,
        *,
        chunks: int = 0,
        bytes_written: int = 0,
        documents: int = 0,
        error: str = "",
    ) -> "SourceRegistration":
        """A copy updated with the outcome of one run (keeps registration intact)."""
        return SourceRegistration(
            source_id=self.source_id,
            uri=self.uri,
            origin=self.origin,
            connector=self.connector,
            provider=self.provider,
            title=self.title,
            license=self.license,
            tags=self.tags,
            options=self.options,
            trusted=self.trusted,
            enabled=self.enabled,
            status=status,
            last_run=_now(),
            chunks=chunks,
            bytes=bytes_written,
            documents=documents,
            error=error[:500],
        )


class SourceRegistry:
    """A SQLite-backed, namespaced registry of :class:`SourceRegistration`."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._store = SqliteStore(self.path)

    def register(self, registration: SourceRegistration) -> SourceRegistration:
        """Add or replace one registration, keyed by ``source_id``."""
        self._store.save(_NAMESPACE, registration.source_id, registration.to_primitive())
        return registration

    def get(self, source_id: str) -> SourceRegistration | None:
        data = self._store.load(_NAMESPACE, source_id)
        if not isinstance(data, dict):
            return None
        return SourceRegistration.from_primitive(data)

    def list(self) -> list[SourceRegistration]:
        """All registrations, newest first when timestamps are present."""
        items = []
        for key in self._store.list_keys(_NAMESPACE):
            data = self._store.load(_NAMESPACE, key)
            if isinstance(data, dict):
                try:
                    items.append(SourceRegistration.from_primitive(data))
                except (KeyError, ValueError, TypeError):
                    continue
        items.sort(key=lambda item: (item.last_run, item.source_id), reverse=True)
        return items

    def remove(self, source_id: str) -> bool:
        """Forget one registration. Returns whether it existed."""
        return self._store.delete(_NAMESPACE, source_id)

    def add(
        self,
        uri: str,
        *,
        origin: str = LOCAL,
        connector: str = "",
        provider: str = "",
        title: str = "",
        license: str = "",
        tags: tuple[str, ...] = (),
        options: Mapping[str, Any] | None = None,
        trusted: bool = False,
        enabled: bool = True,
        source_id: str = "",
    ) -> SourceRegistration:
        """Convenience: build a registration with a stable id and store it.

        The id defaults to a slug of the uri so the same uri registers once. An
        external source is never trusted by default here; the caller must pass
        ``trusted=True`` explicitly after the promotion flow.
        """
        registration = SourceRegistration(
            source_id=source_id or source_slug(uri),
            uri=uri,
            origin=origin,
            connector=connector,
            provider=provider,
            title=title,
            license=license,
            tags=tuple(tags),
            options=dict(options or {}),
            trusted=trusted,
            enabled=enabled,
        )
        return self.register(registration)


def source_slug(uri: str) -> str:
    """A stable, filesystem- and SQL-safe id derived from a uri."""
    return re.sub(r"[^0-9a-zA-Z]+", "-", uri.strip()).strip("-").lower() or "source"


def source_registry_path(store_root: str | Path) -> Path:
    """Where the source registry lives, beside the ledger under the store root."""
    return Path(store_root) / "ingest" / "sources.sqlite"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "DONE",
    "FAILED",
    "LOCAL",
    "PENDING",
    "REMOTE",
    "RUNNING",
    "SourceRegistration",
    "SourceRegistry",
    "source_registry_path",
    "source_slug",
]
