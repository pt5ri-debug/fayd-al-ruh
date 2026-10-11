"""Resolve a source to the connector that can read it.

The registry is the one place that knows how URIs map to connectors, so the
pipeline and the CLI stay dumb: they ask for a connector by URI and never branch
on file type themselves. Resolution is by scheme first (``sqlite://``,
``https://``), then by file suffix (``.csv``, ``.docx``), and a connector may
also *probe* a spec it does not claim statically -- the web connector uses this
to refuse a non-allow-listed host with a clear error.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import PurePosixPath
from urllib.parse import urlparse

from ingest.errors import ConnectorError
from ingest.records import Connector, ConnectorInfo, SourceSpec


class ConnectorRegistry:
    """An ordered set of connectors with scheme/suffix resolution."""

    def __init__(self, connectors: Iterable[Connector] = ()) -> None:
        self._connectors: list[Connector] = []
        self._by_name: dict[str, Connector] = {}
        for connector in connectors:
            self.register(connector)

    def register(self, connector: Connector) -> Connector:
        info = connector.info
        if info.name in self._by_name:
            raise ValueError(f"Connector {info.name!r} is already registered.")
        self._connectors.append(connector)
        self._by_name[info.name] = connector
        return connector

    def get(self, name: str) -> Connector | None:
        return self._by_name.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(connector.info.name for connector in self._connectors)

    def infos(self) -> tuple[ConnectorInfo, ...]:
        return tuple(connector.info for connector in self._connectors)

    def resolve(self, spec: SourceSpec) -> Connector:
        """Pick a connector for ``spec``, or raise :class:`ConnectorError`."""
        scheme = _scheme(spec.uri)
        if scheme:
            for connector in self._connectors:
                if scheme in connector.info.schemes:
                    return connector
        suffix = _suffix(spec.uri)
        if suffix:
            for connector in self._connectors:
                if suffix in connector.info.suffixes:
                    return connector
        for connector in self._connectors:
            if connector.info.default:
                return connector
        raise ConnectorError(
            f"No connector handles {spec.uri!r} "
            f"(scheme {scheme or 'none'}, suffix {suffix or 'none'}). "
            f"Known connectors: {', '.join(self.names()) or 'none'}."
        )


def _scheme(uri: str) -> str:
    """The URI scheme, but only when it is a real one (not a Windows drive)."""
    parsed = urlparse(uri)
    if not parsed.scheme:
        return ""
    # ``C:\path`` parses as scheme ``c``; a single-letter scheme with no ``//``
    # is a Windows drive, not a scheme we handle.
    if len(parsed.scheme) == 1 and not uri.startswith(f"{parsed.scheme}://"):
        return ""
    return parsed.scheme.lower()


def _suffix(uri: str) -> str:
    parsed = urlparse(uri)
    path = parsed.path if parsed.scheme else uri
    name = PurePosixPath(path.replace("\\", "/")).name
    dot = name.rfind(".")
    return name[dot:].lower() if dot > 0 else ""


def build_default_registry(*, allow_web: bool = False) -> ConnectorRegistry:
    """The registry the CLI and pipeline use by default.

    ``allow_web`` is off by default: the internet connector is only registered
    when the caller has explicitly opted in, so a plain local run cannot even
    resolve a URL -- the deny-by-default posture the project uses elsewhere.
    """
    from ingest.connectors.files import FileConnector
    from ingest.connectors.sql import SqlConnector

    registry = ConnectorRegistry()
    registry.register(FileConnector())
    registry.register(SqlConnector())
    if allow_web:
        from ingest.connectors.web import WebConnector

        registry.register(WebConnector())
    return registry


__all__ = ["ConnectorRegistry", "build_default_registry"]
