"""Error reporting for ingestion: never let one bad source abort a batch.

Ingestion runs over many sources, and a single unreadable file, a locked
database, or a refused web fetch must not stop the rest. This module gives the
pipeline a typed error (:class:`SourceError`) and an :class:`ErrorLog` that
collects them, mirroring how :class:`core.tools.ToolResult` lets an agent loop
continue after a failed tool call.

Messages are kept free of file *contents* and credentials: a report names the
source and the reason, never the data it held.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class IngestionError(Exception):
    """Base class for ingestion failures that a caller may want to catch."""


class ConnectorError(IngestionError):
    """A connector could not read a source (bad format, missing dependency, ...)."""


class IngestCancelled(IngestionError):
    """A cooperative cancel stopped the run; committed work is kept, the rest is not.

    This is *not* a connector failure: the caller asked to stop, so it must not
    be recorded as a failed source, and the job must be marked ``cancelled``
    rather than ``done``. The pipeline checks for it separately from the other
    errors so a cancel never inflates the failure count.
    """


class IngestInterrupted(IngestionError):
    """A deliberate interruption at a controlled point (tests, embedded callers).

    Models a crash between a durably committed batch and its checkpoint: the run
    stops where it was told to, already-written work stays, and the next run
    resumes from the last checkpoint. It is never retried and never counted as a
    failure, so the pipeline re-raises it instead of folding it into ``failed``.
    """


@dataclass(frozen=True)
class SourceError:
    """One source that failed, and why."""

    source: str
    kind: str
    message: str

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("SourceError source cannot be empty.")
        if not self.kind.strip():
            raise ValueError("SourceError kind cannot be empty.")

    def as_dict(self) -> dict[str, str]:
        return {"source": self.source, "kind": self.kind, "message": self.message}


@dataclass
class ErrorLog:
    """Collects per-source failures so a run can report them at the end."""

    errors: list[SourceError] = field(default_factory=list)

    def add(self, source: str, kind: str, message: str) -> SourceError:
        error = SourceError(source=source, kind=kind, message=_sanitise(message))
        self.errors.append(error)
        return error

    def __len__(self) -> int:
        return len(self.errors)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dicts(self) -> list[dict[str, str]]:
        return [error.as_dict() for error in self.errors]

    def summary(self) -> str:
        if not self.errors:
            return "no errors"
        return f"{len(self.errors)} source(s) failed"


_SENSITIVE_HINTS = ("password", "passwd", "secret", "token", "apikey", "api_key", "key=")


def _sanitise(message: str) -> str:
    """Drop anything that looks like an embedded credential from a message.

    Drivers sometimes put a full connection string in an exception; a report
    must not carry it. When a sensitive word is present we keep only the first
    line and redact the rest, rather than trying to parse every format.
    """
    lowered = message.lower()
    if any(hint in lowered for hint in _SENSITIVE_HINTS):
        first = message.splitlines()[0] if message.splitlines() else message
        return first.split("//", 1)[0] + "[credentials redacted]"
    return message.splitlines()[0] if message.splitlines() else message


__all__ = [
    "ConnectorError",
    "ErrorLog",
    "IngestCancelled",
    "IngestInterrupted",
    "IngestionError",
    "SourceError",
]
