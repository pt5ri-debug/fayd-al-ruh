"""Resumable ingestion jobs: remember how far a long import got.

A 100 GB ingest can run for a long time, so it must survive an interruption. A
:class:`JobState` records the source, the connector, and the index of the last
source *item* completed. The pipeline saves it after every item, and a re-run
with the same job id starts from that position instead of the beginning.

State is tiny JSON -- a position and running totals, never document text -- so
writing it after every item is cheap. One file per job under a jobs directory
keeps jobs independent and easy to list or delete.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

#: Job status values.
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"


@dataclass
class JobState:
    """How far one ingestion job got, and what it has done so far."""

    job_id: str
    source: str
    connector: str = ""
    position: int = 0
    records: int = 0
    chunks: int = 0
    bytes_written: int = 0
    status: str = RUNNING
    updated_at: str = ""
    failures: list[str] = field(default_factory=list)
    """Actionable per-item/per-batch failure messages, capped by the writer."""

    def __post_init__(self) -> None:
        if not self.job_id.strip():
            raise ValueError("JobState job_id cannot be empty.")
        if not self.source.strip():
            raise ValueError("JobState source cannot be empty.")

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def as_dict(self) -> dict:
        return asdict(self)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file + fsync + rename).

    A job file is the *only* record of how far a resumable run got, so a crash
    while saving must leave either the old file or the new one -- never a
    half-written file that would make a resumed run misread its position.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


class JobStore:
    """A directory of :class:`JobState` files, one per job."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, job_id: str) -> Path:
        return self.root / f"{_safe_name(job_id)}.json"

    def save(self, state: JobState) -> JobState:
        state.touch()
        _atomic_write_text(
            self._path(state.job_id),
            json.dumps(state.as_dict(), ensure_ascii=False, indent=2),
        )
        return state

    def load(self, job_id: str) -> JobState | None:
        path = self._path(job_id)
        if not path.exists():
            return None
        return JobState(**json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[JobState]:
        if not self.root.exists():
            return []
        states = []
        for path in sorted(self.root.glob("*.json")):
            try:
                states.append(JobState(**json.loads(path.read_text(encoding="utf-8"))))
            except (ValueError, TypeError):
                continue
        return states

    def remove(self, job_id: str) -> bool:
        path = self._path(job_id)
        if path.exists():
            path.unlink()
            return True
        return False


def write_atomic_text(path: Path, text: str) -> None:
    """Public wrapper over the atomic write (used by the SQL job store)."""
    _atomic_write_text(path, text)


def _safe_name(job_id: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in job_id)


__all__ = [
    "CANCELLED",
    "DONE",
    "FAILED",
    "RUNNING",
    "JobState",
    "JobStore",
    "write_atomic_text",
]
