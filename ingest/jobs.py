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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

#: Job status values.
RUNNING = "running"
DONE = "done"
FAILED = "failed"


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

    def __post_init__(self) -> None:
        if not self.job_id.strip():
            raise ValueError("JobState job_id cannot be empty.")
        if not self.source.strip():
            raise ValueError("JobState source cannot be empty.")

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def as_dict(self) -> dict:
        return asdict(self)


class JobStore:
    """A directory of :class:`JobState` files, one per job."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, job_id: str) -> Path:
        return self.root / f"{_safe_name(job_id)}.json"

    def save(self, state: JobState) -> JobState:
        state.touch()
        self.root.mkdir(parents=True, exist_ok=True)
        self._path(state.job_id).write_text(
            json.dumps(state.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
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


def _safe_name(job_id: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in job_id)


__all__ = ["DONE", "FAILED", "RUNNING", "JobState", "JobStore"]
