"""The durable history of everything the Training Center did.

Every learning session, import, crawl, dataset build, training run, and
evaluation should be findable later, with what it produced. The project already
keeps the *authoritative* records for most of those (the run log, the dataset
manifests, the knowledge-base manifest); this module is the one **joined, human
view** over them, plus a small durable journal of the Training Center's own
actions, so a user can see one timeline instead of five.

The journal is stored through the project's own
:class:`storage.json_store.JsonStore`, is bounded, and is corruption-tolerant: a
broken file reads as empty rather than crashing a window. It is dependency-free
and Tk-free.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from storage.json_store import JsonStore

from core.paths import data_dir
from core.training_center.records import HistoryEntry

#: The JsonStore namespace and key the journal lives under.
NAMESPACE = "training_center"
KEY = "history"

#: How many entries to keep; the journal is a timeline, not an archive.
DEFAULT_LIMIT = 500

#: The kinds a history entry can have, for filtering.
KINDS: tuple[str, ...] = (
    "import",
    "crawl",
    "ask",
    "review",
    "dataset",
    "training",
    "evaluation",
    "pipeline",
    "knowledge",
    "settings",
)


@dataclass(frozen=True)
class HistoryLine:
    """One line of the joined timeline: the Training Center's own view."""

    at: str
    kind: str
    summary: str
    outcome: str = ""
    source: str = "journal"

    def describe(self) -> str:
        when = self.at.replace("T", " ")[:19]
        suffix = f" ({self.outcome})" if self.outcome else ""
        return f"{when}  [{self.kind}] {self.summary}{suffix}"


class TrainingHistory:
    """The durable journal plus a joined view over the project's own records."""

    def __init__(self, *, root: Path | str | None = None, store: JsonStore | None = None) -> None:
        self.root = Path(root) if root is not None else data_dir()
        self._store = store or JsonStore(self.root)

    # -- the journal -------------------------------------------------------
    def record(self, kind: str, summary: str, *, outcome: str = "", **detail: object) -> HistoryEntry:
        """Append one entry, keeping the journal bounded."""
        entry = HistoryEntry(kind=kind, summary=summary, outcome=outcome, detail=dict(detail))
        entries = list(self.entries())
        entries.append(entry)
        if len(entries) > DEFAULT_LIMIT:
            entries = entries[-DEFAULT_LIMIT:]
        self._save(entries)
        return entry

    def entries(self) -> tuple[HistoryEntry, ...]:
        """The journal, oldest first. A corrupt file reads as empty."""
        raw = self._store.load(NAMESPACE, KEY)
        if not isinstance(raw, list):
            return ()
        entries: list[HistoryEntry] = []
        for item in raw:
            try:
                entries.append(HistoryEntry.from_primitive(item))
            except (ValueError, TypeError):
                continue
        return tuple(entries)

    def clear(self) -> int:
        count = len(self.entries())
        self._save([])
        return count

    def _save(self, entries: Iterable[HistoryEntry]) -> None:
        self._store.save(NAMESPACE, KEY, [entry.to_primitive() for entry in entries])

    # -- filtering ---------------------------------------------------------
    def search(self, *, kind: str = "", query: str = "", limit: int = 100) -> tuple[HistoryLine, ...]:
        """The joined timeline, filtered by kind and free text, newest first."""
        lines = list(self.timeline())
        if kind:
            lines = [line for line in lines if line.kind == kind]
        if query.strip():
            needle = query.strip().lower()
            lines = [
                line
                for line in lines
                if needle in line.summary.lower() or needle in line.outcome.lower()
            ]
        lines.sort(key=lambda line: line.at, reverse=True)
        return tuple(lines[: max(1, limit)])

    def timeline(self) -> tuple[HistoryLine, ...]:
        """Every line of the joined timeline: the journal plus the real records."""
        lines: list[HistoryLine] = [
            HistoryLine(at=entry.at, kind=entry.kind, summary=entry.summary,
                        outcome=entry.outcome, source="journal")
            for entry in self.entries()
        ]
        lines.extend(self._dataset_lines())
        lines.extend(self._run_lines())
        lines.extend(self._knowledge_lines())
        return tuple(lines)

    def _studio(self):
        """The dataset studio for this history's own root.

        ``self.root`` defaults to the project's data folder, so this reads the same
        store the default :class:`DatasetStudio` would while honouring a caller's
        own root (the CLI's ``--root``, a test).
        """
        from ml.training.datasets import DatasetBuilder

        from core.training_center.datasets import DatasetStudio

        return DatasetStudio(builder=DatasetBuilder(root=self.root / "training" / "datasets"))

    def _dataset_lines(self) -> list[HistoryLine]:
        try:
            rows = self._studio().rows()
        except Exception:  # pragma: no cover - defensive
            return []
        return [
            HistoryLine(
                at=row.created_at, kind="dataset",
                summary=f"dataset {row.key}: {row.total} example(s)",
                outcome="eligible" if row.eligible else row.status, source="datasets",
            )
            for row in rows
        ]

    def _run_lines(self) -> list[HistoryLine]:
        try:
            from core.training_center.jobs import TrainingJobManager

            runs = TrainingJobManager(root=self.root / "training").coordinator.trainer.runs()
        except Exception:  # pragma: no cover - defensive
            return []
        return [
            HistoryLine(
                at=run.ended_at or run.started_at, kind="training",
                summary=f"training {run.run_id}: {run.method} on {run.dataset_id}@v{run.dataset_version}",
                outcome=run.status, source="runs",
            )
            for run in runs
        ]

    def _knowledge_lines(self) -> list[HistoryLine]:
        try:
            from core.knowledge_base import KnowledgeBaseStore

            store = KnowledgeBaseStore(self.root / "knowledge_base")
            manifest = store.load_manifest()
        except Exception:  # pragma: no cover - defensive
            return []
        if not manifest:
            return []
        at = str(manifest.get("updated_at", "") or manifest.get("created_at", "") or "")
        chunks = manifest.get("chunks", manifest.get("total_chunks", ""))
        summary = "knowledge base built"
        if chunks:
            summary += f": {chunks} chunk(s)"
        return [HistoryLine(at=at, kind="knowledge", summary=summary, source="knowledge_base")]

    def describe(self, *, limit: int = 30) -> str:
        lines = self.search(limit=limit)
        if not lines:
            return "No training-center history yet."
        return "\n".join(line.describe() for line in lines)

    def stats(self) -> dict[str, int]:
        lines = self.timeline()
        counts: dict[str, int] = {}
        for line in lines:
            counts[line.kind] = counts.get(line.kind, 0) + 1
        counts["total"] = len(lines)
        return counts


__all__ = [
    "DEFAULT_LIMIT",
    "KEY",
    "KINDS",
    "NAMESPACE",
    "HistoryLine",
    "TrainingHistory",
]
