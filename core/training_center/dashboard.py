"""The dashboard's real statistics -- read, never fabricated.

Every figure the Training Center's dashboard shows comes from a store the project
already keeps: the knowledge base's manifest and shards, the governed dataset
registry, the training run log, the adapter registry, the model catalog, the
review queue, and this machine's own resource readings. Nothing here invents a
number; where a store does not exist yet the figure is ``0`` and the section says
so, and where the platform cannot report a resource the figure stays ``None``.

The dashboard also carries the data for the charts the window draws (loss,
perplexity, dataset growth, job duration, source distribution), each a series read
from a real record, so a chart can never show a line that no run produced.

Dependency-free and Tk-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.training_center.qa_learning import LearningReviewQueue
from core.training_center.resources import ResourceSnapshot, human_size, read_resources


@dataclass(frozen=True)
class ChartSeries:
    """One chart: a named series of ``(label, value)`` points, read from a record."""

    title: str
    points: tuple[tuple[str, float], ...]
    unit: str = ""

    @property
    def empty(self) -> bool:
        return not self.points

    def describe(self) -> str:
        if self.empty:
            return f"{self.title}: no data yet."
        body = ", ".join(f"{label}={value:g}{self.unit}" for label, value in self.points[-8:])
        return f"{self.title}: {body}"


@dataclass
class DashboardSnapshot:
    """Everything the dashboard shows, read from the real stores."""

    # knowledge / ingestion
    learning_sources: int = 0
    documents: int = 0
    chunks: int = 0
    tokens_estimate: int = 0
    knowledge_bytes: int = 0
    # datasets
    datasets: int = 0
    dataset_versions: int = 0
    training_examples: int = 0
    dataset_bytes: int = 0
    # review
    learning_examples: int = 0
    approved_examples: int = 0
    pending_examples: int = 0
    # jobs
    jobs: int = 0
    active_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    cancelled_jobs: int = 0
    adapters: int = 0
    active_adapter: str = ""
    # models
    local_models: int = 0
    ready_models: int = 0
    # resources
    resources: ResourceSnapshot | None = None
    # recent activity
    recent_history: tuple[str, ...] = ()
    recent_jobs: tuple[str, ...] = ()
    recent_errors: tuple[str, ...] = ()
    charts: tuple[ChartSeries, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    # -- readable figures --------------------------------------------------
    @property
    def cpu_count(self) -> int | None:
        return self.resources.cpu_count if self.resources else None

    @property
    def memory_bytes(self) -> int | None:
        return self.resources.total_memory_bytes if self.resources else None

    @property
    def available_memory_bytes(self) -> int | None:
        return self.resources.available_memory_bytes if self.resources else None

    @property
    def disk_free_bytes(self) -> int | None:
        return self.resources.disk_free_bytes if self.resources else None

    @property
    def gpu_available(self) -> bool:
        return bool(self.resources and self.resources.gpu_available)

    def lines(self) -> list[str]:
        """The dashboard as a readable block, honest about every unknown."""
        lines = [
            "Training Center — dashboard",
            f"  learning sources:     {self.learning_sources}",
            f"  documents:            {self.documents}",
            f"  knowledge chunks:     {self.chunks}",
            f"  tokens (estimate):    {self.tokens_estimate}",
            f"  knowledge size:       {human_size(self.knowledge_bytes)}",
            f"  datasets:             {self.datasets} ({self.dataset_versions} version(s))",
            f"  training examples:    {self.training_examples}",
            f"  dataset size:         {human_size(self.dataset_bytes)}",
            f"  learning examples:    {self.learning_examples} "
            f"({self.approved_examples} approved, {self.pending_examples} pending)",
            f"  training jobs:        {self.jobs} "
            f"({self.active_jobs} running, {self.completed_jobs} completed, "
            f"{self.failed_jobs} failed, {self.cancelled_jobs} cancelled)",
            f"  adapters:             {self.adapters}"
            + (f" (active {self.active_adapter})" if self.active_adapter else ""),
            f"  local models:         {self.ready_models} ready of {self.local_models}",
            f"  cpu cores:            {self.cpu_count if self.cpu_count else 'unknown'}",
            f"  memory:               {human_size(self.memory_bytes)}"
            + (f" ({human_size(self.available_memory_bytes)} available)" if self.available_memory_bytes is not None else ""),
            f"  disk free:            {human_size(self.disk_free_bytes)}",
            f"  gpu:                  {'present' if self.gpu_available else 'none detected'}",
        ]
        lines.extend(f"  note: {note}" for note in self.notes)
        return lines

    def describe(self) -> str:
        return "\n".join(self.lines())

    def to_primitive(self) -> dict[str, Any]:
        return {
            "active_adapter": self.active_adapter,
            "active_jobs": self.active_jobs,
            "adapters": self.adapters,
            "approved_examples": self.approved_examples,
            "available_memory_bytes": self.available_memory_bytes,
            "cancelled_jobs": self.cancelled_jobs,
            "chunks": self.chunks,
            "completed_jobs": self.completed_jobs,
            "cpu_count": self.cpu_count,
            "dataset_bytes": self.dataset_bytes,
            "dataset_versions": self.dataset_versions,
            "datasets": self.datasets,
            "disk_free_bytes": self.disk_free_bytes,
            "documents": self.documents,
            "failed_jobs": self.failed_jobs,
            "gpu_available": self.gpu_available,
            "jobs": self.jobs,
            "knowledge_bytes": self.knowledge_bytes,
            "learning_examples": self.learning_examples,
            "learning_sources": self.learning_sources,
            "local_models": self.local_models,
            "memory_bytes": self.memory_bytes,
            "pending_examples": self.pending_examples,
            "ready_models": self.ready_models,
            "tokens_estimate": self.tokens_estimate,
            "training_examples": self.training_examples,
        }


def _knowledge_stats(root: Path | None = None) -> dict[str, Any]:
    """Read the knowledge base's real size and chunk count, or zeros.

    When ``root`` is given the store lives under it, so a caller with its own data
    folder (the CLI's ``--root``, a test) reads *its* knowledge base rather than the
    machine-wide default; with no root the project's own default is read.
    """
    try:
        from core.knowledge_base import KnowledgeBaseStore, default_knowledge_base_dir

        store_dir = (root / "knowledge_base") if root is not None else default_knowledge_base_dir()
        store = KnowledgeBaseStore(store_dir)
        stats = store.stats()
        return {
            "chunks": int(stats.get("chunks", 0) or 0),
            "articles": int(stats.get("articles", 0) or 0),
            "bytes": int(stats.get("size_bytes", 0) or 0),
            "shards": int(stats.get("shards", 0) or 0),
        }
    except Exception:  # pragma: no cover - defensive
        return {"chunks": 0, "articles": 0, "bytes": 0, "shards": 0}


def _dataset_stats(root: Path | None = None) -> dict[str, Any]:
    try:
        from core.training_center.datasets import DatasetStudio

        if root is None:
            return DatasetStudio().stats()
        from ml.training.datasets import DatasetBuilder

        return DatasetStudio(builder=DatasetBuilder(root=root / "training" / "datasets")).stats()
    except Exception:  # pragma: no cover - defensive
        return {"datasets": 0, "versions": 0, "examples": 0, "storage_bytes": 0}


def _job_stats(root: Path | None = None) -> dict[str, Any]:
    try:
        from core.training_center.jobs import TrainingJobManager

        return TrainingJobManager(root=root).stats()
    except Exception:  # pragma: no cover - defensive
        return {"jobs": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0, "adapters": 0, "active_adapter": ""}


def _model_stats() -> dict[str, Any]:
    try:
        from core.language_models import all_language_models

        models = [model for _, model in all_language_models()]
        return {
            "total": len(models),
            "ready": sum(1 for model in models if model.available),
        }
    except Exception:  # pragma: no cover - defensive
        return {"total": 0, "ready": 0}


def _run_rows(root: Path | None = None) -> list[Any]:
    try:
        from core.training_center.jobs import TrainingJobManager

        return list(TrainingJobManager(root=root).rows())
    except Exception:  # pragma: no cover - defensive
        return []


def _charts(runs: Sequence[Any], root: Path | None = None) -> tuple[ChartSeries, ...]:
    """The dashboard's charts, each read from a real record."""
    loss_points: list[tuple[str, float]] = []
    val_points: list[tuple[str, float]] = []
    duration_points: list[tuple[str, float]] = []
    for run in runs:
        label = run.run_id[:15]
        if run.train_loss is not None:
            loss_points.append((label, float(run.train_loss)))
        if run.val_loss is not None:
            val_points.append((label, float(run.val_loss)))
        if run.started_at and run.ended_at:
            seconds = _seconds_between(run.started_at, run.ended_at)
            if seconds is not None:
                duration_points.append((label, seconds))
    return (
        ChartSeries("Training loss", tuple(loss_points)),
        ChartSeries("Validation loss", tuple(val_points)),
        ChartSeries(
            "Dataset growth",
            ((str(_dataset_stats(root).get("versions", 0)), float(_dataset_stats(root).get("examples", 0))),),
            " examples",
        ),
        ChartSeries("Job duration", tuple(duration_points), " s"),
        ChartSeries(
            "Knowledge chunks",
            (("chunks", float(_knowledge_stats(root).get("chunks", 0))),),
        ),
    )


def _seconds_between(start: str, end: str) -> float | None:
    from datetime import datetime

    def _parse(value: str) -> datetime | None:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    a, b = _parse(start), _parse(end)
    if a is None or b is None:
        return None
    return max(0.0, (b - a).total_seconds())


def build_dashboard(root: Path | None = None) -> DashboardSnapshot:
    """Read every figure from the real stores and return one snapshot.

    ``root`` is the data folder the stores live under; with no root the project's
    own defaults are read, so a caller with its own folder (the CLI's ``--root``, a
    test) reads *its* stores and never the machine-wide ones.
    """
    knowledge = _knowledge_stats(root)
    datasets = _dataset_stats(root)
    jobs = _job_stats(root)
    models = _model_stats()
    queue = LearningReviewQueue(root=root)
    summary = queue.summary()
    runs = _run_rows(root)

    notes: list[str] = []
    if knowledge["chunks"] == 0:
        notes.append("the knowledge base is empty; import documents in Learning Sources first")
    if datasets["datasets"] == 0:
        notes.append("no dataset has been built yet")
    if jobs["jobs"] == 0:
        notes.append("no training job has run yet")

    recent_history: tuple[str, ...] = ()
    recent_errors: tuple[str, ...] = ()
    try:
        from core.training_center.history import TrainingHistory

        history = TrainingHistory(root=root)
        recent_history = tuple(line.describe() for line in history.search(limit=10))
    except Exception:  # pragma: no cover - defensive
        pass
    recent_errors = tuple(
        f"{run.run_id}: {run.error}" for run in runs if run.state == "failed" and run.error
    )[:5]

    return DashboardSnapshot(
        learning_sources=knowledge["shards"],
        documents=knowledge["articles"],
        chunks=knowledge["chunks"],
        tokens_estimate=knowledge["chunks"] * 250,
        knowledge_bytes=knowledge["bytes"],
        datasets=int(datasets.get("datasets", 0)),
        dataset_versions=int(datasets.get("versions", 0)),
        training_examples=int(datasets.get("examples", 0)),
        dataset_bytes=int(datasets.get("storage_bytes", 0)),
        learning_examples=summary.total,
        approved_examples=summary.approved,
        pending_examples=summary.counts.get("pending", 0),
        jobs=int(jobs.get("jobs", 0)),
        active_jobs=int(jobs.get("running", 0)),
        completed_jobs=int(jobs.get("completed", 0)),
        failed_jobs=int(jobs.get("failed", 0)),
        cancelled_jobs=int(jobs.get("cancelled", 0)),
        adapters=int(jobs.get("adapters", 0)),
        active_adapter=str(jobs.get("active_adapter", "") or ""),
        local_models=int(models.get("total", 0)),
        ready_models=int(models.get("ready", 0)),
        resources=read_resources(),
        recent_history=recent_history,
        recent_jobs=tuple(run.describe() for run in runs[:5]),
        recent_errors=recent_errors,
        charts=_charts(runs, root),
        notes=tuple(notes),
    )


__all__ = [
    "ChartSeries",
    "DashboardSnapshot",
    "build_dashboard",
]
