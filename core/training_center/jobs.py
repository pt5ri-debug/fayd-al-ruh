"""The training-job manager: one place a run is started, watched, and stopped.

The controlled Phase-6 pipeline (:mod:`ml.training`) already owns everything a
training job *is*: the feasibility check, the run record, the cooperative cancel
flag, checkpoint resumption, and the base-model-preservation proof. This module
adds only what a *user interface* needs and the pipeline does not provide:

* one :class:`TrainingJobManager` that builds a model and dataset, runs the
  feasibility check, and starts the job on a **worker thread**, so a long run
  never freezes a window;
* a small :class:`JobHandle` with the cancellation the interface exposes;
* a :class:`JobView` that renders the run log as rows a table can draw.

Nothing here trains on its own and nothing here touches a model weight: every
start goes through :meth:`ml.training.coordinator.TrainingCoordinator.start`,
which is the only path that may. The manager is dependency-free and Tk-free; a
window drains its progress callbacks on its own clock, and a test drives it with
a fake coordinator, so the whole life cycle is exercised with no weights and no
long run.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml.training.config import TrainingConfig, resolve_preset
from ml.training.datasets import DatasetValidationError
from ml.training.trainer import RUN_CANCELLED, RUN_COMPLETED, RUN_FAILED, RUN_RUNNING, TrainingJob

#: The job states the interface shows, mapped from the pipeline's own run status.
STATE_QUEUED = "queued"
STATE_PREPARING = "preparing"
STATE_RUNNING = "running"
STATE_PAUSED = "paused"
STATE_COMPLETED = "completed"
STATE_FAILED = "failed"
STATE_CANCELLED = "cancelled"

JOB_STATES: tuple[str, ...] = (
    STATE_QUEUED,
    STATE_PREPARING,
    STATE_RUNNING,
    STATE_PAUSED,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_CANCELLED,
)

#: The pipeline's run statuses mapped onto the interface's job states.
_RUN_TO_JOB: dict[str, str] = {
    "pending": STATE_QUEUED,
    "running": STATE_RUNNING,
    RUN_RUNNING: STATE_RUNNING,
    RUN_COMPLETED: STATE_COMPLETED,
    RUN_FAILED: STATE_FAILED,
    RUN_CANCELLED: STATE_CANCELLED,
}


@dataclass(frozen=True)
class JobView:
    """One training run, as a table row the window and the CLI can draw."""

    run_id: str
    model_id: str
    dataset_id: str
    dataset_version: int
    method: str
    state: str
    progress: float
    started_at: str
    ended_at: str
    train_loss: float | None
    val_loss: float | None
    learning_rate: float | None
    steps: int
    adapter_path: str
    error: str

    def describe(self) -> str:
        percent = f"{self.progress * 100:.0f}%"
        loss = f"loss {self.train_loss:.4f}" if self.train_loss is not None else "loss —"
        return (
            f"{self.run_id} [{self.state}] {percent} · {self.method} · "
            f"{self.model_id} · {self.dataset_id}@v{self.dataset_version} · {loss}"
        )


@dataclass
class JobHandle:
    """A cancellable handle for one job running on a worker thread."""

    run_id: str
    job: TrainingJob
    thread: threading.Thread
    result: Any = None
    error: BaseException | None = None
    done: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        """Trip the cooperative cancel flag; the run stops at its next step."""
        self.job.cancel()

    @property
    def finished(self) -> bool:
        return self.done.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self.done.wait(timeout)


class TrainingJobManager:
    """Start, watch, and stop training jobs over the controlled pipeline."""

    def __init__(self, coordinator: Any = None, *, root: Any = None) -> None:
        self._coordinator = coordinator
        self.root = root
        self._handles: dict[str, JobHandle] = {}
        self._lock = threading.Lock()

    @property
    def coordinator(self) -> Any:
        """The Phase-6 coordinator, built lazily so importing this is free."""
        if self._coordinator is None:
            from ml.training.coordinator import TrainingCoordinator

            self._coordinator = TrainingCoordinator(root=self.root) if self.root else TrainingCoordinator()
        return self._coordinator

    # -- reading -----------------------------------------------------------
    def rows(self) -> tuple[JobView, ...]:
        """Every recorded run, newest first, as interface rows."""
        runs = list(self.coordinator.trainer.runs())
        runs.sort(key=lambda run: run.started_at, reverse=True)
        return tuple(self._view(run) for run in runs)

    def get(self, run_id: str) -> JobView | None:
        run = self.coordinator.trainer.get(run_id)
        return self._view(run) if run is not None else None

    def _view(self, run: Any) -> JobView:
        metrics = dict(getattr(run, "train_metrics", {}) or {})
        steps = int(metrics.get("steps", 0) or 0)
        max_steps = int(dict(getattr(run, "config", {}) or {}).get("max_steps", 0) or 0)
        progress = (steps / max_steps) if max_steps else (1.0 if run.finished else 0.0)
        return JobView(
            run_id=run.run_id,
            model_id=run.model_id,
            dataset_id=run.dataset_id,
            dataset_version=run.dataset_version,
            method=run.method,
            state=_RUN_TO_JOB.get(run.status, run.status),
            progress=min(1.0, max(0.0, progress)),
            started_at=run.started_at,
            ended_at=run.ended_at,
            train_loss=metrics.get("train_loss"),
            val_loss=metrics.get("val_loss"),
            learning_rate=metrics.get("learning_rate"),
            steps=steps,
            adapter_path=getattr(run, "adapter_path", "") or "",
            error=getattr(run, "error", "") or "",
        )

    # -- feasibility -------------------------------------------------------
    def feasibility(
        self,
        model: Any,
        dataset_id: str,
        config: TrainingConfig,
        *,
        version: int | None = None,
    ) -> Any:
        """The pre-run feasibility report, without starting anything."""
        dataset = (
            self.coordinator.datasets.load(dataset_id, version)
            if version is not None
            else self.coordinator.datasets.latest(dataset_id)
        )
        return self.coordinator.feasibility(model, dataset, config)

    # -- starting ----------------------------------------------------------
    def build_config(
        self,
        *,
        model_id: str,
        dataset_id: str,
        dataset_version: int,
        preset: str = "cpu-safe",
        max_steps: int | None = None,
        rank: int | None = None,
        batch_size: int | None = None,
        method: str | None = None,
        owner: str = "",
    ) -> TrainingConfig:
        """A validated configuration from a preset plus named overrides."""
        values: dict[str, Any] = {
            "base_model_id": model_id,
            "dataset_id": dataset_id,
            "dataset_version": dataset_version,
            "owner": owner,
        }
        if preset:
            values.update(resolve_preset(preset))
        if max_steps is not None:
            values["max_steps"] = max_steps
        if rank is not None:
            values["rank"] = rank
            values["alpha"] = rank * 2
        if batch_size is not None:
            values["batch_size"] = batch_size
        if method:
            values["method"] = method
        config = TrainingConfig(**values)
        config.validate()
        return config

    def start(
        self,
        model: Any,
        dataset_id: str,
        config: TrainingConfig,
        *,
        tokenizer: Any = None,
        resume_from: str | Path | None = None,
        on_step: Callable[[int, float], None] | None = None,
        on_done: Callable[[Any], None] | None = None,
        require_feasible: bool = True,
        version: int | None = None,
    ) -> JobHandle:
        """Start one governed training run on a worker thread.

        The feasibility check runs **first, on this thread**, so an impossible run
        is refused before a worker is ever started. The job then runs on a worker
        thread and its result is handed to ``on_done`` (a window's callback), so
        the interface thread is never blocked.
        """
        dataset = (
            self.coordinator.datasets.load(dataset_id, version)
            if version is not None
            else self.coordinator.datasets.latest(dataset_id)
        )
        if dataset is None:
            raise DatasetValidationError(f"No dataset {dataset_id!r} was found.")
        report = self.coordinator.feasibility(model, dataset, config)
        if require_feasible and not report.ok:
            from ml.training.trainer import TrainingError

            raise TrainingError("Training refused: " + "; ".join(report.blockers or ["not feasible"]))
        if tokenizer is None:
            from ml.llm.tokenizer import ByteTokenizer

            tokenizer = ByteTokenizer()

        # The run id is generated by the pipeline; pre-read the trainer's own naming
        # so the handle can be registered before the thread starts (a job that
        # finishes instantly must still be findable) and the cancellable job knows
        # its own id.
        from ml.training.trainer import _run_id as make_run_id

        run_id = config.run_id or make_run_id(config.base_model_id, config.dataset_id, config.dataset_version)
        job = TrainingJob(run_id=run_id)
        run_id_holder: dict[str, str] = {"id": run_id}

        def _run() -> None:
            handle = None
            with self._lock:
                handle = self._handles.get(run_id_holder["id"])
            try:
                result = self.coordinator.start(
                    model, dataset, config, tokenizer=tokenizer,
                    on_step=on_step, job=job, resume_from=resume_from,
                    require_feasible=require_feasible,
                )
                if handle is not None:
                    handle.result = result
            except BaseException as error:
                if handle is not None:
                    handle.error = error
            finally:
                if handle is not None:
                    handle.done.set()
                if on_done is not None:
                    on_done(handle)

        thread = threading.Thread(target=_run, name=f"training-{run_id}", daemon=True)
        handle = JobHandle(run_id=run_id, job=job, thread=thread)
        with self._lock:
            self._handles[run_id] = handle
        thread.start()
        return handle

    def start_and_wait(
        self,
        model: Any,
        dataset_id: str,
        config: TrainingConfig,
        *,
        tokenizer: Any = None,
        resume_from: str | Path | None = None,
        on_step: Callable[[int, float], None] | None = None,
        version: int | None = None,
    ) -> JobHandle:
        """Start a run and block until it finishes (for the CLI, never a window)."""
        handle = self.start(
            model, dataset_id, config, tokenizer=tokenizer, resume_from=resume_from,
            on_step=on_step, version=version,
        )
        handle.wait()
        return handle

    def cancel(self, run_id: str) -> bool:
        """Cancel a running job by id; returns whether a live handle was found."""
        with self._lock:
            handle = self._handles.get(run_id)
        if handle is None:
            return False
        handle.cancel()
        return True

    def handles(self) -> tuple[JobHandle, ...]:
        with self._lock:
            return tuple(self._handles.values())

    def active(self) -> tuple[JobHandle, ...]:
        """Handles whose thread has not finished yet."""
        return tuple(handle for handle in self.handles() if not handle.finished)

    def resume_from_adapter(self, run_id: str) -> str | None:
        """The adapter directory a completed run left, for a resume."""
        run = self.coordinator.trainer.get(run_id)
        if run is None or not run.finished:
            return None
        adapter = getattr(run, "adapter_path", "") or ""
        return adapter or None

    def stats(self) -> dict[str, Any]:
        rows = self.rows()
        return {
            "jobs": len(rows),
            "running": sum(1 for row in rows if row.state == STATE_RUNNING),
            "completed": sum(1 for row in rows if row.state == STATE_COMPLETED),
            "failed": sum(1 for row in rows if row.state == STATE_FAILED),
            "cancelled": sum(1 for row in rows if row.state == STATE_CANCELLED),
            "adapters": len(self.coordinator.registry.adapters()),
            "active_adapter": self.coordinator.registry.active_id(),
        }


__all__ = [
    "JOB_STATES",
    "STATE_CANCELLED",
    "STATE_COMPLETED",
    "STATE_FAILED",
    "STATE_PAUSED",
    "STATE_PREPARING",
    "STATE_QUEUED",
    "STATE_RUNNING",
    "JobHandle",
    "JobView",
    "TrainingJobManager",
]
