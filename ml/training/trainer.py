"""The adapter trainer: one training job's whole life cycle, recorded.

A training run is a **job** with a record: it is started, may be cancelled, may be
resumed from a checkpoint, and always ends with a :class:`TrainingRun` that carries
every field the task requires -- the run id, the model and dataset with their
versions, the configuration, the seed, the software and hardware, the start and end
time, the training and validation metrics, the resource consumption, the output
adapter path, and the status (with the error when it failed).

Hard guarantees, enforced here rather than documented:

* the **base model file is never written** -- the trainer records the base's digest
  before and after and marks the run failed if it changed;
* the **adapter is written to its own directory**, never over an existing one (a
  second run with the same id is refused);
* a **failed run leaves the source data alone** and writes no adapter;
* a **cancelled run** stops at the next step, records what it did, and is not
  reported as finished.

The trainer drives an :class:`~ml.training.adapters.AdapterBackend`, so a test
injects a tiny backend and exercises the whole life cycle with no model.
"""

from __future__ import annotations

import json
import platform
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.paths import training_adapters_dir, training_runs_dir

from ml.training.adapters import AdapterBackend, AdapterError, AdapterResult, backend_for
from ml.training.config import TrainingConfig
from ml.training.datasets import BuiltDataset
from ml.training.resources import ResourceMonitor

RUN_PENDING = "pending"
RUN_RUNNING = "running"
RUN_COMPLETED = "completed"
RUN_FAILED = "failed"
RUN_CANCELLED = "cancelled"

TERMINAL_STATES = (RUN_COMPLETED, RUN_FAILED, RUN_CANCELLED)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _run_id(model_id: str, dataset_id: str, version: int) -> str:
    # Microseconds keep the id unique when two runs start in the same second --
    # which is exactly what a resume-from run does, and a collision here made
    # the trainer refuse to run ("the run id ... already exists").
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    safe_model = "".join(c if c.isalnum() or c in "-_" else "-" for c in model_id)[:24]
    return f"{stamp}-{safe_model or 'model'}-{dataset_id}-v{version}"


@dataclass
class TrainingRun:
    """The durable record of one training run."""

    run_id: str
    model_id: str = ""
    model_version: str = ""
    dataset_id: str = ""
    dataset_version: int = 0
    method: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    software: dict[str, str] = field(default_factory=dict)
    hardware: dict[str, str] = field(default_factory=dict)
    started_at: str = ""
    ended_at: str = ""
    train_metrics: dict[str, float] = field(default_factory=dict)
    validation_metrics: dict[str, float] = field(default_factory=dict)
    resources: dict[str, str] = field(default_factory=dict)
    adapter_path: str = ""
    checkpoint_path: str = ""
    status: str = RUN_PENDING
    error: str = ""
    cancelled: bool = False
    resumed_from: str = ""
    owner: str = ""

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL_STATES

    def to_primitive(self) -> dict[str, Any]:
        return {
            "adapter_path": self.adapter_path,
            "cancelled": self.cancelled,
            "checkpoint_path": self.checkpoint_path,
            "config": self.config,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "ended_at": self.ended_at,
            "error": self.error,
            "hardware": self.hardware,
            "method": self.method,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "owner": self.owner,
            "resources": self.resources,
            "resumed_from": self.resumed_from,
            "run_id": self.run_id,
            "seed": self.seed,
            "software": self.software,
            "started_at": self.started_at,
            "status": self.status,
            "train_metrics": self.train_metrics,
            "validation_metrics": self.validation_metrics,
        }

    @classmethod
    def from_primitive(cls, data: object) -> TrainingRun:
        if not isinstance(data, dict):
            raise ValueError("A training run must be a mapping.")
        run_id = str(data.get("run_id", "")).strip()
        if not run_id:
            raise ValueError("A training run needs a run_id.")
        return cls(
            run_id=run_id,
            model_id=str(data.get("model_id", "")),
            model_version=str(data.get("model_version", "")),
            dataset_id=str(data.get("dataset_id", "")),
            dataset_version=int(data.get("dataset_version", 0) or 0),
            method=str(data.get("method", "")),
            config=dict(data.get("config", {}) or {}),
            seed=int(data.get("seed", 0) or 0),
            software=dict(data.get("software", {}) or {}),
            hardware=dict(data.get("hardware", {}) or {}),
            started_at=str(data.get("started_at", "")),
            ended_at=str(data.get("ended_at", "")),
            train_metrics=dict(data.get("train_metrics", {}) or {}),
            validation_metrics=dict(data.get("validation_metrics", {}) or {}),
            resources=dict(data.get("resources", {}) or {}),
            adapter_path=str(data.get("adapter_path", "")),
            checkpoint_path=str(data.get("checkpoint_path", "")),
            status=str(data.get("status", RUN_PENDING)),
            error=str(data.get("error", "")),
            cancelled=bool(data.get("cancelled", False)),
            resumed_from=str(data.get("resumed_from", "")),
            owner=str(data.get("owner", "")),
        )

    def describe(self) -> str:
        when = self.ended_at or self.started_at
        text = (
            f"{self.run_id} [{self.status}] {self.method} · {self.model_id} · "
            f"{self.dataset_id}@v{self.dataset_version} · {when}"
        )
        if self.train_metrics.get("final_loss") is not None:
            text += f" · loss {self.train_metrics['final_loss']:.4f}"
        if self.error:
            text += f" · error: {self.error}"
        return text


@dataclass
class TrainingJob:
    """A cancellable handle for one run. ``cancel()`` trips the cooperative flag."""

    run_id: str
    _cancelled: bool = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def should_cancel(self) -> bool:
        return self._cancelled


@dataclass(frozen=True)
class TrainingResult:
    """What a training call returned: the run record and the adapter's result."""

    run: TrainingRun
    result: AdapterResult | None = None
    adapter_dir: Path | None = None

    @property
    def ok(self) -> bool:
        return self.run.status == RUN_COMPLETED

    def describe(self) -> str:
        text = self.run.describe()
        if self.result is not None:
            text += f"\n  {self.result.describe()}"
        if self.adapter_dir is not None:
            text += f"\n  adapter: {self.adapter_dir}"
        return text


class TrainingError(RuntimeError):
    """A training run could not be started or completed."""


class AdapterTrainer:
    """Drive one adapter training run and record everything it did.

    The trainer is deliberately model-agnostic: it takes a base model object, a
    built dataset, and a validated config, and it uses the adapter backend to do
    the arithmetic. ``model_factory`` and ``backend_factory`` are the injection
    seams, so a test drives a fake model and a fake backend.
    """

    def __init__(
        self,
        *,
        runs_dir: Path | str | None = None,
        adapters_dir: Path | str | None = None,
        backend_factory: Callable[..., AdapterBackend] = backend_for,
        model_factory: Callable[[Any], Any] | None = None,
        monitor_factory: Callable[[], ResourceMonitor] | None = None,
        disk_path: Path | str | None = None,
    ) -> None:
        self.runs_dir = Path(runs_dir) if runs_dir is not None else training_runs_dir()
        self.adapters_dir = (
            Path(adapters_dir) if adapters_dir is not None else training_adapters_dir()
        )
        self.backend_factory = backend_factory
        self.model_factory = model_factory
        self.monitor_factory = monitor_factory or ResourceMonitor
        self.disk_path = Path(disk_path) if disk_path is not None else self.runs_dir

    # -- the run -----------------------------------------------------------
    def train(
        self,
        model: Any,
        dataset: BuiltDataset,
        config: TrainingConfig,
        *,
        tokenizer: Any,
        on_step: Callable[[int, float], None] | None = None,
        job: TrainingJob | None = None,
        resume_from: Path | str | None = None,
        model_version: str = "",
        owner: str = "",
    ) -> TrainingResult:
        """Run one training job, recording a :class:`TrainingRun` either way."""
        config.validate()
        if not dataset.eligible:
            raise TrainingError(
                f"{dataset.record.key} is not eligible for training: "
                f"{dataset.record.status}"
            )
        run = TrainingRun(
            run_id=config.run_id or _run_id(config.base_model_id, config.dataset_id, config.dataset_version),
            model_id=config.base_model_id,
            model_version=model_version,
            dataset_id=config.dataset_id,
            dataset_version=config.dataset_version,
            method=config.method,
            config=config.to_primitive(),
            seed=config.seed,
            software=_software_versions(),
            hardware=_hardware_summary(),
            started_at=_now(),
            status=RUN_RUNNING,
            owner=owner or config.owner,
            resumed_from=str(resume_from) if resume_from else "",
        )
        run_dir = self.runs_dir / run.run_id
        if run_dir.exists():
            raise TrainingError(
                f"The run id {run.run_id} already exists; refusing to overwrite a run."
            )
        run_dir.mkdir(parents=True, exist_ok=False)
        adapter_dir = self.adapters_dir / run.run_id

        monitor = self.monitor_factory()
        monitor.start()
        base_digest_before = _model_digest(model)
        backend: AdapterBackend | None = None
        result: AdapterResult | None = None
        try:
            backend = self.backend_factory(
                method=config.method,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
                seed=config.seed,
            )
            self._bind(backend, model)
            if resume_from is not None:
                backend.load(Path(resume_from))
                run.resumed_from = str(resume_from)
            else:
                self._attach(backend, model)
            texts = [example.text for example in dataset.split.train]
            if not texts:
                raise TrainingError("The dataset's training split is empty.")

            def _step(step: int, loss: float) -> None:
                monitor.sample()
                if on_step is not None:
                    on_step(step, loss)
                if step % config.checkpoint_every == 0:
                    self._checkpoint(backend, run_dir, step)

            result = backend.train(
                texts,
                config=config,
                tokenizer=tokenizer,
                on_step=_step,
                should_cancel=job.should_cancel if job is not None else None,
            )
            # The base must be exactly as it was; a changed base is a bug.
            if _model_digest(model) != base_digest_before:
                raise TrainingError(
                    "The base model changed during training; an adapter must leave "
                    "the base untouched."
                )
            if result.cancelled:
                run.status = RUN_CANCELLED
                run.cancelled = True
            else:
                run.status = RUN_COMPLETED
                adapter_dir.mkdir(parents=True, exist_ok=True)
                backend.save(adapter_dir)
                run.adapter_path = str(adapter_dir)
            run.train_metrics = {
                "initial_loss": result.initial_loss,
                "final_loss": result.final_loss,
                "improvement": result.improvement,
                "steps": float(result.steps),
                "trainable_parameters": float(result.trainable_parameters),
            }
            run.validation_metrics = self._validation_loss(backend, model, dataset, tokenizer)
        except (AdapterError, TrainingError, ValueError) as exc:
            run.status = RUN_FAILED
            run.error = str(exc)
            # A failed run leaves no half-written adapter behind.
            if adapter_dir.exists():
                shutil.rmtree(adapter_dir, ignore_errors=True)
        except Exception as exc:  # pragma: no cover - unexpected, still recorded
            run.status = RUN_FAILED
            run.error = f"{type(exc).__name__}: {exc}"
            if adapter_dir.exists():
                shutil.rmtree(adapter_dir, ignore_errors=True)
        finally:
            monitor.sample()
            run.ended_at = _now()
            run.resources = {
                "peak_memory": str(monitor.peak_bytes) if monitor.peak_bytes is not None else "unknown",
                "grew_by": str(monitor.grew_by) if monitor.grew_by is not None else "unknown",
                "samples": str(len(monitor.samples)),
            }
            self._write_run(run_dir, run)

        return TrainingResult(run=run, result=result, adapter_dir=adapter_dir if run.adapter_path else None)

    # -- helpers -----------------------------------------------------------
    def _bind(self, backend: AdapterBackend, model: Any) -> None:
        if hasattr(backend, "bind"):
            backend.bind(model)

    def _attach(self, backend: AdapterBackend, model: Any) -> None:
        if hasattr(backend, "attach"):
            backend.attach(model)

    def _checkpoint(self, backend: AdapterBackend, run_dir: Path, step: int) -> None:
        try:
            backend.save(run_dir / "checkpoint")
        except (AdapterError, OSError):
            # A checkpoint is best-effort; its failure never fails the run.
            pass

    def _validation_loss(
        self,
        backend: AdapterBackend,
        model: Any,
        dataset: BuiltDataset,
        tokenizer: Any,
    ) -> dict[str, float]:
        """The held-out validation loss, measured with the adapter applied."""
        texts = [example.text for example in dataset.split.validation]
        if not texts or tokenizer is None:
            return {}
        try:
            from ml.training.evaluation import measure_loss

            loss = measure_loss(backend, model, texts, tokenizer)
            return {"loss": float(loss)} if loss is not None else {}
        except Exception:  # pragma: no cover - validation is best-effort
            return {}

    def _write_run(self, run_dir: Path, run: TrainingRun) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(
            json.dumps(run.to_primitive(), indent=2, sort_keys=True), encoding="utf-8"
        )

    # -- reading -----------------------------------------------------------
    def runs(self) -> tuple[TrainingRun, ...]:
        if not self.runs_dir.is_dir():
            return ()
        out: list[TrainingRun] = []
        for folder in sorted(self.runs_dir.iterdir()):
            path = folder / "run.json"
            if not path.is_file():
                continue
            try:
                out.append(TrainingRun.from_primitive(json.loads(path.read_text(encoding="utf-8"))))
            except (ValueError, OSError):
                continue
        return tuple(out)

    def get(self, run_id: str) -> TrainingRun | None:
        path = self.runs_dir / run_id / "run.json"
        if not path.is_file():
            return None
        try:
            return TrainingRun.from_primitive(json.loads(path.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            return None

    def cleanup(self, run_id: str, *, remove_adapter: bool = True) -> bool:
        """Remove a failed or cancelled run's artifacts; source data is untouched."""
        run = self.get(run_id)
        if run is None:
            return False
        if run.status == RUN_COMPLETED:
            raise TrainingError(
                f"{run_id} completed; a completed run is never deleted automatically."
            )
        shutil.rmtree(self.runs_dir / run_id, ignore_errors=True)
        if remove_adapter:
            shutil.rmtree(self.adapters_dir / run_id, ignore_errors=True)
        return True


def _model_digest(model: Any) -> str:
    """A stable digest of a model's parameters, so an unchanged base is provable."""
    import hashlib

    import numpy as np

    hasher = hashlib.blake2b(digest_size=16)
    for array in model.parameters():
        hasher.update(np.ascontiguousarray(array, dtype=np.float32).tobytes())
    return hasher.hexdigest()


def _software_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import numpy

        versions["numpy"] = numpy.__version__
    except Exception:  # pragma: no cover - numpy is a base dependency
        versions["numpy"] = "unknown"
    try:
        from core.optional_deps import FEATURE_TRAINING, status

        versions["training_extra"] = "installed" if status(FEATURE_TRAINING).python_ok else "absent"
    except Exception:  # pragma: no cover - defensive
        versions["training_extra"] = "unknown"
    return versions


def _hardware_summary() -> dict[str, str]:
    try:
        from core.model_resources import system_resources

        resources = system_resources()
        return {
            "cpu_count": str(resources.cpu_count or "unknown"),
            "total_memory_bytes": str(resources.total_memory_bytes or "unknown"),
            "gpu": resources.gpu.describe() if hasattr(resources.gpu, "describe") else "unknown",
            "gpu_available": str(bool(getattr(resources.gpu, "available", False))),
        }
    except Exception:  # pragma: no cover - defensive
        return {"cpu_count": "unknown", "gpu": "unknown"}


__all__ = [
    "RUN_CANCELLED",
    "RUN_COMPLETED",
    "RUN_FAILED",
    "RUN_PENDING",
    "RUN_RUNNING",
    "TERMINAL_STATES",
    "AdapterTrainer",
    "TrainingError",
    "TrainingJob",
    "TrainingResult",
    "TrainingRun",
]
