"""Model versioning and rollback for the project's own trained model.

The task is explicit: never modify the base model in place, keep every training
run, keep a candidate until it is accepted, and always be able to return to the
last stable version. This module is that bookkeeping.

On disk (all under the git-ignored data directory)::

    data/cognition/models/
        base.npz              the immutable base model (a copy, never written again)
        versions/v1.npz ...   every accepted version
        candidate.npz         the version being evaluated
        registry.json         the log: dataset, settings, results, date, parent
        active.json           which version is live

A :class:`ModelVersion` is a small frozen record; :class:`ModelRegistry` reads and
writes the log atomically through the project's own :class:`storage.json_store.JsonStore`.

Nothing here trains anything -- it only stores, compares, promotes, and rolls back.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from core.paths import data_dir
from storage.json_store import JsonStore

#: The model-store folder under the data directory.
_NAMESPACE = "cognition/model_registry"
_KEY = "versions"
_ACTIVE_KEY = "active"


def models_root() -> Path:
    """The folder holding the base model, the versions, and the candidate."""
    return data_dir() / "cognition" / "models"


def base_path() -> Path:
    return models_root() / "base.npz"


def candidate_path() -> Path:
    return models_root() / "candidate.npz"


def version_path(number: int) -> Path:
    return models_root() / "versions" / f"v{number}.npz"


def _as_int(value: object, fallback: int = 0) -> int:
    """Read a stored integer, falling back rather than raising on bad user data."""
    if isinstance(value, bool) or not isinstance(value, int):
        return fallback
    return int(value)


def _as_float(value: object, fallback: float = 0.0) -> float:
    """Read a stored float, falling back rather than raising on bad user data."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return float(value)


def _as_dict(value: object) -> dict[str, object]:
    """Read a stored settings object, or an empty one when it is not a mapping."""
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    return {}


@dataclass(frozen=True)
class Evaluation:
    """One measured result, before and after, for a training run.

    ``lower_is_better`` exists because the project's own yardstick -- held-out
    perplexity -- improves when it *falls*. The :attr:`delta` is always expressed
    so that a **positive value means an improvement**, whichever direction the
    raw metric moves, so the acceptance gate never has to know which metric it is
    reading.
    """

    metric: str
    before: float
    after: float
    lower_is_better: bool = False

    @property
    def delta(self) -> float:
        raw = self.before - self.after if self.lower_is_better else self.after - self.before
        return round(raw, 6)

    @property
    def improved(self) -> bool:
        return self.delta > 0

    def describe(self) -> str:
        sign = "+" if self.delta >= 0 else ""
        return f"{self.metric}: {self.before:.4f} -> {self.after:.4f} ({sign}{self.delta:.4f})"


@dataclass(frozen=True)
class ModelVersion:
    """The record of one training run and the version it produced."""

    number: int
    path: str
    parent: int
    dataset: str
    examples: int
    steps: int
    settings: dict[str, object] = field(default_factory=dict)
    evaluations: tuple[Evaluation, ...] = ()
    accepted: bool = False
    reason: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    id: str = ""

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError("A version number must be at least 1.")
        if not self.id:
            object.__setattr__(self, "id", f"v{self.number}")

    def to_primitive(self) -> dict[str, object]:
        return {
            "id": self.id,
            "number": self.number,
            "path": self.path,
            "parent": self.parent,
            "dataset": self.dataset,
            "examples": self.examples,
            "steps": self.steps,
            "settings": dict(self.settings),
            "evaluations": [
                {
                    "metric": e.metric,
                    "before": e.before,
                    "after": e.after,
                    "lower_is_better": e.lower_is_better,
                }
                for e in self.evaluations
            ],
            "accepted": self.accepted,
            "reason": self.reason,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_primitive(cls, data: dict[str, object]) -> ModelVersion:
        raw_evaluations = data.get("evaluations")
        evaluations = tuple(
            Evaluation(
                metric=str(e.get("metric", "")),
                before=_as_float(e.get("before")),
                after=_as_float(e.get("after")),
                lower_is_better=bool(e.get("lower_is_better", False)),
            )
            for e in (raw_evaluations if isinstance(raw_evaluations, list) else [])
            if isinstance(e, dict)
        )
        created = data.get("created_at")
        try:
            created_at = datetime.fromisoformat(str(created)) if created else datetime.now(UTC)
        except ValueError:
            created_at = datetime.now(UTC)
        return cls(
            number=_as_int(data.get("number"), 1),
            path=str(data.get("path", "")),
            parent=_as_int(data.get("parent"), 0),
            dataset=str(data.get("dataset", "")),
            examples=_as_int(data.get("examples"), 0),
            steps=_as_int(data.get("steps"), 0),
            settings=_as_dict(data.get("settings")),
            evaluations=evaluations,
            accepted=bool(data.get("accepted", False)),
            reason=str(data.get("reason", "")),
            created_at=created_at,
            id=str(data.get("id", "")),
        )

    def describe(self) -> str:
        verdict = "accepted" if self.accepted else "not accepted"
        detail = "; ".join(e.describe() for e in self.evaluations)
        return (
            f"v{self.number} ({verdict}) from {self.examples} example(s), "
            f"{self.steps} step(s)" + (f" -- {detail}" if detail else "")
        )


class ModelRegistry:
    """The version log, the candidate, and the active pointer."""

    def __init__(self, store: JsonStore | None = None) -> None:
        self.store = store or JsonStore(data_dir())

    # -- reading ------------------------------------------------------------
    def versions(self) -> tuple[ModelVersion, ...]:
        data = self.store.load(_NAMESPACE, _KEY)
        if not isinstance(data, list):
            return ()
        return tuple(
            ModelVersion.from_primitive(d) for d in data if isinstance(d, dict)
        )

    def latest(self) -> ModelVersion | None:
        versions = self.versions()
        return versions[-1] if versions else None

    def accepted_versions(self) -> tuple[ModelVersion, ...]:
        return tuple(v for v in self.versions() if v.accepted)

    def last_stable(self) -> ModelVersion | None:
        accepted = self.accepted_versions()
        return accepted[-1] if accepted else None

    def active_number(self) -> int:
        data = self.store.load(_NAMESPACE, _ACTIVE_KEY)
        if isinstance(data, dict):
            return int(data.get("number", 0) or 0)
        return 0

    def active_path(self) -> Path | None:
        """The file the active version points at (the base when none is active)."""
        number = self.active_number()
        if number >= 1:
            path = version_path(number)
            if path.exists():
                return path
        return base_path() if base_path().exists() else None

    # -- writing ------------------------------------------------------------
    def ensure_base(self, source: Path) -> Path:
        """Copy the shipped checkpoint to ``base.npz`` once; never overwrite it.

        The tokenizer sidecar is copied with it, so the base can be evaluated and
        resumed against the exact vocabulary it was trained with.
        """
        target = base_path()
        if not target.exists() and source.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            tokenizer = source.with_suffix(".tokenizer.json")
            if tokenizer.exists():
                shutil.copy2(tokenizer, target.with_suffix(".tokenizer.json"))
        return target

    def next_number(self) -> int:
        versions = self.versions()
        return (versions[-1].number + 1) if versions else 1

    def record(self, version: ModelVersion) -> ModelVersion:
        """Append a version to the log."""
        data = [v.to_primitive() for v in self.versions()]
        data.append(version.to_primitive())
        self.store.save(_NAMESPACE, _KEY, data)
        return version

    def set_active(self, number: int) -> None:
        self.store.save(_NAMESPACE, _ACTIVE_KEY, {"number": number})

    def clear_active(self) -> None:
        self.store.save(_NAMESPACE, _ACTIVE_KEY, {"number": 0})

    # -- comparison / acceptance -------------------------------------------
    def should_accept(
        self,
        evaluations: tuple[Evaluation, ...],
        *,
        max_regression: float = 0.02,
        min_improvement: float = 0.0,
    ) -> tuple[bool, str]:
        """Accept only when no metric regresses badly and at least one improves.

        This is the task's rule: do not adopt a version that gains a little on one
        task while losing a lot on others. A metric that falls by more than
        ``max_regression`` blocks acceptance outright; otherwise at least one
        metric must improve by more than ``min_improvement``.
        """
        if not evaluations:
            return False, "no evaluation was run"
        regressions = [e for e in evaluations if e.delta < -abs(max_regression)]
        if regressions:
            worst = min(regressions, key=lambda e: e.delta)
            return False, f"regresses {worst.metric} by {worst.delta:+.4f}"
        improvements = [e for e in evaluations if e.delta > abs(min_improvement)]
        if not improvements:
            return False, "no measurable improvement"
        best = max(improvements, key=lambda e: e.delta)
        return True, f"improves {best.metric} by {best.delta:+.4f}"

    # -- rollback -----------------------------------------------------------
    def rollback(self) -> ModelVersion | None:
        """Return to the last accepted version, dropping a failed candidate."""
        stable = self.last_stable()
        if stable is None:
            self.clear_active()
            return None
        self.set_active(stable.number)
        return stable

    def report(self) -> str:
        """A readable summary of the version log."""
        versions = self.versions()
        active = self.active_number()
        if not versions:
            return "No trained versions yet; the base model is in use."
        lines = [f"Active: v{active}" if active else "Active: base model"]
        for version in versions:
            mark = "*" if version.number == active else " "
            lines.append(f"{mark} {version.describe()}")
        return "\n".join(lines)


__all__ = [
    "Evaluation",
    "ModelRegistry",
    "ModelVersion",
    "base_path",
    "candidate_path",
    "models_root",
    "version_path",
]
