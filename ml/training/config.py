"""The training configuration manager: one validated, reproducible config object.

A training run is only reproducible if its configuration is a **recorded, validated
value** rather than a set of CLI flags a reader has to reconstruct. :class:`TrainingConfig`
is that value: it carries the base model, the dataset, the method, the LoRA
hyper-parameters, the batch/accumulation settings, the seed, and the resource
limits, and :meth:`TrainingConfig.validate` refuses an impossible combination
before a run starts.

Defaults are deliberately **conservative** for the stated hardware (Intel i7,
16 GB RAM, no usable CUDA GPU): a small LoRA rank, a small batch, gradient
accumulation, and a hard memory ceiling. A caller can raise them, and the config
records that it was raised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ml.training.compatibility import (
    METHOD_LORA_INPROCESS,
    METHODS,
)

#: LoRA rank bounds. A larger rank learns more and needs more memory.
MIN_RANK = 1
MAX_RANK = 256
DEFAULT_RANK = 4

#: The scaling factor; a common default is ``2 * rank``.
MIN_ALPHA = 1
MAX_ALPHA = 1024

DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 2
DEFAULT_EPOCHS = 1
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_MAX_STEPS = 200
DEFAULT_SEED = 0

#: The memory ceiling the trainer enforces, in bytes (default 4 GB).
DEFAULT_MEMORY_LIMIT_BYTES = 4 * 1024**3

#: The minimum free disk the trainer insists on before it writes anything (2 GB).
DEFAULT_DISK_FLOOR_BYTES = 2 * 1024**3

#: How often a checkpoint is written (in optimizer steps).
DEFAULT_CHECKPOINT_EVERY = 25


class TrainingConfigError(ValueError):
    """The configuration is impossible or self-contradictory."""


@dataclass(frozen=True)
class TrainingConfig:
    """A validated, reproducible training configuration."""

    # -- what to train ---------------------------------------------------
    base_model_id: str = ""
    dataset_id: str = ""
    dataset_version: int = 0
    method: str = METHOD_LORA_INPROCESS
    owner: str = ""
    run_id: str = ""

    # -- LoRA hyper-parameters -------------------------------------------
    rank: int = DEFAULT_RANK
    alpha: int = DEFAULT_RANK * 2
    dropout: float = 0.05

    # -- optimisation ----------------------------------------------------
    batch_size: int = DEFAULT_BATCH_SIZE
    grad_accum: int = DEFAULT_GRAD_ACCUM
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    max_steps: int = DEFAULT_MAX_STEPS
    seed: int = DEFAULT_SEED
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY

    # -- resource limits -------------------------------------------------
    memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES
    disk_floor_bytes: int = DEFAULT_DISK_FLOOR_BYTES
    timeout_seconds: float = 0.0
    #: Recorded so a reader sees which knobs were raised from the conservative default.
    overrides: tuple[str, ...] = field(default=())

    def validate(self) -> None:
        """Refuse an impossible configuration, naming exactly what is wrong."""
        if self.method not in METHODS:
            raise TrainingConfigError(
                f"Unknown method {self.method!r}; expected one of {METHODS}."
            )
        if not self.base_model_id:
            raise TrainingConfigError("A base model id is required.")
        if not self.dataset_id:
            raise TrainingConfigError("A dataset id is required.")
        if self.dataset_version < 0:
            raise TrainingConfigError("The dataset version cannot be negative.")
        if not MIN_RANK <= self.rank <= MAX_RANK:
            raise TrainingConfigError(
                f"LoRA rank {self.rank} is outside [{MIN_RANK}, {MAX_RANK}]."
            )
        if not MIN_ALPHA <= self.alpha <= MAX_ALPHA:
            raise TrainingConfigError(
                f"LoRA alpha {self.alpha} is outside [{MIN_ALPHA}, {MAX_ALPHA}]."
            )
        if not 0.0 <= self.dropout < 1.0:
            raise TrainingConfigError("LoRA dropout must be in [0, 1).")
        if self.batch_size < 1:
            raise TrainingConfigError("The batch size must be at least 1.")
        if self.grad_accum < 1:
            raise TrainingConfigError("Gradient accumulation must be at least 1.")
        if self.epochs < 1:
            raise TrainingConfigError("There must be at least one epoch.")
        if not 0.0 < self.learning_rate < 1.0:
            raise TrainingConfigError("The learning rate must be in (0, 1).")
        if self.max_steps < 1:
            raise TrainingConfigError("max_steps must be at least 1.")
        if self.checkpoint_every < 1:
            raise TrainingConfigError("checkpoint_every must be at least 1.")
        if self.memory_limit_bytes < 64 * 1024**2:
            raise TrainingConfigError(
                "The memory limit is below 64 MB, which no training run can fit in."
            )
        if self.disk_floor_bytes < 0:
            raise TrainingConfigError("The disk floor cannot be negative.")
        if self.timeout_seconds < 0:
            raise TrainingConfigError("The timeout cannot be negative.")

    @property
    def effective_batch(self) -> int:
        """The batch size the optimiser actually sees (batch × accumulation)."""
        return self.batch_size * self.grad_accum

    def with_overrides(self, **changes: Any) -> TrainingConfig:
        """Return a new config with named changes, recording which knobs moved."""
        unknown = set(changes) - set(self.__dataclass_fields__)
        if unknown:
            raise TrainingConfigError(f"Unknown configuration field(s): {sorted(unknown)}.")
        merged = {**self.to_primitive(), **changes}
        merged.pop("overrides", None)
        defaults = TrainingConfig()
        overrides = tuple(
            sorted(
                key
                for key, value in merged.items()
                if value != getattr(defaults, key, None)
                and key
                not in (
                    "base_model_id",
                    "dataset_id",
                    "dataset_version",
                    "method",
                    "owner",
                    "run_id",
                )
            )
        )
        config = TrainingConfig(**merged, overrides=overrides)
        config.validate()
        return config

    def to_primitive(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "base_model_id": self.base_model_id,
            "batch_size": self.batch_size,
            "checkpoint_every": self.checkpoint_every,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "disk_floor_bytes": self.disk_floor_bytes,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "grad_accum": self.grad_accum,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "memory_limit_bytes": self.memory_limit_bytes,
            "method": self.method,
            "overrides": list(self.overrides),
            "owner": self.owner,
            "rank": self.rank,
            "run_id": self.run_id,
            "seed": self.seed,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_primitive(cls, data: object) -> TrainingConfig:
        if not isinstance(data, dict):
            raise TrainingConfigError("A training config must be a mapping.")
        fields = {
            key: data[key]
            for key in cls.__dataclass_fields__
            if key in data and key != "overrides"
        }
        config = cls(**fields, overrides=tuple(data.get("overrides", ()) or ()))
        return config

    def describe(self) -> str:
        return (
            f"{self.method} on {self.base_model_id} · dataset "
            f"{self.dataset_id}@v{self.dataset_version} · rank {self.rank}/alpha "
            f"{self.alpha} · batch {self.batch_size}×{self.grad_accum} · "
            f"{self.max_steps} steps · lr {self.learning_rate:g} · seed {self.seed}"
        )


#: A conservative starting config for the stated hardware.
def default_config(
    base_model_id: str = "", dataset_id: str = "", dataset_version: int = 0
) -> TrainingConfig:
    """The conservative default, pre-filled with a base model and dataset."""
    config = TrainingConfig(
        base_model_id=base_model_id,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
    )
    return config


#: The one place a preset's numbers live, so the CLI and the window agree.
PRESETS: dict[str, dict[str, Any]] = {
    "cpu-safe": {
        "rank": 4,
        "alpha": 8,
        "batch_size": 2,
        "grad_accum": 4,
        "max_steps": 100,
    },
    "cpu-balanced": {
        "rank": 8,
        "alpha": 16,
        "batch_size": 4,
        "grad_accum": 2,
        "max_steps": 200,
    },
    "quality": {
        "rank": 16,
        "alpha": 32,
        "batch_size": 8,
        "grad_accum": 1,
        "max_steps": 400,
    },
}


def resolve_preset(name: str) -> dict[str, Any]:
    """The hyper-parameters of a named preset; raises for an unknown name."""
    if name not in PRESETS:
        raise TrainingConfigError(
            f"Unknown preset {name!r}; expected one of {tuple(PRESETS)}."
        )
    return dict(PRESETS[name])


def preset_names() -> tuple[str, ...]:
    return tuple(PRESETS)


def preset_summaries() -> tuple[tuple[str, str], ...]:
    return (
        ("cpu-safe", "the smallest run: rank 4, batch 2×4, 100 steps"),
        ("cpu-balanced", "the default: rank 8, batch 4×2, 200 steps"),
        ("quality", "the largest run: rank 16, batch 8, 400 steps (slow on CPU)"),
    )


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CHECKPOINT_EVERY",
    "DEFAULT_DISK_FLOOR_BYTES",
    "DEFAULT_EPOCHS",
    "DEFAULT_GRAD_ACCUM",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MEMORY_LIMIT_BYTES",
    "DEFAULT_RANK",
    "DEFAULT_SEED",
    "MAX_ALPHA",
    "MAX_RANK",
    "MIN_ALPHA",
    "MIN_RANK",
    "PRESETS",
    "TrainingConfig",
    "TrainingConfigError",
    "default_config",
    "preset_names",
    "preset_summaries",
    "resolve_preset",
]
