"""Sharded, resumable training schedule for the language model.

A resume replays the *same* corpus by default: the shuffled windows are drawn
from the whole token stream again, so more steps deepen the fit on material the
model has already seen rather than reaching new material. This module adds a
schedule that splits the document universe into a fixed number of shards and
walks them one per run, recording which shards were consumed in a sidecar so a
later `--resume` trains on the *next* shard.

The shards are a deterministic partition of `ml.llm.corpus.corpus_units()`
(whole documents, shuffled by a fixed seed, dealt round-robin), so the consumed
shards can always be reconstructed from the sidecar alone. The sidecar is run
state, like the optimiser state: never committed, and rebuilt fresh when absent.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEDULE_SUFFIX = ".schedule.json"
DEFAULT_SHARD_COUNT = 5


def schedule_path(checkpoint: str | Path) -> Path:
    """The schedule sidecar for a checkpoint (``llm.npz`` -> ``llm.schedule.json``)."""
    source = Path(checkpoint)
    return source.with_suffix(SCHEDULE_SUFFIX)


@dataclass(frozen=True)
class ShardPlan:
    """A deterministic partition of the document universe into ``count`` shards."""

    count: int = DEFAULT_SHARD_COUNT

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("Shard count must be at least 1.")

    def shards(self, units: Sequence[str], *, seed: int = 0) -> list[list[str]]:
        """Partition whole documents into ``count`` balanced shards.

        Documents are shuffled with a fixed seed, then dealt round-robin so the
        shards are balanced and together cover every document exactly once.
        """
        items = [unit for unit in units if unit.strip()]
        buckets: list[list[str]] = [[] for _ in range(self.count)]
        if not items:
            return buckets
        order = np.random.default_rng(seed).permutation(len(items))
        for position, index in enumerate(order):
            buckets[position % self.count].append(items[int(index)])
        return buckets

    def next_index(self, consumed: Sequence[int]) -> int:
        """The shard to train on next, cycling once every shard is consumed."""
        return len(consumed) % self.count


@dataclass
class ScheduleState:
    """Which shards a run has trained on, and the plan they belong to."""

    count: int
    seed: int
    consumed: list[int] = field(default_factory=list)
    units: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "seed": self.seed,
            "consumed": list(self.consumed),
            "units": self.units,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ScheduleState:
        count = data.get("count")
        seed = data.get("seed", 0)
        consumed = data.get("consumed", [])
        units = data.get("units", 0)
        if not isinstance(count, int) or not isinstance(seed, int):
            raise ValueError("Schedule count and seed must be integers.")
        if not isinstance(consumed, list) or not all(isinstance(i, int) for i in consumed):
            raise ValueError("Schedule consumed must be a list of shard indices.")
        if not isinstance(units, int):
            raise ValueError("Schedule units must be an integer.")
        if count < 1:
            raise ValueError("Schedule count must be at least 1.")
        return cls(count=count, seed=seed, consumed=[int(i) for i in consumed], units=units)

    def matches(self, plan: ShardPlan, seed: int) -> bool:
        """Whether this state was recorded for ``plan`` and ``seed``."""
        return self.count == plan.count and self.seed == seed


def load_schedule(path: str | Path) -> ScheduleState | None:
    """Read a schedule sidecar, or ``None`` when it is absent."""
    source = Path(path)
    if not source.exists():
        return None
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise ValueError(f"Schedule {source} is unreadable: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"Schedule {source} is unreadable.")
    return ScheduleState.from_dict(data)


def save_schedule(state: ScheduleState, path: str | Path) -> None:
    """Persist the consumed shards and the plan they belong to."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(state.as_dict(), indent=2, sort_keys=True), encoding="utf-8")


def select_shard(
    units: Sequence[str],
    plan: ShardPlan,
    state: ScheduleState,
    *,
    seed: int,
) -> tuple[list[str], int]:
    """The documents of the next unconsumed shard, and its index.

    ``state`` is advanced by the caller *after* a successful run, so a crashed
    run does not silently consume a shard.
    """
    shards = plan.shards(units, seed=seed)
    index = plan.next_index(state.consumed)
    return shards[index], index


def describe_schedule(path: str | Path) -> dict[str, object] | None:
    """A manifest-friendly summary of a schedule sidecar, or ``None`` if absent."""
    state = load_schedule(path)
    if state is None:
        return None
    index = state.count and (len(state.consumed) % state.count)
    return {
        "count": state.count,
        "seed": state.seed,
        "consumed": list(state.consumed),
        "next": index,
        "units": state.units,
    }


__all__ = [
    "DEFAULT_SHARD_COUNT",
    "SCHEDULE_SUFFIX",
    "ScheduleState",
    "ShardPlan",
    "describe_schedule",
    "load_schedule",
    "save_schedule",
    "schedule_path",
    "select_shard",
]
