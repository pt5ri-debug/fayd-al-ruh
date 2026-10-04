"""Data layer: replay buffers, feature hashing, and train/val splits.

The learning signal here is grounded in the project's own symbolic substrate.
Experiences are captured from the sandbox tasks and encoded into fixed-size
vectors with a deterministic hasher, so the same facts always map to the same
features across runs and machines.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


def _stable_index(token: str, size: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % size


def _stable_sign(token: str) -> float:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return 1.0 if digest[0] % 2 == 0 else -1.0


class FeatureHasher:
    """Deterministic bag-of-tokens encoder with signed hashing."""

    def __init__(self, size: int = 64) -> None:
        if size < 1:
            raise ValueError("Feature size must be at least 1.")
        self.size = size

    def encode(self, tokens: Iterable[str]) -> np.ndarray:
        vector = np.zeros(self.size, dtype=np.float64)
        count = 0
        for token in tokens:
            if not token:
                continue
            vector[_stable_index(token, self.size)] += _stable_sign(token)
            count += 1
        if count:
            vector /= np.sqrt(count)
        return vector

    def encode_many(self, token_lists: Iterable[Iterable[str]]) -> np.ndarray:
        rows = [self.encode(tokens) for tokens in token_lists]
        if not rows:
            return np.zeros((0, self.size), dtype=np.float64)
        return np.vstack(rows)


@dataclass
class Transition:
    """One observed (state, action, reward, next_state, done) experience."""

    state_tokens: tuple[str, ...]
    action: int
    reward: float
    next_tokens: tuple[str, ...]
    done: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "state_tokens": list(self.state_tokens),
            "action": self.action,
            "reward": self.reward,
            "next_tokens": list(self.next_tokens),
            "done": self.done,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transition":
        raw_state = data.get("state_tokens", [])
        raw_next = data.get("next_tokens", [])
        state_tokens = tuple(str(token) for token in raw_state) if isinstance(raw_state, (list, tuple)) else ()
        next_tokens = tuple(str(token) for token in raw_next) if isinstance(raw_next, (list, tuple)) else ()
        return cls(
            state_tokens,
            int(data["action"]),  # type: ignore[arg-type]
            float(data["reward"]),  # type: ignore[arg-type]
            next_tokens,
            bool(data.get("done", False)),
        )


@dataclass
class ReplayBuffer:
    """A bounded, append-only store of transitions."""

    capacity: int = 10_000
    _items: list[Transition] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("Replay capacity must be at least 1.")

    def add(self, transition: Transition) -> None:
        self._items.append(transition)
        if len(self._items) > self.capacity:
            del self._items[0]

    def extend(self, transitions: Iterable[Transition]) -> None:
        for transition in transitions:
            self.add(transition)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def items(self) -> tuple[Transition, ...]:
        return tuple(self._items)

    def clear(self) -> None:
        self._items.clear()

    def to_dicts(self) -> list[dict[str, object]]:
        return [transition.to_dict() for transition in self._items]

    def load_dicts(self, payload: Iterable[dict[str, object]]) -> None:
        self._items = [Transition.from_dict(item) for item in payload][-self.capacity :]


@dataclass(frozen=True)
class Dataset:
    """Feature matrix plus integer labels."""

    features: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        if self.features.shape[0] != self.labels.shape[0]:
            raise ValueError("Features and labels must have the same length.")

    def __len__(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_classes(self) -> int:
        return int(self.labels.max()) + 1 if len(self) else 0


def build_classification_dataset(
    transitions: Iterable[Transition],
    hasher: FeatureHasher,
    *,
    reward_threshold: float = 0.0,
) -> Dataset:
    """Label each transition 1 when its reward clears the threshold, else 0."""
    items = list(transitions)
    if not items:
        return Dataset(np.zeros((0, hasher.size)), np.zeros(0, dtype=np.int64))
    features = hasher.encode_many(item.state_tokens for item in items)
    labels = np.array(
        [1 if item.reward > reward_threshold else 0 for item in items], dtype=np.int64
    )
    return Dataset(features, labels)


def train_val_split(
    dataset: Dataset, *, val_fraction: float = 0.2, seed: int = 0
) -> tuple[Dataset, Dataset]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")
    count = len(dataset)
    if count < 2:
        return dataset, dataset
    generator = np.random.default_rng(seed)
    order = generator.permutation(count)
    val_size = max(1, int(round(count * val_fraction)))
    val_idx, train_idx = order[:val_size], order[val_size:]
    return (
        Dataset(dataset.features[train_idx], dataset.labels[train_idx]),
        Dataset(dataset.features[val_idx], dataset.labels[val_idx]),
    )
