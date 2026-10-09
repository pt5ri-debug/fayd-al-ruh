"""Run state that lets language-model training resume instead of restarting.

The model weights live in the committed checkpoint (``fayd-llm.npz``). The Adam
moment estimates and the step counter are *run state*, not release content, so
they go in a sidecar file next to the checkpoint and are never committed. When
the sidecar is present a resumed run continues from the exact optimiser moments
and step it stopped at; when it is missing, training falls back to a fresh
optimiser, which is the previous behaviour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ml.llm.transformer import TransformerLM
from ml.models import Adam

STATE_SUFFIX = ".state.npz"


def state_path(checkpoint: str | Path) -> Path:
    """The optimiser-state sidecar for a checkpoint (``llm.npz`` -> ``llm.state.npz``)."""
    return Path(checkpoint).with_suffix(STATE_SUFFIX)


def save_training_state(optimizer: Adam, steps: int, path: str | Path) -> None:
    """Persist Adam's moments and the cumulative step count."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "steps": np.array([int(steps)], dtype=np.int64),
        "t": np.array([int(optimizer.state.t)], dtype=np.int64),
    }
    for index, moment in enumerate(optimizer.state.m):
        arrays[f"m{index}"] = moment
    for index, moment in enumerate(optimizer.state.v):
        arrays[f"v{index}"] = moment
    np.savez_compressed(target, **arrays)  # type: ignore[arg-type]


def load_training_state(path: str | Path, model: TransformerLM) -> tuple[Adam, int] | None:
    """Rebuild an Adam optimiser and step count for ``model``, or ``None`` if absent.

    The stored moments must line up with the model's parameters; a mismatch means
    the sidecar belongs to a different architecture, which is a hard error rather
    than a silently wrong resume.
    """
    source = Path(path)
    if not source.exists():
        return None
    data = np.load(source, allow_pickle=False)
    expected = len(model.parameters())
    if int(data["t"].shape[0]) != 1 or "m0" not in data:
        raise ValueError(f"Optimiser state {source} is unreadable.")
    moments = [f"m{index}" for index in range(expected)]
    missing = [key for key in moments if key not in data]
    if missing:
        raise ValueError(
            f"Optimiser state {source} does not match the model's {expected} parameters."
        )
    optimizer = Adam(model.parameters())
    optimizer.state.t = int(data["t"][0])
    optimizer.state.m = [data[f"m{index}"] for index in range(expected)]
    optimizer.state.v = [data[f"v{index}"] for index in range(expected)]
    steps = int(data["steps"][0])
    return optimizer, steps


__all__ = ["STATE_SUFFIX", "load_training_state", "save_training_state", "state_path"]
