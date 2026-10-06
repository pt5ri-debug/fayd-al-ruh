"""Resolve where Fayd Al-Ruh reads bundled assets and writes its data.

Two modes matter:

- **Source checkout** (development): assets and data stay inside the repository,
  so ``python -m pytest`` and ``python -m ml`` behave exactly as before.
- **Frozen build** (portable): assets ship inside the bundle (PyInstaller
  ``_MEIPASS``) and data is written next to the executable, so the folder can be
  copied to a USB stick and keep its state.

``FAYD_DATA_DIR`` overrides the data location in either mode.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ENV_DATA_DIR = "FAYD_DATA_DIR"


def is_frozen() -> bool:
    """True when running from a PyInstaller-style bundle rather than source."""
    return bool(getattr(sys, "frozen", False))


def _source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def bundle_root() -> Path:
    """Read-only assets: the source tree, or the bundle's extraction directory."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return _source_root()


def executable_dir() -> Path:
    """Folder of the running program: the executable's folder, or the repo root."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return _source_root()


def data_dir() -> Path:
    """Writable folder for state, artifacts, and trained checkpoints.

    A frozen build is portable: data sits next to the executable. A source build
    keeps the historical repository layout unless ``FAYD_DATA_DIR`` is set.
    """
    override = os.environ.get(ENV_DATA_DIR)
    if override:
        return Path(override).expanduser().resolve()
    if is_frozen():
        return executable_dir() / "data"
    return _source_root()


def artifacts_dir() -> Path:
    """Where ``python -m ml`` writes learned artifacts by default."""
    return data_dir() / "ml" / "artifacts"


def checkpoints_dir() -> Path:
    """Where ``python -m ml.llm`` writes the trained language-model checkpoint."""
    return data_dir() / "ml" / "llm" / "checkpoints"


def default_checkpoint() -> Path:
    """Prefer a user-retrained checkpoint, then fall back to the bundled one."""
    candidate = checkpoints_dir() / "fayd-llm.npz"
    if candidate.exists():
        return candidate
    return bundle_root() / "ml" / "llm" / "checkpoints" / "fayd-llm.npz"


__all__ = [
    "ENV_DATA_DIR",
    "is_frozen",
    "bundle_root",
    "executable_dir",
    "data_dir",
    "artifacts_dir",
    "checkpoints_dir",
    "default_checkpoint",
]
