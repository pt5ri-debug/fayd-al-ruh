"""Resolve where Fayd Al-Ruh reads bundled assets and writes its data.

Two modes matter:

- **Source checkout** (development): code and bundled assets stay in the
  repository, while learned data (artifacts, checkpoints, knowledge base,
  learned documents) is kept in a ``data/`` folder beside the code. That folder
  is git-ignored, so retraining never dirties the working tree and a program
  update can fast-forward without disturbing what the user has learned.
- **Frozen build** (portable): assets ship inside the bundle (PyInstaller
  ``_MEIPASS``) and data is written next to the executable, so the folder can be
  copied to a USB stick and keep its state.

``FAYD_DATA_DIR`` overrides the data location in either mode.
"""

from __future__ import annotations

import os
import shutil
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

    A frozen build is portable: data sits next to the executable. A source
    checkout keeps its learned data in a ``data/`` folder beside the code (which
    git ignores), so retraining never dirties the working tree and a program
    update can fast-forward without touching what the user has learned.
    ``FAYD_DATA_DIR`` overrides the location in either mode.
    """
    override = os.environ.get(ENV_DATA_DIR)
    if override:
        return Path(override).expanduser().resolve()
    if is_frozen():
        return executable_dir() / "data"
    return _source_root() / "data"


def artifacts_dir() -> Path:
    """Where ``python -m ml`` writes learned artifacts by default."""
    return data_dir() / "ml" / "artifacts"


def checkpoints_dir() -> Path:
    """Where ``python -m ml.llm`` writes the trained language-model checkpoint."""
    return data_dir() / "ml" / "llm" / "checkpoints"


def models_dir() -> Path:
    """Where optional learned models (e.g. a diffusion checkpoint) are cached.

    Kept under the writable data folder (git-ignored), never in the repository, so
    a downloaded model never dirties the working tree or bloats a program update.
    """
    return data_dir() / "models"


def ultron_models_dir() -> Path:
    """Where the Ultron general-language models (GGUF files) live.

    Ultron's model is *not* an in-project checkpoint: it is a general, advanced
    model the user drops into ``data/ultron`` (or downloads there with
    ``python -m cli ultron download --approve``). The folder is git-ignored, so a
    multi-gigabyte model never dirties the working tree or bloats a program
    update. ``FAYD_DATA_DIR`` moves it with the rest of the data.
    """
    return data_dir() / "ultron"


def vision_models_dir() -> Path:
    """Where the Vision petroleum-engineering models (GGUF files) live.

    Vision is the project's *petroleum* local model (built on the OGAI Reasoner
    weights): like Ultron it is *not* an in-project checkpoint, but a GGUF file
    the user drops into ``data/vision`` (or downloads there with
    ``python -m cli vision download --approve``). The folder is git-ignored, so a
    multi-gigabyte model never dirties the working tree or bloats a program
    update. ``FAYD_DATA_DIR`` moves it with the rest of the data.
    """
    return data_dir() / "vision"


def coder_models_dir() -> Path:
    """Where the Code Studio's in-process coding models (GGUF files) live.

    Like Ultron's models, these are *not* in-project checkpoints: they are GGUF
    files the user drops into ``data/coder`` (or downloads there with
    ``python -m cli studio download --approve``). Running the model in-process
    means no ``ollama serve`` daemon has to sit in the background holding RAM.
    The folder is git-ignored, so a multi-gigabyte model never dirties the
    working tree or bloats a program update.
    """
    return data_dir() / "coder"


def presentation_models_dir() -> Path:
    """Where the Presentation Studio's in-process authoring models live.

    The Studio drafts outlines and slide content with a GGUF file the user drops
    into ``data/presentation`` (or downloads there with
    ``python -m cli presentation model-download --approve``). Like Ultron's and
    the Code Studio's models these are *not* in-project checkpoints, so the
    folder is git-ignored and a multi-gigabyte model never dirties the working
    tree or bloats a program update. ``FAYD_DATA_DIR`` moves it with the rest of
    the data.
    """
    return data_dir() / "presentation"


def qwen_speech_models_dir() -> Path:
    """Where the Qwen3 speech models (ASR and TTS) live.

    Qwen3-ASR and Qwen3-TTS are *not* in-project checkpoints: they are
    ``transformers`` model folders the user drops into ``data/qwen-speech`` (or
    downloads there with ``python -m cli qwen-asr download --approve`` and
    ``python -m cli qwen-tts download --approve``). One folder holds both
    families, keyed by a registry, so a machine that installs one can install the
    other beside it. The folder is git-ignored, so a multi-gigabyte model never
    dirties the working tree or bloats a program update.
    """
    return data_dir() / "qwen-speech"


def studio_dir() -> Path:
    """Where the Code Studio keeps its own state (git worktrees, run logs).

    The Studio edits the project through *git worktrees* under this folder: one
    branch per change, checked out beside the code so it can be tested in
    isolation and merged back with a button. The folder is git-ignored, so a
    half-finished change never dirties the working tree and a program update
    (which only fast-forwards tracked files) never touches it.
    """
    return data_dir() / "studio"


def studio_workspaces_dir() -> Path:
    """The parent folder of every Studio worktree (one directory per change)."""
    return studio_dir() / "workspaces"


def bundled_artifacts_dir() -> Path:
    """Read-only artifacts inside the code or bundle, used as a fallback."""
    return bundle_root() / "ml" / "artifacts"


def bundled_checkpoints_dir() -> Path:
    """Read-only language-model checkpoints inside the code or bundle."""
    return bundle_root() / "ml" / "llm" / "checkpoints"


def default_checkpoint() -> Path:
    """Prefer a user-retrained checkpoint, then fall back to the bundled one."""
    candidate = checkpoints_dir() / "fayd-llm.npz"
    if candidate.exists():
        return candidate
    return bundled_checkpoints_dir() / "fayd-llm.npz"


def ensure_data_layout() -> None:
    """One-time upgrade step: gather pre-existing source data under ``data/``.

    Before this, a source build wrote ML artifacts, the knowledge base, learned
    documents, and the learning cursor straight into the repository. Those still
    exist on an upgrading machine, so on first run we copy them into ``data/``
    once; git-updated copies of the tracked artifacts keep working, and nothing
    the user trained is lost. Best-effort and silent: any failure simply leaves
    the old locations in place.
    """
    if is_frozen() or os.environ.get(ENV_DATA_DIR):
        return
    root = _source_root()
    data = root / "data"
    if data.exists():
        return

    pairs = (
        (root / "ml" / "artifacts", data / "ml" / "artifacts"),
        (root / "ml" / "llm" / "checkpoints", data / "ml" / "llm" / "checkpoints"),
        (root / "knowledge_base", data / "knowledge_base"),
        (root / "documents", data / "documents"),
    )
    if not any(source.exists() for source, _target in pairs):
        return
    try:
        for source, target in pairs:
            if source.exists() and not target.exists():
                shutil.copytree(source, target)
        cursor = root / "learned_cursor.json"
        if cursor.exists() and not (data / "learned_cursor.json").exists():
            data.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cursor, data / "learned_cursor.json")
    except OSError:
        pass


__all__ = [
    "ENV_DATA_DIR",
    "is_frozen",
    "bundle_root",
    "executable_dir",
    "data_dir",
    "artifacts_dir",
    "checkpoints_dir",
    "models_dir",
    "ultron_models_dir",
    "coder_models_dir",
    "presentation_models_dir",
    "studio_dir",
    "studio_workspaces_dir",
    "bundled_artifacts_dir",
    "bundled_checkpoints_dir",
    "default_checkpoint",
    "ensure_data_layout",
]
