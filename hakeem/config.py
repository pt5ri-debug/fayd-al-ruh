"""Configuration for Hakeem, the project's scholar of religion and history.

Every tunable lives in this one place: the model and server, timeouts, how much
context is sent, and where his database sits. Hakeem speaks through the **same**
model as Natasha, the companion, so the model defaults are the companion's own
(see :mod:`hakeem.model`); only the storage location differs, so the two
characters keep separate memories and sessions.

Nothing here duplicates a setting that already exists elsewhere in Fayd Al-Ruh,
and no absolute path is hard-coded -- the data folder is discovered through
:mod:`core.paths`, exactly like the rest of the project.

Hakeem is a *conversational scholar layer*, not the cognitive core: this
configuration never touches entities, relations, evidence, beliefs, or the
knowledge graph.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from core.paths import data_dir

from hakeem.model import (
    BACKEND_AUTO,
    BACKEND_NAMES,
    DEFAULT_CONTEXT,
    DEFAULT_GPU_LAYERS,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_THREADS,
    MAX_CONTEXT,
)

#: Environment overrides, following the project's ``FAYD_*`` convention, so a
#: user can point Hakeem elsewhere without editing code.
ENV_MODEL = "FAYD_HAKEEM_MODEL"
ENV_BACKEND = "FAYD_HAKEEM_BACKEND"
ENV_MODEL_FILE = "FAYD_HAKEEM_MODEL_FILE"
ENV_OLLAMA_URL = "FAYD_HAKEEM_OLLAMA_URL"
ENV_DB = "FAYD_HAKEEM_DB"
ENV_TIMEOUT = "FAYD_HAKEEM_TIMEOUT"
ENV_THINK = "FAYD_HAKEEM_THINK"

#: Default name of the database file inside the writable data folder. It is a
#: *separate* file from the companion's, so Hakeem's memories and sessions can
#: never mix with Natasha's.
DEFAULT_DB_FILENAME = "hakeem.db"

#: Where backups are written, relative to the database's own folder.
BACKUP_DIRNAME = "hakeem-backups"


def default_db_path() -> Path:
    """Where Hakeem's SQLite database lives (the git-ignored ``data/`` folder)."""
    override = os.environ.get(ENV_DB)
    if override:
        return Path(override).expanduser()
    return data_dir() / DEFAULT_DB_FILENAME


def backup_dir(db_path: Path | None = None) -> Path:
    """The folder that holds rotated backups for a database file."""
    return (db_path or default_db_path()).parent / BACKUP_DIRNAME


@dataclass(frozen=True)
class HakeemConfig:
    """Everything that configures Hakeem, resolved in one object.

    The fields mirror the companion's, because he speaks through the same model;
    ``persistent=False`` keeps the whole character in memory -- useful for a test,
    or for a user who wants no lasting record at all.
    """

    # -- model connection -------------------------------------------------
    model: str = DEFAULT_MODEL
    #: Which runtime speaks for Hakeem: ``auto`` (prefer the in-process GGUF when
    #: the extra and a model file are present, else Ollama), ``gguf`` (force the
    #: in-process model), or ``ollama`` (force the server).
    backend: str = BACKEND_AUTO
    #: An explicit GGUF file to load, overriding the folder scan. ``None`` uses the
    #: companion's ``data/natasha`` folder, since it is the same model.
    model_file: Path | None = None
    base_url: str = DEFAULT_OLLAMA_URL
    #: Generous by default: a reasoning model on a CPU-only machine can take
    #: minutes for one turn, so a short timeout reports a false failure.
    timeout: float = 600.0
    #: A scholar's answer benefits from a steady, non-random register; a touch
    #: lower than the companion's, but still warm enough to be natural.
    temperature: float = 0.6
    #: Room for a reasoning model's hidden reasoning trace *and* its reply.
    max_tokens: int = 4096
    #: Whether a reasoning model may emit its hidden reasoning trace.
    think: bool = True
    #: Context window for the in-process model. Ignored by the Ollama runtime.
    context: int = DEFAULT_CONTEXT
    #: GPU offload and thread count for the in-process model; both CPU-friendly.
    gpu_layers: int = DEFAULT_GPU_LAYERS
    threads: int = DEFAULT_THREADS

    # -- storage ----------------------------------------------------------
    db_path: Path | None = field(default_factory=default_db_path)
    persistent: bool = True
    #: How many rotated backups to keep; older ones are pruned so backups
    #: never grow without bound.
    backup_keep: int = 5

    # -- context budget ---------------------------------------------------
    #: How many recent messages form the direct context window.
    history_window: int = 12
    #: At most this many relevant memories are injected per request.
    memory_limit: int = 6
    #: A session is summarised once it holds more than this many messages.
    summary_threshold: int = 40
    #: Keep this many most recent summaries per session.
    summary_limit: int = 3
    #: Hard character budget for the assembled prompt.
    context_chars: int = 12000

    # -- tone -------------------------------------------------------------
    #: When false, the dialogue tone stays neutral and is never persisted.
    emotion_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("A model name is required.")
        if self.backend.strip().lower() not in BACKEND_NAMES:
            raise ValueError(f"backend must be one of {', '.join(BACKEND_NAMES)}.")
        if not self.base_url.strip():
            raise ValueError("An Ollama base URL is required.")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive.")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2].")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1.")
        if not 1 <= self.context <= MAX_CONTEXT:
            raise ValueError(f"context must be in [1, {MAX_CONTEXT}].")
        if self.history_window < 1:
            raise ValueError("history_window must be at least 1.")
        if self.memory_limit < 0:
            raise ValueError("memory_limit cannot be negative.")
        if self.summary_threshold < 1:
            raise ValueError("summary_threshold must be at least 1.")
        if self.context_chars < 500:
            raise ValueError("context_chars is too small to be useful.")
        if self.backup_keep < 0:
            raise ValueError("backup_keep cannot be negative.")

    @classmethod
    def from_env(cls, **overrides: object) -> HakeemConfig:
        """Build a config from the environment, then apply explicit overrides."""
        env: dict[str, object] = {}
        model = os.environ.get(ENV_MODEL)
        if model:
            env["model"] = model
        backend = os.environ.get(ENV_BACKEND)
        if backend:
            env["backend"] = backend
        model_file = os.environ.get(ENV_MODEL_FILE)
        if model_file:
            env["model_file"] = Path(model_file).expanduser()
        url = os.environ.get(ENV_OLLAMA_URL)
        if url:
            env["base_url"] = url
        db = os.environ.get(ENV_DB)
        if db:
            env["db_path"] = Path(db).expanduser()
        timeout = os.environ.get(ENV_TIMEOUT)
        if timeout:
            env["timeout"] = float(timeout)
        think = os.environ.get(ENV_THINK)
        if think:
            env["think"] = think.strip().lower() in {"1", "true", "yes", "on"}
        env.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**env)  # type: ignore[arg-type]

    def with_overrides(self, **overrides: object) -> HakeemConfig:
        """A copy with ``overrides`` applied (``None`` values are ignored)."""
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})  # type: ignore[arg-type]

    def describe(self) -> str:
        """A one-line, honest description of the connection and storage."""
        storage = str(self.db_path) if self.persistent else "in-memory (no lasting record)"
        if self.backend.strip().lower() == "ollama":
            where = f"{self.model} at {self.base_url}"
        else:
            where = f"{self.model} ({self.backend})"
        return f"{where}; storage: {storage}"


__all__ = [
    "BACKUP_DIRNAME",
    "DEFAULT_DB_FILENAME",
    "DEFAULT_MODEL",
    "DEFAULT_OLLAMA_URL",
    "ENV_BACKEND",
    "ENV_DB",
    "ENV_MODEL",
    "ENV_MODEL_FILE",
    "ENV_OLLAMA_URL",
    "ENV_THINK",
    "ENV_TIMEOUT",
    "HakeemConfig",
    "backup_dir",
    "default_db_path",
]
