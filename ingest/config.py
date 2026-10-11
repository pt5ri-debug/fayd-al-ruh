"""Configuration for the ingestion subsystem, via the project's own mechanism.

Ingestion has a handful of tunables -- where the store lives, how big chunks are,
which shard size, whether a re-run is incremental, and whether the (denied by
default) internet connector may be used. Rather than invent a new settings file,
this reuses the two mechanisms the project already has:

* :class:`storage.json_store.JsonStore` persists the record, exactly as
  :class:`core.preferences.UserPreferences` persists Jarvis preferences;
* environment overrides follow :mod:`core.paths`' ``FAYD_*`` convention, here
  ``FAYD_INGEST_<FIELD>``.

Resolution order is *explicit argument -> environment -> stored value ->
built-in default*, so a caller can always pin a value and the environment can
override a stored file without editing it. Nothing here enables the web
connector on its own: ``allow_web`` defaults to ``False`` and only a caller (or
an explicit environment value) turns it on.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

from core.knowledge_base import (
    DEFAULT_CHUNK_CHARS,
    DEFAULT_MIN_CHUNK,
    DEFAULT_OVERLAP,
    default_knowledge_base_dir,
)
from ingest.pipeline import DEFAULT_SHARD_BYTES
from storage.json_store import JsonStore

#: Environment prefix, mirroring ``core.paths.ENV_DATA_DIR`` ("FAYD_DATA_DIR").
ENV_PREFIX = "FAYD_INGEST_"

_NAMESPACE = "ingest"
_KEY = "config"


@dataclass(frozen=True)
class IngestConfig:
    """The ingestion tunables, with the same defaults the CLI and pipeline use.

    ``store_dir`` is empty by default so it resolves to
    :func:`core.knowledge_base.default_knowledge_base_dir` at use time; that keeps
    a frozen build and a source checkout each pointing at their own ``data/``
    folder without the config having to know which it is.
    """

    store_dir: str = ""
    ledger: str = ""
    jobs_dir: str = ""
    chunk_chars: int = DEFAULT_CHUNK_CHARS
    overlap: int = DEFAULT_OVERLAP
    min_chunk: int = DEFAULT_MIN_CHUNK
    shard_bytes: int = DEFAULT_SHARD_BYTES
    incremental: bool = True
    allow_web: bool = False
    retries: int = 2
    """Bounded per-item retries for a transient local read failure."""
    normalize: bool = True
    """Normalize imported text (Unicode NFC + structure) before chunking.

    The plain pass is conservative: it never removes meaningful characters,
    punctuation, or diacritics. It only unifies line endings, drops control
    characters, and collapses whitespace.
    """
    fold_arabic: bool = False
    """Also unify Arabic letter forms (diacritics/tatweel dropped) -- opt-in.

    Off by default so Arabic letter distinctions and diacritics are preserved;
    turn it on only when matching the retriever's token normalization matters
    more than keeping the original forms.
    """

    def __post_init__(self) -> None:
        if self.chunk_chars < 1:
            raise ValueError("chunk_chars must be at least 1.")
        if not 0 <= self.overlap < self.chunk_chars:
            raise ValueError("overlap must be in [0, chunk_chars).")
        if self.min_chunk < 0:
            raise ValueError("min_chunk cannot be negative.")
        if self.shard_bytes < 1:
            raise ValueError("shard_bytes must be at least 1.")
        if self.retries < 0:
            raise ValueError("retries cannot be negative.")

    # -- resolution -------------------------------------------------------

    def resolved_store_dir(self) -> Path:
        """The store root: the configured path, else the project default."""
        return Path(self.store_dir) if self.store_dir else default_knowledge_base_dir()

    def resolved_ledger_path(self) -> Path:
        """The dedup ledger path, kept beside the store root under ``ingest/``."""
        from ingest.pipeline import ledger_path

        return Path(self.ledger) if self.ledger else ledger_path(self.resolved_store_dir())

    def resolved_jobs_dir(self) -> Path:
        """The resumable-job directory, beside the ledger under ``ingest/``."""
        from ingest.pipeline import ingest_jobs_dir

        return Path(self.jobs_dir) if self.jobs_dir else ingest_jobs_dir(self.resolved_store_dir())

    # -- serialisation ----------------------------------------------------

    def to_primitive(self) -> dict[str, object]:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @classmethod
    def from_primitive(cls, data: Mapping[str, object]) -> "IngestConfig":
        """Rebuild a config from stored data, ignoring unknown or invalid fields.

        A malformed stored value falls back to the built-in default for that
        field rather than failing the whole load, so a hand-edited file never
        makes the subsystem unstartable.
        """
        known = {field.name: field for field in fields(cls)}
        values: dict[str, object] = {}
        for name, field in known.items():
            if name not in data:
                continue
            try:
                values[name] = _coerce(data[name], field.type)
            except (TypeError, ValueError):
                continue
        try:
            return cls(**values)  # type: ignore[arg-type]
        except ValueError:
            return cls()


def _coerce(value: object, _type: object) -> object:
    """Coerce a stored/env string back to the field's declared type."""
    # ``from __future__ import annotations`` makes ``field.type`` a string, so we
    # compare as text rather than relying on the resolved type object.
    name = _type if isinstance(_type, str) else getattr(_type, "__name__", "")
    if name.endswith("bool"):
        return _as_bool(value)
    if name.endswith("int"):
        return int(str(value).strip())
    return str(value)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"Not a boolean: {value!r}")


def _env_overrides(environ: Mapping[str, str] | None) -> dict[str, object]:
    """``FAYD_INGEST_<FIELD>`` values keyed by field name, invalid ones dropped."""
    source = os.environ if environ is None else environ
    overrides: dict[str, object] = {}
    for field in fields(IngestConfig):
        raw = source.get(f"{ENV_PREFIX}{field.name.upper()}")
        if raw is None:
            continue
        try:
            overrides[field.name] = _coerce(raw, field.type)
        except (TypeError, ValueError):
            continue
    return overrides


def _config_store(root: str | Path | None) -> JsonStore:
    return JsonStore(Path(root) if root is not None else data_config_dir())


def data_config_dir() -> Path:
    """Where the ingestion config file lives: under the writable data folder."""
    from core.paths import data_dir

    return data_dir() / "ingest" / "config"


def load_ingest_config(
    root: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    **overrides: object,
) -> IngestConfig:
    """Load the stored config, apply environment overrides, then explicit ones.

    A malformed *stored* file never blocks startup: :meth:`IngestConfig.from_primitive`
    drops an unreadable field and falls back to the built-in default. But an
    *override* the caller (or the environment) supplied is checked strictly, so a
    bad combination such as ``chunk_chars`` below ``overlap`` raises a clear
    ``ValueError`` instead of being silently discarded. Passing ``**overrides``
    (e.g. ``store_dir=...``) pins a field regardless of the environment.
    """
    stored = _config_store(root).load(_NAMESPACE, _KEY)
    base = IngestConfig.from_primitive(stored) if isinstance(stored, dict) else IngestConfig()
    merged = base.to_primitive()
    overrides = dict(overrides)
    merged.update(_env_overrides(environ))
    merged.update({key: value for key, value in overrides.items() if key in merged})
    return IngestConfig(**merged)  # type: ignore[arg-type]


def save_ingest_config(config: IngestConfig, root: str | Path | None = None) -> None:
    """Persist ``config`` through the project's ``JsonStore``."""
    _config_store(root).save(_NAMESPACE, _KEY, config.to_primitive())


__all__ = [
    "ENV_PREFIX",
    "IngestConfig",
    "data_config_dir",
    "load_ingest_config",
    "save_ingest_config",
]
