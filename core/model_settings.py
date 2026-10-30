"""Per-model inference settings, and the one place their precedence is decided.

The project already stores inference parameters -- ``core.character_config.ModelConfig``
holds ``temperature``/``top_p``/``context_length``/``max_tokens`` -- but no engine
reads them, so the *effective* settings today are the constants hard-coded in each
engine (Ultron 4096/512/0.7/0.95, Vision 4096/512/0.6/0.95, the two Studios 8192).
That is fine for a mouth with one model, and wrong for a model manager: the same
GGUF can be run by several mouths, and a user who tunes one should not have to
edit three modules.

This module is the missing store, and it is deliberately **additive**: it does not
touch an engine's constants and does not replace the character configuration. It
adds one record per *model identity* and one function that resolves what a caller
should actually use:

    explicit argument
      → this store, keyed by the model's identity
        → the character's own ModelConfig (when the caller is a character mouth)
          → the engine's own default

:func:`effective_settings` is that function, in one place, so no two callers can
disagree. The identity key is a **stable** string -- ``repo/filename`` for a
catalog model, ``custom:<path>`` for a user-added file, ``fayd-llm`` for the
trained checkpoint -- never an internal id that changes between runs (the lesson
:mod:`core.knowledge_carry` already records).

Persistence is :class:`storage.json_store.JsonStore`, which writes atomically, so
a crash cannot leave a half-written record. A corrupt or unreadable store yields
defaults rather than raising, and a save is :func:`core.best_effort.best_effort`
(a read-only data folder must not crash a window) while still being *recorded*.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from core.best_effort import best_effort
from core.paths import data_dir

#: The JsonStore namespace every model's settings live under.
NAMESPACE = "model_settings"

#: The bounds. They mirror the ceilings the engines already enforce, so a value
#: accepted here can never be rejected by the engine that receives it.
TEMPERATURE_RANGE = (0.0, 2.0)
TOP_P_RANGE = (0.0, 1.0)
CONTEXT_RANGE = (1, 32768)
MAX_TOKENS_RANGE = (1, 4096)
THREADS_RANGE = (0, 256)
GPU_LAYERS_RANGE = (0, 1024)

#: What a model gets when nobody has chosen anything: the same numbers Ultron
#: has always used, so the default path is byte-identical to today's behaviour.
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_CONTEXT = 4096
DEFAULT_MAX_TOKENS = 512
DEFAULT_THREADS = 0
#: CPU by default, always. GPU offload is never turned on from a guess.
DEFAULT_GPU_LAYERS = 0

#: The identifier the trained checkpoint is keyed by (see ``core.language_models``).
TRAINED_IDENTITY = "fayd-llm"


class SettingsError(ValueError):
    """A setting is outside its allowed range, with the reason stated."""


@dataclass(frozen=True)
class ModelSettings:
    """One model's inference settings.

    Every field has a working default, so a model nobody has tuned still resolves
    to the numbers the project already used. ``threads=0`` and ``gpu_layers=0``
    mean "let llama.cpp decide" and "CPU only", exactly as the engines' own
    constants do.
    """

    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    context: int = DEFAULT_CONTEXT
    max_tokens: int = DEFAULT_MAX_TOKENS
    threads: int = DEFAULT_THREADS
    gpu_layers: int = DEFAULT_GPU_LAYERS

    def __post_init__(self) -> None:
        _check_range("temperature", self.temperature, TEMPERATURE_RANGE)
        _check_range("top_p", self.top_p, TOP_P_RANGE)
        _check_int_range("context", self.context, CONTEXT_RANGE)
        _check_int_range("max_tokens", self.max_tokens, MAX_TOKENS_RANGE)
        _check_int_range("threads", self.threads, THREADS_RANGE)
        _check_int_range("gpu_layers", self.gpu_layers, GPU_LAYERS_RANGE)
        if self.max_tokens > self.context:
            raise SettingsError(
                f"max_tokens ({self.max_tokens}) cannot exceed context ({self.context}): "
                "a reply has to fit in the window it is generated in."
            )

    def with_value(self, field_name: str, value: object) -> ModelSettings:
        """A copy with one field replaced, validated. Raises on an unknown field."""
        if field_name not in _FIELDS:
            raise SettingsError(
                f"Unknown setting {field_name!r}. Known settings: {', '.join(sorted(_FIELDS))}."
            )
        # ``replace`` is typed per-field, so a dynamic name cannot be checked
        # statically; the constructor's own validation runs either way, which is
        # what actually keeps a bad value out.
        return replace(self, **{field_name: value})  # type: ignore[arg-type]

    def to_primitive(self) -> dict[str, float | int]:
        """A deterministic mapping -- sorted keys, JSON-safe values."""
        return {
            "context": int(self.context),
            "gpu_layers": int(self.gpu_layers),
            "max_tokens": int(self.max_tokens),
            "temperature": float(self.temperature),
            "threads": int(self.threads),
            "top_p": float(self.top_p),
        }

    @classmethod
    def from_primitive(cls, data: object) -> ModelSettings:
        """Read a stored record, falling back per field on anything unreadable.

        A single bad value does not discard the whole record: the rest of the
        user's choices are kept, and the unusable one returns to its default. A
        value of the wrong type is treated the same way, so a hand-edited file
        cannot make a window unusable.
        """
        if not isinstance(data, dict):
            return cls()
        return cls(
            temperature=_clamped_float(data.get("temperature"), DEFAULT_TEMPERATURE, TEMPERATURE_RANGE),
            top_p=_clamped_float(data.get("top_p"), DEFAULT_TOP_P, TOP_P_RANGE),
            context=_clamped_int(data.get("context"), DEFAULT_CONTEXT, CONTEXT_RANGE),
            max_tokens=_clamped_int(data.get("max_tokens"), DEFAULT_MAX_TOKENS, MAX_TOKENS_RANGE),
            threads=_clamped_int(data.get("threads"), DEFAULT_THREADS, THREADS_RANGE),
            gpu_layers=_clamped_int(data.get("gpu_layers"), DEFAULT_GPU_LAYERS, GPU_LAYERS_RANGE),
        )

    def describe(self) -> str:
        """A one-line, readable summary for a status surface."""
        return (
            f"temperature {self.temperature:g}, top_p {self.top_p:g}, "
            f"context {self.context}, max_tokens {self.max_tokens}, "
            f"threads {self.threads or 'auto'}, gpu_layers {self.gpu_layers}"
        )


_FIELDS: tuple[str, ...] = (
    "temperature", "top_p", "context", "max_tokens", "threads", "gpu_layers",
)


def _check_range(name: str, value: object, bounds: tuple[float, float]) -> None:
    low, high = bounds
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SettingsError(f"{name} must be a number.")
    if not low <= float(value) <= high:
        raise SettingsError(f"{name} must be in [{low:g}, {high:g}] (got {value!r}).")


def _check_int_range(name: str, value: object, bounds: tuple[int, int]) -> None:
    low, high = bounds
    if not isinstance(value, int) or isinstance(value, bool):
        raise SettingsError(f"{name} must be a whole number.")
    if not low <= value <= high:
        raise SettingsError(f"{name} must be in [{low}, {high}] (got {value!r}).")


def _as_float(value: object, fallback: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    return float(value)


def _as_int(value: object, fallback: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return fallback
    return int(value)


def _clamped_float(value: object, fallback: float, bounds: tuple[float, float]) -> float:
    """Read a stored float, clamping it into range rather than raising.

    A record is *user data*: a hand-edited file with ``temperature: 99`` must not
    make the reader raise, and silently dropping the field would discard the other
    choices beside it. Clamping keeps the record usable and the value in range --
    the same thing a runtime would do with it anyway. A value of the wrong *type*
    falls back, because clamping a non-number would be inventing one.
    """
    low, high = bounds
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float(fallback)
    return min(max(float(value), float(low)), float(high))


def _clamped_int(value: object, fallback: int, bounds: tuple[int, int]) -> int:
    """Read a stored int, clamping it into range rather than raising."""
    low, high = bounds
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return int(fallback)
    return min(max(int(value), int(low)), int(high))


def _store(root: Path | str | None = None):
    from storage.json_store import JsonStore

    return JsonStore(Path(root) if root is not None else data_dir())


def load_settings(
    identity: str, *, root: Path | str | None = None
) -> ModelSettings:
    """The saved settings for ``identity``, or the project defaults.

    An unknown identity, a store that cannot be read, and a record of the wrong
    shape all yield the defaults -- so a listing never fails because one record
    drifted. The identity is used as the store *key*, so it must be a stable
    string; an empty one yields the defaults rather than writing under "".
    """
    key = _normalise_identity(identity)
    if not key:
        return ModelSettings()
    try:
        saved = _store(root).load(NAMESPACE, key)
    except Exception:
        return ModelSettings()
    return ModelSettings.from_primitive(saved)


def save_settings(
    identity: str,
    settings: ModelSettings,
    *,
    root: Path | str | None = None,
) -> ModelSettings:
    """Persist ``settings`` for ``identity`` and return what was stored.

    The write is :func:`core.best_effort.best_effort`, so a read-only data folder
    cannot crash a window -- but the failure is *recorded* rather than discarded,
    so a real problem is still discoverable.
    """
    key = _normalise_identity(identity)
    if not key:
        raise SettingsError("A model identity is required to save its settings.")
    with best_effort(f"save the settings for {key}"):
        _store(root).save(NAMESPACE, key, settings.to_primitive())
    return settings


def update_settings(
    identity: str,
    *,
    root: Path | str | None = None,
    **changes: object,
) -> ModelSettings:
    """Apply ``changes`` to the saved settings for ``identity`` and store them.

    This is the safe-update path: the current record is read, each named field is
    validated against its range, and the merged result is written once. An
    invalid value raises :class:`SettingsError` *before* anything is stored, so a
    bad edit can never leave a half-applied record.
    """
    current = load_settings(identity, root=root)
    updated = current
    for field_name, value in changes.items():
        if value is None:
            continue
        updated = updated.with_value(field_name, value)
    return save_settings(identity, updated, root=root)


def reset_settings(identity: str, *, root: Path | str | None = None) -> ModelSettings:
    """Return ``identity`` to the project defaults and store that."""
    return save_settings(identity, ModelSettings(), root=root)


def _normalise_identity(identity: str) -> str:
    """The store key for a model identity. Empty in, empty out."""
    return str(identity or "").strip()


def effective_settings(
    identity: str,
    *,
    root: Path | str | None = None,
    character: object = None,
    explicit: ModelSettings | None = None,
) -> ModelSettings:
    """Resolve what a model should actually be run with, in one documented order.

    1. ``explicit`` -- a caller that knows what it wants wins outright.
    2. the saved record for ``identity``, **when one exists**;
    3. ``character`` -- a :class:`core.character_config.ModelConfig` (or anything
       exposing the same four fields), when the caller is a character's mouth;
    4. the project defaults.

    Step 2 and step 3 are deliberately ordered this way: a setting a user chose
    *for this model* is more specific than a setting stored for the character that
    happens to run it. Both are consulted, so a character's configuration is never
    silently ignored -- it is simply outranked by a per-model choice.

    A ``character`` whose values are out of range is skipped rather than allowed to
    raise: a stale character file must not stop a model from being used.
    """
    if explicit is not None:
        return explicit
    saved = _saved_or_none(identity, root=root)
    if saved is not None:
        return saved
    if character is not None:
        from_character = _settings_from_character(character)
        if from_character is not None:
            return from_character
    return ModelSettings()


def _saved_or_none(identity: str, *, root: Path | str | None) -> ModelSettings | None:
    """The saved record for ``identity``, or ``None`` when there is genuinely none.

    Distinguishes "the user saved the defaults" from "nothing is saved", which is
    what lets step 3 of :func:`effective_settings` ever run.
    """
    key = _normalise_identity(identity)
    if not key:
        return None
    try:
        raw = _store(root).load(NAMESPACE, key)
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    return ModelSettings.from_primitive(raw)


def _settings_from_character(character: object) -> ModelSettings | None:
    """A character's ``ModelConfig`` as :class:`ModelSettings`, or ``None``."""
    mapping = {
        "temperature": getattr(character, "temperature", None),
        "top_p": getattr(character, "top_p", None),
        "context": getattr(character, "context_length", None),
        "max_tokens": getattr(character, "max_tokens", None),
    }
    if all(value is None for value in mapping.values()):
        return None
    merged: dict[str, object] = {}
    for field_name, value in mapping.items():
        if value is not None:
            merged[field_name] = value
    try:
        return ModelSettings(**merged)  # type: ignore[arg-type]
    except SettingsError:
        # A stale character value out of range is skipped, not fatal: the model
        # still runs with the project defaults.
        return None


def settings_store_path(root: Path | str | None = None) -> Path:
    """Where the settings records live, for a diagnostic or a backup."""
    base = Path(root) if root is not None else data_dir()
    return base / NAMESPACE


__all__ = [
    "DEFAULT_CONTEXT",
    "DEFAULT_GPU_LAYERS",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_THREADS",
    "DEFAULT_TOP_P",
    "NAMESPACE",
    "TRAINED_IDENTITY",
    "ModelSettings",
    "SettingsError",
    "effective_settings",
    "load_settings",
    "reset_settings",
    "save_settings",
    "settings_store_path",
    "update_settings",
]
