"""Language models the user adds by hand, in one small registry.

The project ships a catalog of known-good GGUF models per mouth (Ultron's,
Vision's, Banner's, the companion's), and a dedicated model window can offer
*every* GGUF the project has (:mod:`core.language_models`). That still leaves one
gap the user asked to close: a GGUF file already on this machine -- a model the
catalog does not name -- cannot be chosen from a window at all.

This module is that registry. A user points a per-model window at a ``.gguf`` file
and it is remembered, with a readable title, under a stable ``custom:<slug>`` key.
The file is referenced **in place** rather than copied: a GGUF is often several
gigabytes, so copying it into ``data/`` would double the disk cost for nothing.
The honest consequence is that a file the user later moves or deletes is reported
as unavailable (its real path is checked on every read), exactly like a catalog
model whose weights are missing.

The registry is plain persistence -- no model is loaded, no Tk is imported, and
the cognitive core is untouched -- so it is dependency-free and testable with no
display. A save is :func:`core.best_effort.best_effort` (a read-only data folder
must not crash a window) but the failure is recorded rather than lost, and a
corrupt store falls back to an empty registry so a window always opens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from core.best_effort import best_effort
from core.paths import data_dir

#: The JsonStore namespace the custom models live under.
NAMESPACE = "custom_models"

#: The prefix a custom model's key carries, so it can never be confused with a
#: catalog key (a catalog key is a bare name like ``qwen2.5-7b``).
CUSTOM_PREFIX = "custom:"

#: A filesystem-safe slug is derived from the file's stem; anything outside this
#: set becomes a single dash, so a key is stable and readable.
_SLUG_KEEP = re.compile(r"[^a-z0-9]+")


def is_custom_key(key: str | None) -> bool:
    """True when ``key`` names a user-added model rather than a catalog one."""
    return bool(key) and str(key).strip().startswith(CUSTOM_PREFIX)


def _slug(text: str) -> str:
    """A readable, filesystem-safe slug for a model key."""
    cleaned = _SLUG_KEEP.sub("-", text.strip().lower()).strip("-")
    return cleaned or "model"


def model_key(path: str | Path, title: str = "") -> str:
    """The stable ``custom:<slug>`` key for a file, derived from its stem/title.

    The slug comes from the file's stem (or the given title), so a key is
    readable and stable for a given file. Two files with the same stem resolve to
    the same slug; :func:`add_model` disambiguates with a numeric suffix rather
    than overwriting.
    """
    stem = Path(path).stem if str(path).strip() else ""
    source = title.strip() or stem or "model"
    return f"{CUSTOM_PREFIX}{_slug(source)}"


@dataclass(frozen=True)
class CustomModel:
    """One user-added GGUF model: its key, a readable title, and its real file."""

    key: str
    title: str
    path: Path

    def available(self) -> bool:
        """True when the file is still where the user put it."""
        return self.path.is_file()

    def to_primitive(self) -> dict[str, str]:
        return {"key": self.key, "title": self.title, "path": str(self.path)}


def _store():
    from storage.json_store import JsonStore

    return JsonStore(data_dir())


def _clean_title(title: str, path: Path) -> str:
    return (title or "").strip() or path.stem


def list_models() -> tuple[CustomModel, ...]:
    """Every remembered custom model, in insertion-independent (sorted) order.

    A corrupt store, an unreadable entry, or a value of the wrong shape is skipped
    rather than raising, so a window always opens with whatever is readable.
    """
    try:
        raw = _store().load(NAMESPACE, "registry")
    except Exception:
        return ()
    if not isinstance(raw, dict):
        return ()
    entries = raw.get("models")
    if not isinstance(entries, list):
        return ()
    models: list[CustomModel] = []
    seen: set[str] = set()
    for item in entries:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        path = str(item.get("path") or "").strip()
        if not key or not path or key in seen or not is_custom_key(key):
            continue
        seen.add(key)
        models.append(
            CustomModel(key=key, title=str(item.get("title") or "").strip() or Path(path).stem,
                        path=Path(path))
        )
    models.sort(key=lambda model: model.title.casefold())
    return tuple(models)


def find_model(key: str | None) -> CustomModel | None:
    """The custom model ``key`` names, or ``None``."""
    wanted = (key or "").strip()
    if not wanted:
        return None
    for model in list_models():
        if model.key == wanted:
            return model
    return None


def _save(models: list[CustomModel]) -> None:
    with best_effort("save the custom-model registry"):
        _store().save(
            NAMESPACE,
            "registry",
            {"models": [model.to_primitive() for model in models]},
        )


def add_model(path: str | Path, *, title: str = "") -> CustomModel:
    """Remember ``path`` as a selectable model and return its entry.

    Raises :class:`ValueError` when there is no file there, or it is not a
    ``.gguf`` -- a bad choice is reported rather than stored. An existing entry for
    the same file is replaced (so re-adding updates the title); a different file
    whose slug collides gets a numeric suffix rather than overwriting the first.
    """
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise ValueError(f"There is no file at {candidate}.")
    if candidate.suffix.lower() != ".gguf":
        raise ValueError(
            "Choose a GGUF file (a model file ending in .gguf)."
        )
    resolved = candidate.resolve()
    key = model_key(resolved, title)
    existing = list(list_models())
    same_path = [model for model in existing if model.path.resolve() == resolved]
    if same_path:
        # Re-adding the same file updates its title; drop the stale entries first.
        existing = [model for model in existing if model.path.resolve() != resolved]
    taken = {model.key for model in existing}
    base = key
    counter = 2
    while key in taken:
        key = f"{base}-{counter}"
        counter += 1
    entry = CustomModel(key=key, title=_clean_title(title, resolved), path=resolved)
    existing.append(entry)
    _save(existing)
    return entry


def remove_model(key: str) -> bool:
    """Forget the custom model ``key``. ``True`` when one was removed."""
    models = list_models()
    kept = [model for model in models if model.key != (key or "").strip()]
    if len(kept) == len(models):
        return False
    _save(kept)
    return True


__all__ = [
    "CUSTOM_PREFIX",
    "NAMESPACE",
    "CustomModel",
    "add_model",
    "find_model",
    "is_custom_key",
    "list_models",
    "model_key",
    "remove_model",
]
