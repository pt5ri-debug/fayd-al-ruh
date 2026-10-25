"""Per-mouth settings: the character's name, logo, reading voice, and model.

Each answering mouth -- Jarvis, Ultron, Vision, Banner, Natasha, Hakeem, and Cloud
-- gets its own dedicated chat window (see :mod:`gui.model_chat`). Three of that
window's controls belong to the *mouth* and must survive a restart, each kept
separate from the others:

* the **logo** (an image the user may replace, copied into ``data/characters``);
* the **reading voice** the mouth's answers are read aloud with -- exclusive to
  that mouth, so choosing a voice for Ultron never changes Vision's;
* the **language model** the mouth answers with.

One profile per mouth is persisted through :class:`storage.json_store.JsonStore`
under a single namespace, so the whole set is one small, inspectable file tree.
Nothing here loads a model, imports Tk, or touches the cognitive core: it is
plain persistence, dependency-free and testable with no display.

The module is honest about failure: an unreadable store, a saved voice or model
that is no longer offered, and a saved logo that has been deleted all fall back
to the default rather than raising, so a stale file can never make a window
unusable. A copy of a chosen logo is bounded and validated by extension, and a
failure to copy is recorded through :func:`core.best_effort.best_effort` rather
than swallowed invisibly.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path

from core.best_effort import best_effort
from core.language_models import MOUTHS, find_language_model
from core.paths import characters_dir

#: The JsonStore namespace every mouth profile lives under.
NAMESPACE = "mouths"

#: The image extensions a logo may use. Kept to what Tk can load on every
#: platform (``tk.PhotoImage`` reads PNG, GIF, PPM/PGM; Pillow widens this when it
#: is installed), so a chosen logo always has a chance of being shown.
LOGO_SUFFIXES: tuple[str, ...] = (".png", ".gif", ".ppm", ".pgm", ".jpg", ".jpeg", ".webp")

#: The default display name per mouth, used until the user renames one.
DEFAULT_NAMES: dict[str, str] = {
    "sources": "Jarvis",
    "model": "Fayd",
    "ultron": "Ultron",
    "vision": "Vision",
    "banner": "Banner",
    "cloud": "Cloud",
    "natasha": "Natasha",
    "hakeem": "Hakeem",
}


@dataclass(frozen=True)
class MouthProfile:
    """One mouth's saved character: its name, its logo, its voice, and its model.

    Every field is optional: an empty ``logo`` means "draw the default badge",
    and an empty ``voice``/``model`` means "use the mouth's own default". A profile
    is frozen so a caller can hand it around without another thread mutating it.
    """

    mouth: str
    name: str
    logo: str = ""
    voice: str = ""
    model: str = ""

    def with_logo(self, filename: str) -> MouthProfile:
        return replace(self, logo=filename)

    def with_name(self, name: str) -> MouthProfile:
        return replace(self, name=name.strip() or DEFAULT_NAMES.get(self.mouth, self.mouth))

    def with_voice(self, voice: str) -> MouthProfile:
        return replace(self, voice=voice)

    def with_model(self, model: str) -> MouthProfile:
        return replace(self, model=model)

    def logo_path(self) -> Path | None:
        """The saved logo file, or ``None`` when none is set or it is gone."""
        if not self.logo:
            return None
        candidate = characters_dir() / self.logo
        return candidate if candidate.is_file() else None

    def to_primitive(self) -> dict[str, str]:
        return {
            "name": self.name,
            "logo": self.logo,
            "voice": self.voice,
            "model": self.model,
        }


def default_profile(mouth: str) -> MouthProfile:
    """The profile a mouth starts with: its own name and no overrides."""
    return MouthProfile(mouth=mouth, name=DEFAULT_NAMES.get(mouth, mouth))


def _store():
    from storage.json_store import JsonStore

    return JsonStore(characters_dir().parent)


def load_profile(mouth: str) -> MouthProfile:
    """Read a mouth's saved profile, falling back to the default on any problem.

    A missing profile, a store that cannot be read, or a saved value of the wrong
    shape all yield the default rather than raising, so a window always opens.
    """
    base = default_profile(mouth)
    try:
        saved = _store().load(NAMESPACE, mouth)
    except Exception:
        return base
    if not isinstance(saved, dict):
        return base
    name = str(saved.get("name") or "").strip()
    return MouthProfile(
        mouth=mouth,
        name=name or base.name,
        logo=str(saved.get("logo") or "").strip(),
        voice=str(saved.get("voice") or "").strip(),
        model=str(saved.get("model") or "").strip(),
    )


def save_profile(profile: MouthProfile) -> MouthProfile:
    """Persist ``profile``. Returns it, so a caller can chain.

    A store error is swallowed through :func:`best_effort`: saving a cosmetic
    preference must not fail just because the data folder is read-only, but the
    failure is still recorded rather than lost.
    """
    with best_effort("save a mouth profile"):
        _store().save(NAMESPACE, profile.mouth, profile.to_primitive())
    return profile


def set_name(mouth: str, name: str) -> MouthProfile:
    """Rename a mouth's character and save it."""
    return save_profile(load_profile(mouth).with_name(name))


def set_voice(mouth: str, voice: str) -> MouthProfile:
    """Remember the reading voice for one mouth (exclusive to it) and save it."""
    return save_profile(load_profile(mouth).with_voice(voice))


def set_model(mouth: str, model: str) -> MouthProfile:
    """Remember the language model for one mouth and save it."""
    return save_profile(load_profile(mouth).with_model(model))


def logo_filename(mouth: str, source: str | Path) -> str:
    """The stable filename a mouth's logo is copied to, from ``source``.

    The name is ``<mouth><suffix>`` so it is obvious which mouth owns it, and the
    suffix is normalised (lower-cased, ``.jpeg`` folded to ``.jpg``) so two copies
    of the same picture do not accumulate.
    """
    suffix = Path(source).suffix.lower()
    if suffix == ".jpeg":
        suffix = ".jpg"
    if suffix not in LOGO_SUFFIXES:
        suffix = ".png"
    return f"{mouth}{suffix}"


def save_logo(mouth: str, source: str | Path) -> MouthProfile:
    """Copy ``source`` in as ``mouth``'s logo and remember it.

    Raises :class:`ValueError` for an unsupported extension, so a bad choice is
    reported rather than silently stored; the copy itself is best-effort (a
    read-only data folder must not crash the window) but its failure is recorded.
    Any previous logo for the mouth is removed, so only one file per mouth exists.
    """
    path = Path(source)
    if not path.is_file():
        raise ValueError(f"There is no image at {path}.")
    if path.suffix.lower() not in LOGO_SUFFIXES:
        raise ValueError(
            "Choose a PNG, GIF, JPEG, or WebP image "
            f"(got {path.suffix or 'no extension'})."
        )
    folder = characters_dir()
    filename = logo_filename(mouth, path)
    target = folder / filename
    with best_effort("copy a mouth logo"):
        folder.mkdir(parents=True, exist_ok=True)
        # Drop any other logo the mouth had, whatever extension it used.
        for stale in folder.glob(f"{mouth}.*"):
            if stale != target and stale.suffix.lower() in LOGO_SUFFIXES:
                with best_effort("remove a replaced mouth logo"):
                    stale.unlink()
        shutil.copy2(path, target)
    profile = load_profile(mouth).with_logo(filename)
    return save_profile(profile)


def clear_logo(mouth: str) -> MouthProfile:
    """Remove a mouth's logo (back to the default badge) and save the change."""
    current = load_profile(mouth)
    path = current.logo_path()
    if path is not None:
        with best_effort("remove a mouth logo"):
            path.unlink()
    return save_profile(current.with_logo(""))


def resolve_voice(mouth: str, available: tuple[str, ...] = ()) -> str:
    """The voice key a mouth should use: its saved one, else the first available.

    A saved voice that is no longer installed (its engine was removed, or a
    ``system:``/``model:`` voice uninstalled) falls back to the first available
    key rather than leaving the menu with nothing selected. ``available`` is what
    the caller can actually offer; an empty tuple means "the caller will resolve
    the default itself", and the saved value is returned unchanged.
    """
    saved = load_profile(mouth).voice
    if not available:
        return saved
    if saved and saved in available:
        return saved
    return available[0]


def resolve_model(mouth: str) -> str:
    """The language-model key a mouth should use: its saved one when still offered.

    A saved model that left the catalog falls back to ``""`` -- meaning "the
    mouth's own default" -- rather than pointing the loader at a model it does not
    know, so a stale choice can never break the mouth.
    """
    saved = load_profile(mouth).model
    if saved and find_language_model(mouth, saved) is not None:
        return saved
    return ""


def all_profiles() -> tuple[MouthProfile, ...]:
    """Every known mouth's profile, in the catalog's own order."""
    return tuple(load_profile(mouth) for mouth in MOUTHS)


__all__ = [
    "DEFAULT_NAMES",
    "LOGO_SUFFIXES",
    "NAMESPACE",
    "MouthProfile",
    "all_profiles",
    "clear_logo",
    "default_profile",
    "load_profile",
    "logo_filename",
    "resolve_model",
    "resolve_voice",
    "save_logo",
    "save_profile",
    "set_model",
    "set_name",
    "set_voice",
]
