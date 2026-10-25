"""Per-engine speech-studio settings: the chosen voice and mood, and the logo.

Each *speech engine* the project can speak through -- the voices the OS installed,
VoiceTut-TTS over audio.cpp, Piper, and Qwen3-TTS -- gets its own studio window (see
:mod:`gui.voice_studio`). Two of that window's controls belong to the engine and
must survive a restart, each kept separate from the others:

* the **voice** the studio reads documents with -- exclusive to that engine, so
  choosing a Piper voice never changes the Qwen3-TTS studio's;
* the **mood** (an emotional state such as happy, sad, excited, rapturous, in love,
  or angry) applied to that voice -- also exclusive, and applied the moment it is
  chosen;
* the **logo** the studio shows for itself (an image the user may replace, copied
  into ``data/characters`` beside the per-mouth logos).

One profile per engine is persisted through :class:`storage.json_store.JsonStore`
under a single namespace, so the whole set is one small, inspectable file tree.
Nothing here loads a model, imports Tk, or touches the cognitive core: it is plain
persistence, dependency-free and testable with no display.

The module is honest about failure: an unreadable store, a saved voice or mood that
is no longer offered, and a saved logo that has been deleted all fall back to the
default rather than raising, so a stale file can never make a window unusable.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path

from core.best_effort import best_effort
from core.paths import characters_dir
from core.voice_moods import DEFAULT_MOOD_KEY, mood_keys
from core.voices import (
    SOURCE_AUDIOCPP,
    SOURCE_PIPER,
    SOURCE_QWEN,
    SOURCE_SYSTEM,
)

#: The JsonStore namespace every speech-studio profile lives under.
NAMESPACE = "voice_studios"

#: The four speech-engine sources that get their own studio window, in the order
#: the composer row shows them (right to left in the window; this is left-to-right
#: for the data). The value is the human title the window and the button carry --
#: exactly the names the user asked for.
STUDIO_SOURCES: tuple[tuple[str, str], ...] = (
    (SOURCE_SYSTEM, "Installed Voices (Arabic + English)"),
    (SOURCE_AUDIOCPP, "VoiceTut-TTS (Local Arabic)"),
    (SOURCE_PIPER, "Piper (Arabic + English)"),
    (SOURCE_QWEN, "Qwen3-TTS (English)"),
)

#: The image extensions a studio logo may use. Kept to what Tk can load on every
#: platform (``tk.PhotoImage`` reads PNG, GIF, PPM/PGM; Pillow widens this when it
#: is installed), so a chosen logo always has a chance of being shown.
LOGO_SUFFIXES: tuple[str, ...] = (".png", ".gif", ".ppm", ".pgm", ".jpg", ".jpeg", ".webp")

#: The default display name per source, used until the user renames one.
DEFAULT_NAMES: dict[str, str] = {
    SOURCE_SYSTEM: "Installed Voices",
    SOURCE_AUDIOCPP: "VoiceTut-TTS",
    SOURCE_PIPER: "Piper",
    SOURCE_QWEN: "Qwen3-TTS",
}


def studio_title(source: str) -> str:
    """The human title of a speech-engine studio (its button and window label)."""
    for key, title in STUDIO_SOURCES:
        if key == source:
            return title
    return source


def source_names() -> tuple[str, ...]:
    """Every source that has a studio, in menu order."""
    return tuple(source for source, _title in STUDIO_SOURCES)


@dataclass(frozen=True)
class VoiceStudioProfile:
    """One engine's saved studio: its name, its logo, its voice, and its mood.

    Every field is optional: an empty ``logo`` means "draw the default badge", an
    empty ``voice`` means "the engine's first available voice", and an empty
    ``mood`` means the neutral mood. A profile is frozen so a caller can hand it
    around without another thread mutating it.
    """

    source: str
    name: str
    logo: str = ""
    voice: str = ""
    mood: str = ""

    def with_logo(self, filename: str) -> VoiceStudioProfile:
        return replace(self, logo=filename)

    def with_name(self, name: str) -> VoiceStudioProfile:
        return replace(self, name=name.strip() or DEFAULT_NAMES.get(self.source, self.source))

    def with_voice(self, voice: str) -> VoiceStudioProfile:
        return replace(self, voice=voice)

    def with_mood(self, mood: str) -> VoiceStudioProfile:
        return replace(self, mood=mood)

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
            "mood": self.mood,
        }


def default_profile(source: str) -> VoiceStudioProfile:
    """The profile a studio starts with: its own name and no overrides."""
    return VoiceStudioProfile(source=source, name=DEFAULT_NAMES.get(source, source))


def _store():
    from storage.json_store import JsonStore

    return JsonStore(characters_dir().parent)


def load_profile(source: str) -> VoiceStudioProfile:
    """Read a studio's saved profile, falling back to the default on any problem.

    A missing profile, a store that cannot be read, or a saved value of the wrong
    shape all yield the default rather than raising, so a window always opens.
    """
    base = default_profile(source)
    try:
        saved = _store().load(NAMESPACE, source)
    except Exception:
        return base
    if not isinstance(saved, dict):
        return base
    name = str(saved.get("name") or "").strip()
    return VoiceStudioProfile(
        source=source,
        name=name or base.name,
        logo=str(saved.get("logo") or "").strip(),
        voice=str(saved.get("voice") or "").strip(),
        mood=str(saved.get("mood") or "").strip(),
    )


def save_profile(profile: VoiceStudioProfile) -> VoiceStudioProfile:
    """Persist ``profile``. Returns it, so a caller can chain.

    A store error is swallowed through :func:`best_effort`: saving a cosmetic
    preference must not fail just because the data folder is read-only, but the
    failure is still recorded rather than lost.
    """
    with best_effort("save a voice studio profile"):
        _store().save(NAMESPACE, profile.source, profile.to_primitive())
    return profile


def set_name(source: str, name: str) -> VoiceStudioProfile:
    """Rename a studio and save it."""
    return save_profile(load_profile(source).with_name(name))


def set_voice(source: str, voice: str) -> VoiceStudioProfile:
    """Remember the reading voice for one engine (exclusive to it) and save it."""
    return save_profile(load_profile(source).with_voice(voice))


def set_mood(source: str, mood: str) -> VoiceStudioProfile:
    """Remember the mood for one engine (exclusive to it) and save it."""
    return save_profile(load_profile(source).with_mood(mood))


def logo_filename(source: str, source_path: str | Path) -> str:
    """The stable filename a studio's logo is copied to, from ``source_path``.

    The name is ``voice-<source><suffix>`` so it is obvious which studio owns it and
    it can never collide with a per-mouth logo (which is ``<mouth><suffix>``).
    """
    suffix = Path(source_path).suffix.lower()
    if suffix == ".jpeg":
        suffix = ".jpg"
    if suffix not in LOGO_SUFFIXES:
        suffix = ".png"
    return f"voice-{source}{suffix}"


def save_logo(source: str, source_path: str | Path) -> VoiceStudioProfile:
    """Copy ``source_path`` in as ``source``'s logo and remember it.

    Raises :class:`ValueError` for an unsupported extension, so a bad choice is
    reported rather than silently stored; the copy itself is best-effort (a
    read-only data folder must not crash the window) but its failure is recorded.
    Any previous logo for the studio is removed, so only one file per studio exists.
    """
    path = Path(source_path)
    if not path.is_file():
        raise ValueError(f"There is no image at {path}.")
    if path.suffix.lower() not in LOGO_SUFFIXES:
        raise ValueError(
            "Choose a PNG, GIF, JPEG, or WebP image "
            f"(got {path.suffix or 'no extension'})."
        )
    folder = characters_dir()
    filename = logo_filename(source, path)
    target = folder / filename
    with best_effort("copy a voice studio logo"):
        folder.mkdir(parents=True, exist_ok=True)
        for stale in folder.glob(f"voice-{source}.*"):
            if stale != target and stale.suffix.lower() in LOGO_SUFFIXES:
                with best_effort("remove a replaced voice studio logo"):
                    stale.unlink()
        shutil.copy2(path, target)
    return save_profile(load_profile(source).with_logo(filename))


def clear_logo(source: str) -> VoiceStudioProfile:
    """Remove a studio's logo (back to the default badge) and save the change."""
    current = load_profile(source)
    path = current.logo_path()
    if path is not None:
        with best_effort("remove a voice studio logo"):
            path.unlink()
    return save_profile(current.with_logo(""))


def resolve_voice(source: str, available: tuple[str, ...] = ()) -> str:
    """The voice key a studio should use: its saved one, else the first available.

    A saved voice that is no longer installed (its engine was removed, or a voice
    uninstalled) falls back to the first available key rather than leaving the menu
    with nothing selected. ``available`` is what the caller can actually offer; an
    empty tuple means "the caller will resolve the default itself", and the saved
    value is returned unchanged.
    """
    saved = load_profile(source).voice
    if not available:
        return saved
    if saved and saved in available:
        return saved
    return available[0]


def resolve_mood(source: str) -> str:
    """The mood key a studio should use: its saved one, else the neutral mood.

    A saved mood that left the catalog falls back to ``neutral`` rather than
    pointing the synthesizer at a state it does not know, so a stale choice can
    never break a studio.
    """
    saved = load_profile(source).mood
    if saved and saved in mood_keys():
        return saved
    return DEFAULT_MOOD_KEY


def all_profiles() -> tuple[VoiceStudioProfile, ...]:
    """Every known studio's profile, in the catalog's own order."""
    return tuple(load_profile(source) for source in source_names())


__all__ = [
    "DEFAULT_NAMES",
    "LOGO_SUFFIXES",
    "NAMESPACE",
    "STUDIO_SOURCES",
    "VoiceStudioProfile",
    "all_profiles",
    "clear_logo",
    "default_profile",
    "load_profile",
    "logo_filename",
    "resolve_mood",
    "resolve_voice",
    "save_logo",
    "save_profile",
    "set_mood",
    "set_name",
    "set_voice",
    "source_names",
    "studio_title",
]
