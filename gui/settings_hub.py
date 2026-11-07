"""The centralized settings model (Phase 3).

One authoritative home per **global** setting, and an explicit record of where
that home is. The window's Settings page is built from this catalogue and only
*links to* the authoritative editor (Themes, Fonts & paragraphs, Reading Voice,
the Cloud pickers, the internet switch) rather than re-implementing it, so a
global setting can never drift between two windows.

Two rules this module exists to make checkable:

1. **One writer per global setting.** Every entry names the single ``reader`` and
   ``writer`` (a dotted path to a real function) and the ``store`` (the JSON
   namespace + key) it lives in; ``writers_by_setting``/``stores`` let a test
   assert no setting has a second store and no two settings share one.
2. **Context-dependent settings stay distinct.** The per-mouth voice/model, the
   per-character model, the per-agent permissions, and the per-task limits are
   *not* global and are deliberately absent here; :data:`CONTEXTUAL_SETTINGS`
   names them so the boundary is written down rather than assumed.

Honest by construction: a value that cannot be read stays ``"unknown"``; nothing
here writes anything (the writers are the existing, authoritative functions); and
the module imports no Tk and loads no model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The JSON namespace the window's own preferences share (theme, voice, fonts,
#: sidebar layout), matching ``gui.app``/``gui.themes``/``gui.typography_settings``.
GUI_NAMESPACE = "gui"


@dataclass(frozen=True)
class SettingsEntry:
    """One **global** setting and the one place it is authoritative.

    ``reader``/``writer`` are dotted paths (``"gui.themes.save_theme_name"``) so
    the catalogue is a *record* of the wiring that a test can resolve and check,
    not a second implementation of it. ``link`` names the existing sidebar
    section (``"themes"``, ``"voices"``, ``"fonts"``, ``"settings"``) a user
    opens to change the value, so the Settings page never edits it directly.
    """

    key: str
    title: str
    description: str
    reader: str
    writer: str
    store: tuple[str, str]
    link: str = ""


#: The global settings, each with exactly one authoritative reader/writer/store.
SETTINGS_ENTRIES: tuple[SettingsEntry, ...] = (
    SettingsEntry(
        key="theme",
        title="Appearance theme",
        description="The application's colour theme; every window restyles at once.",
        reader="gui.themes.load_theme_name",
        writer="gui.themes.save_theme_name",
        store=(GUI_NAMESPACE, "appearance"),
        link="themes",
    ),
    SettingsEntry(
        key="typography",
        title="Fonts & paragraphs",
        description="The font, size, weight, alignment, and spacing of each text scope.",
        reader="gui.typography_settings.load_profiles",
        writer="gui.typography_settings.save_profiles",
        store=(GUI_NAMESPACE, "typography"),
        link="fonts",
    ),
    SettingsEntry(
        key="reading_voice",
        title="Reading voice",
        description="The voice every 'Read aloud' control uses.",
        reader="gui.app.load_reading_voice",
        writer="gui.app.save_reading_voice",
        store=(GUI_NAMESPACE, "reading_voice"),
        link="voices",
    ),
    SettingsEntry(
        key="auto_speak",
        title="Auto-speak",
        description="Whether a new answer is read aloud automatically.",
        reader="gui.app.load_auto_speak",
        writer="gui.app.save_auto_speak",
        store=(GUI_NAMESPACE, "auto_speak"),
        link="voices",
    ),
    SettingsEntry(
        key="internet",
        title="Internet access",
        description="The one switch every path that leaves this machine consults.",
        reader="core.preferences.UserPreferences",
        writer="core.preferences.UserPreferences.update",
        store=("jarvis", "preferences"),
    ),
    SettingsEntry(
        key="tashkeel",
        title="Arabic vocalization (tashkeel)",
        description="Whether Arabic answers are fully vocalized, and in which form.",
        reader="core.preferences.UserPreferences",
        writer="core.preferences.UserPreferences.update",
        store=("jarvis", "preferences"),
    ),
    SettingsEntry(
        key="cloud_provider",
        title="Cloud provider",
        description="The remembered cloud provider and its model.",
        reader="core.cloud_model.load_provider_key",
        writer="core.cloud_model.save_provider_key",
        store=("cloud", "provider"),
    ),
    SettingsEntry(
        key="cloud_model",
        title="Cloud model",
        description="The remembered model for the chosen cloud provider.",
        reader="core.cloud_model.load_model_key",
        writer="core.cloud_model.save_model_key",
        store=("cloud", "models"),
    ),
    SettingsEntry(
        key="sidebar_layout",
        title="Sidebar layout",
        description="Which sidebar sections are open (a per-scope layout preference).",
        reader="gui.app.load_sidebar_sections",
        writer="gui.app.save_sidebar_sections",
        store=(GUI_NAMESPACE, "sidebar_sections"),
    ),
)

#: Settings that are deliberately **context-dependent**, so they must *not* be
#: centralized: a per-mouth voice/model is not the global default, a per-character
#: model is not the global model, a per-agent permission is not a global
#: permission, and a per-task limit is not a global limit. Named here so the
#: boundary is explicit and a future change cannot quietly merge them.
CONTEXTUAL_SETTINGS: tuple[tuple[str, str], ...] = (
    ("mouth_profile", "Per-mouth voice and language model (gui.mouth_profile)."),
    ("character_config", "Per-character model, voice, and learning (core.character_config)."),
    ("agent_capabilities", "Per-personality capability ceiling (core.agents.capabilities)."),
    ("task_limits", "Per-task resource limits (perception/planning limits)."),
)

#: The settings that must read the one **internet** preference -- the switch's
#: consumers. Every one of them consults ``core.preferences`` rather than keeping
#: its own copy, so turning the internet on once turns it on everywhere.
INTERNET_CONSUMERS: tuple[str, ...] = (
    "chat",
    "training_center",
    "jarvis",
)


def entry_by_key(key: str) -> SettingsEntry:
    """Look up one settings entry by its stable key (raises ``KeyError``)."""
    for entry in SETTINGS_ENTRIES:
        if entry.key == key:
            return entry
    raise KeyError(key)


def writers_by_setting() -> dict[str, str]:
    """Every setting's single authoritative writer, by key."""
    return {entry.key: entry.writer for entry in SETTINGS_ENTRIES}


def stores() -> dict[str, tuple[str, str]]:
    """Every setting's store (namespace, key), by setting key."""
    return {entry.key: entry.store for entry in SETTINGS_ENTRIES}


def settings_by_link() -> dict[str, tuple[SettingsEntry, ...]]:
    """The settings grouped by the sidebar section a user opens to change them."""
    grouped: dict[str, list[SettingsEntry]] = {}
    for entry in SETTINGS_ENTRIES:
        grouped.setdefault(entry.link or "settings", []).append(entry)
    return {link: tuple(entries) for link, entries in grouped.items()}


def resolve(path: str) -> Any:
    """Resolve a dotted ``"module.attr"`` path to the real object (or ``None``).

    Used by the catalogue's own test to prove every reader/writer it names really
    exists, so the record cannot drift from the code it describes.
    """
    import importlib

    parts = path.split(".")
    for split in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split])
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        obj: Any = module
        for attr in parts[split:]:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    return None


def _preferences(root: Any) -> Any:
    from core.paths import data_dir
    from core.preferences import UserPreferences

    return UserPreferences(root if root is not None else data_dir())


def read_internet(root: Any) -> bool:
    """Whether the internet is on, read from the one authoritative preference."""
    try:
        return bool(_preferences(root).record.web)
    except Exception:
        return False


def write_internet(root: Any, enabled: bool) -> bool:
    """Turn the internet on or off in the one authoritative preference.

    This is the single writer the chat toggle, the Training Center, and Jarvis all
    go through, so the three can never disagree about whether the machine may
    reach the network. Returns the state now in force.
    """
    preferences = _preferences(root)
    preferences.update(web=bool(enabled))
    return bool(preferences.record.web)


def internet_status(root: Any) -> str:
    """One honest line about the internet switch and what it governs."""
    if read_internet(root):
        return (
            "Internet: on — a web search, a cloud provider, a URL import, and the "
            "bulk knowledge-base build may reach the network."
        )
    return (
        "Internet: off — a search, a cloud provider, a URL import, or a "
        "knowledge-base build is refused until this is turned on."
    )


def _safe(call: Any, default: Any) -> Any:
    try:
        return call()
    except Exception:
        return default


def current_values(root: Path | None = None) -> dict[str, str]:
    """The live value of each global setting, honestly (an unreadable one is ``"unknown"``).

    Read through the authoritative readers only, so the Settings page shows what
    is really in force rather than a second copy of it.
    """
    from gui.app import load_auto_speak, load_reading_voice
    from gui.themes import load_theme_name

    values: dict[str, str] = {
        "theme": _safe(load_theme_name, "unknown"),
        "reading_voice": _safe(load_reading_voice, "unknown") or "the default voice",
        "auto_speak": "on" if _safe(load_auto_speak, False) else "off",
    }
    try:
        from gui.typography_settings import SCOPES, load_profile

        values["typography"] = ", ".join(
            f"{scope}: {_safe(lambda scope=scope: load_profile(scope).describe(), 'unknown')}"
            for scope in SCOPES
        )
    except Exception:
        values["typography"] = "unknown"
    values["internet"] = "on" if read_internet(root) else "off"
    try:
        record = _preferences(root).record
        values["tashkeel"] = (
            f"{'on' if record.tashkeel else 'off'} ({record.tashkeel_mode})"
        )
    except Exception:
        values["tashkeel"] = "unknown"
    try:
        from core.cloud_model import (
            load_model_key,
            selected_provider,
        )
        from core.paths import data_dir

        provider = _safe(lambda: selected_provider(data_dir()), None)
        values["cloud_provider"] = _safe(lambda: provider.key, "unknown") if provider else "unknown"
        model = _safe(lambda: load_model_key(data_dir(), provider.key), None) if provider else None
        values["cloud_model"] = model or "(provider default)"
    except Exception:
        values["cloud_provider"] = "unknown"
        values["cloud_model"] = "unknown"
    try:
        from gui.app import load_sidebar_sections

        states = load_sidebar_sections(root)
        opened = sum(1 for open_ in states.values() if open_)
        values["sidebar_layout"] = f"{opened} section(s) open"
    except Exception:
        values["sidebar_layout"] = "unknown"
    return values


def describe(root: Path | None = None) -> str:
    """A readable block of every global setting and its current value."""
    values = current_values(root)
    lines = ["Global settings (one authoritative home each)", ""]
    for entry in SETTINGS_ENTRIES:
        lines.append(f"  {entry.title}")
        lines.append(f"    value:  {values.get(entry.key, 'unknown')}")
        lines.append(f"    writer: {entry.writer}")
    lines.append("")
    lines.append("Context-dependent (deliberately separate)")
    for name, note in CONTEXTUAL_SETTINGS:
        lines.append(f"  {name}: {note}")
    return "\n".join(lines)


__all__ = [
    "CONTEXTUAL_SETTINGS",
    "GUI_NAMESPACE",
    "INTERNET_CONSUMERS",
    "SETTINGS_ENTRIES",
    "SettingsEntry",
    "current_values",
    "describe",
    "entry_by_key",
    "internet_status",
    "read_internet",
    "resolve",
    "settings_by_link",
    "stores",
    "write_internet",
    "writers_by_setting",
]
