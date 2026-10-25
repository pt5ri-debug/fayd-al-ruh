"""The speech-engine studio window: its identity, its voice, its mood, its document.

The feature this pins: pressing one of the composer row's speech-engine buttons
opens a window of that engine's own, holding down the left side, in the order
asked for, (1) the engine's logo and name, changeable; (2) the reading voice,
drawn only from that engine's own voices, exclusive and remembered; (3) the mood,
applied at once and remembered; and below them the document that is read aloud --
plus a microphone in the composer.

The window parts are skipped unless a display (or Xvfb) is available, exactly like
``tests/test_gui.py``. The service they drive is exercised headless in
``tests/test_voice_studio.py``; here the *view* is what is checked.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest
from core import voice_studio_profile as profiles

# The fake reader/voice used to open a studio with no real speech engine.
from tests.test_voice_studio import FakeReader, _display_available


def _png(path: Path, size: int = 4) -> Path:
    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"".join(b"\x00" + b"\xff\x00\x00" * size for _ in range(size))
    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n"
    data += chunk(b"IHDR", header)
    data += chunk(b"IDAT", zlib.compress(raw))
    data += chunk(b"IEND", b"")
    path.write_bytes(data)
    return path


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("FAYD_DATA_DIR", str(root))
    return root


def _studio(source: str = "piper"):
    from gui.voice_studio import VoiceStudioService

    return VoiceStudioService(source, reader=FakeReader())


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_shows_the_identity_the_voice_and_the_mood(data_root) -> None:
    """The left side carries the logo, the voice menu, and the mood menu, in order."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.voice_studio_dialog import VoiceStudioDialog

    root = tk.Tk()
    try:
        service = _studio()
        dialog = VoiceStudioDialog(
            root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        root.update()
        # 1. The logo and the name.
        assert dialog._logo_canvas.winfo_ismapped()
        assert dialog._name_var.get() == service.name
        # 2. The reading voice, exclusive to this engine.
        assert dialog._voice_var.get() == service.voice_key()
        # 3. The mood menu.
        assert dialog._mood_menu.winfo_ismapped()
        # And a microphone in the composer.
        assert dialog._composer.mic_button is not None
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_keeps_both_scrollbars_and_a_bounded_geometry(data_root) -> None:
    """Both columns always show a scrollbar; the opening size fits the work area."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.voice_studio_dialog import (
        DEFAULT_HEIGHT,
        DEFAULT_WIDTH,
        VoiceStudioDialog,
    )

    root = tk.Tk()
    try:
        dialog = VoiceStudioDialog(
            root, _studio(), theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        root.update()
        assert dialog._sidebar_scrollbar.winfo_ismapped()
        assert dialog._scrollbar.winfo_ismapped()
        assert DEFAULT_WIDTH == 1180
        assert 0 < DEFAULT_HEIGHT <= root.winfo_screenheight()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_rename_and_logo_change_persist(data_root, tmp_path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.voice_studio_dialog import VoiceStudioDialog

    root = tk.Tk()
    try:
        service = _studio()
        dialog = VoiceStudioDialog(
            root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        root.update()
        dialog._name_var.set("My Piper")
        dialog._on_rename()
        root.update()
        assert service.name == "My Piper"
        assert profiles.load_profile("piper").name == "My Piper"
        # A logo can be set and cleared through the service the window drives.
        service.set_logo(_png(tmp_path / "logo.png"))
        dialog._refresh_identity()
        root.update()
        assert dialog._logo_photo is not None
        service.clear_logo()
        dialog._refresh_identity()
        root.update()
        assert dialog._logo_photo is None
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_mood_choice_applies_and_is_remembered(data_root) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.voice_studio_dialog import VoiceStudioDialog

    root = tk.Tk()
    try:
        service = _studio()
        dialog = VoiceStudioDialog(
            root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        root.update()
        dialog._on_mood_change("happy")
        root.update()
        assert service.mood_key() == "happy"
        assert profiles.load_profile("piper").mood == "happy"
        # The menu's button shows a readable label, not the raw key.
        assert "Happy" in dialog._mood_menu.cget("text")
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_voice_choice_is_remembered_and_exclusive(data_root) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.voice_studio_dialog import VoiceStudioDialog

    root = tk.Tk()
    try:
        service = _studio("system")
        dialog = VoiceStudioDialog(
            root, service, theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        root.update()
        dialog._on_voice_change("man-2")
        root.update()
        assert profiles.load_profile("system").voice == "man-2"
        # Another engine is untouched.
        assert profiles.load_profile("piper").voice == ""
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_closes_cleanly_and_calls_back(data_root) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.voice_studio_dialog import VoiceStudioDialog

    root = tk.Tk()
    closed: list[bool] = []
    try:
        dialog = VoiceStudioDialog(
            root, _studio(), theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono",
            on_close=lambda: closed.append(True),
        )
        root.update()
        dialog._close()
        root.update()
        assert closed == [True]
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_adds_the_transcript_messages(data_root) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.voice_studio_dialog import VoiceStudioDialog

    root = tk.Tk()
    try:
        dialog = VoiceStudioDialog(
            root, _studio(), theme=DARK_THEME, font="DejaVu Sans", mono="DejaVu Sans Mono"
        )
        root.update()
        before = len(dialog._rows)
        dialog._add_message("note", "A test note about the studio.")
        root.update()
        assert len(dialog._rows) == before + 1
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_composer_row_opens_the_studio(data_root, tmp_path) -> None:
    """Clicking a speech-engine chip opens that engine's studio window."""
    import tkinter as tk

    from gui.app import VOICE_ENGINE_BUTTONS, FaydGUI

    from tests.test_gui_design import _service as build_service

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=build_service(tmp_path))
        root.geometry("1180x780")
        for _ in range(6):
            root.update()
        key = VOICE_ENGINE_BUTTONS[0][0]
        app._open_voice_studio(key)
        root.update()
        dialogs = app._voice_studio_dialogs
        assert key in dialogs
        assert dialogs[key].window.winfo_exists()
        # Opening it again raises the same window rather than a second one.
        first = dialogs[key]
        app._open_voice_studio(key)
        assert app._voice_studio_dialogs[key] is first
    finally:
        root.destroy()
