"""Window-level tests for Doc's Studio (skipped with no display).

These drive the *real* Tk window against the *real* ``DocsStudioService`` and an
injected vision runner, so the page is exercised the way a user drives it: open a
document from the ＋ menu, read its text, page through it, and ask a question --
all with no model, no weights, and no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.docs_model import HEADER
from gui.docs_studio import DocsStudioService


def _display_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


class RecordingRunner:
    """A deterministic stand-in for the vision model (no extra, no weights)."""

    def __init__(self, answer: str = "Porosity is the void fraction of a rock.") -> None:
        self.answer = answer

    def chat(self, messages: list[dict], **kwargs) -> str:
        return self.answer


def _wait_until(root, predicate, *, timeout: float = 15.0) -> bool:
    """Run the real Tk main loop until ``predicate`` holds (or time runs out)."""
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        root.update()
        time.sleep(0.01)
    return predicate()


def _all_text_widgets(parent) -> list:
    import tkinter as tk

    found = []
    for child in parent.winfo_children():
        if isinstance(child, tk.Text):
            found.append(child)
        found.extend(_all_text_widgets(child))
    return found


def _transcript_text(dialog) -> str:
    return "\n".join(widget.get("1.0", "end") for widget in _all_text_widgets(dialog._transcript))


def _dialog(tmp_path: Path, runner=None):
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.docs_dialog import DocsDialog

    service = DocsStudioService(
        runner=runner or RecordingRunner(), model_root=tmp_path / "models"
    )
    root = tk.Tk()
    root.withdraw()
    dialog = DocsDialog(
        root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
    )
    root.update()
    return root, dialog, service


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_builds_with_the_chat_window_style(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        assert dialog.window.title()
        assert dialog._transcript is not None
        assert dialog._question is not None
        # The same title-bar trio every window installs.
        assert dialog._window_controls is not None
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_opens_a_document_and_shows_its_text(tmp_path: Path) -> None:
    document = tmp_path / "report.md"
    document.write_text("# Report\n\nPorosity is the void fraction of a rock.\n", encoding="utf-8")
    root, dialog, _service = _dialog(tmp_path)
    try:
        dialog._load(str(document))
        assert _wait_until(root, lambda: "Porosity is the void fraction" in _transcript_text(dialog))
        assert dialog._current_key
        assert dialog._document_list.size() == 1
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_answers_a_question(tmp_path: Path) -> None:
    document = tmp_path / "report.md"
    document.write_text("Porosity is the void fraction of a rock.\n", encoding="utf-8")
    root, dialog, _service = _dialog(tmp_path)
    try:
        dialog._load(str(document))
        assert _wait_until(root, lambda: bool(dialog._current_key))
        dialog._question_var.set("What is porosity?")
        dialog._on_ask()
        # Wait for the *answer* row (the honesty header), not just the echoed
        # document text, so the assertion is about the model's reply.
        assert _wait_until(root, lambda: HEADER in _transcript_text(dialog))
        assert "Porosity is the void fraction" in _transcript_text(dialog)
        assert "not grounded in cited sources" in _transcript_text(dialog)
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_says_so_when_no_model_is_installed(tmp_path: Path) -> None:
    document = tmp_path / "report.md"
    document.write_text("hello\n", encoding="utf-8")
    root, dialog, _service = _dialog(tmp_path, runner=None)
    try:
        # Replace the injected runner with nothing, so the real "missing model"
        # path is exercised: the window must name the next step, not crash.
        dialog.service._runner = None  # type: ignore[attr-defined]
        dialog._load(str(document))
        assert _wait_until(root, lambda: bool(dialog._current_key))
        dialog._question_var.set("what is this?")
        dialog._on_ask()
        assert _wait_until(root, lambda: "Cannot answer" in _transcript_text(dialog)
                           or "docs" in _transcript_text(dialog).lower())
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_refuses_a_question_with_no_document(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        dialog._question_var.set("what is porosity?")
        dialog._on_ask()
        assert _wait_until(root, lambda: "Open a document first" in _transcript_text(dialog))
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_closes_a_document(tmp_path: Path) -> None:
    document = tmp_path / "report.md"
    document.write_text("Porosity controls storage.\n", encoding="utf-8")
    root, dialog, _service = _dialog(tmp_path)
    try:
        dialog._load(str(document))
        assert _wait_until(root, lambda: bool(dialog._current_key))
        dialog._on_close_document()
        root.update()
        assert dialog._current_key == ""
        assert dialog._document_list.size() == 0
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_shows_the_model_status_honestly(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    try:
        note = dialog._model_note.cget("text")
        assert note
        assert "ready" in note.lower() or "docs" in note.lower()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_previews_a_vision_page(tmp_path: Path) -> None:
    # A plain image is a one-page document whose content is the picture itself, so
    # the preview must actually show it rather than report a text layer.
    image = tmp_path / "shot.png"
    image.write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    )
    root, dialog, _service = _dialog(tmp_path)
    try:
        dialog._load(str(image))
        assert _wait_until(root, lambda: bool(dialog._current_key))
        assert dialog._page_numbers == [1]
        assert dialog._page_label.cget("text").startswith("1 / 1")
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_reads_a_scanned_document_and_shows_its_contents(tmp_path: Path) -> None:
    # A plain image is a one-page document with no text layer, so the window must
    # *read* it (transcribe it) and show the contents -- not only the page image.
    image = tmp_path / "scan.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    root, dialog, _service = _dialog(tmp_path)
    try:
        dialog._load(str(image))
        assert _wait_until(root, lambda: HEADER in _transcript_text(dialog))
        text = _transcript_text(dialog)
        assert "Porosity is the void fraction" in text
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_names_the_next_step_when_a_scanned_file_needs_the_model(
    tmp_path: Path,
) -> None:
    image = tmp_path / "scan.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    root, dialog, service = _dialog(tmp_path)
    try:
        # Clear the client's injected runner so the real "model not installed" path
        # is exercised: the window must name the next step, not show a blank file.
        service._client._runner = None  # type: ignore[attr-defined]
        dialog._load(str(image))
        assert _wait_until(root, lambda: "not ready yet" in _transcript_text(dialog))
        assert "docs extra" in _transcript_text(dialog)
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_the_window_closes_cleanly(tmp_path: Path) -> None:
    root, dialog, _service = _dialog(tmp_path)
    dialog._close()
    assert dialog._closed is True
    root.destroy()
