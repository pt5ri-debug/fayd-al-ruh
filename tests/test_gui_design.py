"""The redesigned interface: operating modes, Markdown rendering, and the window.

The headless parts (modes, persistence, Markdown, the service's mode wiring) run
without a display; the window parts are skipped unless a display (or Xvfb) is
available, exactly like ``tests/test_gui.py``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, cast

import pytest

from core.command_router import NaturalLanguageRouter, RouteKind
from core.jarvis import JarvisMode, JarvisTurn
from core.tools import build_sandbox
from gui import markdown as md
from gui import modes
from gui.modes import HistoryItem
from gui.service import ChatService


class FakeSession:
    """A minimal stand-in for ``JarvisMode`` with a real sandboxed router."""

    def __init__(self, router: NaturalLanguageRouter | None = None) -> None:
        self.router = router
        self.assistant: Any = SimpleNamespace(name="grounded")
        self.voice = SimpleNamespace(assistant=self.assistant)
        self.help_text = "help"
        self.memory: Any = None
        self._conversation = SimpleNamespace(id="CNV-fake")
        self.saved_ids: tuple[str, ...] = ()

    @property
    def conversation(self) -> SimpleNamespace:
        return self._conversation

    def handle(self, message: str, *, require_wake: bool = False, speak: bool = True) -> JarvisTurn:
        return JarvisTurn(heard=message, response="ok", woke=True, routed="answer")

    def reset_conversation(self) -> SimpleNamespace:
        self._conversation = SimpleNamespace(id="CNV-new")
        return self._conversation

    def save_session(self) -> str:
        return f"Saved this conversation as {self.conversation.id}."

    def resume_session(self, conversation_id: str) -> bool:
        self._conversation = SimpleNamespace(id=conversation_id)
        return True

    def list_sessions(self) -> tuple[str, ...]:
        return self.saved_ids


def _service(tmp_path, *, operating_mode: str | None = None, read_write: bool = False) -> ChatService:
    sandbox = build_sandbox(tmp_path, read_only=not read_write)
    session = FakeSession(NaturalLanguageRouter(sandbox))
    return ChatService(
        mode=cast(JarvisMode, session),
        operating_mode=operating_mode,
        data_root=tmp_path,
    )


# --- modes ----------------------------------------------------------------


def test_mode_labels_are_exact() -> None:
    assert [mode.label for mode in modes.MODES] == ["AI Mode", "Jarvics Mode"]
    assert modes.AI_MODE == "AI Mode"
    assert modes.JARVICS_MODE == "Jarvics Mode"


def test_resolve_mode_by_key_label_and_default() -> None:
    assert modes.resolve_mode("ai").key == "ai"
    assert modes.resolve_mode("jarvics").key == "jarvics"
    assert modes.resolve_mode("AI Mode").key == "ai"
    assert modes.resolve_mode("jarvics mode").key == "jarvics"
    assert modes.resolve_mode(None).key == "ai"
    assert modes.resolve_mode("nonsense").key == "ai"  # never raises


def test_mode_persists_and_reloads(tmp_path) -> None:
    assert modes.load_mode_key(tmp_path) == "ai"
    modes.save_mode_key(tmp_path, "jarvics")
    assert modes.load_mode_key(tmp_path) == "jarvics"
    # An unknown value falls back to the default rather than breaking the window.
    modes.save_mode_key(tmp_path, "not-a-mode")
    assert modes.load_mode_key(tmp_path) == "jarvics"


def test_answer_only_router_never_plans_a_tool(tmp_path) -> None:
    real = NaturalLanguageRouter(build_sandbox(tmp_path, read_only=False))
    answer = modes.AnswerOnlyRouter(real.sandbox)
    # A sentence the real router would run as a tool is answered, not executed.
    assert real.route("list the files in notes/").kind is RouteKind.TOOL
    assert answer.route("list the files in notes/").kind is RouteKind.ANSWER


def test_history_groups_by_date() -> None:
    now = datetime(2026, 10, 11, tzinfo=timezone.utc)
    items = [
        HistoryItem("a", "today", now),
        HistoryItem("b", "yesterday", now - timedelta(days=1)),
        HistoryItem("c", "week", now - timedelta(days=3)),
        HistoryItem("d", "month", now - timedelta(days=20)),
        HistoryItem("e", "old", now - timedelta(days=99)),
        HistoryItem("f", "undated", None),
    ]
    grouped = dict(modes.group_by_date(items, now=now))
    assert [i.session_id for i in grouped["Today"]] == ["a"]
    assert [i.session_id for i in grouped["Yesterday"]] == ["b"]
    assert [i.session_id for i in grouped["Previous 7 days"]] == ["c"]
    assert [i.session_id for i in grouped["Previous 30 days"]] == ["d"]
    assert [i.session_id for i in grouped["Older"]] == ["e", "f"]


def test_search_items_matches_label_and_snippet() -> None:
    items = (
        HistoryItem("a", "Porosity basics", None, snippet="pore space"),
        HistoryItem("b", "Drilling mud", None, snippet="mud weight"),
    )
    assert [i.session_id for i in modes.search_items(items, "porosity")] == ["a"]
    assert [i.session_id for i in modes.search_items(items, "mud weight")] == ["b"]
    assert modes.search_items(items, "") == items


# --- markdown -------------------------------------------------------------


def test_render_markdown_recognises_blocks() -> None:
    text = (
        "# Title\n\nA paragraph.\n\n- one\n- two\n\n1. first\n\n> quote\n\n"
        "```python\nx = 1\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n---\n"
    )
    kinds = [block.kind for block in md.render_markdown(text)]
    assert md.HEADING in kinds
    assert kinds.count(md.BULLET) == 2
    assert md.NUMBER in kinds
    assert md.QUOTE in kinds
    assert md.CODE in kinds
    assert md.TABLE in kinds
    assert md.RULE in kinds


def test_inline_marks_bold_and_code_but_not_multiplication() -> None:
    segments = md.parse_inline("**bold** and `code` and 2 * 3")
    kinds = [(s.kind, s.text) for s in segments]
    assert ("bold", "bold") in kinds
    assert ("code", "code") in kinds
    # The multiplication stays plain text -- no accidental italic.
    assert any(s.kind == "text" and "2 * 3" in s.text for s in segments)


def test_highlight_code_classifies_tokens() -> None:
    spans = md.highlight_code("def f(x):\n    return 'a'  # note 42")
    kinds = {s.kind for s in spans}
    assert "keyword" in kinds
    assert "string" in kinds
    assert "comment" in kinds


def test_iter_styled_lines_round_trips_the_text() -> None:
    text = "# H\n\n- a\n- b\n\n```py\nx = 1\n```\n\nplain 2 * 3"
    styled = md.iter_styled_lines(text)
    assert "\n".join(line.text for line in styled) == text
    assert any(line.kind == md.HEADING for line in styled)
    assert any(line.in_code for line in styled)


# --- service mode wiring --------------------------------------------------


def test_service_defaults_to_ai_mode(tmp_path) -> None:
    service = _service(tmp_path)
    assert service.operating_mode.label == "AI Mode"
    assert service.tools_enabled is False
    # AI Mode answers with the answer-only router, so no tool can be planned.
    assert service.mode.router is service._answer_router
    assert service.mode.router.route("list the files in notes/").kind is RouteKind.ANSWER


def test_service_switching_modes_swaps_the_router_and_persists(tmp_path) -> None:
    service = _service(tmp_path)
    spec = service.set_operating_mode("jarvics")
    assert spec.label == "Jarvics Mode"
    assert service.tools_enabled is True
    assert service.mode.router is service._tool_router
    assert modes.load_mode_key(tmp_path) == "jarvics"
    # A tool sentence is now recognised as a tool plan.
    preview = service.preview_execution("list the files in notes/")
    assert preview.kind == "tool"


def test_service_reads_a_saved_mode(tmp_path) -> None:
    modes.save_mode_key(tmp_path, "jarvics")
    service = _service(tmp_path)
    assert service.operating_mode_key == "jarvics"


def test_preview_marks_a_write_as_needing_confirmation(tmp_path) -> None:
    service = _service(tmp_path, read_write=True)
    preview = service.preview_execution("write to file notes/a.txt: permeability is high")
    assert preview.kind == "tool"
    assert preview.needs_confirmation is True
    assert any(step["tool"] == "write_file" for step in preview.steps)
    # A read-only request does not need confirmation.
    read = service.preview_execution("list the files")
    assert read.needs_confirmation is False


def test_service_lists_tool_names(tmp_path) -> None:
    service = _service(tmp_path)
    names = service.tool_names
    assert "calculator" in names
    assert "list_dir" in names


def test_saved_session_summaries_are_newest_first(tmp_path) -> None:
    from core.conversation import ConversationMemory
    from storage.json_store import JsonStore

    memory = ConversationMemory(JsonStore(tmp_path / "jarvis"))
    older = memory.start("older")
    older.add_user("what is porosity?")
    memory.save(older)
    newer = memory.start("newer")
    newer.add_user("what is permeability?")
    newer.created_at = datetime.now(timezone.utc)
    memory.save(newer)

    session = FakeSession()
    session.memory = memory
    session.saved_ids = (older.id, newer.id)
    service = ChatService(mode=cast(JarvisMode, session), data_root=tmp_path)

    items = service.saved_session_summaries()
    assert {item.session_id for item in items} == {older.id, newer.id}
    # The label is the first user turn, so the sidebar shows a readable title.
    assert any("permeability" in item.label for item in items)


# --- window (only when a display is available) ----------------------------


def _display_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


def _pillow_available() -> bool:
    try:
        import PIL  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_mode_switch_preserves_messages(tmp_path) -> None:
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path))
        root.update()
        app._add_bubble("assistant", "keep me across the mode switch")
        assert app.service.operating_mode.label == "AI Mode"
        app._set_mode(modes.resolve_mode("jarvics"))
        root.update()
        assert app.service.operating_mode.label == "Jarvics Mode"
        texts = [entry.text for entry in app._rows]
        assert any("keep me across the mode switch" in text for text in texts)
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_renders_markdown_tags(tmp_path) -> None:
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path))
        root.update()
        text = "# Heading\n\n- item\n\n**bold** and `code`"
        bubble = app._add_bubble("assistant", text)
        root.update()
        assert bubble.get("1.0", "end-1c") == text  # exact round-trip
        assert bubble.tag_ranges("heading")
        assert bubble.tag_ranges("bullet")
        assert bubble.tag_ranges("bold")
        assert bubble.tag_ranges("inline_code")
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_asks_confirmation_before_a_write(tmp_path) -> None:
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path, read_write=True))
        root.update()
        app._set_mode(modes.resolve_mode("jarvics"))
        root.update()
        app._entry.insert("1.0", "write to file notes/a.txt: permeability is high")
        app._on_send()
        root.update()
        # The plan is shown and the tool has not run: it waits for approval.
        assert app._pending_message is not None
        assert app._confirm_bar is not None and app._confirm_bar.winfo_ismapped()
        app._resolve_confirmation(False)  # cancel
        root.update()
        assert app._pending_message is None
        assert not (app._confirm_bar is not None and app._confirm_bar.winfo_ismapped())
        assert not (tmp_path / "notes" / "a.txt").exists()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_sidebar_collapses_and_reopens(tmp_path) -> None:
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path))
        root.geometry("1200x800")
        for _ in range(5):
            root.update()
        assert app._sidebar_visible is True
        app._toggle_sidebar()
        root.update()
        assert app._sidebar_visible is False
        assert not app._sidebar.winfo_ismapped()
        app._toggle_sidebar()
        root.update()
        assert app._sidebar_visible is True
        assert app._sidebar.winfo_ismapped()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_tools_section_reflects_the_mode(tmp_path) -> None:
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path))
        root.update()
        labels = [w.cget("text") for w in app._tools_panel.winfo_children()]
        assert any("calculator" in text for text in labels)
        # AI Mode dims them and says so; Jarvics Mode enables them.
        assert any("Jarvics Mode" in text for text in labels)
        app._set_mode(modes.resolve_mode("jarvics"))
        root.update()
        labels = [w.cget("text") for w in app._tools_panel.winfo_children()]
        assert any("Available in Jarvics Mode" in text for text in labels)
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_image_dialog_puts_the_preview_on_the_right(tmp_path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.image_gen import ImageGenerationService
    from gui.image_gen_dialog import ImageGenDialog

    root = tk.Tk()
    try:
        service = ImageGenerationService(root=tmp_path)
        dialog = ImageGenDialog(
            root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        root.update()
        # The preview lives in its own column, to the right of the controls.
        assert int(dialog._controls_column.grid_info()["column"]) == 0
        assert int(dialog._side_column.grid_info()["column"]) == 1
        assert dialog._side_column.master is dialog._controls_column.master
    finally:
        root.destroy()



@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_image_dialog_preview_is_a_third_of_the_window(tmp_path) -> None:
    """The preview column is 30% of the window width, before and after a render."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.image_gen import ImageGenerationService
    from gui.image_gen_dialog import PREVIEW_FRACTION, ImageGenDialog

    root = tk.Tk()
    try:
        root.geometry("1600x1000")
        service = ImageGenerationService(root=tmp_path)
        dialog = ImageGenDialog(
            root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        for _ in range(20):
            root.update()
        window_width = dialog.window.winfo_width()
        assert PREVIEW_FRACTION == 0.30
        # The visible preview box is 30% of the window width, before any render.
        preview_before = dialog._preview.winfo_width()
        assert preview_before > 1
        assert abs(preview_before - int(window_width * PREVIEW_FRACTION)) <= 2

        report = service.generate("a red spiral 512x512")
        assert report.ok
        dialog._show_preview(report.path)
        for _ in range(20):
            root.update()
        # The generated image fits the preview's own width -- unchanged by the
        # render, so the picture is as wide as the box it replaced. With Pillow
        # (the [image]/[ocr] extra) the fit is exact; Tk's integer ``subsample``
        # fallback can only round down, so it is never wider.
        assert dialog._photo is not None
        assert dialog._photo.width() <= preview_before
        assert dialog._preview.winfo_width() == preview_before
        if _pillow_available():
            assert dialog._photo.width() == preview_before

        # Resizing the window re-fits the picture to the new 30% preview box.
        dialog.window.geometry("800x600")
        for _ in range(40):
            root.update()
        resized = dialog.window.winfo_width()
        assert abs(dialog._preview.winfo_width() - int(resized * PREVIEW_FRACTION)) <= 2
        assert dialog._photo.width() <= dialog._preview.winfo_width()
        assert dialog._photo.width() < preview_before
    finally:
        root.destroy()


def _sample_image(tmp_path):
    """A real PNG on disk, written by the project's own generator (no Pillow needed)."""
    from core.imagegen import ImageRequest, ProceduralImageGenerator

    image = ProceduralImageGenerator().generate(
        ImageRequest(prompt="a blue rig", width=512, height=384)
    )
    path = tmp_path / "source.png"
    path.write_bytes(image.png)
    return path


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_edit_dialog_puts_the_preview_on_the_right(tmp_path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.image_edit import ImageEditService
    from gui.image_edit_dialog import ImageEditDialog

    root = tk.Tk()
    try:
        service = ImageEditService(root=tmp_path)
        dialog = ImageEditDialog(
            root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        root.update()
        assert int(dialog._controls_column.grid_info()["column"]) == 0
        assert int(dialog._side_column.grid_info()["column"]) == 1
        assert dialog._side_column.master is dialog._controls_column.master
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_edit_dialog_preview_is_a_third_of_the_window(tmp_path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.image_edit import ImageEditService
    from gui.image_edit_dialog import PREVIEW_FRACTION, ImageEditDialog

    root = tk.Tk()
    try:
        root.geometry("1600x1000")
        service = ImageEditService(root=tmp_path)
        dialog = ImageEditDialog(
            root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        for _ in range(20):
            root.update()
        window_width = dialog.window.winfo_width()
        assert PREVIEW_FRACTION == 0.30
        preview_before = dialog._preview.winfo_width()
        assert preview_before > 1
        assert abs(preview_before - int(window_width * PREVIEW_FRACTION)) <= 2

        # Loading a picture scales it to the preview's own width, unchanged by the
        # load -- the picture is as wide as the box it sits in.
        dialog._load_source(str(_sample_image(tmp_path)))
        for _ in range(20):
            root.update()
        assert dialog._photo is not None
        assert dialog._photo.width() <= preview_before
        assert dialog._preview.winfo_width() == preview_before
        if _pillow_available():
            assert dialog._photo.width() == preview_before

        # Resizing the window re-fits the picture to the new 30% preview box.
        dialog.window.geometry("800x600")
        for _ in range(40):
            root.update()
        resized = dialog.window.winfo_width()
        assert abs(dialog._preview.winfo_width() - int(resized * PREVIEW_FRACTION)) <= 2
        assert dialog._photo.width() <= dialog._preview.winfo_width()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_edit_dialog_applies_an_instruction_and_undoes_it(tmp_path) -> None:
    import time
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.image_edit import ImageEditService
    from gui.image_edit_dialog import ImageEditDialog

    root = tk.Tk()
    try:
        service = ImageEditService(root=tmp_path)
        dialog = ImageEditDialog(
            root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        dialog._load_source(str(_sample_image(tmp_path)))
        root.update()
        assert dialog._apply_button.cget("state") == "normal"
        assert dialog._undo_button.cget("state") == "disabled"

        dialog._instruction.delete("1.0", "end")
        dialog._instruction.insert("1.0", "rotate 90 and grayscale")
        dialog._on_apply()
        for _ in range(200):
            root.update()
            time.sleep(0.01)
        assert "Saved to" in dialog._status.cget("text")
        assert "rotate 90°" in dialog._status.cget("text")
        assert dialog._undo_button.cget("state") == "normal"
        assert dialog._photo is not None
        assert (dialog._session.raster.width, dialog._session.raster.height) == (384, 512)

        dialog._on_undo()
        root.update()
        assert "Undid" in dialog._status.cget("text")
        assert dialog._undo_button.cget("state") == "disabled"
        assert (dialog._session.raster.width, dialog._session.raster.height) == (512, 384)
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_edit_dialog_reports_an_unrecognised_instruction(tmp_path) -> None:
    import time
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.image_edit import ImageEditService
    from gui.image_edit_dialog import ImageEditDialog

    root = tk.Tk()
    try:
        service = ImageEditService(root=tmp_path)
        dialog = ImageEditDialog(
            root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        dialog._load_source(str(_sample_image(tmp_path)))
        root.update()
        dialog._instruction.delete("1.0", "end")
        dialog._instruction.insert("1.0", "paint the sky purple")
        dialog._on_apply()
        for _ in range(200):
            root.update()
            time.sleep(0.01)
        # The failure is shown in the danger colour, and the image is unchanged.
        assert "could not find an edit" in dialog._status.cget("text")
        assert dialog._status.cget("fg") == DARK_THEME.danger
        assert dialog._undo_button.cget("state") == "disabled"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_attach_menu_offers_image_editing(tmp_path) -> None:
    """The ＋ menu has a labelled entry that opens the edit window."""
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path))
        root.update()
        # The ＋ menu names the entry, next to the generate one.
        menu = app._build_attach_menu()
        end = menu.index("end")
        assert end is not None
        labels = [
            menu.entrycget(index, "label")
            for index in range(end + 1)
            if menu.type(index) == "command"
        ]
        assert "Generate an image from text…" in labels
        assert "Edit an image from text…" in labels
        # The opener builds the window without raising.
        app._open_image_edit()
        root.update()
        assert app._image_edit_dialog is not None
        assert app._image_edit_dialog.window.title() == "Edit an image from text"
        app._image_edit_dialog._close()
        root.update()
        assert app._image_edit_dialog is None
    finally:
        root.destroy()
