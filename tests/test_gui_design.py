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
def test_window_sidebar_hides_the_mouth_actions(tmp_path) -> None:
    """The four "Answer from ..." rows are hidden; the header selector still switches."""
    import tkinter as tk

    from gui.actions import visible_actions
    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path))
        app._toggle_settings()
        root.update()

        labels = [button.cget("text") for button in app._action_buttons]
        for mouth in ("Ultron", "Vision", "Banner", "Cloud"):
            assert not any(f"Answer from {mouth}" in text for text in labels)
        assert len(app._action_buttons) == len(visible_actions())
        # The header selector remains the live mouth switch.
        assert app._source_var.get() == app.service.source
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_settings_hides_the_answering_mouth_radios(tmp_path) -> None:
    """The sidebar settings no longer list the answering mouths; the header does."""
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path))
        app._toggle_settings()
        root.update()

        def walk(widget, out):
            for child in widget.winfo_children():
                out.append(child)
                walk(child, out)

        widgets: list = []
        walk(app._settings_panel, widgets)
        # No radio for any mouth remains in the panel, and the header's variable
        # still tracks the live mouth (the header selector is the one switch).
        mouth_keys = {"sources", "model", "ultron", "vision", "banner", "cloud"}
        radios = [w for w in widgets if isinstance(w, tk.Radiobutton)]
        assert not any(w.cget("value") in mouth_keys for w in radios)
        assert app._source_var.get() == app.service.source
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_has_no_tools_section_in_the_sidebar(tmp_path) -> None:
    """The sidebar's **Tools** list is gone; the tools themselves are unchanged."""
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path))
        root.update()
        # The panel is not built any more, and the service still knows the tools.
        assert not hasattr(app, "_tools_panel")
        assert app.service.tool_names
        # Switching mode no longer needs a panel to paint into (it used to call
        # _render_tools); it must not raise.
        app._set_mode(modes.resolve_mode("jarvics"))
        root.update()
        assert app.service.tools_enabled is True
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
def test_edit_dialog_offers_the_learned_backend(tmp_path) -> None:
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
        # Deterministic is the default, so the download button is hidden.
        assert dialog._backend_var.get() == "deterministic"
        assert not dialog._download_button.winfo_ismapped()
        # Switching to the learned backend reveals the download button and names
        # the model honestly.
        dialog._backend_var.set("learned")
        dialog._on_backend_change()
        root.update()
        assert dialog._download_button.winfo_ismapped()
        note = dialog._backend_note.cget("text")
        assert "instruct-pix2pix" in note
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_edit_dialog_learned_backend_without_the_model_reports_honestly(tmp_path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.image_edit import ImageEditService
    from gui.image_edit_dialog import ImageEditDialog

    root = tk.Tk()
    try:
        # The service's root is the tmp dir, so the cached weights are not found:
        # applying a learned edit must say so instead of running a CPU edit.
        service = ImageEditService(root=tmp_path)
        dialog = ImageEditDialog(
            root, service, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont"
        )
        dialog._load_source(str(_sample_image(tmp_path)))
        dialog._backend_var.set("learned")
        dialog._on_backend_change()
        dialog._instruction.delete("1.0", "end")
        dialog._instruction.insert("1.0", "make the person smile")
        dialog._on_apply()
        root.update()
        assert "not ready" in dialog._status.cget("text")
        assert dialog._status.cget("fg") == DARK_THEME.danger
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


# --- the shared chat composer (only when a display is available) -----------


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_shared_composer_placeholder_is_not_the_user_text() -> None:
    """The hint is an overlay label, so the input holds only the user's words."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.chat_composer import ChatComposer
    from gui.typography import resolve

    root = tk.Tk()
    try:
        composer = ChatComposer(
            root, theme=DARK_THEME, typography=resolve(root),
            placeholder="What do you want to build?", on_submit=lambda: None,
        )
        composer.wrap.pack(fill="x")
        root.update()
        assert composer.text() == ""
        assert composer.placeholder_label.winfo_ismapped()
        composer.set_text("hello")
        root.update()
        assert composer.text() == "hello"
        assert not composer.placeholder_label.winfo_ismapped()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_shared_composer_disables_send_for_whitespace_only() -> None:
    """Send is refused for an empty or whitespace-only message."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.chat_composer import ChatComposer
    from gui.typography import resolve

    root = tk.Tk()
    try:
        composer = ChatComposer(
            root, theme=DARK_THEME, typography=resolve(root),
            placeholder="Ask", on_submit=lambda: None,
        )
        root.update()
        assert composer._submit_enabled and not composer.text().strip()
        # The disc is greyed while there is nothing to send.
        assert composer.send_button.itemcget(composer.send_button._shape, "fill") == DARK_THEME.disabled
        composer.set_text("   ")
        root.update()
        assert composer.send_button.itemcget(composer.send_button._shape, "fill") == DARK_THEME.disabled
        composer.set_text("a real question")
        root.update()
        assert composer.send_button.itemcget(composer.send_button._shape, "fill") == DARK_THEME.primary
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_shared_composer_enter_submits_and_shift_enter_does_not() -> None:
    """Enter sends; Shift+Enter keeps the newline the Text widget inserts."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.chat_composer import ChatComposer
    from gui.typography import resolve

    root = tk.Tk()
    try:
        calls: list[int] = []
        composer = ChatComposer(
            root, theme=DARK_THEME, typography=resolve(root),
            placeholder="Ask", on_submit=lambda: calls.append(1),
        )
        composer.wrap.pack(fill="x")
        root.update()
        composer.entry.focus_force()
        root.update()
        composer.entry.event_generate("<Return>")
        root.update()
        assert calls == [1]
        # Shift+Enter inserts a newline and never submits.
        composer.entry.event_generate("<Shift-Return>")
        root.update()
        assert calls == [1]
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_shared_composer_grows_then_caps_within_its_limits() -> None:
    """The input grows with the message and then stops at its maximum height."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.chat_composer import ChatComposer
    from gui.typography import resolve

    root = tk.Tk()
    try:
        composer = ChatComposer(
            root, theme=DARK_THEME, typography=resolve(root),
            placeholder="Ask", on_submit=lambda: None, min_lines=1, max_lines=3,
        )
        composer.wrap.pack(fill="x")
        root.update()
        assert int(composer.entry.cget("height")) == 1
        composer.set_text("one\ntwo\nthree\nfour\nfive")
        root.update()
        assert int(composer.entry.cget("height")) == 3
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_shared_composer_mirrors_a_callers_string_variable() -> None:
    """A screen that drives its panel through a StringVar keeps it in step."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.chat_composer import ChatComposer
    from gui.typography import resolve

    root = tk.Tk()
    try:
        var = tk.StringVar(value="")
        composer = ChatComposer(
            root, theme=DARK_THEME, typography=resolve(root),
            placeholder="Ask", on_submit=lambda: None, variable=var,
        )
        root.update()
        composer.set_text("typed here")
        root.update()
        assert var.get() == "typed here"
        var.set("set outside")
        root.update()
        assert composer.text() == "set outside"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_shared_composer_has_no_attach_button_when_not_supported() -> None:
    """A screen with no attachments gets no ＋ button that would lie."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.chat_composer import ChatComposer
    from gui.typography import resolve

    root = tk.Tk()
    try:
        without = ChatComposer(
            root, theme=DARK_THEME, typography=resolve(root),
            placeholder="Ask", on_submit=lambda: None, with_attach=False,
        )
        without.wrap.pack(fill="x")
        root.update()
        assert without.attach_button is None
        with_attach = ChatComposer(
            root, theme=DARK_THEME, typography=resolve(root),
            placeholder="Ask", on_submit=lambda: None,
        )
        with_attach.wrap.pack(fill="x")
        root.update()
        assert with_attach.attach_button is not None
        assert with_attach.attach_button.winfo_ismapped()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_main_window_composer_carries_both_mode_selectors(tmp_path) -> None:
    """The bottom-left toolbar has the Code Studio and Chat Studio pills."""
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path))
        root.update()
        code = app._code_mode_selector
        chat = app._chat_mode_selector
        assert code.winfo_ismapped() and chat.winfo_ismapped()
        assert "Code Studio" in code.cget("text")
        assert "AI Mode" in chat.cget("text")

        # The Chat Studio pill lists the real operating modes and switches them.
        from gui.app import MODES

        chat_menu = chat.nametowidget(chat.cget("menu"))
        labels = [
            chat_menu.entrycget(index, "label")
            for index in range(chat_menu.index("end") + 1)
        ]
        assert labels == [mode.label for mode in MODES]
        chat_menu.invoke(1)
        root.update()
        assert app.service.operating_mode.label == labels[1]
        assert labels[1] in chat.cget("text")

        # The Code Studio pill opens the one real code surface.
        code_menu = code.nametowidget(code.cget("menu"))
        assert code_menu.type(0) == "command"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_main_window_composer_has_attach_dot_mic_and_send(tmp_path) -> None:
    """The composer carries the ＋, the status dot, the microphone, and Send."""
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path))
        root.update()
        composer = app._composer
        assert composer.attach_button is not None and composer.attach_button.winfo_ismapped()
        assert composer.status_dot is not None and composer.status_dot.winfo_ismapped()
        assert composer.mic_button is not None and composer.mic_button.winfo_ismapped()
        assert composer.send_button.winfo_ismapped()
        # The toolbars sit below the input, and Send is rightmost.
        assert composer.send_button.winfo_rootx() > composer.mic_button.winfo_rootx()
        assert composer.send_button.winfo_rooty() > composer.entry.winfo_rooty()
    finally:
        root.destroy()


# --- every window's title bar ------------------------------------------------


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_every_window_title_bar_has_the_three_controls(tmp_path) -> None:
    """Minimize, maximize, and close live on the main window and every dialog.

    One shared trio, built by ``install_window_controls``, is what makes "every
    window" true rather than aspirational: the main window is checked here, and
    each dialog is checked where it is exercised, so a window that forgot the
    controls fails rather than quietly differing from the rest.
    """
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.window_controls import _CLOSE_GLYPH, _MAXIMIZE_GLYPH, _MINIMIZE_GLYPH

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path))
        root.update()
        controls = app._window_controls
        assert controls.frame.winfo_ismapped()
        glyphs = [
            child.cget("text") for child in controls.frame.winfo_children()
        ]
        assert glyphs == [_MINIMIZE_GLYPH, _MAXIMIZE_GLYPH, _CLOSE_GLYPH]
        # The close control is the window's own close path, not a second one.
        assert controls.close_button is not None
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_title_bars_use_the_dark_grey_band(tmp_path) -> None:
    """The title bar is the dark grey token in both appearances."""
    import tkinter as tk

    from gui.app import DARK_THEME, LIGHT_THEME, FaydGUI

    # A dark grey band, clearly darker than the raised surfaces it sits among,
    # and not the page base it used to be.
    assert DARK_THEME.titlebar != DARK_THEME.base
    assert DARK_THEME.titlebar != LIGHT_THEME.titlebar

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(tmp_path), theme="dark")
        root.update()
        bar = app._window_controls.frame.master
        assert bar.cget("bg") == DARK_THEME.titlebar
        app._toggle_theme()
        root.update()
        assert app._window_controls.frame.master.cget("bg") == LIGHT_THEME.titlebar
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_a_destroyed_window_releases_its_title_bar_handlers() -> None:
    """A closed window must not be kept alive by its title-bar controls.

    Each button's ``command`` holds a bound method on the controller, which holds
    the window, which owns the buttons -- a cycle a destroyed window only leaves
    when the cyclic collector happens to run, and on a worker thread the Tcl call
    that follows deadlocks (which is how a deck export used to hang). The
    controls drop the handler and clear both commands with the window, so the
    chain is broken the moment the window dies rather than whenever a collection
    happens to run.
    """
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.typography import for_root
    from gui.window_controls import install_window_controls

    root = tk.Tk()
    try:
        root.withdraw()
        window = tk.Toplevel(root)
        header = tk.Frame(window)
        header.pack(fill="x")
        closed: list[int] = []
        controls = install_window_controls(
            window, header, theme=DARK_THEME, typography=for_root(root),
            on_close=lambda: closed.append(1),
        )
        root.update()
        assert controls._close is not None

        window.destroy()
        root.update()

        # The caller's handler is dropped and no button still calls back into it.
        assert controls._close is None
        # The buttons themselves are gone, and the caller's handler was never
        # called again, so a stray event cannot reach the closed window.
        assert not controls.maximize_button.winfo_exists()
        assert closed == []
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_shared_composer_toolbar_controls_have_a_subtle_hover() -> None:
    """The ＋, the mode pills, and Send tint on hover, then restore their rest colour.

    Tk's ``activebackground`` only paints while the button is held, so the
    reference design's hover needs a real ``<Enter>``/``<Leave>`` treatment; the
    control must return to exactly its resting colour on leave.
    """
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.chat_composer import ChatComposer
    from gui.typography import resolve

    root = tk.Tk()
    try:
        composer = ChatComposer(
            root, theme=DARK_THEME, typography=resolve(root),
            placeholder="Ask", on_submit=lambda: None,
        )
        composer.wrap.pack(fill="x")
        selector = composer.add_mode_selector(
            icon="⌘", label="Code Studio", options=[("Open", lambda: None)],
        )
        root.update()

        attach = composer.attach_button
        assert attach is not None
        for widget, rest in (
            (attach, DARK_THEME.surface),
            (selector, DARK_THEME.surface),
        ):
            assert str(widget.cget("bg")) == rest
            widget.event_generate("<Enter>")
            root.update()
            assert str(widget.cget("bg")) == DARK_THEME.surface_raised
            widget.event_generate("<Leave>")
            root.update()
            assert str(widget.cget("bg")) == rest

        # The send disc is a canvas, so its hover is the oval's fill, not a ``bg``.
        # Type a message first: an empty box greys Send out, and a disabled disc
        # deliberately does not lighten (see the next test).
        composer.set_text("hello")
        root.update()
        send = composer.send_button
        rest_fill = str(send.itemcget(send._shape, "fill"))
        send.event_generate("<Enter>")
        root.update()
        hovered = str(send.itemcget(send._shape, "fill"))
        assert hovered != rest_fill
        send.event_generate("<Leave>")
        root.update()
        assert str(send.itemcget(send._shape, "fill")) == rest_fill
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_shared_composer_disabled_send_does_not_lighten_on_hover() -> None:
    """A disabled Send disc keeps its disabled fill when the pointer passes over it."""
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.chat_composer import ChatComposer
    from gui.typography import resolve

    root = tk.Tk()
    try:
        composer = ChatComposer(
            root, theme=DARK_THEME, typography=resolve(root),
            placeholder="Ask", on_submit=lambda: None,
        )
        composer.wrap.pack(fill="x")
        root.update()
        # An empty box already greys Send out.
        disabled_fill = str(composer.send_button.itemcget(composer.send_button._shape, "fill"))
        composer.send_button.event_generate("<Enter>")
        root.update()
        assert str(composer.send_button.itemcget(composer.send_button._shape, "fill")) == disabled_fill
    finally:
        root.destroy()


def test_composer_hover_blend_derives_from_the_theme_colour() -> None:
    """``_blend`` mixes toward the other colour and passes odd input through."""
    from gui.chat_composer import _blend

    assert _blend("#000000", "#FFFFFF", 0.5) == "#808080"
    assert _blend("#C9B974", "#FFFFFF", 0.0) == "#C9B974"
    # A non-#rrggbb colour (a name, or #rgb) is returned untouched, never mangled.
    assert _blend("gold", "#FFFFFF", 0.2) == "gold"
    assert _blend("#abc", "#FFFFFF", 0.2) == "#abc"
