"""The ChatGPT-style GUI: the headless service, actions, and CLI wiring.

Tkinter is only touched when a window is actually opened, so these tests drive
the logic that a window would use without needing a display.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

from cli import menu as cli_menu
from core.jarvis import JarvisMode, JarvisTurn
from gui import actions as gui_actions
from gui.service import ChatReply, ChatService


class FakeSession:
    """A stand-in for ``JarvisMode`` that records calls and returns known turns."""

    def __init__(self, turn: JarvisTurn | None = None) -> None:
        self.turn = turn
        self.heard: list[str] = []
        self.saved = 0
        self.resumed: list[str] = []
        self.assistant: Any = SimpleNamespace(name="grounded")
        self.help_text = "help"
        self.voice = SimpleNamespace(assistant=self.assistant)
        self._conversation = SimpleNamespace(id="CNV-fake")
        self.saved_ids: tuple[str, ...] = ()

    @property
    def conversation(self) -> SimpleNamespace:
        return self._conversation

    def handle(self, message: str, *, require_wake: bool = False, speak: bool = True) -> JarvisTurn:
        self.heard.append(message)
        assert require_wake is False
        assert speak is False
        return self.turn or JarvisTurn(heard=message, response="ok", woke=True, routed="answer")

    def reset_conversation(self) -> SimpleNamespace:
        self._conversation = SimpleNamespace(id="CNV-new")
        return self._conversation

    def save_session(self) -> str:
        self.saved += 1
        return f"Saved this conversation as {self.conversation.id}."

    def resume_session(self, conversation_id: str) -> bool:
        self.resumed.append(conversation_id)
        if conversation_id in self.saved_ids or conversation_id == self.conversation.id:
            self._conversation = SimpleNamespace(id=conversation_id)
            return True
        return False

    def list_sessions(self) -> tuple[str, ...]:
        return self.saved_ids


def _service(mode: FakeSession, **kwargs: Any) -> ChatService:
    return ChatService(mode=cast(JarvisMode, mode), **kwargs)


def _service_with_retriever() -> ChatService:
    """A service backed by a real assistant, so attachments have a store to join."""
    from core.assistant import build_default_assistant

    session = FakeSession()
    session.assistant = build_default_assistant()
    return _service(session)


# --- service --------------------------------------------------------------


def test_service_attach_file_indexes_and_detaches(tmp_path) -> None:
    import zipfile

    from core.assistant import build_default_assistant

    session = FakeSession()
    session.assistant = build_default_assistant()
    service = ChatService(mode=cast(JarvisMode, session))
    docx = tmp_path / "well.docx"
    body = "<w:p><w:r><w:t>Zephyrium-9 flowed at 4200 barrels per day.</w:t></w:r></w:p>"
    xml = (
        '<?xml version="1.0"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    with zipfile.ZipFile(docx, "w") as archive:
        archive.writestr("word/document.xml", xml)

    notice = service.attach_file(docx)

    assert notice.ok is True
    assert len(service.attachments) == 1
    doc_id = service.attachments[0].doc_ids[0]
    assert session.assistant.retriever.documents.get(doc_id) is not None
    # A question can now retrieve the attached text and cite it.
    from core.assistant import GroundedAssistant

    assert isinstance(session.assistant, GroundedAssistant)
    answer = session.assistant.ask("What did Zephyrium-9 flow at?")
    assert any("attachment" in citation for citation in answer.citations)

    removed = service.remove_attachment(service.attachments[0])

    assert removed == 1
    assert session.assistant.retriever.documents.get(doc_id) is None
    assert service.attachments == []


def test_service_attach_file_reports_failure(tmp_path) -> None:
    from core.assistant import build_default_assistant

    session = FakeSession()
    session.assistant = build_default_assistant()
    service = ChatService(mode=cast(JarvisMode, session))

    notice = service.attach_file(tmp_path / "missing.docx")

    assert notice.ok is False
    assert service.attachments == []


def test_service_answers_and_renders_sources() -> None:
    turn = JarvisTurn(
        heard="what is porosity?",
        response="A fraction of pore space.",
        woke=True,
        routed="answer",
        citations=("Reservoir rock properties (reservoir)",),
    )
    service = _service(FakeSession(turn))

    reply = service.ask("what is porosity?")

    assert isinstance(reply, ChatReply)
    assert "A fraction of pore space." in reply.text
    assert "Sources:" in reply.text
    assert reply.citations == ("Reservoir rock properties (reservoir)",)
    assert reply.mode == "sources"


def test_service_marks_control_turns() -> None:
    turn = JarvisTurn(heard="help", response="controls", woke=True, routed="control", intent="help")
    service = _service(FakeSession(turn))

    reply = service.ask("help")

    assert reply.is_control is True
    assert reply.text.startswith("[help]")


def test_service_hints_when_sources_cannot_answer() -> None:
    from ai.chat import MIND_HINT, _NO_SOURCE_MARKER

    turn = JarvisTurn(
        heard="q",
        response=f"not enough evidence {_NO_SOURCE_MARKER}",
        woke=True,
        routed="answer",
        citations=("some source",),
    )
    service = _service(FakeSession(turn))

    reply = service.ask("q")

    assert reply.hint == MIND_HINT


def test_service_new_conversation_and_save() -> None:
    session = FakeSession()
    service = _service(session)

    new_id = service.new_conversation()
    status = service.save()

    assert new_id == "CNV-new"
    assert session.saved == 1
    assert service.session == "CNV-new"
    assert "Saved" in status


def test_service_resumes_a_saved_session() -> None:
    session = FakeSession()
    service = _service(session, session="CNV-fake")

    assert service.session == "CNV-fake"
    assert session.resumed == ["CNV-fake"]


def test_service_switches_between_model_and_sources() -> None:
    session = FakeSession()
    service = _service(session)
    service.switcher._loader = lambda: "MODEL"  # type: ignore[assignment]

    assert service.source == "sources"
    service.use_model()
    assert service.source == "model"
    assert service.mode.assistant is not None
    service.use_sources()
    assert service.source == "sources"


def test_service_lists_and_resumes_saved_sessions() -> None:
    session = FakeSession()
    session.saved_ids = ("CNV-fake", "CNV-other")
    service = _service(session)

    assert service.list_sessions() == ("CNV-fake", "CNV-other")

    assert service.resume("CNV-fake") is True
    assert service.session == "CNV-fake"
    assert service.resume("CNV-missing") is False
    assert session.resumed == ["CNV-fake", "CNV-missing"]


# --- actions --------------------------------------------------------------


def test_every_action_is_callable_and_has_a_unique_key() -> None:
    keys = [action.key for action in gui_actions.ACTIONS]
    assert len(keys) == len(set(keys))
    for action in gui_actions.ACTIONS:
        assert callable(action.command)
        assert action.label


def test_actions_cover_numbered_menu_items_five_through_eleven() -> None:
    """Learn, insight, internet, web, and both knowledge entries are present."""
    keys = {action.key for action in gui_actions.ACTIONS}
    for key in (
        "learn",
        "insight",
        "internet-learner",
        "web",
        "knowledge-build",
        "knowledge-stats",
    ):
        assert key in keys, f"missing action {key!r}"


def test_action_labels_match_cli_menu_items_five_to_eleven() -> None:
    """Every numbered CLI item from 5 to 11 appears once, with the same label."""
    from cli.menu import visible_menu

    cli_items = visible_menu()[4:11]  # 1-based items 5..11
    gui_labels = [action.label for action in gui_actions.ACTIONS]
    for title, _command, _argv in cli_items:
        assert gui_labels.count(title) == 1, f"{title!r} must appear exactly once"


def test_actions_never_duplicate_a_gui_control() -> None:
    """The window's own controls own the first menu items; the sidebar must not repeat them."""
    forbidden = {"gui", "voice", "agent", "jarvis", "chat", "mind-chat", "new-chat", "save"}
    keys = {action.key for action in gui_actions.ACTIONS}
    assert not (keys & forbidden)

    labels = [action.label for action in gui_actions.ACTIONS]
    assert len(labels) == len(set(labels)), "two actions share a label"


def test_web_action_asks_for_a_question() -> None:
    action = gui_actions.action_by_key("web")
    assert action.needs_input is True
    assert "--approve" in action.argv


def test_action_by_key_round_trips_and_rejects_unknown() -> None:
    action = gui_actions.ACTIONS[0]
    assert gui_actions.action_by_key(action.key) is action
    with pytest.raises(KeyError):
        gui_actions.action_by_key("does-not-exist")


def test_action_commands_are_the_cli_menu_functions() -> None:
    for action in gui_actions.ACTIONS:
        assert action.command.__module__ == "cli.menu"


def test_saved_session_items_mark_the_current_chat() -> None:
    items = gui_actions.saved_session_items(("CNV-1", "CNV-2"), current="CNV-2")

    assert [item.session_id for item in items] == ["CNV-1", "CNV-2"]
    assert items[1].is_current is True
    assert items[0].is_current is False
    assert "current" in items[1].label
    assert items[0].key == "session:CNV-1"


def test_saved_session_items_are_empty_without_saves() -> None:
    assert gui_actions.saved_session_items(()) == ()


# --- CLI wiring must stay import-safe -------------------------------------


def test_gui_command_is_registered_in_the_cli() -> None:
    assert "run_gui" in cli_menu.__all__

    from cli.__main__ import _COMMANDS

    assert _COMMANDS["gui"] is cli_menu.run_gui


def test_importing_the_cli_does_not_import_tkinter() -> None:
    """A headless machine without Tk must still be able to run ``python -m cli``."""
    code = (
        "import cli.menu, cli.__main__, sys;"
        "assert 'tkinter' not in sys.modules, 'cli imported tkinter eagerly'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_gui_parser_defaults() -> None:
    from gui.options import build_parser

    args = build_parser().parse_args([])
    assert args.no_web is False
    assert args.mind is False
    assert args.session is None


# --- window (only when a display is available) ----------------------------


def _display_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_builds_and_sends_a_message() -> None:
    import time
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        root.update()
        app._entry.insert("1.0", "hello")
        app._on_send()
        for _ in range(40):
            root.update()
            time.sleep(0.02)
        texts = [bubble.get("1.0", "end-1c") for bubble in app._bubbles]
        assert any("hello" in text for text in texts)
        assert any("ok" in text for text in texts)
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_bubbles_are_selectable_and_copyable() -> None:
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        root.update()
        app._add_bubble("assistant", "porosity is the fraction of pore space")
        root.update()
        bubble = app._bubbles[-1]
        # The text is selectable, so a copy can pick it up.
        bubble.tag_add("sel", "1.0", "end-1c")
        app._on_copy()
        root.update()
        assert "porosity" in root.clipboard_get()
        # With no selection, Ctrl+C must not clobber the clipboard.
        bubble.tag_remove("sel", "1.0", "end-1c")
        root.clipboard_clear()
        root.clipboard_append("keep me")
        root.update()
        app._on_copy()
        root.update()
        assert root.clipboard_get() == "keep me"
        # The context menu can copy the whole conversation, then clear it.
        app._on_copy_conversation()
        root.update()
        assert "porosity" in root.clipboard_get()
        app._clear_transcript()
        assert app._bubbles == []
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_coalesces_a_burst_of_progress_lines() -> None:
    import time
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        root.update()
        app._streaming_label = app._add_bubble("note", "training — live output:")
        app._last_render = time.monotonic()  # simulate a render just happened
        for i in range(500):
            app._append_progress_line(f"line {i}")
        app._flush_progress()  # rate-limited: nothing rendered yet
        assert "line 499" not in app._streaming_label.get("1.0", "end-1c")
        app._flush_progress(force=True)  # the forced flush renders them all
        text = app._streaming_label.get("1.0", "end-1c")
        assert "line 0" in text and "line 499" in text
        assert text.count("line") == 500
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_starts_with_a_hero_and_hides_it_after_a_message() -> None:
    import time
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        root.update()
        assert app._empty is not None  # the welcome/empty state is shown
        app._entry.insert("1.0", "hello")
        app._on_send()
        for _ in range(40):
            root.update()
            time.sleep(0.02)
        assert app._empty is None  # the first message replaces it
        # Clearing the transcript brings the hero back.
        app._clear_transcript()
        root.update()
        assert app._empty is not None
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_per_bubble_copy_button_copies_that_message() -> None:
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        root.update()
        first = app._add_bubble("assistant", "the first answer")
        app._add_bubble("assistant", "the second answer")
        root.clipboard_clear()
        app._copy_widget(first)
        root.update()
        assert root.clipboard_get() == "the first answer"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_shows_and_clears_a_thinking_indicator() -> None:
    import time
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        root.update()
        app._entry.insert("1.0", "hello")
        app._on_send()
        root.update()
        assert app._thinking_row is not None  # a pending answer is visible
        assert app._thinking_row.winfo_ismapped()
        for _ in range(40):
            root.update()
            time.sleep(0.02)
        # The indicator is hidden (not mapped) once the answer lands.
        assert app._thinking_row is not None
        assert not app._thinking_row.winfo_ismapped()
        assert app._busy is False
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_theme_toggle_switches_palette_and_persists(tmp_path, monkeypatch) -> None:
    import tkinter as tk

    from gui import app as app_module
    from gui.app import DARK_THEME, FaydGUI, LIGHT_THEME

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        root.update()
        assert app._theme is DARK_THEME
        app._add_bubble("assistant", "keep me across the switch")
        app._toggle_theme()
        root.update()
        assert app._theme is LIGHT_THEME
        # The transcript survives the rebuild.
        texts = [bubble.get("1.0", "end-1c") for bubble in app._bubbles]
        assert any("keep me across the switch" in text for text in texts)
        # The choice is written where a new launch will read it.
        assert app_module.load_theme_name() == "light"
        # And back again.
        app._toggle_theme()
        root.update()
        assert app._theme is DARK_THEME
        assert app_module.load_theme_name() == "dark"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_sidebar_has_a_scrollbar_shown_only_on_overflow() -> None:
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        root.geometry("1160x420")  # short enough that the sidebar overflows
        for _ in range(10):
            root.update()
        # The actions overflow a short window, so the scrollbar is mapped.
        app._toggle_sidebar_scrollbar()
        root.update()
        assert app._side_scrollbar.winfo_ismapped()
        root.geometry("1160x2000")  # tall enough to fit everything
        for _ in range(10):
            root.update()
        app._toggle_sidebar_scrollbar()
        root.update()
        assert not app._side_scrollbar.winfo_ismapped()
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_attaches_a_document_and_shows_a_chip(tmp_path) -> None:
    import tkinter as tk
    import zipfile

    from gui.app import FaydGUI

    docx = tmp_path / "well.docx"
    body = "<w:p><w:r><w:t>The Zephyrium-9 well flowed at 4200 bopd.</w:t></w:r></w:p>"
    xml = (
        '<?xml version="1.0"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    with zipfile.ZipFile(docx, "w") as archive:
        archive.writestr("word/document.xml", xml)

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service_with_retriever())
        root.update()
        assert app.service.attachments == []
        notice = app._on_attach(str(docx))
        root.update()
        assert notice.ok is True
        assert len(app.service.attachments) == 1
        # A removable chip is shown for the attached file.
        assert app._attachment_chips
        assert app._attachments_row.winfo_ismapped()
        # Detaching removes it from the service and the chips.
        app._detach(app.service.attachments[0])
        root.update()
        assert app.service.attachments == []
        assert app._attachment_chips == []
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_attachment_failure_is_reported_not_raised(tmp_path) -> None:
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service_with_retriever())
        root.update()
        notice = app._on_attach(str(tmp_path / "missing.docx"))
        root.update()
        assert notice.ok is False
        assert app.service.attachments == []
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_shows_actions_and_saved_chats() -> None:
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.actions import ACTIONS

    session = FakeSession()
    session.saved_ids = ("CNV-alpha", "CNV-beta")
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(session))
        root.update()
        # One button per action, none of them repeated.
        assert len(app._action_buttons) == len(ACTIONS)
        labels = [button.cget("text") for button in app._session_buttons]
        assert any("CNV-alpha" in text for text in labels)
        assert any("CNV-beta" in text for text in labels)
        # Tapping a saved chat marks it current, and the list refreshes in place.
        app._on_resume_session(
            gui_actions.SessionItem("session:CNV-alpha", "CNV-alpha", "CNV-alpha")
        )
        root.update()
        assert app.service.session == "CNV-alpha"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_tolerates_a_broken_session_store() -> None:
    import tkinter as tk

    from gui.app import FaydGUI

    class BrokenSession(FakeSession):
        def list_sessions(self) -> tuple[str, ...]:
            raise OSError("store is unreadable")

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(BrokenSession()))
        root.update()
        labels = [button.cget("text") for button in app._session_buttons]
        assert labels == ["No saved chats yet."]
    finally:
        root.destroy()
