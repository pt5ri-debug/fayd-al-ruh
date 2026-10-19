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
from gui import modes as gui_modes
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


def test_service_attach_file_puts_the_text_in_the_conversation(tmp_path) -> None:
    """A mouth that never retrieves (Ultron) still sees an attached file's text."""
    import zipfile

    from core.assistant import build_default_assistant
    from core.conversation import Conversation

    session = FakeSession()
    session.assistant = build_default_assistant()
    session._conversation = cast(Any, Conversation(title="real"))
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
    conversation = session.conversation
    # The whole file's text is reference material in the conversation itself.
    assert len(conversation.context_messages()) == 1
    assert "Zephyrium-9" in conversation.render_context()

    # Detaching takes both the evidence and the reference material back out.
    service.remove_attachment(service.attachments[0])

    assert conversation.context_messages() == ()
    assert conversation.render_context() == ""


def test_service_new_conversation_drops_attachment_reference_material(tmp_path) -> None:
    import zipfile

    from core.assistant import build_default_assistant
    from core.conversation import Conversation

    session = FakeSession()
    session.assistant = build_default_assistant()
    session._conversation = cast(Any, Conversation(title="real"))
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
    service.attach_file(docx)

    service.new_conversation()

    # The reference-material turns belonged to the previous conversation.
    assert service._context_turns == []


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


def test_actions_cover_the_mirrored_menu_items() -> None:
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


def test_screen_actions_are_hidden_from_the_sidebar() -> None:
    """The screens the ＋ menu already opens leave the sidebar.

    Entries 11..14 -- the Import Center, Code Studio, Presentation Studio, and
    image generation -- are reachable from the composer's ＋ menu, so their
    sidebar rows are hidden. They stay in ``ACTIONS`` (the CLI still resolves
    them) and are absent from ``visible_actions()``.
    """
    hidden = {action.key for action in gui_actions.ACTIONS if action.hidden}
    assert {"import-center", "code-studio", "presentation", "image"} <= hidden
    visible = {action.key for action in gui_actions.visible_actions()}
    assert not ({"import-center", "code-studio", "presentation", "image"} & visible)
    for key in ("import-center", "code-studio", "presentation", "image"):
        assert gui_actions.action_by_key(key).key == key


#: The visible CLI menu entries the window provides through its own controls
#: (the window itself, the reading-voice picker, Jarvis, Ultron, Vision, Banner,
#: Cloud and its free-provider hint, Natasha's companion action, and the two
#: text-chat mouths). The sidebar mirrors a later block instead of repeating them.
_WINDOW_OWNED_MENU_LABELS = frozenset(
    {
        "Open the graphical interface (ChatGPT-style)",
        "Read text aloud: choose a voice (men, women, boys, girls)",
        "Jarvis mode (listen, act, speak)",
        "Ultron (general local model, any subject)",
        "Vision (petroleum local model: reservoir, drilling, PVT)",
        "Banner (presentation local model: outlines and slide content)",
        "Cloud (remote model: ChatGPT, Claude, Gemini; needs an API key)",
        "Cloud: free providers (no card) and how to start",
        "Natasha (companion: lasting memory, tone, and relationship)",
        "Chat with the AI (text, like Jarvis without voice)",
        "Chat with the AI (trained model, no sources)",
    }
)

#: The visible CLI entries the sidebar mirrors, in order (the first one after the
#: window-owned controls through the Code Studio screen).
_MIRRORED_MENU_START = 14
_MIRRORED_MENU_STOP = 21

#: The Qwen3 speech entries sit before the window-owned controls (the window shows
#: their status in the reading-voice picker), so they are mirrored as their own
#: block rather than inside the range above.
_QWEN_MENU_LABELS = frozenset(
    {
        "Qwen3 speech: status of the local ASR + TTS models",
        "Qwen3 speech: download the local ASR model (~3.4 GB)",
        "Qwen3 speech: download the local TTS model (~3.4 GB)",
    }
)


def test_action_labels_match_the_mirrored_cli_menu_block() -> None:
    """The sidebar mirrors the CLI menu block the window has no control for.

    The window owns its own controls for the GUI, the reading voice, Jarvis,
    Ultron, and the two text-chat mouths, so it mirrors the next block instead.
    Each mirrored entry appears once, with the CLI's own label, and no
    window-owned label leaks into the sidebar.
    """
    from cli.menu import visible_menu

    mirrored = visible_menu()[_MIRRORED_MENU_START:_MIRRORED_MENU_STOP]
    gui_labels = [action.label for action in gui_actions.ACTIONS]
    for title, _command, _argv in mirrored:
        assert gui_labels.count(title) == 1, f"{title!r} must appear exactly once"
    for title in _WINDOW_OWNED_MENU_LABELS:
        assert title not in gui_labels, f"{title!r} is a window control, not a sidebar action"
    for title in _QWEN_MENU_LABELS:
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
def test_window_fits_a_short_model_answer_without_clipping_its_last_line() -> None:
    """A two-line model answer must show its body, not just its header.

    A regression guard: ``count("1.0", "end-1c", "displaylines")`` undercounts by
    one, which sized a short answer one line short. For a model answer -- a header
    line, a blank line and the body -- that clipped the body entirely, leaving a
    bubble that showed only the header and its Copy button.
    """
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        root.update()
        text = "header line\n\nporosity is the fraction of pore space"
        bubble = app._add_bubble("assistant", text)
        root.update()
        logical = int(bubble.index("end-1c").split(".")[0])
        assert int(bubble.cget("height")) >= logical
        # Nothing is scrolled out of view: the whole message is visible.
        first, last = bubble.yview()
        assert first <= 0.0 and last >= 1.0
        assert bubble.get("1.0", "end-1c") == text
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
        app._toggle_settings()  # the Actions list makes the sidebar long enough to overflow
        root.geometry("1160x300")  # short enough that the sidebar overflows
        for _ in range(10):
            root.update()
        app._toggle_sidebar_scrollbar()
        root.update()
        assert app._side_scrollbar.winfo_ismapped()
        root.geometry("1160x2400")  # tall enough to fit everything
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
    from gui.actions import visible_actions

    session = FakeSession()
    session.saved_ids = ("CNV-alpha", "CNV-beta")
    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(session))
        root.update()
        # The Actions list is behind Settings; opening it builds one button per
        # *visible* action (the hidden mouth switches are not drawn), none repeated.
        app._toggle_settings()
        root.update()
        assert len(app._action_buttons) == len(visible_actions())
        drawn = [button.cget("text") for button in app._action_buttons]
        assert not any("Answer from Ultron" in text for text in drawn)
        assert not any("Answer from Vision" in text for text in drawn)
        assert not any("Answer from Banner" in text for text in drawn)
        assert not any("Answer from Cloud" in text for text in drawn)
        labels = [button.cget("text") for button in app._session_buttons]
        assert any("CNV-alpha" in text for text in labels)
        assert any("CNV-beta" in text for text in labels)
        # Tapping a saved chat marks it current, and the list refreshes in place.
        app._on_resume_session(
            gui_modes.HistoryItem("CNV-alpha", "CNV-alpha", None)
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


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_uses_the_openhands_palette() -> None:
    """The window draws with OpenHands' cool-grey tokens in both appearances."""
    import tkinter as tk

    from gui.app import DARK_THEME, LIGHT_THEME, FaydGUI

    # Dark base and surface match the OpenHands cool-grey-950/925 fills exactly.
    assert DARK_THEME.base == "#0B0E14"
    assert DARK_THEME.surface == "#21252F"
    assert DARK_THEME.primary == "#C9B974"

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()), theme="dark")
        root.update()
        assert app.root.cget("bg") == DARK_THEME.base
        assert app._entry.cget("bg") == DARK_THEME.surface
        app._toggle_theme()
        root.update()
        assert app.root.cget("bg") == LIGHT_THEME.base
        assert app._entry.cget("bg") == LIGHT_THEME.surface
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_composer_is_a_rounded_card() -> None:
    """The composer is a rounded surface panel holding the input and send button."""
    import tkinter as tk

    from gui.app import RADIUS_LARGE, FaydGUI, RoundedBox

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        root.geometry("1200x780")
        for _ in range(10):
            root.update()
        assert isinstance(app._composer_box, RoundedBox)
        assert app._composer_box._corner == RADIUS_LARGE
        # The entry and the send button both live inside the rounded panel.
        assert app._entry.winfo_toplevel() is root
        assert app._entry.master is app._composer_box.inner
        assert app._send.master is app._composer_box.inner
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_composer_is_lifted_by_half_its_box_height() -> None:
    """The composer sits half an input box above the bottom of the transcript."""
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        root.geometry("1200x780")
        for _ in range(10):
            root.update()
        box_height = app._composer_box.winfo_height()
        assert box_height > 1
        # The lift equals half the input box height, and the composer box keeps
        # that much extra space below it inside the main column.
        assert app._composer_lift == box_height // 2
        main_bottom = app._main_frame.winfo_rooty() + app._main_frame.winfo_height()
        box_bottom = app._composer_box.winfo_rooty() + app._composer_box.winfo_height()
        assert main_bottom - box_bottom >= box_height // 2
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_messages_are_centred_in_a_readable_column() -> None:
    """Rows are indented into a centred column; user bubbles sit on the right."""
    import tkinter as tk

    from gui.app import _COLUMN_MAX, FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        root.geometry("1200x780")
        for _ in range(10):
            root.update()
        pad = app._column_pad()
        assert pad > 20  # a wide window centres the column rather than hugging the edge
        user = app._add_bubble("user", "short question")
        assistant = app._add_bubble("assistant", "a longer answer " * 6)
        for _ in range(10):
            root.update()
        # The assistant reply spans the column; the user bubble is narrower and
        # right-aligned inside the same column.
        box = user.master.master  # the user's RoundedBox, inside its row
        assert box is not None
        assert assistant.master is not None
        assert assistant.master.winfo_width() == _COLUMN_MAX
        assert box.winfo_width() <= _COLUMN_MAX
        assert box.winfo_x() > 0  # pushed toward the right within its row
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_send_button_is_a_circle() -> None:
    """The chat send control is a circular disc, not a rectangular button."""
    import tkinter as tk

    from gui.app import SEND_BUTTON_SIZE, CircleButton, FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        root.geometry("1200x780")
        for _ in range(10):
            root.update()
        assert isinstance(app._send, CircleButton)
        # A square side length is what makes the drawn oval a circle.
        assert app._send.winfo_width() == app._send.winfo_height() == SEND_BUTTON_SIZE
        # It stays inside the composer card, beside the input.
        assert app._send.master is app._composer_box.inner
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_composer_has_no_gold_focus_outline() -> None:
    """Focusing the input does not paint a gold border around the composer."""
    import tkinter as tk

    from gui.app import FaydGUI, RoundedBox

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        root.geometry("1200x780")
        for _ in range(10):
            root.update()
        # The composer's border is the neutral stroke; there is no focus recolour.
        assert isinstance(app._composer_box, RoundedBox)
        assert app._composer_box._outline == app._theme.border
        app._entry.focus_force()
        root.update()
        assert app._composer_box._outline == app._theme.border
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_bubbles_size_to_their_text() -> None:
    """A bubble's width and height follow the amount of text it holds."""
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        root.geometry("1200x780")
        for _ in range(10):
            root.update()
        short = app._add_bubble("user", "hi")
        longer = app._add_bubble("user", "what is the permeability of this core sample")
        root.update()
        # A longer question makes a wider bubble than a two-character one.
        assert longer.winfo_width() > short.winfo_width()
        # Both stay inside the readable column.
        assert longer.winfo_width() <= app._available_column()

        brief = app._add_bubble("assistant", "ok")
        tall = app._add_bubble("assistant", "a sentence about rock. " * 30)
        root.update()
        # More text wraps to more lines, so the bubble is taller.
        assert int(tall.cget("height")) > int(brief.cget("height"))
        assert tall.winfo_width() <= app._available_column()
    finally:
        root.destroy()


# --- reading aloud (text to speech) ---------------------------------------


class _FakeVoiceEngine:
    """A synthesizer that claims a real engine and returns non-silent audio."""

    name = "fake-engine"

    def synthesize(self, text):
        from core.speech import AudioClip

        return AudioClip(tuple([1000, -1000] * 400), sample_rate=16_000)


class _FakePlayback:
    """A stoppable playback that reports itself as still playing."""

    def __init__(self) -> None:
        self.stopped = False

    @property
    def playing(self) -> bool:
        return not self.stopped

    def stop(self) -> None:
        self.stopped = True

    def wait(self, *, timeout: float | None = None) -> None:
        return None


class _FakeSpeaker:
    """A player that records the clips it was asked to start."""

    def __init__(self) -> None:
        self.started: list = []

    def available(self) -> bool:
        return True

    def play(self, clip) -> None:
        self.started.append(clip)

    def start(self, clip) -> _FakePlayback:
        self.started.append(clip)
        return _FakePlayback()


def _wait(app, root, predicate, tries: int = 120) -> bool:
    """Pump the Tk event loop until ``predicate`` holds (or give up)."""
    import time

    for _ in range(tries):
        root.update()
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_reads_a_message_aloud_and_picks_a_voice() -> None:
    """The reading controls: a per-message button, a voice picker, honest notes."""
    import tkinter as tk

    from core.voices import VOICE_CATALOG
    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        root.update()

        # The settings panel exposes one radio per catalog voice, grouped.
        app._toggle_settings()
        root.update()

        def walk(widget, out):
            for child in widget.winfo_children():
                out.append(child)
                walk(child, out)

        widgets: list = []
        walk(app._settings_panel, widgets)
        keys = {voice.key for voice in VOICE_CATALOG}
        radios = [
            w for w in widgets
            if isinstance(w, tk.Radiobutton) and w.cget("value") in keys
        ]
        assert len(radios) == len(VOICE_CATALOG)

        # Choosing a voice applies it to the service and persists the key.
        app._on_voice_change("girl-3")
        assert app._voice_key == "girl-3"
        assert app.service.voice_key() == "girl-3"

        # The auto-speak toggle persists too.
        app._auto_speak.set(True)
        app._on_auto_speak_toggle()
        assert app._auto_speak.get() is True

        # A fake real engine + player, installed *after* the voice change (which
        # rebuilds the engine), so the test never depends on a system voice.
        app.service.reader._synthesizer = _FakeVoiceEngine()
        speaker = _FakeSpeaker()
        app.service.reader._speaker = speaker  # type: ignore[assignment]

        # Reading a message plays it and reports the duration honestly.
        app._speak_text("porosity is the fraction of pore space")
        assert _wait(app, root, lambda: app._playback is not None)
        notes = [entry.text for entry in app._rows if entry.role == "note"]
        assert any("reading aloud" in note.lower() for note in notes)
        assert speaker.started, "the clip was rendered and played"
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_speaker_icon_toggles_reading_on_and_off() -> None:
    """A second click on the speaker control stops the reading it started."""
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        app.service.reader._synthesizer = _FakeVoiceEngine()
        app.service.reader._speaker = _FakeSpeaker()  # type: ignore[assignment]
        root.update()

        widget = app._add_bubble("assistant", "porosity is the fraction of pore space")
        root.update()
        # The message footer carries the read-aloud control.
        footer = widget.master.winfo_children()[-1]
        label = next(
            w for w in footer.winfo_children()
            if isinstance(w, tk.Label) and "Read aloud" in str(w.cget("text"))
        )

        # First click: the label flips to "Stop" and playback starts.
        app._toggle_speak(label, "porosity is the fraction of pore space")
        assert _wait(app, root, lambda: app._playback is not None)
        assert "Stop" in str(label.cget("text"))

        # Second click on the same control: the reading stops, label restored.
        app._toggle_speak(label, "porosity is the fraction of pore space")
        assert app._playback is None
        assert "Read aloud" in str(label.cget("text"))
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_microphone_reports_honestly_when_no_recorder_is_installed() -> None:
    """With no recorder installed, the microphone says so and names the next step."""
    import tkinter as tk

    from core.speech import MicrophoneRecorder
    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        root.update()
        if MicrophoneRecorder().available():
            return  # a real recorder exists here; the honest-note path needs none
        app._toggle_mic()
        root.update()
        notes = [entry.text for entry in app._rows if entry.role == "note"]
        assert any("no microphone recorder is installed" in note.lower() for note in notes)
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_microphone_transcribes_into_the_composer() -> None:
    """The microphone flow places the transcript in the input box, ready to edit."""
    import tkinter as tk

    from core.speech import AudioClip
    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))

        class _Recording:
            recording = True

            def result(self, *, timeout=None):
                return AudioClip(tuple([500, -500] * 100))

        app.service.microphone_available = lambda: True  # type: ignore[method-assign]
        app.service.recognizer_name = lambda: "qwen3-asr"  # type: ignore[method-assign]
        app.service.start_recording = lambda: _Recording()  # type: ignore[assignment,method-assign,return-value]
        app.service.stop_recording = lambda: _Recording()  # type: ignore[assignment,method-assign,return-value]
        app.service.transcribe = lambda clip: "ما هي المسامية؟"  # type: ignore[method-assign]
        root.update()

        app._toggle_mic()
        root.update()
        assert app._recording is not None
        app._toggle_mic()  # stop and transcribe
        assert _wait(app, root, lambda: "المسامية" in app._entry.get("1.0", "end"))
        assert "المسامية" in app._entry.get("1.0", "end")
        notes = [entry.text for entry in app._rows if entry.role == "note"]
        assert any("Heard" in note for note in notes)
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_speaks_honestly_when_no_engine_is_installed() -> None:
    import tkinter as tk

    from core.speech import SilentSynthesizer
    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        app.service.reader._synthesizer = SilentSynthesizer()
        root.update()
        app._speak_text("hello there")
        root.update()
        notes = [entry.text for entry in app._rows if entry.role == "note"]
        assert any("no speech engine is installed" in note.lower() for note in notes)
        # The note names the real next step (the platform's install command), not a
        # vague "go install something".
        assert any("install-engine" in note for note in notes)
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_offers_to_install_the_speech_engine() -> None:
    import tkinter as tk

    from core import speech
    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        root.update()
        # The sidebar offers the install entry while no engine is present.
        if not app.service.reading_available():
            app._toggle_settings()
            root.update()
            labels = [w.cget("text") for w in app._settings_panel.winfo_children() if "text" in w.keys()]
            assert any("Install a speech engine" in label for label in labels)

        # Rendering a successful install refreshes the reader and confirms it.
        refreshed = []
        app.service.refresh_reading = lambda: refreshed.append(True)  # type: ignore[method-assign]
        app._render_speech_install(
            speech.SpeechInstallResult(ok=True, message="Installed a speech engine (espeak-ng).")
        )
        root.update()
        notes = [entry.text for entry in app._rows if entry.role == "note"]
        assert any("Installed a speech engine" in note for note in notes)
        assert refreshed == [True]
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_markdown_is_stripped_before_reading() -> None:
    from gui.app import _strip_markdown

    spoken = _strip_markdown("# Title\n\n**bold** and `code`\n\n- item one\n- item two")
    assert "#" not in spoken and "**" not in spoken and "`" not in spoken
    assert "bold" in spoken and "code" in spoken and "item one" in spoken



@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_opens_the_presentation_studio_from_its_menu() -> None:
    """The Studio opens from the window and reuses one dialog, not a second AI."""
    import tkinter as tk

    from gui.actions import ACTIONS
    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        service = _service(FakeSession())
        app = FaydGUI(root, service=service)
        for _ in range(20):
            root.update()

        # The sidebar action exists and the Studio shares this window's session.
        assert any(action.key == "presentation" for action in ACTIONS)
        assert service.presentation is not None

        app._open_presentation()
        for _ in range(30):
            root.update()
        dialog = app._presentation_dialog
        assert dialog is not None
        assert dialog.window.title() == "Presentation Studio"
        # A second open reuses the window instead of stacking another one.
        app._open_presentation()
        assert app._presentation_dialog is dialog
        dialog._close()
        assert app._presentation_dialog is None
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_shows_the_free_cloud_providers() -> None:
    """The window points a user who does not want to pay at the no-card providers.

    The Settings panel carries a one-line "Free (no card)" note, and the button
    posts the providers with their key variable and signup link -- the same list
    the CLI's ``cloud free`` prints, so the two never disagree.
    """
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_service(FakeSession()))
        root.update()
        app._toggle_settings()
        root.update()

        def walk(widget, out):
            for child in widget.winfo_children():
                out.append(child)
                walk(child, out)

        widgets: list = []
        walk(app._settings_panel, widgets)
        labels = [w.cget("text") for w in widgets if isinstance(w, tk.Label)]
        assert any("Free (no card)" in text for text in labels)
        assert any("Gemini" in text for text in labels)

        app._show_free_cloud_providers()
        root.update()
        posted = app._bubbles[-1].get("1.0", "end-1c")
        assert "GEMINI_API_KEY" in posted
        assert "https://aistudio.google.com/apikey" in posted
    finally:
        root.destroy()
