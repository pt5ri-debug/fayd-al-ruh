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
        self.assistant = SimpleNamespace(name="grounded")
        self.help_text = "help"
        self.voice = SimpleNamespace(assistant=self.assistant)
        self._conversation = SimpleNamespace(id="CNV-fake")

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
        return conversation_id == self.conversation.id


def _service(mode: FakeSession, **kwargs: Any) -> ChatService:
    return ChatService(mode=cast(JarvisMode, mode), **kwargs)


# --- service --------------------------------------------------------------


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


# --- actions --------------------------------------------------------------


def test_every_action_is_callable_and_has_a_unique_key() -> None:
    keys = [action.key for action in gui_actions.ACTIONS]
    assert len(keys) == len(set(keys))
    for action in gui_actions.ACTIONS:
        assert callable(action.command)
        assert action.label


def test_action_by_key_round_trips_and_rejects_unknown() -> None:
    action = gui_actions.ACTIONS[0]
    assert gui_actions.action_by_key(action.key) is action
    with pytest.raises(KeyError):
        gui_actions.action_by_key("does-not-exist")


def test_action_commands_are_the_cli_menu_functions() -> None:
    for action in gui_actions.ACTIONS:
        assert action.command.__module__ == "cli.menu"


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
        texts = [bubble.cget("text") for bubble in app._bubbles]
        assert any("hello" in text for text in texts)
        assert any("ok" in text for text in texts)
    finally:
        root.destroy()
