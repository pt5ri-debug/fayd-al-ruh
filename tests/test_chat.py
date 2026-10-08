"""The chat command: talking with the AI in text, like Jarvis without voice."""

from __future__ import annotations

import json

from ai import chat
from core.assistant import GroundedAssistant
from core.command_router import NaturalLanguageRouter
from core.curriculum_knowledge import seed_curriculum
from core.documents import build_document_store
from core.domain_knowledge import seed_ontology
from core.jarvis import TEXT_HELP_TEXT, JarvisMode
from core.ontology import Ontology
from core.rag import RagRetriever
from core.speech import SilentSynthesizer, TranscriptRecognizer, VoiceAssistant
from core.tool_agent import ToolAgent
from core.tools import build_sandbox


def _assistant():
    ontology = Ontology()
    seed_ontology(ontology)
    seed_curriculum(ontology)
    return GroundedAssistant(RagRetriever(build_document_store(), ontology=ontology))


def _mode(tmp_path):
    sandbox = build_sandbox(tmp_path)
    assistant = _assistant()
    return JarvisMode(
        assistant,
        router=NaturalLanguageRouter(sandbox),
        agent=ToolAgent(sandbox),
        voice=VoiceAssistant(assistant, recognizer=TranscriptRecognizer(), synthesizer=SilentSynthesizer()),
        help_text=TEXT_HELP_TEXT,
    )


# --- turn rendering -------------------------------------------------------


def test_format_turn_renders_an_answer_with_sources(tmp_path) -> None:
    turn = _mode(tmp_path).handle("what is porosity?", speak=False)
    text = chat.format_turn(turn)
    assert turn.response in text
    assert "Sources:" in text


def test_format_turn_renders_a_control(tmp_path) -> None:
    turn = _mode(tmp_path).handle("help", speak=False)
    text = chat.format_turn(turn)
    assert text.startswith("[help]")
    assert "new conversation" in text


def test_format_turn_renders_a_tool_run(tmp_path) -> None:
    turn = _mode(tmp_path).handle("calculate 2 * (3 + 4)", speak=False)
    text = chat.format_turn(turn)
    assert "[ok ] calculator: 14" in text


def test_format_turn_marks_ignored_and_silence() -> None:
    from core.jarvis import JarvisTurn

    ignored = JarvisTurn(heard="x", response="", woke=False, routed="ignored")
    silence = JarvisTurn(heard="", response="", woke=False, routed="silence")
    assert chat.format_turn(ignored) == "(no wake word; ignored)"
    assert chat.format_turn(silence) == "(no speech detected)"


def test_turn_payload_is_machine_readable(tmp_path) -> None:
    turn = _mode(tmp_path).handle("what is porosity?", speak=False)
    payload = chat.turn_payload(turn)
    assert payload["routed"] == "answer"
    assert payload["citations"]
    assert "used_web" in payload and "learned" in payload


def test_turn_payload_includes_tool_steps(tmp_path) -> None:
    turn = _mode(tmp_path).handle("calculate 2 * (3 + 4)", speak=False)
    payload = chat.turn_payload(turn)
    assert payload["steps"][0]["tool"] == "calculator"
    assert payload["steps"][0]["ok"] is True


# --- interactive loop -----------------------------------------------------


def test_run_loop_answers_then_quits(tmp_path, monkeypatch, capsys) -> None:
    mode = _mode(tmp_path)
    answers = iter(["what is porosity?", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert chat.run_loop(mode, as_json=False) == 0
    out = capsys.readouterr().out
    assert chat.INTRO in out
    assert "Sources:" in out


def test_run_loop_handles_eof(tmp_path, monkeypatch, capsys) -> None:
    mode = _mode(tmp_path)

    def _raise(_prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise)
    assert chat.run_loop(mode, as_json=False) == 0


def test_run_loop_autosaves_when_requested(tmp_path, monkeypatch) -> None:
    mode = _mode(tmp_path)
    saved: list[int] = []

    def _record_save() -> str:
        saved.append(1)
        return "saved"

    monkeypatch.setattr(mode, "save_session", _record_save)
    answers = iter(["what is porosity?", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    chat.run_loop(mode, as_json=False, autosave=True)
    assert saved


# --- command line ---------------------------------------------------------


def test_main_one_shot_prints_an_answer(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(chat, "build_session", lambda **_kw: _mode(tmp_path))
    assert chat.main(["what is porosity?"]) == 0
    out = capsys.readouterr().out
    assert "Sources:" in out


def test_main_json_one_shot(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(chat, "build_session", lambda **_kw: _mode(tmp_path))
    assert chat.main(["--json", "what is porosity?"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["routed"] == "answer"


def test_main_list_sessions_reports_none(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(chat, "build_session", lambda **_kw: _mode(tmp_path))
    assert chat.main(["--list-sessions"]) == 0
    assert "No saved sessions yet." in capsys.readouterr().out


def test_main_resumes_a_saved_session(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from ai.jarvis import build_session

    first = build_session(text_only=True)
    first.handle("what is porosity?", speak=False)
    session_id = first.conversation.id
    first.save_session()

    second = build_session(text_only=True)
    monkeypatch.setattr(chat, "build_session", lambda **_kw: second)
    assert chat.main(["--session", session_id, "and permeability?"]) == 0
    assert second.conversation.id == session_id
    assert "Resumed session" in capsys.readouterr().out


# --- text-only session wiring --------------------------------------------


def test_build_session_text_only_uses_silent_speech(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from ai.jarvis import build_session

    mode = build_session(text_only=True)
    assert mode.voice.recognizer.name == "transcript"
    assert mode.voice.synthesizer.name == "silent"
    assert mode.mute is True
    assert mode.help_text == TEXT_HELP_TEXT


def test_text_session_help_avoids_voice_wording(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from ai.jarvis import build_session

    mode = build_session(text_only=True)
    turn = mode.handle("help", speak=False)
    assert turn.response == TEXT_HELP_TEXT
    assert "mute my voice" not in turn.response


# --- menu / CLI wiring ----------------------------------------------------


def test_chat_is_wired_into_the_menu_and_cli() -> None:
    import cli.__main__ as cli_entry
    from cli import menu

    assert "chat" in cli_entry._COMMANDS
    titles = [title for title, _action, _args in menu.MENU]
    assert any("Chat with the AI" in title for title in titles)


def test_run_chat_dispatches_to_the_module(monkeypatch) -> None:
    from cli import menu

    captured: dict[str, object] = {}

    def fake_run_module(name: str, argv) -> int:
        captured["name"] = name
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(menu, "_run_module", fake_run_module)
    assert menu.run_chat(("--json",)) == 0
    assert captured == {"name": "ai.chat", "argv": ["--json"]}


def test_jarvis_and_chat_share_the_session_builder(monkeypatch) -> None:
    """Both commands go through ai.jarvis.build_session, so gates stay in one place."""
    from ai import chat as chat_module
    from ai import jarvis as jarvis_module

    seen: dict[str, object] = {}

    def fake_build_session(**kwargs):
        seen.clear()
        seen.update(kwargs)
        return _mode_for_wiring()

    monkeypatch.setattr(chat_module, "build_session", fake_build_session)
    chat_module.main(["--list-sessions"])
    assert seen.get("text_only") is True

    monkeypatch.setattr(jarvis_module, "build_session", fake_build_session)
    jarvis_module.main(["hello"])
    assert "text_only" not in seen


def _mode_for_wiring():
    from core.jarvis import JarvisTurn

    class _Stub:
        def list_sessions(self):
            return ()

        def handle(self, *args, **kwargs):
            return JarvisTurn(heard="", response="", woke=True, routed="answer")

        def play(self, _turn):
            return ""

        def resume_session(self, _session_id):
            return False

        def save_session(self):
            return ""

    return _Stub()
