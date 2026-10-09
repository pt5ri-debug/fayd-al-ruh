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


def test_run_mind_chat_adds_the_mind_flag(monkeypatch) -> None:
    from cli import menu

    captured: dict[str, object] = {}

    def fake_run_module(name: str, argv) -> int:
        captured["name"] = name
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(menu, "_run_module", fake_run_module)
    assert menu.run_mind_chat(()) == 0
    assert captured == {"name": "ai.chat", "argv": ["--mind"]}

    assert menu.run_mind_chat(("--json",)) == 0
    assert captured == {"name": "ai.chat", "argv": ["--mind", "--json"]}

    # An explicit --mind must not be duplicated.
    assert menu.run_mind_chat(("--mind",)) == 0
    assert captured == {"name": "ai.chat", "argv": ["--mind"]}


def test_mind_chat_is_wired_into_the_menu_and_cli() -> None:
    import cli.__main__ as cli_entry
    from cli import menu

    assert "mind-chat" in cli_entry._COMMANDS
    titles = [title for title, _action, _args in menu.MENU]
    assert any("trained model" in title for title in titles)


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


# --- model mouth ("from the model") ---------------------------------------


class _FakeMindModel:
    """A stand-in model whose ``generator.generate`` echoes a fixed continuation."""

    def __init__(self, continuation: str) -> None:
        self._continuation = continuation

        class _Generator:
            def __init__(self, continuation: str) -> None:
                self._continuation = continuation

            def generate(self, prompt: str, *, max_new_tokens: int = 64, seed: int | None = None) -> str:
                return prompt + self._continuation

        self.generator = _Generator(continuation)


def _switcher_mode(tmp_path, continuation="model prose"):
    """A chat mode plus a switcher wired to a fake model (no numpy needed)."""
    mode = _mode(tmp_path)
    switcher = chat.ModelSwitcher(
        mode, loader=lambda: _FakeMindModel(continuation), max_new_tokens=16
    )
    return mode, switcher


def test_match_switch_recognises_model_and_sources_phrases() -> None:
    assert chat._match_switch("from the model") == "model"
    assert chat._match_switch("Model on.") == "model"
    assert chat._match_switch("من الموديل") == "model"
    assert chat._match_switch("from the sources") == "sources"
    assert chat._match_switch("model off") == "sources"
    assert chat._match_switch("من المصادر") == "sources"


def test_match_switch_leaves_real_questions_alone() -> None:
    assert chat._match_switch("what is porosity?") is None
    assert chat._match_switch("how does the model predict the next state?") is None
    assert chat._match_switch("") is None


def test_switcher_swaps_the_mouth_and_keeps_the_grounded_one(tmp_path) -> None:
    mode, switcher = _switcher_mode(tmp_path)
    grounded = switcher.grounded
    grounded_help = switcher.grounded_help

    switcher.use_model()
    assert mode.assistant is switcher.mind
    assert mode.voice.assistant is switcher.mind
    assert mode.help_text != grounded_help
    assert chat._match_switch("from the model")

    switcher.use_sources()
    assert mode.assistant is grounded
    assert mode.voice.assistant is grounded
    assert mode.help_text == grounded_help


def test_model_answer_has_no_sources_and_carries_the_header(tmp_path) -> None:
    mode, switcher = _switcher_mode(tmp_path, continuation="the model's own words")
    switcher.use_model()
    turn = mode.handle("what is porosity?", require_wake=False, speak=False)

    assert "the model's own words" in turn.response
    assert "not grounded in cited sources" in turn.response
    assert turn.citations == ()
    assert "Sources:" not in chat.format_turn(turn)


def test_switcher_loads_the_model_lazily(tmp_path) -> None:
    mode = _mode(tmp_path)
    calls: list[int] = []

    def _loader():
        calls.append(1)
        return _FakeMindModel("late")

    switcher = chat.ModelSwitcher(mode, loader=_loader)
    assert calls == []
    switcher.use_model()
    assert calls == [1]
    switcher.use_model()
    assert calls == [1]  # loaded once, reused


def test_run_loop_switches_mouths_between_turns(tmp_path, monkeypatch, capsys) -> None:
    mode, switcher = _switcher_mode(tmp_path, continuation="model words")
    answers = iter(["from the model", "what is porosity?", "from the sources", "what is porosity?", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert chat.run_loop(mode, as_json=False, switcher=switcher) == 0
    out = capsys.readouterr().out
    assert "[model] Now answering from the trained model" in out
    assert "model words" in out
    assert "[sources] Now answering from retrieved passages" in out
    assert "Sources:" in out


def test_run_loop_switch_load_failure_is_reported_not_fatal(tmp_path, monkeypatch, capsys) -> None:
    mode = _mode(tmp_path)

    def _bad_loader():
        raise FileNotFoundError("no model here")

    switcher = chat.ModelSwitcher(mode, loader=_bad_loader)
    answers = iter(["from the model", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert chat.run_loop(mode, as_json=False, switcher=switcher) == 0
    assert "Could not load the trained model" in capsys.readouterr().out


def test_run_loop_hints_the_model_when_the_sources_fail(tmp_path, monkeypatch, capsys) -> None:
    """In sources mode, an unanswerable question points the user at the model."""
    mode, switcher = _switcher_mode(tmp_path)
    mode.handle = lambda *a, **k: _turn("There is not enough evidence on file to answer this question about x.")
    answers = iter(["what is x?", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    chat.run_loop(mode, as_json=False, switcher=switcher)
    out = capsys.readouterr().out
    assert "not enough evidence" in out
    assert "from the model" in out


def test_run_loop_hint_is_silent_in_model_mode(tmp_path, monkeypatch, capsys) -> None:
    mode, switcher = _switcher_mode(tmp_path)
    switcher.use_model()
    mode.handle = lambda *a, **k: _turn("There is not enough evidence on file to answer this question about x.")
    answers = iter(["what is x?", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    chat.run_loop(mode, as_json=False, switcher=switcher)
    assert "sources could not answer" not in capsys.readouterr().out


def _turn(response: str):
    from core.jarvis import JarvisTurn

    return JarvisTurn(heard="", response=response, woke=True, routed="answer")


def test_main_without_mind_does_not_touch_the_model(tmp_path, monkeypatch, capsys) -> None:
    """The default chat must not load a model: it stays the grounded assistant."""
    loaded: list[int] = []

    def _explode(*_args, **_kwargs):
        loaded.append(1)
        raise AssertionError("the model must not load without --mind")

    monkeypatch.setattr(chat.mind_module, "load_mind_model", _explode)
    monkeypatch.setattr(chat, "build_session", lambda **_kw: _mode(tmp_path))
    assert chat.main(["what is porosity?"]) == 0
    assert loaded == []


def test_main_mind_missing_checkpoint_exits_cleanly(tmp_path, monkeypatch, capsys) -> None:
    def _missing(*_args, **_kwargs):
        raise FileNotFoundError("No local language model at /nowhere.npz.")

    monkeypatch.setattr(chat.mind_module, "load_mind_model", _missing)
    monkeypatch.setattr(chat, "build_session", lambda **_kw: _mode(tmp_path))
    assert chat.main(["--mind", "hello"]) == 1
    out = capsys.readouterr().out
    assert "Could not load the trained model" in out
    assert "python -m cli llm" in out


def test_main_mind_answers_from_the_model(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(chat.mind_module, "load_mind_model", lambda *_a, **_k: _FakeMindModel("mind output"))
    monkeypatch.setattr(chat, "build_session", lambda **_kw: _mode(tmp_path))
    assert chat.main(["--mind", "what is porosity?"]) == 0
    out = capsys.readouterr().out
    assert "mind output" in out
    assert "not grounded in cited sources" in out


# --- Jarvis is untouched --------------------------------------------------


def test_chat_module_imports_jarvis_but_not_the_other_way(tmp_path) -> None:
    """The dependency is one-way: chat -> jarvis, never jarvis -> chat."""
    import ai.jarvis as jarvis_module

    assert not hasattr(jarvis_module, "mind")
    assert not hasattr(jarvis_module, "ModelSwitcher")


def test_jarvis_build_session_signature_has_no_mind_parameters() -> None:
    import inspect

    from ai.jarvis import build_session

    parameters = inspect.signature(build_session).parameters
    assert "mind" not in parameters
    assert "model" not in parameters
    assert "answer_from_model" not in parameters

