"""Ultron: the general local language model, its CLI, menu, and chat/GUI wiring.

Everything heavy is injected, so these tests need neither ``llama-cpp-python``
nor a multi-gigabyte model file: the engine is exercised through a fake runner,
and the CLI/menu/chat/GUI wiring is checked by import and dispatch.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ai import ultron as ultron_cli
from ai import ultron_mouth
from ai.chat import MOUTH_MODEL, MOUTH_SOURCES, MOUTH_ULTRON
from cli import menu as cli_menu
from core import optional_deps
from core.conversation import Conversation
from core.optional_deps import FEATURE_ULTRON
from core.ultron import (
    CATALOG,
    DEFAULT_SPEC_KEY,
    HEADER,
    NAME,
    UltronEngine,
    UltronUnavailable,
    download_model,
    find_model_file,
    model_is_cached,
    resolve_spec,
    spec_names,
    spec_summaries,
)
from gui import actions as gui_actions


class FakeRunner:
    """A stand-in for the llama-cpp runner: records calls, echoes a reply."""

    def __init__(self, reply: str = "Hello from Ultron.") -> None:
        self.reply_text = reply
        self.calls: list[tuple[list[dict[str, str]], dict]] = []

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        self.calls.append((messages, kwargs))
        return self.reply_text


class StreamingRunner(FakeRunner):
    """A runner that streams its reply in pieces."""

    def stream(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        yield "Hello "
        yield "from "
        yield "Ultron."


# --- engine ----------------------------------------------------------------


def test_engine_builds_the_system_then_history_then_prompt() -> None:
    engine = UltronEngine(runner=FakeRunner())
    messages = engine.build_messages([("user", "hi"), ("assistant", "hello")], "how are you?")
    assert messages[0]["role"] == "system"
    assert "every field of life" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "hi"}
    assert messages[2] == {"role": "assistant", "content": "hello"}
    assert messages[-1] == {"role": "user", "content": "how are you?"}


def test_engine_reply_returns_the_runner_text() -> None:
    runner = FakeRunner("The answer is 42.")
    engine = UltronEngine(runner=runner)
    assert engine.reply("meaning of life?") == "The answer is 42."
    messages, kwargs = runner.calls[0]
    assert messages[-1]["content"] == "meaning of life?"
    assert "temperature" in kwargs and "max_tokens" in kwargs


def test_engine_rejects_an_empty_prompt() -> None:
    engine = UltronEngine(runner=FakeRunner())
    with pytest.raises(ValueError):
        engine.reply("   ")


def test_engine_streams_when_the_runner_can() -> None:
    engine = UltronEngine(runner=StreamingRunner())
    assert "".join(engine.stream_reply("hi")) == "Hello from Ultron."


def test_engine_stream_falls_back_to_one_piece() -> None:
    engine = UltronEngine(runner=FakeRunner("one shot"))
    assert list(engine.stream_reply("hi")) == ["one shot"]


def test_engine_validates_its_settings() -> None:
    with pytest.raises(ValueError):
        UltronEngine(runner=FakeRunner(), context=0)
    with pytest.raises(ValueError):
        UltronEngine(runner=FakeRunner(), max_tokens=0)
    with pytest.raises(ValueError):
        UltronEngine(runner=FakeRunner(), temperature=5.0)
    with pytest.raises(ValueError):
        UltronEngine(runner=FakeRunner(), top_p=0.0)


def test_injected_runner_is_always_available() -> None:
    assert UltronEngine(runner=FakeRunner()).is_available() is True


def test_engine_without_a_model_is_unavailable_and_explains(tmp_path: Path) -> None:
    engine = UltronEngine(root=tmp_path)
    assert engine.is_available() is False
    with pytest.raises(UltronUnavailable):
        engine.reply("hi")


# --- model discovery in the data folder ------------------------------------


def test_find_model_file_prefers_the_registry(tmp_path: Path) -> None:
    (tmp_path / "small.gguf").write_bytes(b"x" * 10)
    (tmp_path / "big.gguf").write_bytes(b"x" * 100)
    # Without a registry the largest file wins.
    chosen = find_model_file(tmp_path)
    assert chosen is not None and chosen.name == "big.gguf"
    # A registry entry overrides the size heuristic.
    (tmp_path / "model.json").write_text(json.dumps({"file": "small.gguf"}), encoding="utf-8")
    chosen = find_model_file(tmp_path)
    assert chosen is not None and chosen.name == "small.gguf"


def test_find_model_file_is_none_when_empty(tmp_path: Path) -> None:
    assert find_model_file(tmp_path) is None
    assert model_is_cached(tmp_path) is False


def test_find_model_file_ignores_a_stale_registry(tmp_path: Path) -> None:
    (tmp_path / "present.gguf").write_bytes(b"x")
    (tmp_path / "model.json").write_text(json.dumps({"file": "missing.gguf"}), encoding="utf-8")
    chosen = find_model_file(tmp_path)
    assert chosen is not None and chosen.name == "present.gguf"


def test_engine_uses_a_model_file_it_finds(tmp_path: Path) -> None:
    (tmp_path / "any.gguf").write_bytes(b"x")
    engine = UltronEngine(root=tmp_path)
    assert engine.model_path is not None
    assert engine.model_path.name == "any.gguf"


# --- the catalog -----------------------------------------------------------


def test_catalog_has_a_default_and_named_specs() -> None:
    assert DEFAULT_SPEC_KEY in CATALOG
    assert spec_names() == tuple(CATALOG)
    for key, title, size, summary in spec_summaries():
        assert key in CATALOG and title and summary and size > 0


def test_resolve_spec_defaults_and_rejects_unknown() -> None:
    assert resolve_spec(None).key == DEFAULT_SPEC_KEY
    assert resolve_spec("qwen2.5-3b").params == "3B"
    with pytest.raises(ValueError):
        resolve_spec("not-a-model")


# --- download gate ---------------------------------------------------------


def test_download_needs_approval_by_default(tmp_path: Path) -> None:
    with pytest.raises(UltronUnavailable):
        download_model("qwen2.5-3b", root=tmp_path)


def test_download_returns_a_cached_file_without_the_network(tmp_path: Path) -> None:
    # A file that is already there is returned before any hub import happens.
    spec = resolve_spec("qwen2.5-3b")
    (tmp_path / spec.filename).write_bytes(b"weights")
    path = download_model("qwen2.5-3b", root=tmp_path, approved=True)
    assert path.name == spec.filename


# --- the optional extra ----------------------------------------------------


def test_ultron_feature_is_registered() -> None:
    spec = optional_deps.feature(FEATURE_ULTRON)
    assert spec.requirements_file == "requirements-ultron.txt"
    assert "llama_cpp" in spec.import_names
    # ``status`` must never raise even with the extra and the model absent.
    assert optional_deps.status(FEATURE_ULTRON).key == FEATURE_ULTRON


def test_ultron_pip_hint_points_at_the_requirements_file() -> None:
    assert "requirements-ultron.txt" in optional_deps.pip_install_command(FEATURE_ULTRON)


# --- the CLI ---------------------------------------------------------------


def test_status_line_is_honest_without_a_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    line = ultron_cli.status_line(UltronEngine(root=tmp_path))
    assert NAME in line


def test_cli_models_lists_the_catalog(capsys) -> None:
    assert ultron_cli.main(["models"]) == 0
    out = capsys.readouterr().out
    assert DEFAULT_SPEC_KEY in out
    for key in spec_names():
        assert key in out


def test_cli_models_json(capsys) -> None:
    assert ultron_cli.main(["models", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert {entry["key"] for entry in payload} == set(spec_names())


def test_cli_download_without_approve_is_refused(capsys) -> None:
    assert ultron_cli.main(["download"]) == 1
    assert "--approve" in capsys.readouterr().out


def test_cli_status_reports_plainly(capsys) -> None:
    assert ultron_cli.main(["status"]) in (0, 1)
    assert NAME in capsys.readouterr().out


def test_cli_bare_question_is_normalized_to_chat() -> None:
    assert ultron_cli._normalize_argv(["what", "is", "rust?"]) == [
        "chat",
        "what",
        "is",
        "rust?",
    ]
    assert ultron_cli._normalize_argv(["download", "--approve"])[0] == "download"
    assert ultron_cli._normalize_argv(["--help"])[0] == "--help"


def test_cli_one_shot_uses_an_injected_runner(tmp_path, monkeypatch, capsys) -> None:
    """A one-shot question is answered through the engine, no model file needed."""
    engine = UltronEngine(runner=FakeRunner("Paris."))

    def fake_engine(_args):
        return engine

    monkeypatch.setattr(ultron_cli, "_engine", fake_engine)
    code = ultron_cli.main(["who", "built", "the", "Eiffel", "Tower?"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Paris." in out
    assert HEADER in out


def test_cli_one_shot_in_a_named_session_persists(tmp_path, monkeypatch, capsys) -> None:
    """A `--session` one-shot turn is saved, so the next run keeps context."""
    from core.conversation import ConversationMemory
    from storage.json_store import JsonStore

    engine = UltronEngine(runner=FakeRunner("Paris."))
    memory = ConversationMemory(JsonStore(tmp_path))
    monkeypatch.setattr(ultron_cli, "_engine", lambda _args: engine)
    monkeypatch.setattr(ultron_cli, "_memory", lambda: memory)

    code = ultron_cli.main(["--session", "trip", "Where", "is", "the", "Eiffel", "Tower?"])
    capsys.readouterr()
    assert code == 0

    reloaded = ConversationMemory(JsonStore(tmp_path)).load("trip")
    assert reloaded is not None
    assert reloaded.id == "trip"
    assert [message.text for message in reloaded.messages][-2:] == [
        "Where is the Eiffel Tower?",
        "Paris.",
    ]


def test_cli_interactive_quits_immediately(monkeypatch, capsys) -> None:
    engine = UltronEngine(runner=FakeRunner())
    monkeypatch.setattr(ultron_cli, "_engine", lambda _args: engine)
    monkeypatch.setattr("builtins.input", lambda _prompt: "quit")
    assert ultron_cli.main([]) == 0


# --- menu and CLI dispatch -------------------------------------------------


def test_menu_shows_ultron_directly_below_jarvis() -> None:
    labels = [title for title, _action, _argv in cli_menu.visible_menu()]
    jarvis = next(i for i, label in enumerate(labels) if label.startswith("Jarvis mode"))
    assert labels[jarvis + 1] == "Ultron (general local model, any subject)"


def test_menu_action_runs_the_ultron_module(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_module(name: str, argv) -> int:
        captured["name"] = name
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(cli_menu, "_run_module", fake_run_module)
    assert cli_menu.run_ultron(["--session", "s"]) == 0
    assert captured["name"] == "ai.ultron"
    assert captured["argv"] == ["--session", "s"]


def test_cli_dispatches_ultron() -> None:
    from cli import __main__ as cli_entry

    assert "ultron" in cli_entry._COMMANDS


def test_gui_sidebar_has_an_ultron_action() -> None:
    keys = {action.key for action in gui_actions.ACTIONS}
    assert "ultron" in keys
    action = gui_actions.action_by_key("ultron")
    # The sidebar entry is the *chat mouth*, so it runs the chat with --ultron
    # rather than a separate Ultron program.
    assert action.command is cli_menu.run_ultron_chat


def test_gui_ultron_action_selects_the_mouth_in_the_same_chat() -> None:
    """Ultron answers in the project's own chat, so the sidebar entry switches the mouth."""
    labels = [action.label for action in gui_actions.ACTIONS]
    assert "Answer from Ultron (general local model)" in labels


def test_menu_offers_the_ultron_chat_mouth() -> None:
    """Ultron is reachable as a *chat mouth* too, not only as the separate program."""
    labels = [title for title, _action, _argv in cli_menu.visible_menu()]
    assert "Ultron (general local model, any subject)" in labels
    assert cli_menu.run_ultron_chat.__module__ == "cli.menu"


def test_run_ultron_chat_adds_the_flag_once(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_module(name: str, argv) -> int:
        captured["name"] = name
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(cli_menu, "_run_module", fake_run_module)
    assert cli_menu.run_ultron_chat([]) == 0
    assert captured["name"] == "ai.chat"
    assert captured["argv"] == ["--ultron"]
    # A caller that already passed the flag is left exactly as it was.
    cli_menu.run_ultron_chat(["--ultron", "--json"])
    assert captured["argv"] == ["--ultron", "--json"]


# --- the Ultron mouth inside the chat --------------------------------------


class FakeSession:
    """A stand-in for ``JarvisMode`` that the chat's switcher can swap a mouth into."""

    def __init__(self) -> None:
        self.assistant: Any = SimpleNamespace(name="grounded")
        self.help_text = "help"
        self.voice = SimpleNamespace(assistant=self.assistant)


def _ultron_assistant(reply: str = "Ultron says hello.") -> ultron_mouth.UltronAssistant:
    return ultron_mouth.UltronAssistant(UltronEngine(runner=FakeRunner(reply)))


def _switcher(reply: str = "Ultron says hello."):
    from ai.chat import ModelSwitcher

    session = FakeSession()
    switcher = ModelSwitcher(
        cast(Any, session),
        loader=lambda: (_ for _ in ()).throw(AssertionError("the trained model must not load")),
        ultron_loader=lambda: _ultron_assistant(reply),
    )
    return session, switcher


def test_switcher_starts_on_the_sources() -> None:
    _session, switcher = _switcher()
    assert switcher.mouth == MOUTH_SOURCES
    assert switcher.active is False


def test_switcher_ultron_swaps_the_mouth_and_keeps_the_grounded_one() -> None:
    session, switcher = _switcher()
    grounded = switcher.grounded
    grounded_help = switcher.grounded_help

    switcher.use_ultron()

    assert switcher.mouth == MOUTH_ULTRON
    assert switcher.active is True
    assert session.assistant is switcher.ultron
    assert session.voice.assistant is switcher.ultron
    assert session.help_text != grounded_help
    assert ultron_mouth.ULTRON_HELP in session.help_text

    switcher.use_sources()
    assert switcher.mouth == MOUTH_SOURCES
    assert session.assistant is grounded
    assert session.help_text == grounded_help


def test_switcher_loads_ultron_lazily_and_reuses_it() -> None:
    from ai.chat import ModelSwitcher

    session = FakeSession()
    calls: list[int] = []

    def _loader():
        calls.append(1)
        return _ultron_assistant()

    switcher = ModelSwitcher(
        cast(Any, session),
        loader=lambda: None,
        ultron_loader=_loader,
    )
    assert calls == []
    switcher.use_ultron()
    switcher.use_ultron()
    assert calls == [1]


def test_switch_recognises_the_ultron_phrases() -> None:
    from ai import chat

    assert chat._match_switch("from ultron") == MOUTH_ULTRON
    assert chat._match_switch("Ultron on.") == MOUTH_ULTRON
    assert chat._match_switch("من ألترون") == MOUTH_ULTRON
    assert chat._match_switch("from the model") == MOUTH_MODEL
    assert chat._match_switch("from the sources") == MOUTH_SOURCES
    # A real question is never captured as a switch.
    assert chat._match_switch("how does ultron work?") is None


def test_ultron_assistant_records_the_turn_and_carries_no_citations() -> None:
    assistant = _ultron_assistant("The answer is 42.")
    conversation = Conversation(title="t")
    answer = assistant.ask("meaning of life?", conversation=conversation)

    assert "The answer is 42." in answer.text
    assert HEADER in answer.text
    assert answer.citations == ()
    assert answer.used_model is True
    # Both the question and the answer are in the same session, so a follow-up
    # keeps context and the chat's own transcript shows them.
    assert [message.text for message in conversation.messages][0] == "meaning of life?"


def test_ultron_assistant_passes_prior_turns_to_the_model() -> None:
    runner = FakeRunner("Paris.")
    assistant = ultron_mouth.UltronAssistant(UltronEngine(runner=runner))
    conversation = Conversation(title="t")
    conversation.add_user("What is the capital of France?")
    conversation.add_assistant(f"{HEADER}\n\nParis.")

    assistant.ask("and its river?", conversation=conversation)

    messages, _kwargs = runner.calls[-1]
    # The system prompt, the prior exchange (with the honesty header stripped),
    # then the new question.
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "What is the capital of France?"}
    assert messages[2] == {"role": "assistant", "content": "Paris."}
    assert messages[-1] == {"role": "user", "content": "and its river?"}


def test_ultron_assistant_reports_an_empty_reply_honestly() -> None:
    assistant = _ultron_assistant("")
    answer = assistant.ask("anything?", conversation=Conversation(title="t"))
    assert ultron_mouth.ULTRON_EMPTY in answer.text


def test_ultron_assistant_reads_an_attached_file_from_the_conversation() -> None:
    runner = FakeRunner("Zephyrium-9 flowed at 4200 barrels per day.")
    assistant = ultron_mouth.UltronAssistant(UltronEngine(runner=runner))
    conversation = Conversation(title="t")
    conversation.add_context(
        "Attached file: well.docx\n\nZephyrium-9 flowed at 4200 barrels per day."
    )

    assistant.ask("What did Zephyrium-9 flow at?", conversation=conversation)

    messages, _kwargs = runner.calls[-1]
    prompt = messages[-1]["content"]
    # The attached file's text reaches the model, in front of the question, so
    # Ultron can read it without any retriever of its own.
    assert ultron_mouth.REFERENCE_LABEL in prompt
    assert "Zephyrium-9" in prompt
    assert prompt.rstrip().endswith("What did Zephyrium-9 flow at?")


def test_ultron_assistant_sends_the_bare_question_without_an_attachment() -> None:
    runner = FakeRunner()
    assistant = ultron_mouth.UltronAssistant(UltronEngine(runner=runner))

    assistant.ask("hello", conversation=Conversation(title="t"))

    messages, _kwargs = runner.calls[-1]
    assert messages[-1] == {"role": "user", "content": "hello"}


def test_ultron_assistant_marks_a_cut_attachment() -> None:
    runner = FakeRunner()
    assistant = ultron_mouth.UltronAssistant(UltronEngine(runner=runner), reference_chars=20)
    conversation = Conversation(title="t")
    conversation.add_context("Attached file: long.docx\n\n" + "word " * 50)

    assistant.ask("summarise it", conversation=conversation)

    prompt = runner.calls[-1][0][-1]["content"]
    assert "not shown here" in prompt


def test_load_ultron_assistant_raises_when_unavailable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    with pytest.raises(UltronUnavailable):
        ultron_mouth.load_ultron_assistant()


def test_chat_main_reports_a_missing_ultron_honestly(tmp_path: Path, monkeypatch, capsys) -> None:
    """``chat --ultron`` without the extra/model exits non-zero with the real next step."""
    from ai import chat

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert chat.main(["--ultron", "--no-web"]) == 1
    out = capsys.readouterr().out
    assert "Could not load Ultron" in out
    assert "requirements-ultron.txt" in out


def test_chat_accepts_the_ultron_options(tmp_path: Path, monkeypatch, capsys) -> None:
    """The Ultron flags are real; ``--list-sessions`` returns before a mouth loads."""
    from ai import chat

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    code = chat.main(
        [
            "--list-sessions",
            "--ultron",
            "--ultron-model",
            "x.gguf",
            "--ultron-tokens",
            "64",
            "--ultron-temperature",
            "0.2",
        ]
    )
    assert code == 0
    assert "sessions" in capsys.readouterr().out.lower()


def test_run_loop_switches_to_ultron_and_back(monkeypatch, capsys) -> None:
    from ai import chat
    from core.jarvis import JarvisTurn

    session, switcher = _switcher()

    class _Mode:
        def handle(self, *a, **k):
            return JarvisTurn(
                heard="hi", response="Ultron says hello.", woke=True, routed="answer"
            )

    answers = iter(["from ultron", "hi", "from the sources", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert chat.run_loop(_Mode(), as_json=False, switcher=switcher) == 0
    out = capsys.readouterr().out
    assert "[ultron] Now answering from the general local model" in out
    assert "[sources] Now answering from retrieved passages" in out


def test_run_loop_reports_a_missing_ultron_without_switching(monkeypatch, capsys) -> None:
    from ai import chat
    from core.jarvis import JarvisTurn

    session = FakeSession()
    switcher = chat.ModelSwitcher(
        cast(Any, session),
        loader=lambda: None,
        ultron_loader=lambda: (_ for _ in ()).throw(UltronUnavailable("no model file")),
    )

    class _Mode:
        def handle(self, *a, **k):
            return JarvisTurn(heard="hi", response="ok", woke=True, routed="answer")

    answers = iter(["from ultron", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert chat.run_loop(_Mode(), as_json=False, switcher=switcher) == 0
    out = capsys.readouterr().out
    assert "Could not load Ultron" in out
    assert "requirements-ultron.txt" in out
    # The failed switch left the grounded mouth in place.
    assert switcher.mouth == MOUTH_SOURCES


# --- the window's model selector (needs a display) -------------------------


def _display_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


def _gui_service() -> Any:
    from gui.service import ChatService

    return ChatService(
        mode=cast(Any, FakeSession()),
        data_root=Path.cwd(),
        ultron_model=None,
    )


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_model_selector_offers_ultron(tmp_path) -> None:
    import tkinter as tk

    from gui.app import MOUTH_CHOICES, MOUTH_LABELS, FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_gui_service())
        root.update()
        # The header's selector lists all three mouths, Ultron included.
        labels = [
            app._source_menu["menu"].entrycget(i, "label")
            for i in range(app._source_menu["menu"].index("end") + 1)
        ]
        assert MOUTH_LABELS[MOUTH_ULTRON] in labels
        assert [label for _key, label in MOUTH_CHOICES] == labels
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_switches_to_ultron_in_the_same_chat(tmp_path) -> None:
    import tkinter as tk

    from gui.app import MOUTH_LABELS, FaydGUI

    root = tk.Tk()
    try:
        service = _gui_service()
        service.switcher._ultron_loader = lambda: _ultron_assistant("A windowed reply.")
        app = FaydGUI(root, service=service)
        root.update()
        app._add_bubble("assistant", "keep me across the mouth switch")

        app._on_source_menu(MOUTH_LABELS[MOUTH_ULTRON])
        root.update()

        # The same chat now answers from Ultron: no new window, transcript intact.
        assert service.source == MOUTH_ULTRON
        assert app._source_choice_label() == MOUTH_LABELS[MOUTH_ULTRON]
        texts = [entry.text for entry in app._rows]
        assert any("keep me across the mouth switch" in text for text in texts)
        assert any("Ultron" in text for text in texts)

        # And back to the sources.
        app._on_source_menu(MOUTH_LABELS[MOUTH_SOURCES])
        root.update()
        assert service.source == MOUTH_SOURCES
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_reports_a_missing_ultron_in_the_transcript(tmp_path) -> None:
    import tkinter as tk

    from gui.app import MOUTH_LABELS, FaydGUI

    root = tk.Tk()
    try:
        service = _gui_service()
        service.switcher._ultron_loader = lambda: (_ for _ in ()).throw(
            UltronUnavailable("no model file")
        )
        app = FaydGUI(root, service=service)
        root.update()

        app._on_source_menu(MOUTH_LABELS[MOUTH_ULTRON])
        root.update()

        # The mouth did not change, and the real next step is shown.
        assert service.source == MOUTH_SOURCES
        assert app._source_choice_label() == MOUTH_LABELS[MOUTH_SOURCES]
        texts = [entry.text for entry in app._rows]
        assert any("Could not load Ultron" in text for text in texts)
        assert any("requirements-ultron.txt" in text for text in texts)
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_ultron_sidebar_action_switches_the_mouth(tmp_path) -> None:
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.actions import action_by_key

    root = tk.Tk()
    try:
        service = _gui_service()
        service.switcher._ultron_loader = lambda: _ultron_assistant()
        app = FaydGUI(root, service=service)
        root.update()

        app._on_action(action_by_key("ultron"))
        root.update()

        assert service.source == MOUTH_ULTRON
        # No separate window was opened.
        assert getattr(app, "_ultron_dialog", None) is None
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_switches_mouth_before_the_settings_panel_is_opened(tmp_path) -> None:
    """The header selector works on a fresh window, before Settings is expanded."""
    import tkinter as tk

    from gui.app import MOUTH_LABELS, FaydGUI

    root = tk.Tk()
    try:
        service = _gui_service()
        service.switcher._ultron_loader = lambda: _ultron_assistant()
        app = FaydGUI(root, service=service)
        root.update()
        assert app._settings_open is False

        app._on_source_menu(MOUTH_LABELS[MOUTH_ULTRON])
        root.update()
        assert service.source == MOUTH_ULTRON

        # Opening Settings afterwards keeps the header's choice; the panel no longer
        # shows a mouth radio, so the variable is what stays in step.
        app._toggle_settings()
        root.update()
        assert app._source_var.get() == MOUTH_ULTRON
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_hero_names_all_three_mouths(tmp_path) -> None:
    """The welcome hint must not forget Ultron now that it is a mouth here too."""
    import tkinter as tk

    from gui.app import FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_gui_service())
        root.update()

        def walk(widget, out):
            for child in widget.winfo_children():
                out.append(child)
                walk(child, out)

        widgets: list = []
        walk(app._empty, widgets)
        hints = [
            w.cget("text")
            for w in widgets
            if isinstance(w, tk.Label) and "Answering mouth" in str(w.cget("text"))
        ]
        assert hints, "the hero must explain the answering mouths"
        assert "Ultron" in hints[0]
    finally:
        root.destroy()
