"""Vision: the petroleum local language model, its CLI, menu, and chat/GUI wiring.

Everything heavy is injected, so these tests need neither ``llama-cpp-python``
nor a multi-gigabyte model file: the engine is exercised through a fake runner,
and the CLI/menu/chat/GUI wiring is checked by import and dispatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ai import vision_model as vision_cli
from ai import vision_mouth
from ai.chat import MOUTH_MODEL, MOUTH_SOURCES, MOUTH_ULTRON, MOUTH_VISION
from cli import menu as cli_menu
from core import optional_deps
from core.conversation import Conversation
from core.optional_deps import FEATURE_VISION
from core.vision_model import (
    CATALOG,
    DEFAULT_SPEC_KEY,
    HEADER,
    NAME,
    VisionEngine,
    VisionUnavailable,
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

    def __init__(self, reply: str = "Hello from Vision.") -> None:
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
        yield "Vision."


# --- engine ----------------------------------------------------------------


def test_engine_builds_the_system_then_history_then_prompt() -> None:
    engine = VisionEngine(runner=FakeRunner())
    messages = engine.build_messages([("user", "hi"), ("assistant", "hello")], "how are you?")
    assert messages[0]["role"] == "system"
    assert "petroleum" in messages[0]["content"].lower()
    assert messages[1] == {"role": "user", "content": "hi"}
    assert messages[2] == {"role": "assistant", "content": "hello"}
    assert messages[-1] == {"role": "user", "content": "how are you?"}


def test_engine_reply_returns_the_runner_text() -> None:
    runner = FakeRunner("The material balance equation is ...")
    engine = VisionEngine(runner=runner)
    assert engine.reply("what is the MBE?") == "The material balance equation is ..."
    _messages, kwargs = runner.calls[0]
    assert kwargs["max_tokens"] == engine.max_tokens
    assert kwargs["temperature"] == engine.temperature


def test_engine_rejects_an_empty_prompt() -> None:
    engine = VisionEngine(runner=FakeRunner())
    with pytest.raises(ValueError):
        engine.reply("   ")


def test_engine_streams_when_the_runner_can() -> None:
    engine = VisionEngine(runner=StreamingRunner())
    assert "".join(engine.stream_reply("hi")) == "Hello from Vision."


def test_engine_stream_falls_back_to_one_piece() -> None:
    engine = VisionEngine(runner=FakeRunner("whole"))
    assert list(engine.stream_reply("hi")) == ["whole"]


def test_engine_validates_its_settings() -> None:
    with pytest.raises(ValueError):
        VisionEngine(context=0)
    with pytest.raises(ValueError):
        VisionEngine(max_tokens=0)
    with pytest.raises(ValueError):
        VisionEngine(temperature=3.0)
    with pytest.raises(ValueError):
        VisionEngine(top_p=0.0)


def test_injected_runner_is_always_available() -> None:
    assert VisionEngine(runner=FakeRunner()).is_available() is True


def test_engine_without_a_model_is_unavailable_and_explains(tmp_path: Path) -> None:
    engine = VisionEngine(root=tmp_path)
    assert engine.is_available() is False
    with pytest.raises(VisionUnavailable):
        engine.reply("anything")
    assert "no local model file yet" in engine.describe().lower()


def test_find_model_file_prefers_the_registry(tmp_path: Path) -> None:
    small = tmp_path / "small.gguf"
    small.write_bytes(b"x")
    big = tmp_path / "big.gguf"
    big.write_bytes(b"x" * 100)
    (tmp_path / "model.json").write_text(json.dumps({"file": "small.gguf"}), encoding="utf-8")
    assert find_model_file(tmp_path) == small


def test_find_model_file_is_none_when_empty(tmp_path: Path) -> None:
    assert find_model_file(tmp_path) is None
    assert model_is_cached(tmp_path) is False


def test_find_model_file_ignores_a_stale_registry(tmp_path: Path) -> None:
    present = tmp_path / "present.gguf"
    present.write_bytes(b"x")
    (tmp_path / "model.json").write_text(json.dumps({"file": "gone.gguf"}), encoding="utf-8")
    assert find_model_file(tmp_path) == present


def test_engine_uses_a_model_file_it_finds(tmp_path: Path) -> None:
    (tmp_path / "vision.gguf").write_bytes(b"x")
    engine = VisionEngine(root=tmp_path)
    assert engine.model_path == tmp_path / "vision.gguf"


# --- the download catalog --------------------------------------------------


def test_catalog_has_a_default_and_named_specs() -> None:
    assert DEFAULT_SPEC_KEY in CATALOG
    assert resolve_spec(None) is CATALOG[DEFAULT_SPEC_KEY]
    assert "ogai-reasoner" in spec_names()
    for _key, title, size, summary in spec_summaries():
        assert title and summary
        assert size > 0


def test_resolve_spec_defaults_and_rejects_unknown() -> None:
    assert resolve_spec("OGAI-REASONER").key == "ogai-reasoner"
    with pytest.raises(ValueError):
        resolve_spec("nope")


def test_download_needs_approval_by_default(tmp_path: Path) -> None:
    with pytest.raises(VisionUnavailable):
        download_model("ogai-reasoner", root=tmp_path)


def test_download_returns_a_cached_file_without_the_network(tmp_path: Path) -> None:
    # A file that is already there is returned before any hub import happens.
    spec = resolve_spec("ogai-reasoner")
    cached = tmp_path / spec.filename
    cached.write_bytes(b"x")
    assert download_model("ogai-reasoner", root=tmp_path, approved=True) == cached


def test_vision_feature_is_registered() -> None:
    assert FEATURE_VISION in optional_deps.FEATURES
    spec = optional_deps.feature(FEATURE_VISION)
    assert spec.requirements_file == "requirements-vision.txt"


def test_vision_pip_hint_points_at_the_requirements_file() -> None:
    hint = optional_deps.pip_install_command(FEATURE_VISION)
    assert "requirements-vision.txt" in hint


def test_status_line_is_honest_without_a_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    line = vision_cli.status_line()
    assert NAME in line
    assert "no local model" in line.lower()


# --- the CLI ---------------------------------------------------------------


def test_cli_models_lists_the_catalog(capsys) -> None:
    assert vision_cli.run_models(cast(argparse.Namespace, SimpleNamespace(as_json=False))) == 0
    out = capsys.readouterr().out
    assert "ogai-reasoner" in out


def test_cli_models_json(capsys) -> None:
    assert vision_cli.run_models(cast(argparse.Namespace, SimpleNamespace(as_json=True))) == 0
    data = json.loads(capsys.readouterr().out)
    assert any(entry["key"] == "ogai-reasoner" for entry in data)


def test_cli_download_without_approve_is_refused(capsys) -> None:
    args = cast(argparse.Namespace, SimpleNamespace(approve=False, model=None, as_json=False))
    assert vision_cli.run_download(args) == 1
    assert "needs --approve" in capsys.readouterr().out


def test_cli_status_reports_plainly(capsys) -> None:
    assert vision_cli.run_status(cast(argparse.Namespace, SimpleNamespace(as_json=False))) == 0
    assert NAME in capsys.readouterr().out


def test_cli_bare_question_is_normalized_to_chat() -> None:
    assert vision_cli._normalize_argv(["what is porosity?"])[0] == "chat"
    assert vision_cli._normalize_argv(["download"]) == ["download"]
    assert vision_cli._normalize_argv(["--help"]) == ["--help"]


def test_cli_one_shot_uses_an_injected_runner(tmp_path, monkeypatch, capsys) -> None:
    engine = VisionEngine(runner=FakeRunner("Porosity is the void fraction."))
    monkeypatch.setattr(vision_cli, "_engine", lambda _args: engine)
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert vision_cli.main(["what is porosity?"]) == 0
    out = capsys.readouterr().out
    assert "Porosity is the void fraction." in out
    assert HEADER in out


def test_cli_interactive_quits_immediately(monkeypatch, capsys) -> None:
    engine = VisionEngine(runner=FakeRunner())
    monkeypatch.setattr(vision_cli, "_engine", lambda _args: engine)
    monkeypatch.setattr("builtins.input", lambda _prompt: "quit")
    assert vision_cli.main([]) == 0


# --- menu and CLI dispatch -------------------------------------------------


def test_menu_shows_vision_directly_below_ultron() -> None:
    labels = [title for title, _action, _argv in cli_menu.visible_menu()]
    ultron = next(i for i, label in enumerate(labels) if label.startswith("Ultron"))
    assert labels[ultron + 1] == "Vision (petroleum local model: reservoir, drilling, PVT)"


def test_menu_action_runs_the_vision_module(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_module(name: str, argv) -> int:
        captured["name"] = name
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(cli_menu, "_run_module", fake_run_module)
    assert cli_menu.run_vision(["--session", "s"]) == 0
    assert captured["name"] == "ai.vision_model"
    assert captured["argv"] == ["--session", "s"]


def test_cli_dispatches_vision() -> None:
    from cli import __main__ as cli_entry

    assert "vision" in cli_entry._COMMANDS
    assert "vision-chat" in cli_entry._COMMANDS


def test_gui_sidebar_has_a_vision_action() -> None:
    keys = {action.key for action in gui_actions.ACTIONS}
    assert "vision" in keys
    action = gui_actions.action_by_key("vision")
    # The sidebar entry is the *chat mouth*, so it runs the chat with --vision
    # rather than a separate Vision program.
    assert action.command is cli_menu.run_vision_chat


def test_gui_vision_action_selects_the_mouth_in_the_same_chat() -> None:
    """Vision answers in the project's own chat, so the sidebar entry switches the mouth."""
    labels = [action.label for action in gui_actions.ACTIONS]
    assert "Answer from Vision (petroleum local model)" in labels


def test_menu_offers_the_vision_chat_mouth() -> None:
    """Vision is reachable as a *chat mouth* too, not only as the separate program."""
    assert cli_menu.run_vision_chat.__module__ == "cli.menu"


def test_run_vision_chat_adds_the_flag_once(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_module(name: str, argv) -> int:
        captured["name"] = name
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(cli_menu, "_run_module", fake_run_module)
    assert cli_menu.run_vision_chat([]) == 0
    assert captured["name"] == "ai.chat"
    assert captured["argv"] == ["--vision"]
    # A caller that already passed the flag is left exactly as it was.
    cli_menu.run_vision_chat(["--vision", "--json"])
    assert captured["argv"] == ["--vision", "--json"]


# --- the Vision mouth inside the chat --------------------------------------


class FakeSession:
    """A stand-in for ``JarvisMode`` that the chat's switcher can swap a mouth into."""

    def __init__(self) -> None:
        self.assistant: Any = SimpleNamespace(name="grounded")
        self.help_text = "help"
        self.voice = SimpleNamespace(assistant=self.assistant)


def _vision_assistant(reply: str = "Vision says hello.") -> vision_mouth.VisionAssistant:
    return vision_mouth.VisionAssistant(VisionEngine(runner=FakeRunner(reply)))


def _switcher(reply: str = "Vision says hello."):
    from ai.chat import ModelSwitcher

    session = FakeSession()
    switcher = ModelSwitcher(
        cast(Any, session),
        loader=lambda: (_ for _ in ()).throw(AssertionError("the trained model must not load")),
        vision_loader=lambda: _vision_assistant(reply),
    )
    return session, switcher


def test_switcher_vision_swaps_the_mouth_and_keeps_the_grounded_one() -> None:
    session, switcher = _switcher()
    grounded = switcher.grounded
    grounded_help = switcher.grounded_help

    switcher.use_vision()

    assert switcher.mouth == MOUTH_VISION
    assert switcher.active is True
    assert session.assistant is switcher.vision
    assert session.voice.assistant is switcher.vision
    assert session.help_text != grounded_help
    assert vision_mouth.VISION_HELP in session.help_text

    switcher.use_sources()
    assert switcher.mouth == MOUTH_SOURCES
    assert session.assistant is grounded
    assert session.help_text == grounded_help


def test_switcher_loads_vision_lazily_and_reuses_it() -> None:
    from ai.chat import ModelSwitcher

    session = FakeSession()
    calls: list[int] = []

    def _loader():
        calls.append(1)
        return _vision_assistant()

    switcher = ModelSwitcher(
        cast(Any, session),
        loader=lambda: None,
        vision_loader=_loader,
    )
    assert calls == []
    switcher.use_vision()
    switcher.use_vision()
    assert calls == [1]


def test_switch_recognises_the_vision_phrases() -> None:
    from ai import chat

    assert chat._match_switch("from vision") == MOUTH_VISION
    assert chat._match_switch("Vision on.") == MOUTH_VISION
    assert chat._match_switch("من فيجن") == MOUTH_VISION
    assert chat._match_switch("from ultron") == MOUTH_ULTRON
    assert chat._match_switch("from the model") == MOUTH_MODEL
    assert chat._match_switch("from the sources") == MOUTH_SOURCES
    # A real question is never captured as a switch.
    assert chat._match_switch("how does vision work?") is None


def test_vision_assistant_records_the_turn_and_carries_no_citations() -> None:
    assistant = _vision_assistant("The answer is 42.")
    conversation = Conversation(title="t")
    answer = assistant.ask("meaning of life?", conversation=conversation)

    assert "The answer is 42." in answer.text
    assert HEADER in answer.text
    assert answer.citations == ()
    assert answer.used_model is True
    assert [message.text for message in conversation.messages][0] == "meaning of life?"


def test_vision_assistant_passes_prior_turns_to_the_model() -> None:
    runner = FakeRunner("Paris.")
    assistant = vision_mouth.VisionAssistant(VisionEngine(runner=runner))
    conversation = Conversation(title="t")
    conversation.add_user("What is the capital of France?")
    conversation.add_assistant(f"{HEADER}\n\nParis.")

    assistant.ask("and its river?", conversation=conversation)

    messages, _kwargs = runner.calls[-1]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "What is the capital of France?"}
    assert messages[2] == {"role": "assistant", "content": "Paris."}
    assert messages[-1] == {"role": "user", "content": "and its river?"}


def test_vision_assistant_reports_an_empty_reply_honestly() -> None:
    assistant = _vision_assistant("")
    answer = assistant.ask("anything?", conversation=Conversation(title="t"))
    assert vision_mouth.VISION_EMPTY in answer.text


def test_vision_assistant_reads_an_attached_file_from_the_conversation() -> None:
    runner = FakeRunner("Zephyrium-9 flowed at 4200 barrels per day.")
    assistant = vision_mouth.VisionAssistant(VisionEngine(runner=runner))
    conversation = Conversation(title="t")
    conversation.add_context(
        "Attached file: well.docx\n\nZephyrium-9 flowed at 4200 barrels per day."
    )

    assistant.ask("What did Zephyrium-9 flow at?", conversation=conversation)

    messages, _kwargs = runner.calls[-1]
    prompt = messages[-1]["content"]
    assert vision_mouth.REFERENCE_LABEL in prompt
    assert "Zephyrium-9" in prompt
    assert prompt.rstrip().endswith("What did Zephyrium-9 flow at?")


def test_vision_assistant_sends_the_bare_question_without_an_attachment() -> None:
    runner = FakeRunner()
    assistant = vision_mouth.VisionAssistant(VisionEngine(runner=runner))

    assistant.ask("hello", conversation=Conversation(title="t"))

    messages, _kwargs = runner.calls[-1]
    assert messages[-1] == {"role": "user", "content": "hello"}


def test_vision_assistant_marks_a_cut_attachment() -> None:
    runner = FakeRunner()
    assistant = vision_mouth.VisionAssistant(VisionEngine(runner=runner), reference_chars=20)
    conversation = Conversation(title="t")
    conversation.add_context("Attached file: long.docx\n\n" + "word " * 50)

    assistant.ask("summarise it", conversation=conversation)

    prompt = runner.calls[-1][0][-1]["content"]
    assert "not shown here" in prompt


def test_load_vision_assistant_raises_when_unavailable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    with pytest.raises(VisionUnavailable):
        vision_mouth.load_vision_assistant()


def test_chat_main_reports_a_missing_vision_honestly(tmp_path: Path, monkeypatch, capsys) -> None:
    """``chat --vision`` without the extra/model exits non-zero with the real next step."""
    from ai import chat

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert chat.main(["--vision", "--no-web"]) == 1
    out = capsys.readouterr().out
    assert "Could not load Vision" in out
    assert "requirements-vision.txt" in out


def test_chat_accepts_the_vision_options(tmp_path: Path, monkeypatch, capsys) -> None:
    """The Vision flags are real; ``--list-sessions`` returns before a mouth loads."""
    from ai import chat

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    code = chat.main(
        [
            "--list-sessions",
            "--vision",
            "--vision-model",
            "x.gguf",
            "--vision-tokens",
            "64",
            "--vision-temperature",
            "0.2",
        ]
    )
    assert code == 0
    assert "sessions" in capsys.readouterr().out.lower()


def test_run_loop_switches_to_vision_and_back(monkeypatch, capsys) -> None:
    from ai import chat
    from core.jarvis import JarvisTurn

    session, switcher = _switcher()

    class _Mode:
        def handle(self, *a, **k):
            return JarvisTurn(
                heard="hi", response="Vision says hello.", woke=True, routed="answer"
            )

    answers = iter(["from vision", "hi", "from the sources", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert chat.run_loop(_Mode(), as_json=False, switcher=switcher) == 0
    out = capsys.readouterr().out
    assert "[vision] Now answering from the petroleum local model" in out
    assert "[sources] Now answering from retrieved passages" in out


def test_run_loop_reports_a_missing_vision_without_switching(monkeypatch, capsys) -> None:
    from ai import chat
    from core.jarvis import JarvisTurn

    session = FakeSession()
    switcher = chat.ModelSwitcher(
        cast(Any, session),
        loader=lambda: None,
        vision_loader=lambda: (_ for _ in ()).throw(VisionUnavailable("no model file")),
    )

    class _Mode:
        def handle(self, *a, **k):
            return JarvisTurn(heard="hi", response="ok", woke=True, routed="answer")

    answers = iter(["from vision", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert chat.run_loop(_Mode(), as_json=False, switcher=switcher) == 0
    out = capsys.readouterr().out
    assert "Could not load Vision" in out
    assert "requirements-vision.txt" in out
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
        vision_model=None,
    )


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_model_selector_offers_vision(tmp_path) -> None:
    import tkinter as tk

    from gui.app import MOUTH_CHOICES, MOUTH_LABELS, FaydGUI

    root = tk.Tk()
    try:
        app = FaydGUI(root, service=_gui_service())
        root.update()
        labels = [
            app._source_menu["menu"].entrycget(i, "label")
            for i in range(app._source_menu["menu"].index("end") + 1)
        ]
        assert MOUTH_LABELS[MOUTH_VISION] in labels
        assert [label for _key, label in MOUTH_CHOICES] == labels
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_switches_to_vision_in_the_same_chat(tmp_path) -> None:
    import tkinter as tk

    from gui.app import MOUTH_LABELS, FaydGUI

    root = tk.Tk()
    try:
        service = _gui_service()
        service.switcher._vision_loader = lambda: _vision_assistant("A windowed reply.")
        app = FaydGUI(root, service=service)
        root.update()
        app._add_bubble("assistant", "keep me across the mouth switch")

        app._on_source_menu(MOUTH_LABELS[MOUTH_VISION])
        root.update()

        # The same chat now answers from Vision: no new window, transcript intact.
        assert service.source == MOUTH_VISION
        assert app._source_choice_label() == MOUTH_LABELS[MOUTH_VISION]
        texts = [entry.text for entry in app._rows]
        assert any("keep me across the mouth switch" in text for text in texts)
        assert any("Vision" in text for text in texts)

        app._on_source_menu(MOUTH_LABELS[MOUTH_SOURCES])
        root.update()
        assert service.source == MOUTH_SOURCES
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_reports_a_missing_vision_in_the_transcript(tmp_path) -> None:
    import tkinter as tk

    from gui.app import MOUTH_LABELS, FaydGUI

    root = tk.Tk()
    try:
        service = _gui_service()
        service.switcher._vision_loader = lambda: (_ for _ in ()).throw(
            VisionUnavailable("no model file")
        )
        app = FaydGUI(root, service=service)
        root.update()

        app._on_source_menu(MOUTH_LABELS[MOUTH_VISION])
        root.update()

        # The mouth did not change, and the real next step is shown.
        assert service.source == MOUTH_SOURCES
        assert app._source_choice_label() == MOUTH_LABELS[MOUTH_SOURCES]
        texts = [entry.text for entry in app._rows]
        assert any("Could not load Vision" in text for text in texts)
        assert any("requirements-vision.txt" in text for text in texts)
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_vision_sidebar_action_switches_the_mouth(tmp_path) -> None:
    import tkinter as tk

    from gui.app import FaydGUI
    from gui.actions import action_by_key

    root = tk.Tk()
    try:
        service = _gui_service()
        service.switcher._vision_loader = lambda: _vision_assistant()
        app = FaydGUI(root, service=service)
        root.update()

        app._on_action(action_by_key("vision"))
        root.update()

        assert service.source == MOUTH_VISION
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_hero_names_all_four_mouths(tmp_path) -> None:
    """The welcome hint must not forget Vision now that it is a mouth here too."""
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
        assert "Vision" in hints[0]
    finally:
        root.destroy()
