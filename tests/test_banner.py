"""Banner: the presentation local language model, its CLI, menu, and chat/GUI wiring.

Banner is deliberately a *thin* layer over the Presentation Studio's own model
(:mod:`core.presentation_model`), so several tests below assert exactly that: the
catalog, the model folder, and the download gate are the Studio's, not a second
copy. Everything heavy is injected, so these tests need neither
``llama-cpp-python`` nor a multi-gigabyte model file: the engine is exercised
through a fake runner, and the CLI/menu/chat/GUI wiring is checked by import and
dispatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ai import banner as banner_cli
from ai import banner_mouth
from ai.chat import MOUTH_BANNER, MOUTH_MODEL, MOUTH_SOURCES, MOUTH_ULTRON, MOUTH_VISION
from cli import menu as cli_menu
from core import optional_deps
from core import presentation_model
from core.banner import (
    CATALOG,
    DEFAULT_SPEC_KEY,
    HEADER,
    NAME,
    BannerEngine,
    BannerUnavailable,
    download_model,
    find_model_file,
    model_is_cached,
    resolve_spec,
    spec_names,
    spec_summaries,
)
from core.conversation import Conversation
from core.optional_deps import FEATURE_BANNER
from gui import actions as gui_actions


class FakeRunner:
    """A stand-in for the llama-cpp runner: records calls, echoes a reply."""

    def __init__(self, reply: str = "Hello from Banner.") -> None:
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
        yield "Banner."


# --- the thin layer over the Studio's model --------------------------------


def test_banner_reuses_the_studios_catalog_and_default() -> None:
    """One catalog, one default: Banner must not fork the Studio's model list."""
    assert CATALOG is presentation_model.CATALOG
    assert DEFAULT_SPEC_KEY == presentation_model.DEFAULT_SPEC_KEY == "qwen3-8b"
    assert resolve_spec(None) is CATALOG["qwen3-8b"]


def test_banner_unavailable_is_a_presentation_model_unavailable() -> None:
    """A caller that catches the Studio's error still catches Banner's."""
    assert issubclass(BannerUnavailable, presentation_model.PresentationModelUnavailable)


def test_banner_engine_uses_the_studios_runner() -> None:
    """The engine's default runner is the Studio's in-process GGUF runner."""
    import inspect

    source = inspect.getsource(BannerEngine._active_runner)
    assert "LlamaCppPresentationRunner" in source


# --- engine ----------------------------------------------------------------


def test_engine_builds_the_system_then_history_then_prompt() -> None:
    engine = BannerEngine(runner=FakeRunner())
    messages = engine.build_messages([("user", "hi"), ("assistant", "hello")], "an outline?")
    assert messages[0]["role"] == "system"
    assert "presentation" in messages[0]["content"].lower()
    assert messages[1] == {"role": "user", "content": "hi"}
    assert messages[2] == {"role": "assistant", "content": "hello"}
    assert messages[-1] == {"role": "user", "content": "an outline?"}


def test_engine_reply_returns_the_runner_text() -> None:
    runner = FakeRunner("Section 1\n  - point")
    engine = BannerEngine(runner=runner)
    assert engine.reply("an outline for well testing") == "Section 1\n  - point"
    _messages, kwargs = runner.calls[0]
    assert kwargs["max_tokens"] == engine.max_tokens
    assert kwargs["temperature"] == engine.temperature


def test_engine_rejects_an_empty_prompt() -> None:
    engine = BannerEngine(runner=FakeRunner())
    with pytest.raises(ValueError):
        engine.reply("   ")


def test_engine_streams_when_the_runner_can() -> None:
    engine = BannerEngine(runner=StreamingRunner())
    assert "".join(engine.stream_reply("hi")) == "Hello from Banner."


def test_engine_stream_falls_back_to_one_piece() -> None:
    engine = BannerEngine(runner=FakeRunner("whole"))
    assert list(engine.stream_reply("hi")) == ["whole"]


def test_engine_validates_its_settings() -> None:
    with pytest.raises(ValueError):
        BannerEngine(context=0)
    with pytest.raises(ValueError):
        BannerEngine(max_tokens=0)
    with pytest.raises(ValueError):
        BannerEngine(temperature=3.0)
    with pytest.raises(ValueError):
        BannerEngine(top_p=0.0)


def test_injected_runner_is_always_available() -> None:
    assert BannerEngine(runner=FakeRunner()).is_available() is True


def test_engine_without_a_model_is_unavailable_and_explains(tmp_path: Path) -> None:
    engine = BannerEngine(root=tmp_path)
    assert engine.is_available() is False
    with pytest.raises(BannerUnavailable):
        engine.reply("anything")
    assert "no local model file yet" in engine.describe().lower()


def test_engine_uses_a_model_file_it_finds(tmp_path: Path) -> None:
    (tmp_path / "banner.gguf").write_bytes(b"x")
    engine = BannerEngine(root=tmp_path)
    assert engine.model_path == tmp_path / "banner.gguf"


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


# --- the download catalog --------------------------------------------------


def test_catalog_has_a_default_and_named_specs() -> None:
    assert DEFAULT_SPEC_KEY in CATALOG
    assert resolve_spec(None) is CATALOG[DEFAULT_SPEC_KEY]
    assert "qwen3-8b" in spec_names()
    for _key, title, size, summary in spec_summaries():
        assert title and summary
        assert size > 0


def test_resolve_spec_defaults_and_rejects_unknown() -> None:
    assert resolve_spec("QWEN3-8B").key == "qwen3-8b"
    # Banner reuses the Studio's resolver, so its error is the Studio's type; a
    # Banner caller still catches it, since BannerUnavailable subclasses it.
    with pytest.raises(presentation_model.PresentationModelUnavailable):
        resolve_spec("nope")


def test_download_needs_approval_by_default(tmp_path: Path) -> None:
    # The permission/human-approval gate is the Studio's own, reused unchanged.
    with pytest.raises(presentation_model.PresentationModelUnavailable):
        download_model("qwen3-8b", root=tmp_path)


def test_download_returns_a_cached_file_without_the_network(tmp_path: Path) -> None:
    spec = resolve_spec("qwen3-8b")
    cached = tmp_path / spec.filename
    cached.write_bytes(b"x")
    assert download_model("qwen3-8b", root=tmp_path, approved=True) == cached


def test_banner_feature_is_registered() -> None:
    assert FEATURE_BANNER in optional_deps.FEATURES
    spec = optional_deps.feature(FEATURE_BANNER)
    assert spec.requirements_file == "requirements-banner.txt"


def test_banner_pip_hint_points_at_the_requirements_file() -> None:
    hint = optional_deps.pip_install_command(FEATURE_BANNER)
    assert "requirements-banner.txt" in hint


def test_banner_requirements_file_includes_the_studios_runtime() -> None:
    """Banner has no extra pins of its own: it includes the Studio's runtime."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "requirements-banner.txt").read_text(encoding="utf-8")
    assert "-r requirements-presentation.txt" in text


def test_status_line_is_honest_without_a_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    line = banner_cli.status_line()
    assert NAME in line
    assert "no local model" in line.lower()


# --- the CLI ---------------------------------------------------------------


def test_cli_models_lists_the_catalog(capsys) -> None:
    assert banner_cli.run_models(cast(argparse.Namespace, SimpleNamespace(as_json=False))) == 0
    assert "qwen3-8b" in capsys.readouterr().out


def test_cli_models_json(capsys) -> None:
    assert banner_cli.run_models(cast(argparse.Namespace, SimpleNamespace(as_json=True))) == 0
    data = json.loads(capsys.readouterr().out)
    assert any(entry["key"] == "qwen3-8b" for entry in data)


def test_cli_download_without_approve_is_refused(capsys) -> None:
    args = cast(argparse.Namespace, SimpleNamespace(approve=False, model=None, as_json=False))
    assert banner_cli.run_download(args) == 1
    assert "needs --approve" in capsys.readouterr().out


def test_cli_status_reports_plainly(capsys) -> None:
    assert banner_cli.run_status(cast(argparse.Namespace, SimpleNamespace(as_json=False))) == 0
    assert NAME in capsys.readouterr().out


def test_cli_bare_question_is_normalized_to_chat() -> None:
    assert banner_cli._normalize_argv(["an outline?"])[0] == "chat"
    assert banner_cli._normalize_argv(["download"]) == ["download"]
    assert banner_cli._normalize_argv(["--help"]) == ["--help"]


def test_cli_one_shot_uses_an_injected_runner(tmp_path, monkeypatch, capsys) -> None:
    engine = BannerEngine(runner=FakeRunner("Section 1\n  - Well testing"))
    monkeypatch.setattr(banner_cli, "_engine", lambda _args: engine)
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert banner_cli.main(["an outline for a well-testing deck"]) == 0
    out = capsys.readouterr().out
    assert "Section 1" in out
    assert HEADER in out


def test_cli_interactive_quits_immediately(monkeypatch, capsys) -> None:
    engine = BannerEngine(runner=FakeRunner())
    monkeypatch.setattr(banner_cli, "_engine", lambda _args: engine)
    monkeypatch.setattr("builtins.input", lambda _prompt: "quit")
    assert banner_cli.main([]) == 0


# --- menu and CLI dispatch -------------------------------------------------


def test_menu_shows_banner_directly_below_vision() -> None:
    labels = [title for title, _action, _argv in cli_menu.visible_menu()]
    vision = next(i for i, label in enumerate(labels) if label.startswith("Vision"))
    assert labels[vision + 1] == "Banner (presentation local model: outlines and slide content)"


def test_menu_action_runs_the_banner_module(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_module(name: str, argv) -> int:
        captured["name"] = name
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(cli_menu, "_run_module", fake_run_module)
    assert cli_menu.run_banner(["--session", "s"]) == 0
    assert captured["name"] == "ai.banner"
    assert captured["argv"] == ["--session", "s"]


def test_cli_dispatches_banner() -> None:
    from cli import __main__ as cli_entry

    assert "banner" in cli_entry._COMMANDS
    assert "banner-chat" in cli_entry._COMMANDS


def test_gui_sidebar_has_a_banner_action() -> None:
    keys = {action.key for action in gui_actions.ACTIONS}
    assert "banner" in keys
    action = gui_actions.action_by_key("banner")
    assert action.command is cli_menu.run_banner_chat


def test_gui_banner_action_is_hidden_from_the_sidebar() -> None:
    """Banner is offered by the header's model selector, so its sidebar row is hidden."""
    action = gui_actions.action_by_key("banner")
    assert action.hidden is True
    assert action not in gui_actions.visible_actions()
    assert "Answer from Banner (presentation local model)" not in [
        a.label for a in gui_actions.visible_actions()
    ]


def test_run_banner_chat_adds_the_flag_once(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_module(name: str, argv) -> int:
        captured["name"] = name
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(cli_menu, "_run_module", fake_run_module)
    assert cli_menu.run_banner_chat([]) == 0
    assert captured["name"] == "ai.chat"
    assert captured["argv"] == ["--banner"]
    cli_menu.run_banner_chat(["--banner", "--json"])
    assert captured["argv"] == ["--banner", "--json"]


# --- the Banner mouth inside the chat --------------------------------------


class FakeSession:
    """A stand-in for ``JarvisMode`` that the chat's switcher can swap a mouth into."""

    def __init__(self) -> None:
        self.assistant: Any = SimpleNamespace(name="grounded")
        self.help_text = "help"
        self.voice = SimpleNamespace(assistant=self.assistant)


def _banner_assistant(reply: str = "Banner says hello.") -> banner_mouth.BannerAssistant:
    return banner_mouth.BannerAssistant(BannerEngine(runner=FakeRunner(reply)))


def _switcher(reply: str = "Banner says hello."):
    from ai.chat import ModelSwitcher

    session = FakeSession()
    switcher = ModelSwitcher(
        cast(Any, session),
        loader=lambda: (_ for _ in ()).throw(AssertionError("the trained model must not load")),
        banner_loader=lambda: _banner_assistant(reply),
    )
    return session, switcher


def test_switcher_banner_swaps_the_mouth_and_keeps_the_grounded_one() -> None:
    session, switcher = _switcher()
    grounded = switcher.grounded
    grounded_help = switcher.grounded_help

    switcher.use_banner()

    assert switcher.mouth == MOUTH_BANNER
    assert switcher.active is True
    assert session.assistant is switcher.banner
    assert session.voice.assistant is switcher.banner
    assert session.help_text != grounded_help
    assert banner_mouth.BANNER_HELP in session.help_text

    switcher.use_sources()
    assert switcher.mouth == MOUTH_SOURCES
    assert session.assistant is grounded
    assert session.help_text == grounded_help


def test_switcher_loads_banner_lazily_and_reuses_it() -> None:
    from ai.chat import ModelSwitcher

    session = FakeSession()
    calls: list[int] = []

    def _loader():
        calls.append(1)
        return _banner_assistant()

    switcher = ModelSwitcher(
        cast(Any, session),
        loader=lambda: None,
        banner_loader=_loader,
    )
    assert calls == []
    switcher.use_banner()
    switcher.use_banner()
    assert calls == [1]


def test_switch_recognises_the_banner_phrases() -> None:
    from ai import chat

    assert chat._match_switch("from banner") == MOUTH_BANNER
    assert chat._match_switch("Banner on.") == MOUTH_BANNER
    assert chat._match_switch("من بانر") == MOUTH_BANNER
    assert chat._match_switch("from vision") == MOUTH_VISION
    assert chat._match_switch("from ultron") == MOUTH_ULTRON
    assert chat._match_switch("from the model") == MOUTH_MODEL
    assert chat._match_switch("from the sources") == MOUTH_SOURCES
    # A real question is never captured as a switch.
    assert chat._match_switch("what is a banner ad?") is None


def test_banner_assistant_records_the_turn_and_carries_no_citations() -> None:
    assistant = _banner_assistant("Section 1\n  - point")
    conversation = Conversation(title="t")
    answer = assistant.ask("an outline?", conversation=conversation)

    assert "Section 1" in answer.text
    assert HEADER in answer.text
    assert answer.citations == ()
    assert answer.used_model is True
    assert [message.text for message in conversation.messages][0] == "an outline?"


def test_banner_assistant_passes_prior_turns_to_the_model() -> None:
    runner = FakeRunner("Section 1")
    assistant = banner_mouth.BannerAssistant(BannerEngine(runner=runner))
    conversation = Conversation(title="t")
    conversation.add_user("Outline a deck on well testing.")
    conversation.add_assistant(f"{HEADER}\n\nSection 1")

    assistant.ask("now the bullets", conversation=conversation)

    messages, _kwargs = runner.calls[-1]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "Outline a deck on well testing."}
    # The honesty header a previous turn carried is dropped before it is fed back.
    assert messages[2] == {"role": "assistant", "content": "Section 1"}
    assert messages[-1] == {"role": "user", "content": "now the bullets"}


def test_banner_assistant_reports_an_empty_reply_honestly() -> None:
    assistant = _banner_assistant("")
    answer = assistant.ask("anything?", conversation=Conversation(title="t"))
    assert banner_mouth.BANNER_EMPTY in answer.text


def test_banner_assistant_reads_an_attached_file_from_the_conversation() -> None:
    runner = FakeRunner("The slide should say 4200 bbl/d.")
    assistant = banner_mouth.BannerAssistant(BannerEngine(runner=runner))
    conversation = Conversation(title="t")
    conversation.add_context("Attached file: well.docx\n\nZephyrium-9 flowed at 4200 bbl/d.")

    assistant.ask("make a slide about the flow rate", conversation=conversation)

    prompt = runner.calls[-1][0][-1]["content"]
    assert banner_mouth.REFERENCE_LABEL in prompt
    assert "Zephyrium-9" in prompt
    assert prompt.rstrip().endswith("make a slide about the flow rate")


def test_banner_assistant_sends_the_bare_question_without_an_attachment() -> None:
    runner = FakeRunner()
    assistant = banner_mouth.BannerAssistant(BannerEngine(runner=runner))

    assistant.ask("hello", conversation=Conversation(title="t"))

    messages, _kwargs = runner.calls[-1]
    assert messages[-1] == {"role": "user", "content": "hello"}


def test_banner_assistant_marks_a_cut_attachment() -> None:
    runner = FakeRunner()
    assistant = banner_mouth.BannerAssistant(BannerEngine(runner=runner), reference_chars=20)
    conversation = Conversation(title="t")
    conversation.add_context("Attached file: long.docx\n\n" + "word " * 50)

    assistant.ask("summarise it", conversation=conversation)

    prompt = runner.calls[-1][0][-1]["content"]
    assert "not shown here" in prompt


def test_load_banner_assistant_raises_when_unavailable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    with pytest.raises(BannerUnavailable):
        banner_mouth.load_banner_assistant()


def test_chat_main_reports_a_missing_banner_honestly(tmp_path: Path, monkeypatch, capsys) -> None:
    """``chat --banner`` without the extra/model exits non-zero with the real next step."""
    from ai import chat

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    assert chat.main(["--banner", "--no-web"]) == 1
    out = capsys.readouterr().out
    assert "Could not load Banner" in out
    assert "requirements-banner.txt" in out


def test_chat_accepts_the_banner_options(tmp_path: Path, monkeypatch, capsys) -> None:
    """The Banner flags are real; ``--list-sessions`` returns before a mouth loads."""
    from ai import chat

    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    code = chat.main(
        [
            "--list-sessions",
            "--banner",
            "--banner-model",
            "x.gguf",
            "--banner-tokens",
            "64",
            "--banner-temperature",
            "0.2",
        ]
    )
    assert code == 0
    assert "sessions" in capsys.readouterr().out.lower()


def test_run_loop_switches_to_banner_and_back(monkeypatch, capsys) -> None:
    from ai import chat
    from core.jarvis import JarvisTurn

    _session, switcher = _switcher()

    class _Mode:
        def handle(self, *a, **k):
            return JarvisTurn(heard="hi", response="Banner says hello.", woke=True, routed="answer")

    answers = iter(["from banner", "hi", "from the sources", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert chat.run_loop(_Mode(), as_json=False, switcher=switcher) == 0
    out = capsys.readouterr().out
    assert "[banner] Now answering from the presentation local model" in out
    assert "[sources] Now answering from retrieved passages" in out


def test_run_loop_reports_a_missing_banner_without_switching(monkeypatch, capsys) -> None:
    from ai import chat
    from core.jarvis import JarvisTurn

    session = FakeSession()
    switcher = chat.ModelSwitcher(
        cast(Any, session),
        loader=lambda: None,
        banner_loader=lambda: (_ for _ in ()).throw(BannerUnavailable("no model file")),
    )

    class _Mode:
        def handle(self, *a, **k):
            return JarvisTurn(heard="hi", response="ok", woke=True, routed="answer")

    answers = iter(["from banner", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert chat.run_loop(_Mode(), as_json=False, switcher=switcher) == 0
    out = capsys.readouterr().out
    assert "Could not load Banner" in out
    assert "requirements-banner.txt" in out
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

    return ChatService(mode=cast(Any, FakeSession()), data_root=Path.cwd(), banner_model=None)


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_model_selector_offers_banner() -> None:
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
        assert MOUTH_LABELS[MOUTH_BANNER] in labels
        assert [label for _key, label in MOUTH_CHOICES] == labels
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_window_switches_to_banner_in_the_same_chat() -> None:
    import tkinter as tk

    from gui.app import MOUTH_LABELS, FaydGUI

    root = tk.Tk()
    try:
        service = _gui_service()
        service.switcher._banner_loader = lambda: _banner_assistant("A windowed reply.")
        app = FaydGUI(root, service=service)
        root.update()
        app._add_bubble("assistant", "keep me across the mouth switch")

        app._on_source_menu(MOUTH_LABELS[MOUTH_BANNER])
        root.update()

        assert service.source == MOUTH_BANNER
        assert app._source_choice_label() == MOUTH_LABELS[MOUTH_BANNER]
        texts = [entry.text for entry in app._rows]
        assert any("keep me across the mouth switch" in text for text in texts)
    finally:
        root.destroy()
