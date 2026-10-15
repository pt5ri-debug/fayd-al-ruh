"""Ultron: the general local language model, its CLI, menu, and GUI wiring.

Everything heavy is injected, so these tests need neither ``llama-cpp-python``
nor a multi-gigabyte model file: the engine is exercised through a fake runner,
and the CLI/menu/GUI wiring is checked by import and dispatch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai import ultron as ultron_cli
from cli import menu as cli_menu
from core import optional_deps
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
    assert action.command is cli_menu.run_ultron


def test_gui_ultron_action_is_not_a_plain_command() -> None:
    """The window opens its own Ultron chat, so the sidebar entry is a real window."""
    # The entry must exist in the actions list so the window can special-case it.
    labels = [action.label for action in gui_actions.ACTIONS]
    assert "Ultron (general local model, any subject)" in labels


# --- GUI window (needs a display) ------------------------------------------


def _display_available() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
    except Exception:
        return False
    root.destroy()
    return True


@pytest.mark.skipif(not _display_available(), reason="no display available")
def test_ultron_dialog_builds_and_replies(tmp_path) -> None:
    import tkinter as tk

    from gui.app import DARK_THEME
    from gui.ultron_dialog import UltronDialog

    root = tk.Tk()
    root.withdraw()
    try:
        engine = UltronEngine(runner=FakeRunner("A windowed reply."))
        dialog = UltronDialog(root, engine, theme=DARK_THEME, font="TkDefaultFont", mono="TkFixedFont")
        dialog._input.insert("1.0", "hello")
        dialog._on_send()
        # The worker thread posts the result; drain it on the main thread.
        import time

        for _ in range(400):
            dialog.window.update()
            if not dialog._busy:
                break
            time.sleep(0.01)
        assert "A windowed reply." in dialog._transcript.get("1.0", "end")
        dialog.window.destroy()
    finally:
        root.destroy()
