"""Doc's Studio as the user reaches it: the CLI, the menu, the sidebar, the ＋ menu.

These are the wiring tests: the ``python -m cli docs`` command, its place in the
numbered menu and the hidden GUI sidebar action, the ＋ menu entry, the
``ChatService`` property, and the optional-deps feature. The document engine and
the model layer have their own tests in ``tests/test_docs_studio.py``; here the
point is that the *front ends* reach the same service and never invent a second
one.
"""

from __future__ import annotations

import json

import pytest

from cli import docs as docs_cli
from cli import menu as cli_menu
from core import optional_deps
from core.optional_deps import FEATURE_DOCS
from gui import actions as gui_actions


class RecordingRunner:
    """A deterministic stand-in for the vision model (no extra, no weights)."""

    def __init__(self, answer: str = "The document says porosity is the void fraction.") -> None:
        self.answer = answer

    def chat(self, messages: list[dict], **kwargs) -> str:
        return self.answer


@pytest.fixture()
def document(tmp_path):
    path = tmp_path / "report.md"
    path.write_text("# Report\n\nPorosity is the void fraction of a rock.\n", encoding="utf-8")
    return path


@pytest.fixture()
def wired(monkeypatch):
    """Make the CLI build a service with an injected runner, so no model is needed."""
    from gui.docs_studio import DocsStudioService

    def build() -> DocsStudioService:
        return DocsStudioService(runner=RecordingRunner())

    monkeypatch.setattr(docs_cli, "_service", build)


# --- the CLI ---------------------------------------------------------------


def test_status_runs_and_prints_a_summary(capsys) -> None:
    code = docs_cli.main(["status"])
    out = capsys.readouterr().out
    assert code == docs_cli.EXIT_OK
    assert out.strip()


def test_models_lists_every_catalog_entry(capsys) -> None:
    code = docs_cli.main(["models"])
    out = capsys.readouterr().out
    assert code == docs_cli.EXIT_OK
    assert "qwen3-vl-4b" in out
    assert "qwen3-vl-2b" in out


def test_models_json_is_machine_readable(capsys) -> None:
    code = docs_cli.main(["models", "--json"])
    rows = json.loads(capsys.readouterr().out)
    assert code == docs_cli.EXIT_OK
    assert {row["key"] for row in rows} >= {"qwen3-vl-2b", "qwen3-vl-4b", "qwen3-vl-8b"}
    assert all({"key", "title", "size_gb", "summary"} <= set(row) for row in rows)


def test_open_shows_the_document_and_its_text(wired, document, capsys) -> None:
    code = docs_cli.main(["open", str(document)])
    out = capsys.readouterr().out
    assert code == docs_cli.EXIT_OK
    assert "report.md" in out
    assert "Porosity is the void fraction" in out


def test_open_json_reports_the_honest_facts(wired, document, capsys) -> None:
    code = docs_cli.main(["open", str(document), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == docs_cli.EXIT_OK
    assert payload["filename"] == "report.md"
    assert payload["pages"] == 1
    assert payload["scanned"] is False
    assert payload["vision_pages"] == []


def test_text_prints_the_extracted_text(wired, document, capsys) -> None:
    code = docs_cli.main(["text", str(document)])
    out = capsys.readouterr().out
    assert code == docs_cli.EXIT_OK
    assert "Porosity is the void fraction" in out


def test_pages_reports_the_text_layer(wired, document, capsys) -> None:
    code = docs_cli.main(["pages", str(document)])
    out = capsys.readouterr().out
    assert code == docs_cli.EXIT_OK
    assert "page 1" in out
    assert "text" in out


def test_ask_prints_the_answer_with_the_honesty_header(wired, document, capsys) -> None:
    code = docs_cli.main(["ask", str(document), "what is porosity?"])
    out = capsys.readouterr().out
    assert code == docs_cli.EXIT_OK
    assert "not grounded in cited sources" in out
    assert "void fraction" in out


def test_ask_json_is_machine_readable(wired, document, capsys) -> None:
    code = docs_cli.main(["ask", str(document), "q", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == docs_cli.EXIT_OK
    assert payload["ok"] is True
    assert payload["pages_used"] == [1]


def test_ask_accepts_a_page_selection(wired, document, capsys) -> None:
    code = docs_cli.main(["ask", str(document), "q", "--pages", "1", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == docs_cli.EXIT_OK
    assert payload["pages_used"] == [1]


def test_ask_rejects_a_bad_page_selection(wired, document, capsys) -> None:
    code = docs_cli.main(["ask", str(document), "q", "--pages", "zero"])
    assert code == docs_cli.EXIT_USAGE
    assert "pages" in capsys.readouterr().out.lower()


def test_ask_without_a_model_exits_with_the_model_code(document, capsys) -> None:
    code = docs_cli.main(["ask", str(document), "q"])
    out = capsys.readouterr().out
    assert code == docs_cli.EXIT_MODEL
    assert "Cannot answer" in out


def test_open_missing_document_exits_with_the_document_code(wired, tmp_path, capsys) -> None:
    code = docs_cli.main(["open", str(tmp_path / "nope.md")])
    assert code == docs_cli.EXIT_DOCUMENT
    assert "Could not read the document" in capsys.readouterr().out


def test_open_unsupported_document_exits_with_the_document_code(wired, tmp_path, capsys) -> None:
    path = tmp_path / "old.doc"
    path.write_text("x", encoding="utf-8")
    code = docs_cli.main(["open", str(path)])
    assert code == docs_cli.EXIT_DOCUMENT
    assert ".docx" in capsys.readouterr().out


def test_download_without_approval_is_refused(capsys) -> None:
    code = docs_cli.main(["download", "qwen3-vl-4b"])
    out = capsys.readouterr().out
    assert code == docs_cli.EXIT_MODEL
    assert "denied" in out.lower() or "approval" in out.lower()


def test_unknown_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit):
        docs_cli.main(["nope"])


# --- the wiring ------------------------------------------------------------


def test_menu_exposes_the_docs_actions() -> None:
    labels = [label for label, _runner, _args in cli_menu.MENU]
    assert any("Doc's Studio" in label for label in labels)
    assert any("download the local vision model" in label for label in labels)
    assert "run_docs" in cli_menu.__all__


def test_menu_runs_the_docs_module(monkeypatch) -> None:
    seen: list[tuple[str, tuple]] = []

    def fake_run(module: str, argv):
        seen.append((module, tuple(argv)))
        return 0

    monkeypatch.setattr(cli_menu, "_run_module", fake_run)
    assert cli_menu.run_docs(("status",)) == 0
    assert seen == [("cli.docs", ("status",))]


def test_main_dispatches_the_docs_command() -> None:
    from cli import __main__ as entry

    assert "docs" in entry._COMMANDS


def test_sidebar_action_is_hidden_because_the_menu_opens_it() -> None:
    # The ＋ menu already opens the Studio, so its sidebar row is hidden -- the
    # same rule the Import Center, the Code Studio, and the image pages follow.
    hidden = {action.key for action in gui_actions.ACTIONS if action.hidden}
    visible = {action.key for action in gui_actions.visible_actions()}
    assert "docs-studio" in hidden
    assert "docs-studio" not in visible
    assert gui_actions.action_by_key("docs-studio") is not None


def test_optional_deps_declares_the_docs_feature() -> None:
    feature = optional_deps.feature(FEATURE_DOCS)
    assert feature.requirements_file == "requirements-docs.txt"
    assert "llama_cpp" in feature.import_names
    assert "huggingface_hub" in feature.import_names
    command = optional_deps.install_command(FEATURE_DOCS)
    assert command is not None
    assert "requirements-docs.txt" in " ".join(command)
    assert optional_deps.requirements_path(FEATURE_DOCS).is_file()


def test_docs_requirements_file_names_the_runtime() -> None:
    text = optional_deps.requirements_path(FEATURE_DOCS).read_text(encoding="utf-8")
    assert "llama-cpp-python" in text
    assert "pypdfium2" in text


def test_paths_expose_the_docs_folders(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from core import paths

    assert paths.docs_models_dir() == tmp_path / "docs"
    assert paths.docs_pages_dir() == tmp_path / "docs" / "pages"


def _chat_service(tmp_path):
    """A real ``ChatService`` over a fake session, the way the GUI tests build one."""
    from types import SimpleNamespace
    from typing import cast

    from core.command_router import NaturalLanguageRouter
    from core.jarvis import JarvisMode
    from core.tools import build_sandbox
    from gui.service import ChatService

    sandbox = build_sandbox(tmp_path, read_only=True)
    session = SimpleNamespace(
        router=NaturalLanguageRouter(sandbox),
        assistant=SimpleNamespace(name="grounded"),
        help_text="help",
        memory=None,
        conversation=SimpleNamespace(id="CNV-fake"),
        handle=lambda message, require_wake=False, speak=True: SimpleNamespace(
            answer_text="A grounded answer.", response="A grounded answer."
        ),
    )
    return ChatService(mode=cast(JarvisMode, session), data_root=tmp_path)


def test_chat_service_exposes_the_docs_engine(tmp_path) -> None:
    service = _chat_service(tmp_path)
    assert service.docs is service.docs
    assert service.docs.model_names()


def test_chat_service_borrows_its_own_assistant(tmp_path) -> None:
    service = _chat_service(tmp_path)
    # The grounded path answers through this session's own loop, so no second AI
    # is introduced: it must not raise when the session is available.
    answer = service.docs_assistant_answer("what is porosity?")
    assert isinstance(answer, str)
    assert answer
