"""The optional-feature installer as the user reaches it: CLI, menu, and hints.

These are the wiring tests for :mod:`core.optional_deps`: the ``python -m cli
deps`` command, its place in the numbered menu and the GUI sidebar, and the
attachment hint that replaced the dead-end "pip install fayd-al-ruh[ocr]".
"""

from __future__ import annotations

import pytest

from cli import deps as deps_cli
from cli import menu as cli_menu
from core import optional_deps
from core.optional_deps import FEATURE_IMAGE, FEATURE_OCR, FEATURE_PDF
from gui import actions as gui_actions
from gui.attachments import _ocr_attachment_hint


# --- the CLI command -------------------------------------------------------


def test_deps_report_lists_every_feature(capsys):
    code = deps_cli.main([])
    out = capsys.readouterr().out
    assert "PDF support" in out
    assert "OCR support" in out
    assert "Learned image backend" in out
    # 0 when everything is present, 1 when something is missing -- never a crash.
    assert code in (0, 1)


def test_deps_report_exits_nonzero_when_a_feature_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(
        optional_deps,
        "status",
        lambda key: optional_deps.FeatureStatus(key=key, python_ok=False, system_ok=False),
    )
    code = deps_cli.main(["--check"])
    out = capsys.readouterr().out
    assert code == 1
    assert "requirements-ocr.txt" in out


def test_deps_install_runs_the_installer_and_reports(monkeypatch, capsys):
    monkeypatch.setattr(optional_deps, "is_available", lambda key: False)

    def fake_install(key, *, progress=None):
        if progress is not None:
            progress("Installing…")
        return optional_deps.InstallResult(key=key, ok=True, message="OCR support installed.")

    monkeypatch.setattr(optional_deps, "install", fake_install)
    code = deps_cli.main(["--install", "ocr"])
    out = capsys.readouterr().out
    assert code == 0
    assert "OCR support installed." in out


def test_deps_install_skips_an_already_present_feature(monkeypatch, capsys):
    monkeypatch.setattr(optional_deps, "is_available", lambda key: True)
    called = False

    def fake_install(key, *, progress=None):  # pragma: no cover - must not run
        nonlocal called
        called = True
        return optional_deps.InstallResult(key=key, ok=True)

    monkeypatch.setattr(optional_deps, "install", fake_install)
    assert deps_cli.main(["--install", "pdf"]) == 0
    assert called is False
    assert "already installed" in capsys.readouterr().out.lower()


def test_deps_install_failure_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(optional_deps, "is_available", lambda key: False)
    monkeypatch.setattr(
        optional_deps,
        "install",
        lambda key, *, progress=None: optional_deps.InstallResult(
            key=key, ok=False, message="pip failed"
        ),
    )
    assert deps_cli.main(["--install", "ocr"]) == 1
    assert "pip failed" in capsys.readouterr().out


# --- menu and CLI registration --------------------------------------------


def test_deps_is_registered_in_the_cli() -> None:
    from cli.__main__ import _COMMANDS

    assert "run_deps" in cli_menu.__all__
    assert _COMMANDS["deps"] is cli_menu.run_deps


def test_deps_is_in_the_visible_menu() -> None:
    labels = [title for title, _action, _args in cli_menu.visible_menu()]
    assert any("optional PDF/OCR/image support" in label for label in labels)


def test_deps_is_in_the_gui_sidebar() -> None:
    action = gui_actions.action_by_key("deps")
    assert action.command is cli_menu.run_deps


# --- the attachment hint points at a working next step --------------------


def test_ocr_hint_without_the_extra_points_at_the_installer(monkeypatch):
    monkeypatch.setattr("core.pdf_ocr.ocr_available", lambda: False)
    message = "'scan.pdf' contains no extractable text (it may be a scanned image)."
    hint = _ocr_attachment_hint(message, ocr=True)
    assert "Install OCR support" in hint
    assert "requirements-ocr.txt" in hint


def test_ocr_hint_when_ocr_ran_but_found_nothing(monkeypatch):
    monkeypatch.setattr("core.pdf_ocr.ocr_available", lambda: True)
    message = "'scan.pdf' contains no extractable text (it may be a scanned image)."
    hint = _ocr_attachment_hint(message, ocr=True)
    assert "found no readable text" in hint
    assert "requirements-ocr.txt" not in hint


def test_non_ocr_failure_is_returned_unchanged():
    message = "Could not read the file: it is encrypted."
    assert _ocr_attachment_hint(message, ocr=False) == message


@pytest.mark.parametrize("key", [FEATURE_PDF, FEATURE_OCR, FEATURE_IMAGE])
def test_install_hint_never_names_the_unpublished_package(key):
    assert "fayd-al-ruh[" not in optional_deps.install_hint(key)


# --- the OCR *system* half is offered too (mirroring the speech engine) ----


def test_deps_install_ocr_also_installs_the_tesseract_engine(monkeypatch, capsys):
    """pip installs the packages; the Tesseract engine is a system package."""
    from core import pdf_ocr

    monkeypatch.setattr(optional_deps, "is_available", lambda key: False)
    monkeypatch.setattr(
        optional_deps,
        "install",
        lambda key, *, progress=None: optional_deps.InstallResult(
            key=key, ok=True, message="OCR support installed."
        ),
    )
    # The packages are present (so the engine step is meaningful) but the engine is
    # not; pinning this keeps the test independent of the [ocr] extra.
    monkeypatch.setattr(pdf_ocr, "missing_ocr_python_packages", lambda: ())
    monkeypatch.setattr(pdf_ocr, "ocr_available", lambda: False)
    monkeypatch.setattr(
        pdf_ocr,
        "install_ocr_engine",
        lambda **kwargs: pdf_ocr.OcrInstallResult(ok=True, message="Installed the Tesseract OCR engine."),
    )
    code = deps_cli.main(["--install", "ocr"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Installed the Tesseract OCR engine." in out


def test_deps_install_ocr_reports_a_failed_engine_install(monkeypatch, capsys):
    from core import pdf_ocr

    monkeypatch.setattr(optional_deps, "is_available", lambda key: False)
    monkeypatch.setattr(
        optional_deps,
        "install",
        lambda key, *, progress=None: optional_deps.InstallResult(
            key=key, ok=True, message="OCR support installed."
        ),
    )
    monkeypatch.setattr(pdf_ocr, "missing_ocr_python_packages", lambda: ())
    monkeypatch.setattr(pdf_ocr, "ocr_available", lambda: False)
    monkeypatch.setattr(
        pdf_ocr,
        "install_ocr_engine",
        lambda **kwargs: pdf_ocr.OcrInstallResult(ok=False, message="The install command failed."),
    )
    assert deps_cli.main(["--install", "ocr"]) == 1
    assert "failed" in capsys.readouterr().out.lower()


def test_deps_install_ocr_skips_the_engine_when_the_packages_are_absent(monkeypatch, capsys):
    """With no OCR packages, the engine step is skipped (a package manager would be for nothing)."""
    from core import pdf_ocr

    monkeypatch.setattr(optional_deps, "is_available", lambda key: False)
    monkeypatch.setattr(
        optional_deps,
        "install",
        lambda key, *, progress=None: optional_deps.InstallResult(
            key=key, ok=True, message="OCR support installed."
        ),
    )
    monkeypatch.setattr(pdf_ocr, "missing_ocr_python_packages", lambda: ("pytesseract",))
    called = []
    monkeypatch.setattr(
        pdf_ocr, "install_ocr_engine", lambda **kwargs: called.append(True) or pdf_ocr.OcrInstallResult(ok=True)
    )
    assert deps_cli.main(["--install", "ocr"]) == 0
    assert called == []


def test_deps_report_names_the_engine_step_when_only_the_engine_is_missing(monkeypatch, capsys):
    """Packages present, engine missing: the report shows the system-package step."""
    from core import pdf_ocr

    monkeypatch.setattr(
        optional_deps,
        "status",
        lambda key: optional_deps.FeatureStatus(
            key=key,
            python_ok=key == FEATURE_OCR,
            system_ok=key != FEATURE_OCR,
            missing_python=(),
        ),
    )
    monkeypatch.setattr(
        pdf_ocr, "ocr_install_plan", lambda: pdf_ocr.OcrInstallPlan("linux", ("apt",), "run apt")
    )
    code = deps_cli.main(["--check"])
    out = capsys.readouterr().out
    assert code == 1
    assert "run apt" in out
