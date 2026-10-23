"""The optional-feature installer: honest status, working commands, no false promises.

The user reported a scanned-PDF attach failing with "install the optional
packages ... pip install fayd-al-ruh[ocr]" -- a command that cannot work, because
the project is not published on PyPI. These tests pin the fix: the app now knows
a command that *does* work from a source checkout, offers to run it, and never
claims pip can install the Tesseract system package.

Nothing here touches the network or the real pip: :func:`optional_deps.install`
takes an injected runner, and the availability probes are monkeypatched, so the
command construction and the result handling are tested as real code paths.
"""

from __future__ import annotations

import pytest
from core import optional_deps
from core.optional_deps import (
    FEATURE_IMAGE,
    FEATURE_OCR,
    FEATURE_PDF,
    FEATURE_PIPER,
    FeatureStatus,
    install,
    install_command,
    install_hint,
    is_available,
    missing_imports,
    pip_install_command,
    requirements_path,
    status,
)

# --- status is honest about the two kinds of missing piece ----------------


def test_status_reports_missing_python_packages(monkeypatch):
    monkeypatch.setattr(optional_deps, "missing_imports", lambda key: ("pypdfium2", "pytesseract"))
    state = status(FEATURE_OCR)
    assert state.python_ok is False
    assert state.system_ok is False
    assert state.available is False
    assert "not installed" in str(state)


def test_status_reports_a_missing_system_engine(monkeypatch):
    # The Python packages are present, but Tesseract itself is not reachable.
    monkeypatch.setattr(optional_deps, "missing_imports", lambda key: ())
    monkeypatch.setattr(optional_deps, "_ocr_system_ok", lambda: False)
    state = status(FEATURE_OCR)
    assert state.python_ok is True
    assert state.system_ok is False
    assert state.available is False
    assert "system engine" in str(state)


def test_pdf_has_no_system_dependency(monkeypatch):
    monkeypatch.setattr(optional_deps, "missing_imports", lambda key: ())
    assert status(FEATURE_PDF).system_ok is True
    assert is_available(FEATURE_PDF) is True


def test_missing_imports_names_only_the_absent_ones(monkeypatch):
    import importlib.util

    def fake_find_spec(name: str):
        return None if name == "pytesseract" else object()

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    assert missing_imports(FEATURE_OCR) == ("pytesseract",)


# --- the learned image backend has a second half (the weights) ------------


def test_image_status_reports_missing_python_packages(monkeypatch):
    monkeypatch.setattr(optional_deps, "missing_imports", lambda key: ("torch", "diffusers"))
    state = status(FEATURE_IMAGE)
    assert state.python_ok is False
    assert state.system_ok is False
    assert state.available is False
    assert "not installed" in str(state)


def test_image_status_reports_uncached_weights(monkeypatch):
    # The packages are present, but the one-time weights download has not run.
    monkeypatch.setattr(optional_deps, "missing_imports", lambda key: ())
    monkeypatch.setattr(optional_deps, "_image_weights_cached", lambda: False)
    state = status(FEATURE_IMAGE)
    assert state.python_ok is True
    assert state.system_ok is False
    assert state.available is False
    assert "weights" in str(state)


def test_image_is_available_when_packages_and_weights_are_present(monkeypatch):
    monkeypatch.setattr(optional_deps, "missing_imports", lambda key: ())
    monkeypatch.setattr(optional_deps, "_image_weights_cached", lambda: True)
    assert is_available(FEATURE_IMAGE) is True


# --- the bundled requirements files are real and correct ------------------


@pytest.mark.parametrize(
    "key", [FEATURE_PDF, FEATURE_OCR, FEATURE_IMAGE, FEATURE_PIPER]
)
def test_requirements_file_exists_and_lists_its_imports(key):
    path = requirements_path(key)
    assert path.exists(), f"{path} must ship with the project"
    text = path.read_text(encoding="utf-8").lower()
    for name in optional_deps.feature(key).import_names:
        # ``PIL`` is provided by the ``Pillow`` distribution.
        dist = "pillow" if name == "PIL" else name
        assert dist in text


def test_ocr_requirements_name_the_tesseract_system_packages():
    text = requirements_path(FEATURE_OCR).read_text(encoding="utf-8")
    assert "tesseract-ocr" in text
    assert "tesseract-ocr-ara" in text


# --- the install command works from a source checkout ---------------------


def test_install_command_uses_the_requirements_file_and_this_interpreter():
    command = install_command(FEATURE_OCR)
    assert command is not None
    assert command[1:] == ("-m", "pip", "install", "-r", str(requirements_path(FEATURE_OCR)))


def test_install_command_is_none_in_a_frozen_build(monkeypatch):
    monkeypatch.setattr(optional_deps, "is_frozen", lambda: True)
    assert install_command(FEATURE_OCR) is None


def test_install_command_is_none_when_the_file_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(optional_deps, "requirements_path", lambda key: tmp_path / "absent.txt")
    assert install_command(FEATURE_OCR) is None


def test_pip_install_command_is_the_short_form():
    assert pip_install_command(FEATURE_OCR) == "python -m pip install -r requirements-ocr.txt"


def test_install_hint_names_the_working_command_and_the_system_note():
    hint = install_hint(FEATURE_OCR)
    assert "requirements-ocr.txt" in hint
    assert "tesseract" in hint.lower()
    # It must not send the user to the unpublished package again.
    assert "fayd-al-ruh[ocr]" not in hint


def test_pdf_install_hint_has_no_system_note():
    hint = install_hint(FEATURE_PDF)
    assert "requirements-pdf.txt" in hint
    assert "system package" not in hint.lower()


# --- install runs the command and reports the outcome --------------------


def test_install_succeeds_when_pip_succeeds_and_the_imports_appear(monkeypatch):
    monkeypatch.setattr(optional_deps, "missing_imports", lambda key: ())
    seen: list[tuple[str, ...]] = []

    def runner(argv):
        seen.append(tuple(argv))
        return 0, "Successfully installed pypdfium2"

    result = install(FEATURE_OCR, runner=runner)
    assert result.ok is True
    assert result.ran is True
    assert result.returncode == 0
    assert seen and seen[0][1:] == ("-m", "pip", "install", "-r", str(requirements_path(FEATURE_OCR)))
    assert "installed" in result.message.lower()


def test_install_reports_a_pip_failure_without_raising(monkeypatch):
    monkeypatch.setattr(optional_deps, "missing_imports", lambda key: ("pytesseract",))

    def runner(argv):
        return 1, "ERROR: no matching distribution found"

    result = install(FEATURE_OCR, runner=runner)
    assert result.ok is False
    assert result.returncode == 1
    assert "no matching distribution" in result.message


def test_install_reports_when_pip_finishes_but_the_import_still_fails(monkeypatch):
    monkeypatch.setattr(optional_deps, "missing_imports", lambda key: ("pytesseract",))
    result = install(FEATURE_OCR, runner=lambda argv: (0, "ok"))
    assert result.ok is False
    assert "still cannot be imported" in result.message


def test_install_handles_pip_being_unlaunchable(monkeypatch):
    monkeypatch.setattr(optional_deps, "missing_imports", lambda key: ())

    def runner(argv):
        raise OSError("no such file")

    result = install(FEATURE_OCR, runner=runner)
    assert result.ok is False
    assert "Could not run pip" in result.message


def test_install_cannot_run_in_a_frozen_build(monkeypatch):
    monkeypatch.setattr(optional_deps, "is_frozen", lambda: True)
    called = False

    def runner(argv):  # pragma: no cover - must not be reached
        nonlocal called
        called = True
        return 0, ""

    result = install(FEATURE_OCR, runner=runner)
    assert result.ok is False
    assert result.ran is False
    assert called is False
    assert "requirements-ocr.txt" in result.message


def test_install_reports_the_system_note_when_ocr_engine_is_still_absent(monkeypatch):
    monkeypatch.setattr(optional_deps, "missing_imports", lambda key: ())
    monkeypatch.setattr(optional_deps, "_ocr_system_ok", lambda: False)
    result = install(FEATURE_OCR, runner=lambda argv: (0, "ok"))
    assert result.ok is True
    assert "tesseract" in result.message.lower()


def test_install_reports_the_download_step_when_image_weights_are_absent(monkeypatch):
    monkeypatch.setattr(optional_deps, "missing_imports", lambda key: ())
    monkeypatch.setattr(optional_deps, "_image_weights_cached", lambda: False)
    result = install(FEATURE_IMAGE, runner=lambda argv: (0, "ok"))
    assert result.ok is True
    assert "download" in result.message.lower()


def test_image_install_hint_names_the_working_command_and_the_download():
    hint = install_hint(FEATURE_IMAGE)
    assert "requirements-image.txt" in hint
    assert "download" in hint.lower()
    # It must not send the user to the unpublished package.
    assert "fayd-al-ruh[" not in hint


def test_install_passes_progress_lines(monkeypatch):
    monkeypatch.setattr(optional_deps, "missing_imports", lambda key: ())
    messages: list[str] = []
    install(FEATURE_OCR, runner=lambda argv: (0, "ok"), progress=messages.append)
    assert messages and "ocr" in messages[0].lower()


# --- the feature registry is complete ------------------------------------


def test_unknown_feature_raises_key_error():
    with pytest.raises(KeyError):
        optional_deps.feature("nope")


def test_feature_status_available_requires_both_halves():
    assert FeatureStatus(key=FEATURE_OCR, python_ok=True, system_ok=True).available is True
    assert FeatureStatus(key=FEATURE_OCR, python_ok=True, system_ok=False).available is False
