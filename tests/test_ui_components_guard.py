"""Guards that the shared components stay the one implementation (Phase 1).

These tests read the real ``gui/`` package source with ``ast`` and fail when a
dialog grows back its own copy of a component the library already owns. They are
the regression gate for the Phase-1 de-duplication: a new window that types its
own ``_button`` (instead of delegating to :func:`gui.components.button`) is
caught here, not by a reviewer's eye.
"""

from __future__ import annotations

import ast
import pathlib

GUI = pathlib.Path(__file__).resolve().parent.parent / "gui"


def _methods(path: pathlib.Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, ast.FunctionDef] = {}
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
            found.setdefault(fn.name, fn)
    return found


def _builds_a_raw_button(fn: ast.FunctionDef) -> bool:
    """True when the method constructs a ``tk.Button`` itself rather than delegating."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            func = node.func
            # tk.Button(...) / tkinter.Button(...)
            if isinstance(func, ast.Attribute) and func.attr == "Button":
                return True
    return False


def test_no_dialog_reimplements_its_own_button() -> None:
    """Every ``_button`` must delegate to the shared factory, not build a widget."""
    offenders: list[str] = []
    for path in sorted(GUI.glob("*_dialog.py")):
        for name, fn in _methods(path).items():
            if name == "_button" and _builds_a_raw_button(fn):
                offenders.append(f"{path.name}::{name}")
    assert offenders == [], f"these still hand-roll a tk.Button: {offenders}"


def test_no_dialog_reimplements_a_section() -> None:
    """A ``_section`` must delegate to :class:`gui.components.Section`."""
    offenders: list[str] = []
    for path in sorted(GUI.glob("*_dialog.py")):
        methods = _methods(path)
        if "_section" not in methods:
            continue
        fn = methods["_section"]
        source = ast.get_source_segment(path.read_text(encoding="utf-8"), fn) or ""
        if "components" not in source and "Section(" not in source:
            offenders.append(f"{path.name}::_section")
    assert offenders == [], f"these still hand-roll a section: {offenders}"


def test_components_module_exposes_the_library() -> None:
    from gui import components

    for name in (
        "Button",
        "Section",
        "DialogHeader",
        "StatusLine",
        "EmptyState",
        "LoadingState",
        "SearchInput",
        "SettingsRow",
        "SettingsSection",
        "Notification",
        "NotificationCenter",
        "WorkerPump",
        "button",
    ):
        assert hasattr(components, name), name
        assert name in components.__all__, name
