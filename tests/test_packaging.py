"""The packaging contract must stay installable and expose one entry point.

``pip install .`` used to fail outright (an empty ``build-system.requires`` left
``setuptools.build_meta`` unimportable, so the backend could never load) and the
project shipped no console script. These checks pin the fix cheaply: they read
``pyproject.toml`` and import the entry point, so a regression is caught without
building a wheel.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_build_backend_is_declared_with_its_requirement():
    data = _pyproject()
    build = data["build-system"]
    assert build["build-backend"] == "setuptools.build_meta"
    # The backend cannot be imported without this; an empty list breaks the build.
    assert any(req.startswith("setuptools") for req in build["requires"])


def test_console_script_points_at_an_importable_callable():
    scripts = _pyproject()["project"]["scripts"]
    assert scripts["fayd-al-ruh"] == "cli.__main__:main"
    module_name, _, attribute = scripts["fayd-al-ruh"].partition(":")
    module = importlib.import_module(module_name)
    assert callable(getattr(module, attribute))


def test_discovery_covers_every_real_package_and_skips_the_tests():
    include = _pyproject()["tool"]["setuptools"]["packages"]["find"]["include"]
    patterns = {pattern.rstrip("*") for pattern in include}
    found = {
        path.parent.name
        for path in ROOT.glob("*/__init__.py")
        if path.parent.name != "tests"
    }
    assert found <= patterns, f"packages missing from discovery: {found - patterns}"
    assert "tests*" in _pyproject()["tool"]["setuptools"]["packages"]["find"]["exclude"]


def test_runtime_dependencies_are_declared():
    dependencies = _pyproject()["project"]["dependencies"]
    names = {dep.split(">")[0].split("=")[0].split("[")[0].strip() for dep in dependencies}
    assert {"numpy", "requests"} <= names


def test_every_optional_import_is_ignored_for_type_checking():
    """An optional dependency imported lazily must not fail mypy on a base install.

    The project's own rule: every optional import name carries a mypy override.
    This scans the *actual* imports of the optional feature modules and checks
    each third-party one is covered, so adding an extra without its override is
    caught here rather than in CI's mypy step.
    """
    import ast
    import sys

    overrides: set[str] = set()
    for entry in _pyproject()["tool"]["mypy"]["overrides"]:
        for name in entry.get("module", []):
            # "piper.*" covers the "piper" top-level name and its submodules.
            overrides.add(name.split(".")[0] if name.endswith(".*") else name)

    local = {"core", "storage", "ai", "ml", "gui", "cli", "ingest"}
    # The base runtime dependencies ship their own type information, so they are
    # installed and checked rather than ignored.
    stubbed = {
        dep.split(">")[0].split("=")[0].split("[")[0].split("<")[0].strip()
        for dep in _pyproject()["project"]["dependencies"]
    }
    optional_modules = [
        "core/piper_speech.py",
        "core/qwen_speech.py",
        "core/pptx_engine.py",
        "core/imagegen_diffusion.py",
        "core/ultron.py",
        "core/pdf_extract.py",
    ]
    missing: set[str] = set()
    for relative in optional_modules:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top in local or top in stubbed or top in sys.stdlib_module_names or top in overrides:
                    continue
                missing.add(f"{relative}: {top}")
    assert not missing, f"optional imports without a mypy override: {sorted(missing)}"

