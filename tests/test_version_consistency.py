"""The version string must not drift between its three published copies.

``core/version.py`` is the source of truth; ``pyproject.toml`` carries it for
packaging and the README's status line shows it to a reader. A release that
bumps one and forgets another is a silent lie about what is shipped, so the
three are pinned together here.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from core.version import __version__

ROOT = Path(__file__).resolve().parent.parent


def test_pyproject_matches_the_source_of_truth():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["version"] == __version__


def test_readme_status_line_matches_the_source_of_truth():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"^- Version:\s*(\S+)\s*$", readme, re.MULTILINE)
    assert match, "README status block should state the version"
    assert match.group(1) == __version__


def test_version_is_a_plain_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__)
