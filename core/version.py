"""Single source of truth for the project version.

Kept in one place so the package metadata, the release manifest, and any
user-facing banner agree. ``pyproject.toml`` and the README's status line are
updated in step with this value; ``tests/test_version_consistency.py`` fails if
any of the three drifts, so "one source of truth" is enforced rather than
merely intended.
"""

from __future__ import annotations

__version__ = "2.3.0"

__all__ = ["__version__"]
