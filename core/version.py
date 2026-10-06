"""Single source of truth for the project version.

Kept in one place so the package metadata, the release manifest, and any
user-facing banner agree. ``pyproject.toml`` is updated in step with this value.
"""

from __future__ import annotations

__version__ = "2.2.0"

__all__ = ["__version__"]
