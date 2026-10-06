"""Frozen-build entry point.

PyInstaller needs a plain script rather than ``python -m cli``; this calls the
same ``cli.main`` the developer command uses, so behaviour is identical.
"""

from __future__ import annotations

from cli import main

if __name__ == "__main__":
    raise SystemExit(main())
