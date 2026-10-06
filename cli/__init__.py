"""Portable command-line front end for Fayd Al-Ruh.

The package keeps the developer workflow intact (``python -m pytest``,
``python -m ml``, ``python -m evaluation``) while giving a frozen, double-click
build a single entry point and a data folder that travels next to it.
"""

from __future__ import annotations

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    from cli.__main__ import main as _main

    return _main(argv)
