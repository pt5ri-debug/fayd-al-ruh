"""A ChatGPT-style desktop interface for Fayd Al-Ruh.

This package adds a graphical front end *alongside* the existing command-line
interface; it does not replace or modify it. ``python -m cli`` keeps working
exactly as before, and ``python -m cli gui`` (or ``python -m gui``) opens the
window.

It leans on what the project already has:

- :mod:`gui.service` builds the same text-only Jarvis session that
  ``python -m cli chat`` uses, and the same ``ModelSwitcher`` that swaps the
  answering mouth between grounded sources and the trained model.
- :mod:`gui.app` is the Tkinter window (part of the standard library, so no new
  dependency is added).
"""

from __future__ import annotations

__all__: list[str] = []
