"""Argument parsing for the GUI command.

Kept in its own module so ``python -m cli gui --help``-style introspection, and
the tests, never need Tkinter or a display just to read the options.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """The GUI command's arguments, mirroring the ``chat`` command's options."""
    parser = argparse.ArgumentParser(
        prog="gui",
        description="Open the ChatGPT-style graphical interface for Fayd Al-Ruh.",
    )
    parser.add_argument("--root", default=None, help="sandbox root directory for tools")
    parser.add_argument("--read-write", action="store_true", help="allow writing tools in the sandbox")
    parser.add_argument("--no-web", action="store_true", help="answer only from internal knowledge")
    parser.add_argument("--mind", action="store_true", help="start answering from the trained model")
    parser.add_argument("--model-checkpoint", default=None, help="path to a language-model checkpoint")
    parser.add_argument(
        "--ultron",
        action="store_true",
        help="start answering from Ultron, the general local model (any subject)",
    )
    parser.add_argument("--ultron-model", default=None, help="a specific .gguf file for Ultron to load")
    parser.add_argument(
        "--vision",
        action="store_true",
        help="start answering from Vision, the petroleum local model (reservoir, drilling, PVT)",
    )
    parser.add_argument("--vision-model", default=None, help="a specific .gguf file for Vision to load")
    parser.add_argument(
        "--banner",
        action="store_true",
        help="start answering from Banner, the presentation local model (outlines and slide content)",
    )
    parser.add_argument("--banner-model", default=None, help="a specific .gguf file for Banner to load")
    parser.add_argument("--session", default=None, help="resume a saved session by id")
    parser.add_argument(
        "--mode",
        default=None,
        choices=("ai", "jarvics"),
        help="start in 'ai' mode (conversation only) or 'jarvics' mode (conversation plus sandboxed tools)",
    )
    return parser


__all__ = ["build_parser"]
