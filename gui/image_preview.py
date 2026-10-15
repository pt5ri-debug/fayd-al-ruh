"""Shared image-preview helpers for the window's image pages.

Both the "Generate an image from text" and "Edit an image from text" pages show a
picture scaled to a preview box. The scaling is identical, so it lives here once:
Pillow is used for a smooth, exact resize when it is installed, and Tk's own
integer ``subsample`` is the dependency-free fallback so a base install still
shows a fitting preview.

Kept Tk-only (imported lazily with the window) and free of project imports, so
either page can use it without pulling in the other.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import cast


def load_scaled(path: str | Path, width: int) -> tk.PhotoImage | None:
    """Load ``path`` scaled to about ``width`` pixels, or ``None`` if unreadable.

    Pillow is used when present for a smooth, exact resize. Without it the image is
    reduced with Tk's own ``subsample`` so it still fits the preview instead of
    overflowing, which keeps a base install usable.
    """
    try:
        photo = tk.PhotoImage(file=str(path))
    except tk.TclError:
        return None
    if width <= 1 or photo.width() <= width:
        return photo
    try:
        from PIL import Image, ImageTk
    except ImportError:
        return subsample_to(photo, width)
    try:
        image = Image.open(path)
        ratio = width / max(1, image.width)
        height = max(1, round(image.height * ratio))
        resized = image.resize((width, height), Image.Resampling.LANCZOS)
        # ImageTk's image is a Tk image at runtime; its stub type is not a
        # tkinter.PhotoImage, so cast to keep the return type honest.
        return cast(tk.PhotoImage, ImageTk.PhotoImage(resized))
    except (OSError, ValueError):
        return subsample_to(photo, width)


def subsample_to(photo: tk.PhotoImage, width: int) -> tk.PhotoImage:
    """Reduce a Tk image with integer ``subsample`` until it fits ``width``."""
    factor = max(1, -(-photo.width() // max(1, width)))
    return photo.subsample(factor) if factor > 1 else photo


__all__ = ["load_scaled", "subsample_to"]
