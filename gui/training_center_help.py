"""The Training Center's **Help** window: everything, explained, in Arabic and English.

Reached from the **?** control beside the window's minimize / maximize / close
buttons (see :func:`gui.window_controls.install_window_controls`), this window
answers the three questions a reader actually has before pressing anything:

* **What is this place for?** -- the Training Center in two paragraphs.
* **How does it work, in order?** -- the six steps from a source to a trained,
  measured model, so a newcomer knows where to start and what follows what.
* **What does this control do?** -- every section and all sixty-four controls,
  each with its one-line summary, what it returns, whether it asks for
  confirmation, whether it writes, and the equivalent command.

It is built the way every other window in the project is built -- the same
:class:`~gui.app.Theme` tokens, the shared :class:`~gui.rich_text` Markdown
painter, and the three :class:`~gui.window_controls` title-bar buttons -- and its
text is **generated from the catalogue** (:mod:`core.training_center.help`), so it
can never describe a control differently from the window that draws it. A
language switch toggles the whole window between Arabic and English; the Arabic
page is right-aligned, because that is how Arabic is read.
"""

from __future__ import annotations

import re
import tkinter as tk
from collections.abc import Callable
from typing import Any

from core.training_center.help import LANGUAGES, help_text, help_title, resolve_language

from gui.dialog_base import ThemedDialog
from gui.typography import resolve
from gui.window_controls import fit_geometry

DEFAULT_WIDTH = 900
DEFAULT_HEIGHT = 760
MIN_WIDTH = 640
MIN_HEIGHT = 480

_LANGUAGE_LABELS: dict[str, str] = {"ar": "العربية", "en": "English"}


class TrainingCenterHelpDialog(ThemedDialog):
    """The Training Center's bilingual explanation window."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        *,
        theme: Any,
        font: str,
        mono: str,
        overrides: object | None = None,
        list_overrides: object | None = None,
        language: str = "ar",
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self.theme = theme
        self.font = font
        self.mono = mono
        self._type = resolve(parent, ui_family=font, mono_family=mono, overrides=overrides)
        self._overrides = overrides
        self._list_overrides = list_overrides
        self._on_close = on_close
        self._language = resolve_language(language)

        self.window = tk.Toplevel(parent)
        self.window.title(help_title(self._language))
        self.window.configure(bg=theme.base)
        self.window.geometry(
            fit_geometry(self.window, DEFAULT_WIDTH, DEFAULT_HEIGHT, min_height=MIN_HEIGHT)
        )
        self.window.minsize(MIN_WIDTH, MIN_HEIGHT)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)

        self._themed_build()

    # -- the window is rebuilt on a language switch, so it is the build that reads
    #    the language rather than a hand-written redraw.

    def _after_build(self) -> None:
        self._render_help()

    def _build(self) -> None:
        theme = self.theme
        bar = tk.Frame(self.window, bg=theme.titlebar)
        bar.pack(fill="x")
        inner = tk.Frame(bar, bg=theme.titlebar)
        inner.pack(fill="x", padx=16, pady=(12, 10))
        self._title_label = tk.Label(
            inner, text=help_title(self._language), bg=theme.titlebar, fg=theme.contrast,
            font=self._type.font("heading3", weight="bold"), anchor="w",
        )
        self._title_label.pack(side="left", anchor="w")

        from gui.components import button
        from gui.window_controls import install_window_controls

        self._language_button = button(
            inner, self._language_switch_label(), self._toggle_language, theme=theme,
            typography=self._type, variant="secondary", small=True,
        )
        self._language_button.pack(side="right", padx=(0, 12))
        self._controls = install_window_controls(
            self.window, inner, theme=theme, typography=self._type, on_close=self._close
        )
        tk.Frame(self.window, bg=theme.titlebar_edge, height=1).pack(fill="x")

        body = tk.Frame(self.window, bg=theme.base)
        body.pack(fill="both", expand=True, padx=16, pady=12)
        wrap = tk.Frame(body, bg=theme.surface, highlightthickness=1,
                        highlightbackground=theme.border_subtle)
        wrap.pack(fill="both", expand=True)
        inner = tk.Frame(wrap, bg=theme.surface)
        inner.pack(fill="both", expand=True, padx=12, pady=12)
        scrollbar = tk.Scrollbar(inner, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        self._text = tk.Text(
            inner, wrap="word", bd=0, highlightthickness=0, bg=theme.surface,
            fg=theme.foreground, font=self._type.font("body"), padx=8, pady=4,
            yscrollcommand=scrollbar.set, cursor="arrow",
        )
        self._text.pack(side="left", fill="both", expand=True)
        scrollbar.configure(command=self._text.yview)
        self._text.configure(state="disabled")

    def _render_help(self) -> None:
        """Paint the current language's guide into the text area."""
        from gui.rich_text import insert_markdown

        self.window.title(help_title(self._language))
        self._title_label.configure(text=help_title(self._language))
        self._language_button.configure(text=self._language_switch_label())
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        insert_markdown(
            self._text, self._as_markdown(help_text(self._language)),
            theme=self.theme, typography=self._type,
        )
        # Arabic reads right to left, so its page is right-aligned whatever the
        # reader's own paragraph setting is; the English page follows the profile.
        justify = "right" if self._language == "ar" else self._type.justify()
        if justify != "left":
            self._text.tag_configure("_help_align", justify=justify)
            self._text.tag_add("_help_align", "1.0", "end")
        self._text.configure(state="disabled")
        self._text.yview_moveto(0.0)

    def _language_switch_label(self) -> str:
        other = "en" if self._language == "ar" else "ar"
        return f"🌐  {_LANGUAGE_LABELS[other]}"

    def _toggle_language(self) -> None:
        self._language = "en" if self._language == "ar" else "ar"
        self.retype(self._overrides, self._list_overrides)

    @staticmethod
    def _as_markdown(text: str) -> str:
        """Lightly mark the plain guide up so the shared painter styles it.

        The guide is authored as plain text (so the CLI prints it verbatim); the
        window turns the forms it uses into Markdown the shared renderer already
        understands: an ``=``-underlined title becomes ``#``, a ``-``-underlined
        heading becomes ``##``, a ``■`` section line becomes ``###``, a ``•`` line
        becomes a bullet, and a numbered flow step is emboldened. Every other line
        is emitted with its indentation removed, so nothing is read as a code
        block.
        """
        lines = text.splitlines()
        out: list[str] = []
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped and set(stripped) <= {"="} and len(stripped) >= 3:
                continue  # the underline belongs to the title above it
            if stripped and set(stripped) <= {"-"} and len(stripped) >= 3:
                continue  # the underline belongs to the heading above it
            if not stripped:
                out.append("")
                continue
            nxt = lines[index + 1].strip() if index + 1 < len(lines) else ""
            if nxt and set(nxt) <= {"="} and len(nxt) >= 3:
                out.append(f"# {stripped}")
            elif nxt and set(nxt) <= {"-"} and len(nxt) >= 3:
                out.append(f"## {stripped}")
            elif stripped.startswith("■ "):
                out.append(f"### {stripped[2:]}")
            elif stripped.startswith("• "):
                out.append(f"- {stripped[2:]}")
            elif re.match(r"^\d+[.)]\s", stripped):
                out.append(f"**{stripped}**")
            else:
                out.append(stripped)
        return "\n".join(out)

    def _close(self) -> None:
        if self._on_close is not None:
            self._on_close()
        try:
            self.window.destroy()
        except tk.TclError:
            pass


__all__ = ["DEFAULT_HEIGHT", "DEFAULT_WIDTH", "LANGUAGES", "TrainingCenterHelpDialog"]
