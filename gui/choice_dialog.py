"""A small modal dialog whose field is an *editable* dropdown.

Several Training Center controls used to ask for a name in a plain text box --
*which provider?*, *which mouth?*, *which character?*, *the dataset id*, *the run
id*, *the setting key* -- so a user had to remember the exact spelling and could
type a value the engine would reject. This module replaces those with one control
that offers the real choices **and** still accepts a typed value, which is what
the project's other prompts (``simpledialog.askstring``) cannot do.

Tk has no widget that is at once a dropdown and a free-text field, so the dialog
composes one: an editable :class:`ttk.Combobox` (its ``values`` are the readable
labels, so the list reads like a menu) inside a :class:`simpledialog.Dialog`,
whose OK/Enter reads whatever text is in the box. The value returned is the
picked row's *value*, or -- when the user typed something the list does not hold
-- exactly what was typed, so a name that is not yet in the list (a dataset the
user is about to build, a run id from another machine) is still accepted.

It is honest about the two halves: a labelled row maps back to its value, a bare
value the list holds is returned as itself, and anything else is returned
verbatim. Nothing is invented and nothing is silently rejected.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Sequence
from tkinter import simpledialog, ttk
from typing import Any


def normalise_choices(
    values: Sequence[str | tuple[str, str]],
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    """Split ``values`` into ``(rows, displays)``.

    A row is ``(value, label)``; a plain string is its own value *and* its own
    label. ``displays`` is what the dropdown shows, in order, de-duplicated so a
    repeated label never shadows a value.
    """
    rows: list[tuple[str, str]] = []
    displays: list[str] = []
    seen: set[str] = set()
    for entry in values:
        if isinstance(entry, tuple):
            value, label = entry
        else:
            value = label = entry
        value = str(value)
        label = str(label) if label else value
        rows.append((value, label))
        if label not in seen:
            displays.append(label)
            seen.add(label)
    return tuple(rows), tuple(displays)


def resolve_choice(text: str, rows: Sequence[tuple[str, str]]) -> str:
    """Turn what is in the box back into a value.

    A known label maps to its value; a bare value is returned as itself; any
    other text is returned verbatim, so a name that is not in the list yet is
    still usable. This is the *one* place the mapping is decided, so the dialog
    and its tests can never disagree.
    """
    text = text.strip()
    for value, label in rows:
        if label == text:
            return value
    return text


def display_for(value: str, rows: Sequence[tuple[str, str]]) -> str:
    """The label to show for ``value`` (or the value itself when unknown)."""
    if not value:
        return ""
    for candidate, label in rows:
        if candidate == value:
            return label
    return value


# Backwards-compatible private alias (kept so an existing caller keeps working).
_normalise = normalise_choices


class ChoiceDialog(simpledialog.Dialog):
    """A modal dialog with one editable dropdown.

    ``values`` is a sequence of ``(value, label)`` pairs (or plain strings); the
    dropdown shows the labels and returns the matching value. ``initial`` presets
    the box: it is looked up as a value first, then a label, then shown as typed.
    """

    def __init__(
        self,
        parent: tk.Misc,
        title: str,
        prompt: str,
        *,
        values: Sequence[str | tuple[str, str]] = (),
        initial: str = "",
        theme: Any = None,
        typography: Any = None,
    ) -> None:
        self._prompt = prompt
        self._rows, self._displays = normalise_choices(values)
        self._initial = initial
        self._theme = theme
        self._type = typography
        #: The chosen (or typed) value, or ``None`` when the dialog was cancelled.
        self.result: str | None = None
        super().__init__(parent, title)

    # -- simpledialog.Dialog API ------------------------------------------

    def body(self, master: tk.Frame) -> tk.Widget:
        bg = getattr(self._theme, "surface", None) or master.cget("bg")
        fg = getattr(self._theme, "foreground", None) or "black"
        master.configure(bg=bg)
        font = self._type.font("body") if self._type is not None else None
        tk.Label(
            master, text=self._prompt, bg=bg, fg=fg, justify="left", anchor="w",
            wraplength=440, **({"font": font} if font else {}),
        ).pack(fill="x", pady=(0, 6))
        self._var = tk.StringVar(value=display_for(self._initial, self._rows))
        self._box = ttk.Combobox(
            master, textvariable=self._var, values=list(self._displays),
            **({"font": font} if font else {}),
        )
        self._box.pack(fill="x")
        self._box.select_range(0, "end")
        return self._box

    def validate(self) -> bool:
        # The field is editable on purpose, so any text (including a value the
        # list does not hold) is accepted; an empty box is a cancel.
        return True

    def apply(self) -> None:
        self.result = resolve_choice(self._var.get(), self._rows)


def ask_choice(
    parent: tk.Misc,
    title: str,
    prompt: str,
    *,
    values: Sequence[str | tuple[str, str]] = (),
    initial: str = "",
    theme: Any = None,
    typography: Any = None,
) -> str:
    """Ask for one value with an editable dropdown; ``""`` when cancelled.

    This is the dropdown counterpart of :func:`tkinter.simpledialog.askstring`,
    so a caller swaps one call for the other and gains the list of real choices
    while keeping the ability to type a value.
    """
    dialog = ChoiceDialog(
        parent, title, prompt, values=values, initial=initial, theme=theme, typography=typography,
    )
    return dialog.result or ""


__all__ = ["ChoiceDialog", "ask_choice"]
