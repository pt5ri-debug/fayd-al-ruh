"""The message footer every chat-style window shows under an answer.

The main chat window draws a small row beneath each message: a **Copy** control
and a **Read aloud / Stop** toggle, both selectable from the transcript. That is
the behaviour the user asked to see *everywhere* -- in the seven per-model chat
windows and in each speech-engine studio -- so the wording, the styling, and the
toggle semantics live here once and every window installs them, instead of each
one growing its own copy that drifts.

Two pieces:

* :func:`strip_markdown` -- the same plain-prose pass the main window speaks, so
  an engine reads words rather than ``**`` and backticks. It is the shared home
  for the rule that used to live only in :mod:`gui.app`.
* :class:`MessageFooter` -- the footer row and its two controls. It is
  presentation only: it never decides what a message means. It calls back into
  the window for the two actions (copy the widget's text, toggle the reading) so
  a window keeps its own playback, voice, and transcript wiring.

The reading control follows the *one reading at a time* rule: the window tells
the footer which control is currently playing (through :meth:`set_playing`) and
the footer paints that one ``⏹ Stop`` and every other ``🔊 Read aloud``.
"""

from __future__ import annotations

import re
import tkinter as tk
from collections.abc import Callable
from typing import Any, Literal

#: The two labels the reading control alternates between.
READ_LABEL = "🔊 Read aloud"
STOP_LABEL = "⏹ Stop"
COPY_LABEL = "⧉ Copy"

#: The Tk pack sides a footer's controls may take.
Side = Literal["left", "right", "top", "bottom"]


def strip_markdown(text: str) -> str:
    """Turn rendered Markdown back into plain prose, so a reader speaks words.

    A speech engine should not read ``**``, backticks, or a list bullet aloud, so
    the markers are removed while the words and the code are kept. The rule is the
    main window's own, shared here so every window reads aloud the same way.
    """
    text = re.sub(r"```[^\n]*\n?", "", text)          # fenced code markers
    text = re.sub(r"`([^`]*)`", r"\1", text)          # inline code
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)  # headings
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)       # bullets
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)     # ordered lists
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)           # block quotes
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)               # links -> label
    text = re.sub(r"(\*\*|__|\*|_|~~)", "", text)                      # emphasis
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)  # rules
    return text


def add_provenance_label(
    parent: tk.Misc,
    provenance: str,
    *,
    theme: Any,
    typography: Any,
) -> tk.Label | None:
    """Paint one dim "what answered / what it rested on" line, or nothing.

    A message's provenance is two short lines (``core.answer_provenance``); this
    renders them as a single small dim label above the message's controls, in the
    window's own theme, so a window that does not go through the turn formatter
    still states the model and the sources. An empty ``provenance`` adds no
    widget, so a control or a tool run carries no line rather than a blank one.
    """
    if not provenance.strip():
        return None
    label = tk.Label(
        parent,
        text=provenance,
        bg=theme.base,
        fg=theme.text_dim,
        font=typography.font("2xs"),
        anchor="w",
        justify="left",
        wraplength=720,
    )
    label.pack(side="top", fill="x", anchor="w")
    return label


class MessageFooter:
    """A message's footer row: a Copy control and a Read aloud / Stop toggle.

    ``parent`` is the frame the controls are packed into (the window owns it).
    ``text`` is the message's own text, and ``widget`` the ``tk.Text`` that holds
    it (what Copy selects from). ``on_toggle`` is called with the footer when the
    reading control is clicked; the window starts or stops the reading and then
    calls :meth:`set_playing` to paint the right label. ``theme``/``typography``
    style the controls the way every other window does.

    ``provenance`` is the optional "what answered this, and what it rested on"
    line (``core.answer_provenance``). When given, it is painted as a small dim
    line above the controls, so a window that does not render its answers through
    the turn formatter still says which model produced the message and which
    sources it used -- the same fact every chat window shows, in one place.

    The two controls are ``tk.Label``s bound to ``<Button-1>`` with a hover tint,
    exactly like the main window's, so a footer reads the same in every window.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        text: str,
        widget: tk.Text,
        theme: Any,
        typography: Any,
        on_toggle: Callable[[MessageFooter], None],
        side: Side = "left",
        copy_side: Side | None = None,
        provenance: str = "",
    ) -> None:
        self.theme = theme
        self.typography = typography
        self.text = text
        self.widget = widget
        self._on_toggle = on_toggle
        self._playing = False

        self.frame = tk.Frame(parent, bg=theme.base)
        self.frame.pack(fill="x", pady=(2, 0))

        self.provenance = add_provenance_label(
            self.frame, provenance, theme=theme, typography=typography
        )

        self.copy = tk.Label(
            self.frame, text=COPY_LABEL, bg=theme.base, fg=theme.text_dim,
            font=typography.font("2xs"), cursor="hand2",
        )
        self.copy.pack(side=copy_side or side, padx=(0 if copy_side else 10, 0))
        self.copy.bind("<Button-1>", lambda _e: self._copy())
        self.copy.bind("<Enter>", lambda _e: self.copy.configure(fg=theme.foreground))
        self.copy.bind("<Leave>", lambda _e: self.copy.configure(fg=theme.text_dim))

        self.speaker: tk.Label | None = None
        if text.strip():
            speaker = tk.Label(
                self.frame, text=READ_LABEL, bg=theme.base, fg=theme.text_dim,
                font=typography.font("2xs"), cursor="hand2",
            )
            self.speaker = speaker
            speaker.pack(side=side, padx=(10, 0))
            speaker.bind("<Button-1>", lambda _e: self._on_toggle(self))
            speaker.bind("<Enter>", lambda _e: speaker.configure(fg=theme.foreground))
            speaker.bind("<Leave>", lambda _e: self._speaker_leave())

    # -------------------------------------------------------------- controls

    def _copy(self) -> None:
        """Copy this message's text to the clipboard (selection first)."""
        try:
            selected = self.widget.get("sel.first", "sel.last")
        except tk.TclError:
            selected = ""
        body = selected or self.text
        self.widget.clipboard_clear()
        self.widget.clipboard_append(body)

    def _speaker_leave(self) -> None:
        if self.speaker is not None:
            self.speaker.configure(fg=self.theme.foreground if self._playing else self.theme.text_dim)

    @property
    def playing(self) -> bool:
        """True while this footer's message is the one being read aloud."""
        return self._playing

    def set_playing(self, playing: bool) -> None:
        """Paint the reading control as playing (``⏹ Stop``) or idle."""
        self._playing = playing
        if self.speaker is None:
            return
        with _suppress():
            self.speaker.configure(
                text=STOP_LABEL if playing else READ_LABEL,
                fg=self.theme.foreground if playing else self.theme.text_dim,
            )

    def clear_commands(self) -> None:
        """Drop the bound callbacks (called when the window is destroyed).

        A destroyed window that still owns a bound command sits in a reference
        cycle only the cyclic collector can break, and that can run on a worker
        thread where a Tcl call deadlocks -- the same hazard ``gui.chat_composer``
        guards against. Clearing the commands here lets the window die at once.
        """
        for control in (self.copy, self.speaker):
            if control is not None:
                with _suppress():
                    control.unbind("<Button-1>")


def _suppress():
    import contextlib

    return contextlib.suppress(tk.TclError)


__all__ = [
    "COPY_LABEL",
    "READ_LABEL",
    "STOP_LABEL",
    "MessageFooter",
    "add_provenance_label",
    "strip_markdown",
]
