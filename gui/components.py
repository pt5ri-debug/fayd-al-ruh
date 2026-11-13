"""The shared UI component library every window builds on.

Seventeen dialogs each grew their own ``_button``, thirteen their own
``_build_header``, and twelve their own worker-thread/queue plumbing. The code
was *nearly* identical, which is worse than different: the windows drifted by a
pixel or a font size here and there and stopped reading as one program, and each
copy of the thread-safety dance was a fresh chance to get it wrong.

This module is the one place those components live. It is **Tk-only** (it draws
real widgets), so ``cli/`` must never import it -- exactly like
:mod:`gui.chat_composer` and :mod:`gui.sidebar`. It imports nothing from
:mod:`gui.app`, so a dialog can build on it without a circular import; it takes
the duck-typed ``Theme`` and :class:`~gui.typography.Typography` it is handed.

The components are deliberately small and behavioural, not a framework:

* :class:`Button` / :func:`button` -- one flat button with named variants, so the
  seventeen copies become one call.
* :class:`Section` -- a titled card (the five ``_section`` copies).
* :class:`DialogHeader` -- the title/subtitle/spinner/window-controls bar (the
  thirteen ``_build_header`` copies).
* :class:`StatusLine` -- the dim status label (the nine ``_set_status`` copies).
* :class:`EmptyState` / :class:`LoadingState` -- the two states a pane shows
  before it has content.
* :class:`WorkerPump` -- the worker-thread + queue + UI-thread-drain pattern, in
  one place, so a window cannot get the thread-safety wrong.
* :class:`SearchInput` -- a labelled search box.
* :class:`SettingsRow` / :class:`SettingsSection` -- a labelled control row and
  the group it lives in.
* :class:`Notification` / :class:`NotificationCenter` -- a small, honest list of
  background results.

Every component takes the same ``theme``/``typography`` the other shared widgets
do, and every button clears its own ``command`` on ``<Destroy>`` for the same
reason :mod:`gui.chat_composer` does: a bound method kept in a ``command`` holds
the whole window in a cycle only the cyclic collector can break, and that can run
on a worker thread, where a Tcl call deadlocks.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from gui.design import RADIUS, SIZE, SPACE

#: Tk's anchor values, narrowed so a widget construction type-checks.
Anchor = Literal["nw", "n", "ne", "w", "center", "e", "sw", "s", "se"]

#: The button variants. ``primary`` is the one call to action on a screen (a gold
#: fill); ``secondary`` is the ordinary raised-surface button; ``ghost`` is a
#: borderless button for a toolbar; ``danger`` is the destructive one.
VARIANT_PRIMARY = "primary"
VARIANT_SECONDARY = "secondary"
VARIANT_GHOST = "ghost"
VARIANT_DANGER = "danger"
BUTTON_VARIANTS: tuple[str, ...] = (
    VARIANT_PRIMARY,
    VARIANT_SECONDARY,
    VARIANT_GHOST,
    VARIANT_DANGER,
)


def _token(theme: Any, name: str) -> str:
    """Read a theme colour, falling back to a neutral grey if it is missing.

    A component must not crash because a caller passed a partial theme (a test
    stand-in); a missing token reads as a mid grey rather than raising.
    """
    return str(getattr(theme, name, "#808080"))


class Button(tk.Button):
    """A flat, themed button with a named variant -- the one button factory.

    Replaces the seventeen hand-rolled ``_button`` helpers. The variant chooses
    the fill and the label colour; ``small`` chooses the compact font and padding
    the sidebar and the cards use. The caller's ``command`` is released on
    ``<Destroy>`` (see the module docstring), so a closed window is never kept
    alive by one of its own buttons.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        text: str,
        command: Callable[[], Any] | None,
        theme: Any,
        typography: Any,
        variant: str = VARIANT_SECONDARY,
        small: bool = False,
        anchor: Anchor = "center",
        width: int | None = None,
        padx: int | None = None,
        pady: int | None = None,
    ) -> None:
        if variant not in BUTTON_VARIANTS:
            raise ValueError(f"unknown button variant {variant!r}")
        self._command_ref: Callable[[], Any] | None = command
        fill, ink, hover = _variant_colours(theme, variant)
        role = "2xs" if small else "xs"
        super().__init__(
            parent,
            text=text,
            command=self._invoke,
            bg=fill,
            fg=ink,
            activebackground=hover,
            activeforeground=ink,
            disabledforeground=_token(theme, "text_dim"),
            font=typography.font(role, weight="bold"),
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            cursor="hand2",
            anchor=anchor,
            padx=padx if padx is not None else (10 if small else 12),
            pady=pady if pady is not None else (4 if small else 5),
        )
        if width is not None:
            self.configure(width=width)
        self._hover = hover
        self._resting = fill
        self.bind("<Enter>", self._on_enter, add="+")
        self.bind("<Leave>", self._on_leave, add="+")
        self.bind("<Destroy>", self._on_destroy, add="+")

    def _invoke(self) -> None:
        if self._command_ref is not None:
            self._command_ref()

    def _on_enter(self, _event: tk.Event) -> None:
        if str(self.cget("state")) != "disabled":
            self.configure(bg=self._hover)

    def _on_leave(self, _event: tk.Event) -> None:
        self.configure(bg=self._resting)

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is not self:
            return
        self._command_ref = None
        try:
            self.configure(command="")
        except tk.TclError:
            pass

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the control, keeping the hover state honest."""
        self.configure(state="normal" if enabled else "disabled")
        if not enabled:
            self.configure(bg=self._resting)


def _variant_colours(theme: Any, variant: str) -> tuple[str, str, str]:
    """``(fill, ink, hover)`` for a variant, from the theme's own tokens."""
    if variant == VARIANT_PRIMARY:
        return (
            _token(theme, "primary"),
            _token(theme, "primary_foreground"),
            _token(theme, "interactive_active"),
        )
    if variant == VARIANT_DANGER:
        return (
            _token(theme, "danger"),
            _token(theme, "primary_foreground"),
            _token(theme, "interactive_active"),
        )
    if variant == VARIANT_GHOST:
        return (
            _token(theme, "base"),
            _token(theme, "foreground"),
            _token(theme, "interactive_hover"),
        )
    return (
        _token(theme, "surface_raised"),
        _token(theme, "foreground"),
        _token(theme, "interactive_active"),
    )


def button(
    parent: tk.Misc,
    text: str,
    command: Callable[[], Any] | None,
    *,
    theme: Any,
    typography: Any,
    variant: str = VARIANT_SECONDARY,
    small: bool = False,
    anchor: Anchor = "center",
    **kwargs: Any,
) -> Button:
    """Build and return a :class:`Button` (the concise call form)."""
    return Button(
        parent,
        text=text,
        command=command,
        theme=theme,
        typography=typography,
        variant=variant,
        small=small,
        anchor=anchor,
        **kwargs,
    )


class Section:
    """A titled section the caller fills with rows -- the one ``_section`` factory.

    The five ``_section`` copies drew one of two shapes: a **card** (a rounded
    surface, a bold title, an inset body) or a **plain** surface frame with the
    title above the caller's rows. This keeps both, chosen by ``card``, and
    exposes the frame the caller fills as ``body``. It is a *small* wrapper, not a
    layout engine.

    ``card=True`` (default) is the character window's shape: a :class:`RoundedBox`
    card whose ``body`` is inset by the card padding. ``card=False`` is the import
    and image windows' shape: a plain surface frame with the title packed above
    ``body``, which the caller packs its own rows into.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        title: str,
        theme: Any,
        typography: Any,
        outer_bg: str | None = None,
        card: bool = True,
        padded: bool = True,
    ) -> None:
        outer = outer_bg or _token(theme, "base")
        self.card = card
        if card:
            from gui.chat_composer import RoundedBox

            self.frame = tk.Frame(parent, bg=outer)
            self.frame.pack(fill="x", padx=SPACE.xl, pady=(SPACE.sm, SPACE.xxs))
            box = RoundedBox(
                self.frame,
                bg=outer,
                fill=_token(theme, "surface"),
                outline=_token(theme, "border_subtle"),
                corner=RADIUS.card,
                parent_bg=outer,
            )
            box.pack(fill="x")
            tk.Label(
                box.inner,
                text=title,
                bg=_token(theme, "surface"),
                fg=_token(theme, "contrast"),
                font=typography.font("sm", weight="bold"),
                anchor="w",
            ).pack(anchor="w", padx=SPACE.card_pad, pady=(SPACE.card_gap, SPACE.xxs))
            self.body = tk.Frame(box.inner, bg=_token(theme, "surface"))
            self.body.pack(
                fill="x",
                padx=SPACE.card_pad if padded else 0,
                pady=(0, SPACE.lg if padded else 0),
            )
        else:
            self.frame = tk.Frame(parent, bg=_token(theme, "surface"))
            self.frame.pack(fill="x", padx=SPACE.xxl, pady=(SPACE.card_gap, 0))
            tk.Label(
                self.frame,
                text=title,
                bg=_token(theme, "surface"),
                fg=_token(theme, "contrast"),
                font=typography.font("sm", weight="bold"),
                anchor="w",
            ).pack(fill="x", padx=SPACE.lg, pady=(SPACE.card_gap, SPACE.xs))
            self.body = self.frame


class StatusLine:
    """A one-line, dim status label that can also show an error in the danger colour."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        theme: Any,
        typography: Any,
        anchor: Anchor = "w",
        role: str = "2xs",
        wraplength: int = 0,
    ) -> None:
        self._theme = theme
        self.label = tk.Label(
            parent,
            text="",
            bg=_token(theme, "surface"),
            fg=_token(theme, "text_secondary"),
            font=typography.font(role),
            anchor=anchor,
            justify="left",
            wraplength=wraplength,
        )

    def pack(self, **kwargs: Any) -> tk.Label:
        self.label.pack(**kwargs)
        return self.label

    def set(self, message: str, *, error: bool = False) -> None:
        """Show ``message``, in the danger colour when ``error`` is true."""
        self.label.configure(
            text=message,
            fg=_token(self._theme, "danger") if error else _token(self._theme, "text_secondary"),
        )

    @property
    def text(self) -> str:
        return str(self.label.cget("text"))


class DialogHeader:
    """The title/subtitle band every dialog draws, with its window controls.

    The thirteen ``_build_header`` copies all drew the same three things: a bold
    title, a dim subtitle, and the minimize/maximize/close trio on the right. This
    builds them once, installs the controls through :mod:`gui.window_controls`,
    and returns the created widgets so a window can keep its own references.

    Two optional pieces cover what the copies had *in addition* to the trio:

    * ``extra`` -- a callback that draws into the header's inner frame, so a
      window with its own header control (a Cancel button, a live title label, a
      text spinner) keeps it while the band itself stays one implementation. The
      callback receives the ``inner`` frame and may ``pack(side="right")`` into it.
    * ``cancel`` -- the ordinary "Cancel" button, packed on the right, exposed as
      :attr:`cancel_button`; :meth:`set_busy` shows it exactly while a job runs, so
      a window does not wire that dance itself.
    """

    def __init__(
        self,
        window: tk.Tk | tk.Toplevel,
        *,
        title: str,
        subtitle: str = "",
        theme: Any,
        typography: Any,
        on_close: Callable[[], None] | None = None,
        on_help: Callable[[], None] | None = None,
        spinner_length: int = SIZE.spinner,
        show_spinner: bool = True,
        extra: Callable[[tk.Frame], None] | None = None,
        cancel: Callable[[], None] | None = None,
    ) -> None:
        from tkinter import ttk

        from gui.window_controls import install_window_controls

        self.theme = theme
        self._type = typography
        bar = tk.Frame(window, bg=_token(theme, "titlebar"))
        bar.pack(fill="x")
        inner = tk.Frame(bar, bg=_token(theme, "titlebar"))
        inner.pack(fill="x", padx=SPACE.xl, pady=(SPACE.lg, SPACE.md))
        titles = tk.Frame(inner, bg=_token(theme, "titlebar"))
        titles.pack(side="left", anchor="w")
        self.title_label = tk.Label(
            titles,
            text=title,
            bg=_token(theme, "titlebar"),
            fg=_token(theme, "contrast"),
            font=typography.font("heading3", weight="bold"),
            anchor="w",
        )
        self.title_label.pack(anchor="w")
        self.subtitle_label = tk.Label(
            titles,
            text=subtitle,
            bg=_token(theme, "titlebar"),
            fg=_token(theme, "text_secondary"),
            font=typography.font("2xs"),
            anchor="w",
            justify="left",
            wraplength=820,
        )
        if subtitle:
            self.subtitle_label.pack(anchor="w", pady=(SPACE.xxs, 0))
        self.spinner: Any = None
        if show_spinner:
            self.spinner = ttk.Progressbar(inner, mode="indeterminate", length=spinner_length)
        self.cancel_button: tk.Button | None = None
        if cancel is not None:
            self.cancel_button = button(
                inner, "Cancel", cancel, theme=theme, typography=typography,
                variant="secondary", small=True,
            )
        self.controls = install_window_controls(
            window, inner, theme=theme, typography=typography, on_close=on_close,
            on_help=on_help,
        )
        self.inner = inner
        if extra is not None:
            extra(inner)
        tk.Frame(window, bg=_token(theme, "titlebar_edge"), height=1).pack(fill="x")
        self.bar = bar

    def set_busy(self, busy: bool) -> None:
        """Show or hide the busy spinner (and the Cancel button, when one exists)."""
        if busy:
            if self.spinner is not None:
                self.spinner.pack(side="right", padx=SPACE.md)
                self.spinner.start(12)
            if self.cancel_button is not None:
                self.cancel_button.pack(side="right", padx=(0, SPACE.md))
            return
        if self.spinner is not None:
            self.spinner.stop()
            self.spinner.pack_forget()
        if self.cancel_button is not None:
            self.cancel_button.pack_forget()


class HeroHeader:
    """The large "hero" title band the five form-style dialogs draw.

    A different visual variant from :class:`DialogHeader` (a ``heading1`` title
    with a subtitle and one or more dim note lines), but the same *component*: it
    builds the title row, installs the window controls, and returns them, so the
    five copies become one. ``notes`` are the dim lines under the subtitle (the
    honesty line, the accepted-formats line, the unsupported-formats line).
    """

    def __init__(
        self,
        window: tk.Tk | tk.Toplevel,
        *,
        title: str,
        subtitle: str = "",
        notes: tuple[str, ...] = (),
        theme: Any,
        typography: Any,
        on_close: Callable[[], None] | None = None,
        on_help: Callable[[], None] | None = None,
        padx: int = 20,
        wraplength: int = 980,
        extra: Callable[[tk.Frame], None] | None = None,
    ) -> None:
        from gui.window_controls import install_window_controls

        bar = tk.Frame(window, bg=_token(theme, "titlebar"))
        bar.pack(fill="x")
        header = tk.Frame(bar, bg=_token(theme, "titlebar"))
        header.pack(fill="x", padx=padx, pady=(SPACE.xl, SPACE.md))
        title_row = tk.Frame(header, bg=_token(theme, "titlebar"))
        title_row.pack(fill="x")
        self.title_label = tk.Label(
            title_row, text=title, bg=_token(theme, "titlebar"), fg=_token(theme, "contrast"),
            font=typography.font("heading1", weight="bold"), anchor="w",
        )
        self.title_label.pack(side="left")
        self.controls = install_window_controls(
            window, title_row, theme=theme, typography=typography, on_close=on_close,
            on_help=on_help,
        )
        self.subtitle_label: tk.Label | None = None
        if subtitle:
            self.subtitle_label = tk.Label(
                header, text=subtitle, bg=_token(theme, "titlebar"),
                fg=_token(theme, "text_secondary"), font=typography.font("xs"),
                anchor="w", justify="left", wraplength=wraplength,
            )
            self.subtitle_label.pack(fill="x", pady=(SPACE.xs, 0))
        self.note_labels: list[tk.Label] = []
        for note in notes:
            label = tk.Label(
                header, text=note, bg=_token(theme, "titlebar"), fg=_token(theme, "text_dim"),
                font=typography.font("2xs"), anchor="w", justify="left", wraplength=wraplength,
            )
            label.pack(fill="x", pady=(SPACE.sm, 0))
            self.note_labels.append(label)
        if extra is not None:
            extra(title_row)
        tk.Frame(window, bg=_token(theme, "titlebar_edge"), height=1).pack(fill="x")
        self.bar = bar
        self.header = header
        self.title_row = title_row


class EmptyState:
    """The centred "nothing here yet" panel a pane shows before it has content."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        title: str,
        hint: str = "",
        theme: Any,
        typography: Any,
    ) -> None:
        self.frame = tk.Frame(parent, bg=_token(theme, "base"))
        tk.Label(
            self.frame,
            text=title,
            bg=_token(theme, "base"),
            fg=_token(theme, "foreground"),
            font=typography.font("lg", weight="bold"),
            anchor="center",
            justify="center",
        ).pack(pady=(SPACE.xl, SPACE.xs))
        if hint:
            tk.Label(
                self.frame,
                text=hint,
                bg=_token(theme, "base"),
                fg=_token(theme, "text_dim"),
                font=typography.font("sm"),
                anchor="center",
                justify="center",
                wraplength=520,
            ).pack(pady=(0, SPACE.xl))

    def pack(self, **kwargs: Any) -> tk.Frame:
        self.frame.pack(**kwargs)
        return self.frame

    def destroy(self) -> None:
        self.frame.destroy()


class LoadingState:
    """A small indeterminate progress bar with a caption, for a pane that is busy."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        caption: str = "Working…",
        theme: Any,
        typography: Any,
    ) -> None:
        from tkinter import ttk

        self.frame = tk.Frame(parent, bg=_token(theme, "base"))
        self.bar = ttk.Progressbar(self.frame, mode="indeterminate", length=SIZE.spinner)
        self.bar.pack(pady=(SPACE.xl, SPACE.xs))
        self.caption = tk.Label(
            self.frame,
            text=caption,
            bg=_token(theme, "base"),
            fg=_token(theme, "text_secondary"),
            font=typography.font("sm"),
        )
        self.caption.pack()

    def start(self) -> None:
        self.bar.start(12)

    def stop(self) -> None:
        self.bar.stop()

    def pack(self, **kwargs: Any) -> tk.Frame:
        self.frame.pack(**kwargs)
        return self.frame

    def destroy(self) -> None:
        self.bar.stop()
        self.frame.destroy()


class SearchInput:
    """A labelled search box with a live ``on_change`` callback.

    The box is a plain ``tk.Entry``; the label is optional. The callback fires on
    every keystroke, so a caller redraws a list as the user types.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        theme: Any,
        typography: Any,
        label: str = "",
        placeholder: str = "",
        on_change: Callable[[str], None] | None = None,
        width: int = 20,
    ) -> None:
        self.frame = tk.Frame(parent, bg=_token(theme, "surface"))
        if label:
            tk.Label(
                self.frame,
                text=label,
                bg=_token(theme, "surface"),
                fg=_token(theme, "text_secondary"),
                font=typography.font("2xs", weight="bold"),
                anchor="w",
            ).pack(anchor="w")
        self.variable = tk.StringVar(value="")
        self.entry = tk.Entry(
            self.frame,
            textvariable=self.variable,
            width=width,
            bg=_token(theme, "surface_raised"),
            fg=_token(theme, "foreground"),
            insertbackground=_token(theme, "primary"),
            relief="flat",
            font=typography.font("sm"),
            highlightthickness=0,
        )
        self.entry.pack(fill="x", ipady=5, pady=(SPACE.xxs, 0))
        self._placeholder = placeholder
        self._on_change = on_change
        if on_change is not None:
            self.variable.trace_add("write", lambda *_: on_change(self.variable.get()))

    def pack(self, **kwargs: Any) -> tk.Frame:
        self.frame.pack(**kwargs)
        return self.frame

    @property
    def query(self) -> str:
        return self.variable.get().strip()

    def clear(self) -> None:
        self.variable.set("")


class SettingsRow:
    """One labelled setting: a bold label, an optional hint, and the caller's control."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        label: str,
        theme: Any,
        typography: Any,
        hint: str = "",
    ) -> None:
        self.frame = tk.Frame(parent, bg=_token(theme, "surface"))
        self.frame.pack(fill="x", padx=SPACE.lg, pady=SPACE.xs)
        self.label = tk.Label(
            self.frame,
            text=label,
            bg=_token(theme, "surface"),
            fg=_token(theme, "foreground"),
            font=typography.font("xs", weight="bold"),
            anchor="w",
        )
        self.label.pack(side="left")
        self.control = tk.Frame(self.frame, bg=_token(theme, "surface"))
        self.control.pack(side="right")
        self.hint: tk.Label | None = None
        if hint:
            self.hint = tk.Label(
                self.frame,
                text=hint,
                bg=_token(theme, "surface"),
                fg=_token(theme, "text_dim"),
                font=typography.font("2xs"),
                anchor="w",
                justify="left",
                wraplength=420,
            )
            self.hint.pack(side="left", padx=(SPACE.md, 0))


class SettingsSection:
    """A named group of :class:`SettingsRow`s under a header."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        title: str,
        theme: Any,
        typography: Any,
        summary: str = "",
    ) -> None:
        self.frame = tk.Frame(parent, bg=_token(theme, "surface"))
        self.frame.pack(fill="x", pady=(SPACE.md, 0))
        tk.Label(
            self.frame,
            text=title,
            bg=_token(theme, "surface"),
            fg=_token(theme, "contrast"),
            font=typography.font("sm", weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=SPACE.lg, pady=(SPACE.sm, 0))
        if summary:
            tk.Label(
                self.frame,
                text=summary,
                bg=_token(theme, "surface"),
                fg=_token(theme, "text_dim"),
                font=typography.font("2xs"),
                anchor="w",
                justify="left",
                wraplength=560,
            ).pack(fill="x", padx=SPACE.lg, pady=(0, SPACE.xs))
        self.body = tk.Frame(self.frame, bg=_token(theme, "surface"))
        self.body.pack(fill="x")

    def row(self, label: str, *, hint: str = "", theme: Any, typography: Any) -> SettingsRow:
        """Add and return a :class:`SettingsRow` to this section."""
        return SettingsRow(self.body, label=label, theme=theme, typography=typography, hint=hint)


# --------------------------------------------------------------- notifications


@dataclass(frozen=True)
class Notification:
    """One background result a window wants to show: what happened and how bad."""

    title: str
    detail: str = ""
    level: str = "info"  # info | success | warning | error
    at: float = 0.0


#: The notification levels, in the order a reader should weight them.
NOTIFICATION_LEVELS: tuple[str, ...] = ("info", "success", "warning", "error")


class NotificationCenter:
    """A small, bounded list of background results, drawn into a parent frame.

    It holds *what actually happened* -- a finished training run, a failed
    download -- and never a fabricated progress line. The list is bounded, so a
    long session cannot grow it without limit, and the newest is drawn first.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        theme: Any,
        typography: Any,
        limit: int = 50,
    ) -> None:
        self.theme = theme
        self._type = typography
        self._limit = max(1, int(limit))
        self._items: list[Notification] = []
        self.frame = tk.Frame(parent, bg=_token(theme, "surface"))
        self.frame.pack(fill="x")
        self._list = tk.Frame(self.frame, bg=_token(theme, "surface"))
        self._list.pack(fill="x")
        self._empty = tk.Label(
            self._list,
            text="No notifications yet.",
            bg=_token(theme, "surface"),
            fg=_token(theme, "text_dim"),
            font=typography.font("2xs"),
            anchor="w",
        )
        self._empty.pack(fill="x", padx=SPACE.lg, pady=SPACE.xs)

    @property
    def items(self) -> tuple[Notification, ...]:
        return tuple(self._items)

    def add(self, notification: Notification) -> None:
        """Record a notification and redraw the list (newest first)."""
        if notification.level not in NOTIFICATION_LEVELS:
            raise ValueError(f"unknown notification level {notification.level!r}")
        self._items.insert(0, notification)
        del self._items[self._limit :]
        self._redraw()

    def clear(self) -> None:
        self._items.clear()
        self._redraw()

    def _redraw(self) -> None:
        for child in self._list.winfo_children():
            child.destroy()
        if not self._items:
            self._empty = tk.Label(
                self._list,
                text="No notifications yet.",
                bg=_token(self.theme, "surface"),
                fg=_token(self.theme, "text_dim"),
                font=self._type.font("2xs"),
                anchor="w",
            )
            self._empty.pack(fill="x", padx=SPACE.lg, pady=SPACE.xs)
            return
        colours = {
            "info": _token(self.theme, "text_secondary"),
            "success": _token(self.theme, "success"),
            "warning": _token(self.theme, "primary"),
            "error": _token(self.theme, "danger"),
        }
        for item in self._items:
            row = tk.Frame(self._list, bg=_token(self.theme, "surface"))
            row.pack(fill="x", padx=SPACE.lg, pady=1)
            tk.Label(
                row,
                text="•",
                bg=_token(self.theme, "surface"),
                fg=colours[item.level],
                font=self._type.font("sm", weight="bold"),
            ).pack(side="left", padx=(0, SPACE.xs))
            text = item.title if not item.detail else f"{item.title} — {item.detail}"
            tk.Label(
                row,
                text=text,
                bg=_token(self.theme, "surface"),
                fg=colours[item.level],
                font=self._type.font("2xs"),
                anchor="w",
                justify="left",
                wraplength=560,
            ).pack(side="left", fill="x", expand=True)


# ------------------------------------------------------------------- worker


class WorkerPump:
    """The worker-thread + queue + UI-thread-drain pattern, in one place.

    Twelve dialogs wrote this dance themselves, and each copy was a fresh chance
    to touch Tk from a worker thread (which Tk forbids). This class owns the one
    correct shape:

    * :meth:`start` runs ``job`` on a daemon worker thread;
    * the result (or the exception) is put on a :class:`queue.Queue`;
    * the UI thread drains that queue on its own clock through ``after``, never
      from the worker;
    * :meth:`cancel` sets a cooperative flag the job can read, and drops a late
      result rather than delivering it to a closed window.

    The caller supplies ``finish(result, error)``, which runs on the UI thread.
    """

    #: How often the UI thread checks the queue (milliseconds).
    INTERVAL_MS = 60

    def __init__(
        self,
        window: tk.Misc,
        *,
        on_finish: Callable[[Any, Exception | None], None],
        on_busy: Callable[[bool], None] | None = None,
    ) -> None:
        self._window = window
        self._on_finish = on_finish
        self._on_busy = on_busy
        self._results: queue.Queue[tuple[Any, Exception | None]] = queue.Queue()
        self._busy = False
        self._cancelled = False
        self._ticket: str | None = None

    @property
    def busy(self) -> bool:
        return self._busy

    def start(self, job: Callable[[], Any]) -> None:
        """Run ``job`` on a worker thread and schedule the UI-thread drain."""
        if self._busy:
            return
        self._busy = True
        self._cancelled = False
        if self._on_busy is not None:
            self._on_busy(True)

        def run() -> None:
            try:
                result = job()
                error: Exception | None = None
            except Exception as exc:  # reported to the caller, never swallowed
                result, error = None, exc
            self._results.put((result, error))

        threading.Thread(target=run, daemon=True).start()
        self._schedule()

    def cancel(self) -> None:
        """Ask the running job to stop (cooperative; the job checks the flag)."""
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def _schedule(self) -> None:
        try:
            self._ticket = self._window.after(self.INTERVAL_MS, self._drain)
        except tk.TclError:
            self._ticket = None

    def _drain(self) -> None:
        self._ticket = None
        try:
            result, error = self._results.get_nowait()
        except queue.Empty:
            if self._busy:
                self._schedule()
            return
        self._busy = False
        if self._on_busy is not None:
            self._on_busy(False)
        if self._cancelled:
            # The user cancelled: the late result is dropped, never delivered to a
            # window that has moved on.
            self._cancelled = False
            return
        self._on_finish(result, error)

    def close(self) -> None:
        """Cancel a pending ``after`` ticket so a closed window is not called into."""
        if self._ticket is not None:
            try:
                self._window.after_cancel(self._ticket)
            except tk.TclError:
                pass
            self._ticket = None
        self._busy = False


__all__ = [
    "BUTTON_VARIANTS",
    "NOTIFICATION_LEVELS",
    "VARIANT_DANGER",
    "VARIANT_GHOST",
    "VARIANT_PRIMARY",
    "VARIANT_SECONDARY",
    "Button",
    "DialogHeader",
    "EmptyState",
    "HeroHeader",
    "LoadingState",
    "Notification",
    "NotificationCenter",
    "SearchInput",
    "Section",
    "SettingsRow",
    "SettingsSection",
    "StatusLine",
    "WorkerPump",
    "button",
]
