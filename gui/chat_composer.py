"""The shared chat composer: one input panel used by every chat-style screen.

The window, the Code Studio, the Presentation Studio (and its final editor), and
the image screens all let the user type a message and act on it. Each one grew
its own panel, so they drifted apart. This module is the one place that shape is
defined, following the OpenHands chat composer:

* a wide, generously rounded panel on the window's own surface;
* a spacious multiline text area with a muted placeholder and no visible border;
* a compact toolbar on the bottom-left (＋ attach, and any mode selectors the
  caller supplies) and the action controls on the bottom-right (a subtle status
  dot, an optional microphone, and a circular send disc).

The panel is presentation only: it never decides what a message means. The caller
wires :meth:`ChatComposer.set_submit` to its own submit handler, so every existing
integration (the chat's ``_on_send``, the Studio's ``_on_discuss``, the Code
Studio's ``_on_ask_model``) is untouched.

Nothing here imports :mod:`gui.app`, so the window can build on this module
without a circular import; the panel only uses the duck-typed ``Theme`` tokens
and :class:`gui.typography.Typography` it is handed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import weakref

import tkinter as tk

from gui.typography import Typography, for_root

#: The HeroUI radius pair the rest of the window uses (``gui.app``).
RADIUS_SMALL = 5
RADIUS_LARGE = 20

#: The send control is a disc, so its side length is a single diameter in pixels.
SEND_BUTTON_SIZE = 36

#: The composer grows with the text up to this many lines, then scrolls inside
#: itself, so a long message never pushes the surrounding layout around.
DEFAULT_MAX_LINES = 12

#: The smallest height (in text lines) the input area keeps, so the panel reads
#: as a multi-line composer from the first keystroke.
DEFAULT_MIN_LINES = 2


def _blend(base: str, other: str, amount: float) -> str:
    """Mix ``base`` toward ``other`` by ``amount`` (0..1), returning ``#rrggbb``.

    Used for the send disc's hover, so the lightened shade is derived from the
    theme's own primary rather than a second hard-coded colour. A value that is
    not a plain ``#rrggbb`` string is returned unchanged, so a named or
    ``#rgb`` colour can never make the control disappear on hover.
    """
    if not (isinstance(base, str) and base.startswith("#") and len(base) == 7):
        return base
    try:
        base_rgb = tuple(int(base[i:i + 2], 16) for i in (1, 3, 5))
        other_rgb = tuple(int(other[i:i + 2], 16) for i in (1, 3, 5))
    except ValueError:
        return base
    mixed = tuple(round(b + (o - b) * amount) for b, o in zip(base_rgb, other_rgb))
    return "#" + "".join(f"{channel:02X}" for channel in mixed)


class RoundedBox(tk.Canvas):
    """A canvas that paints a rounded-rectangle surface, then holds real widgets.

    Tk has no corner radius, so a rounded panel (the composer, a user bubble) is
    a Canvas that draws the rounded fill and border and hosts its children in an
    interior frame placed with :meth:`place`. ``corner`` is the HeroUI radius
    (5 for small controls, 20 for the composer and message cards).
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        bg: str,
        fill: str,
        outline: str,
        corner: int = RADIUS_SMALL,
        width_pad: int = 24,
        height_pad: int = 8,
        parent_bg: str | None = None,
    ) -> None:
        super().__init__(
            parent, bg=parent_bg or bg, highlightthickness=0, borderwidth=0,
            takefocus=0,
        )
        self._fill = fill
        self._outline = outline
        self._corner = corner
        self._pad_x = width_pad // 2
        self._pad_y = height_pad // 2
        self._shape: int | None = None
        self.inner = tk.Frame(self, bg=fill)
        # relwidth/height fill the canvas exactly, with the padding as an offset,
        # so the interior can never spill past the rounded background.
        self.inner.place(
            x=self._pad_x, y=self._pad_y, relwidth=1, width=-self._pad_x * 2
        )
        self.inner.bind("<Configure>", self._sync_height)
        self.bind("<Configure>", self._redraw)

    def _sync_height(self, _event: tk.Event | None = None) -> None:
        """Match the canvas height to its content, growing with wrapped text."""
        wanted = self.inner.winfo_reqheight() + self._pad_y * 2
        if wanted > 2 and self.winfo_reqheight() != wanted:
            self.configure(height=wanted)

    def _redraw(self, _event: tk.Event | None = None) -> None:
        self.delete("shape")
        w, h = self.winfo_width(), self.winfo_height()
        if w <= 2 or h <= 2:
            return
        self._shape = self._rounded(1, 1, w - 1, h - 1, self._corner)
        # Draw first, then push it behind the interior frame.
        self.tag_lower(self._shape)

    def _rounded(self, x1: int, y1: int, x2: int, y2: int, r: int) -> int:
        points = [
            x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
            x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
            x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
        ]
        return self.create_polygon(
            points, smooth=True, splinesteps=24, tags=("shape",),
            fill=self._fill, outline=self._outline, width=1,
        )


class CircleButton(tk.Canvas):
    """A circular button: a canvas-painted disc that still behaves like a button.

    Tk's ``Button`` cannot be round, so the send control is a Canvas that draws a
    filled circle and a glyph, and forwards clicks to a command. ``enabled`` (and
    the fill/ink) can be changed later without rebuilding it, which is how the
    window greys the control out while it is busy.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        size: int,
        bg: str,
        fill: str,
        ink: str,
        text: str,
        command: Callable[[], Any],
        typography: Typography | None = None,
    ) -> None:
        super().__init__(
            parent, width=size, height=size, bg=bg, highlightthickness=0,
            borderwidth=0, takefocus=0, cursor="hand2",
        )
        self._size = size
        self._fill = fill
        self._ink = ink
        self._command: Callable[[], Any] | None = command
        self._enabled = True
        self._shape = self.create_oval(1, 1, size - 1, size - 1, fill=fill, outline="")
        scale = typography or for_root(None)
        self._glyph = self.create_text(
            size // 2, size // 2, text=text, fill=ink,
            font=scale.font("lg", weight="bold"),
        )
        self.bind("<Button-1>", self._on_click)
        # A hover lightens the disc slightly (only while it is enabled), so the
        # primary action answers the pointer without a heavy ring or a size change.
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        # ``_command`` is a plain attribute, so Tk's own destroy never drops it: a
        # bound method there would keep the caller in a reference cycle that only
        # the cyclic collector could break -- and that can run on a worker thread,
        # where a Tcl call deadlocks. Clear it with the widget.
        self.bind("<Destroy>", self._on_destroy, add="+")

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is self:
            self._command = None

    def _on_click(self, _event: tk.Event) -> None:
        if self._enabled and self._command is not None:
            self._command()

    def _on_enter(self, _event: tk.Event) -> None:
        if self._enabled:
            self.itemconfigure(self._shape, fill=_blend(self._fill, "#FFFFFF", 0.16))

    def _on_leave(self, _event: tk.Event) -> None:
        self.itemconfigure(self._shape, fill=self._fill)

    def set_state(self, *, enabled: bool, fill: str, ink: str) -> None:
        """Repaint the disc for an enabled/disabled state."""
        self._enabled = enabled
        self._fill = fill
        self._ink = ink
        self.itemconfigure(self._shape, fill=fill)
        self.itemconfigure(self._glyph, fill=ink)
        self.configure(cursor="hand2" if enabled else "arrow")


class Tooltip:
    """A small hover hint for a control, shown after a short delay."""

    def __init__(self, widget: tk.Widget, text: str, theme: Any, typography: Typography) -> None:
        self._widget = widget
        self._text = text
        self._theme = theme
        self._type = typography
        self._window: tk.Toplevel | None = None
        self._after: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event: tk.Event) -> None:
        self._after = self._widget.after(450, self._show)

    def _show(self) -> None:
        if self._window is not None:
            return
        window = tk.Toplevel(self._widget)
        window.wm_overrideredirect(True)
        window.configure(bg=self._theme.border)
        tk.Label(
            window, text=self._text, bg=self._theme.surface, fg=self._theme.foreground,
            font=self._type.font("xs"), padx=8, pady=4, relief="flat",
        ).pack()
        x = self._widget.winfo_rootx() + 4
        y = self._widget.winfo_rooty() - 32
        window.wm_geometry(f"+{max(x, 0)}+{max(y, 0)}")
        self._window = window

    def _hide(self, _event: tk.Event | None = None) -> None:
        if self._after is not None:
            self._widget.after_cancel(self._after)
            self._after = None
        if self._window is not None:
            self._window.destroy()
            self._window = None


class ModeSelector(tk.Menubutton):
    """A compact pill selector: an icon, a label, and a downward chevron.

    The composer's bottom-left toolbar carries two of these -- the Code Studio
    and Chat Studio selectors of the reference design -- so a mode is switched
    from the input panel itself rather than a separate menu bar. It is a real
    menu button: the caller supplies the ``(label, callback)`` options, so the
    widget never invents a mode the screen does not actually have.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        theme: Any,
        typography: Typography,
        icon: str,
        label: str,
        options: list[tuple[str, Callable[[], Any]]],
        tip: str = "",
    ) -> None:
        self.theme = theme
        self._type = typography
        self._icon = icon
        super().__init__(
            parent, text=f"{icon}  {label}  ⌄", bg=theme.surface, fg=theme.text_secondary,
            activebackground=theme.surface_raised, activeforeground=theme.contrast,
            font=typography.font("xs", weight="bold"), relief="flat", borderwidth=0,
            highlightthickness=0, padx=8, pady=3, cursor="hand2", indicatoron=False,
        )
        menu = tk.Menu(
            self, tearoff=0, bg=theme.surface, fg=theme.foreground,
            activebackground=theme.interactive_active, activeforeground=theme.contrast,
            font=typography.font("sm"), borderwidth=0,
        )
        for text, command in options:
            menu.add_command(label=text, command=command)
        self.configure(menu=menu)
        if tip:
            Tooltip(self, tip, theme, typography)
        # The subtle hover tint of the reference design (Tk's ``activebackground``
        # only paints while the button is held).
        self.bind("<Enter>", lambda _e: self.configure(bg=theme.surface_raised), add="+")
        self.bind("<Leave>", lambda _e: self.configure(bg=theme.surface), add="+")

    def set_label(self, label: str) -> None:
        """Show the active choice (the caller keeps the menu itself in step)."""
        self.configure(text=f"{self._icon}  {label}  ⌄")


def _variable_write_callback(composer: ChatComposer) -> Callable[..., None]:
    """A trace callback that does not keep ``composer`` alive (see ``__init__``)."""
    reference = weakref.ref(composer)

    def callback(*_args: Any) -> None:
        target = reference()
        if target is not None:
            target._on_variable_write()

    return callback


class ChatComposer:
    """One wide, rounded chat input panel, shared by every chat-style screen.

    The panel is built once and lives in the caller's layout; the caller keeps
    its own attributes (``_entry``/``_question``/``_request_var`` …) pointed at
    the widgets this exposes, so the existing submit handlers keep working.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        theme: Any,
        typography: Typography,
        placeholder: str,
        on_submit: Callable[[], Any],
        corner: int = RADIUS_LARGE,
        outer_bg: str | None = None,
        min_lines: int = DEFAULT_MIN_LINES,
        max_lines: int = DEFAULT_MAX_LINES,
        variable: tk.StringVar | None = None,
        with_attach: bool = True,
        with_status: bool = True,
    ) -> None:
        self.theme = theme
        self._type = typography
        # ``on_submit`` is a bound method of the caller, so it is dropped when the
        # panel's window goes away (see :meth:`_on_destroy`): a strong reference kept
        # after that would hold the caller (and its whole widget tree) in a
        # reference cycle that only the cyclic collector could break -- and that can
        # run on a worker thread, where a Tcl call deadlocks. The caller owns the
        # handler; this panel only calls it while it is alive.
        self._on_submit: Callable[[], Any] | None = on_submit
        self._placeholder = placeholder
        self._min_lines = min_lines
        self._max_lines = max_lines
        self._toolbar_widgets: list[tk.Widget] = []
        # A caller that already drives its panel through a ``StringVar`` keeps
        # that variable in step with the text area, in both directions, so the
        # existing handlers and tests that read the variable are untouched. The
        # variable is held *weakly*: a strong reference here would outlive the
        # widget the caller attached it to, leaving the variable to be freed only
        # by the cyclic collector -- which can run on a worker thread, where Tcl
        # cannot be called. The caller owns its variable; this panel only reads it.
        self._variable_ref: weakref.ref[tk.StringVar] | None = (
            weakref.ref(variable) if variable is not None else None
        )
        self._trace_id: str | None = None
        self._syncing = False
        # The external gate (a screen that is busy disables Send); the content
        # gate is separate, so an empty box is never submittable either.
        self._submit_enabled = True

        self.wrap = tk.Frame(parent, bg=outer_bg or theme.base)
        self.box = RoundedBox(
            self.wrap, bg=outer_bg or theme.base, fill=theme.surface,
            outline=theme.border, corner=corner, parent_bg=outer_bg or theme.base,
        )
        self.box.pack(fill="x")
        self.inner = self.box.inner

        self.entry = tk.Text(
            self.inner, height=min_lines, width=1, bg=theme.surface, fg=theme.contrast,
            insertbackground=theme.primary, font=typography.font("chatInput"),
            relief="flat", wrap="word", padx=6, pady=10,
            highlightthickness=0, borderwidth=0,
        )
        self.entry.pack(side="top", fill="both", expand=True, padx=10, pady=(8, 2))
        self.entry.bind("<Return>", self._on_return)
        self.entry.bind("<Shift-Return>", self._on_shift_return)
        self.entry.bind("<<Modified>>", self._on_modified)
        self.entry.bind("<FocusIn>", self._on_focus_in)
        self.entry.bind("<FocusOut>", self._on_focus_out)
        if self._variable_ref is not None:
            # The trace must not hold this panel strongly either: Tcl keeps the
            # callback alive for the variable's lifetime, so a bound method here
            # would make the variable and the panel a reference cycle.
            variable = self._variable_ref()
            if variable is not None:
                self._trace_id = variable.trace_add("write", _variable_write_callback(self))
        # One place releases what a destroyed panel must not keep alive: the
        # caller's bound submit handler, its variable, and its variable's trace.
        self.wrap.bind("<Destroy>", self._on_destroy, add="+")

        # The placeholder is an overlay *label*, not text inside the entry, so the
        # input widget always holds the user's own words and nothing else -- the
        # existing handlers can read ``entry.get(...)`` and never see the hint.
        self.placeholder_label = tk.Label(
            self.entry, text=placeholder, bg=theme.surface, fg=theme.text_dim,
            font=typography.font("chatInput"), anchor="w", justify="left",
            cursor="xterm", takefocus=0,
        )
        self.placeholder_label.bind("<Button-1>", self._on_placeholder_click)

        # The panel is one card: the input strip across the top, then a single
        # toolbar row beneath it. The input and the Send disc are direct children
        # of the card's interior (the window's tests assert exactly that), while
        # the dot, the microphone, and any caller action live in a small group
        # frame that sits just left of Send.
        self.right_tools = tk.Frame(self.inner, bg=theme.surface)
        self.left_tools = tk.Frame(self.inner, bg=theme.surface)
        # The ＋ control only exists where the screen really has attachments or
        # extra options; a screen without them gets no button that would lie.
        self.attach_button: tk.Button | None = None
        if with_attach:
            self.attach_button = self._toolbar_button(
                self.left_tools, "＋", "Attach a file to this chat"
            )

        # The status dot: a subtle, non-interactive indicator of whether the
        # screen is idle or working. It carries no invented meaning; the caller
        # sets its colour through :meth:`set_status_dot`.
        self.status_dot: tk.Canvas | None = None
        self._dot: int | None = None
        if with_status:
            self.status_dot = tk.Canvas(
                self.right_tools, width=10, height=10, bg=theme.surface,
                highlightthickness=0, takefocus=0,
            )
            self._dot = self.status_dot.create_oval(
                1, 1, 9, 9, fill=theme.text_dim, outline=""
            )
            self.status_dot.pack(side="left", padx=(0, 8))

        # The microphone is optional, so a screen with no voice input simply never
        # calls :meth:`add_mic` rather than showing a control that does nothing.
        self.mic_button: tk.Button | None = None

        self.send_button = CircleButton(
            self.inner, size=SEND_BUTTON_SIZE, bg=theme.surface,
            fill=theme.primary, ink=theme.primary_foreground,
            text="↑", command=self._submit, typography=typography,
        )

        # Packing the input first (top) leaves the remaining strip for the
        # toolbar, so the actions sit under the input and never overlap it. The
        # input's own height comes from its line count (see :meth:`_autosize`),
        # so the card grows with the message rather than the layout stretching.
        self.entry.pack(side="top", fill="x", padx=10, pady=(8, 2))
        # Rightmost is Send; the dot/mic/actions group sits immediately left of it.
        self.send_button.pack(side="right", padx=(0, 10), pady=(0, 8))
        self.right_tools.pack(side="right", padx=(0, 4), pady=(0, 8))
        if self.attach_button is not None:
            self.attach_button.pack(side="left", padx=(10, 0), pady=(0, 8))
        self.left_tools.pack(side="left", padx=(0, 4), pady=(0, 8))

        self._sync_placeholder()
        self._update_send_state()

    # ------------------------------------------------------------- build API

    def add_mic(self, command: Callable[[], Any], tip: str = "Dictate with the microphone") -> tk.Button:
        """Add the microphone control beside Send (call before the panel is shown)."""
        button = tk.Button(
            self.right_tools, text="🎤", command=command,
            bg=self.theme.surface, fg=self.theme.text_dim,
            activebackground=self.theme.surface, activeforeground=self.theme.primary,
            font=self._type.font("lg"), relief="flat", borderwidth=0,
            padx=8, pady=0, cursor="hand2",
        )
        # After the dot, so the group reads [dot] [mic] [actions] left to right.
        button.pack(side="left", padx=(0, 4))
        Tooltip(button, tip, self.theme, self._type)
        # Hover tints the background only: the foreground is the recording state
        # (``set_mic_recording`` recolours it), so it must not be restored here.
        self._bind_hover(button, self.theme.surface_raised)
        self.mic_button = button
        return button

    def add_mode_selector(
        self,
        *,
        icon: str,
        label: str,
        options: list[tuple[str, Callable[[], Any]]],
        tip: str = "",
    ) -> ModeSelector:
        """Add a compact mode pill to the bottom-left toolbar, beside ＋."""
        selector = ModeSelector(
            self.left_tools, theme=self.theme, typography=self._type,
            icon=icon, label=label, options=options, tip=tip,
        )
        selector.pack(side="left", padx=(6, 0))
        self._toolbar_widgets.append(selector)
        return selector

    def add_tool(self, widget: tk.Widget) -> tk.Widget:
        """Place a caller-built action control on the bottom-right toolbar."""
        widget.pack(side="left", padx=(6, 0))
        self._toolbar_widgets.append(widget)
        return widget

    def _toolbar_button(self, parent: tk.Misc, glyph: str, tip: str) -> tk.Button:
        button = tk.Button(
            parent, text=glyph, bg=self.theme.surface, fg=self.theme.text_dim,
            activebackground=self.theme.surface_raised, activeforeground=self.theme.contrast,
            font=self._type.font("xl", weight="bold"), relief="flat", borderwidth=0,
            padx=10, pady=0, cursor="hand2",
        )
        Tooltip(button, tip, self.theme, self._type)
        self._bind_hover(button, self.theme.surface_raised)
        return button

    def _bind_hover(self, widget: tk.Widget, hover_bg: str, hover_fg: str | None = None) -> None:
        """Give a toolbar control the subtle hover tint the reference design shows.

        Tk has no CSS ``:hover``; ``activebackground`` only paints while the mouse
        button is held, so a real hover is bound here. It is added with ``add="+"``
        so the tooltip's own ``<Enter>``/``<Leave>`` handlers still run, and it
        restores the control's own resting colour on leave rather than assuming one.
        """
        rest_bg = str(widget.cget("bg"))
        rest_fg = str(widget.cget("fg"))

        def on_enter(_event: tk.Event) -> None:
            widget.configure(**{"bg": hover_bg})
            if hover_fg is not None:
                widget.configure(**{"fg": hover_fg})

        def on_leave(_event: tk.Event) -> None:
            widget.configure(**{"bg": rest_bg})
            if hover_fg is not None:
                widget.configure(**{"fg": rest_fg})

        widget.bind("<Enter>", on_enter, add="+")
        widget.bind("<Leave>", on_leave, add="+")

    def set_tooltip(self, widget: tk.Widget, text: str) -> None:
        """Attach a hover hint to a control the panel built for the caller."""
        Tooltip(widget, text, self.theme, self._type)

    # -------------------------------------------------------------- behavior

    def set_submit(self, on_submit: Callable[[], Any]) -> None:
        """Point the panel's Send control (and Enter) at a submit handler."""
        self._on_submit = on_submit

    def _submit(self) -> None:
        if self._on_submit is not None:
            self._on_submit()

    def _on_return(self, _event: tk.Event) -> str:
        self._submit()
        return "break"

    def _on_shift_return(self, _event: tk.Event) -> None:
        """Shift+Enter inserts a newline (the Text default), so return nothing."""
        return None

    def _on_modified(self, _event: tk.Event) -> None:
        """Re-measure, mirror, and re-hide the hint whenever the text changes."""
        if not self.entry.edit_modified():
            return
        self.entry.edit_modified(False)
        self._sync_placeholder()
        self._autosize()
        self._sync_variable()
        self._update_send_state()

    def _sync_placeholder(self) -> None:
        """Show the overlay hint only while the input holds no real text."""
        if self.text().strip():
            self.placeholder_label.place_forget()
        else:
            self._place_placeholder()

    def _place_placeholder(self) -> None:
        self.placeholder_label.place(in_=self.entry, x=8, y=10, anchor="nw")

    def _on_placeholder_click(self, _event: tk.Event) -> str:
        self.entry.focus_set()
        return "break"

    def _on_focus_in(self, _event: tk.Event) -> None:
        # The subtle focus treatment is the caret and the I-beam cursor, not a
        # recoloured border: the project deliberately keeps the composer's hairline
        # the neutral stroke (``test_window_composer_has_no_gold_focus_outline``),
        # because a bright ring around the input reads as heavy rather than quiet.
        self._sync_placeholder()
        self.box.configure(cursor="xterm")

    def _on_focus_out(self, _event: tk.Event) -> None:
        self._sync_placeholder()

    def _live_variable(self) -> tk.StringVar | None:
        """The caller's variable, if it is still alive (the panel holds it weakly)."""
        return self._variable_ref() if self._variable_ref is not None else None

    def _on_destroy(self, event: tk.Event) -> None:
        """Drop the caller's handler and the variable trace when the panel dies."""
        if event.widget is not self.wrap:
            return
        # The caller's bound method is dropped so the destroyed window is not kept
        # alive by a cycle only the cyclic collector could break.
        self._on_submit = None
        self._toolbar_widgets.clear()
        variable = self._live_variable()
        if variable is not None and self._trace_id is not None:
            try:
                variable.trace_remove("write", self._trace_id)
            except tk.TclError:  # the variable's interpreter already went away
                pass
        self._trace_id = None

    def _on_variable_write(self) -> None:
        """Mirror a caller's ``StringVar`` into the text area (one direction)."""
        variable = self._live_variable()
        if self._syncing or variable is None:
            return
        wanted = variable.get()
        if wanted == self.text():
            return
        self.set_text(wanted)

    def _sync_variable(self) -> None:
        """Mirror the text area back into the caller's ``StringVar``."""
        variable = self._live_variable()
        if variable is None:
            return
        self._syncing = True
        try:
            variable.set(self.text())
        finally:
            self._syncing = False

    # --------------------------------------------------------- placeholder

    def text(self) -> str:
        """The user's own text; the overlay hint is never part of it."""
        return self.entry.get("1.0", "end-1c")

    def set_text(self, text: str) -> None:
        """Replace the content (used for a transcript or a suggested prompt)."""
        self._syncing = True
        try:
            self.entry.delete("1.0", "end")
            if text:
                self.entry.insert("1.0", text)
            self.entry.edit_modified(False)
        finally:
            self._syncing = False
        self._sync_placeholder()
        self._autosize()
        self._sync_variable()
        self._update_send_state()

    def clear(self) -> None:
        """Empty the panel and show the hint again."""
        self.set_text("")

    def focus(self) -> None:
        self.entry.focus_set()

    # ------------------------------------------------------------- autosize

    def _line_height(self) -> int:
        try:
            return max(1, int(self.entry.tk.call("font", "metrics", self.entry.cget("font"), "-linespace")))
        except tk.TclError:  # pragma: no cover - a font the platform cannot measure
            return 20

    def _autosize(self) -> None:
        """Grow the input with its content, capped so the panel scrolls instead."""
        self.entry.update_idletasks()
        measured = self.entry.count("1.0", "end-1c", "displaylines")
        lines = int(measured[0]) if measured else 1
        lines = max(self._min_lines, min(self._max_lines, lines))
        if int(self.entry.cget("height")) != lines:
            self.entry.configure(height=lines)

    # ------------------------------------------------------------ send state

    def _update_send_state(self) -> None:
        """Grey the Send disc out while the panel holds nothing to send."""
        self._repaint_send()

    def set_send_enabled(self, enabled: bool) -> None:
        """Enable or disable the Send control (the caller's own gate)."""
        self._submit_enabled = enabled
        self._repaint_send()

    def _repaint_send(self) -> None:
        enabled = self._submit_enabled and bool(self.text().strip())
        self.send_button.set_state(
            enabled=enabled,
            fill=self.theme.primary if enabled else self.theme.disabled,
            ink=self.theme.primary_foreground if enabled else self.theme.text_dim,
        )

    def set_status_dot(self, colour: str) -> None:
        """Colour the small status dot (idle/working), without inventing meaning."""
        if self.status_dot is not None and self._dot is not None:
            self.status_dot.itemconfigure(self._dot, fill=colour)

    def set_mic_recording(self, recording: bool) -> None:
        """Reflect a recording in progress on the microphone control, if any."""
        if self.mic_button is None:
            return
        self.mic_button.configure(
            fg=self.theme.primary if recording else self.theme.text_dim,
            text="⏺" if recording else "🎤",
        )
