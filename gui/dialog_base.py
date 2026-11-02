"""A live theme switch for every dialog window.

The main window can be rebuilt for a new palette in one call, because it owns
its own layout. The fifteen dialogs could not: each builds its widgets once in
``__init__`` and never again, so a theme chosen while a Studio was open left
that window in the old colours until it was closed and reopened.

This module is the one place that gap is closed. A dialog that mixes in
:class:`ThemedDialog` calls :meth:`~ThemedDialog._themed_build` instead of
``self._build()`` in its ``__init__`` and moves whatever it used to do right
after the build into :meth:`~ThemedDialog._after_build`. The window then calls
:meth:`~ThemedDialog.retheme` on every open dialog when the theme changes, so
one switch restyles every window at once.

The mixin records the messages a window shows -- by wrapping its
``_add_message`` -- and replays them after the rebuild, so the conversation a
user was reading is still there. Each message's text is refreshed from its live
widget first, but only for a *plain* row (a streamed trace, a diff): a Markdown
row is replayed from the source that produced it, because reading a rendered
``Text`` back would lose the headings and lists it was drawn with. The window's
own state -- its service, the selected item, the composer's text -- lives on the
instance and survives the rebuild untouched.

Nothing here imports :mod:`gui.app`, so a dialog can build on this module
without a circular import; it uses only the duck-typed ``Theme`` it is handed.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _Message:
    """One recorded ``_add_message`` call, so a rebuild can replay it."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)
    widget: Any = None


def _alive(widget: Any) -> bool:
    """Whether a Tk widget still exists (a rebuild destroys the old ones)."""
    try:
        return bool(widget.winfo_exists())
    except (tk.TclError, AttributeError):
        return False


def _plain(kwargs: dict[str, Any]) -> bool:
    """Whether a recorded message is plain text rather than rendered Markdown.

    Only a plain row's *widget* text is a faithful copy of what was asked for; a
    Markdown row renders to text with its headings and bullets already stripped,
    so its source is replayed instead.
    """
    return bool(kwargs.get("mono_body")) or kwargs.get("rich") is False


def _with_text(
    args: tuple[Any, ...], kwargs: dict[str, Any], text: str
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """``(args, kwargs)`` with the message's own text replaced by ``text``."""
    if len(args) >= 2:
        return (args[0], text, *args[2:]), kwargs
    if "text" in kwargs:
        updated = dict(kwargs)
        updated["text"] = text
        return args, updated
    return args, kwargs


class ThemedDialog:
    """Mixin: a ``Toplevel`` dialog that can be restyled without reopening.

    A subclass sets ``self.theme`` before :meth:`_themed_build`, defines
    ``_build`` exactly as before, and puts its post-build work in
    :meth:`_after_build` -- which runs once after the first build *and* again
    after every rebuild, so it must be safe to run twice (a refresh of labels
    from the service, never a state change).
    """

    theme: Any
    window: tk.Toplevel

    #: Subclasses define ``_build`` and (when they show messages)
    #: ``_add_message``; both are declared here so the mixin can reference them
    #: without knowing a concrete dialog. The annotation is type-only -- it adds
    #: no attribute at runtime, so ``getattr(self, "_add_message", None)`` still
    #: tells the truth about whether a dialog shows messages.
    _build: Any
    _add_message: Any

    #: Per-message bookkeeping lists a rebuild must empty before replaying, so a
    #: row is not counted twice. ``_rows``/``_row_frames``/``_cards`` hold the
    #: message widgets; ``_footers`` holds the Copy/Read-aloud bars under them.
    _MESSAGE_LISTS = ("_rows", "_row_frames", "_cards", "_footers")
    #: Attributes that point at one of the message widgets (a live trace, the
    #: row being read aloud), cleared so a stale one is never touched again.
    _TRANSIENT_ATTRS = (
        "_playing_footer",
        "_live",
        "_trace",
        "_review",
        "_streaming_label",
        "_thinking",
        "_thinking_row",
    )

    # -- first build ------------------------------------------------------

    def _themed_build(self) -> None:
        """Build the window for the first time, then run :meth:`_after_build`."""
        self._message_log: list[_Message] = []
        self._replaying = False
        self._recorder_installed = False
        self._initial_count = 0
        if callable(getattr(self, "_add_message", None)):
            self._install_recorder()
        self._build()
        self._after_build()
        # Everything drawn so far is the window's own opening content; it is
        # recreated by every build, so it is *not* replayed on a re-theme.
        self._initial_count = len(self._message_log)

    def _install_recorder(self) -> None:
        """Wrap ``_add_message`` once, so every message is remembered."""
        if self._recorder_installed:
            return
        original = type(self)._add_message.__get__(self)

        def recorded(*args: Any, **kwargs: Any) -> Any:
            widget = original(*args, **kwargs)
            if not self._replaying:
                self._message_log.append(
                    _Message(args=args, kwargs=dict(kwargs), widget=widget)
                )
            return widget

        self._add_message = recorded  # type: ignore[method-assign]
        self._recorder_installed = True

    def _after_build(self) -> None:
        """Refresh the window's labels from its service after a build.

        Runs once after the first build and again after every rebuild, so it
        must be idempotent -- a read, never a state change. The default does
        nothing.
        """

    # -- re-theme ---------------------------------------------------------

    def retheme(self, theme: Any) -> None:
        """Restyle this window for ``theme`` in place, keeping its messages."""
        if getattr(theme, "name", None) == getattr(self.theme, "name", None):
            return
        self.theme = theme
        self.window.configure(bg=theme.base)
        self._rebuild_keeping_messages()

    def retype(self, overrides: Any = None, list_overrides: Any = None) -> None:
        """Rebuild this window with new typography profiles, keeping its messages.

        The window's interface font comes from ``overrides`` and its list boxes'
        font from ``list_overrides``; a change to either is applied the same way a
        theme change is, by rebuilding in place and replaying the messages the
        user's own actions added. A window that never resolves a scale (or that
        has no ``_type``) is left alone.
        """
        if not hasattr(self, "_type"):
            return
        self._overrides = overrides
        if list_overrides is not None or hasattr(self, "_list_type"):
            self._list_overrides = list_overrides
        self._refresh_type()
        self._rebuild_keeping_messages()

    def _refresh_type(self) -> None:
        """Recompute ``self._type`` (and the list scale) from the window's profiles."""
        from gui.typography import resolve

        font = getattr(self, "font", "") or None
        mono = getattr(self, "mono", "") or None
        self._type = resolve(
            self.window, ui_family=font, mono_family=mono, overrides=getattr(self, "_overrides", None)
        )
        if hasattr(self, "_list_type"):
            self._list_type = resolve(
                self.window, ui_family=font, mono_family=mono,
                overrides=getattr(self, "_list_overrides", None),
            )

    def _rebuild_keeping_messages(self) -> None:
        """Destroy the window's body and rebuild it, replaying user messages."""
        carried: list[_Message] = []
        if self._recorder_installed:
            self._snapshot_messages()
            carried = list(self._message_log[self._initial_count :])
            self._clear_message_state()

        # The window's own initial messages -- the ones ``_build`` and
        # ``_after_build`` draw every time -- are recreated below, so they are
        # *not* replayed; only the messages the user's own actions added are.
        # Recording stays off for the whole rebuild, so nothing is logged twice,
        # and the log is then reset to just the carried set.
        self._replaying = True
        try:
            self._destroy_body()
            self._build()
            self._after_build()
            for message in carried:
                message.widget = self._add_message(*message.args, **message.kwargs)
            self._message_log = list(carried)
            self._initial_count = 0
        finally:
            self._replaying = False

    def _destroy_body(self) -> None:
        """Remove every child of the window so ``_build`` can lay it out again."""
        for child in list(self.window.winfo_children()):
            child.destroy()

    def _snapshot_messages(self) -> None:
        """Copy a plain message's live text back into its record before replay."""
        for message in self._message_log:
            if not _plain(message.kwargs):
                continue
            widget = message.widget
            if widget is None or not _alive(widget):
                continue
            try:
                text = widget.get("1.0", "end-1c")
            except tk.TclError:
                continue
            message.args, message.kwargs = _with_text(message.args, message.kwargs, text)

    def _clear_message_state(self) -> None:
        """Empty the per-message bookkeeping a replay will rebuild."""
        for name in self._MESSAGE_LISTS:
            value = getattr(self, name, None)
            if isinstance(value, list):
                value.clear()
        photos = getattr(self, "_avatar_photos", None)
        if isinstance(photos, dict):
            photos.clear()
        for name in self._TRANSIENT_ATTRS:
            if hasattr(self, name):
                setattr(self, name, None)


__all__ = ["ThemedDialog"]
