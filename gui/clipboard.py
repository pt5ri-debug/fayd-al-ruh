"""Copy, cut, paste, and select-all everywhere in the program.

Tk gives a ``Text`` and an ``Entry`` a working clipboard on X11 through their
class bindings (``<<Copy>>``/``<<Paste>>``/``<<Cut>>``), but that is not enough
for this program, and three gaps were measured on the real windows:

* **Select-all is missing.** ``<<SelectAll>>`` is bound to ``Control-Key-slash``,
  never to the ``Ctrl+A`` a user actually presses, so ``Ctrl+A`` does nothing in
  any field of any window.
* **A read-only message can be pasted into.** The transcript's read-only guard
  exempts *any* Control/Command chord, so ``Ctrl+V`` inserts clipboard text into a
  message that is meant to be read-only -- a silent corruption of the transcript.
* **A dialog is a different toplevel.** The main window binds its copy/select
  shortcuts on its own root, so those bindings reach only widgets the root owns;
  a ``Toplevel`` dialog -- the Characters window, a Studio -- is outside that
  scope and inherits none of it.

This module closes all three in one place. :func:`install_app_clipboard` binds
the clipboard actions on every *text widget class* (``Text``, ``Entry``, the ttk
variants), which Tk dispatches to every widget of that class in *every* toplevel
of the interpreter, including widgets created after the call -- so one
installation covers the whole program, and it replaces the class's own binding
rather than adding to it, so an action runs exactly once.
:func:`make_read_only` keeps a message selectable and copyable while refusing
typing, paste, and cut -- and, unlike the guard it replaces, it does not exempt
the Control chords that carry the paste. :class:`TextContextMenu` adds the
right-click menu a desktop user expects, with only the actions the clicked widget
can actually perform.

Everything here is presentation only: it moves text the user already selected
between a widget and the system clipboard and never touches program state.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable, Iterable
from tkinter import ttk
from typing import Any

#: The clipboard actions every text widget gets, as virtual-event names.
CLIPBOARD_EVENTS = ("Copy", "Cut", "Paste", "SelectAll")

#: The physical keys a user presses for each action, so a binding is not at the
#: mercy of a platform's virtual-event mapping.
_PHYSICAL_KEYS = {
    "copy": ("<Control-Key-c>", "<Control-Key-C>"),
    "cut": ("<Control-Key-x>", "<Control-Key-X>"),
    "paste": ("<Control-Key-v>", "<Control-Key-V>"),
    "select_all": ("<Control-Key-a>", "<Control-Key-A>"),
}

#: The Tcl class names of every widget that holds text. ``bind_class`` on these
#: names reaches every such widget in *every* toplevel of the interpreter,
#: including a dialog and a widget built later, which is what makes one
#: installation cover the whole program.
_TEXT_CLASS_NAMES = ("Text", "Entry", "TEntry", "TCombobox", "TSpinbox", "Spinbox")

#: The keys a read-only widget still accepts: moving the caret and selecting.
_NAVIGATION_KEYS = frozenset(
    {"Left", "Right", "Up", "Down", "Home", "End", "Prior", "Next"}
)

#: The widget classes a clipboard action can be applied to. ``tk.Spinbox`` is
#: its own widget (not an ``Entry`` subclass) but carries the same selection
#: API, so it is listed explicitly.
_TEXT_CLASSES = (tk.Text, tk.Entry, ttk.Entry, tk.Spinbox)


def is_text_widget(widget: Any) -> bool:
    """Whether ``widget`` holds editable or selectable text."""
    return isinstance(widget, _TEXT_CLASSES)


def is_text_area(widget: Any) -> bool:
    """Whether ``widget`` is a multi-line ``Text`` rather than a single-line field."""
    return isinstance(widget, tk.Text)


def _clipboard_read(widget: tk.Misc) -> str:
    """The clipboard's text, or ``""`` when it holds none this program can read."""
    try:
        return widget.clipboard_get()
    except tk.TclError:
        return ""


def _clipboard_write(widget: tk.Misc, text: str) -> None:
    if not text:
        return
    widget.clipboard_clear()
    widget.clipboard_append(text)


def _selection(widget: Any) -> str:
    """The widget's selected text, or ``""`` when nothing is selected."""
    try:
        if is_text_area(widget):
            if widget.tag_ranges("sel"):
                return widget.get("sel.first", "sel.last")
            return ""
        if widget.selection_present():
            return widget.selection_get()
    except tk.TclError:
        return ""
    return ""


def _clear_selection(widget: Any) -> None:
    """Delete the selection, so an insert replaces it rather than joins it."""
    try:
        if is_text_area(widget):
            if widget.tag_ranges("sel"):
                widget.delete("sel.first", "sel.last")
        elif widget.selection_present():
            widget.delete("sel.first", "sel.last")
    except tk.TclError:
        pass


def _select_all(widget: Any) -> None:
    try:
        if is_text_area(widget):
            widget.tag_add("sel", "1.0", "end-1c")
            widget.mark_set("insert", "1.0")
        else:
            widget.selection_range(0, "end")
    except tk.TclError:
        pass


def copy_selection(widget: Any) -> str:
    """Copy the widget's selection to the clipboard; return it (``""`` if none)."""
    text = _selection(widget)
    if text:
        _clipboard_write(widget, text)
    return text


def cut_selection(widget: Any) -> str:
    """Copy then delete the widget's selection; return what was cut."""
    text = _selection(widget)
    if not text:
        return ""
    _clipboard_write(widget, text)
    _clear_selection(widget)
    return text


def paste_clipboard(widget: Any) -> str:
    """Insert the clipboard's text at the caret; return what was pasted."""
    text = _clipboard_read(widget)
    if not text:
        return ""
    _clear_selection(widget)
    try:
        widget.insert("insert", text)
    except tk.TclError:
        return ""
    return text


# --------------------------------------------------------------- read-only


def make_read_only(widget: tk.Text) -> None:
    """Keep a ``Text`` selectable and copyable while refusing every edit.

    Typing, paste, and cut are refused; copy and select-all still work, because
    those two change nothing. A previous guard exempted *any* Control/Command
    chord so copy would keep working, but that also let ``Ctrl+V`` paste into a
    read-only message -- a silent corruption of the transcript. Here the two
    harmless chords are named explicitly and every other Control chord is
    refused, so ``Ctrl+V``/``Ctrl+X``/``Shift+Insert`` are all blocked while
    ``Ctrl+C`` and ``Ctrl+A`` are not. The clipboard-changing virtual events are
    blocked as well, as a second line of defence against a programmatic paste.
    """

    def block(event: tk.Event) -> str:
        if int(event.state) & 0x0004 or int(event.state) & 0x0008:  # Control / Command
            if event.keysym in {"c", "C", "a", "A"}:
                return ""
            return "break"
        if event.keysym in _NAVIGATION_KEYS:
            return ""
        return "break"

    widget.bind("<Key>", block)
    for sequence in ("<<Paste>>", "<<Cut>>", "<<Clear>>", "<<PasteSelection>>"):
        widget.bind(sequence, lambda _event: "break")
    try:
        widget._fayd_read_only = True  # type: ignore[attr-defined]
    except tk.TclError:  # pragma: no cover
        pass


def is_read_only(widget: Any) -> bool:
    """Whether :func:`make_read_only` has been applied to ``widget``.

    A ``Text`` is not put in Tk's ``disabled`` state -- that would stop the
    keyboard events copy and select-all need -- so its read-only-ness is marked
    on the widget instead, which is what the right-click menu reads to decide
    whether Paste and Cut apply.
    """
    return bool(getattr(widget, "_fayd_read_only", False))


# ------------------------------------------------------------ app-wide setup


def _install_action(root: tk.Misc, name: str, action: Callable[[Any], Any]) -> None:
    """Bind one clipboard action to every text widget class.

    A ``bind_class`` binding reaches every widget of that class in every
    toplevel, so one call covers the main window *and* each dialog, and a widget
    built later is covered too. It *replaces* the class's own binding rather than
    adding to it, so the action runs exactly once -- a read-only message's guard
    can no longer be pre-empted by Tk's own class binding, and a paste cannot be
    applied twice. The virtual event and the physical chords are both bound,
    because a platform's virtual-event mapping (``<<Copy>>`` on
    ``<Control-Key-c>``) is not something to depend on.
    """

    def handler(event: tk.Event) -> str:
        widget = event.widget
        if is_text_widget(widget):
            action(widget)
        return "break"

    sequences = [_VIRTUAL[name], *_PHYSICAL_KEYS[name]]
    for class_name in _TEXT_CLASS_NAMES:
        for sequence in sequences:
            try:
                root.bind_class(class_name, sequence, handler)
            except tk.TclError:  # pragma: no cover - a class Tk does not know
                continue


#: The virtual-event name of each action (``copy`` -> ``<<Copy>>``).
_VIRTUAL = {
    "copy": "<<Copy>>",
    "cut": "<<Cut>>",
    "paste": "<<Paste>>",
    "select_all": "<<SelectAll>>",
}


#: The one context-menu controller per application, kept as an attribute of the
#: real root window. Every toplevel of a root shares the root, so a dialog that
#: installs on its own window finds the menu already in place instead of binding
#: a second one to the same ``all`` tag -- and the attribute is released with the
#: root, so nothing outlives the window.
_CONTEXT_MENU_ATTR = "_fayd_context_menu"


def _application_root(widget: tk.Misc) -> tk.Misc:
    """The real root window of ``widget``'s tree.

    A dialog passes its own ``Toplevel``; the menu must outlive it and belong to
    the application, so it is built on the root the whole tree hangs from.
    """
    try:
        return widget._root()  # type: ignore[attr-defined,no-any-return]
    except (AttributeError, tk.TclError):  # pragma: no cover - a detached widget
        return widget


def _existing_menu(root: tk.Misc) -> TextContextMenu | None:
    menu = getattr(_application_root(root), _CONTEXT_MENU_ATTR, None)
    if menu is not None and _menu_alive(menu):
        return menu
    return None


def install_app_clipboard(
    root: tk.Misc,
    extras: Callable[[Any], Iterable[tuple[str, Callable[[], None]]]] | None = None,
) -> None:
    """Give every text widget of this application copy, cut, paste, select-all.

    Called once per root, and safe to call again: the *bindings* are made with
    ``bind_class`` (which reaches every text widget in every toplevel and
    replaces rather than stacks) and the right-click menu is kept on the
    application root, so a dialog that installs on its own window does not add a
    second menu to the one already there. ``extras`` contributes the caller's
    own right-click entries; the first caller to supply them wins, which in
    practice is the main window.
    """
    existing = _existing_menu(root)
    if existing is not None:
        if extras is not None and existing.extras is None:
            existing.extras = extras
        return
    _install_action(root, "copy", copy_selection)
    _install_action(root, "cut", cut_selection)
    _install_action(root, "paste", paste_clipboard)
    _install_action(root, "select_all", _select_all)
    install_context_menu(root, extras=extras)


def _menu_alive(controller: TextContextMenu) -> bool:
    try:
        return bool(controller.root.winfo_exists())
    except tk.TclError:  # pragma: no cover - a root already torn down
        return False


# ------------------------------------------------------------ context menu


class TextContextMenu:
    """A right-click menu offering the actions the clicked widget can perform.

    A read-only message is offered Copy and Select all but not Cut or Paste; an
    editable field is offered all four. An ``extras`` callback may contribute
    further entries for a particular widget -- the main window's transcript-wide
    *Copy conversation* and *Clear transcript*, say -- so one menu serves the
    whole application without a second one competing for the same right-click.
    The entries are rebuilt per click, so what is shown always matches the widget
    under the pointer.
    """

    def __init__(
        self,
        root: tk.Misc,
        extras: Callable[[Any], Iterable[tuple[str, Callable[[], None]]]] | None = None,
    ) -> None:
        self.root = root
        self.extras = extras
        self.widget: Any = None
        self.menu = tk.Menu(root, tearoff=0)

    # -- entry states ---------------------------------------------------

    def _editable(self) -> bool:
        widget = self.widget
        if widget is None:
            return False
        if is_read_only(widget):
            return False
        try:
            return str(widget.cget("state")) != "disabled"
        except tk.TclError:
            return True

    def _has_selection(self) -> bool:
        return bool(_selection(self.widget)) if self.widget is not None else False

    def _has_clipboard(self) -> bool:
        return bool(_clipboard_read(self.root)) if self.widget is not None else False

    def _build(self) -> None:
        """Lay the menu out for the widget the click landed on."""
        self.menu.delete(0, "end")
        editable = self._editable()
        self.menu.add_command(
            label="Cut", command=self._cut,
            state="normal" if editable and self._has_selection() else "disabled",
        )
        self.menu.add_command(
            label="Copy", command=self._copy,
            state="normal" if self._has_selection() else "disabled",
        )
        self.menu.add_command(
            label="Paste", command=self._paste,
            state="normal" if editable and self._has_clipboard() else "disabled",
        )
        self.menu.add_separator()
        self.menu.add_command(label="Select all", command=self._select_all)
        if self.extras is not None:
            for label, command in self.extras(self.widget) or ():
                self.menu.add_separator()
                self.menu.add_command(label=label, command=command)

    # -- actions --------------------------------------------------------

    def _cut(self) -> None:
        if self.widget is not None:
            cut_selection(self.widget)

    def _copy(self) -> None:
        if self.widget is not None:
            copy_selection(self.widget)

    def _paste(self) -> None:
        if self.widget is not None:
            paste_clipboard(self.widget)

    def _select_all(self) -> None:
        if self.widget is not None:
            _select_all(self.widget)

    # -- popup ----------------------------------------------------------

    def popup(self, event: tk.Event) -> str:
        widget = event.widget
        if not is_text_widget(widget):
            return ""
        self.widget = widget
        self._build()
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()
        return "break"


def install_context_menu(
    root: tk.Misc,
    extras: Callable[[Any], Iterable[tuple[str, Callable[[], None]]]] | None = None,
) -> TextContextMenu:
    """Bind a right-click clipboard menu on every text widget of this application.

    Returns the controller so a caller can keep it (the main window holds its
    own to hang the transcript-wide actions on). The menu is kept on the
    application root, so calling this again from a dialog returns the menu
    already in place instead of binding a second one to the same ``all`` tag.
    """
    existing = _existing_menu(root)
    if existing is not None:
        if extras is not None and existing.extras is None:
            existing.extras = extras
        return existing
    application_root = _application_root(root)
    controller = TextContextMenu(application_root, extras=extras)
    setattr(application_root, _CONTEXT_MENU_ATTR, controller)
    root.bind_all("<Button-3>", controller.popup, add="+")
    root.bind_all("<Button-2>", controller.popup, add="+")
    return controller


__all__ = [
    "CLIPBOARD_EVENTS",
    "TextContextMenu",
    "copy_selection",
    "cut_selection",
    "install_app_clipboard",
    "install_context_menu",
    "is_read_only",
    "is_text_area",
    "is_text_widget",
    "make_read_only",
    "paste_clipboard",
]
