"""The Studio settings window: the defaults a new companion and chat start from.

Opened from the Studio's **Settings…** control, this window edits the Studio's own
settings -- the default language model, conversation mode, content level, context
and answer limits, the memory recall limit, the voice engine, the interface
language, and the adult-mode switch -- through
:class:`gui.romance.RomanceStudioService`. Every control writes a real setting
(clamped to its documented range) the moment it is changed, so nothing here is a
placeholder; the values live in the Studio's own database and survive a restart.

It is a thin form over the service, and Tk is imported lazily with the rest of the
window layer.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from typing import Any

from gui.dialog_base import ThemedDialog
from gui.romance import (
    SETTING_ADULT_MODE,
    SETTING_CONTEXT_LENGTH,
    SETTING_DEFAULT_CONTENT_LEVEL,
    SETTING_DEFAULT_MODE,
    SETTING_DEFAULT_MODEL,
    SETTING_INTERFACE_LANGUAGE,
    SETTING_LABELS,
    SETTING_MAX_TOKENS,
    SETTING_MEMORY_LIMIT,
    SETTING_TEMPERATURE,
    SETTING_VOICE_ENABLED,
    SETTING_VOICE_ENGINE,
    RomanceStudioService,
)
from gui.typography import resolve

TITLE = "Companion Studio settings"
SUBTITLE = (
    "The defaults a new companion and a new conversation start from. Everything "
    "stays on this machine; adult content is off by default."
)


def _on_event(action: Callable[[], Any]) -> Callable[[Any], None]:
    """Wrap a zero-argument action for a Tk event binding (which passes one)."""

    def handler(_event: Any) -> None:
        action()

    return handler


#: The settings rendered as a dropdown, in the order they appear.
_COMBO_KEYS: tuple[str, ...] = (
    SETTING_DEFAULT_MODEL,
    SETTING_DEFAULT_MODE,
    SETTING_DEFAULT_CONTENT_LEVEL,
    SETTING_VOICE_ENGINE,
    SETTING_INTERFACE_LANGUAGE,
)

#: The settings rendered as a numeric field, with its bounds.
_NUMBER_KEYS: tuple[str, ...] = (
    SETTING_CONTEXT_LENGTH,
    SETTING_MAX_TOKENS,
    SETTING_TEMPERATURE,
    SETTING_MEMORY_LIMIT,
)

#: The settings rendered as an on/off checkbox.
_CHECK_KEYS: tuple[str, ...] = (SETTING_VOICE_ENABLED,)


class StudioSettingsDialog(ThemedDialog):
    """Edit the Romance & Companion Studio's own settings."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        service: RomanceStudioService,
        *,
        theme: Any,
        font: str,
        mono: str,
        overrides: object | None = None,
        list_overrides: object | None = None,
        on_close: Callable[[], None] | None = None,
        on_applied: Callable[[], None] | None = None,
    ) -> None:
        self.service = service
        self.theme = theme
        self.font = font
        self.mono = mono
        self._type = resolve(parent, ui_family=font, mono_family=mono, overrides=overrides)
        self._overrides = overrides
        self._list_overrides = list_overrides
        self._list_type = resolve(parent, ui_family=font, mono_family=mono, overrides=list_overrides)
        self._on_close = on_close
        self._on_applied = on_applied
        self._closed = False
        self._vars: dict[str, tk.StringVar] = {}
        self._combo_rows: dict[str, tuple[tuple[str, str], ...]] = {}
        self._check_vars: dict[str, tk.BooleanVar] = {}

        self.window = tk.Toplevel(parent)
        self.window.title(TITLE)
        self.window.configure(bg=theme.base)
        self.window.geometry("620x640")
        self.window.minsize(520, 480)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._themed_build()

    # -- build ------------------------------------------------------------

    def _build(self) -> None:
        from gui.components import DialogHeader

        header = DialogHeader(
            self.window, title=TITLE, subtitle=SUBTITLE, theme=self.theme,
            typography=self._type, on_close=self._close,
        )
        self._spinner = header.spinner

        body = tk.Frame(self.window, bg=self.theme.base)
        body.pack(fill="both", expand=True)
        self._canvas = tk.Canvas(body, bg=self.theme.base, highlightthickness=0)
        self._canvas.pack(side="left", fill="both", expand=True)
        scroll = tk.Scrollbar(body, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self._form = tk.Frame(self._canvas, bg=self.theme.base)
        self._window = self._canvas.create_window((0, 0), window=self._form, anchor="nw")
        self._form.bind("<Configure>", lambda _e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", lambda e: self._canvas.itemconfigure(self._window, width=e.width))

        self._build_rows()
        self._build_adult()
        self._build_footer()

    def _after_build(self) -> None:
        self._load()

    def _build_rows(self) -> None:
        theme = self.theme
        for key, label, hint in SETTING_LABELS:
            row = tk.Frame(self._form, bg=theme.surface)
            row.pack(fill="x", padx=16, pady=(10 if key == SETTING_DEFAULT_MODEL else 4, 0))
            head = tk.Frame(row, bg=theme.surface)
            head.pack(fill="x", padx=12, pady=(10, 0))
            tk.Label(
                head, text=label, bg=theme.surface, fg=theme.foreground,
                font=self._type.font("xs", weight="bold"), anchor="w",
            ).pack(side="left")
            if hint:
                tk.Label(
                    head, text=hint, bg=theme.surface, fg=theme.text_dim,
                    font=self._type.font("2xs"), anchor="e",
                ).pack(side="right")
            field = tk.Frame(row, bg=theme.surface)
            field.pack(fill="x", padx=12, pady=(2, 10))
            if key in _COMBO_KEYS:
                self._make_combo(field, key)
            elif key in _NUMBER_KEYS:
                self._make_number(field, key)
            elif key in _CHECK_KEYS:
                self._make_check(field, key)

    def _make_combo(self, parent: tk.Frame, key: str) -> None:
        from tkinter import ttk

        rows = self.service.setting_choices(key)
        self._combo_rows[key] = tuple(rows) or (("", "(none)"),)
        var = self._vars.setdefault(key, tk.StringVar(value=""))
        box = ttk.Combobox(
            parent, textvariable=var, values=[label for _v, label in self._combo_rows[key]],
            state="readonly", font=self._type.font("sm"),
        )
        box.pack(fill="x")
        box.bind("<<ComboboxSelected>>", _on_event(lambda: self._write_combo(key)))

    def _make_number(self, parent: tk.Frame, key: str) -> None:
        theme = self.theme
        var = self._vars.setdefault(key, tk.StringVar(value=""))
        entry = tk.Entry(
            parent, textvariable=var, width=10, relief="flat", borderwidth=1,
            highlightthickness=0, bg=theme.surface_raised, fg=theme.foreground,
            insertbackground=theme.foreground, font=self._type.font("sm"),
        )
        entry.pack(side="left", ipady=3, padx=1)
        entry.bind("<FocusOut>", _on_event(lambda: self._write_number(key)))
        entry.bind("<Return>", _on_event(lambda: self._write_number(key)))

    def _make_check(self, parent: tk.Frame, key: str) -> None:
        theme = self.theme
        var = self._check_vars.setdefault(key, tk.BooleanVar(value=False))
        tk.Checkbutton(
            parent, text="On", variable=var, bg=theme.surface, fg=theme.foreground,
            activebackground=theme.surface, activeforeground=theme.foreground,
            selectcolor=theme.base, font=self._type.font("sm"), anchor="w",
            highlightthickness=0, borderwidth=0,
            command=lambda: self._write_check(key),
        ).pack(fill="x")

    def _build_adult(self) -> None:
        theme = self.theme
        row = tk.Frame(self._form, bg=theme.surface)
        row.pack(fill="x", padx=16, pady=(10, 0))
        tk.Label(
            row, text="Adult content mode", bg=theme.surface, fg=theme.foreground,
            font=self._type.font("xs", weight="bold"), anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 0))
        tk.Label(
            row,
            text="Off by default. Turning it on records your adult-eligibility confirmation; "
                 "every fictional participant must be 18 or older.",
            bg=theme.surface, fg=theme.text_dim, font=self._type.font("2xs"),
            anchor="w", justify="left", wraplength=560,
        ).pack(fill="x", padx=12)
        self._adult_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            row, text="I am an adult and I enable adult content.", variable=self._adult_var,
            bg=theme.surface, fg=theme.foreground, activebackground=theme.surface,
            activeforeground=theme.foreground, selectcolor=theme.base,
            font=self._type.font("sm"), anchor="w", highlightthickness=0, borderwidth=0,
            command=self._write_adult,
        ).pack(fill="x", padx=12, pady=(2, 12))

    def _build_footer(self) -> None:
        theme = self.theme
        bar = tk.Frame(self._form, bg=theme.base)
        bar.pack(fill="x", padx=16, pady=(14, 18))
        self._button(bar, "Close", self._close, variant="primary").pack(side="left")
        self._button(bar, "Reset to defaults", self._reset).pack(side="left", padx=(8, 0))
        self._status_label = tk.Label(
            bar, text="", bg=theme.base, fg=theme.text_dim, font=self._type.font("2xs"), anchor="w",
        )
        self._status_label.pack(side="left", padx=(10, 0))

    # -- load / write -----------------------------------------------------

    def _load(self) -> None:
        values = self.service.settings()
        for key, _label, _hint in SETTING_LABELS:
            value = values.get(key, "")
            if key in _COMBO_KEYS:
                self._combo_set(key, value)
            elif key in _NUMBER_KEYS:
                self._vars.setdefault(key, tk.StringVar()).set(value)
            elif key in _CHECK_KEYS:
                self._check_vars.setdefault(key, tk.BooleanVar()).set(value == "on")
        try:
            self._adult_var.set(bool(self.service.safety_status().get("adult_mode")))
        except Exception:
            self._adult_var.set(False)

    def _combo_set(self, key: str, value: str) -> None:
        self._vars.setdefault(key, tk.StringVar())
        for row_value, label in self._combo_rows.get(key, ()):
            if row_value == value:
                self._vars[key].set(label)
                return
        rows = self._combo_rows.get(key, ())
        if rows:
            self._vars[key].set(rows[0][1])

    def _combo_value(self, key: str) -> str:
        shown = self._vars[key].get()
        for value, label in self._combo_rows.get(key, ()):
            if label == shown:
                return value
        return shown

    def _write_combo(self, key: str) -> None:
        value = self._combo_value(key)
        if key == SETTING_DEFAULT_CONTENT_LEVEL and value == "adult":
            if not self._ensure_adult():
                self._load()
                return
        self._store(key, value)

    def _write_number(self, key: str) -> None:
        self._store(key, self._vars[key].get())

    def _write_check(self, key: str) -> None:
        self._store(key, "on" if self._check_vars[key].get() else "off")

    def _write_adult(self) -> None:
        enabled = bool(self._adult_var.get())
        if enabled and not self._ensure_adult():
            self._adult_var.set(False)
            return
        try:
            self.service.set_adult_mode(enabled, confirmed=enabled)
        except Exception as error:
            self._status(str(error))
            self._adult_var.set(False)
            return
        self._store(SETTING_ADULT_MODE, "on" if enabled else "off", announce=False)
        self._status("Adult mode is " + ("on." if enabled else "off."))

    def _ensure_adult(self) -> bool:
        """Make sure adult mode is on (asking for confirmation once), or refuse."""
        try:
            if self.service.safety_status().get("adult_mode"):
                return True
        except Exception:
            pass
        from tkinter import messagebox

        if not messagebox.askyesno(
            "Adult content",
            "Adult content is off by default. Do you confirm you are an adult and "
            "want to enable it? Every fictional participant must be 18 or older.",
            parent=self.window,
        ):
            return False
        try:
            self.service.set_adult_mode(True, confirmed=True)
        except Exception as error:
            self._status(str(error))
            return False
        self._adult_var.set(True)
        return True

    def _store(self, key: str, value: str, *, announce: bool = True) -> None:
        try:
            clean = self.service.set_setting(key, value)
        except Exception as error:
            self._status(str(error))
            return
        if key in _NUMBER_KEYS:
            self._vars[key].set(clean)
        if announce:
            self._status("Saved.")
        if self._on_applied is not None:
            self._on_applied()

    def _reset(self) -> None:
        from tkinter import messagebox

        if not messagebox.askyesno("Reset", "Reset every Studio setting to its default?", parent=self.window):
            return
        try:
            self.service.reset_settings()
            self.service.set_adult_mode(False, confirmed=False)
        except Exception as error:
            self._status(str(error))
            return
        self._load()
        self._status("Reset to the defaults.")
        if self._on_applied is not None:
            self._on_applied()

    # -- helpers ----------------------------------------------------------

    def _button(self, parent: tk.Misc, text: str, command: Any, *, variant: str = "secondary") -> tk.Button:
        from gui.components import button

        return button(
            parent, text, command, theme=self.theme, typography=self._type,
            variant=variant, small=True,
        )

    def _status(self, text: str) -> None:
        try:
            self._status_label.configure(text=text)
        except (AttributeError, tk.TclError):
            pass

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.window.destroy()
        finally:
            if self._on_close is not None:
                self._on_close()


__all__ = ["StudioSettingsDialog"]
