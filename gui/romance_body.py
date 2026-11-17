"""The Body & shape studio: build a character's whole body, shape by shape.

The window the user asked for: opened from **My Persona Manager** or the
**Character Manager**, it shows the character's own body in the middle of the
window -- a woman's figure for a woman, a man's for a man, following the
character's own recorded gender -- with every measurement of the body editable
from the left, and ready-made body templates to start from and then adjust.

The engine is :mod:`core.romance.body` (itself a thin layer over the project's own
:mod:`core.scene3d_figures` / :mod:`core.scene3d` / :mod:`core.scene3d_render`),
so the window collects clicks and shows the render; it never re-implements a
figure or a renderer. A render runs on a worker thread whose result the UI thread
drains, because Tk is not thread-safe, and the picture is stored through the
Studio's own avatar manager, so **Use as picture** lands exactly where an upload
would.

Built the way every other window is built: the shared :class:`~gui.themes.Theme`
tokens, :class:`gui.components.DialogHeader`/``button``, the shared
:class:`gui.sidebar.SidebarColumn`, and :class:`gui.dialog_base.ThemedDialog` so a
theme or font change restyles it in place.
"""

from __future__ import annotations

import base64
import queue
import threading
import tkinter as tk
from collections.abc import Callable
from typing import Any

from gui.dialog_base import ThemedDialog
from gui.romance import RomanceStudioService
from gui.sidebar import SidebarColumn
from gui.typography import resolve

TITLE = "Body & shape"

_SUBTITLE = (
    "Build the character's body: pick a template, then shape every measurement. "
    "The picture is drawn by the project's own renderer and can become the "
    "character's picture."
)

#: The two camera framings, as (key, button label).
_VIEW_BUTTONS: tuple[tuple[str, str], ...] = (("full", "Full body"), ("face", "Face and head"))

#: The preview size the window renders at (a real PNG it shows).
_PREVIEW_WIDTH = 420
_PREVIEW_HEIGHT = 600


class BodyStudioDialog(ThemedDialog):
    """The body builder for one character, with live preview and every dial."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        service: RomanceStudioService,
        character_id: str,
        *,
        role: str = "companion",
        theme: Any,
        font: str,
        mono: str,
        overrides: object | None = None,
        list_overrides: object | None = None,
        on_close: Callable[[], None] | None = None,
        on_changed: Callable[[str], None] | None = None,
    ) -> None:
        self.service = service
        self.character_id = character_id
        self.role = role
        self.theme = theme
        self.font = font
        self.mono = mono
        self._type = resolve(parent, ui_family=font, mono_family=mono, overrides=overrides)
        self._overrides = overrides
        self._list_overrides = list_overrides
        self._on_close = on_close
        self._on_changed = on_changed
        self._closed = False

        self._params = service.body_parameters(character_id)
        self._view = "full"
        self._dial_vars: dict[str, tk.DoubleVar] = {}
        self._dial_labels: dict[str, tk.Label] = {}
        self._colour_vars: dict[str, tk.StringVar] = {}
        self._swatches: dict[str, tk.Canvas] = {}
        self._syncing = False
        self._pending_render: str | None = None
        self._photo: Any = None
        self._preview_png: bytes = b""
        self._busy = False
        self._results: queue.Queue[Any] = queue.Queue()
        self._pump_ticket: str | None = None
        self._children: list[Any] = []

        name = self._character_name()
        self.window = tk.Toplevel(parent)
        self.window.title(f"{TITLE} — {name}" if name else TITLE)
        self.window.configure(bg=theme.base)
        self.window.geometry("1120x780")
        self.window.minsize(880, 600)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._themed_build()

    # -- helpers ----------------------------------------------------------

    def _character_name(self) -> str:
        try:
            bundle = self.service.character(self.character_id)
        except Exception:
            return ""
        return str(getattr(bundle, "name", "") or "")

    def _plural(self) -> str:
        return "persona" if self.role == "user" else "companion"

    # -- build ------------------------------------------------------------

    def _build(self) -> None:
        from gui.components import DialogHeader

        header = DialogHeader(
            self.window,
            title=TITLE,
            subtitle=_SUBTITLE,
            theme=self.theme,
            typography=self._type,
            on_close=self._close,
        )
        self._spinner = header.spinner

        body = tk.Frame(self.window, bg=self.theme.base)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=0, minsize=380)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        sidebar_parent = tk.Frame(body, bg=self.theme.surface)
        sidebar_parent.grid(row=0, column=0, sticky="nsew")
        self._column = SidebarColumn(
            sidebar_parent,
            theme=self.theme,
            typography=self._type,
            scope=f"romance-body-{self._plural()}",
            width=380,
        )
        self._build_sidebar(self._column)

        centre = tk.Frame(body, bg=self.theme.base)
        centre.grid(row=0, column=1, sticky="nsew")
        self._build_centre(centre)

        self._status = tk.Label(
            self.window,
            text="",
            bg=self.theme.base,
            fg=self.theme.text_dim,
            font=self._type.font("2xs"),
            anchor="w",
            justify="left",
            wraplength=1040,
        )
        self._status.pack(fill="x", padx=16, pady=(0, 8))

    def _after_build(self) -> None:
        self._sync_controls()
        self._render_now()

    # -- sidebar ----------------------------------------------------------

    def _build_sidebar(self, column: SidebarColumn) -> None:
        theme = self.theme

        templates = column.add("templates", "Templates", icon="🧍", open=True)
        tk.Label(
            templates,
            text="Start from a ready-made body, then adjust any measurement below.",
            bg=theme.surface,
            fg=theme.text_dim,
            font=self._type.font("2xs"),
            anchor="w",
            justify="left",
            wraplength=340,
        ).pack(fill="x", pady=(0, 4))
        for preset_key, title in self.service.body_presets():
            self._button(
                templates, title, _binder(self._apply_preset, preset_key),
            )

        build_section = column.add("build", "Body type", icon="⚧", open=True)
        tk.Label(
            build_section,
            text="The build family the figure is drawn from.",
            bg=theme.surface,
            fg=theme.text_dim,
            font=self._type.font("2xs"),
            anchor="w",
            justify="left",
            wraplength=340,
        ).pack(fill="x", pady=(0, 4))
        row = tk.Frame(build_section, bg=theme.surface)
        row.pack(fill="x")
        for build_key, title in self.service.body_builds():
            self._button(row, title, _binder(self._set_build, build_key))

        for group_key, title, dials in self.service.body_dial_groups():
            section = column.add(group_key, title, icon="🎚", open=(group_key == "proportions"))
            for dial in dials:
                self._build_dial(section, dial)

    def _build_dial(self, parent: tk.Frame, dial: Any) -> None:
        theme = self.theme
        holder = tk.Frame(parent, bg=theme.surface)
        holder.pack(fill="x", pady=2)
        head = tk.Frame(holder, bg=theme.surface)
        head.pack(fill="x")
        tk.Label(
            head,
            text=dial.label,
            bg=theme.surface,
            fg=theme.foreground,
            font=self._type.font("2xs", weight="bold"),
            anchor="w",
        ).pack(side="left")
        value_label = tk.Label(
            head,
            text="",
            bg=theme.surface,
            fg=theme.text_dim,
            font=self._type.font("2xs"),
            anchor="e",
        )
        value_label.pack(side="right")
        self._dial_labels[dial.key] = value_label

        if dial.kind == "colour":
            colour_var = self._colour_vars.setdefault(dial.key, tk.StringVar(value=""))
            field = tk.Frame(holder, bg=theme.surface)
            field.pack(fill="x", pady=(2, 0))
            swatch = tk.Canvas(
                field, width=22, height=16, bg=theme.surface_raised, highlightthickness=1,
                highlightbackground=theme.border_subtle,
            )
            swatch.pack(side="left")
            self._swatches[dial.key] = swatch
            entry = tk.Entry(
                field, textvariable=colour_var, relief="flat", borderwidth=1, highlightthickness=0,
                bg=theme.surface_raised, fg=theme.foreground, insertbackground=theme.foreground,
                font=self._type.font("2xs"),
            )
            entry.pack(side="left", fill="x", expand=True, padx=(6, 0), ipady=3)
            colour_var.trace_add("write", _colour_trace(self._on_colour, dial.key, colour_var))
            return

        if dial.kind == "choice":
            # A choice dial (the build family) is driven by its own buttons in the
            # Body type section, so no slider is drawn for it here.
            value_label.configure(text=str(getattr(self._params, dial.key, "")))
            return

        dial_var = self._dial_vars.setdefault(dial.key, tk.DoubleVar(value=0.0))
        scale = tk.Scale(
            holder,
            from_=dial.minimum,
            to=dial.maximum,
            resolution=dial.step,
            orient="horizontal",
            variable=dial_var,
            showvalue=False,
            highlightthickness=0,
            borderwidth=0,
            troughcolor=theme.surface_raised,
            bg=theme.surface,
            activebackground=theme.primary,
            length=320,
        )
        scale.pack(fill="x", pady=(1, 0))
        dial_var.trace_add("write", _dial_trace(self._on_dial, dial.key, dial_var))

    # -- centre -----------------------------------------------------------

    def _build_centre(self, parent: tk.Frame) -> None:
        theme = self.theme
        head = tk.Frame(parent, bg=theme.base)
        head.pack(fill="x", padx=16, pady=(12, 4))
        self._name_label = tk.Label(
            head,
            text="",
            bg=theme.base,
            fg=theme.contrast,
            font=self._type.font("heading3", weight="bold"),
            anchor="w",
        )
        self._name_label.pack(anchor="w")
        self._summary_label = tk.Label(
            head,
            text="",
            bg=theme.base,
            fg=theme.text_dim,
            font=self._type.font("2xs"),
            anchor="w",
            justify="left",
            wraplength=680,
        )
        self._summary_label.pack(anchor="w")

        views = tk.Frame(parent, bg=theme.base)
        views.pack(fill="x", padx=16, pady=(4, 4))
        for view_key, label in _VIEW_BUTTONS:
            self._button(views, label, _binder(self._set_view, view_key))

        stage = tk.Frame(parent, bg=theme.surface_raised)
        stage.pack(fill="both", expand=True, padx=16, pady=(4, 8))
        self._canvas = tk.Label(
            stage,
            text="Rendering the body…",
            bg=theme.surface_raised,
            fg=theme.text_dim,
            font=self._type.font("sm"),
        )
        self._canvas.pack(fill="both", expand=True)

        actions = tk.Frame(parent, bg=theme.base)
        actions.pack(fill="x", padx=16, pady=(0, 8))
        self._button(actions, "Render", self._render_now, variant="primary")
        self._button(actions, "Use as the picture", self._use_as_avatar)
        self._button(actions, "Reset the shape", self._reset_shape)
        self._button(actions, "Close", self._close)

        note = tk.Label(
            parent,
            text=(
                "A stylised, parametric figure drawn locally by the project's own CPU "
                "rasteriser — not a licensed likeness or a photograph."
            ),
            bg=theme.base,
            fg=theme.text_dim,
            font=self._type.font("2xs"),
            anchor="w",
            justify="left",
            wraplength=680,
        )
        note.pack(fill="x", padx=16, pady=(0, 8))

    # -- control helpers --------------------------------------------------

    def _button(self, parent: tk.Misc, text: str, command: Callable[[], None], *, variant: str = "secondary") -> tk.Button:
        from gui.components import button

        return button(parent, text=text, command=command, theme=self.theme, typography=self._type, variant=variant)

    def _sync_controls(self) -> None:
        """Push the current parameters onto every control (a read of the dials)."""
        self._syncing = True
        try:
            for dial in self._all_dials():
                value = getattr(self._params, dial.key)
                if dial.kind == "colour":
                    colour_var = self._colour_vars.get(dial.key)
                    if colour_var is not None:
                        colour_var.set(_colour_text(value))
                    self._paint_swatch(dial.key)
                    continue
                dial_var = self._dial_vars.get(dial.key)
                if dial_var is not None:
                    try:
                        dial_var.set(round(float(value), 4))
                    except tk.TclError:
                        pass
                label = self._dial_labels.get(dial.key)
                if label is not None:
                    try:
                        label.configure(text=f"{float(value):.2f}")
                    except (tk.TclError, ValueError):
                        pass
        finally:
            self._syncing = False

    def _all_dials(self) -> tuple[Any, ...]:
        dials: list[Any] = []
        for _key, _title, group in self.service.body_dial_groups():
            dials.extend(group)
        return tuple(dials)

    def _paint_swatch(self, key: str) -> None:
        swatch = self._swatches.get(key)
        if swatch is None:
            return
        value = getattr(self._params, key, None)
        if not isinstance(value, tuple):
            return
        try:
            swatch.configure(bg=_hex(value))
        except tk.TclError:
            pass

    # -- dial callbacks ---------------------------------------------------

    def _on_dial(self, key: str, var: tk.DoubleVar) -> None:
        if self._syncing:
            return
        try:
            value = float(var.get())
        except (tk.TclError, ValueError):
            return
        try:
            self._params = self.service.set_body_dial(self._params, key, value)
        except KeyError:
            return
        label = self._dial_labels.get(key)
        if label is not None:
            try:
                label.configure(text=f"{value:.2f}")
            except tk.TclError:
                pass
        self._schedule_render()

    def _on_colour(self, key: str, var: tk.StringVar) -> None:
        if self._syncing:
            return
        try:
            raw = var.get()
        except tk.TclError:
            return
        colour = _parse_colour(raw)
        if colour is None:
            return
        try:
            self._params = self.service.set_body_dial(self._params, key, colour)
        except KeyError:
            return
        self._paint_swatch(key)
        self._schedule_render()

    def _set_build(self, key: str) -> None:
        try:
            self._params = self.service.set_body_dial(self._params, "build", key)
        except KeyError:
            return
        self._sync_controls()
        self._render_now()

    def _apply_preset(self, key: str) -> None:
        try:
            self._params = self.service.apply_body_preset(self._params, key)
        except KeyError as error:
            self._set_status(str(error))
            return
        self._sync_controls()
        self._set_status(f"Template applied: {key}.")
        self._render_now()

    def _reset_shape(self) -> None:
        self._params = self.service.body_parameters(self.character_id)
        self._sync_controls()
        self._set_status("Shape reset to the character's starting body.")
        self._render_now()

    def _set_view(self, key: str) -> None:
        self._view = key if key in ("full", "face") else "full"
        self._render_now()

    # -- rendering --------------------------------------------------------

    def _cancel_pending_render(self) -> None:
        """Cancel the debounce ticket, if one is pending (a no-op otherwise)."""
        ticket = self._pending_render
        if ticket is None:
            return
        try:
            self.window.after_cancel(ticket)
        except (tk.TclError, ValueError):
            pass
        self._pending_render = None

    def _schedule_render(self) -> None:
        """Coalesce a burst of dial changes into one render (a short debounce)."""
        self._cancel_pending_render()
        try:
            self._pending_render = self.window.after(220, self._render_now)
        except tk.TclError:
            self._pending_render = None

    def _render_now(self) -> None:
        self._cancel_pending_render()
        if self._closed or self._busy:
            return
        params = self._params
        view = self._view
        width, height = _PREVIEW_WIDTH, _PREVIEW_HEIGHT

        def job() -> bytes:
            return self.service.render_body(params, view=view, width=width, height=height)

        self._run_background("Rendering the body…", job, self._on_rendered)

    def _on_rendered(self, data: bytes, error: Exception | None) -> None:
        if error is not None:
            self._set_status(f"Could not render: {error}")
            return
        self._preview_png = data or b""
        self._show_preview()
        self._name_label.configure(text=self._character_name() or "Body")
        self._summary_label.configure(text=self.service.body_summary(self._params))
        self._set_status(
            f"Rendered {_PREVIEW_WIDTH}x{_PREVIEW_HEIGHT} ({len(self._preview_png)} bytes)."
        )

    def _show_preview(self) -> None:
        if not self._preview_png:
            self._canvas.configure(image="", text="Nothing to show yet.")
            return
        try:
            self._photo = tk.PhotoImage(data=base64.b64encode(self._preview_png))
        except tk.TclError:
            self._photo = None
            self._canvas.configure(image="", text="The render could not be shown.")
            return
        self._canvas.configure(image=self._photo, text="")

    # -- actions ----------------------------------------------------------

    def _use_as_avatar(self) -> None:
        params = self._params
        character_id = self.character_id

        def job() -> Any:
            return self.service.use_body_as_avatar(character_id, params)

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._set_status(f"Could not set the picture: {error}")
                return
            self._set_status(getattr(result, "describe", lambda: "Picture set.")())
            self._notify()

        self._run_background("Storing the picture…", job, finish)

    def _notify(self) -> None:
        if self._on_changed is not None:
            try:
                self._on_changed(self.character_id)
            except Exception:
                pass

    # -- worker thread ----------------------------------------------------

    def _run_background(
        self,
        message: str,
        job: Callable[[], Any],
        finish: Callable[[Any, Exception | None], None],
    ) -> None:
        self._busy = True
        self._set_status(message)
        try:
            self._spinner.pack(side="right", padx=8)
            self._spinner.start(12)
        except tk.TclError:
            pass

        def run() -> None:
            try:
                self._results.put((finish, job(), None))
            except Exception as error:  # a failed job is reported, never raised
                self._results.put((finish, None, error))

        threading.Thread(target=run, daemon=True).start()
        self._pump_ticket = self.window.after(60, self._pump_results)

    def _pump_results(self) -> None:
        self._pump_ticket = None
        try:
            finish, result, error = self._results.get_nowait()
        except queue.Empty:
            if self._busy and not self._closed:
                self._pump_ticket = self.window.after(60, self._pump_results)
            return
        self._busy = False
        try:
            self._spinner.stop()
            self._spinner.pack_forget()
        except tk.TclError:
            pass
        finish(result, error)

    # -- status and close -------------------------------------------------

    def _set_status(self, text: str) -> None:
        try:
            self._status.configure(text=text)
        except (AttributeError, tk.TclError):
            pass

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pump_ticket is not None:
            try:
                self.window.after_cancel(self._pump_ticket)
            except (tk.TclError, ValueError):
                pass
            self._pump_ticket = None
        for child in self._children:
            try:
                child._close()
            except Exception:
                pass
        try:
            self.window.destroy()
        finally:
            if self._on_close is not None:
                self._on_close()


def _binder(action: Callable[[str], None], key: str) -> Callable[[], None]:
    """Bind ``action`` to one key, as a zero-argument command.

    A tiny named helper rather than a ``lambda`` default argument: a default-argument
    lambda is a construct a type checker cannot infer, and this keeps the intent
    plain -- one button, one key.
    """

    def run() -> None:
        action(key)

    return run


def _dial_trace(
    handler: Callable[[str, tk.DoubleVar], None], key: str, var: tk.DoubleVar
) -> Callable[..., None]:
    """A ``trace_add`` callback for a numeric dial (the value is read from the var)."""

    def callback(*_args: object) -> None:
        handler(key, var)

    return callback


def _colour_trace(
    handler: Callable[[str, tk.StringVar], None], key: str, var: tk.StringVar
) -> Callable[..., None]:
    """A ``trace_add`` callback for a colour field (the value is read from the var)."""

    def callback(*_args: object) -> None:
        handler(key, var)

    return callback


def _colour_text(colour: tuple[float, float, float]) -> str:
    return ", ".join(f"{component:.2f}" for component in colour)


def _parse_colour(text: str) -> tuple[float, float, float] | None:
    """Read ``"r, g, b"`` (each 0..1) from a text field, or ``None`` if unusable."""
    parts = [piece for piece in text.replace(";", ",").split(",") if piece.strip()]
    if len(parts) != 3:
        return None
    values: list[float] = []
    for piece in parts:
        try:
            values.append(min(1.0, max(0.0, float(piece))))
        except ValueError:
            return None
    return (values[0], values[1], values[2])


def _hex(colour: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(min(1.0, max(0.0, c)) * 255):02x}" for c in colour)


__all__ = ["TITLE", "BodyStudioDialog"]
