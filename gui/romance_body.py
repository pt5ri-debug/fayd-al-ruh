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
        self._camera_azimuth = 0.0
        self._camera_elevation = 0.0
        self._camera_zoom = 1.0
        self._mesh: Any = None
        self._edit_vars: dict[str, tk.DoubleVar] = {}
        self._edit_labels: dict[str, tk.Label] = {}
        self._edit_defaults: dict[str, float] = {}
        self._edit_scales: list[tk.Scale] = []
        self._drag_origin: tuple[int, int] | None = None
        self._drag_angles: tuple[float, float] | None = None
        self._dial_vars: dict[str, tk.DoubleVar] = {}
        self._dial_labels: dict[str, tk.Label] = {}
        self._colour_vars: dict[str, tk.StringVar] = {}
        self._swatches: dict[str, tk.Canvas] = {}
        self._syncing = False
        self._pending_render: str | None = None
        self._photo: Any = None
        self._preview_png: bytes = b""
        self._busy = False
        self._render_dirty = False
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
        # Give the column the whole cell: without the weight it keeps only its
        # natural height and its lower sections are clipped off the window.
        sidebar_parent.rowconfigure(0, weight=1)
        sidebar_parent.columnconfigure(0, weight=1)
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

        model = column.add("model", "Model (import & edit)", icon="🗿", open=True)
        self._build_model_section(model)

        camera = column.add("camera", "Camera (orbit)", icon="🎥", open=True)
        self._build_camera_section(camera)

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
            ).pack(side="top", fill="x", pady=1)

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
            if group_key == "pose":
                self._build_pose_section(section)
                continue
            for dial in dials:
                self._build_dial(section, dial)

    def _build_camera_section(self, parent: tk.Frame) -> None:
        """The camera tool: one-click angles, free orbit sliders, and drag-to-look.

        The user asked for a tool that moves the viewpoint around the figure in
        every direction, so this section offers both: named angles for the common
        views, and azimuth/elevation/zoom sliders for any other. The preview
        itself also accepts a click-and-drag, so the camera can be turned the way
        a 3D program turns it.
        """
        theme = self.theme
        tk.Label(
            parent,
            text=(
                "Turn the viewpoint around the body. Pick an angle, drag the sliders, "
                "or click and drag directly on the picture."
            ),
            bg=theme.surface,
            fg=theme.text_dim,
            font=self._type.font("2xs"),
            anchor="w",
            justify="left",
            wraplength=340,
        ).pack(fill="x", pady=(0, 4))

        grid = tk.Frame(parent, bg=theme.surface)
        grid.pack(fill="x")
        for preset_key, title in self.service.body_camera_presets():
            self._button(grid, title, _binder(self._apply_camera_preset, preset_key)).pack(
                side="top", fill="x", pady=1
            )

        self._camera_azimuth_var = tk.DoubleVar(value=self._camera_azimuth)
        self._camera_elevation_var = tk.DoubleVar(value=self._camera_elevation)
        self._camera_zoom_var = tk.DoubleVar(value=self._camera_zoom)
        self._camera_labels: dict[str, tk.Label] = {}
        self._camera_scales: list[tk.Scale] = []
        for key, label, variable, low, high, step in (
            ("azimuth", "Turn (left / right)", self._camera_azimuth_var, -180.0, 180.0, 1.0),
            ("elevation", "Height (up / down)", self._camera_elevation_var, -85.0, 85.0, 1.0),
            ("zoom", "Zoom (near / far)", self._camera_zoom_var, 0.35, 3.0, 0.01),
        ):
            holder = tk.Frame(parent, bg=theme.surface)
            holder.pack(fill="x", pady=(4, 0))
            head = tk.Frame(holder, bg=theme.surface)
            head.pack(fill="x")
            tk.Label(
                head, text=label, bg=theme.surface, fg=theme.foreground,
                font=self._type.font("2xs", weight="bold"), anchor="w",
            ).pack(side="left")
            value_label = tk.Label(
                head, text="", bg=theme.surface, fg=theme.text_dim,
                font=self._type.font("2xs"), anchor="e",
            )
            value_label.pack(side="right")
            self._camera_labels[key] = value_label
            scale = tk.Scale(
                holder, from_=low, to=high, resolution=step, orient="horizontal",
                variable=variable, showvalue=False, highlightthickness=0, borderwidth=0,
                troughcolor=theme.surface_raised, bg=theme.surface,
                activebackground=theme.primary, length=320,
            )
            scale.pack(fill="x", pady=(1, 0))
            self._camera_scales.append(scale)
            variable.trace_add("write", _camera_trace(self._on_camera, key, variable))

        self._button(
            parent, "Reset the camera", self._reset_camera, variant="primary"
        ).pack(fill="x", pady=(6, 0))
        self._sync_camera_labels()

    def _on_camera(self, key: str, variable: tk.DoubleVar) -> None:
        if self._syncing:
            return
        try:
            value = float(variable.get())
        except (tk.TclError, ValueError):
            return
        if key == "azimuth":
            self._camera_azimuth = value
        elif key == "elevation":
            self._camera_elevation = value
        else:
            self._camera_zoom = value
        self._sync_camera_labels()
        self._schedule_render()

    def _sync_camera_labels(self) -> None:
        labels = getattr(self, "_camera_labels", {})
        values = {
            "azimuth": f"{self._camera_azimuth:.0f}°",
            "elevation": f"{self._camera_elevation:.0f}°",
            "zoom": f"{self._camera_zoom:.2f}×",
        }
        for key, text in values.items():
            label = labels.get(key)
            if label is not None:
                try:
                    label.configure(text=text)
                except tk.TclError:
                    pass

    def _set_camera(
        self, *, azimuth: float | None = None, elevation: float | None = None, zoom: float | None = None
    ) -> None:
        """Move the camera and push the new values onto the sliders, then render."""
        if azimuth is not None:
            self._camera_azimuth = max(-180.0, min(180.0, azimuth))
        if elevation is not None:
            self._camera_elevation = max(-85.0, min(85.0, elevation))
        if zoom is not None:
            self._camera_zoom = max(0.35, min(3.0, zoom))
        self._syncing = True
        try:
            self._camera_azimuth_var.set(round(self._camera_azimuth, 2))
            self._camera_elevation_var.set(round(self._camera_elevation, 2))
            self._camera_zoom_var.set(round(self._camera_zoom, 2))
        except (tk.TclError, AttributeError):
            pass
        finally:
            self._syncing = False
        self._sync_camera_labels()
        self._render_now()

    def _apply_camera_preset(self, key: str) -> None:
        try:
            azimuth, elevation = self.service.body_camera_preset(key)
        except KeyError as error:
            self._set_status(str(error))
            return
        self._set_camera(azimuth=azimuth, elevation=elevation)
        self._set_status(f"Camera set to {key}.")

    def _reset_camera(self) -> None:
        self._set_camera(azimuth=0.0, elevation=0.0, zoom=1.0)
        self._set_status("Camera reset to the front view.")

    # -- drag to orbit ----------------------------------------------------

    def _on_preview_press(self, event: Any) -> None:
        self._drag_origin = (event.x, event.y)
        self._drag_angles = (self._camera_azimuth, self._camera_elevation)
        try:
            self._canvas.configure(cursor="fleur")
        except tk.TclError:
            pass

    def _on_preview_drag(self, event: Any) -> None:
        if self._drag_origin is None or self._drag_angles is None:
            return
        start_x, start_y = self._drag_origin
        base_azimuth, base_elevation = self._drag_angles
        self._camera_azimuth = max(-180.0, min(180.0, base_azimuth + (event.x - start_x) * 0.6))
        self._camera_elevation = max(
            -85.0, min(85.0, base_elevation - (event.y - start_y) * 0.6)
        )
        self._syncing = True
        try:
            self._camera_azimuth_var.set(round(self._camera_azimuth, 2))
            self._camera_elevation_var.set(round(self._camera_elevation, 2))
        except (tk.TclError, AttributeError):
            pass
        finally:
            self._syncing = False
        self._sync_camera_labels()
        self._schedule_render()

    def _on_preview_release(self, _event: Any) -> None:
        self._drag_origin = None
        self._drag_angles = None
        try:
            self._canvas.configure(cursor="")
        except tk.TclError:
            pass

    # -- the imported / edited model --------------------------------------

    def _build_model_section(self, parent: tk.Frame) -> None:
        """Import a ready-made human model, show it, and edit it part by part.

        The user asked for exactly this: open a mesh the project did not build,
        see it in the window, change the size or shape of any part, and add a
        missing part. The engine is :mod:`core.body_editor` (through the
        service); the window only collects the choice and shows the result.
        """
        theme = self.theme
        tk.Label(
            parent,
            text=(
                "Open a ready-made model (OBJ, STL, or PLY) to see and edit it here, "
                "or turn the built body into an editable model. Then pick a part and "
                "resize, move, or rotate it — or add a missing part."
            ),
            bg=theme.surface,
            fg=theme.text_dim,
            font=self._type.font("2xs"),
            anchor="w",
            justify="left",
            wraplength=340,
        ).pack(fill="x", pady=(0, 4))

        self._model_label = tk.Label(
            parent,
            text="Showing the built figure.",
            bg=theme.surface,
            fg=theme.text_dim,
            font=self._type.font("2xs"),
            anchor="w",
            justify="left",
            wraplength=340,
        )
        self._model_label.pack(fill="x", pady=(0, 4))

        self._button(parent, "Import a model…", self._import_model).pack(fill="x", pady=1)
        self._button(parent, "Use the built body as the model", self._use_figure_as_model).pack(
            fill="x", pady=1
        )
        self._button(parent, "Back to the built figure", self._clear_model).pack(fill="x", pady=1)

        tk.Label(
            parent,
            text="Part to edit",
            bg=theme.surface,
            fg=theme.foreground,
            font=self._type.font("2xs", weight="bold"),
            anchor="w",
        ).pack(fill="x", pady=(6, 0))
        self._region_var = tk.StringVar(value="")
        self._region_menu = tk.OptionMenu(parent, self._region_var, "")
        self._region_menu.configure(
            bg=theme.surface_raised,
            fg=theme.foreground,
            activebackground=theme.primary,
            activeforeground=theme.base,
            highlightthickness=0,
            borderwidth=0,
            anchor="w",
            font=self._type.font("2xs"),
        )
        self._region_menu["menu"].configure(bg=theme.surface_raised, fg=theme.foreground)
        self._region_menu.pack(fill="x", pady=(1, 4))
        self._fill_region_menu()

        for key, label, low, high, step in (
            ("scale_x", "Part width", 0.5, 2.0, 0.05),
            ("scale_y", "Part height", 0.5, 2.0, 0.05),
            ("scale_z", "Part depth", 0.5, 2.0, 0.05),
            ("move_x", "Move left / right", -0.3, 0.3, 0.005),
            ("move_y", "Move up / down", -0.3, 0.3, 0.005),
            ("move_z", "Move forward / back", -0.3, 0.3, 0.005),
            ("rotate_y", "Rotate the part", -90.0, 90.0, 1.0),
        ):
            holder = tk.Frame(parent, bg=theme.surface)
            holder.pack(fill="x", pady=(4, 0))
            head = tk.Frame(holder, bg=theme.surface)
            head.pack(fill="x")
            tk.Label(
                head, text=label, bg=theme.surface, fg=theme.foreground,
                font=self._type.font("2xs", weight="bold"), anchor="w",
            ).pack(side="left")
            value_label = tk.Label(
                head, text="", bg=theme.surface, fg=theme.text_dim,
                font=self._type.font("2xs"), anchor="e",
            )
            value_label.pack(side="right")
            self._edit_labels[key] = value_label
            variable = tk.DoubleVar(value=1.0 if key.startswith("scale") else 0.0)
            self._edit_vars[key] = variable
            self._edit_defaults[key] = variable.get()
            scale = tk.Scale(
                holder, from_=low, to=high, resolution=step, orient="horizontal",
                variable=variable, showvalue=False, highlightthickness=0, borderwidth=0,
                troughcolor=theme.surface_raised, bg=theme.surface,
                activebackground=theme.primary, length=320,
            )
            scale.pack(fill="x", pady=(1, 0))
            self._edit_scales.append(scale)
            variable.trace_add("write", _camera_trace(self._on_edit_slider, key, variable))
        self._sync_edit_labels()

        self._button(parent, "Apply to the chosen part", self._apply_part_edit, variant="primary").pack(
            fill="x", pady=(6, 0)
        )
        self._button(parent, "Reset the sliders", self._reset_edit_sliders).pack(fill="x", pady=1)

        tk.Label(
            parent,
            text="Add a missing part",
            bg=theme.surface,
            fg=theme.foreground,
            font=self._type.font("2xs", weight="bold"),
            anchor="w",
        ).pack(fill="x", pady=(8, 0))
        self._part_var = tk.StringVar(value="")
        self._part_menu = tk.OptionMenu(parent, self._part_var, "")
        self._part_menu.configure(
            bg=theme.surface_raised,
            fg=theme.foreground,
            activebackground=theme.primary,
            activeforeground=theme.base,
            highlightthickness=0,
            borderwidth=0,
            anchor="w",
            font=self._type.font("2xs"),
        )
        self._part_menu["menu"].configure(bg=theme.surface_raised, fg=theme.foreground)
        self._part_menu.pack(fill="x", pady=(1, 2))
        self._fill_part_menu()
        tk.Label(
            parent,
            text=(
                "Place it where the chosen part is and it fuses into the body as one "
                "skin. Widen 'Add size' for a larger piece."
            ),
            bg=theme.surface,
            fg=theme.text_dim,
            font=self._type.font("2xs"),
            anchor="w",
            justify="left",
            wraplength=340,
        ).pack(fill="x", pady=(0, 2))
        add_row = tk.Frame(parent, bg=theme.surface)
        add_row.pack(fill="x")
        tk.Label(
            add_row, text="Add size", bg=theme.surface, fg=theme.foreground,
            font=self._type.font("2xs"), anchor="w",
        ).pack(side="left")
        self._add_size_var = tk.DoubleVar(value=1.0)
        tk.Scale(
            add_row, from_=0.3, to=3.0, resolution=0.05, orient="horizontal",
            variable=self._add_size_var, showvalue=False, highlightthickness=0, borderwidth=0,
            troughcolor=theme.surface_raised, bg=theme.surface,
            activebackground=theme.primary, length=240,
        ).pack(side="right", fill="x", expand=True)
        self._button(parent, "Add the part to the body", self._add_part, variant="primary").pack(
            fill="x", pady=(4, 0)
        )

    def _fill_region_menu(self) -> None:
        menu = self._region_menu["menu"]
        menu.delete(0, "end")
        options = self.service.body_model_region_titles()
        if not options:
            return
        if not self._region_var.get():
            self._region_var.set(options[0][0])
        for key, title in options:
            menu.add_command(label=title, command=_binder(self._set_region, key))

    def _fill_part_menu(self) -> None:
        menu = self._part_menu["menu"]
        menu.delete(0, "end")
        options = self.service.body_model_part_shapes()
        if not options:
            return
        if not self._part_var.get():
            self._part_var.set(options[0][0])
        for key, title in options:
            menu.add_command(label=title, command=_binder(self._set_part_shape, key))

    def _set_region(self, key: str) -> None:
        self._region_var.set(key)
        self._reset_edit_sliders()

    def _set_part_shape(self, key: str) -> None:
        self._part_var.set(key)

    def _on_edit_slider(self, key: str, _variable: tk.DoubleVar) -> None:
        self._sync_edit_labels()

    def _sync_edit_labels(self) -> None:
        for key, label in getattr(self, "_edit_labels", {}).items():
            variable = self._edit_vars.get(key)
            if variable is None:
                continue
            try:
                value = float(variable.get())
            except (tk.TclError, ValueError):
                continue
            text = f"{value:.2f}" if key.startswith("scale") else f"{value:+.2f}"
            try:
                label.configure(text=text)
            except tk.TclError:
                pass

    def _reset_edit_sliders(self) -> None:
        for key, variable in self._edit_vars.items():
            try:
                variable.set(self._edit_defaults.get(key, 0.0))
            except tk.TclError:
                pass
        self._sync_edit_labels()

    def _import_model(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            parent=self.window,
            title="Open a human model",
            filetypes=[
                ("Human models", "*.obj *.stl *.ply"),
                ("Wavefront OBJ", "*.obj"),
                ("STL", "*.stl"),
                ("PLY", "*.ply"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._busy_model("Reading the model…", lambda: self.service.load_body_model(path))

    def _use_figure_as_model(self) -> None:
        self._busy_model(
            "Turning the built body into a model…",
            lambda: self.service.body_model_from_figure(self._params),
        )

    def _busy_model(self, message: str, job: Callable[[], Any]) -> None:
        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._set_status(f"Could not load the model: {error}")
                return
            self._mesh = result
            self._model_label.configure(text=self.service.body_model_summary(result))
            self._set_status("Model ready — pick a part to edit, or add a part.")
            self._render_now()

        self._run_background(message, job, finish)

    def _clear_model(self) -> None:
        self._mesh = None
        self._model_label.configure(text="Showing the built figure.")
        self._set_status("Back to the built figure.")
        self._render_now()

    def _apply_part_edit(self) -> None:
        if self._mesh is None:
            self._set_status("Import a model or turn the built body into one first.")
            return
        region_key = self._region_var.get()
        if not region_key:
            self._set_status("Pick the part to edit.")
            return
        scale: tuple[float, float, float] = (
            float(self._edit_vars["scale_x"].get()),
            float(self._edit_vars["scale_y"].get()),
            float(self._edit_vars["scale_z"].get()),
        )
        move: tuple[float, float, float] = (
            float(self._edit_vars["move_x"].get()),
            float(self._edit_vars["move_y"].get()),
            float(self._edit_vars["move_z"].get()),
        )
        rotate: tuple[float, float, float] = (0.0, float(self._edit_vars["rotate_y"].get()), 0.0)
        mesh = self._mesh
        params = self._params

        def job() -> Any:
            return self.service.edit_body_model_region(
                mesh, params, region_key, scale=scale, move=move, rotate=rotate
            )

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._set_status(f"Could not edit the part: {error}")
                return
            self._mesh = result
            self._model_label.configure(text=self.service.body_model_summary(result))
            self._set_status(f"Applied the edit to {region_key}.")
            self._reset_edit_sliders()
            self._render_now()

        self._run_background("Editing the part…", job, finish)

    def _add_part(self) -> None:
        if self._mesh is None:
            self._set_status("Import a model or turn the built body into one first.")
            return
        shape = self._part_var.get()
        if not shape:
            self._set_status("Pick the shape to add.")
            return
        centre = self._region_centre(self._region_var.get())
        try:
            size = 0.09 * float(self._add_size_var.get())
        except (tk.TclError, ValueError):
            size = 0.09
        mesh = self._mesh

        def job() -> Any:
            return self.service.add_body_model_part(mesh, shape=shape, centre=centre, size=size)

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._set_status(f"Could not add the part: {error}")
                return
            self._mesh = result
            self._model_label.configure(text=self.service.body_model_summary(result))
            self._set_status(f"Added a {shape} part.")
            self._render_now()

        self._run_background("Adding the part…", job, finish)

    def _region_centre(self, region_key: str) -> tuple[float, float, float]:
        """The centre of a named region, so a new part is added where it belongs."""
        for region in self.service.body_model_region_objects(self._params):
            if region.key == region_key:
                return (
                    (region.lo[0] + region.hi[0]) * 0.5,
                    (region.lo[1] + region.hi[1]) * 0.5,
                    (region.lo[2] + region.hi[2]) * 0.5,
                )
        return (0.0, self._params.height * 0.5, 0.0)

    def _build_pose_section(self, parent: tk.Frame) -> None:
        theme = self.theme
        tk.Label(
            parent,
            text="Choose a ready-made position, or describe one in words.",
            bg=theme.surface,
            fg=theme.text_dim,
            font=self._type.font("2xs"),
            anchor="w",
            justify="left",
            wraplength=340,
        ).pack(fill="x", pady=(0, 4))
        grid = tk.Frame(parent, bg=theme.surface)
        grid.pack(fill="x")
        for pose_key, title in self.service.body_poses():
            self._button(grid, title, _binder(self._set_pose, pose_key)).pack(
                side="top", fill="x", pady=1
            )

        field = tk.Frame(parent, bg=theme.surface)
        field.pack(fill="x", pady=(6, 0))
        self._pose_var = tk.StringVar(value="")
        entry = tk.Entry(
            field,
            textvariable=self._pose_var,
            relief="flat",
            borderwidth=1,
            highlightthickness=0,
            bg=theme.surface_raised,
            fg=theme.foreground,
            insertbackground=theme.foreground,
            font=self._type.font("2xs"),
        )
        entry.pack(fill="x", ipady=3)
        entry.bind("<Return>", lambda _event: self._apply_pose_text())
        self._button(field, "Apply the described pose", self._apply_pose_text, variant="primary").pack(
            fill="x", pady=(4, 0)
        )

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

        # The action row and the honesty note are packed **before** the expanding
        # preview, anchored to the bottom: a widget packed after a ``fill="both",
        # expand=True`` sibling is left with no space and is never mapped, which is
        # how Render / Use as the picture / Close had become unreachable.
        actions = tk.Frame(parent, bg=theme.base)
        actions.pack(side="bottom", fill="x", padx=16, pady=(0, 8))
        self._action_row = actions
        self._button(actions, "Render", self._render_now, variant="primary")
        self._button(actions, "Use as the picture", self._use_as_avatar)
        self._button(actions, "Reset the shape", self._reset_shape)
        self._button(actions, "Close", self._close)

        note = tk.Label(
            parent,
            text=(
                "A parametric figure drawn locally by the project's own CPU rasteriser: "
                "a subdivided, smooth-shaded mesh with per-part colour — not a licensed "
                "likeness, a photograph, or a commercial render."
            ),
            bg=theme.base,
            fg=theme.text_dim,
            font=self._type.font("2xs"),
            anchor="w",
            justify="left",
            wraplength=680,
        )
        note.pack(side="bottom", fill="x", padx=16, pady=(0, 8))

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
        # Click-and-drag on the picture turns the camera, so the viewpoint can be
        # moved the way a 3D program moves it.
        self._canvas.bind("<ButtonPress-1>", self._on_preview_press)
        self._canvas.bind("<B1-Motion>", self._on_preview_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_preview_release)

    # -- control helpers --------------------------------------------------

    def _button(self, parent: tk.Misc, text: str, command: Callable[[], None], *, variant: str = "secondary") -> tk.Button:
        """A themed button, **placed** in its row.

        The control is packed here so a caller that does not chain its own
        ``.pack(...)`` still shows it: an unplaced Tk widget has no geometry
        manager, so it is never drawn and can never be clicked -- which is how the
        body window's Render / Use as the picture / templates / views controls all
        ended up invisible. A caller that packs it again only re-applies the
        geometry (``pack`` is idempotent).
        """
        from gui.components import button

        widget = button(
            parent, text=text, command=command, theme=self.theme,
            typography=self._type, variant=variant,
        )
        if not widget.winfo_manager():
            widget.pack(side="left", padx=(0, 6), pady=1)
        return widget

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

    def _set_pose(self, key: str) -> None:
        try:
            self._params = self.service.set_body_pose(self._params, key)
        except KeyError as error:
            self._set_status(str(error))
            return
        self._set_status(f"Pose set to {key}.")
        self._render_now()

    def _apply_pose_text(self) -> None:
        try:
            text = self._pose_var.get()
        except tk.TclError:
            return
        self._params, message = self.service.apply_body_pose_instruction(self._params, text)
        self._set_status(message)
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
        if self._closed:
            return
        if self._busy:
            # A render is already running (a fast drag emits many requests). Mark
            # the state dirty so the *latest* camera/pose is rendered as soon as
            # the current one finishes, instead of the last change being dropped.
            self._render_dirty = True
            return
        self._render_dirty = False
        params = self._params
        view = self._view
        width, height = _PREVIEW_WIDTH, _PREVIEW_HEIGHT
        azimuth = self._camera_azimuth
        elevation = self._camera_elevation
        zoom = self._camera_zoom

        mesh = self._mesh

        def job() -> bytes:
            return self.service.render_body(
                params,
                view=view,
                width=width,
                height=height,
                azimuth=azimuth,
                elevation=elevation,
                zoom=zoom,
                mesh=mesh,
            )

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
        if self._render_dirty and not self._closed:
            self._schedule_render()

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


def _edit_trace(action: Callable[[str, tk.DoubleVar], None], key: str) -> Callable[..., None]:
    """A trace callback for a model-edit slider (passes the key, ignores the rest)."""
    def handler(_name: str, _index: str, _op: str) -> None:
        variable = None
        action(key, variable)  # type: ignore[arg-type]

    return handler


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


def _camera_trace(
    handler: Callable[[str, tk.DoubleVar], None], key: str, var: tk.DoubleVar
) -> Callable[..., None]:
    """A ``trace_add`` callback for a camera slider (the value is read from the var)."""

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
