"""Scene Studio: build, pose, and render a 3D figure locally, in one window.

Reached from the chat composer's ＋ menu, this page is the project's own
**local 3D figure studio**:

* **Figures** -- add a built-in parametric humanoid (adult, athletic, slim,
  child) that is generated in code and bound to a real skeleton, so it can be
  posed.
* **Props** -- add a ground plane, a platform, a cube, a column, and so on.
* **Pose** -- pick a bone and rotate it, or apply a named pose (wave, walk, sit,
  T-pose, reach); the mesh deforms with real linear-blend skinning.
* **Camera** -- frame the scene, or orbit the camera around it.
* **Render** -- draw the scene with the project's own CPU software rasteriser
  (a real z-buffered rasteriser with Blinn-Phong shading) and save a PNG.
* **Command box** -- type a plain instruction ("add figure athletic", "pose
  forearm.L 0 0 -90", "render 900 640"); the same interpreter the CLI uses runs it.

The page is a thin view over :class:`gui.scene3d.SceneStudioService`, so the whole
workflow is testable with no display. Tk is imported lazily with the rest of the
window, so the CLI never touches it. Rendering happens on a worker thread whose
result is handed back through a queue the UI thread drains, so the window never
freezes.

Everything stays honest and local: the figures are original and generated here,
nothing is downloaded, no asset store is contacted, and the renderer is a CPU
rasteriser -- it is not a GPU renderer or a commercial render engine, and the
page says so.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import tkinter as tk
from tkinter import filedialog, ttk

from core.scene3d import Vec3
from core.scene3d_io import SceneIOError
from gui.app import RADIUS_LARGE, RoundedBox
from gui.chat_composer import ChatComposer
from gui.image_preview import load_scaled
from gui.scene3d import (
    HONESTY_LINE,
    POSE_PRESETS,
    SceneStudioError,
    SceneStudioService,
    command_help,
)
from gui.typography import resolve
from gui.window_controls import install_window_controls

TITLE = "Scene Studio"
SUBTITLE = (
    "Build a figure, add props, pose the skeleton, move the camera, and render — "
    "all on this machine, with no GPU, no download, and no asset store."
)

#: The viewport column is this fraction of the window width.
VIEWPORT_FRACTION = 0.42

#: Horizontal padding, in pixels.
SECTION_PAD = 16
VIEWPORT_PAD = 10

#: The transcript/status column reads as a centred column, capped like the chat's.
_COLUMN_MAX = 860

#: A message taller than this scrolls inside its own card rather than the column.
_MAX_BUBBLE_LINES = 26

#: The render size used for the live viewport preview.
VIEWPORT_WIDTH = 640
VIEWPORT_HEIGHT = 460


class SceneStudioDialog:
    """A window that builds, poses, and renders a local 3D scene."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        service: SceneStudioService,
        *,
        theme: Any,
        font: str,
        mono: str,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self.service = service
        self.theme = theme
        self.font = font
        self.mono = mono
        self._type = resolve(parent, ui_family=font, mono_family=mono)
        self._on_close = on_close
        self._closed = False
        self._busy = False
        self._photo: tk.PhotoImage | None = None
        self._viewport_source = ""
        self._viewport_fit_width = 0
        self._object_names: list[str] = []
        self._bone_names: list[str] = []
        self._row_frames: list[tuple[str, str, tk.Frame, tk.Text, RoundedBox | None]] = []

        self._results: queue.Queue[
            tuple[Callable[[Any, Exception | None], None], Any, Exception | None]
        ] = queue.Queue()
        self._progress: queue.Queue[str] = queue.Queue()

        self._command_var = tk.StringVar(value="")

        self.window = tk.Toplevel(parent)
        self.window.title(TITLE)
        self.window.configure(bg=theme.base)
        self.window.geometry("1280x760")
        self.window.minsize(1040, 660)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._build()
        self._refresh_all()
        self._add_message("note", HONESTY_LINE)
        self.window.after(120, self._render_viewport)

    # -- layout -----------------------------------------------------------

    def _build(self) -> None:
        self._build_header()
        theme = self.theme
        body = tk.Frame(self.window, bg=theme.base)
        body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=0, minsize=300)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, weight=0)
        self._body = body
        body.bind("<Configure>", self._on_body_configure)
        self._build_sidebar(body)
        self._build_centre(body)
        self._build_viewport(body)
        self._layout_columns()

    def _build_header(self) -> None:
        theme = self.theme
        bar = tk.Frame(self.window, bg=theme.titlebar)
        bar.pack(fill="x")
        inner = tk.Frame(bar, bg=theme.titlebar)
        inner.pack(fill="x", padx=16, pady=(12, 8))
        titles = tk.Frame(inner, bg=theme.titlebar)
        titles.pack(side="left", anchor="w")
        tk.Label(
            titles, text=TITLE, bg=theme.titlebar, fg=theme.contrast,
            font=self._type.font("heading3", weight="bold"), anchor="w",
        ).pack(anchor="w")
        tk.Label(
            titles, text=SUBTITLE, bg=theme.titlebar, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=900,
        ).pack(anchor="w", pady=(2, 0))
        self._spinner = ttk.Progressbar(inner, mode="indeterminate", length=120)
        self._window_controls = install_window_controls(
            self.window, inner, theme=theme, typography=self._type, on_close=self._close
        )
        tk.Frame(self.window, bg=theme.titlebar_edge, height=1).pack(fill="x")

    def _section_label(self, parent: tk.Misc, text: str) -> None:
        tk.Label(
            parent, text=text.upper(), bg=self.theme.surface, fg=self.theme.text_dim,
            font=self._type.font("2xs", weight="bold"), anchor="w",
        ).pack(fill="x", padx=14, pady=(10, 2))

    def _button(self, parent: tk.Misc, text: str, command: Any) -> tk.Button:
        return tk.Button(
            parent, text=text, command=command, relief="flat", borderwidth=0,
            highlightthickness=0, bg=self.theme.surface_raised, fg=self.theme.foreground,
            activebackground=self.theme.interactive_active, activeforeground=self.theme.contrast,
            font=self._type.font("2xs", weight="bold"), padx=10, pady=4, cursor="hand2",
        )

    # -- sidebar ----------------------------------------------------------

    def _build_sidebar(self, parent: tk.Frame) -> None:
        theme = self.theme
        sidebar = tk.Frame(parent, bg=theme.surface)
        sidebar.grid(row=0, column=0, sticky="nsew")

        brand = tk.Frame(sidebar, bg=theme.surface)
        brand.pack(fill="x", padx=14, pady=(14, 4))
        dot = tk.Canvas(brand, width=24, height=24, bg=theme.surface, highlightthickness=0)
        dot.create_oval(1, 1, 23, 23, fill=theme.primary, outline="")
        dot.create_text(
            12, 13, text="◈", fill=theme.primary_foreground,
            font=self._type.font("sm", weight="bold"),
        )
        dot.pack(side="left")
        tk.Label(
            brand, text="Scene Studio", bg=theme.surface, fg=theme.contrast,
            font=self._type.font("body", weight="bold"),
        ).pack(side="left", padx=(8, 0))

        actions = tk.Frame(sidebar, bg=theme.surface)
        actions.pack(fill="x", padx=12, pady=(6, 2))
        self._new_button = self._button(actions, "New scene", self._on_new_scene)
        self._new_button.pack(fill="x", pady=(0, 4))
        self._open_button = self._button(actions, "Open a model or scene…", self._on_open)
        self._open_button.pack(fill="x", pady=(0, 4))
        self._save_button = self._button(actions, "Save the scene…", self._on_save)
        self._save_button.pack(fill="x", pady=(0, 4))
        self._export_button = self._button(actions, "Export OBJ…", self._on_export)
        self._export_button.pack(fill="x")

        self._section_label(sidebar, "Figures")
        figures = tk.Frame(sidebar, bg=theme.surface)
        figures.pack(fill="x", padx=12, pady=(0, 4))
        for preset in self.service.figure_preset_names():
            self._button(figures, preset.capitalize(), lambda p=preset: self._on_add_figure(p)).pack(
                fill="x", pady=(0, 3)
            )

        self._section_label(sidebar, "Props")
        props = tk.Frame(sidebar, bg=theme.surface)
        props.pack(fill="x", padx=12, pady=(0, 4))
        for key in self.service.prop_names():
            self._button(props, key.capitalize(), lambda k=key: self._on_add_prop(k)).pack(
                fill="x", pady=(0, 3)
            )

        self._section_label(sidebar, "Poses")
        poses = tk.Frame(sidebar, bg=theme.surface)
        poses.pack(fill="x", padx=12, pady=(0, 12))
        for pose in POSE_PRESETS:
            self._button(poses, pose.capitalize(), lambda p=pose: self._on_pose_preset(p)).pack(
                fill="x", pady=(0, 3)
            )

    # -- centre: status transcript + command box --------------------------

    def _build_centre(self, parent: tk.Frame) -> None:
        theme = self.theme
        col = tk.Frame(parent, bg=theme.base)
        col.grid(row=0, column=1, sticky="nsew")

        self._build_composer(col)

        container = tk.Frame(col, bg=theme.base)
        container.pack(fill="both", expand=True)
        self._canvas = tk.Canvas(container, bg=theme.base, highlightthickness=0)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._scrollbar = tk.Scrollbar(container, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        self._transcript = tk.Frame(self._canvas, bg=theme.base)
        self._window = self._canvas.create_window((0, 0), window=self._transcript, anchor="nw")
        self._transcript.bind("<Configure>", self._on_transcript_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        for widget in (self._canvas, self._transcript):
            widget.bind("<MouseWheel>", self._on_mousewheel)
            widget.bind("<Button-4>", self._on_mousewheel)
            widget.bind("<Button-5>", self._on_mousewheel)

    def _build_composer(self, parent: tk.Frame) -> None:
        """The shared rounded composer card, used here as the command box."""
        composer = ChatComposer(
            parent, theme=self.theme, typography=self._type,
            placeholder="Command — e.g. \"add figure athletic\", \"pose wave\", \"render\"…",
            on_submit=self._on_command, outer_bg=self.theme.base, with_attach=False,
            variable=self._command_var,
        )
        composer.wrap.pack(side="bottom", fill="x", padx=self._column_pad(), pady=(8, 12))
        self._composer = composer
        self._command = composer.entry
        self._run_button = self._button(composer.right_tools, "Run", self._on_command)
        composer.add_tool(self._run_button)

    # -- viewport ---------------------------------------------------------

    def _build_viewport(self, parent: tk.Frame) -> None:
        theme = self.theme
        side = tk.Frame(parent, bg=theme.base)
        side.grid(row=0, column=2, sticky="nsew")
        self._side_column = side

        card = tk.Frame(side, bg=theme.surface)
        card.pack(fill="both", expand=True, padx=(0, SECTION_PAD), pady=(SECTION_PAD, SECTION_PAD))
        head = tk.Frame(card, bg=theme.surface)
        head.pack(fill="x", padx=VIEWPORT_PAD, pady=(VIEWPORT_PAD, 4))
        tk.Label(
            head, text="VIEWPORT", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs", weight="bold"),
        ).pack(side="left")
        self._viewport_label = tk.Label(
            head, text="—", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"),
        )
        self._viewport_label.pack(side="right")

        self._viewport = tk.Label(
            card, bg=theme.base, fg=theme.text_dim, font=self._type.font("2xs"),
            text="Render the scene to see it here.", wraplength=320, justify="center",
        )
        self._viewport.pack(fill="both", expand=True, padx=VIEWPORT_PAD, pady=(0, 6))

        camera = tk.Frame(card, bg=theme.surface)
        camera.pack(fill="x", padx=VIEWPORT_PAD, pady=(0, 4))
        self._button(camera, "⟲ Orbit", lambda: self._on_orbit(-30)).pack(side="left")
        self._button(camera, "Frame all", self._on_frame).pack(side="left", padx=(6, 0))
        self._button(camera, "Orbit ⟳", lambda: self._on_orbit(30)).pack(side="left", padx=(6, 0))

        render_row = tk.Frame(card, bg=theme.surface)
        render_row.pack(fill="x", padx=VIEWPORT_PAD, pady=(0, VIEWPORT_PAD))
        self._render_button = self._button(render_row, "Render and save…", self._on_render_save)
        self._render_button.pack(side="left")
        self._wireframe_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            render_row, text="Wireframe", variable=self._wireframe_var,
            command=self._render_viewport, bg=theme.surface, fg=theme.text_secondary,
            activebackground=theme.surface, activeforeground=theme.contrast,
            selectcolor=theme.surface_raised, highlightthickness=0, borderwidth=0,
            font=self._type.font("2xs"),
        ).pack(side="right")

        self._section_label(side, "Objects")
        list_wrap = tk.Frame(side, bg=theme.surface)
        list_wrap.pack(fill="x", padx=12, pady=(0, 6))
        self._object_list = tk.Listbox(
            list_wrap, bg=theme.base, fg=theme.foreground, relief="flat",
            highlightthickness=0, borderwidth=0, selectbackground=theme.interactive_active,
            selectforeground=theme.contrast, font=self._type.font("xs"), activestyle="none",
            exportselection=False, height=6,
        )
        self._object_list.pack(fill="x")
        self._object_list.bind("<<ListboxSelect>>", self._on_select_object)

        self._section_label(side, "Bones")
        bone_wrap = tk.Frame(side, bg=theme.surface)
        bone_wrap.pack(fill="x", padx=12, pady=(0, 6))
        self._bone_list = tk.Listbox(
            bone_wrap, bg=theme.base, fg=theme.foreground, relief="flat",
            highlightthickness=0, borderwidth=0, selectbackground=theme.interactive_active,
            selectforeground=theme.contrast, font=self._type.font("xs"), activestyle="none",
            exportselection=False, height=6,
        )
        self._bone_list.pack(fill="x")
        self._bone_list.bind("<<ListboxSelect>>", self._on_select_bone)

        pose_row = tk.Frame(side, bg=theme.surface)
        pose_row.pack(fill="x", padx=12, pady=(0, 12))
        self._button(pose_row, "− X", lambda: self._rotate_bone("x", -15)).pack(side="left")
        self._button(pose_row, "+ X", lambda: self._rotate_bone("x", 15)).pack(side="left", padx=(4, 0))
        self._button(pose_row, "− Y", lambda: self._rotate_bone("y", -15)).pack(side="left", padx=(4, 0))
        self._button(pose_row, "+ Y", lambda: self._rotate_bone("y", 15)).pack(side="left", padx=(4, 0))
        self._button(pose_row, "− Z", lambda: self._rotate_bone("z", -15)).pack(side="left", padx=(4, 0))
        self._button(pose_row, "+ Z", lambda: self._rotate_bone("z", 15)).pack(side="left", padx=(4, 0))
        self._button(side, "Clear the pose", self._on_clear_pose).pack(fill="x", padx=12, pady=(0, 12))

    def _on_body_configure(self, _event: tk.Event | None = None) -> None:
        self._layout_columns()

    def _layout_columns(self) -> None:
        """Pin the viewport column to ``VIEWPORT_FRACTION`` of the window width."""
        try:
            width = self.window.winfo_width()
        except tk.TclError:
            return
        if width <= 1:
            return
        self._body.columnconfigure(
            2,
            minsize=max(280, int(width * VIEWPORT_FRACTION) + 2 * SECTION_PAD + 2 * VIEWPORT_PAD),
        )
        if self._viewport_source:
            self.window.after_idle(self._refit_viewport)

    def _refit_viewport(self) -> None:
        if not self._viewport_source:
            return
        self.window.update_idletasks()
        width = self._viewport.winfo_width()
        if width > 1 and width != self._viewport_fit_width:
            self._show_viewport(self._viewport_source)

    def _show_viewport(self, path: str) -> None:
        self._viewport_source = path
        width = self._viewport.winfo_width()
        if width <= 1:
            width = 340
        photo = load_scaled(path, width)
        if photo is None:
            self._photo = None
            self._viewport.configure(image="", text="The render could not be shown.")
            return
        self._photo = photo
        self._viewport.configure(image=photo, text="")
        self._viewport_fit_width = width

    # -- column geometry --------------------------------------------------

    def _column_pad(self) -> int:
        try:
            width = self._transcript.winfo_width() or self._canvas.winfo_width()
        except (tk.TclError, AttributeError):
            width = _COLUMN_MAX
        if width <= _COLUMN_MAX:
            return 16
        return max(16, (width - _COLUMN_MAX) // 2)

    def _available_column(self) -> int:
        try:
            width = self._canvas.winfo_width()
        except (tk.TclError, AttributeError):
            return _COLUMN_MAX
        if width <= 2:
            return _COLUMN_MAX
        return max(160, min(_COLUMN_MAX, width - 2 * self._column_pad()))

    def _bubble_fonts(self):
        from tkinter import font as tkfont

        return (
            tkfont.Font(font=self._type.font("body")),
            tkfont.Font(font=self._type.mono("code")),
        )

    def _bubble_content_width(self, text: str, *, role: str) -> int:
        normal, mono = self._bubble_fonts()
        widest = 0
        for line in text.splitlines() or [""]:
            widest = max(widest, normal.measure(line))
        del mono
        if role == "user":
            return max(80, min(widest, int(_COLUMN_MAX * 0.84) - 52))
        return max(120, min(widest, self._available_column()))

    def _bubble_text_width(self, text: str, *, role: str) -> int:
        normal, _mono = self._bubble_fonts()
        char_px = max(1, normal.measure("0"))
        content = self._bubble_content_width(text, role=role)
        return max(1, content // char_px)

    def _bubble_card_width(self, text: str) -> int:
        return self._bubble_content_width(text, role="user") + 52

    # -- transcript -------------------------------------------------------

    def _add_message(
        self, role: str, text: str, *, title: str = "", mono_body: bool = False
    ) -> tk.Text:
        theme = self.theme
        pad = self._column_pad()
        row = tk.Frame(self._transcript, bg=theme.base)
        row.pack(fill="x", padx=(pad, pad), pady=6)
        card: RoundedBox | None = None

        if role == "user":
            card = RoundedBox(
                row, bg=theme.user_bubble, fill=theme.user_bubble, outline=theme.user_bubble,
                corner=RADIUS_LARGE, parent_bg=theme.base,
            )
            card.pack(side="right", anchor="e")
            widget = tk.Text(
                card.inner, height=1, wrap="word", relief="flat", borderwidth=0,
                highlightthickness=0, bg=theme.user_bubble, fg=theme.user_bubble_fg,
                font=self._type.font("body"), padx=14, pady=8, cursor="xterm", takefocus=1, width=0,
            )
            widget.insert("1.0", text)
            widget.pack(fill="both", expand=True)
            card.configure(width=self._bubble_card_width(text))
            widget.configure(width=self._bubble_text_width(text, role="user"))
        else:
            head = tk.Frame(row, bg=theme.base)
            head.pack(fill="x", pady=(0, 4))
            if role in ("assistant", "scene"):
                self._avatar(head)
                padx = (6, 0)
            else:
                padx = (0, 0)
            tk.Label(
                head, text=title or ("Scene Studio" if role != "note" else "Note"),
                bg=theme.base, fg=theme.text_dim, font=self._type.font("xs", weight="bold"),
                anchor="w",
            ).pack(side="left", padx=padx)
            widget = tk.Text(
                row, height=1, wrap="word", relief="flat", borderwidth=0, highlightthickness=0,
                bg=theme.base, fg=theme.foreground,
                font=self._type.mono("code") if mono_body else self._type.font("body"),
                padx=0, pady=0, cursor="xterm", takefocus=1,
            )
            widget.insert("1.0", text)
            widget.pack(anchor="w")
            widget.configure(width=self._bubble_text_width(text, role=role))

        self._row_frames.append((role, text, row, widget, card))
        self._make_read_only(widget)
        for wheel in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(wheel, self._on_bubble_wheel, add="+")
        self._fit_height(widget)
        self._scroll_to_bottom()
        return widget

    def _avatar(self, parent: tk.Misc) -> None:
        badge = tk.Canvas(parent, width=24, height=24, bg=self.theme.base, highlightthickness=0)
        badge.create_oval(1, 1, 23, 23, fill=self.theme.primary, outline="")
        badge.create_text(
            12, 13, text="◈", fill=self.theme.primary_foreground,
            font=self._type.font("sm", weight="bold"),
        )
        badge.pack(side="left")

    def _fit_height(self, widget: tk.Text) -> None:
        widget.update_idletasks()
        measured = widget.count("1.0", "end", "displaylines")
        lines = int(measured[0]) if measured else 1
        widget.configure(height=min(max(lines, 1), _MAX_BUBBLE_LINES))

    @staticmethod
    def _make_read_only(widget: tk.Text) -> None:
        def block(event: tk.Event) -> str:
            if int(event.state) & 0x0004 or int(event.state) & 0x0008:
                return ""
            if event.keysym in {"Left", "Right", "Up", "Down", "Home", "End", "Prior", "Next"}:
                return ""
            return "break"

        for sequence in ("<Key>", "<<Paste>>", "<<Cut>>", "<<Clear>>"):
            widget.bind(sequence, block)

    def _on_bubble_wheel(self, event: tk.Event) -> str:
        widget = event.widget
        spans = widget.yview() if isinstance(widget, tk.Text) else (0.0, 1.0)
        if spans[0] <= 0.0 and spans[1] >= 1.0:
            self._on_mousewheel(event)
            return "break"
        return ""

    def _on_transcript_configure(self, _event: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._toggle_scrollbar()

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)
        self._reflow()

    def _reflow(self) -> None:
        pad = self._column_pad()
        for _role, _text, row, _widget, _card in self._row_frames:
            row.pack_configure(padx=(pad, pad))
        for role, text, _row, widget, card in self._row_frames:
            try:
                if card is not None:
                    card.configure(width=self._bubble_card_width(text))
                widget.configure(width=self._bubble_text_width(text, role=role))
            except tk.TclError:
                continue
            self._fit_height(widget)

    def _toggle_scrollbar(self) -> None:
        first, last = self._canvas.yview()
        if first <= 0.0 and last >= 1.0:
            self._scrollbar.pack_forget()
        else:
            self._scrollbar.pack(side="right", fill="y")

    def _on_mousewheel(self, event: tk.Event) -> None:
        delta = -1 * (event.delta // 120) if getattr(event, "delta", 0) else (
            -1 if getattr(event, "num", 0) == 4 else 1
        )
        self._canvas.yview_scroll(delta, "units")
        self._toggle_scrollbar()

    def _scroll_to_bottom(self) -> None:
        self._canvas.update_idletasks()
        self._canvas.yview_moveto(1.0)
        self._toggle_scrollbar()

    # -- background work --------------------------------------------------

    def _run_background(
        self,
        message: str,
        job: Callable[[], Any],
        finish: Callable[[Any, Exception | None], None],
    ) -> None:
        self._busy = True
        self._set_status(message)
        self._set_busy_buttons(False)
        self._spinner.pack(side="right", padx=8)
        self._spinner.start(12)

        def run() -> None:
            try:
                result = job()
                self._results.put((finish, result, None))
            except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
                self._results.put((finish, None, error))

        threading.Thread(target=run, daemon=True).start()
        self.window.after(60, self._pump_results)

    def _pump_results(self) -> None:
        latest = ""
        while True:
            try:
                latest = self._progress.get_nowait()
            except queue.Empty:
                break
        if latest:
            self._set_status(latest)
        try:
            finish, result, error = self._results.get_nowait()
        except queue.Empty:
            if self._busy and not self._closed:
                self.window.after(60, self._pump_results)
            return
        self._busy = False
        self._spinner.stop()
        self._spinner.pack_forget()
        self._set_busy_buttons(True)
        finish(result, error)

    def _set_busy_buttons(self, enabled: bool) -> None:
        state: Literal["normal", "disabled"] = "normal" if enabled else "disabled"
        for button in (self._new_button, self._open_button, self._save_button,
                       self._export_button, self._render_button, self._run_button):
            button.configure(state=state)

    def _set_status(self, message: str, *, error: bool = False) -> None:
        self._viewport_label.configure(
            text=message[:48], fg=self.theme.danger if error else self.theme.text_secondary
        )

    # -- state ------------------------------------------------------------

    def _refresh_all(self) -> None:
        self._refresh_objects()
        self._refresh_bones()

    def _refresh_objects(self) -> None:
        self._object_list.delete(0, "end")
        self._object_names = []
        for name, kind, visible in self.service.object_summaries():
            marker = "" if visible else "  (hidden)"
            self._object_list.insert("end", f"{name}  ·  {kind}{marker}")
            self._object_names.append(name)

    def _refresh_bones(self) -> None:
        self._bone_list.delete(0, "end")
        self._bone_names = []
        selected = self._selected_object()
        if selected is None:
            return
        for bone in self.service.bone_names(selected):
            self._bone_list.insert("end", bone)
            self._bone_names.append(bone)

    def _selected_object(self) -> str | None:
        selection = self._object_list.curselection()
        if not selection:
            return None
        index = int(selection[0])
        if 0 <= index < len(self._object_names):
            return self._object_names[index]
        return None

    def _selected_bone(self) -> str | None:
        selection = self._bone_list.curselection()
        if not selection:
            return None
        index = int(selection[0])
        if 0 <= index < len(self._bone_names):
            return self._bone_names[index]
        return None

    def _on_select_object(self, _event: tk.Event | None = None) -> None:
        self._refresh_bones()

    def _on_select_bone(self, _event: tk.Event | None = None) -> None:
        return

    # -- actions ----------------------------------------------------------

    def _on_new_scene(self) -> None:
        self.service.new_scene("Untitled scene")
        self._refresh_all()
        self._add_message("note", "Started a new scene with a ground plane.")
        self._render_viewport()

    def _on_add_figure(self, preset: str) -> None:
        obj = self.service.add_figure(preset)
        self._refresh_all()
        self._add_message("note", f"Added the **{obj.name}** figure ({preset}).")
        self.service.frame_camera()
        self._render_viewport()

    def _on_add_prop(self, key: str) -> None:
        obj = self.service.add_prop(key)
        self._refresh_all()
        self._add_message("note", f"Added the **{obj.name}** prop.")
        self._render_viewport()

    def _on_pose_preset(self, preset: str) -> None:
        target = self._selected_object() or self._first_figure()
        if target is None:
            self._add_message("note", "Add a figure first, then apply a pose.", title="Problem")
            return
        try:
            self.service.apply_pose_preset(target, preset)
        except SceneStudioError as error:
            self._add_message("note", str(error), title="Problem")
            return
        self._add_message("note", f"Applied the **{preset}** pose to {target}.")
        self._render_viewport()

    def _on_clear_pose(self) -> None:
        target = self._selected_object() or self._first_figure()
        if target is None:
            return
        try:
            self.service.clear_pose(target)
        except (SceneStudioError, KeyError):
            return
        self._add_message("note", f"Cleared the pose of {target}.")
        self._render_viewport()

    def _rotate_bone(self, axis: str, degrees: float) -> None:
        target = self._selected_object() or self._first_figure()
        bone = self._selected_bone()
        if target is None or bone is None:
            self._add_message("note", "Select a figure and a bone first.", title="Problem")
            return
        obj = self.service.scene.find_object(target)
        current = obj.pose.get(bone, Vec3())
        delta = Vec3(
            degrees if axis == "x" else 0.0,
            degrees if axis == "y" else 0.0,
            degrees if axis == "z" else 0.0,
        )
        try:
            self.service.pose_bone(target, bone, current + delta)
        except SceneStudioError as error:
            self._add_message("note", str(error), title="Problem")
            return
        self._render_viewport()

    def _first_figure(self) -> str | None:
        for obj in self.service.scene.objects:
            if obj.is_figure():
                return obj.name
        return None

    def _on_frame(self) -> None:
        self.service.frame_camera()
        self._render_viewport()

    def _on_orbit(self, degrees: float) -> None:
        self.service.orbit_camera(degrees)
        self._render_viewport()

    def _on_render_save(self) -> None:
        default = f"{Path(self.service.scene.name).stem or 'scene'}.png"
        path = filedialog.asksaveasfilename(
            parent=self.window, title="Save the render", defaultextension=".png",
            initialfile=default, filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return
        self._render_viewport(save_path=path, announce=True)

    def _render_viewport(self, save_path: str | None = None, announce: bool = False) -> None:
        """Render the scene on a worker thread and show (and optionally save) it."""
        if not self.service.scene.visible_objects():
            self._viewport.configure(image="", text="Add a figure or a prop to see it here.")
            self._photo = None
            return
        wireframe = self._wireframe_var.get()
        target = save_path or str(
            self.service.renders_dir / "_viewport.png"
        )

        def job() -> Any:
            return self.service.render(
                width=VIEWPORT_WIDTH,
                height=VIEWPORT_HEIGHT,
                wireframe=wireframe,
                path=target,
            )

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("note", f"Could not render: {error}", title="Problem")
                return
            if not result.ok:
                self._set_status(result.message, error=True)
                return
            self._show_viewport(result.path)
            if announce:
                self._add_message("note", result.summary(), title="Render", mono_body=True)

        self._run_background("Rendering…", job, finish)

    def _on_command(self) -> None:
        text = self._command.get("1.0", "end-1c").strip()
        if not text:
            return
        self._command_var.set("")
        self._add_message("user", text)
        output = self.service.run_command(text)
        self._add_message("note", output, title="Command", mono_body="\n" in output)
        self._refresh_all()
        self._render_viewport()

    # -- files ------------------------------------------------------------

    def _on_open(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.window, title="Open a model or scene",
            filetypes=[
                ("3D models and scenes", "*.obj *.faydscene"),
                ("Wavefront OBJ", "*.obj"),
                ("Scene Studio scenes", "*.faydscene"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._load(path)

    def _load(self, path: str) -> None:
        self._add_message("note", f"Opening **{Path(path).name}**…", title="Working")

        def job() -> Any:
            return self.service.open_model(path)

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("note", f"Could not open the file: {error}", title="Problem")
                return
            if not result.ok:
                self._add_message("note", result.message, title="Problem")
                return
            self._refresh_all()
            self._add_message("note", result.summary(), title="Opened")
            self.service.frame_camera()
            self._render_viewport()

        self._run_background(f"Reading {Path(path).name}…", job, finish)

    def _on_save(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.window, title="Save the scene", defaultextension=".faydscene",
            initialfile=f"{self.service.scene.name}.faydscene",
            filetypes=[("Scene Studio scene", "*.faydscene"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            saved = self.service.save_scene(path)
        except SceneIOError as error:
            self._add_message("note", f"Could not save the scene: {error}", title="Problem")
            return
        self._add_message("note", f"Saved the scene to {saved}.")

    def _on_export(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.window, title="Export the scene as OBJ", defaultextension=".obj",
            initialfile=f"{self.service.scene.name}.obj",
            filetypes=[("Wavefront OBJ", "*.obj"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            saved = self.service.export_obj(path)
        except SceneIOError as error:
            self._add_message("note", f"Could not export the scene: {error}", title="Problem")
            return
        self._add_message("note", f"Exported the scene to {saved}.")

    # -- lifecycle --------------------------------------------------------

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.window.destroy()
        finally:
            if self._on_close is not None:
                self._on_close()


__all__ = [
    "SceneStudioDialog",
    "SUBTITLE",
    "TITLE",
    "VIEWPORT_FRACTION",
    "VIEWPORT_HEIGHT",
    "VIEWPORT_WIDTH",
    "command_help",
]
