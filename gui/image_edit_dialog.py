"""The "Edit an image from text" screen: upload a picture and change it with words.

Reached from the ＋ menu, this page turns a written instruction into an edit of a
picture the user already has:

* **Choose an image…** -- loads a PNG or PPM into the page;
* an instruction box, e.g. "rotate 90 and brighten" or "make the person smile";
* a **backend** choice -- *Deterministic* (the dependency-free keyword editor) or
  *Learned* (an InstructPix2Pix diffusion model), which is the only one that can
  make a *semantic* change: an expression, a pose, the movement;
* **Apply** -- edits the current picture on a worker thread, shows the result, and
  saves it under the data folder;
* **Undo** -- steps back to the previous version (works for both backends);
* **Save a copy…** -- writes the current picture wherever the user chooses.

It is a thin view over :class:`gui.image_edit.ImageEditService`. The page says
plainly which backend ran: the deterministic editor lists the operations it
understood and names the words it did not, and the learned editor states the
model, the preset, and the sampler settings, and warns that a CPU edit is slow.

The class is Tk-only and imported lazily with the rest of the window, so the CLI
never touches it.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog
from typing import Any

from core.imagegen_diffusion import DEFAULT_EDIT_PRESET, EDIT_PRESETS
from core.optional_deps import FEATURE_IMAGE, pip_install_command

from gui.chat_composer import ChatComposer
from gui.image_edit import (
    BACKEND_DETERMINISTIC,
    BACKEND_LEARNED,
    EXAMPLE_INSTRUCTION,
    LEARNED_EXAMPLE_INSTRUCTION,
    READABLE_SUFFIXES,
    ImageEditService,
)
from gui.image_preview import load_scaled
from gui.typography import resolve
from gui.window_controls import install_window_controls

TITLE = "Edit an image from text"
SUBTITLE = (
    "Upload a picture, then describe the change in words. Choose Deterministic for "
    "exact pixel operations (rotate, crop, brightness…), or Learned for a real "
    "instruction-following model that can change an expression, a pose, or the "
    "movement — “make the person smile”, “change the pose to a wave”."
)
HONESTY_LINE = (
    "Deterministic edits are plain pixel operations on your file; the original is "
    "never modified, and words it does not recognise are reported, not guessed at. "
    "Learned edits are drawn by a diffusion model: they re-synthesise the picture, "
    "take tens of seconds to a few minutes on a CPU, and are a plausible edit, not a "
    "guaranteed faithful one."
)
INSTRUCTION_HINT = (
    "Deterministic examples: “rotate 90”, “crop to a square”, “grayscale”, "
    "“brighten 20%”.  Learned examples: “make the person smile”, “change the pose "
    "to a wave”, “turn the sky to a sunset”."
)

#: The preview column (and so the shown picture) is this fraction of the window
#: width, both before an edit and after one -- the same rule the generate page uses.
PREVIEW_FRACTION = 0.30

#: Horizontal padding, in pixels: the side padding of every section card, and the
#: extra padding inside the preview card. They are named so the preview column can
#: be sized so the *visible* preview box is exactly ``PREVIEW_FRACTION`` of the
#: window width.
SECTION_PAD = 20
PREVIEW_PAD = 12


class ImageEditDialog:
    """A window that edits an uploaded image from a written instruction."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        service: ImageEditService,
        *,
        theme: Any,
        font: str,
        mono: str,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self.edit_service = service
        self.theme = theme
        self.font = font
        self.mono = mono
        self._type = resolve(parent, ui_family=font, mono_family=mono)
        self._on_close = on_close
        self._busy = False
        self._session: Any = None
        #: Edits run on a worker thread; the result arrives through this queue and
        #: is applied on the main thread by the poll pump, as Tk requires.
        self._results: queue.Queue[tuple[str, Any, Exception | None]] = queue.Queue()
        self._photo: tk.PhotoImage | None = None
        self._preview_source: str = ""
        #: The column width the shown image was last fitted to, so a resize only
        #: rescales when the width actually changed.
        self._preview_fit_width: int = 0

        self._backend_var = tk.StringVar(value=BACKEND_DETERMINISTIC)
        self._preset_var = tk.StringVar(value=DEFAULT_EDIT_PRESET)

        self.window = tk.Toplevel(parent)
        self.window.title(TITLE)
        self.window.configure(bg=theme.base)
        self.window.geometry("1040x660")
        self.window.minsize(820, 600)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._build()

    # -- layout -----------------------------------------------------------

    def _build(self) -> None:
        theme = self.theme
        bar = tk.Frame(self.window, bg=theme.titlebar)
        bar.pack(fill="x")
        header = tk.Frame(bar, bg=theme.titlebar)
        header.pack(fill="x", padx=20, pady=(18, 8))
        title_row = tk.Frame(header, bg=theme.titlebar)
        title_row.pack(fill="x")
        tk.Label(
            title_row, text=TITLE, bg=theme.titlebar, fg=theme.contrast,
            font=self._type.font("heading1", weight="bold"), anchor="w",
        ).pack(side="left")
        self._window_controls = install_window_controls(
            self.window, title_row, theme=theme, typography=self._type, on_close=self._close
        )
        tk.Label(
            header, text=SUBTITLE, bg=theme.titlebar, fg=theme.text_secondary,
            font=self._type.font("xs"), anchor="w", justify="left", wraplength=980,
        ).pack(fill="x", pady=(4, 0))
        tk.Label(
            header, text=HONESTY_LINE, bg=theme.titlebar, fg=theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=980,
        ).pack(fill="x", pady=(6, 0))
        tk.Frame(self.window, bg=theme.titlebar_edge, height=1).pack(fill="x")

        # Two columns: the controls on the left, the preview on the right. The
        # preview column is pinned to PREVIEW_FRACTION of the window width (see
        # _layout_columns), so the picture that appears there is the same width
        # before an edit and after one.
        body = tk.Frame(self.window, bg=theme.base)
        body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0)
        controls = tk.Frame(body, bg=theme.base)
        controls.grid(row=0, column=0, sticky="nsew")
        side = tk.Frame(body, bg=theme.base)
        side.grid(row=0, column=1, sticky="nsew")
        self._controls_column = controls
        self._side_column = side
        self._body = body
        body.bind("<Configure>", self._on_body_configure)

        source_frame = self._section("1.  Your image", parent=controls)
        source_row = tk.Frame(source_frame, bg=theme.surface)
        source_row.pack(fill="x", padx=12, pady=(2, 4))
        self._choose_button = self._button(source_row, "Choose an image…", self._on_choose)
        self._choose_button.pack(side="left")
        self._source_label = tk.Label(
            source_row, text="No image chosen.", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("xs"), anchor="w",
        )
        self._source_label.pack(side="left", padx=(10, 0))
        tk.Label(
            source_frame, text=f"Supported now: {', '.join(READABLE_SUFFIXES)}.",
            bg=theme.surface, fg=theme.text_dim, font=self._type.font("2xs"), anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 8))

        options = self._section("2.  Editor", parent=controls)
        backend_row = tk.Frame(options, bg=theme.surface)
        backend_row.pack(fill="x", padx=12, pady=(2, 4))
        tk.Label(
            backend_row, text="Backend", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("xs"),
        ).pack(side="left", padx=(0, 6))
        for name, label in (
            (BACKEND_DETERMINISTIC, "Deterministic (default)"),
            (BACKEND_LEARNED, "Learned (model)"),
        ):
            tk.Radiobutton(
                backend_row, text=label, value=name, variable=self._backend_var,
                command=self._on_backend_change, bg=theme.surface, fg=theme.foreground,
                selectcolor=theme.base, activebackground=theme.surface, font=self._type.font("xs"),
                highlightthickness=0, borderwidth=0,
            ).pack(side="left", padx=(0, 12))
        self._download_button = self._button(
            backend_row, "Download model…", self._on_download_model
        )
        self._backend_note = tk.Label(
            options, text="", bg=theme.surface, fg=theme.text_dim, font=self._type.font("2xs"),
            anchor="w", justify="left", wraplength=560,
        )
        self._backend_note.pack(fill="x", padx=12, pady=(0, 4))

        preset_row = tk.Frame(options, bg=theme.surface)
        preset_row.pack(fill="x", padx=12, pady=(2, 8))
        tk.Label(
            preset_row, text="Change size", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("xs"),
        ).pack(side="left", padx=(0, 6))
        for name, title in ((p.name, p.title) for p in EDIT_PRESETS.values()):
            tk.Radiobutton(
                preset_row, text=title, value=name, variable=self._preset_var,
                command=self._on_preset_change, bg=theme.surface, fg=theme.foreground,
                selectcolor=theme.base, activebackground=theme.surface, font=self._type.font("xs"),
                highlightthickness=0, borderwidth=0,
            ).pack(side="left", padx=(0, 12))

        # 3. Instruction: the shared chat composer, so the text entry and its Apply
        # action read the same as every other screen's input panel.
        self._composer = ChatComposer(
            controls, theme=theme, typography=self._type,
            placeholder=INSTRUCTION_HINT, on_submit=self._on_apply,
            outer_bg=theme.base, with_attach=False, min_lines=3,
        )
        self._composer.wrap.pack(fill="x", padx=20, pady=(8, 4))
        self._instruction = self._composer.entry
        self._apply_button = self._button(
            self._composer.right_tools, "Apply", self._on_apply
        )
        self._composer.add_tool(self._apply_button)
        self._apply_button.configure(state="disabled")
        self._undo_button = self._button(
            self._composer.right_tools, "Undo", self._on_undo
        )
        self._composer.add_tool(self._undo_button)
        self._undo_button.configure(state="disabled")
        self._save_button = self._button(
            self._composer.right_tools, "Save a copy…", self._on_save_copy
        )
        self._composer.add_tool(self._save_button)
        self._save_button.configure(state="disabled")

        actions = tk.Frame(self.window, bg=theme.base)
        actions.pack(fill="x", padx=20, pady=(6, 4))
        self._status = tk.Label(
            actions, text="", bg=theme.base, fg=theme.text_secondary,
            font=self._type.font("xs"), anchor="w", justify="left", wraplength=520,
        )
        self._status.pack(side="left", padx=12)

        preview = self._section("4.  Preview", parent=side)
        preview.pack_configure(fill="both", expand=True)
        self._preview = tk.Label(
            preview, bg=theme.surface, fg=theme.text_dim, font=self._type.font("xs"),
            width=1, height=1, anchor="center", justify="center",
        )
        self._preview.pack(fill="both", expand=True, padx=12, pady=(2, 10))
        self._preview.configure(text="No image yet.")
        self._refresh_backend_note()
        self._layout_columns()

    # -- preview geometry -------------------------------------------------

    def _on_body_configure(self, _event: tk.Event | None = None) -> None:
        self._layout_columns()

    def _layout_columns(self) -> None:
        """Pin the preview column to ``PREVIEW_FRACTION`` of the window width."""
        try:
            width = self.window.winfo_width()
        except tk.TclError:  # the window is closing
            return
        if width <= 1:
            return
        self._body.columnconfigure(
            1, minsize=max(160, int(width * PREVIEW_FRACTION) + 2 * SECTION_PAD + 2 * PREVIEW_PAD)
        )
        if self._preview_source:
            self.window.after_idle(self._refit_preview)

    def _refit_preview(self) -> None:
        """Rescale the shown image when the preview column width has changed."""
        if not self._preview_source:
            return
        self.window.update_idletasks()
        width = self._preview_pixel_width()
        if width > 1 and width != self._preview_fit_width:
            self._show_preview(self._preview_source)

    def _preview_pixel_width(self) -> int:
        try:
            available = self._preview.winfo_width()
        except tk.TclError:
            return 0
        return max(1, available)

    def _section(self, title: str, *, parent: tk.Misc | None = None) -> tk.Frame:
        theme = self.theme
        frame = tk.Frame(parent or self.window, bg=theme.surface)
        frame.pack(fill="x", padx=20, pady=(10, 0))
        tk.Label(
            frame, text=title, bg=theme.surface, fg=theme.contrast,
            font=self._type.font("sm", weight="bold"), anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 4))
        return frame

    def _button(self, parent: tk.Misc, text: str, command: Any) -> tk.Button:
        theme = self.theme
        return tk.Button(
            parent, text=text, command=command, bg=theme.surface_raised,
            fg=theme.foreground, activebackground=theme.interactive_active,
            activeforeground=theme.contrast, relief="flat", borderwidth=0,
            font=self._type.font("xs", weight="bold"), padx=12, pady=4, cursor="hand2",
        )

    # -- backend selection ------------------------------------------------

    def _service_for_backend(self) -> ImageEditService:
        backend = self._backend_var.get()
        if backend == self.edit_service.backend:
            return self.edit_service
        return self.edit_service.for_backend(backend, preset=self._preset_var.get())

    def _on_backend_change(self) -> None:
        self._refresh_backend_note()

    def _on_preset_change(self) -> None:
        self._refresh_backend_note()

    def _refresh_backend_note(self) -> None:
        backend = self._backend_var.get()
        status = {
            name: (available, note)
            for name, available, note in self.edit_service.backend_status()
        }
        available, note = status.get(backend, (False, "Unknown backend."))
        if backend == BACKEND_LEARNED:
            self._download_button.pack(side="left", padx=(8, 0))
            self._download_button.configure(state="disabled" if self._busy else "normal")
            available = self._service_for_backend().available()
            note = f"{note}  Selected change size: {self._preset_var.get()}."
        else:
            self._download_button.pack_forget()
        prefix = "Ready." if available else "Not ready."
        self._backend_note.configure(text=f"{prefix}  {note}")

    def _on_download_model(self) -> None:
        if self._busy:
            return
        self._busy = True
        self._apply_button.configure(state="disabled")
        self._download_button.configure(state="disabled")
        self._set_status("Downloading the editing model… (one-time, ~4 GB)")

        def run() -> None:
            try:
                report = self.edit_service.download_model(approved=True)
                self._results.put(("download", report, None))
            except Exception as error:
                self._results.put(("download", None, error))

        threading.Thread(target=run, daemon=True).start()
        self.window.after(60, self._pump)

    # -- behaviour --------------------------------------------------------

    def _on_choose(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.window,
            title="Choose an image to edit",
            filetypes=[
                ("Images", "*.png *.ppm"),
                ("PNG image", "*.png"),
                ("PPM image", "*.ppm"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._load_source(path)

    def _load_source(self, path: str) -> None:
        try:
            self._session = self.edit_service.load(path)
        except Exception as error:
            self._set_status(f"Could not open that image: {error}", error=True)
            return
        self._source_label.configure(
            text=f"{Path(path).name}  ·  {self._session.raster.width}×{self._session.raster.height}",
            fg=self.theme.foreground,
        )
        self._apply_button.configure(state="normal")
        self._save_button.configure(state="normal")
        self._undo_button.configure(state="disabled")
        if not self._composer.text().strip():
            self._composer.set_text(EXAMPLE_INSTRUCTION)
        self._show_preview(path)
        self._set_status("Image loaded. Describe an edit and press Apply.")

    def _on_apply(self) -> None:
        if self._busy or self._session is None:
            return
        instruction = self._composer.text().strip()
        if not instruction:
            self._set_status("Write an instruction first.", error=True)
            return
        service = self._service_for_backend()
        if service.learned and not service.available():
            self._set_status(
                "The learned editor is not ready — install its packages with "
                f"{pip_install_command(FEATURE_IMAGE)} and download the model, "
                "or choose Deterministic.",
                error=True,
            )
            return
        self._busy = True
        self._apply_button.configure(state="disabled")
        self._download_button.configure(state="disabled")
        self._set_status("Editing…")

        def run() -> None:
            try:
                report = service.apply(self._session, instruction)
                self._results.put(("apply", report, None))
            except Exception as error:
                self._results.put(("apply", None, error))

        threading.Thread(target=run, daemon=True).start()
        self.window.after(60, self._pump)

    def _pump(self) -> None:
        """Collect a finished edit on the main thread and update the window."""
        try:
            kind, report, error = self._results.get_nowait()
        except queue.Empty:
            if self._busy:
                self.window.after(60, self._pump)
            return
        if kind == "download":
            self._finish_download(report, error)
        else:
            self._finish(report, error)

    def _finish_download(self, report: Any, error: Exception | None) -> None:
        self._busy = False
        self._apply_button.configure(state="normal")
        if error is not None:
            self._set_status(f"Could not download the model: {error}", error=True)
        elif not report.ok:
            self._set_status(report.message, error=True)
        else:
            self._set_status(report.message)
        self._refresh_backend_note()

    def _finish(self, report: Any, error: Exception | None) -> None:
        self._busy = False
        self._apply_button.configure(state="normal")
        self._download_button.configure(state="disabled" if not self._backend_var.get() == BACKEND_LEARNED else "normal")
        if error is not None:
            self._set_status(f"Could not edit the image: {error}", error=True)
            return
        assert report is not None
        if not report.ok:
            self._set_status(report.message, error=True)
            return
        self._show_preview(report.path)
        self._undo_button.configure(state="normal")
        if report.learned:
            note = report.description or "learned model edit."
        else:
            note = f"Applied: {', '.join(report.operations)}."
            if report.unknown:
                note += f"  Not understood: {', '.join(report.unknown)}."
        self._set_status(
            f"Saved to {report.path}  ·  {report.width}×{report.height}  ·  {note}"
        )

    def _on_undo(self) -> None:
        if self._busy or self._session is None:
            return
        if not self.edit_service.undo(self._session):
            self._set_status("Nothing to undo.", error=True)
            return
        self._undo_button.configure(
            state="normal" if self._session.undo_stack else "disabled"
        )
        self._render_raster()
        self._set_status("Undid the last edit.")

    def _render_raster(self) -> None:
        """Show the session's current raster (used after Undo, which has no new file)."""
        try:
            path = self.edit_service.render_working(self._session)
        except OSError as error:
            self._set_status(f"Could not render the preview: {error}", error=True)
            return
        self._show_preview(path)

    def _show_preview(self, path: str) -> None:
        """Scale the shown image to the preview column's width."""
        self._preview_source = path
        width = self._preview_pixel_width()
        photo = load_scaled(path, width)
        if photo is None:
            self._photo = None
            self._preview.configure(
                image="", text=f"Preview unavailable in this Tk build.\nFile: {path}"
            )
            return
        self._photo = photo
        self._preview_fit_width = width
        self._preview.configure(image=photo, text="")

    def _on_save_copy(self) -> None:
        if self._session is None:
            return
        target = filedialog.asksaveasfilename(
            parent=self.window, title="Save the image as", defaultextension=".png",
            initialfile=Path(self._session.source_path).stem + "-edited.png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not target:
            return
        report = self.edit_service.save_copy(self._session, target)
        self._set_status(report.message, error=not report.ok)

    def _set_status(self, message: str, *, error: bool = False) -> None:
        self._status.configure(
            text=message, fg=self.theme.danger if error else self.theme.text_secondary
        )

    def _close(self) -> None:
        if self._on_close is not None:
            self._on_close()
        self.window.destroy()


__all__ = [
    "HONESTY_LINE",
    "INSTRUCTION_HINT",
    "LEARNED_EXAMPLE_INSTRUCTION",
    "PREVIEW_FRACTION",
    "SUBTITLE",
    "TITLE",
    "ImageEditDialog",
]
