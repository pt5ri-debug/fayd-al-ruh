"""The "Edit an image from text" screen: upload a picture and change it with words.

Reached from the ＋ menu, this page turns a written instruction into an edit of a
picture the user already has:

* **Choose an image…** -- loads a PNG or PPM into the page;
* an instruction box, e.g. "rotate 90 and brighten, then crop to a square";
* **Apply** -- interprets the instruction and edits the current picture, then
  shows the result in the page and saves it under the data folder;
* **Undo** -- steps back to the previous version;
* **Save a copy…** -- writes the current picture wherever the user chooses.

It is a thin view over :class:`gui.image_edit.ImageEditService`. The editor is a
deterministic keyword interpreter, not a learned instruction-following model, and
the page says so plainly: it lists the operations it understood and names the
words it did not, so nothing is silently ignored. A learned editor could be
dropped in behind the same service without touching this page.

The class is Tk-only and imported lazily with the rest of the window, so the CLI
never touches it.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog

from core.imageedit import apply_plan
from gui.image_edit import EXAMPLE_INSTRUCTION, READABLE_SUFFIXES, ImageEditService
from gui.image_preview import load_scaled

TITLE = "Edit an image from text"
SUBTITLE = (
    "Upload a picture, then describe the change in words. This editor is a "
    "deterministic keyword interpreter, not a learned model: it understands a "
    "documented set of operations (rotate, flip, resize, crop, brightness, "
    "contrast, saturation, grayscale, invert, sepia, warm/cool, blur, sharpen, "
    "threshold) and applies them in the order you write them."
)
HONESTY_LINE = (
    "Every edit is a plain pixel operation on your file; the original is never "
    "modified. Words the editor does not recognise are reported, not guessed at."
)
INSTRUCTION_HINT = (
    "Examples: “rotate 90”, “flip horizontally”, “resize to 512x512”, "
    "“crop to a square”, “grayscale”, “brighten 20%”, “blur 3”, “invert”."
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
        self._on_close = on_close
        self._busy = False
        self._session: Any = None
        #: Edits run on a worker thread; the result arrives through this queue and
        #: is applied on the main thread by the poll pump, as Tk requires.
        self._results: queue.Queue[tuple[Any, Exception | None]] = queue.Queue()
        self._photo: tk.PhotoImage | None = None
        self._preview_source: str = ""
        #: The column width the shown image was last fitted to, so a resize only
        #: rescales when the width actually changed.
        self._preview_fit_width: int = 0
        self._original_raster: Any = None

        self._instruction_var = tk.StringVar(value="")

        self.window = tk.Toplevel(parent)
        self.window.title(TITLE)
        self.window.configure(bg=theme.base)
        self.window.geometry("1040x620")
        self.window.minsize(820, 580)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._build()

    # -- layout -----------------------------------------------------------

    def _build(self) -> None:
        theme = self.theme
        header = tk.Frame(self.window, bg=theme.base)
        header.pack(fill="x", padx=20, pady=(18, 8))
        tk.Label(
            header, text=TITLE, bg=theme.base, fg=theme.contrast,
            font=(self.font, 15, "bold"), anchor="w",
        ).pack(fill="x")
        tk.Label(
            header, text=SUBTITLE, bg=theme.base, fg=theme.text_secondary,
            font=(self.font, 9), anchor="w", justify="left", wraplength=980,
        ).pack(fill="x", pady=(4, 0))
        tk.Label(
            header, text=HONESTY_LINE, bg=theme.base, fg=theme.text_dim,
            font=(self.font, 8), anchor="w", justify="left", wraplength=980,
        ).pack(fill="x", pady=(6, 0))

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
            font=(self.font, 9), anchor="w",
        )
        self._source_label.pack(side="left", padx=(10, 0))
        tk.Label(
            source_frame, text=f"Supported now: {', '.join(READABLE_SUFFIXES)}.",
            bg=theme.surface, fg=theme.text_dim, font=(self.font, 8), anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 8))

        instruction_frame = self._section("2.  Instruction", parent=controls)
        self._instruction = tk.Text(
            instruction_frame, height=3, wrap="word", relief="flat", borderwidth=0,
            highlightthickness=1, highlightbackground=theme.border,
            highlightcolor=theme.primary, bg=theme.surface, fg=theme.foreground,
            insertbackground=theme.primary, font=(self.font, 11), padx=8, pady=8,
        )
        self._instruction.pack(fill="x", padx=12, pady=(2, 4))
        tk.Label(
            instruction_frame, text=INSTRUCTION_HINT, bg=theme.surface,
            fg=theme.text_dim, font=(self.font, 8), anchor="w", justify="left",
            wraplength=560,
        ).pack(fill="x", padx=12, pady=(0, 8))

        actions = tk.Frame(self.window, bg=theme.base)
        actions.pack(fill="x", padx=20, pady=(6, 4))
        self._apply_button = self._button(actions, "Apply", self._on_apply)
        self._apply_button.pack(side="left")
        self._apply_button.configure(state="disabled")
        self._undo_button = self._button(actions, "Undo", self._on_undo)
        self._undo_button.pack(side="left", padx=8)
        self._undo_button.configure(state="disabled")
        self._save_button = self._button(actions, "Save a copy…", self._on_save_copy)
        self._save_button.pack(side="left")
        self._save_button.configure(state="disabled")
        self._status = tk.Label(
            actions, text="", bg=theme.base, fg=theme.text_secondary,
            font=(self.font, 9), anchor="w", justify="left", wraplength=520,
        )
        self._status.pack(side="left", padx=12)

        preview = self._section("3.  Preview", parent=side)
        preview.pack_configure(fill="both", expand=True)
        self._preview = tk.Label(
            preview, bg=theme.surface, fg=theme.text_dim, font=(self.font, 9),
            width=1, height=1, anchor="center", justify="center",
        )
        self._preview.pack(fill="both", expand=True, padx=12, pady=(2, 10))
        self._preview.configure(text="No image yet.")
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
            font=(self.font, 10, "bold"), anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 4))
        return frame

    def _button(self, parent: tk.Misc, text: str, command: Any) -> tk.Button:
        theme = self.theme
        return tk.Button(
            parent, text=text, command=command, bg=theme.surface_raised,
            fg=theme.foreground, activebackground=theme.interactive_active,
            activeforeground=theme.contrast, relief="flat", borderwidth=0,
            font=(self.font, 9, "bold"), padx=12, pady=4, cursor="hand2",
        )

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
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._set_status(f"Could not open that image: {error}", error=True)
            return
        self._original_raster = self._session.raster.copy()
        self._source_label.configure(
            text=f"{Path(path).name}  ·  {self._session.raster.width}×{self._session.raster.height}",
            fg=self.theme.foreground,
        )
        self._apply_button.configure(state="normal")
        self._save_button.configure(state="normal")
        self._undo_button.configure(state="disabled")
        if not self._instruction.get("1.0", "end").strip():
            self._instruction.insert("1.0", EXAMPLE_INSTRUCTION)
        self._show_preview(path)
        self._set_status("Image loaded. Describe an edit and press Apply.")

    def _on_apply(self) -> None:
        if self._busy or self._session is None:
            return
        instruction = self._instruction.get("1.0", "end").strip()
        if not instruction:
            self._set_status("Write an instruction first.", error=True)
            return
        self._busy = True
        self._apply_button.configure(state="disabled")
        self._set_status("Editing…")

        def run() -> None:
            try:
                report = self.edit_service.apply(self._session, instruction)
                self._results.put((report, None))
            except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
                self._results.put((None, error))

        threading.Thread(target=run, daemon=True).start()
        self.window.after(60, self._pump)

    def _pump(self) -> None:
        """Collect a finished edit on the main thread and update the window."""
        try:
            report, error = self._results.get_nowait()
        except queue.Empty:
            if self._busy:
                self.window.after(60, self._pump)
            return
        self._finish(report, error)

    def _finish(self, report: Any, error: Exception | None) -> None:
        self._busy = False
        self._apply_button.configure(state="normal")
        if error is not None:
            self._set_status(f"Could not edit the image: {error}", error=True)
            return
        assert report is not None
        if not report.ok:
            self._set_status(report.message, error=True)
            return
        self._show_preview(report.path)
        self._undo_button.configure(state="normal")
        note = f"Applied: {', '.join(report.operations)}."
        if report.unknown:
            note += f"  Not understood: {', '.join(report.unknown)}."
        self._set_status(
            f"Saved to {report.path}  ·  {report.width}×{report.height}  ·  {note}"
        )

    def _on_undo(self) -> None:
        if self._busy or self._session is None:
            return
        if not self._session.history:
            self._set_status("Nothing to undo.", error=True)
            return
        self._session.history.pop()
        # Replay the remaining instructions from the original image.
        raster = self._original_raster.copy()
        for plan in self._session.history:
            raster = apply_plan(raster, plan)
        self._session.raster = raster
        self._undo_button.configure(
            state="normal" if self._session.history else "disabled"
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
    "ImageEditDialog",
    "INSTRUCTION_HINT",
    "PREVIEW_FRACTION",
    "SUBTITLE",
    "TITLE",
]
