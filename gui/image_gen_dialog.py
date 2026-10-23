"""The "Generate an image from text" screen: the window's text-to-image page.

Reached from the ＋ menu, this page turns a written prompt into an image file:

* a prompt box (a trailing ``WxH`` sets the size, e.g. ``768x512``);
* optional width, height, and seed fields;
* **Generate** -- renders on a worker thread so Tk never blocks, then shows the
  picture in the page and saves it under the data folder;
* **Save a copy…** -- writes the same bytes wherever the user chooses.

It is a thin view over :class:`gui.image_gen.ImageGenerationService`. The default
backend renders with only the standard library and the page says so plainly: this
is a deterministic geometric composition, not a learned text-to-image model. A
real model can be dropped in behind the same service without touching this page.

The class is Tk-only and imported lazily with the rest of the window, so the CLI
never touches it.
"""

from __future__ import annotations

import threading
import tkinter as tk
from collections.abc import Callable
from functools import partial
from pathlib import Path
from tkinter import filedialog
from typing import Any

from core.imagegen_diffusion import DEFAULT_PRESET, PRESETS
from core.optional_deps import FEATURE_IMAGE, pip_install_command

from gui.chat_composer import ChatComposer
from gui.image_preview import load_scaled
from gui.typography import resolve
from gui.window_controls import install_window_controls

TITLE = "Generate an image from text"
SUBTITLE = (
    "Write a prompt and render it to a PNG. The default backend is a dependency-free "
    "procedural renderer: the same prompt and seed always give the same picture, and "
    "the result is an abstract composition, not a learned depiction of the words. "
    "Choose Diffusion (learned) for a photorealistic picture from a real model."
)
HONESTY_LINE = (
    "Procedural output is a geometric composition, not a picture of the subject. "
    "Diffusion output is drawn by a learned model; the Photorealistic preset is the "
    "default and is tuned for clean, lifelike images."
)

#: The preview column (and so the rendered picture) is this fraction of the
#: window's width, both before a render and after one.
PREVIEW_FRACTION = 0.30

#: Horizontal padding, in pixels: the side padding of every section card, and the
#: extra padding inside the preview card. They are named so the preview column can
#: be sized so the *visible* preview box is exactly ``PREVIEW_FRACTION`` of the
#: window width.
SECTION_PAD = 20
PREVIEW_PAD = 12


class ImageGenDialog:
    """A window that renders a text prompt to an image file."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        service: Any,
        *,
        theme: Any,
        font: str,
        mono: str,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self.image_service = service
        self.theme = theme
        self.font = font
        self.mono = mono
        self._type = resolve(parent, ui_family=font, mono_family=mono)
        self._on_close = on_close
        self._busy = False
        self._photo: tk.PhotoImage | None = None
        self._last_path: str = ""
        self._preview_source: str = ""
        #: The column width the shown image was last fitted to, so a resize only
        #: rescales when the width actually changed.
        self._preview_fit_width: int = 0

        self._width_var = tk.StringVar(value="")
        self._height_var = tk.StringVar(value="")
        self._seed_var = tk.StringVar(value="")
        self._backend_var = tk.StringVar(value=service.backend)
        self._preset_var = tk.StringVar(value=DEFAULT_PRESET)

        self.window = tk.Toplevel(parent)
        self.window.title(TITLE)
        self.window.configure(bg=theme.base)
        self.window.geometry("1040x576")
        self.window.minsize(820, 560)
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
        # before and after a render.
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

        # 1. Prompt: the shared chat composer, so the text entry and its Generate
        # action read the same as every other screen's input panel.
        self._composer = ChatComposer(
            controls, theme=theme, typography=self._type,
            placeholder="Describe the image you want…",
            on_submit=self._on_generate, outer_bg=theme.base,
            with_attach=False, min_lines=3,
        )
        self._composer.wrap.pack(fill="x", padx=20, pady=(8, 4))
        self._prompt = self._composer.entry
        self._generate_button = self._button(
            self._composer.right_tools, "Generate", self._on_generate
        )
        self._composer.add_tool(self._generate_button)
        self._save_button = self._button(
            self._composer.right_tools, "Save a copy…", self._on_save_copy
        )
        self._composer.add_tool(self._save_button)
        self._save_button.configure(state="disabled")
        tk.Label(
            controls,
            text="Tip: end the prompt with a size, e.g. “a red spiral 768x512”.",
            bg=theme.base, fg=theme.text_dim, font=self._type.font("2xs"), anchor="w",
        ).pack(fill="x", padx=20, pady=(0, 8))

        options = self._section("2.  Options", parent=controls)
        backend_row = tk.Frame(options, bg=theme.surface)
        backend_row.pack(fill="x", padx=12, pady=(2, 4))
        tk.Label(
            backend_row, text="Backend", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("xs"),
        ).pack(side="left", padx=(0, 6))
        for name, label in (("procedural", "Procedural (default)"), ("diffusion", "Diffusion (learned)")):
            tk.Radiobutton(
                backend_row, text=label, value=name, variable=self._backend_var,
                command=self._on_backend_change, bg=theme.surface, fg=theme.foreground,
                selectcolor=theme.base, activebackground=theme.surface, font=self._type.font("xs"),
                highlightthickness=0, borderwidth=0,
            ).pack(side="left", padx=(0, 12))
        self._download_button = self._button(
            backend_row, "Download model…", self._on_download_model
        )
        self._download_button.pack(side="left", padx=(8, 0))
        self._backend_note = tk.Label(
            options, text="", bg=theme.surface, fg=theme.text_dim, font=self._type.font("2xs"),
            anchor="w", justify="left", wraplength=560,
        )
        self._backend_note.pack(fill="x", padx=12, pady=(0, 4))

        preset_row = tk.Frame(options, bg=theme.surface)
        preset_row.pack(fill="x", padx=12, pady=(2, 4))
        tk.Label(
            preset_row, text="Preset", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("xs"),
        ).pack(side="left", padx=(0, 6))
        for name, title in ((p.name, p.title) for p in PRESETS.values()):
            tk.Radiobutton(
                preset_row, text=title, value=name, variable=self._preset_var,
                command=self._on_preset_change, bg=theme.surface, fg=theme.foreground,
                selectcolor=theme.base, activebackground=theme.surface, font=self._type.font("xs"),
                highlightthickness=0, borderwidth=0,
            ).pack(side="left", padx=(0, 12))

        negative_row = tk.Frame(options, bg=theme.surface)
        negative_row.pack(fill="x", padx=12, pady=(2, 4))
        tk.Label(
            negative_row, text="Negative prompt", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("xs"),
        ).pack(side="left", padx=(0, 6))
        self._negative = tk.Entry(
            negative_row, bg=theme.base, fg=theme.foreground, insertbackground=theme.primary,
            relief="flat", highlightthickness=1, highlightbackground=theme.border,
            font=self._type.font("xs"),
        )
        self._negative.pack(side="left", fill="x", expand=True)
        tk.Label(
            options,
            text="(blank = the preset's photorealistic default: no cartoon, blur, or artefacts)",
            bg=theme.surface, fg=theme.text_dim, font=self._type.font("2xs"), anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 4))

        row = tk.Frame(options, bg=theme.surface)
        row.pack(fill="x", padx=12, pady=(2, 8))
        self._labelled_entry(row, "Width", self._width_var, 6)
        self._labelled_entry(row, "Height", self._height_var, 6)
        self._labelled_entry(row, "Seed", self._seed_var, 10)
        tk.Label(
            row, text="(blank = size from the prompt / a seed from the prompt)",
            bg=theme.surface, fg=theme.text_dim, font=self._type.font("2xs"), anchor="w",
        ).pack(side="left", padx=(8, 0))
        self._refresh_backend_note()

        actions = tk.Frame(self.window, bg=theme.base)
        actions.pack(fill="x", padx=20, pady=(6, 4))
        self._status = tk.Label(
            actions, text="", bg=theme.base, fg=theme.text_secondary,
            font=self._type.font("xs"), anchor="w", justify="left", wraplength=520,
        )
        self._status.pack(side="left", padx=12)

        preview = self._section("3.  Preview", parent=side)
        preview.pack_configure(fill="both", expand=True)
        self._preview = tk.Label(
            preview, bg=theme.surface, fg=theme.text_dim, font=self._type.font("xs"),
            width=1, height=1, anchor="center", justify="center",
        )
        self._preview.pack(fill="both", expand=True, padx=12, pady=(2, 10))
        self._preview.configure(text="No image yet.")
        self._layout_columns()

    # -- preview geometry -------------------------------------------------

    def _on_body_configure(self, _event: tk.Event | None = None) -> None:
        self._layout_columns()

    def _layout_columns(self) -> None:
        """Pin the preview column to ``PREVIEW_FRACTION`` of the window width.

        The window's own width is what the fraction is taken from, so the preview
        (and therefore the image inside it) is the same width before a render and
        after one, and it follows the window when the user resizes it.
        """
        try:
            width = self.window.winfo_width()
        except tk.TclError:  # the window is closing
            return
        if width <= 1:
            return
        # The column is sized so the *visible* preview box (the column minus the
        # section and preview padding) is PREVIEW_FRACTION of the window width.
        self._body.columnconfigure(
            1, minsize=max(160, int(width * PREVIEW_FRACTION) + 2 * SECTION_PAD + 2 * PREVIEW_PAD)
        )
        # Re-fit an image that is already on screen once the new column geometry
        # has settled (deferred, so the measured width is the updated one).
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
        """The width, in pixels, the rendered image is scaled to fit."""
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

    def _labelled_entry(self, parent: tk.Misc, label: str, var: tk.StringVar, width: int) -> None:
        theme = self.theme
        tk.Label(
            parent, text=label, bg=theme.surface, fg=theme.text_secondary, font=self._type.font("xs"),
        ).pack(side="left", padx=(0, 4))
        tk.Entry(
            parent, textvariable=var, width=width, bg=theme.base, fg=theme.foreground,
            insertbackground=theme.primary, relief="flat", highlightthickness=1,
            highlightbackground=theme.border, font=self._type.font("sm"),
        ).pack(side="left", padx=(0, 12))

    def _button(self, parent: tk.Misc, text: str, command: Any) -> tk.Button:
        theme = self.theme
        return tk.Button(
            parent, text=text, command=command, bg=theme.surface_raised,
            fg=theme.foreground, activebackground=theme.interactive_active,
            activeforeground=theme.contrast, relief="flat", borderwidth=0,
            font=self._type.font("xs", weight="bold"), padx=12, pady=4, cursor="hand2",
        )

    # -- behaviour --------------------------------------------------------

    def _service_for_backend(self) -> Any:
        backend = self._backend_var.get()
        if backend == "procedural":
            if self.image_service.backend == "procedural":
                return self.image_service
            return self.image_service.for_backend("procedural")
        preset = self._preset_var.get()
        negative = self._negative.get().strip() or None
        return self.image_service.for_backend(
            "diffusion", preset=preset, negative_prompt=negative
        )

    def _on_backend_change(self) -> None:
        self._refresh_backend_note()

    def _on_preset_change(self) -> None:
        self._refresh_backend_note()

    def _refresh_backend_note(self) -> None:
        backend = self._backend_var.get()
        status = {name: (available, note) for name, available, note in self.image_service.backend_status()}
        available, note = status.get(backend, (False, "Unknown backend."))
        if backend == "procedural":
            self._download_button.pack_forget()
        else:
            self._download_button.pack(side="left", padx=(8, 0))
            self._download_button.configure(state="disabled" if self._busy else "normal")
        if backend == "diffusion":
            # Availability is per model: switching preset can change readiness.
            available = self._service_for_backend().available()
            preset = self._preset_var.get()
            note = f"{note}  Selected preset: {preset}."
        prefix = "Ready." if available else "Not ready."
        self._backend_note.configure(text=f"{prefix}  {note}")

    def _on_download_model(self) -> None:
        if self._busy:
            return
        self._busy = True
        self._generate_button.configure(state="disabled")
        self._download_button.configure(state="disabled")
        self._set_status("Downloading the model… (one-time, ~4 GB)")

        def run() -> None:
            try:
                report = self.image_service.download_model(
                    approved=True, preset=self._preset_var.get()
                )
                self.window.after(0, partial(self._finish_download, report, None))
            except Exception as error:
                self.window.after(0, partial(self._finish_download, None, error))

        threading.Thread(target=run, daemon=True).start()

    def _finish_download(self, report: Any, error: Exception | None) -> None:
        self._busy = False
        self._generate_button.configure(state="normal")
        if error is not None:
            self._set_status(f"Could not download the model: {error}", error=True)
        elif not report.ok:
            self._set_status(report.message, error=True)
        else:
            self._set_status(report.message)
        self._refresh_backend_note()

    def _on_generate(self) -> None:
        if self._busy:
            return
        prompt = self._composer.text().strip()
        if not prompt:
            self._set_status("Write a prompt first.", error=True)
            return
        width = self._optional_int(self._width_var.get())
        height = self._optional_int(self._height_var.get())
        seed = self._optional_int(self._seed_var.get())
        if width is False or height is False or seed is False:
            self._set_status("Width, height, and seed must be whole numbers.", error=True)
            return
        backend = self._backend_var.get()
        service = self._service_for_backend()
        if backend == "diffusion" and not service.available():
            self._set_status(
                "The learned backend is not ready — install its packages with "
                f"{pip_install_command(FEATURE_IMAGE)} and download the model, "
                "or choose Procedural.",
                error=True,
            )
            return
        negative = self._negative.get().strip() or None
        self._busy = True
        self._generate_button.configure(state="disabled")
        self._save_button.configure(state="disabled")
        self._set_status("Rendering…")

        def run() -> None:
            try:
                report = service.generate(
                    prompt, width=width, height=height, seed=seed, negative_prompt=negative
                )
                self.window.after(0, partial(self._finish, report, None))
            except Exception as error:
                self.window.after(0, partial(self._finish, None, error))

        threading.Thread(target=run, daemon=True).start()

    @staticmethod
    def _optional_int(text: str) -> int | bool | None:
        stripped = text.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return False

    def _finish(self, report: Any, error: Exception | None) -> None:
        self._busy = False
        self._generate_button.configure(state="normal")
        if error is not None:
            self._set_status(f"Could not render the image: {error}", error=True)
            return
        assert report is not None
        if not report.ok:
            self._set_status(report.message, error=True)
            return
        self._last_path = report.path
        self._show_preview(report.path)
        self._save_button.configure(state="normal")
        self._set_status(
            f"Saved to {report.path}  ·  {report.width}x{report.height}, "
            f"{report.size_bytes} bytes  ·  {report.backend}"
        )

    def _show_preview(self, path: str) -> None:
        """Scale the rendered image to the preview column's width and show it.

        The picture is fitted to the same width the empty preview occupies, so the
        generated image is exactly as wide as the preview box before it appeared.
        """
        self._preview_source = path
        width = self._preview_pixel_width()
        photo = self._load_scaled(path, width)
        if photo is None:
            self._photo = None
            self._preview.configure(
                image="", text=f"Preview unavailable in this Tk build.\nFile: {path}"
            )
            return
        # Keep a reference so Tk does not garbage-collect the image.
        self._photo = photo
        self._preview_fit_width = width
        self._preview.configure(image=photo, text="")

    def _load_scaled(self, path: str, width: int) -> tk.PhotoImage | None:
        """Load ``path`` scaled to about ``width`` pixels, or ``None`` if unreadable.

        Delegates to :func:`gui.image_preview.load_scaled`, which both image pages
        share so their previews scale identically.
        """
        return load_scaled(path, width)

    def _on_save_copy(self) -> None:
        if not self._last_path:
            return
        target = filedialog.asksaveasfilename(
            parent=self.window, title="Save the image as", defaultextension=".png",
            initialfile=Path(self._last_path).name,
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        )
        if not target:
            return
        try:
            Path(target).write_bytes(Path(self._last_path).read_bytes())
        except OSError as error:
            self._set_status(f"Could not save a copy: {error}", error=True)
            return
        self._set_status(f"Saved a copy to {target}")

    def _set_status(self, message: str, *, error: bool = False) -> None:
        self._status.configure(
            text=message, fg=self.theme.danger if error else self.theme.text_secondary
        )

    def _close(self) -> None:
        if self._on_close is not None:
            self._on_close()
        self.window.destroy()


__all__ = ["HONESTY_LINE", "PREVIEW_FRACTION", "SUBTITLE", "TITLE", "ImageGenDialog"]
