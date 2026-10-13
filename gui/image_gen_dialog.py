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
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog

TITLE = "Generate an image from text"
SUBTITLE = (
    "Write a prompt and render it to a PNG. The default backend is a dependency-free "
    "procedural renderer: the same prompt and seed always give the same picture, and "
    "the result is an abstract composition, not a learned depiction of the words."
)
HONESTY_LINE = (
    "This build has no learned text-to-image model. Images are geometric compositions "
    "derived from the prompt; treat them as decoration, not as a picture of the subject."
)


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
        self._on_close = on_close
        self._busy = False
        self._photo: tk.PhotoImage | None = None
        self._last_path: str = ""

        self._width_var = tk.StringVar(value="")
        self._height_var = tk.StringVar(value="")
        self._seed_var = tk.StringVar(value="")
        self._backend_var = tk.StringVar(value=service.backend)

        self.window = tk.Toplevel(parent)
        self.window.title(TITLE)
        self.window.configure(bg=theme.base)
        self.window.geometry("760x720")
        self.window.minsize(560, 520)
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
            font=(self.font, 9), anchor="w", justify="left", wraplength=700,
        ).pack(fill="x", pady=(4, 0))
        tk.Label(
            header, text=HONESTY_LINE, bg=theme.base, fg=theme.text_dim,
            font=(self.font, 8), anchor="w", justify="left", wraplength=700,
        ).pack(fill="x", pady=(6, 0))

        prompt_frame = self._section("1.  Prompt")
        self._prompt = tk.Text(
            prompt_frame, height=3, wrap="word", relief="flat", borderwidth=0,
            highlightthickness=1, highlightbackground=theme.border,
            highlightcolor=theme.primary, bg=theme.surface, fg=theme.foreground,
            insertbackground=theme.primary, font=(self.font, 11), padx=8, pady=8,
        )
        self._prompt.pack(fill="x", padx=12, pady=(2, 4))
        tk.Label(
            prompt_frame,
            text="Tip: end the prompt with a size, e.g. “a red spiral 768x512”.",
            bg=theme.surface, fg=theme.text_dim, font=(self.font, 8), anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 8))

        options = self._section("2.  Options")
        backend_row = tk.Frame(options, bg=theme.surface)
        backend_row.pack(fill="x", padx=12, pady=(2, 4))
        tk.Label(
            backend_row, text="Backend", bg=theme.surface, fg=theme.text_secondary,
            font=(self.font, 9),
        ).pack(side="left", padx=(0, 6))
        for name, label in (("procedural", "Procedural (default)"), ("diffusion", "Diffusion (learned)")):
            tk.Radiobutton(
                backend_row, text=label, value=name, variable=self._backend_var,
                command=self._on_backend_change, bg=theme.surface, fg=theme.foreground,
                selectcolor=theme.base, activebackground=theme.surface, font=(self.font, 9),
                highlightthickness=0, borderwidth=0,
            ).pack(side="left", padx=(0, 12))
        self._download_button = self._button(
            backend_row, "Download model…", self._on_download_model
        )
        self._download_button.pack(side="left", padx=(8, 0))
        self._backend_note = tk.Label(
            options, text="", bg=theme.surface, fg=theme.text_dim, font=(self.font, 8),
            anchor="w", justify="left", wraplength=700,
        )
        self._backend_note.pack(fill="x", padx=12, pady=(0, 4))

        row = tk.Frame(options, bg=theme.surface)
        row.pack(fill="x", padx=12, pady=(2, 8))
        self._labelled_entry(row, "Width", self._width_var, 6)
        self._labelled_entry(row, "Height", self._height_var, 6)
        self._labelled_entry(row, "Seed", self._seed_var, 10)
        tk.Label(
            row, text="(blank = size from the prompt / a seed from the prompt)",
            bg=theme.surface, fg=theme.text_dim, font=(self.font, 8), anchor="w",
        ).pack(side="left", padx=(8, 0))
        self._refresh_backend_note()

        actions = tk.Frame(self.window, bg=theme.base)
        actions.pack(fill="x", padx=20, pady=(6, 4))
        self._generate_button = self._button(actions, "Generate", self._on_generate)
        self._generate_button.pack(side="left")
        self._save_button = self._button(actions, "Save a copy…", self._on_save_copy)
        self._save_button.pack(side="left", padx=8)
        self._save_button.configure(state="disabled")
        self._status = tk.Label(
            actions, text="", bg=theme.base, fg=theme.text_secondary,
            font=(self.font, 9), anchor="w", justify="left", wraplength=520,
        )
        self._status.pack(side="left", padx=12)

        preview = self._section("3.  Preview")
        self._preview = tk.Label(preview, bg=theme.surface, fg=theme.text_dim, font=(self.font, 9))
        self._preview.pack(fill="both", expand=True, padx=12, pady=(2, 10))
        self._preview.configure(text="No image yet.")

    def _section(self, title: str) -> tk.Frame:
        theme = self.theme
        frame = tk.Frame(self.window, bg=theme.surface)
        frame.pack(fill="x", padx=20, pady=(10, 0))
        tk.Label(
            frame, text=title, bg=theme.surface, fg=theme.contrast,
            font=(self.font, 10, "bold"), anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 4))
        return frame

    def _labelled_entry(self, parent: tk.Misc, label: str, var: tk.StringVar, width: int) -> None:
        theme = self.theme
        tk.Label(
            parent, text=label, bg=theme.surface, fg=theme.text_secondary, font=(self.font, 9),
        ).pack(side="left", padx=(0, 4))
        tk.Entry(
            parent, textvariable=var, width=width, bg=theme.base, fg=theme.foreground,
            insertbackground=theme.primary, relief="flat", highlightthickness=1,
            highlightbackground=theme.border, font=(self.font, 10),
        ).pack(side="left", padx=(0, 12))

    def _button(self, parent: tk.Misc, text: str, command: Any) -> tk.Button:
        theme = self.theme
        return tk.Button(
            parent, text=text, command=command, bg=theme.surface_raised,
            fg=theme.foreground, activebackground=theme.interactive_active,
            activeforeground=theme.contrast, relief="flat", borderwidth=0,
            font=(self.font, 9, "bold"), padx=12, pady=4, cursor="hand2",
        )

    # -- behaviour --------------------------------------------------------

    def _service_for_backend(self) -> Any:
        backend = self._backend_var.get()
        if backend == self.image_service.backend:
            if self.image_service.available():
                return self.image_service
        return self.image_service.for_backend(backend)

    def _on_backend_change(self) -> None:
        self._refresh_backend_note()

    def _refresh_backend_note(self) -> None:
        backend = self._backend_var.get()
        status = dict((name, (available, note)) for name, available, note in self.image_service.backend_status())
        available, note = status.get(backend, (False, "Unknown backend."))
        if backend == "procedural":
            self._download_button.pack_forget()
        else:
            self._download_button.pack(side="left", padx=(8, 0))
            self._download_button.configure(state="disabled" if self._busy else "normal")
        prefix = "Ready." if available else "Not ready."
        self._backend_note.configure(text=f"{prefix}  {note}")

    def _on_download_model(self) -> None:
        if self._busy:
            return
        self._busy = True
        self._generate_button.configure(state="disabled")
        self._download_button.configure(state="disabled")
        self._set_status("Downloading the model… (one-time, ~2.5 GB)")

        def run() -> None:
            try:
                report = self.image_service.download_model(approved=True)
                self.window.after(0, partial(self._finish_download, report, None))
            except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
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
        prompt = self._prompt.get("1.0", "end").strip()
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
                "The learned backend is not ready — install the [image] extra and "
                "download the model, or choose Procedural.",
                error=True,
            )
            return
        self._busy = True
        self._generate_button.configure(state="disabled")
        self._save_button.configure(state="disabled")
        self._set_status("Rendering…")

        def run() -> None:
            try:
                report = service.generate(
                    prompt, width=width, height=height, seed=seed
                )
                self.window.after(0, partial(self._finish, report, None))
            except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
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
        try:
            photo = tk.PhotoImage(file=path)
        except tk.TclError:
            self._photo = None
            self._preview.configure(
                image="", text=f"Preview unavailable in this Tk build.\nFile: {path}"
            )
            return
        # Keep a reference so Tk does not garbage-collect the image.
        self._photo = photo
        self._preview.configure(image=photo, text="")

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


__all__ = ["HONESTY_LINE", "ImageGenDialog", "SUBTITLE", "TITLE"]
