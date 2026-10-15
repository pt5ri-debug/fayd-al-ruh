"""The "Ultron" window: a general, advanced local chat.

Reached from the sidebar (directly below Jarvis), this page is a plain chat with
**Ultron** -- the project's general local model. Unlike the main window (which
answers from the petroleum corpus, with citations) Ultron talks about *anything*:
science, history, code, everyday questions. It runs entirely on this machine
through ``llama-cpp-python``; nothing is sent to a remote server.

The page is deliberately small:

* a transcript that shows the conversation so far;
* an input box and **Send**;
* a status line that says honestly whether the runtime and a model are present,
  with a **Download model…** button (a one-time, human-approved fetch) when they
  are not.

A reply is slow on a CPU, so it is produced on a worker thread and posted back
through a queue; Tk is only ever touched from the main thread, as it requires.
The class is Tk-only and imported lazily with the rest of the window, so the CLI
never touches it.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any

import tkinter as tk

from core.optional_deps import FEATURE_ULTRON, pip_install_command
from core.ultron import (
    DEFAULT_SPEC_KEY,
    HEADER,
    NAME,
    UltronEngine,
    UltronUnavailable,
    find_model_file,
)

TITLE = "Ultron"
SUBTITLE = (
    "A general, advanced local model. Ask about anything — science, history, "
    "technology, everyday questions — not only petroleum engineering. It runs on "
    "this machine; nothing is sent to a remote server."
)
HONESTY_LINE = (
    "Ultron's answers are the model's own. They are not retrieved from the project's "
    "corpus and carry no citations, so treat them as a helpful general model's words, "
    "not as verified sources."
)


class UltronDialog:
    """A window that holds one conversation with Ultron."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        engine: UltronEngine,
        *,
        theme: Any,
        font: str,
        mono: str,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self.engine = engine
        self.theme = theme
        self.font = font
        self.mono = mono
        self._on_close = on_close
        self._busy = False
        #: Replies run on a worker thread; text arrives through this queue and is
        #: appended on the main thread by the poll pump, as Tk requires.
        self._results: queue.Queue[tuple[str, Any, Exception | None]] = queue.Queue()
        self._history: list[tuple[str, str]] = []

        self.window = tk.Toplevel(parent)
        self.window.title(f"{NAME} — general local model")
        self.window.configure(bg=theme.base)
        self.window.geometry("900x640")
        self.window.minsize(680, 520)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._build()
        self._refresh_status()

    # -- layout -----------------------------------------------------------

    def _build(self) -> None:
        theme = self.theme
        header = tk.Frame(self.window, bg=theme.base)
        header.pack(fill="x", padx=20, pady=(18, 8))
        tk.Label(
            header, text=TITLE, bg=theme.base, fg=theme.contrast,
            font=(self.font, 16, "bold"), anchor="w",
        ).pack(fill="x")
        tk.Label(
            header, text=SUBTITLE, bg=theme.base, fg=theme.text_secondary,
            font=(self.font, 9), anchor="w", justify="left", wraplength=860,
        ).pack(fill="x", pady=(4, 0))
        tk.Label(
            header, text=HONESTY_LINE, bg=theme.base, fg=theme.text_dim,
            font=(self.font, 8), anchor="w", justify="left", wraplength=860,
        ).pack(fill="x", pady=(6, 0))

        status_row = tk.Frame(self.window, bg=theme.base)
        status_row.pack(fill="x", padx=20, pady=(4, 6))
        self._status = tk.Label(
            status_row, text="", bg=theme.base, fg=theme.text_dim,
            font=(self.font, 9), anchor="w", justify="left", wraplength=760,
        )
        self._status.pack(side="left")
        self._download_button = tk.Button(
            status_row, text="Download model…", command=self._on_download,
            bg=theme.surface_raised, fg=theme.foreground,
            activebackground=theme.interactive_active, activeforeground=theme.contrast,
            relief="flat", borderwidth=0, font=(self.font, 9, "bold"),
            padx=12, pady=4, cursor="hand2",
        )

        transcript_frame = tk.Frame(self.window, bg=theme.base)
        transcript_frame.pack(fill="both", expand=True, padx=20)
        scrollbar = tk.Scrollbar(transcript_frame, command=lambda *a: self._transcript.yview(*a))
        scrollbar.pack(side="right", fill="y")
        self._transcript = tk.Text(
            transcript_frame, wrap="word", relief="flat", borderwidth=0,
            highlightthickness=1, highlightbackground=theme.border, bg=theme.surface,
            fg=theme.foreground, font=(self.font, 11), padx=10, pady=10,
            yscrollcommand=scrollbar.set, state="disabled", cursor="arrow",
        )
        self._transcript.pack(side="left", fill="both", expand=True)
        self._transcript.tag_configure("user", foreground=theme.primary, font=(self.font, 11, "bold"))
        self._transcript.tag_configure("ultron", foreground=theme.foreground, font=(self.font, 11))
        self._transcript.tag_configure("note", foreground=theme.text_dim, font=(self.font, 9, "italic"))

        composer = tk.Frame(self.window, bg=theme.base)
        composer.pack(fill="x", padx=20, pady=12)
        self._input = tk.Text(
            composer, height=2, wrap="word", relief="flat", borderwidth=0,
            highlightthickness=1, highlightbackground=theme.border,
            highlightcolor=theme.primary, bg=theme.surface, fg=theme.foreground,
            insertbackground=theme.primary, font=(self.font, 11), padx=8, pady=8,
        )
        self._input.pack(side="left", fill="both", expand=True)
        self._input.bind("<Return>", self._on_return)
        self._input.bind("<Shift-Return>", lambda _e: None)
        self._send = tk.Button(
            composer, text="Send", command=self._on_send,
            bg=theme.primary, fg=theme.primary_foreground,
            activebackground=theme.primary, activeforeground=theme.primary_foreground,
            relief="flat", borderwidth=0, font=(self.font, 10, "bold"),
            padx=18, pady=8, cursor="hand2",
        )
        self._send.pack(side="left", padx=(8, 0))
        self._append("note", f"{NAME}: ask anything. Enter sends; Shift+Enter makes a new line.")

    # -- helpers ----------------------------------------------------------

    def _append(self, tag: str, text: str) -> None:
        self._transcript.configure(state="normal")
        prefix = {"user": "You: ", "ultron": f"{NAME}: ", "note": ""}.get(tag, "")
        self._transcript.insert("end", f"{prefix}{text}\n\n", tag)
        self._transcript.configure(state="disabled")
        self._transcript.see("end")

    def _set_status(self, text: str, *, error: bool = False) -> None:
        self._status.configure(text=text, fg=self.theme.danger if error else self.theme.text_dim)

    def _refresh_status(self) -> None:
        if self.engine.is_available():
            self._status.configure(text=self.engine.describe(), fg=self.theme.text_dim)
            self._download_button.pack_forget()
            self._send.configure(state="normal")
            return
        if find_model_file() is None:
            note = (
                "No Ultron model yet. Download one (one-time, ~5 GB) or drop a `.gguf` "
                "file into the data/ultron folder."
            )
        else:
            note = (
                "A model file is present, but the runtime `llama-cpp-python` is not "
                f"installed. Install it with: {pip_install_command(FEATURE_ULTRON)}"
            )
        self._set_status(note)
        self._download_button.pack(side="right")
        self._send.configure(state="disabled")

    # -- behaviour --------------------------------------------------------

    def _on_return(self, _event: tk.Event) -> str:
        self._on_send()
        return "break"

    def _on_send(self) -> None:
        if self._busy or not self.engine.is_available():
            return
        prompt = self._input.get("1.0", "end").strip()
        if not prompt:
            return
        self._input.delete("1.0", "end")
        self._append("user", prompt)
        self._busy = True
        self._send.configure(state="disabled")
        self._set_status(f"{NAME} is thinking… (a CPU reply can take a while)")
        history = tuple(self._history)

        def run() -> None:
            try:
                answer = self.engine.reply(prompt, history=history)
                self._results.put(("reply", (prompt, answer), None))
            except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
                self._results.put(("reply", None, error))

        threading.Thread(target=run, daemon=True).start()
        self.window.after(80, self._pump)

    def _pump(self) -> None:
        try:
            _kind, payload, error = self._results.get_nowait()
        except queue.Empty:
            if self._busy:
                self.window.after(80, self._pump)
            return
        self._finish_reply(payload, error)

    def _finish_reply(self, payload: Any, error: Exception | None) -> None:
        self._busy = False
        self._send.configure(state="normal")
        if error is not None:
            message = (
                str(error)
                if isinstance(error, UltronUnavailable)
                else f"Could not answer: {error}"
            )
            self._append("note", message)
            self._set_status(message, error=True)
            return
        assert payload is not None
        prompt, answer = payload
        self._history.append(("user", prompt))
        self._history.append(("assistant", answer))
        self._append("ultron", answer)
        self._set_status(HEADER)

    def _on_download(self) -> None:
        if self._busy:
            return
        self._busy = True
        self._download_button.configure(state="disabled")
        self._send.configure(state="disabled")
        self._set_status(f"Downloading the {NAME} model… (one-time, ~5 GB)")

        def run() -> None:
            try:
                from core.ultron import download_model

                path = download_model(DEFAULT_SPEC_KEY, approved=True)
                self._results.put(("download", path, None))
            except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
                self._results.put(("download", None, error))

        threading.Thread(target=run, daemon=True).start()
        self.window.after(80, self._pump_download)

    def _pump_download(self) -> None:
        try:
            _kind, payload, error = self._results.get_nowait()
        except queue.Empty:
            self.window.after(80, self._pump_download)
            return
        self._busy = False
        self._download_button.configure(state="normal")
        if error is not None:
            self._set_status(f"Could not download the model: {error}", error=True)
        else:
            self._append("note", f"Model ready at {payload}.")
        self._refresh_status()

    def _close(self) -> None:
        if self._on_close is not None:
            self._on_close()
        self.window.destroy()


__all__ = ["HONESTY_LINE", "SUBTITLE", "TITLE", "UltronDialog"]
