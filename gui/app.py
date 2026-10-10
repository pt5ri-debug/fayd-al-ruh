"""The Tkinter window: a ChatGPT-style front end for Fayd Al-Ruh.

Tkinter ships with CPython, so the GUI adds no dependency. Long work (an answer
or a sidebar command) runs on a worker thread and posts results back through a
queue; Tk is only ever touched from the main thread, as Tk requires.

This module is imported lazily by the CLI, so importing it never opens a window
and never breaks the command-line path.
"""

from __future__ import annotations

import contextlib
import io
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import font as tkfont
from tkinter import simpledialog

from gui.actions import ACTIONS, GuiAction, SessionItem, saved_session_items
from gui.options import build_parser
from gui.service import ChatReply, ChatService

# A ChatGPT-like palette.
SIDEBAR_BG = "#202123"
SIDEBAR_HOVER = "#2A2B32"
MAIN_BG = "#343541"
ASSISTANT_BG = "#444654"
USER_BG = "#2A6FB0"
TEXT_COLOR = "#ECECF1"
MUTED_COLOR = "#9A9AA8"
ACCENT = "#10A37F"


def _pick_font() -> str:
    """A clean UI font, falling back to whatever Tk offers."""
    for name in ("Segoe UI", "Helvetica Neue", "DejaVu Sans", "Helvetica", "Arial"):
        if name in tkfont.families():
            return name
    return "TkDefaultFont"


class LineStream(io.TextIOBase):
    """A text stream that forwards each complete line to a queue.

    Training entry points print their live progress to stderr; the GUI redirects
    stderr through this stream so each line reaches the transcript as it is
    written, instead of arriving only after the run ends.
    """

    def __init__(self, lines: "queue.Queue[str]") -> None:
        self._lines = lines
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._lines.put(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self._lines.put(self._buffer)
        self._buffer = ""


class FaydGUI:
    """The application window and its wiring to :class:`ChatService`."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        service: ChatService | None = None,
        model_checkpoint: str | None = None,
        no_web: bool = False,
        session: str | None = None,
    ) -> None:
        self.root = root
        self.service = service or ChatService(
            no_web=no_web,
            model_checkpoint=model_checkpoint,
            session=session,
        )
        self._results: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._lines: queue.Queue[str] = queue.Queue()
        self._busy = False
        self._streaming_label: tk.Text | None = None
        self._bubbles: list[tk.Text] = []
        # Live output is coalesced and rendered at a bounded rate: a fast training
        # loop can emit thousands of lines, and configuring a label or scrolling
        # on every one would freeze the window.
        self._pending: list[str] = []
        self._last_render = 0.0
        self._font = _pick_font()

        # The transcript is selectable and copyable; Delete clears it, Ctrl+C
        # copies the selection, Ctrl+A selects the whole transcript, and a
        # right-click offers the same as a menu.
        root.bind("<Control-c>", self._on_copy)
        root.bind("<Control-C>", self._on_copy)
        root.bind("<Control-a>", self._on_select_all)
        root.bind("<Control-A>", self._on_select_all)
        root.bind("<Delete>", self._on_clear_transcript)
        root.bind("<Button-3>", self._on_right_click)
        self._clipboard_menu = tk.Menu(root, tearoff=0)
        self._clipboard_menu.add_command(label="Copy", command=self._on_copy)
        self._clipboard_menu.add_command(
            label="Copy conversation", command=self._on_copy_conversation
        )
        self._clipboard_menu.add_command(label="Select all", command=self._on_select_all)
        self._clipboard_menu.add_separator()
        self._clipboard_menu.add_command(label="Clear transcript", command=self._clear_transcript)

        root.title("Fayd Al-Ruh")
        root.geometry("1040x720")
        root.minsize(760, 520)
        root.configure(bg=MAIN_BG)

        self._build_sidebar()
        self._build_main()
        self._set_source_label()
        self._add_bubble(
            "assistant",
            "Welcome to Fayd Al-Ruh. Ask a question in the box below.\n"
            "Use “From sources” for cited answers, or “From model” for the trained model.",
        )
        self._pump()

    # ------------------------------------------------------------------ layout

    def _build_sidebar(self) -> None:
        sidebar = tk.Frame(self.root, bg=SIDEBAR_BG, width=260)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar, text="Fayd Al-Ruh", bg=SIDEBAR_BG, fg=TEXT_COLOR,
            font=(self._font, 15, "bold"), anchor="w", padx=16, pady=14,
        ).pack(fill="x")

        self._status = tk.Label(
            sidebar, text=self._status_text(), bg=SIDEBAR_BG, fg=MUTED_COLOR,
            font=(self._font, 9), anchor="w", justify="left", padx=16, pady=10,
        )
        self._status.pack(side="bottom", fill="x")

        # The actions and saved chats can outgrow the window, so the body scrolls.
        body = tk.Frame(sidebar, bg=SIDEBAR_BG)
        body.pack(side="top", fill="both", expand=True)
        canvas = tk.Canvas(body, bg=SIDEBAR_BG, highlightthickness=0, width=244)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar = tk.Scrollbar(body, command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scrollbar.set)
        panel = tk.Frame(canvas, bg=SIDEBAR_BG)
        window = canvas.create_window((0, 0), window=panel, anchor="nw")
        panel.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        for widget in (canvas, panel):
            widget.bind("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        self._sidebar_button(panel, "＋  New chat", self._on_new_chat).pack(
            fill="x", padx=10, pady=(0, 8)
        )

        tk.Label(
            panel, text="Answering with", bg=SIDEBAR_BG, fg=MUTED_COLOR,
            font=(self._font, 9), anchor="w", padx=16,
        ).pack(fill="x", pady=(6, 2))
        self._source_var = tk.StringVar(value="sources")
        for value, label in (("sources", "From sources (cited)"), ("model", "From trained model")):
            tk.Radiobutton(
                panel, text=label, value=value, variable=self._source_var,
                command=self._on_source_change, bg=SIDEBAR_BG, fg=TEXT_COLOR,
                selectcolor=SIDEBAR_BG, activebackground=SIDEBAR_BG,
                activeforeground=TEXT_COLOR, font=(self._font, 10),
                anchor="w", padx=10, highlightthickness=0, bd=0,
            ).pack(fill="x", padx=6)

        self._sidebar_button(panel, "💾  Save conversation", self._on_save).pack(
            fill="x", padx=10, pady=(8, 10)
        )

        tk.Label(
            panel, text="Actions", bg=SIDEBAR_BG, fg=MUTED_COLOR,
            font=(self._font, 9), anchor="w", padx=16,
        ).pack(fill="x", pady=(4, 2))
        self._actions_panel = tk.Frame(panel, bg=SIDEBAR_BG)
        self._actions_panel.pack(fill="x")
        self._action_buttons = self._populate_actions(self._actions_panel)

        tk.Label(
            panel, text="Saved chats", bg=SIDEBAR_BG, fg=MUTED_COLOR,
            font=(self._font, 9), anchor="w", padx=16,
        ).pack(fill="x", pady=(12, 2))
        self._sessions_panel = tk.Frame(panel, bg=SIDEBAR_BG)
        self._sessions_panel.pack(fill="x")
        self._session_buttons: list[tk.Widget] = []
        self.refresh_sessions()

    def _populate_actions(self, parent: tk.Misc) -> list[tk.Widget]:
        """Fill the Actions section with one button per entry (no duplicates)."""
        buttons: list[tk.Widget] = []
        for action in ACTIONS:
            button = self._action_button(parent, action)
            button.pack(fill="x", padx=10, pady=1)
            buttons.append(button)
        return buttons

    def refresh_sessions(self) -> None:
        """Rebuild the Saved chats list from the store, marking the current one."""
        for widget in self._session_buttons:
            widget.destroy()
        self._session_buttons = []
        try:
            saved = self.service.list_sessions()
        except Exception:  # a broken store must not blank the sidebar
            saved = ()
        items = saved_session_items(saved, current=self.service.session)
        if not items:
            hint = tk.Label(
                self._sessions_panel, text="No saved chats yet.", bg=SIDEBAR_BG,
                fg=MUTED_COLOR, font=(self._font, 9), anchor="w", padx=16, pady=2,
            )
            hint.pack(fill="x")
            self._session_buttons.append(hint)
            return
        for item in items:
            button = self._session_button(self._sessions_panel, item)
            button.pack(fill="x", padx=10, pady=1)
            self._session_buttons.append(button)

    def _session_button(self, parent: tk.Misc, item: SessionItem) -> tk.Label:
        """A sidebar button that resumes one saved chat."""
        label = item.label
        return self._sidebar_button(
            parent, label, lambda: self._on_resume_session(item), small=True
        )

    def _action_button(self, parent: tk.Misc, action: GuiAction) -> tk.Label:
        """A sidebar button bound to one action (avoids a late-bound lambda)."""
        return self._sidebar_button(
            parent, action.label, lambda: self._on_action(action), small=True
        )

    def _sidebar_button(
        self, parent: tk.Misc, text: str, command: Callable[[], Any], *, small: bool = False
    ) -> tk.Label:
        """A flat, hoverable label that behaves like a button."""
        size = 9 if small else 11
        button = tk.Label(
            parent, text=text, bg=SIDEBAR_BG, fg=TEXT_COLOR,
            font=(self._font, size), anchor="w", padx=12,
            pady=(3 if small else 8),
            justify="left", wraplength=216 if small else 0,
        )
        button.bind("<Enter>", lambda _e: button.configure(bg=SIDEBAR_HOVER))
        button.bind("<Leave>", lambda _e: button.configure(bg=SIDEBAR_BG))
        button.bind("<Button-1>", lambda _e: command())
        return button

    def _build_main(self) -> None:
        main = tk.Frame(self.root, bg=MAIN_BG)
        main.pack(side="left", fill="both", expand=True)

        header = tk.Frame(main, bg=MAIN_BG)
        header.pack(fill="x", padx=20, pady=(14, 6))
        tk.Label(
            header, text="Chat", bg=MAIN_BG, fg=TEXT_COLOR,
            font=(self._font, 14, "bold"), anchor="w",
        ).pack(side="left")
        self._mode_label = tk.Label(
            header, text="", bg=MAIN_BG, fg=MUTED_COLOR,
            font=(self._font, 10), anchor="e",
        )
        self._mode_label.pack(side="right")

        # Scrollable transcript.
        container = tk.Frame(main, bg=MAIN_BG)
        container.pack(fill="both", expand=True)
        self._canvas = tk.Canvas(container, bg=MAIN_BG, highlightthickness=0)
        self._canvas.pack(side="left", fill="both", expand=True)
        scrollbar = tk.Scrollbar(container, command=self._canvas.yview)
        scrollbar.pack(side="right", fill="y")
        self._canvas.configure(yscrollcommand=scrollbar.set)

        self._transcript = tk.Frame(self._canvas, bg=MAIN_BG)
        self._window = self._canvas.create_window((0, 0), window=self._transcript, anchor="nw")
        self._transcript.bind("<Configure>", self._on_transcript_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        for widget in (self._canvas, self._transcript):
            widget.bind("<MouseWheel>", self._on_mousewheel)
            widget.bind("<Button-4>", self._on_mousewheel)
            widget.bind("<Button-5>", self._on_mousewheel)

        # Composer.
        composer = tk.Frame(main, bg=MAIN_BG)
        composer.pack(fill="x", padx=20, pady=(6, 16))
        self._entry = tk.Text(
            composer, height=2, bg=ASSISTANT_BG, fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR, font=(self._font, 11),
            relief="flat", wrap="word", padx=10, pady=8,
        )
        self._entry.pack(side="left", fill="both", expand=True)
        self._entry.bind("<Return>", self._on_enter)
        self._entry.bind("<Shift-Return>", lambda _e: None)
        self._send = tk.Button(
            composer, text="Send", command=self._on_send, bg=ACCENT,
            fg="white", activebackground="#0E8C6D", activeforeground="white",
            font=(self._font, 11, "bold"), relief="flat", padx=18, pady=8,
            cursor="hand2",
        )
        self._send.pack(side="right", fill="y", padx=(8, 0))
        self._entry.focus_set()

    # -------------------------------------------------------------- transcript

    def _add_bubble(self, role: str, text: str, *, extra: str = "") -> tk.Text:
        row = tk.Frame(self._transcript, bg=MAIN_BG)
        row.pack(fill="x", padx=16, pady=6)
        if role == "user":
            row.pack_configure(anchor="e")
            bg, fg = USER_BG, "#FFFFFF"
        elif role == "note":
            bg, fg = MAIN_BG, MUTED_COLOR
        else:
            bg, fg = ASSISTANT_BG, TEXT_COLOR

        bubble = tk.Frame(row, bg=bg)
        bubble.pack(side="right" if role == "user" else "left")
        # A read-only Text, not a Label: its text can be selected and copied.
        widget = tk.Text(
            bubble,
            height=1, wrap="word", relief="flat", borderwidth=0,
            highlightthickness=0, bg=bg, fg=fg, font=(self._font, 11),
            padx=14, pady=10, cursor="xterm", takefocus=1,
        )
        widget.insert("1.0", text if not extra else f"{text}\n\n{extra}")
        self._make_read_only(widget)
        widget.pack(fill="x")
        for wheel in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(wheel, self._on_bubble_wheel, add="+")
        self._fit_height(widget)
        self._bubbles.append(widget)
        self._scroll_to_bottom()
        return widget

    @staticmethod
    def _make_read_only(widget: tk.Text) -> None:
        """Keep a Text editable-looking but read-only, yet still selectable/copyable.

        Blocking only the mutating keys (rather than ``state="disabled"``) keeps
        selection, Ctrl+C, and the mouse working normally.
        """
        def block(event: tk.Event) -> str:
            if int(event.state) & 0x0004 or int(event.state) & 0x0008:  # Control / Command
                return ""  # let accelerators (copy, select-all) through
            if event.keysym in {"Left", "Right", "Up", "Down", "Home", "End", "Prior", "Next"}:
                return ""  # allow navigation
            return "break"

        for sequence in ("<Key>", "<<Paste>>", "<<Cut>>", "<<Clear>>"):
            widget.bind(sequence, block)

    def _on_bubble_wheel(self, event: tk.Event) -> str:
        """Scroll the transcript when the bubble has nothing of its own to scroll."""
        widget = event.widget
        spans = widget.yview() if isinstance(widget, tk.Text) else (0.0, 1.0)
        if spans[0] <= 0.0 and spans[1] >= 1.0:
            self._on_mousewheel(event)
            return "break"
        return ""

    def _fit_height(self, widget: tk.Text) -> None:
        """Size a read-only Text to its wrapped content, up to a scroll cap."""
        widget.update_idletasks()
        measured = widget.count("1.0", "end-1c", "displaylines")
        lines = int(measured[0]) if measured else 1
        widget.configure(height=min(max(lines, 1), 30))

    def _bubble_texts(self) -> list[str]:
        return [widget.get("1.0", "end-1c") for widget in self._bubbles]

    def _on_right_click(self, event: tk.Event) -> None:
        try:
            self._clipboard_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._clipboard_menu.grab_release()

    def _copy_text(self, text: str) -> None:
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update_idletasks()

    def _selected_text(self) -> str:
        """The selection from the focused Text widget, falling back to any selection."""
        widget = self.root.focus_get()
        if isinstance(widget, tk.Text):
            try:
                if widget.tag_ranges("sel"):
                    return widget.get("sel.first", "sel.last")
            except tk.TclError:
                pass
        for candidate in self._bubbles:
            try:
                if candidate.tag_ranges("sel"):
                    return candidate.get("sel.first", "sel.last")
            except tk.TclError:
                continue
        return ""

    def _on_copy(self, _event: tk.Event | None = None) -> str:
        """Copy the focused or transcript selection; never clobber it with nothing."""
        selection = self._selected_text()
        if not selection:
            return ""  # nothing selected: leave the clipboard (and composer) alone
        self._copy_text(selection)
        return "break"

    def _on_copy_conversation(self) -> None:
        self._copy_text("\n\n".join(self._bubble_texts()))

    def _on_select_all(self, _event: tk.Event | None = None) -> str:
        if self.root.focus_get() is self._entry:
            return ""  # keep the composer's own editing untouched
        for widget in self._bubbles:
            widget.tag_add("sel", "1.0", "end-1c")
        return "break"

    def _on_clear_transcript(self, _event: tk.Event | None = None) -> str:
        if self.root.focus_get() is self._entry:
            return ""  # Delete while typing edits the composer, never the transcript
        self._clear_transcript()
        return "break"

    def _clear_transcript(self) -> None:
        for widget in list(self._bubbles):
            widget.master.destroy()  # the wrapping row frame
        self._bubbles.clear()

    def _on_transcript_configure(self, _event: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        delta = -1 * (event.delta // 120) if getattr(event, "delta", 0) else (
            -1 if getattr(event, "num", 0) == 4 else 1
        )
        self._canvas.yview_scroll(delta, "units")

    def _scroll_to_bottom(self) -> None:
        self._canvas.update_idletasks()
        self._canvas.yview_moveto(1.0)

    # --------------------------------------------------------------- behavior

    def _set_source_label(self) -> None:
        self._mode_label.configure(text=self._source_label_text())

    def _status_text(self) -> str:
        return f"Session: {self.service.session or 'unsaved'}\nCtrl+Enter or click Send."

    def _on_enter(self, _event: tk.Event) -> str:
        self._on_send()
        return "break"  # keep the newline out of the box

    def _on_send(self) -> None:
        if self._busy:
            return
        message = self._entry.get("1.0", "end").strip()
        if not message:
            return
        self._entry.delete("1.0", "end")
        self._add_bubble("user", message)
        self._start_work(lambda: self.service.ask(message))

    def _on_new_chat(self) -> None:
        conversation_id = self.service.new_conversation()
        self._add_bubble("note", f"Started a new conversation ({conversation_id}).")
        self._update_status()
        self.refresh_sessions()

    def _on_save(self) -> None:
        self._add_bubble("note", self.service.save())
        self._update_status()
        self.refresh_sessions()

    def _on_resume_session(self, item: SessionItem) -> None:
        if self._busy or item.is_current:
            return
        if self.service.resume(item.session_id):
            self._add_bubble("note", f"Resumed conversation {item.session_id}.")
            self._update_status()
            self.refresh_sessions()
        else:
            self._add_bubble("note", f"Could not resume conversation {item.session_id}.")
            self.refresh_sessions()

    def _update_status(self) -> None:
        self._status.configure(text=self._status_text())

    def _on_source_change(self) -> None:
        want = self._source_var.get()
        if want == self.service.source:
            return
        try:
            if want == "model":
                self.service.use_model()
            else:
                self.service.use_sources()
        except (FileNotFoundError, OSError, ValueError) as error:
            self._source_var.set(self.service.source)
            self._add_bubble(
                "note",
                f"Could not load the trained model: {error}\n"
                "Train it first with the “Train the language model” action.",
            )
            return
        self._set_source_label()
        self._add_bubble(
            "note",
            "Now answering from the trained model (no sources)."
            if want == "model"
            else "Now answering from sources, with citations.",
        )

    def _on_action(self, action: GuiAction) -> None:
        if self._busy:
            return
        argv = action.argv
        if action.needs_input:
            question = self._prompt_for_question(action.label)
            if not question:
                return
            argv = (*argv, question)
        self._add_bubble("note", f"Running: {action.label} …")
        self._streaming_label = self._add_bubble("note", f"{action.label} — live output:")
        self._start_work(lambda: self._run_action(action, argv))

    def _prompt_for_question(self, title: str) -> str:
        """Ask for the extra argument an action needs; empty means cancelled."""
        return (simpledialog.askstring(title, "Question:", parent=self.root) or "").strip()

    def _run_action(self, action: GuiAction, argv: tuple[str, ...]) -> str:
        # Capture stdout (the report) while streaming stderr -- the live training
        # progress -- into the transcript line by line, so a long run never looks
        # frozen and the operator can watch the time remaining.
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(LineStream(self._lines)):
            action.command(argv)
        output = buffer.getvalue().strip()
        return output or f"{action.label} finished."

    def _append_progress_line(self, text: str) -> None:
        """Queue one live output line; the pump flushes them at a bounded rate."""
        self._pending.append(text)

    def _flush_progress(self, *, force: bool = False) -> None:
        """Render queued progress lines, coalesced and rate-limited.

        A training loop can emit many lines between pumps; joining them into one
        label update keeps the window responsive instead of reconfiguring a label
        and re-scrolling on every single line.
        """
        if not self._pending:
            return
        now = time.monotonic()
        if not force and now - self._last_render < 0.15:
            return
        text = "\n".join(self._pending)
        self._pending.clear()
        self._last_render = now
        if self._streaming_label is None:
            self._add_bubble("note", text)
        else:
            self._streaming_label.insert("end", f"\n{text}")
            self._fit_height(self._streaming_label)
        self._scroll_to_bottom()

    # -------------------------------------------------------------- threading

    def _start_work(self, job) -> None:
        """Run ``job`` on a worker thread; its result becomes an assistant bubble."""
        self._busy = True
        self._send.configure(state="disabled")
        self._status.configure(text="Working…\nThis can take a moment.")

        def run() -> None:
            try:
                self._results.put(("ok", job()))
            except Exception as error:  # surfaced in the transcript, never crashes Tk
                self._results.put(("error", error))

        threading.Thread(target=run, daemon=True).start()

    def _pump(self) -> None:
        """Drain worker results and live progress lines on the Tk main thread."""
        self._drain_lines(cap=500)
        self._flush_progress()
        try:
            kind, payload = self._results.get_nowait()
        except queue.Empty:
            kind = None
        if kind is not None:
            # The job has ended, so every line it printed is already queued;
            # drain them all into the same live bubble before showing the result.
            self._drain_lines(cap=None)
            self._flush_progress(force=True)
            if kind == "ok":
                self._render_result(payload)
            else:
                self._add_bubble("assistant", f"Something went wrong: {payload}")
            self._busy = False
            self._streaming_label = None
            self._send.configure(state="normal")
            self._update_status()
            self.refresh_sessions()
        self.root.after(100, self._pump)

    def _drain_lines(self, *, cap: int | None) -> None:
        """Move queued live-output lines into the pending buffer, optionally capped."""
        drained = 0
        while cap is None or drained < cap:
            try:
                self._append_progress_line(self._lines.get_nowait())
                drained += 1
            except queue.Empty:
                break

    def _render_result(self, payload: Any) -> None:
        if isinstance(payload, ChatReply):
            self._add_bubble("assistant", payload.text)
            if payload.hint:
                self._add_bubble("note", payload.hint)
        else:
            self._add_bubble("assistant", str(payload))
        self._mode_label.configure(text=self._source_label_text())
        self._entry.focus_set()

    def _source_label_text(self) -> str:
        return (
            "Answering from the trained model (no sources)"
            if self.service.source == "model"
            else "Answering from sources (cited)"
        )


def main(argv: list[str] | None = None) -> int:
    """Launch the GUI. Returns a non-zero code when no display is available."""
    args = build_parser().parse_args(argv)
    try:
        root = tk.Tk()
    except tk.TclError as error:
        print(f"The graphical interface needs a display: {error}")
        print("Use the command line instead: python -m cli chat")
        return 1

    service = ChatService(
        no_web=args.no_web,
        root=Path(args.root) if args.root else None,
        read_write=args.read_write,
        model_checkpoint=args.model_checkpoint,
        session=args.session,
    )
    if args.mind:
        try:
            service.use_model()
        except (FileNotFoundError, OSError, ValueError) as error:
            print(f"Could not load the trained model: {error}")
            return 1

    FaydGUI(root, service=service)
    root.mainloop()
    return 0


__all__ = ["FaydGUI", "main"]
