"""The Code Studio window: edit the project's own code, test it, then merge.

This is the project's "code host in a window". It borrows the model a developer
already uses on a code host -- a branch per change, reviewed and merged when it is
good -- and implements it with the project's own ``git``:

* **A workspace is an isolated copy.** *New workspace* makes a ``git worktree``
  under ``data/studio/workspaces/<slug>`` on its own ``studio/<slug>`` branch, cut
  from ``main``. Everything you and the coding model change happens there, so the
  running checkout is never touched and a broken change cannot stop the program.
* **The coding model edits it.** *Ask the model* sends a plain-language request to
  a local coding model (``qwen2.5-coder:7b`` through Ollama by default) which reads
  and writes files *inside the workspace only*, through a strict JSON protocol.
  Every step it takes is shown, so its work is reviewable rather than magic.
* **Test, review, merge.** *Run checks* runs the project's own test command inside
  the workspace; *Show diff* is the change as a unified diff; and **Merge** appears
  only when the checks have passed and ``main`` has not moved on, so a merge can
  never overwrite newer work. *Discard* throws the whole change away.

The window is a thin view over :class:`gui.code_studio.CodeStudioService`, so all
of the logic is testable with no display. Tk is imported lazily with the rest of
the window, so the CLI never touches it.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from typing import Any

import tkinter as tk
from tkinter import ttk

from core.coder_model import PULL_HINT

TITLE = "Code Studio"
SUBTITLE = (
    "Change the project's own code safely: every change gets its own isolated workspace (a "
    "git worktree on its own branch). Ask the local coding model to make the change, review "
    "the diff, run the tests, then merge — or discard it. The running program is never edited "
    "directly, so a half-finished change can never break it."
)
HONESTY_LINE = (
    "The coding model's edits are real file writes inside the workspace, and nothing it writes "
    "reaches the project until you press Merge — which is refused unless the checks pass and "
    "the base branch has not moved on."
)

#: The default coding model, shown in the status line so the choice is explicit.
DEFAULT_MODEL_NOTE = f"Coding model: qwen2.5-coder:7b (Ollama). If it is missing: {PULL_HINT}"


class CodeStudioDialog:
    """A window for editing, testing, and merging changes to the project."""

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
        self.service = service
        self.theme = theme
        self.font = font
        self.mono = mono
        self._on_close = on_close
        self._busy = False
        self._closed = False
        self._current_slug: str = ""
        self._current_file: str = ""
        self._dirty = False
        # Worker threads hand results back through a queue that the UI thread
        # drains on its own clock. Calling `after` from a worker thread races with
        # the main loop (Tk is not thread-safe), which is what the project's own
        # window avoids the same way.
        self._results: queue.Queue[tuple[Callable[[Any, Exception | None], None], Any, Exception | None]] = (
            queue.Queue()
        )

        self._task_var = tk.StringVar(value="")
        self._request_var = tk.StringVar(value="")

        self.window = tk.Toplevel(parent)
        self.window.title(TITLE)
        self.window.configure(bg=theme.base)
        self.window.geometry("1180x608")
        self.window.minsize(900, 600)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._build()
        self.refresh()

    # -- layout -----------------------------------------------------------

    def _build(self) -> None:
        theme = self.theme
        header = tk.Frame(self.window, bg=theme.base)
        header.pack(fill="x", padx=20, pady=(18, 6))
        tk.Label(
            header, text=TITLE, bg=theme.base, fg=theme.contrast,
            font=(self.font, 15, "bold"), anchor="w",
        ).pack(fill="x")
        tk.Label(
            header, text=SUBTITLE, bg=theme.base, fg=theme.text_secondary,
            font=(self.font, 9), anchor="w", justify="left", wraplength=1120,
        ).pack(fill="x", pady=(4, 0))
        self._status_line = tk.Label(
            header, text=DEFAULT_MODEL_NOTE, bg=theme.base, fg=theme.text_dim,
            font=(self.font, 8), anchor="w", justify="left", wraplength=1120,
        )
        self._status_line.pack(fill="x", pady=(6, 0))

        body = tk.Frame(self.window, bg=theme.base)
        body.pack(fill="both", expand=True, padx=20, pady=(6, 0))
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=0, minsize=300)
        body.columnconfigure(1, weight=1)

        self._build_workspaces(body)
        self._build_work_area(body)
        self._build_action_bar()

    def _build_workspaces(self, parent: tk.Misc) -> None:
        theme = self.theme
        column = tk.Frame(parent, bg=theme.surface, highlightthickness=1,
                          highlightbackground=theme.border_subtle)
        column.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        tk.Label(
            column, text="1.  Workspaces", bg=theme.surface, fg=theme.contrast,
            font=(self.font, 10, "bold"), anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 2))

        self._workspace_list = tk.Listbox(
            column, bg=theme.base, fg=theme.foreground, selectbackground=theme.primary,
            selectforeground=theme.contrast, relief="flat", highlightthickness=1,
            highlightbackground=theme.border, font=(self.font, 9), height=10,
            activestyle="none",
        )
        self._workspace_list.pack(fill="both", expand=True, padx=12, pady=(2, 6))
        self._workspace_list.bind("<<ListboxSelect>>", self._on_workspace_select)

        tk.Label(
            column, text="Describe the change, then create its workspace:",
            bg=theme.surface, fg=theme.text_secondary, font=(self.font, 8),
            anchor="w", justify="left", wraplength=270,
        ).pack(fill="x", padx=12, pady=(2, 2))
        tk.Entry(
            column, textvariable=self._task_var, bg=theme.base, fg=theme.foreground,
            insertbackground=theme.primary, relief="flat", highlightthickness=1,
            highlightbackground=theme.border, font=(self.font, 9),
        ).pack(fill="x", padx=12, pady=(0, 4))
        row = tk.Frame(column, bg=theme.surface)
        row.pack(fill="x", padx=12, pady=(0, 10))
        self._button(row, "New workspace", self._on_create_workspace).pack(side="left")
        self._button(row, "Refresh", self.refresh).pack(side="left", padx=(6, 0))

    def _build_work_area(self, parent: tk.Misc) -> None:
        theme = self.theme
        area = tk.Frame(parent, bg=theme.base)
        area.grid(row=0, column=1, sticky="nsew")

        self._workspace_label = tk.Label(
            area, text="No workspace selected.", bg=theme.base, fg=theme.text_secondary,
            font=(self.font, 10, "bold"), anchor="w",
        )
        self._workspace_label.pack(fill="x", pady=(0, 4))

        notebook = ttk.Notebook(area)
        notebook.pack(fill="both", expand=True)
        self._notebook = notebook

        self._build_editor_tab(notebook)
        self._build_model_tab(notebook)
        self._build_review_tab(notebook)

    def _build_editor_tab(self, notebook: ttk.Notebook) -> None:
        theme = self.theme
        tab = tk.Frame(notebook, bg=theme.surface)
        notebook.add(tab, text="  2.  Files and editor  ")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=0, minsize=230)
        tab.columnconfigure(1, weight=1)

        self._file_list = tk.Listbox(
            tab, bg=theme.base, fg=theme.foreground, selectbackground=theme.primary,
            selectforeground=theme.contrast, relief="flat", highlightthickness=1,
            highlightbackground=theme.border, font=(self.mono, 8), activestyle="none",
        )
        self._file_list.grid(row=0, column=0, sticky="nsew", padx=(12, 8), pady=10)
        self._file_list.bind("<<ListboxSelect>>", self._on_file_select)

        right = tk.Frame(tab, bg=theme.surface)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=10)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        self._file_label = tk.Label(
            right, text="Select a file to open it.", bg=theme.surface, fg=theme.text_secondary,
            font=(self.mono, 9), anchor="w",
        )
        self._file_label.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self._editor = tk.Text(
            right, wrap="none", undo=True, bg=theme.base, fg=theme.foreground,
            insertbackground=theme.primary, relief="flat", highlightthickness=1,
            highlightbackground=theme.border, font=(self.mono, 9), padx=6, pady=6,
        )
        self._editor.grid(row=1, column=0, sticky="nsew")
        self._editor.bind("<<Modified>>", self._on_editor_modified)
        save_row = tk.Frame(right, bg=theme.surface)
        save_row.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self._button(save_row, "Save file", self._on_save_file).pack(side="left")
        tk.Label(
            save_row, text="Edits go to the workspace, not to the running program.",
            bg=theme.surface, fg=theme.text_dim, font=(self.font, 8),
        ).pack(side="left", padx=8)

    def _build_model_tab(self, notebook: ttk.Notebook) -> None:
        theme = self.theme
        tab = tk.Frame(notebook, bg=theme.surface)
        notebook.add(tab, text="  3.  Ask the coding model  ")
        tab.rowconfigure(2, weight=1)
        tab.columnconfigure(0, weight=1)

        tk.Label(
            tab, text="Describe what to change in plain language. The model reads and writes "
            "files inside this workspace only.",
            bg=theme.surface, fg=theme.text_secondary, font=(self.font, 9), anchor="w",
            justify="left", wraplength=760,
        ).grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        entry_row = tk.Frame(tab, bg=theme.surface)
        entry_row.grid(row=1, column=0, sticky="ew", padx=12)
        entry_row.columnconfigure(0, weight=1)
        tk.Entry(
            entry_row, textvariable=self._request_var, bg=theme.base, fg=theme.foreground,
            insertbackground=theme.primary, relief="flat", highlightthickness=1,
            highlightbackground=theme.border, font=(self.font, 10),
        ).grid(row=0, column=0, sticky="ew")
        self._ask_button = self._button(entry_row, "Ask the model", self._on_ask_model)
        self._ask_button.grid(row=0, column=1, padx=(8, 0))

        self._trace = tk.Text(
            tab, wrap="word", bg=theme.base, fg=theme.foreground, relief="flat",
            highlightthickness=1, highlightbackground=theme.border,
            font=(self.mono, 8), padx=6, pady=6,
        )
        self._trace.grid(row=2, column=0, sticky="nsew", padx=12, pady=10)
        self._trace.configure(state="disabled")

    def _build_review_tab(self, notebook: ttk.Notebook) -> None:
        theme = self.theme
        tab = tk.Frame(notebook, bg=theme.surface)
        notebook.add(tab, text="  4.  Diff and checks  ")
        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)

        row = tk.Frame(tab, bg=theme.surface)
        row.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        self._button(row, "Show diff", self._on_show_diff).pack(side="left")
        self._checks_button = self._button(row, "Run checks (tests)", self._on_run_checks)
        self._checks_button.pack(side="left", padx=6)
        self._checks_label = tk.Label(
            row, text="Checks have not run yet.", bg=theme.surface, fg=theme.text_dim,
            font=(self.font, 8),
        )
        self._checks_label.pack(side="left", padx=8)

        self._review = tk.Text(
            tab, wrap="none", bg=theme.base, fg=theme.foreground, relief="flat",
            highlightthickness=1, highlightbackground=theme.border,
            font=(self.mono, 8), padx=6, pady=6,
        )
        self._review.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 10))
        self._review.configure(state="disabled")

    def _build_action_bar(self) -> None:
        theme = self.theme
        bar = tk.Frame(self.window, bg=theme.base)
        bar.pack(fill="x", padx=20, pady=(8, 12))
        self._merge_button = self._button(bar, "Merge into the project", self._on_merge)
        self._merge_button.pack(side="left")
        self._merge_button.configure(state="disabled")
        self._button(bar, "Discard workspace", self._on_discard).pack(side="left", padx=6)
        self._merge_note = tk.Label(
            bar, text="", bg=theme.base, fg=theme.text_secondary, font=(self.font, 9),
            anchor="w", justify="left", wraplength=560,
        )
        self._merge_note.pack(side="left", padx=10)
        self._spinner = ttk.Progressbar(bar, mode="indeterminate", length=140)

    def _button(self, parent: tk.Misc, text: str, command: Any) -> tk.Button:
        theme = self.theme
        return tk.Button(
            parent, text=text, command=command, bg=theme.surface_raised,
            fg=theme.foreground, activebackground=theme.interactive_active,
            activeforeground=theme.contrast, relief="flat", borderwidth=0,
            font=(self.font, 9, "bold"), padx=12, pady=4, cursor="hand2",
        )

    # -- state ------------------------------------------------------------

    def refresh(self) -> None:
        """Reload the workspace list and the model status line."""
        state = self.service.state()
        self._status_line.configure(text=state.coder.summary() if state.coder else "")
        self._workspace_list.delete(0, "end")
        self._workspaces = state.workspaces
        for workspace in state.workspaces:
            label = workspace.slug
            if workspace.checks_passed is True:
                label += "   ✓ checks passed"
            elif workspace.checks_passed is False:
                label += "   ✗ checks failed"
            self._workspace_list.insert("end", label)
        if not state.repository:
            self._workspace_label.configure(
                text="This is not a git checkout, so the Studio cannot make workspaces."
            )
        if self._current_slug and not any(w.slug == self._current_slug for w in state.workspaces):
            self._current_slug = ""
            self._workspace_label.configure(text="No workspace selected.")

    def _selected_slug(self) -> str:
        selection = self._workspace_list.curselection()
        if not selection:
            return ""
        return self._workspaces[selection[0]].slug

    # -- workspaces -------------------------------------------------------

    def _on_workspace_select(self, _event: tk.Event | None = None) -> None:
        slug = self._selected_slug()
        if not slug:
            return
        self._current_slug = slug
        workspace = self.service.workspace(slug)
        self._workspace_label.configure(
            text=f"{workspace.slug}  ·  branch {workspace.branch}  ·  from "
            f"{workspace.base_revision}"
            + (f"  ·  {workspace.task}" if workspace.task else "")
        )
        self._checks_label.configure(
            text=workspace.checks_summary or "Checks have not run yet.",
            fg=self.theme.text_secondary if workspace.checks_passed else self.theme.text_dim,
        )
        self._refresh_files()
        self._refresh_merge_state()
        self._set_trace(workspace.task or "")

    def _on_create_workspace(self) -> None:
        task = self._task_var.get().strip()
        if not task:
            self._set_merge_note("Describe the change first.", error=True)
            return
        if self._busy:
            return
        self._run_background(
            "Creating the workspace…",
            lambda: self.service.create_workspace(task),
            self._after_create,
        )

    def _after_create(self, report: Any, error: Exception | None) -> None:
        if error is not None:
            self._set_merge_note(f"Could not create the workspace: {error}", error=True)
            return
        if not report.ok:
            self._set_merge_note(report.message, error=True)
            return
        self._task_var.set("")
        self.refresh()
        self._set_merge_note(report.message)

    def _on_discard(self) -> None:
        slug = self._current_slug or self._selected_slug()
        if not slug:
            self._set_merge_note("Select a workspace to discard.", error=True)
            return
        report = self.service.remove_workspace(slug)
        self._current_slug = ""
        self._current_file = ""
        self._editor.delete("1.0", "end")
        self._file_list.delete(0, "end")
        self._set_review("")
        self.refresh()
        self._set_merge_note(report.message, error=not report.ok)

    # -- editor -----------------------------------------------------------

    def _refresh_files(self) -> None:
        self._file_list.delete(0, "end")
        if not self._current_slug:
            return
        try:
            entries = self.service.list_files(self._current_slug)
        except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
            self._set_merge_note(f"Could not list files: {error}", error=True)
            return
        self._files = entries
        for entry in entries:
            self._file_list.insert("end", entry.path)

    def _on_file_select(self, _event: tk.Event | None = None) -> None:
        selection = self._file_list.curselection()
        if not selection:
            return
        relative = self._files[selection[0]].path
        report = self.service.read_file(self._current_slug, relative)
        if not report.ok:
            self._set_merge_note(report.message, error=True)
            return
        self._current_file = relative
        self._file_label.configure(text=relative)
        self._editor.delete("1.0", "end")
        self._editor.insert("1.0", report.detail)
        self._editor.edit_modified(False)
        self._dirty = False

    def _on_editor_modified(self, _event: tk.Event | None = None) -> None:
        if self._editor.edit_modified():
            self._dirty = True
            self._file_label.configure(text=f"{self._current_file}  (unsaved)")
            self._editor.edit_modified(False)

    def _on_save_file(self) -> None:
        if not self._current_slug or not self._current_file:
            self._set_merge_note("Open a file first.", error=True)
            return
        content = self._editor.get("1.0", "end-1c")
        report = self.service.save_file(self._current_slug, self._current_file, content)
        self._set_merge_note(report.message, error=not report.ok)
        if report.ok:
            self._dirty = False
            self._file_label.configure(text=self._current_file)
            self._refresh_merge_state(keep_note=True)

    # -- model ------------------------------------------------------------

    def _on_ask_model(self) -> None:
        request = self._request_var.get().strip()
        if not self._current_slug:
            self._set_merge_note("Select a workspace first.", error=True)
            return
        if not request:
            self._set_merge_note("Describe the change you want the model to make.", error=True)
            return
        if self._busy:
            return
        self._set_trace("Asking the coding model…\n")
        self._run_background(
            "The coding model is working…",
            lambda: self.service.run_agent(self._current_slug, request),
            self._after_ask,
        )

    def _after_ask(self, result: Any, error: Exception | None) -> None:
        if error is not None:
            self._set_trace(f"The coding model failed: {error}")
            return
        report, trace = result
        if trace is not None:
            self._set_trace(trace.render())
        else:
            self._set_trace(report.message)
        self._set_merge_note(report.message, error=not report.ok)
        self.refresh()
        self._refresh_files()
        self._refresh_merge_state(keep_note=True)

    # -- diff and checks --------------------------------------------------

    def _on_show_diff(self) -> None:
        if not self._current_slug:
            self._set_merge_note("Select a workspace first.", error=True)
            return
        report = self.service.diff(self._current_slug)
        self._set_review(report.detail if report.ok else report.message)
        self._notebook.select(2)

    def _on_run_checks(self) -> None:
        if not self._current_slug:
            self._set_merge_note("Select a workspace first.", error=True)
            return
        if self._busy:
            return
        self._checks_label.configure(text="Running the checks…", fg=self.theme.text_dim)
        self._run_background(
            "Running the project's checks…",
            lambda: self.service.run_checks(self._current_slug),
            self._after_checks,
        )

    def _after_checks(self, result: Any, error: Exception | None) -> None:
        if error is not None:
            self._checks_label.configure(text=f"Checks could not run: {error}", fg=self.theme.danger)
            return
        report, _check = result
        self._set_review(report.detail or report.message)
        self._checks_label.configure(
            text=report.message, fg=self.theme.text_secondary if report.ok else self.theme.danger
        )
        self.refresh()
        self._refresh_merge_state()

    # -- merge ------------------------------------------------------------

    def _refresh_merge_state(self, *, keep_note: bool = False) -> None:
        """Enable Merge only when the gate allows it.

        ``keep_note`` leaves the current message alone, so a caller that has just
        reported something to the user (a save, a model run) does not have its
        confirmation replaced by the merge hint.
        """
        if not self._current_slug:
            self._merge_button.configure(state="disabled")
            if not keep_note:
                self._merge_note.configure(text="")
            return
        report = self.service.can_merge(self._current_slug)
        self._merge_button.configure(state="normal" if report.ok else "disabled")
        if not keep_note:
            self._merge_note.configure(
                text=report.message, fg=self.theme.text_secondary if report.ok else self.theme.text_dim
            )

    def _on_merge(self) -> None:
        if not self._current_slug or self._busy:
            return
        slug = self._current_slug
        self._run_background(
            f"Merging `{slug}` into the project…",
            lambda: self.service.merge(slug),
            self._after_merge,
        )

    def _after_merge(self, report: Any, error: Exception | None) -> None:
        if error is not None:
            self._set_merge_note(f"Merge failed: {error}", error=True)
            return
        self._set_merge_note(report.message, error=not report.ok)
        if report.ok:
            self._set_review(report.detail)
            self._current_slug = ""
            self._current_file = ""
            self._editor.delete("1.0", "end")
            self._file_list.delete(0, "end")
            self._workspace_label.configure(text="Merged. Create a new workspace for the next change.")
        self.refresh()
        # A merged workspace is gone, so Merge must go back to disabled rather than
        # stay lit for a selection that no longer exists.
        self._refresh_merge_state(keep_note=True)

    # -- helpers ----------------------------------------------------------

    def _run_background(self, message: str, job: Callable[[], Any], finish: Callable[[Any, Exception | None], None]) -> None:
        """Run ``job`` on a worker thread, then hand its result back on the UI thread."""
        self._busy = True
        self._set_merge_note(message)
        self._set_buttons_enabled(False)
        self._spinner.pack(side="right", padx=8)
        self._spinner.start(12)

        def run() -> None:
            try:
                result = job()
                self._results.put((finish, result, None))
            except Exception as error:  # noqa: BLE001 - shown, never crashes Tk
                self._results.put((finish, None, error))

        threading.Thread(target=run, daemon=True).start()
        self._schedule_pump()

    def _schedule_pump(self) -> None:
        """Drain finished jobs on the UI thread, then look again shortly."""
        self._pump_results()
        if not self._closed:
            self.window.after(60, self._schedule_pump)

    def _pump_results(self) -> None:
        while True:
            try:
                finish, result, error = self._results.get_nowait()
            except queue.Empty:
                return
            self._busy = False
            self._spinner.stop()
            self._spinner.pack_forget()
            self._set_buttons_enabled(True)
            finish(result, error)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self._ask_button.configure(state="normal" if enabled else "disabled")
        self._checks_button.configure(state="normal" if enabled else "disabled")
        if not enabled:
            self._merge_button.configure(state="disabled")
        else:
            self._refresh_merge_state()

    def _set_trace(self, text: str) -> None:
        self._trace.configure(state="normal")
        self._trace.delete("1.0", "end")
        self._trace.insert("1.0", text)
        self._trace.configure(state="disabled")

    def _set_review(self, text: str) -> None:
        self._review.configure(state="normal")
        self._review.delete("1.0", "end")
        self._review.insert("1.0", text)
        self._review.configure(state="disabled")

    def _set_merge_note(self, message: str, *, error: bool = False) -> None:
        self._merge_note.configure(
            text=message, fg=self.theme.danger if error else self.theme.text_secondary
        )

    def _close(self) -> None:
        self._closed = True
        if self._on_close is not None:
            self._on_close()
        self.window.destroy()


__all__ = ["DEFAULT_MODEL_NOTE", "HONESTY_LINE", "SUBTITLE", "TITLE", "CodeStudioDialog"]
