"""The Code Studio window: a complete, local IDE for the project's own code.

This is the project's "code host and IDE in a window". It borrows the model a
developer already uses on a code host -- a branch per change, reviewed and applied
when it is good -- and implements it with the project's own ``git``, and it puts
the rest of a developer's day beside it, so the whole workflow happens here with
no cloud service:

* **A workspace is created for you.** Opening the Studio opens a workspace (a
  ``git worktree`` under ``data/studio/workspaces/<slug>`` on its own
  ``studio/<slug>`` branch, cut from ``main``), named automatically in order:
  ``PR1``, ``PR2``, ... Everything you and the coding model change happens there,
  so the running checkout is never touched and a broken change cannot stop the
  program.
* **You just describe the change in the chat.** Type what you want changed in the
  project's own code in the box below and press *Send*. *New change* opens the
  next workspace.
* **The coding model edits it, live.** A local coding model reads and writes files
  *inside the workspace only*, through a strict JSON protocol. The request appears
  as the user's own message, and every step the model takes is shown in the
  transcript as it happens -- first by first, so its work is reviewable.
* **A developer's whole toolbox is on the left.** The tool column groups:
  *Workspaces* (every change, its branch, its state), *Explorer* (the workspace's
  file tree, with new file / new folder / rename / delete), *Editor* (open a file,
  save it, read its outline), *Find & Replace* (search the project, and a
  project-wide literal replace with a preview), *Source Control* (git status,
  stage/unstage, commit, discard, stash, branches, history, diff, blame, and —
  only when you approve it — fetch/pull/push), *Tasks* (run the project's own
  tests, lint, type check, evaluation gate, build, or a command of your own, with
  live output), *Notes* (a TODO/FIXME scan), and *Review and apply* (the merge
  gate).
* **Test, then ask to apply.** When the model finishes, the project's own test
  command runs *inside the workspace*. Only when it passes does the window ask for
  your approval to apply the change to the real project; a failing or unapproved
  change never reaches it.

Everything runs **locally**: the model is local, git is local, the tasks are the
project's own commands, and the one network action (a remote fetch/pull/push) is
gated behind an explicit confirmation. Nothing here calls a cloud service.

The window is a thin view over :class:`gui.code_studio.CodeStudioService`, so all
of the logic is testable with no display. Tk is imported lazily with the rest of
the window, so the CLI never touches it.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import font as tkfont
from typing import Any, Literal

from core.answer_provenance import AnswerProvenance
from core.coder_model import DEFAULT_CODER_MODEL, PULL_HINT
from core.studio_tasks import CommandSpec

from gui import clipboard, rich_text
from gui import markdown as md
from gui.app import RADIUS_LARGE, RoundedBox
from gui.chat_composer import ChatComposer
from gui.dialog_base import ThemedDialog
from gui.message_footer import add_provenance_label
from gui.sidebar import SidebarColumn
from gui.typography import resolve

#: The Studio reads as a centred conversation, capped like the chat window's
#: transcript so long lines stay readable on a wide screen.
_COLUMN_MAX = 820

#: A message taller than this scrolls inside its own card rather than the column.
_MAX_BUBBLE_LINES = 40

#: The width of the left tool column.
_SIDEBAR_WIDTH = 300

#: A sentinel queued when the model has finished and the checks are about to run,
#: so the live bubble can say so. It is identity-compared, never a real step.
CHECKS_MARKER = object()

TITLE = "Code Studio"
SUBTITLE = (
    "Change the project's own code safely: type what you want in the chat below, and the "
    "coding model makes the change in an isolated workspace (a git worktree on its own "
    "branch). Every step is shown live. When the change passes the tests, you are asked to "
    "approve applying it to the project — the running program is never edited directly, so a "
    "half-finished change can never break it."
)
HONESTY_LINE = (
    "Type the change you want in the box below and press Send. The coding model's edits are "
    "real file writes, but they happen in an isolated workspace and nothing reaches the "
    "project until the checks pass and you approve applying the change. The panels on the "
    "left are the rest of a developer's day: browse and edit files, search the whole project, "
    "review git, and run the project's own commands — all inside this workspace."
)

#: The default coding model, shown in the status line so the choice is explicit.
DEFAULT_MODEL_NOTE = f"Coding model: qwen2.5-coder:7b (Ollama). If it is missing: {PULL_HINT}"


class CodeStudioDialog(ThemedDialog):
    """A complete local IDE for editing, testing, and applying project changes."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        service: Any,
        *,
        theme: Any,
        font: str,
        mono: str,
        overrides: object | None = None,
        list_overrides: object | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self.service = service
        self.theme = theme
        self.font = font
        self.mono = mono
        self._type = resolve(parent, ui_family=font, mono_family=mono, overrides=overrides)
        self._overrides = overrides
        self._list_overrides = list_overrides
        self._list_type = resolve(
            parent, ui_family=font, mono_family=mono, overrides=list_overrides
        )
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
        # Live agent steps, pushed from the worker thread and drained on the UI
        # thread (Tk is not thread-safe), so the model's work appears as it happens.
        self._steps: queue.Queue[Any] = queue.Queue()
        # Live task output lines, drained on the UI thread into the task's own
        # transcript row (Tk is not thread-safe, so the worker only queues).
        self._task_lines: queue.Queue[str] = queue.Queue()
        self._live: tk.Text | None = None
        self._task_running = False
        # The workspace a finished run left ready to apply, so the approval applies
        # the change the user was just shown rather than whatever is selected now.
        self._pending_apply: str = ""
        # True while the automatic first workspace is being opened, so its message
        # does not overwrite one the user is reading.
        self._auto_open = False
        # True while the running job's own result is the message to keep, so the
        # button refresh does not replace it.
        self._quiet_job = False
        # The slug being applied, so the result message names it after the run.
        self._applying_slug = ""
        # The catalog tasks, in menu order, so a selection maps back to a spec.
        self._task_specs: list[CommandSpec] = []

        self._request_var = tk.StringVar(value="")
        self._find_var = tk.StringVar(value="")
        self._replace_var = tk.StringVar(value="")
        self._scope_var = tk.StringVar(value="")
        self._commit_var = tk.StringVar(value="")
        self._branch_var = tk.StringVar(value="")
        self._custom_task_var = tk.StringVar(value="")

        self.window = tk.Toplevel(parent)
        self.window.title(TITLE)
        self.window.configure(bg=theme.base)
        self.window.geometry("1280x720")
        self.window.minsize(1000, 640)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._themed_build()

    def _after_build(self) -> None:
        """Run the window's post-build work (first build, and after a re-theme)."""
        self.refresh()
        self._ensure_workspace()

    # -- layout -----------------------------------------------------------

    def _build(self) -> None:
        """Assemble the window the way the chat window is assembled.

        A slim header, a left tool column, and a centred conversation column with
        its composer pinned to the bottom -- the same shape and the same theme
        tokens as the main chat window, so the Studio reads as part of the program
        rather than a separate tool.
        """
        theme = self.theme
        # Set the first time the model's work is shown; the tests read them after.
        self._trace: Any = None
        self._review: Any = None
        self._output: Any = None
        self._row_frames: list[tk.Frame] = []

        self._build_header()

        body = tk.Frame(self.window, bg=theme.base)
        body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=0, minsize=_SIDEBAR_WIDTH)
        body.columnconfigure(1, weight=1)

        self._build_sidebar(body)
        self._build_conversation(body)

    def _build_header(self) -> None:
        """The shared title band (the one :class:`gui.components.DialogHeader`)."""
        from gui.components import DialogHeader

        header = DialogHeader(
            self.window, title=TITLE, subtitle=SUBTITLE, theme=self.theme,
            typography=self._type, on_close=self._close,
        )
        self._spinner = header.spinner
        self._window_controls = header.controls

    # -- sidebar ----------------------------------------------------------

    def _build_sidebar(self, parent: tk.Frame) -> None:
        # A stack of collapsible sections (the shared :class:`~gui.sidebar.SidebarColumn`),
        # so the Studio's left column reads like a graphics program's tool panel
        # rather than one long flat list.
        column = SidebarColumn(
            parent,
            theme=self.theme,
            typography=self._type,
            width=_SIDEBAR_WIDTH,
            scope="code-studio",
            state_root=getattr(self.service, "data_root", None),
        )
        self._sidebar_column = column
        self._sidebar_canvas = column.canvas
        self._sidebar_scrollbar = column.scrollbar
        self._sidebar = column.body

        brand = tk.Frame(column.body, bg=self.theme.surface)
        brand.pack(fill="x", padx=14, pady=(14, 4))
        dot = tk.Canvas(brand, width=24, height=24, bg=self.theme.surface, highlightthickness=0)
        dot.create_oval(1, 1, 23, 23, fill=self.theme.primary, outline="")
        dot.create_text(12, 13, text="ف", fill=self.theme.primary_foreground, font=self._type.font("sm", weight="bold"))
        dot.pack(side="left")
        tk.Label(
            brand, text=TITLE, bg=self.theme.surface, fg=self.theme.contrast,
            font=self._type.font("heading3", weight="bold"), anchor="w",
        ).pack(side="left", padx=(8, 0))

        self._status_line = tk.Label(
            column.body, text=DEFAULT_MODEL_NOTE, bg=self.theme.surface, fg=self.theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=264,
        )
        self._status_line.pack(fill="x", padx=14, pady=(0, 2))

        self._project_line = tk.Label(
            column.body, text="", bg=self.theme.surface, fg=self.theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=264,
        )
        self._project_line.pack(fill="x", padx=14, pady=(0, 8))

        self._build_workspaces_section(column)
        self._build_explorer_section(column)
        self._build_editor_section(column)
        self._build_find_section(column)
        self._build_source_control_section(column)
        self._build_tasks_section(column)
        self._build_notes_section(column)
        self._build_review_section(column)

    # -- workspaces -------------------------------------------------------

    def _build_workspaces_section(self, column: SidebarColumn) -> None:
        sidebar = column.add("workspaces", "Workspaces", icon="🗂")
        self._workspace_list = tk.Listbox(
            sidebar, bg=self.theme.base, fg=self.theme.foreground, selectbackground=self.theme.primary,
            selectforeground=self.theme.primary_foreground, relief="flat", highlightthickness=0,
            font=self._list_type.font("xs"), height=5, activestyle="none",
        )
        self._workspace_list.pack(fill="x", padx=14, pady=(4, 6))
        self._workspace_list.bind("<<ListboxSelect>>", self._on_workspace_select)

        tk.Label(
            sidebar, text="Workspaces are named for you, in order (PR1, PR2, …).",
            bg=self.theme.surface, fg=self.theme.text_secondary, font=self._list_type.font("2xs"),
            anchor="w", justify="left", wraplength=264,
        ).pack(fill="x", padx=14, pady=(0, 4))
        row = tk.Frame(sidebar, bg=self.theme.surface)
        row.pack(fill="x", padx=14, pady=(0, 6))
        self._button(row, "New change", self._on_new_change).pack(side="left")
        self._button(row, "Refresh", self.refresh).pack(side="left", padx=(6, 0))
        self._button(row, "Discard", self._on_discard).pack(side="left", padx=(6, 0))

        self._workspace_label = tk.Label(
            sidebar, text="No workspace selected.", bg=self.theme.surface, fg=self.theme.text_secondary,
            font=self._type.font("xs", weight="bold"), anchor="w", justify="left", wraplength=264,
        )
        self._workspace_label.pack(fill="x", padx=14, pady=(0, 10))

    # -- explorer ---------------------------------------------------------

    def _build_explorer_section(self, column: SidebarColumn) -> None:
        explorer = column.add("explorer", "Explorer", icon="📁")
        self._file_list = tk.Listbox(
            explorer, bg=self.theme.base, fg=self.theme.foreground, selectbackground=self.theme.primary,
            selectforeground=self.theme.primary_foreground, relief="flat", highlightthickness=0,
            font=self._list_type.mono("2xs"), height=10, activestyle="none",
        )
        self._file_list.pack(fill="both", expand=True, padx=14, pady=(4, 4))
        self._file_list.bind("<<ListboxSelect>>", self._on_file_select)

        row = tk.Frame(explorer, bg=self.theme.surface)
        row.pack(fill="x", padx=14, pady=(0, 4))
        self._button(row, "New file", self._on_new_file).pack(side="left")
        self._button(row, "New folder", self._on_new_folder).pack(side="left", padx=(6, 0))
        row2 = tk.Frame(explorer, bg=self.theme.surface)
        row2.pack(fill="x", padx=14, pady=(0, 10))
        self._button(row2, "Rename", self._on_rename_file).pack(side="left")
        self._button(row2, "Copy", self._on_copy_file).pack(side="left", padx=(6, 0))
        self._button(row2, "Delete", self._on_delete_file).pack(side="left", padx=(6, 0))

    # -- editor -----------------------------------------------------------

    def _build_editor_section(self, column: SidebarColumn) -> None:
        editor = column.add("editor", "Editor", icon="✏")
        self._file_label = tk.Label(
            editor, text="Select a file to open it.", bg=self.theme.surface, fg=self.theme.text_secondary,
            font=self._type.mono("2xs"), anchor="w", justify="left", wraplength=264,
        )
        self._file_label.pack(fill="x", padx=14, pady=(4, 4))
        row = tk.Frame(editor, bg=self.theme.surface)
        row.pack(fill="x", padx=14, pady=(0, 4))
        self._button(row, "Save file", self._on_save_file).pack(side="left")
        self._button(row, "Close", self._hide_editor).pack(side="left", padx=(6, 0))
        row2 = tk.Frame(editor, bg=self.theme.surface)
        row2.pack(fill="x", padx=14, pady=(0, 4))
        self._button(row2, "Outline", self._on_outline).pack(side="left")
        self._button(row2, "Blame", self._on_blame).pack(side="left", padx=(6, 0))
        self._button(row2, "Discard file", self._on_discard_file).pack(side="left", padx=(6, 0))
        tk.Label(
            editor, text="Editing writes into the workspace only, never the running project.",
            bg=self.theme.surface, fg=self.theme.text_dim, font=self._type.font("2xs"),
            anchor="w", justify="left", wraplength=264,
        ).pack(fill="x", padx=14, pady=(0, 10))

    # -- find & replace ---------------------------------------------------

    def _build_find_section(self, column: SidebarColumn) -> None:
        find = column.add("find", "Find & Replace", icon="🔍", open=False)
        tk.Label(
            find, text="Find", bg=self.theme.surface, fg=self.theme.text_secondary,
            font=self._type.font("2xs"),
        ).pack(anchor="w", padx=14, pady=(4, 0))
        self._find_entry = tk.Entry(
            find, textvariable=self._find_var, bg=self.theme.base, fg=self.theme.foreground,
            relief="flat", highlightthickness=1, highlightbackground=self.theme.border_subtle,
            font=self._type.font("xs"),
        )
        self._find_entry.pack(fill="x", padx=14, pady=(2, 4))
        self._find_entry.bind("<Return>", lambda _e: self._on_search())

        tk.Label(
            find, text="Limit to one file or folder (optional)", bg=self.theme.surface,
            fg=self.theme.text_secondary, font=self._type.font("2xs"),
        ).pack(anchor="w", padx=14)
        self._scope_entry = tk.Entry(
            find, textvariable=self._scope_var, bg=self.theme.base, fg=self.theme.foreground,
            relief="flat", highlightthickness=1, highlightbackground=self.theme.border_subtle,
            font=self._type.mono("2xs"),
        )
        self._scope_entry.pack(fill="x", padx=14, pady=(2, 4))

        row = tk.Frame(find, bg=self.theme.surface)
        row.pack(fill="x", padx=14, pady=(0, 6))
        self._button(row, "Search", self._on_search).pack(side="left")

        tk.Label(
            find, text="Replace (literal, in the whole project)", bg=self.theme.surface,
            fg=self.theme.text_secondary, font=self._type.font("2xs"),
        ).pack(anchor="w", padx=14)
        self._replace_entry = tk.Entry(
            find, textvariable=self._replace_var, bg=self.theme.base, fg=self.theme.foreground,
            relief="flat", highlightthickness=1, highlightbackground=self.theme.border_subtle,
            font=self._type.font("xs"),
        )
        self._replace_entry.pack(fill="x", padx=14, pady=(2, 4))
        row2 = tk.Frame(find, bg=self.theme.surface)
        row2.pack(fill="x", padx=14, pady=(0, 10))
        self._button(row2, "Preview", self._on_replace_preview).pack(side="left")
        self._button(row2, "Replace all", self._on_replace_apply).pack(side="left", padx=(6, 0))

    # -- source control ---------------------------------------------------

    def _build_source_control_section(self, column: SidebarColumn) -> None:
        scm = column.add("scm", "Source Control", icon="⎇", open=False)
        self._git_label = tk.Label(
            scm, text="", bg=self.theme.surface, fg=self.theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=264,
        )
        self._git_label.pack(fill="x", padx=14, pady=(4, 2))
        self._git_status_list = tk.Listbox(
            scm, bg=self.theme.base, fg=self.theme.foreground, selectbackground=self.theme.primary,
            selectforeground=self.theme.primary_foreground, relief="flat", highlightthickness=0,
            font=self._list_type.mono("2xs"), height=6, activestyle="none", selectmode="extended",
        )
        self._git_status_list.pack(fill="x", padx=14, pady=(2, 4))
        row = tk.Frame(scm, bg=self.theme.surface)
        row.pack(fill="x", padx=14, pady=(0, 4))
        self._button(row, "Stage", self._on_stage).pack(side="left")
        self._button(row, "Unstage", self._on_unstage).pack(side="left", padx=(6, 0))
        self._button(row, "Discard", self._on_discard_changes).pack(side="left", padx=(6, 0))

        self._commit_entry = tk.Entry(
            scm, textvariable=self._commit_var, bg=self.theme.base, fg=self.theme.foreground,
            relief="flat", highlightthickness=1, highlightbackground=self.theme.border_subtle,
            font=self._type.font("xs"),
        )
        self._commit_entry.pack(fill="x", padx=14, pady=(4, 4))
        row2 = tk.Frame(scm, bg=self.theme.surface)
        row2.pack(fill="x", padx=14, pady=(0, 4))
        self._button(row2, "Commit", self._on_commit).pack(side="left")
        self._button(row2, "Stash", self._on_stash).pack(side="left", padx=(6, 0))
        self._button(row2, "Pop stash", self._on_stash_pop).pack(side="left", padx=(6, 0))

        self._branch_entry = tk.Entry(
            scm, textvariable=self._branch_var, bg=self.theme.base, fg=self.theme.foreground,
            relief="flat", highlightthickness=1, highlightbackground=self.theme.border_subtle,
            font=self._type.font("xs"),
        )
        self._branch_entry.pack(fill="x", padx=14, pady=(4, 4))
        row3 = tk.Frame(scm, bg=self.theme.surface)
        row3.pack(fill="x", padx=14, pady=(0, 4))
        self._button(row3, "New branch", self._on_new_branch).pack(side="left")
        self._button(row3, "Switch", self._on_switch_branch).pack(side="left", padx=(6, 0))

        row4 = tk.Frame(scm, bg=self.theme.surface)
        row4.pack(fill="x", padx=14, pady=(0, 4))
        self._button(row4, "History", self._on_history).pack(side="left")
        self._button(row4, "Branches", self._on_branches).pack(side="left", padx=(6, 0))
        self._button(row4, "Diff", self._on_show_diff).pack(side="left", padx=(6, 0))

        tk.Label(
            scm, text="Remote (leaves the machine — asks first)", bg=self.theme.surface,
            fg=self.theme.text_dim, font=self._type.font("2xs"),
        ).pack(anchor="w", padx=14, pady=(6, 2))
        row5 = tk.Frame(scm, bg=self.theme.surface)
        row5.pack(fill="x", padx=14, pady=(0, 10))
        self._button(row5, "Fetch", self._on_fetch).pack(side="left")
        self._button(row5, "Pull", self._on_pull).pack(side="left", padx=(6, 0))
        self._button(row5, "Push", self._on_push).pack(side="left", padx=(6, 0))

    # -- tasks ------------------------------------------------------------

    def _build_tasks_section(self, column: SidebarColumn) -> None:
        tasks = column.add("tasks", "Tasks", icon="▶", open=False)
        self._task_list = tk.Listbox(
            tasks, bg=self.theme.base, fg=self.theme.foreground, selectbackground=self.theme.primary,
            selectforeground=self.theme.primary_foreground, relief="flat", highlightthickness=0,
            font=self._list_type.font("2xs"), height=8, activestyle="none",
        )
        self._task_list.pack(fill="x", padx=14, pady=(4, 4))
        row = tk.Frame(tasks, bg=self.theme.surface)
        row.pack(fill="x", padx=14, pady=(0, 4))
        self._button(row, "Run", self._on_run_task).pack(side="left")
        self._button(row, "Stop", self._on_stop_task).pack(side="left", padx=(6, 0))
        self._button(row, "Clear output", self._on_clear_output).pack(side="left", padx=(6, 0))

        tk.Label(
            tasks, text="A command of your own (run in the workspace)", bg=self.theme.surface,
            fg=self.theme.text_secondary, font=self._type.font("2xs"),
        ).pack(anchor="w", padx=14)
        self._custom_task_entry = tk.Entry(
            tasks, textvariable=self._custom_task_var, bg=self.theme.base, fg=self.theme.foreground,
            relief="flat", highlightthickness=1, highlightbackground=self.theme.border_subtle,
            font=self._type.mono("2xs"),
        )
        self._custom_task_entry.pack(fill="x", padx=14, pady=(2, 4))
        self._custom_task_entry.bind("<Return>", lambda _e: self._on_run_custom_task())
        self._button(tasks, "Run custom command", self._on_run_custom_task).pack(anchor="w", padx=14, pady=(0, 10))

    # -- notes ------------------------------------------------------------

    def _build_notes_section(self, column: SidebarColumn) -> None:
        notes = column.add("notes", "Notes (TODO scan)", icon="🗒", open=False)
        self._notes_label = tk.Label(
            notes, text="Scan the workspace for TODO/FIXME/BUG/XXX/HACK notes.",
            bg=self.theme.surface, fg=self.theme.text_secondary, font=self._type.font("2xs"),
            anchor="w", justify="left", wraplength=264,
        )
        self._notes_label.pack(fill="x", padx=14, pady=(4, 4))
        self._button(notes, "Scan for notes", self._on_scan_notes).pack(anchor="w", padx=14, pady=(0, 10))

    # -- review and apply -------------------------------------------------

    def _build_review_section(self, column: SidebarColumn) -> None:
        review = column.add("review", "Review and apply", icon="✔")
        gate = tk.Frame(review, bg=self.theme.surface)
        gate.pack(fill="x", padx=14, pady=(4, 4))
        self._merge_button = self._button(gate, "Apply to the project", self._on_merge)
        self._merge_button.pack(side="left")
        self._merge_button.configure(state="disabled")
        row = tk.Frame(review, bg=self.theme.surface)
        row.pack(fill="x", padx=14, pady=(0, 4))
        self._button(row, "Run checks", self._on_run_checks).pack(side="left")
        self._button(row, "Export patch", self._on_export_patch).pack(side="left", padx=(6, 0))
        self._checks_label = tk.Label(
            review, text="Checks have not run yet.", bg=self.theme.surface, fg=self.theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=264,
        )
        self._checks_label.pack(fill="x", padx=14)
        self._merge_note = tk.Label(
            review, text="", bg=self.theme.surface, fg=self.theme.text_secondary, font=self._type.font("2xs"),
            anchor="w", justify="left", wraplength=264,
        )
        self._merge_note.pack(fill="x", padx=14, pady=(4, 14))

    # -- conversation -----------------------------------------------------

    def _build_conversation(self, parent: tk.Frame) -> None:
        theme = self.theme
        col = tk.Frame(parent, bg=theme.base)
        col.grid(row=0, column=1, sticky="nsew")

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

        self._build_composer(col)
        self._build_editor_panel(col)

        self._add_message("note", HONESTY_LINE)

    def _build_composer(self, parent: tk.Frame) -> None:
        """The shared rounded composer card, pinned to the bottom like the chat window."""
        theme = self.theme
        composer = ChatComposer(
            parent, theme=theme, typography=self._type,
            placeholder="Describe the code changes you want to make…",
            on_submit=self._on_ask_model, outer_bg=theme.base, with_attach=False,
            variable=self._request_var,
        )
        composer.wrap.pack(side="bottom", fill="x", padx=self._column_pad(), pady=(8, 12))
        self._composer = composer
        # The Studio's own actions sit on the composer's action toolbar, beside the
        # shared Send disc, so the panel is one consistent composer everywhere.
        self._ask_button = self._button(composer.right_tools, "Send", self._on_ask_model)
        composer.add_tool(self._ask_button)
        self._apply_button = self._button(composer.right_tools, "Apply to the project", self._on_merge)
        composer.add_tool(self._apply_button)
        self._apply_button.configure(state="disabled")
        self._checks_button = self._button(composer.right_tools, "Run checks", self._on_run_checks)
        composer.add_tool(self._checks_button)
        composer.add_tool(self._button(composer.right_tools, "Show diff", self._on_show_diff))

    def _build_editor_panel(self, parent: tk.Frame) -> None:
        """The file editor, revealed above the composer when a file is opened."""
        theme = self.theme
        panel = tk.Frame(parent, bg=theme.surface)
        head = tk.Frame(panel, bg=theme.surface)
        head.pack(fill="x", padx=12, pady=(10, 4))
        self._editor_file_label = tk.Label(
            head, text="Select a file to open it.", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.mono("code"), anchor="w",
        )
        self._editor_file_label.pack(side="left")
        self._button(head, "Close", self._hide_editor).pack(side="right")
        self._button(head, "Save file", self._on_save_file).pack(side="right", padx=(0, 6))

        self._editor = tk.Text(
            panel, wrap="none", undo=True, bg=theme.base, fg=theme.foreground,
            insertbackground=theme.primary, relief="flat", highlightthickness=0,
            font=self._type.mono("code"), padx=8, pady=8, height=14,
        )
        self._editor.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self._editor.bind("<<Modified>>", self._on_editor_modified)
        self._editor_panel = panel

    def _show_editor(self) -> None:
        if not self._editor_panel.winfo_ismapped():
            self._editor_panel.pack(side="bottom", fill="x", padx=self._column_pad(), pady=(0, 6))

    def _hide_editor(self) -> None:
        self._editor_panel.pack_forget()

    # -- transcript -------------------------------------------------------

    def _column_pad(self) -> int:
        """Side padding that centres the transcript in a readable column."""
        width = self._canvas.winfo_width() if hasattr(self, "_canvas") else 0
        if width <= 1:
            return 16
        return max(16, (width - _COLUMN_MAX) // 2)

    def _available_column(self) -> int:
        width = self._canvas.winfo_width()
        if width <= 1:
            return _COLUMN_MAX
        return max(160, min(_COLUMN_MAX, width - 2 * self._column_pad()))

    def _bubble_fonts(self) -> tuple[tkfont.Font, tkfont.Font]:
        return tkfont.Font(font=self._type.font("body")), tkfont.Font(font=self._type.mono("code"))

    def _bubble_width(self, text: str, *, role: str = "model") -> int:
        normal, mono = self._bubble_fonts()
        widest = 0
        for line in md.iter_styled_lines(text or " "):
            font = mono if line.in_code else normal
            widest = max(widest, font.measure(line.text))
        if role == "user":
            return max(80, min(widest, int(_COLUMN_MAX * 0.84) - 52))
        return max(160, min(widest, self._available_column()))

    def _add_message(
        self, role: str, text: str, *, title: str = "", mono_body: bool = False,
        rich: bool = True, provenance: str = "",
    ) -> tk.Text:
        """Add one message row in the chat window's style and return its widget."""
        theme = self.theme
        pad = self._column_pad()
        row = tk.Frame(self._transcript, bg=theme.base)
        row.pack(fill="x", padx=(pad, pad), pady=6)
        self._row_frames.append(row)

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
            card.configure(width=self._bubble_width(text, role="user"))
        else:
            head = tk.Frame(row, bg=theme.base)
            head.pack(fill="x", pady=(0, 4))
            if role == "model":
                self._avatar(head)
                padx = (6, 0)
            else:
                padx = (0, 0)
            tk.Label(
                head, text=title or ("Coding model" if role == "model" else "Status"),
                bg=theme.base, fg=theme.text_dim, font=self._type.font("xs", weight="bold"), anchor="w",
            ).pack(side="left", padx=padx)
            widget = tk.Text(
                row, height=1, wrap="word", relief="flat", borderwidth=0, highlightthickness=0,
                bg=theme.base, fg=theme.foreground,
                font=self._type.mono("code") if mono_body else self._type.font("body"),
                padx=0, pady=0, cursor="xterm", takefocus=1,
            )
            rich_text.apply_body_spacing(widget, self._type)
            if rich and not mono_body:
                self._insert_rich(widget, text)
            else:
                widget.insert("1.0", text)
            widget.pack(anchor="w")
            widget.configure(width=self._bubble_width(text))

        self._make_read_only(widget)
        for wheel in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(wheel, self._on_bubble_wheel, add="+")
        self._fit_height(widget)
        if provenance.strip():
            add_provenance_label(row, provenance, theme=theme, typography=self._type)
        self._scroll_to_bottom()
        return widget

    def _avatar(self, parent: tk.Misc) -> None:
        badge = tk.Canvas(parent, width=24, height=24, bg=self.theme.base, highlightthickness=0)
        badge.create_oval(1, 1, 23, 23, fill=self.theme.primary, outline="")
        badge.create_text(
            12, 13, text="ف", fill=self.theme.primary_foreground, font=self._type.font("sm", weight="bold")
        )
        badge.pack(side="left")

    def _fit_height(self, widget: tk.Text) -> None:
        widget.update_idletasks()
        measured = widget.count("1.0", "end", "displaylines")
        lines = int(measured[0]) if measured else 1
        widget.configure(height=min(max(lines, 1), _MAX_BUBBLE_LINES))

    @staticmethod
    def _make_read_only(widget: tk.Text) -> None:
        """Read-only but still selectable and copyable, like a chat message."""
        clipboard.make_read_only(widget)

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
        for row in self._row_frames:
            row.pack_configure(padx=(pad, pad))

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

    # -- markdown rendering ----------------------------------------------

    def _insert_rich(self, widget: tk.Text, text: str) -> None:
        """Render Markdown into a bubble (the window's shared renderer)."""
        rich_text.insert_markdown(widget, text, theme=self.theme, typography=self._type)

    def _button(self, parent: tk.Misc, text: str, command: Any) -> tk.Button:
        """A themed flat button (the shared :func:`gui.components.button`)."""
        from gui.components import button

        return button(
            parent, text, command, theme=self.theme, typography=self._type,
            variant="secondary", small=False, anchor="center",
        )

    # -- state ------------------------------------------------------------

    def refresh(self) -> None:
        """Reload the workspace list, the model status line, and the task menu."""
        state = self.service.state()
        self._status_line.configure(text=state.coder.summary() if state.coder else "")
        project = getattr(state, "project", None)
        kind = getattr(project, "kind", "") if project else ""
        branch = getattr(state, "current_branch", "")
        dirty = " · uncommitted changes" if getattr(state, "dirty", False) else ""
        if kind:
            self._project_line.configure(
                text=f"Project: {kind} · on `{branch}`{dirty}"
                if state.repository
                else f"Project: {kind} · not a git checkout"
            )
        else:
            self._project_line.configure(text="")
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
        self._refresh_task_menu()

    def _selected_slug(self) -> str:
        selection = self._workspace_list.curselection()
        if not selection:
            return ""
        return self._workspaces[selection[0]].slug

    # -- workspaces -------------------------------------------------------

    def _on_workspace_select(self, _event: tk.Event | None = None, *, keep_note: bool = False) -> None:
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
        self._refresh_source_control()
        self._refresh_merge_state(keep_note=keep_note)
        self._set_trace(workspace.task or "")

    def _ensure_workspace(self) -> None:
        """Open a workspace automatically, so the composer is ready to use.

        The Studio only edits the project's own code, so a fresh window opens a
        workspace (``PR1``, the next number) rather than making the user press a
        button first. When the checkout is not a git repository, or a workspace is
        already open, this does nothing.
        """
        if self._current_slug:
            return
        state = self.service.state()
        if not state.repository:
            return
        existing = state.workspaces
        if existing:
            self._select_workspace(existing[0].slug, keep_note=self._auto_open)
            return
        self._auto_open = True
        self._run_background(
            "Opening the first workspace…",
            lambda: self.service.create_workspace(""),
            self._after_create,
            note=False,
        )

    def _select_workspace(self, slug: str, *, keep_note: bool = False) -> None:
        """Select ``slug`` in the list and load it, as a click would."""
        for index, workspace in enumerate(self._workspaces):
            if workspace.slug == slug:
                self._workspace_list.selection_clear(0, "end")
                self._workspace_list.selection_set(index)
                self._workspace_list.see(index)
                self._on_workspace_select(keep_note=keep_note)
                return

    def _on_new_change(self) -> None:
        """Start a fresh workspace for the next change (named automatically)."""
        if self._busy:
            return
        self._run_background(
            "Starting a new change…",
            lambda: self.service.create_workspace(""),
            self._after_create,
        )

    def _after_create(self, report: Any, error: Exception | None) -> None:
        auto = self._auto_open
        self._auto_open = False
        if error is not None:
            self._set_merge_note(f"Could not create the workspace: {error}", error=True)
            return
        if not report.ok:
            self._set_merge_note(report.message, error=True)
            return
        self.refresh()
        # The new workspace is the newest, so it is first in the list.
        if self._workspaces:
            self._select_workspace(self._workspaces[0].slug, keep_note=auto)
        if not auto:
            self._set_merge_note(f"{report.message} Describe the change in the chat below.")

    def _on_discard(self) -> None:
        slug = self._current_slug or self._selected_slug()
        if not slug:
            self._set_merge_note("Select a change to discard.", error=True)
            return
        report = self.service.remove_workspace(slug)
        self._current_slug = ""
        self._current_file = ""
        self._pending_apply = ""
        self._editor.delete("1.0", "end")
        self._file_list.delete(0, "end")
        self._set_review("")
        self.refresh()
        self._set_merge_note(report.message, error=not report.ok)
        # Keep a workspace open, so the next request can be sent right away. Its own
        # message stays on screen, so this step is silent.
        self._auto_open = True
        self._ensure_workspace()

    # -- explorer ---------------------------------------------------------

    def _refresh_files(self) -> None:
        self._file_list.delete(0, "end")
        if not self._current_slug:
            return
        try:
            entries = self.service.list_files(self._current_slug)
        except Exception as error:
            self._set_merge_note(f"Could not list files: {error}", error=True)
            return
        self._files = entries
        for entry in entries:
            self._file_list.insert("end", entry.path)

    def _selected_file(self) -> str:
        selection = self._file_list.curselection()
        if not selection:
            return self._current_file
        return self._files[selection[0]].path

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
        self._editor_file_label.configure(text=relative)
        self._file_label.configure(text=relative)
        self._show_editor()
        self._editor.delete("1.0", "end")
        self._editor.insert("1.0", report.detail)
        self._editor.edit_modified(False)
        self._dirty = False

    def _on_new_file(self) -> None:
        slug = self._current_slug
        if not slug:
            self._set_merge_note("Open a workspace first.", error=True)
            return
        relative = _ask_string(self.window, "New file", "Path inside the workspace (e.g. core/new.py):")
        if not relative:
            return
        report = self.service.create_file(slug, relative, "")
        self._set_merge_note(report.message, error=not report.ok)
        self._refresh_files()

    def _on_new_folder(self) -> None:
        slug = self._current_slug
        if not slug:
            self._set_merge_note("Open a workspace first.", error=True)
            return
        relative = _ask_string(self.window, "New folder", "Path inside the workspace (e.g. core/sub):")
        if not relative:
            return
        report = self.service.create_directory(slug, relative)
        self._set_merge_note(report.message, error=not report.ok)
        self._refresh_files()

    def _on_rename_file(self) -> None:
        slug = self._current_slug
        current = self._selected_file()
        if not slug or not current:
            self._set_merge_note("Select a file to rename.", error=True)
            return
        new_relative = _ask_string(self.window, "Rename", "New path:", initial=current)
        if not new_relative or new_relative == current:
            return
        report = self.service.rename_file(slug, current, new_relative)
        self._set_merge_note(report.message, error=not report.ok)
        self._refresh_files()

    def _on_copy_file(self) -> None:
        slug = self._current_slug
        current = self._selected_file()
        if not slug or not current:
            self._set_merge_note("Select a file to copy.", error=True)
            return
        new_relative = _ask_string(self.window, "Copy", "New path:", initial=current)
        if not new_relative or new_relative == current:
            return
        report = self.service.copy_file(slug, current, new_relative)
        self._set_merge_note(report.message, error=not report.ok)
        self._refresh_files()

    def _on_delete_file(self) -> None:
        slug = self._current_slug
        current = self._selected_file()
        if not slug or not current:
            self._set_merge_note("Select a file to delete.", error=True)
            return
        if not _confirm(self.window, "Delete", f"Delete {current} from the workspace?"):
            return
        report = self.service.delete_file(slug, current)
        self._set_merge_note(report.message, error=not report.ok)
        if report.ok and current == self._current_file:
            self._current_file = ""
            self._hide_editor()
        self._refresh_files()

    # -- editor -----------------------------------------------------------

    def _on_editor_modified(self, _event: tk.Event | None = None) -> None:
        if self._editor.edit_modified():
            self._dirty = True
            self._editor_file_label.configure(text=f"{self._current_file}  (unsaved)")
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
            self._editor_file_label.configure(text=self._current_file)
            self._file_label.configure(text=self._current_file)
            self._refresh_merge_state(keep_note=True)

    def _on_outline(self) -> None:
        if not self._current_slug or not self._current_file:
            self._set_merge_note("Open a file first.", error=True)
            return
        report = self.service.outline(self._current_slug, self._current_file)
        self._set_review(report.describe())

    def _on_blame(self) -> None:
        if not self._current_slug or not self._current_file:
            self._set_merge_note("Open a file first.", error=True)
            return
        report = self.service.git_blame(self._current_slug, self._current_file)
        self._set_review(report.detail if report.ok else report.message)

    def _on_discard_file(self) -> None:
        slug = self._current_slug
        current = self._selected_file()
        if not slug or not current:
            self._set_merge_note("Select a file to discard.", error=True)
            return
        if not _confirm(self.window, "Discard", f"Discard uncommitted changes in {current}?"):
            return
        report = self.service.discard_changes(slug, current)
        self._set_merge_note(report.message, error=not report.ok)
        self._refresh_files()
        self._refresh_source_control()

    # -- find & replace ---------------------------------------------------

    def _on_search(self) -> None:
        slug = self._current_slug
        query = self._find_var.get().strip()
        if not slug:
            self._set_merge_note("Open a workspace first.", error=True)
            return
        if not query:
            self._set_merge_note("Type something to search for.", error=True)
            return
        report = self.service.search(slug, query, relative=self._scope_var.get().strip())
        self._set_review(report.detail or report.message)
        self._set_merge_note(report.message, error=not report.ok)

    def _on_replace_preview(self) -> None:
        self._run_replace(apply=False)

    def _on_replace_apply(self) -> None:
        old = self._find_var.get().strip()
        if not _confirm(self.window, "Replace all", f"Replace every occurrence of {old!r} in the project?"):
            return
        self._run_replace(apply=True)

    def _run_replace(self, *, apply: bool) -> None:
        slug = self._current_slug
        old = self._find_var.get().strip()
        new = self._replace_var.get()
        if not slug:
            self._set_merge_note("Open a workspace first.", error=True)
            return
        if not old:
            self._set_merge_note("Type the text to replace.", error=True)
            return
        report = self.service.replace_across(
            slug, old, new, relative=self._scope_var.get().strip(), apply=apply
        )
        detail = "\n".join([report.message, *report.summary_lines()])
        self._set_review(detail)
        self._set_merge_note(report.message, error=not report.ok)
        if report.ok and report.applied:
            self._refresh_files()

    # -- source control ---------------------------------------------------

    def _refresh_source_control(self) -> None:
        """Redraw the git status list and the branch line for the current workspace."""
        if not hasattr(self, "_git_status_list"):
            return
        self._git_status_list.delete(0, "end")
        if not self._current_slug:
            self._git_label.configure(text="")
            return
        entries = self.service.git_status(self._current_slug)
        self._git_entries = entries
        for entry in entries:
            self._git_status_list.insert("end", entry.describe())
        branch = ""
        try:
            branch = self.service.workspace(self._current_slug).branch
        except Exception:
            branch = ""
        clean = "clean" if not entries else f"{len(entries)} change(s)"
        self._git_label.configure(text=f"On `{branch}` · {clean}")
        # Prefill the commit box with a sensible default the user can edit.
        if not self._commit_var.get().strip():
            task = ""
            try:
                task = self.service.workspace(self._current_slug).task
            except Exception:
                task = ""
            self._commit_var.set(f"Studio: {task}" if task else f"Studio: {self._current_slug}")

    def _selected_status_paths(self) -> tuple[str, ...]:
        selection = self._git_status_list.curselection()
        entries = getattr(self, "_git_entries", ())
        return tuple(entries[i].path for i in selection if i < len(entries))

    def _on_stage(self) -> None:
        slug = self._current_slug
        if not slug:
            self._set_merge_note("Open a workspace first.", error=True)
            return
        paths = self._selected_status_paths()
        report = self.service.stage(slug, *paths)
        self._set_merge_note(report.message, error=not report.ok)
        self._refresh_source_control()

    def _on_unstage(self) -> None:
        slug = self._current_slug
        if not slug:
            self._set_merge_note("Open a workspace first.", error=True)
            return
        paths = self._selected_status_paths()
        report = self.service.unstage(slug, *paths)
        self._set_merge_note(report.message, error=not report.ok)
        self._refresh_source_control()

    def _on_discard_changes(self) -> None:
        slug = self._current_slug
        paths = self._selected_status_paths()
        if not slug or not paths:
            self._set_merge_note("Select file(s) to discard.", error=True)
            return
        if not _confirm(self.window, "Discard", f"Discard changes in {len(paths)} file(s)?"):
            return
        report = self.service.discard_changes(slug, *paths)
        self._set_merge_note(report.message, error=not report.ok)
        self._refresh_files()
        self._refresh_source_control()

    def _on_commit(self) -> None:
        slug = self._current_slug
        message = self._commit_var.get().strip()
        if not slug:
            self._set_merge_note("Open a workspace first.", error=True)
            return
        if not message:
            self._set_merge_note("Write a commit message.", error=True)
            return
        report = self.service.commit(slug, message)
        self._set_merge_note(report.message, error=not report.ok)
        if report.ok:
            self._commit_var.set("")
            self._refresh_source_control()
            self._refresh_merge_state(keep_note=True)

    def _on_stash(self) -> None:
        if not self._current_slug:
            self._set_merge_note("Open a workspace first.", error=True)
            return
        report = self.service.stash(self._current_slug, self._commit_var.get().strip(), include_untracked=True)
        self._set_merge_note(report.message, error=not report.ok)
        self._refresh_files()
        self._refresh_source_control()

    def _on_stash_pop(self) -> None:
        if not self._current_slug:
            self._set_merge_note("Open a workspace first.", error=True)
            return
        report = self.service.stash_pop(self._current_slug)
        self._set_merge_note(report.message, error=not report.ok)
        self._refresh_files()
        self._refresh_source_control()

    def _on_new_branch(self) -> None:
        slug = self._current_slug
        name = self._branch_var.get().strip()
        if not slug:
            self._set_merge_note("Open a workspace first.", error=True)
            return
        if not name:
            self._set_merge_note("Type a branch name.", error=True)
            return
        report = self.service.create_branch(slug, name)
        self._set_merge_note(report.message, error=not report.ok)
        if report.ok:
            self._branch_var.set("")
            self._refresh_source_control()

    def _on_switch_branch(self) -> None:
        slug = self._current_slug
        name = self._branch_var.get().strip()
        if not slug or not name:
            self._set_merge_note("Type a branch name to switch to.", error=True)
            return
        report = self.service.switch_branch(slug, name)
        self._set_merge_note(report.message, error=not report.ok)
        self._refresh_source_control()

    def _on_history(self) -> None:
        if not self._current_slug:
            self._set_merge_note("Open a workspace first.", error=True)
            return
        commits = self.service.git_log(self._current_slug, limit=40)
        lines = [commit.describe() for commit in commits] or ["No commits yet."]
        self._set_review("\n".join(lines))

    def _on_branches(self) -> None:
        if not self._current_slug:
            self._set_merge_note("Open a workspace first.", error=True)
            return
        branches = self.service.git_branches(self._current_slug)
        lines = [branch.describe() for branch in branches] or ["No branches."]
        self._set_review("\n".join(lines))

    def _on_export_patch(self) -> None:
        if not self._current_slug:
            self._set_merge_note("Open a workspace first.", error=True)
            return
        report = self.service.export_patch(self._current_slug)
        if not report.ok:
            self._set_merge_note(report.message, error=True)
            return
        _ask_save(self.window, report.detail, default_name=f"{self._current_slug}.patch")
        self._set_merge_note("Patch ready (use Save in the dialog to keep it).")

    def _remote_action(self, action: str) -> None:
        slug = self._current_slug
        if not slug:
            self._set_merge_note("Open a workspace first.", error=True)
            return
        if not _confirm(
            self.window, f"{action.title()} remote",
            f"`git {action}` contacts the network. Run it now?",
        ):
            return
        report = self.service.remote_action(slug, action)
        self._set_merge_note(report.message, error=not report.ok)
        self._refresh_source_control()

    def _on_fetch(self) -> None:
        self._remote_action("fetch")

    def _on_pull(self) -> None:
        self._remote_action("pull")

    def _on_push(self) -> None:
        self._remote_action("push")

    # -- tasks ------------------------------------------------------------

    def _refresh_task_menu(self) -> None:
        """Fill the task list from the catalog that fits this project."""
        if not hasattr(self, "_task_list"):
            return
        self._task_specs = list(self.service.command_catalog())
        self._task_list.delete(0, "end")
        for spec in self._task_specs:
            self._task_list.insert("end", f"[{spec.category}] {spec.title}")
        # Run the tests first, so the most common action is the pre-selected one.
        for index, spec in enumerate(self._task_specs):
            if spec.key == "tests":
                self._task_list.selection_set(index)
                break

    def _selected_task(self) -> CommandSpec | None:
        selection = self._task_list.curselection()
        if not selection:
            return self._task_specs[0] if self._task_specs else None
        return self._task_specs[selection[0]]

    def _on_run_task(self) -> None:
        slug = self._current_slug
        spec = self._selected_task()
        if not slug or spec is None:
            self._set_merge_note("Open a workspace first.", error=True)
            return
        self._start_task(slug, spec.key, spec.title, "")

    def _on_run_custom_task(self) -> None:
        slug = self._current_slug
        command_line = self._custom_task_var.get().strip()
        if not slug:
            self._set_merge_note("Open a workspace first.", error=True)
            return
        if not command_line:
            self._set_merge_note("Type a command to run.", error=True)
            return
        self._start_task(slug, "custom", command_line, command_line)

    def _start_task(self, slug: str, key: str, title: str, command_line: str) -> None:
        if self._task_running or self._busy:
            self._set_merge_note("A task is already running.", error=True)
            return
        self._task_running = True
        self._set_merge_note(f"Running {title}…")
        # The task's output is a transcript message like every other row, so it is
        # selectable, copyable, and scrolls with the rest of the conversation.
        self._output = self._add_message("note", "", title=f"Task: {title}", mono_body=True)
        self._set_buttons_enabled(False)

        def on_line(line: str) -> None:
            self._task_lines.put(line)

        def job() -> Any:
            return self.service.run_command(slug, key, on_line=on_line, command_line=command_line)

        self._run_background(f"Running {title}…", job, self._after_task)

    def _on_stop_task(self) -> None:
        if not self._task_running:
            return
        self.service.cancel_command()
        self._set_merge_note("Stopping the task…")

    def _after_task(self, outcome: Any, error: Exception | None) -> None:
        self._task_running = False
        self._drain_task_lines()
        if error is not None:
            self._set_merge_note(f"The task could not run: {error}", error=True)
            return
        self._set_merge_note(outcome.summary(), error=not outcome.ok)
        if self._output is not None:
            if not self._output.get("1.0", "end-1c").strip() and outcome.output:
                self._output.insert("1.0", outcome.output)
            self._output.insert("end", "\n" + outcome.summary())
            self._fit_height(self._output)
        self._refresh_source_control()
        self._refresh_merge_state(keep_note=True)

    def _on_clear_output(self) -> None:
        """Forget the last task's output row (it stays in the transcript above)."""
        self._output = None
        self._set_merge_note("The task output is cleared; scroll the transcript to review it.")

    def _drain_task_lines(self) -> None:
        appended = False
        while True:
            try:
                line = self._task_lines.get_nowait()
            except queue.Empty:
                break
            if self._output is not None:
                self._output.insert("end", "\n" + line)
                appended = True
        if appended and self._output is not None:
            self._fit_height(self._output)
            self._scroll_to_bottom()

    # -- notes ------------------------------------------------------------

    def _on_scan_notes(self) -> None:
        if not self._current_slug:
            self._set_merge_note("Open a workspace first.", error=True)
            return
        scan = self.service.scan_notes(self._current_slug)
        self._notes_label.configure(text=scan.describe().splitlines()[0] if scan.notes else "No notes found.")
        self._set_review(scan.describe())

    # -- model ------------------------------------------------------------

    def _on_ask_model(self) -> None:
        request = self._request_var.get().strip()
        if not request:
            self._set_merge_note("Type what you want changed in the chat below.", error=True)
            return
        if self._busy:
            return
        if not self._current_slug:
            # The window opens a workspace on start; if it is missing (a discarded
            # one, or a race), open one before sending so the request is not lost.
            self._set_merge_note("Opening a workspace…")
            self._run_background(
                "Opening a workspace…",
                lambda: self.service.create_workspace(""),
                self._after_create,
            )
            return
        # Show the request as the user's own message, then clear the composer, so
        # the conversation reads like a chat: question first, then the model's work.
        self._add_message("user", request)
        self._request_var.set("")
        self._pending_apply = ""
        self._apply_button.configure(state="disabled")
        # A live bubble the model's steps are appended to as they happen.
        self._live = self._add_message("model", "", title="Coding model", mono_body=True)
        self._run_background(
            "The coding model is working…",
            lambda: self.service.run_change(
                self._current_slug,
                request,
                on_step=self._on_agent_step,
                on_checks=self._on_checks_starting,
            ),
            self._after_ask,
        )

    def _on_agent_step(self, step: Any) -> None:
        """Queue one live step; the UI thread renders it (never touch Tk here)."""
        self._steps.put(step)

    def _on_checks_starting(self) -> None:
        """Called on the worker thread before the checks run; queue a marker."""
        self._steps.put(CHECKS_MARKER)

    def _drain_steps(self) -> None:
        """Append any queued agent steps to the live bubble."""
        appended = False
        while True:
            try:
                step = self._steps.get_nowait()
            except queue.Empty:
                break
            if step is CHECKS_MARKER:
                lines = ["— The model is done; running the project's checks…"]
            else:
                lines = step.live_lines()
            if not lines:
                continue
            if self._live is not None:
                text = self._live.get("1.0", "end-1c")
                prefix = "\n" if text.strip() else ""
                self._live.insert("end", prefix + "\n".join(lines))
                appended = True
        if appended and self._live is not None:
            self._fit_height(self._live)
            self._scroll_to_bottom()

    def _after_ask(self, result: Any, error: Exception | None) -> None:
        self._drain_steps()
        self._live = None
        if error is not None:
            self._set_trace(f"The coding model failed: {error}")
            return
        change = result
        trace = change.trace
        if trace is not None:
            self._set_trace(trace.render())
        else:
            self._set_trace(change.report.message)
        if change.checks is not None:
            self._set_review(change.checks.detail or change.checks.message)
        self._set_merge_note(change.report.message, error=not change.report.ok)
        self.refresh()
        self._refresh_files()
        self._refresh_source_control()
        self._refresh_merge_state(keep_note=True)
        self._offer_apply(change)

    def _offer_apply(self, change: Any) -> None:
        """Ask for the user's approval when a change is ready to apply.

        This is the gate the user asked for: the model's edits live in an isolated
        workspace, and nothing reaches the real project until the checks pass *and*
        the user explicitly approves applying the change.
        """
        if not change.report.ok:
            # The run failed (an unreachable model, a refused edit); its own message
            # is the honest one, so it is left in place.
            self._pending_apply = ""
            self._apply_button.configure(state="disabled")
            return
        if not change.changed:
            self._pending_apply = ""
            self._apply_button.configure(state="disabled")
            self._add_message(
                "note", "The model made no changes to the project.", title="No change"
            )
            self._set_merge_note("The model made no changes to the project.", error=True)
            return
        if change.ready_to_apply:
            self._pending_apply = self._current_slug
            self._apply_button.configure(state="normal")
            self._add_message(
                "note",
                f"The change passed the checks ({', '.join(change.changed)}). "
                "Press **Apply to the project** to put it into فِيض الرُّوح, or "
                "*Discard change* to throw it away.",
                title="Ready to apply",
            )
            self._set_merge_note(
                "The change passed the checks — approve applying it to the project.",
                error=False,
            )
        else:
            self._pending_apply = ""
            self._apply_button.configure(state="disabled")
            failed = change.checks is not None and not change.checks.ok
            reason = (
                "The checks failed, so the change is not applied. Fix it and send again."
                if failed
                else "The change is not ready to apply yet."
            )
            self._add_message("note", reason, title="Not applied")
            self._set_merge_note(reason, error=True)

    # -- diff and checks --------------------------------------------------

    def _on_show_diff(self) -> None:
        if not self._current_slug:
            self._set_merge_note("Select a workspace first.", error=True)
            return
        report = self.service.diff(self._current_slug)
        self._set_review(report.detail if report.ok else report.message)

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
        """Enable **Apply to the project** only when the gate allows it.

        The gate is the checks passing and the base branch not having moved on
        (``can_merge``), so a change can never reach the real project on a stale or
        failing run. The button itself is the user's approval: nothing is applied
        until it is pressed.

        ``keep_note`` leaves the current message alone, so a caller that has just
        reported something to the user (a save, a model run) does not have its
        confirmation replaced by the merge hint.
        """
        if not self._current_slug:
            self._merge_button.configure(state="disabled")
            self._apply_button.configure(state="disabled")
            if not keep_note:
                self._merge_note.configure(text="")
            return
        report = self.service.can_merge(self._current_slug)
        state: Literal["normal", "disabled"] = "normal" if report.ok else "disabled"
        self._merge_button.configure(state=state)
        self._apply_button.configure(state=state)
        if not keep_note:
            self._merge_note.configure(
                text=report.message, fg=self.theme.text_secondary if report.ok else self.theme.text_dim
            )

    def _on_merge(self) -> None:
        if not self._current_slug or self._busy:
            return
        slug = self._current_slug
        self._applying_slug = slug
        self._run_background(
            f"Applying `{slug}` to the project…",
            lambda: self.service.merge(slug),
            self._after_merge,
        )

    def _after_merge(self, report: Any, error: Exception | None) -> None:
        slug = self._applying_slug
        self._applying_slug = ""
        if error is not None:
            self._set_merge_note(f"Apply failed: {error}", error=True)
            return
        if report.ok:
            self._set_merge_note(f"Applied `{slug}` to the project.", error=False)
            self._set_review(report.detail)
            self._add_message(
                "note", "The change is now part of فِيض الرُّوح.", title="Applied"
            )
            self._current_slug = ""
            self._current_file = ""
            self._editor.delete("1.0", "end")
            self._file_list.delete(0, "end")
            self._workspace_label.configure(
                text="Applied. Start a new change for the next request."
            )
        else:
            self._set_merge_note(report.message, error=True)
        self.refresh()
        # An applied workspace is gone, so Apply must go back to disabled rather
        # than stay lit for a selection that no longer exists.
        self._refresh_merge_state(keep_note=True)
        if report.ok:
            # Keep a workspace open for the next request, without replacing the
            # "Applied" confirmation the user is still reading.
            self._auto_open = True
            self._ensure_workspace()

    # -- helpers ----------------------------------------------------------

    def _run_background(
        self,
        message: str,
        job: Callable[[], Any],
        finish: Callable[[Any, Exception | None], None],
        *,
        note: bool = True,
    ) -> None:
        """Run ``job`` on a worker thread, then hand its result back on the UI thread.

        ``note=False`` leaves the status line alone, so an automatic step (opening a
        workspace after a merge or a discard) does not replace the message the user
        is still reading.
        """
        self._busy = True
        self._quiet_job = not note
        if note:
            self._set_merge_note(message)
        self._set_buttons_enabled(False)
        self._spinner.pack(side="right", padx=8)
        self._spinner.start(12)

        def run() -> None:
            try:
                result = job()
                self._results.put((finish, result, None))
            except Exception as error:
                self._results.put((finish, None, error))

        threading.Thread(target=run, daemon=True).start()
        self._schedule_pump()

    def _schedule_pump(self) -> None:
        """Drain finished jobs on the UI thread, then look again shortly."""
        self._pump_results()
        if not self._closed:
            self.window.after(60, self._schedule_pump)

    def _pump_results(self) -> None:
        # Render the model's live steps and the task's live output first, so the
        # window keeps pace with the worker even before the run finishes.
        self._drain_steps()
        self._drain_task_lines()
        while True:
            try:
                finish, result, error = self._results.get_nowait()
            except queue.Empty:
                return
            self._busy = False
            quiet = self._quiet_job
            self._quiet_job = False
            self._spinner.stop()
            self._spinner.pack_forget()
            self._set_buttons_enabled(True, keep_note=quiet)
            finish(result, error)

    def _set_buttons_enabled(self, enabled: bool, *, keep_note: bool = False) -> None:
        self._ask_button.configure(state="normal" if enabled else "disabled")
        self._checks_button.configure(state="normal" if enabled else "disabled")
        if not enabled:
            self._merge_button.configure(state="disabled")
            self._apply_button.configure(state="disabled")
        else:
            self._refresh_merge_state(keep_note=keep_note)

    def _set_trace(self, text: str) -> None:
        """Show the model's work as a transcript message, not a bare panel."""
        if not text:
            return
        self._trace = self._add_message(
            "model", text, title="Coding model", mono_body=True,
            provenance=self._coding_model_line(),
        )

    def _coding_model_line(self) -> str:
        """Which coding model did the work, and where it ran.

        The Studio's service reports its backend honestly (the in-process GGUF,
        else a reachable Ollama, else neither), so the line names the model the
        run really used and its runtime -- never a model that was not loaded.
        """
        status = self.service.status()
        available = bool(getattr(status, "available", False))
        model = getattr(status, "model", "") or DEFAULT_CODER_MODEL
        runtime = getattr(status, "backend", "") or "in-process"
        label = f"Code Studio: {model} ({runtime})." if available else ""
        return AnswerProvenance(
            model_label=label,
            mouth="code",
            model_key=model,
            retrieval_only=not available,
        ).describe()

    def _set_review(self, text: str) -> None:
        """Show a diff (or a checks result) as a monospaced transcript message."""
        if not text:
            return
        self._review = self._add_message("note", text, title="Diff and checks", mono_body=True)

    def _set_merge_note(self, message: str, *, error: bool = False) -> None:
        self._merge_note.configure(
            text=message, fg=self.theme.danger if error else self.theme.text_secondary
        )

    def _close(self) -> None:
        self._closed = True
        if self._on_close is not None:
            self._on_close()
        self.window.destroy()


# -- small modal helpers -----------------------------------------------------


def _ask_string(parent: tk.Misc, title: str, prompt: str, *, initial: str = "") -> str:
    """A one-line modal prompt, or an empty string when cancelled.

    The project avoids heavy modal dialogs where a small prompt will do; this is
    the one place the Studio asks for a name, so every "new file / rename / copy"
    path reads the same way.
    """
    from tkinter import simpledialog

    answer = simpledialog.askstring(title, prompt, initialvalue=initial, parent=parent)
    return (answer or "").strip()


def _confirm(parent: tk.Misc, title: str, message: str) -> bool:
    """A yes/no confirmation, used before anything destructive or networked."""
    from tkinter import messagebox

    return bool(messagebox.askyesno(title, message, parent=parent))


def _ask_save(parent: tk.Misc, text: str, *, default_name: str) -> None:
    """Let the user save a piece of text (a patch, a report) to a file of their choosing."""
    from tkinter import filedialog

    path = filedialog.asksaveasfilename(
        parent=parent, title="Save", initialfile=default_name,
        defaultextension=".patch", filetypes=[("Patch", "*.patch"), ("All files", "*.*")],
    )
    if not path:
        return
    from pathlib import Path

    Path(path).write_text(text, encoding="utf-8")


__all__ = [
    "CHECKS_MARKER",
    "DEFAULT_MODEL_NOTE",
    "HONESTY_LINE",
    "SUBTITLE",
    "TITLE",
    "CodeStudioDialog",
]
