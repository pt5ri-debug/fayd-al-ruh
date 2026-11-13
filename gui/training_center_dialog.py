"""The Training Center window: every learning and training workflow in one place.

Reached from the chat composer's **Training** control (beside **Code Studio**) and
from the sidebar, this window is the one professional surface for the project's
learning: it shows what the project has learned, gathers more, builds governed
datasets, runs the controlled training pipeline, evaluates it, and keeps the whole
history.

It is built the way every other window in the project is built: the same
:class:`~gui.app.Theme` tokens, the shared :class:`~gui.sidebar.SidebarColumn` on
the left, the shared :class:`~gui.chat_composer.ChatComposer` at the bottom, the
three :class:`~gui.window_controls` title-bar buttons, and a worker thread whose
result is handed back through a queue the UI thread drains (Tk is not
thread-safe).

The left sidebar is a stack of collapsible sections, one per area of the task,
each drawn from the **catalogue** (:mod:`core.training_center.catalog`) so a
button's label, hint, and tooltip are the single source of truth and can never
drift from the CLI or the docs:

* Dashboard, Learning Sources, Knowledge Acquisition, Question-and-Answer
  Learning, Dataset Studio, Training Jobs, Model Fine-Tuning, Pretraining &
  Continued Training, Evaluation & Benchmarking, Knowledge Base Management,
  Learning Pipelines, Training History, Resource Monitor, Settings.

Everything is honest: a governed action asks for confirmation first, a capability
the project does not have says so with the real next step, a long run happens off
the interface thread, and no figure is invented.
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from collections.abc import Callable
from tkinter import filedialog, messagebox, simpledialog
from typing import Any

from core.best_effort import best_effort
from core.training_center import catalog
from core.training_center.catalog import capability

from gui.app import RADIUS_LARGE, RoundedBox
from gui.chat_composer import ChatComposer, Tooltip
from gui.choice_dialog import ask_choice
from gui.dialog_base import ThemedDialog
from gui.sidebar import SidebarColumn
from gui.training_center import TrainingCenterService
from gui.typography import resolve
from gui.window_controls import fit_geometry

TITLE = "Training Center"
SUBTITLE = (
    "Learn from sources, build governed datasets, train and evaluate the model, "
    "and keep the whole history — all over the project's own machinery."
)
HONESTY_LINE = (
    "The Training Center composes the project's own learning machinery; it does "
    "not re-implement it and it never fabricates a figure. A governed action asks "
    "first, a capability the project does not have says so with the real next "
    "step, and every long run happens off this window's thread."
)

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 820
MIN_WIDTH = 980
MIN_HEIGHT = 640
SIDEBAR_WIDTH = 320
_COLUMN_MAX = 900


class TrainingCenterDialog(ThemedDialog):
    """The Training Center workspace."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        service: TrainingCenterService,
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
        self._on_close = on_close
        self._closed = False
        self._busy = False
        self._cards: list[tk.Text] = []
        self._results: queue.Queue[
            tuple[Callable[[Any, Exception | None], None], Any, Exception | None]
        ] = queue.Queue()
        self._pump_after: str | None = None
        self._command_var = tk.StringVar(value="")
        self._topic_var = tk.StringVar(value="")
        self._web_toggle: Any = None
        # Live training details stream into the chat area. The worker thread never
        # touches a widget: it puts each line on ``_lines`` and the UI thread's
        # pump drains it into one growing message, coalesced so a fast loop cannot
        # freeze the window (exactly the main window's pattern).
        self._lines: queue.Queue[str] = queue.Queue()
        self._pending: list[str] = []
        self._last_render = 0.0
        self._stream_body: tk.Text | None = None

        self.window = tk.Toplevel(parent)
        self.window.title(f"{TITLE} — Fayd Al-Ruh")
        self.window.configure(bg=theme.base)
        self.window.geometry(self._initial_geometry())
        self.window.minsize(MIN_WIDTH, MIN_HEIGHT)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)

        self._themed_build()

    def _after_build(self) -> None:
        self._refresh_overview()

    def _initial_geometry(self) -> str:
        return fit_geometry(self.window, DEFAULT_WIDTH, DEFAULT_HEIGHT, min_height=MIN_HEIGHT)

    # -- layout -----------------------------------------------------------

    def _build(self) -> None:
        self._build_header()
        theme = self.theme
        body = tk.Frame(self.window, bg=theme.base)
        body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=0, minsize=SIDEBAR_WIDTH)
        body.columnconfigure(1, weight=1)
        self._body = body
        self._build_sidebar(body)
        self._build_conversation(body)

    def _build_header(self) -> None:
        """The shared title band (the one :class:`gui.components.DialogHeader`).

        The busy indicator is a *text* label ("working…"), not a progressbar, so
        it is drawn through the header's ``extra`` slot and ``self._spinner`` still
        points at the label the rest of the window configures. The band also
        carries the **?** help control beside the trio, which opens this window's
        own bilingual guide (:mod:`gui.training_center_help`).
        """
        from gui.components import DialogHeader

        def _extra(inner: tk.Frame) -> None:
            self._spinner = tk.Label(
                inner, text="", bg=self.theme.titlebar, fg=self.theme.primary,
                font=self._type.font("2xs", weight="bold"),
            )
            self._spinner.pack(side="right", padx=(0, 12))

        header = DialogHeader(
            self.window, title=TITLE, subtitle=SUBTITLE, theme=self.theme,
            typography=self._type, on_close=self._close, on_help=self.open_help,
            show_spinner=False, extra=_extra,
        )
        self._window_controls = header.controls

    def open_help(self) -> None:
        """Open (or raise) this window's bilingual help window.

        One window is kept and reused, exactly like every other secondary window
        the program opens, so pressing **?** twice does not stack copies.
        """
        existing = getattr(self, "_help_window", None)
        if existing is not None:
            try:
                if existing.window.winfo_exists():
                    existing.window.lift()
                    existing.window.focus_set()
                    return
            except tk.TclError:
                pass
        from gui.training_center_help import TrainingCenterHelpDialog

        self._help_window: Any = TrainingCenterHelpDialog(
            self.window, theme=self.theme, font=self.font, mono=self.mono,
            overrides=self._overrides, list_overrides=self._list_overrides,
        )

    def _button(self, parent: tk.Misc, text: str, command: Any) -> tk.Button:
        """A themed flat button (the shared :func:`gui.components.button`)."""
        from gui.components import button

        return button(
            parent, text, command, theme=self.theme, typography=self._type,
            variant="secondary", small=True, anchor="w",
        )

    def _choose(
        self,
        title: str,
        prompt: str,
        *,
        values: Any = (),
        initial: str = "",
    ) -> str:
        """Ask for one value with an *editable dropdown* of the real choices.

        The dropdown counterpart of :func:`simpledialog.askstring`: the box lists
        the choices the service really accepts **and** still accepts a typed
        value, so a control asks "which one?" instead of "type the exact name".
        An empty return means the user cancelled.
        """
        return ask_choice(
            self.window, title, prompt, values=tuple(values), initial=initial,
            theme=self.theme, typography=self._type,
        )

    def _require_web(self, what: str) -> bool:
        """Make sure the internet switch is on for an action that needs it.

        When it is off, ask once -- so the option that needs the internet *turns
        it on* for the user -- and only proceed when the user agrees. When the
        user declines, nothing runs and the transcript says why.
        """
        if self.service.web_enabled():
            return True
        if messagebox.askyesno(
            "Turn the internet on?",
            f"{what} needs the internet, and it is off.\n\nTurn it on now? "
            "Turning it on approves cloud providers, URL imports, and the bulk "
            "knowledge-base build for this Training Center.",
            parent=self.window,
        ):
            self.service.set_web(True)
            self._refresh_web_toggle()
            self._add_message("ok", self.service.web_status())
            return True
        self._add_message(
            "error",
            f"{what} needs the internet, which is off. Turn it on in Settings, or "
            "with the Web switch beside the message box.",
        )
        return False

    def _hint(self, parent: tk.Misc, text: str) -> tk.Label:
        label = tk.Label(
            parent, text=text, bg=self.theme.surface, fg=self.theme.text_dim,
            font=self._type.font("2xs"), justify="left", anchor="w",
            wraplength=SIDEBAR_WIDTH - 48,
        )
        label.pack(fill="x", padx=(20, 12), pady=(0, 6))
        return label

    def _section_note(self, section: tk.Frame, key: str) -> None:
        english = catalog.SECTION_SUMMARIES.get(key, "")
        arabic = catalog.SECTION_TITLES_AR.get(key, "")
        if english:
            tk.Label(
                section, text=english, bg=self.theme.surface, fg=self.theme.text_secondary,
                font=self._type.font("2xs"), justify="left", anchor="w",
                wraplength=SIDEBAR_WIDTH - 40,
            ).pack(fill="x", padx=12, pady=(2, 2))
        if arabic:
            tk.Label(
                section, text=arabic, bg=self.theme.surface, fg=self.theme.text_dim,
                font=self._type.font("2xs"), justify="right", anchor="e",
                wraplength=SIDEBAR_WIDTH - 40,
            ).pack(fill="x", padx=12, pady=(0, 6))

    def _cap_button(self, section: tk.Frame, key: str, command: Any) -> tk.Button:
        """A button whose label, hint, and tooltip all come from the catalogue."""
        cap = capability(key)
        text = cap.title if cap is not None else key
        button = self._button(section, text, command)
        button.pack(fill="x", padx=12, pady=(4, 0))
        if cap is not None:
            hint = cap.summary.split(". ")[0]
            if not hint.endswith("."):
                hint += "."
            self._hint(section, hint)
            governed = "  (asks first)" if cap.governed else ""
            Tooltip(
                button,
                f"{cap.title}{governed}\n{cap.summary}\n\nReturns: {cap.outcome}\n{cap.cli}",
                self.theme,
                self._type,
            )
        return button

    # -- sidebar ----------------------------------------------------------

    def _build_sidebar(self, parent: tk.Frame) -> None:
        column = SidebarColumn(
            parent, theme=self.theme, typography=self._type, width=SIDEBAR_WIDTH,
            scope="training-center",
        )
        self._sidebar_column = column
        theme = self.theme

        brand = tk.Frame(column.body, bg=theme.surface)
        brand.pack(fill="x", padx=14, pady=(14, 4))
        dot = tk.Canvas(brand, width=24, height=24, bg=theme.surface, highlightthickness=0)
        dot.create_oval(1, 1, 23, 23, fill=theme.primary, outline="")
        dot.create_text(
            12, 13, text="ت", fill=theme.primary_foreground,
            font=self._type.font("sm", weight="bold"),
        )
        dot.pack(side="left")
        tk.Label(
            brand, text="Training Center", bg=theme.surface, fg=theme.contrast,
            font=self._type.font("body", weight="bold"),
        ).pack(side="left", padx=(8, 0))

        tk.Label(
            column.body,
            text=(
                "Every control is explained: a note under it says what it does, and "
                "hovering shows what it returns and the exact command it runs. The "
                "Settings section has \"Show every control\" to print the whole list."
            ),
            bg=theme.surface, fg=theme.text_dim, font=self._type.font("2xs"),
            justify="left", anchor="w", wraplength=SIDEBAR_WIDTH - 40,
        ).pack(fill="x", padx=14, pady=(0, 8))

        self._build_dashboard_section(column)
        self._build_sources_section(column)
        self._build_knowledge_section(column)
        self._build_qa_section(column)
        self._build_datasets_section(column)
        self._build_jobs_section(column)
        self._build_finetune_section(column)
        self._build_pretrain_section(column)
        self._build_language_model_section(column)
        self._build_evaluation_section(column)
        self._build_kb_section(column)
        self._build_pipelines_section(column)
        self._build_history_section(column)
        self._build_resources_section(column)
        self._build_settings_section(column)

    def _build_dashboard_section(self, column: SidebarColumn) -> None:
        section = column.add("dashboard", "Dashboard", icon="📊")
        self._section_note(section, "dashboard")
        self._cap_button(section, "dashboard.refresh", self._on_dashboard)
        self._cap_button(section, "dashboard.history", self._on_history)

    def _build_sources_section(self, column: SidebarColumn) -> None:
        section = column.add("sources", "Learning Sources", icon="🗂")
        self._section_note(section, "sources")
        self._cap_button(section, "sources.database", self._on_database)
        self._cap_button(section, "sources.internet", self._on_internet)
        self._cap_button(section, "sources.gemini", self._on_gemini)
        self._cap_button(section, "sources.local_model", self._on_local_model)
        self._cap_button(section, "sources.documents", self._on_documents)
        self._cap_button(section, "sources.books", self._on_books)
        self._cap_button(section, "sources.characters", self._on_characters)

    def _build_knowledge_section(self, column: SidebarColumn) -> None:
        section = column.add("knowledge", "Knowledge Acquisition", icon="📚")
        self._section_note(section, "knowledge")
        self._cap_button(section, "knowledge.ingest", self._on_knowledge_ingest)
        self._cap_button(section, "knowledge.search", self._on_knowledge_search)
        self._cap_button(section, "knowledge.reindex", self._on_reindex)
        self._cap_button(section, "knowledge.export", self._on_export_knowledge)

    def _build_qa_section(self, column: SidebarColumn) -> None:
        section = column.add("qa", "Question-and-Answer Learning", icon="❓")
        self._section_note(section, "qa")
        self._cap_button(section, "qa.ask", self._on_ask)
        self._cap_button(section, "qa.generate", self._on_generate)
        self._cap_button(section, "qa.review", self._on_review)
        self._cap_button(section, "qa.export", self._on_export_dataset)

    def _build_datasets_section(self, column: SidebarColumn) -> None:
        section = column.add("datasets", "Dataset Studio", icon="🧩")
        self._section_note(section, "datasets")
        self._cap_button(section, "datasets.list", self._on_datasets)
        self._cap_button(section, "datasets.build", self._on_build_dataset)
        self._cap_button(section, "datasets.validate", self._on_validate_dataset)
        self._cap_button(section, "datasets.export", self._on_export_dataset_file)

    def _build_jobs_section(self, column: SidebarColumn) -> None:
        section = column.add("jobs", "Training Jobs", icon="⚙")
        self._section_note(section, "jobs")
        self._cap_button(section, "jobs.list", self._on_jobs)
        self._cap_button(section, "jobs.feasibility", self._on_feasibility)
        self._cap_button(section, "jobs.start", self._on_start_job)
        self._cap_button(section, "jobs.cancel", self._on_cancel_job)
        self._cap_button(section, "jobs.resume", self._on_resume_job)
        self._cap_button(section, "jobs.retry", self._on_retry_job)

    def _build_finetune_section(self, column: SidebarColumn) -> None:
        section = column.add("finetune", "Model Fine-Tuning", icon="🎛")
        self._section_note(section, "finetune")
        self._cap_button(section, "finetune.compat", self._on_compat)
        self._cap_button(section, "finetune.presets", self._on_presets)
        self._cap_button(section, "finetune.adapters", self._on_adapters)

    def _build_pretrain_section(self, column: SidebarColumn) -> None:
        section = column.add("pretrain", "Pretraining & Continued Training", icon="🏗")
        self._section_note(section, "pretrain")
        self._cap_button(section, "pretrain.status", self._on_pretrain_status)
        self._cap_button(section, "pretrain.estimate", self._on_estimate)
        self._cap_button(section, "pretrain.continue", self._on_continue_training)

    def _build_language_model_section(self, column: SidebarColumn) -> None:
        section = column.add("language_model", "Language Model (Fayd Al-Ruh)", icon="🧠")
        self._section_note(section, "language_model")
        self._cap_button(section, "lm.status", self._on_lm_status)
        self._cap_button(section, "lm.corpus", self._on_lm_corpus)
        self._cap_button(section, "lm.settings.show", self._on_lm_settings)
        self._cap_button(section, "lm.settings.set", self._on_lm_set_setting)
        self._cap_button(section, "lm.train", self._on_lm_train)
        self._cap_button(section, "lm.train.fresh", self._on_lm_train_fresh)
        self._cap_button(section, "lm.train.sharded", self._on_lm_train_sharded)
        self._cap_button(section, "lm.release", self._on_lm_release)
        self._cap_button(section, "lm.evaluate", self._on_lm_evaluate)
        self._cap_button(section, "lm.generate", self._on_lm_generate)
        self._cap_button(section, "lm.weights.install", self._on_lm_install_pack)
        self._cap_button(section, "lm.weights.import", self._on_lm_import_weights)
        self._cap_button(section, "lm.weights.fetch", self._on_lm_fetch_weights)
        self._cap_button(section, "lm.weights.build", self._on_lm_build_learned)
        self._cap_button(section, "lm.weights.learned", self._on_lm_learned)
        self._cap_button(section, "lm.plan", self._on_lm_plan)

    def _build_evaluation_section(self, column: SidebarColumn) -> None:
        section = column.add("evaluation", "Evaluation & Benchmarking", icon="📈")
        self._section_note(section, "evaluation")
        self._cap_button(section, "evaluation.run", self._on_evaluate)
        self._cap_button(section, "evaluation.compare", self._on_compare)
        self._cap_button(section, "evaluation.gate", self._on_gate)

    def _build_kb_section(self, column: SidebarColumn) -> None:
        section = column.add("kb", "Knowledge Base Management", icon="🗄")
        self._section_note(section, "kb")
        self._cap_button(section, "kb.build", self._on_kb_build)
        self._cap_button(section, "kb.stats", self._on_kb_stats)
        self._cap_button(section, "kb.index", self._on_reindex)

    def _build_pipelines_section(self, column: SidebarColumn) -> None:
        section = column.add("pipelines", "Learning Pipelines", icon="🔗")
        self._section_note(section, "pipelines")
        self._cap_button(section, "pipelines.list", self._on_pipeline_list)
        self._cap_button(section, "pipelines.run", self._on_pipeline_run)

    def _build_history_section(self, column: SidebarColumn) -> None:
        section = column.add("history", "Training History", icon="🧾")
        self._section_note(section, "history")
        self._cap_button(section, "history.show", self._on_history)
        self._cap_button(section, "history.search", self._on_history_search)

    def _build_resources_section(self, column: SidebarColumn) -> None:
        section = column.add("resources", "Resource Monitor", icon="🖥")
        self._section_note(section, "resources")
        self._cap_button(section, "resources.read", self._on_resources)
        self._cap_button(section, "pretrain.estimate", self._on_estimate)

    def _build_settings_section(self, column: SidebarColumn) -> None:
        section = column.add("settings", "Settings", icon="⚙")
        self._section_note(section, "settings")
        self._cap_button(section, "settings.web", self._on_toggle_web)
        self._cap_button(section, "settings.show", self._on_settings)
        self._cap_button(section, "settings.set", self._on_set_setting)
        self._cap_button(section, "settings.catalog", self._on_catalog)

    # -- conversation column ----------------------------------------------

    def _build_conversation(self, parent: tk.Frame) -> None:
        theme = self.theme
        col = tk.Frame(parent, bg=theme.base)
        col.grid(row=0, column=1, sticky="nsew")

        self._build_composer(col)

        container = tk.Frame(col, bg=theme.base)
        container.pack(fill="both", expand=True)
        self._canvas = tk.Canvas(container, bg=theme.base, highlightthickness=0)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._scrollbar = tk.Scrollbar(container, command=self._canvas.yview)
        self._scrollbar.pack(side="right", fill="y")
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        self._transcript = tk.Frame(self._canvas, bg=theme.base)
        self._window = self._canvas.create_window((0, 0), window=self._transcript, anchor="nw")
        self._transcript.bind("<Configure>", self._on_transcript_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        for widget in (self._canvas, self._transcript):
            widget.bind("<MouseWheel>", self._on_mousewheel)
            widget.bind("<Button-4>", self._on_mousewheel)
            widget.bind("<Button-5>", self._on_mousewheel)
        self._add_message("note", HONESTY_LINE)

    def _build_composer(self, parent: tk.Frame) -> None:
        composer = ChatComposer(
            parent, theme=self.theme, typography=self._type,
            placeholder="Type a question to gather into the review queue, or a search query…",
            on_submit=self._on_command, outer_bg=self.theme.base, with_attach=False,
            variable=self._command_var,
        )
        composer.wrap.pack(side="bottom", fill="x", padx=self._column_pad(), pady=(8, 12))
        self._composer = composer
        self._command_entry = composer.entry
        self._command_button = self._button(composer.right_tools, "Ask", self._on_command)
        composer.add_tool(self._command_button)
        Tooltip(
            self._command_button,
            "Ask the selected provider the typed question and record the answer as a "
            "pending learning example for review.\nCLI: python -m cli training-center ask --question …",
            self.theme,
            self._type,
        )
        # The internet switch: a real control, beside the Ask button, that governs
        # every external path (cloud, URL import, KB build). It shows the state in
        # force, so it can never disagree with the session it belongs to.
        self._web_toggle = composer.add_web_toggle(
            self._on_web_toggled,
            initial=self.service.web_enabled(),
            available=True,
            tip=self.service.web_status(),
        )

    def _on_web_toggled(self, enabled: bool) -> None:
        now = self.service.set_web(enabled)
        self._refresh_web_toggle()
        self._add_message("ok" if now else "note", self.service.web_status())

    def _refresh_web_toggle(self) -> None:
        toggle = getattr(self, "_web_toggle", None)
        if toggle is None:
            return
        toggle.set_state(self.service.web_enabled(), available=True)

    # -- layout helpers ---------------------------------------------------

    def _column_pad(self) -> int:
        width = self.window.winfo_width() or DEFAULT_WIDTH
        return max(16, (width - SIDEBAR_WIDTH - _COLUMN_MAX) // 2)

    def _on_transcript_configure(self, _event: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        delta = -1 if getattr(event, "delta", 0) > 0 else 1
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        self._canvas.yview_scroll(delta, "units")

    # -- transcript -------------------------------------------------------

    def _add_message(self, role: str, text: str) -> tk.Text:
        theme = self.theme
        pad = self._column_pad()
        outer = tk.Frame(self._transcript, bg=theme.base)
        outer.pack(fill="x", padx=pad, pady=(4, 4))

        if role == "user":
            card = RoundedBox(
                outer, bg=theme.user_bubble, fill=theme.user_bubble,
                outline=theme.user_bubble, corner=RADIUS_LARGE, parent_bg=theme.base,
            )
            card.pack(side="right", anchor="e")
            body = tk.Text(
                card.inner, wrap="word", bg=theme.user_bubble, fg=theme.user_bubble_fg,
                relief="flat", highlightthickness=0, borderwidth=0,
                font=self._type.font("body"), padx=14, pady=8, height=1, width=0,
                cursor="arrow",
            )
            body.insert("1.0", text)
            body.pack(fill="both", expand=True)
        else:
            host = tk.Frame(outer, bg=theme.surface)
            host.pack(side="left", anchor="w", fill="x", expand=True)
            accent = {
                "note": theme.text_secondary,
                "assistant": theme.primary,
                "error": theme.danger,
                "ok": theme.success,
            }.get(role, theme.text_secondary)
            tk.Frame(host, bg=accent, width=3).pack(side="left", fill="y")
            body = tk.Text(
                host, wrap="word", bg=theme.surface, fg=theme.foreground, relief="flat",
                highlightthickness=0, borderwidth=0, font=self._type.font("body"),
                padx=12, pady=8, height=1, cursor="arrow",
            )
            body.insert("1.0", text)
            body.pack(side="left", fill="both", expand=True)

        body.configure(state="disabled")
        body.bind("<MouseWheel>", self._on_mousewheel)
        self._cards.append(body)
        self._fit_card(body)
        self._scroll_to_end()
        return body

    def _fit_card(self, widget: tk.Text) -> None:
        try:
            widget.update_idletasks()
            lines = int(widget.index("end-1c").split(".")[0])
        except tk.TclError:
            return
        widget.configure(height=max(1, min(lines, 40)))

    def _scroll_to_end(self) -> None:
        self._transcript.update_idletasks()
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._canvas.yview_moveto(1.0)

    # -- background work --------------------------------------------------

    def _start_work(
        self, job: Callable[[], Any], finish: Callable[[Any, Exception | None], None]
    ) -> None:
        if self._busy:
            self._add_message("note", "Another operation is still running; please wait.")
            return
        self._busy = True
        self._spinner.configure(text="working…")
        self._pending = []
        self._stream_body = None

        def run() -> None:
            try:
                result = job()
                self._results.put((finish, result, None))
            except Exception as error:  # never traceback into the window
                self._results.put((finish, None, error))

        threading.Thread(target=run, daemon=True).start()
        self._schedule_pump()

    def _schedule_pump(self) -> None:
        if self._closed:
            return
        self._pump_after = self.window.after(60, self._pump_results)

    def _pump_results(self) -> None:
        self._pump_after = None
        if self._closed:
            return
        self._drain_lines(cap=500)
        self._flush_stream()
        try:
            finish, result, error = self._results.get_nowait()
        except queue.Empty:
            if self._busy:
                self._schedule_pump()
            return
        self._busy = False
        self._drain_lines(cap=None)
        self._flush_stream(force=True)
        self._spinner.configure(text="")
        finish(result, error)

    # -- live streaming ---------------------------------------------------

    def _stream_line(self, text: str) -> None:
        """Queue one line of live detail from a worker thread (Tk-safe).

        A worker never touches a widget: it only puts the line here, and the UI
        thread's pump drains it into the growing message. This is the one entry a
        training callback needs, so a run's details reach the chat area live.
        """
        if text:
            self._lines.put(str(text))

    def _drain_lines(self, *, cap: int | None) -> None:
        drained = 0
        while cap is None or drained < cap:
            try:
                self._pending.append(self._lines.get_nowait())
                drained += 1
            except queue.Empty:
                break

    def _flush_stream(self, *, force: bool = False) -> None:
        """Append the queued lines to the streaming message, at a bounded rate."""
        if not self._pending:
            return
        now = time.monotonic()
        if not force and now - self._last_render < 0.15:
            return
        text = "\n".join(self._pending)
        self._pending.clear()
        self._last_render = now
        if self._stream_body is None:
            self._stream_body = self._add_message("note", text)
        else:
            try:
                self._stream_body.configure(state="normal")
                self._stream_body.insert("end", f"\n{text}")
                self._stream_body.configure(state="disabled")
            except tk.TclError:  # the window closed mid-run
                return
        self._fit_card(self._stream_body)
        self._scroll_to_end()

    def _report(self, role: str, title: str, lines: tuple[str, ...] = (), detail: str = "") -> None:
        body = title
        if lines:
            body += "\n" + "\n".join(f"• {line}" for line in lines)
        if detail:
            body += f"\n\n{detail}"
        self._add_message(role, body)

    def _run(self, title: str, job: Callable[[], str], *, ok_role: str = "assistant") -> None:
        """Run ``job`` off the thread and post its returned text as a message."""
        self._add_message("note", title)
        self._start_work(job, lambda result, error: self._post(result, error, ok_role))

    def _step_stream(
        self, run_id: str, step: int, loss: float, max_steps: int, *, started: float
    ) -> None:
        """Queue one live step line, throttled to roughly ten reports per run.

        The callback runs on the training thread, so it only queues a line -- the
        UI thread's pump is what draws it. Reporting the first step, the last, and
        about every tenth in between keeps a long run readable without flooding the
        chat area.
        """
        interval = max(1, max_steps // 10)
        if step != 1 and step != max_steps and step % interval:
            return
        elapsed = max(1e-6, time.monotonic() - started)
        rate = step / elapsed
        eta = (max_steps - step) / rate if rate > 0 else 0.0
        percent = (step / max_steps * 100) if max_steps else 0.0
        self._stream_line(
            f"[{run_id}] step {step}/{max_steps} ({percent:.0f}%) · "
            f"loss {loss:.4f} · {rate:.1f} steps/s · eta {eta:.0f}s"
        )

    def _start_streamed_job(
        self,
        model: Any,
        dataset_id: str,
        config: Any,
        *,
        resume_from: Any = None,
    ) -> Any:
        """Start a governed run and stream its live detail; block until it ends.

        Runs inside a ``_run`` job (already on a worker thread), so it starts the
        pipeline job, wires ``on_step`` to the chat area's live stream, and waits
        for the handle. Every training control goes through here, so the details
        always stream and never only appear at the end.
        """
        max_steps = int(getattr(config, "max_steps", 0) or 0)
        started = time.monotonic()
        holder: dict[str, str] = {"run_id": str(getattr(config, "run_id", "") or "")}

        def on_step(step: int, loss: float) -> None:
            self._step_stream(holder["run_id"] or "run", step, loss, max_steps, started=started)

        handle = self.service.start_job(
            model, dataset_id, config, resume_from=resume_from, on_step=on_step
        )
        holder["run_id"] = handle.run_id
        self._stream_line(f"[{handle.run_id}] training on {dataset_id!r} started.")
        handle.wait()
        return handle

    def _describe_handle(self, handle: Any) -> str:
        if handle.error is not None:
            return f"The run failed: {handle.error}"
        result = handle.result
        return result.describe() if result is not None else "The run produced no result."

    def _post(self, result: Any, error: Exception | None, ok_role: str) -> None:
        if error is not None:
            self._add_message("error", f"{type(error).__name__}: {error}")
            return
        self._add_message(ok_role, str(result))

    # -- overview ---------------------------------------------------------

    def _refresh_overview(self) -> None:
        try:
            snapshot = self.service.dashboard()
        except Exception as error:
            self._add_message("error", f"Could not read the dashboard: {error}")
            return
        self._add_message("assistant", snapshot.describe())
        for chart in snapshot.charts:
            self._add_message("note", chart.describe())

    # -- dashboard --------------------------------------------------------

    def _on_dashboard(self) -> None:
        self._run("Reading the dashboard…", lambda: self.service.dashboard().describe())

    def _on_history(self) -> None:
        def job() -> str:
            lines = self.service.history_lines(limit=40)
            if not lines:
                return "No training-center history yet."
            return "\n".join(line.describe() for line in lines)

        self._run("Reading the history…", job)

    def _on_history_search(self) -> None:
        query = simpledialog.askstring("Search the history", "Text to find:", parent=self.window)
        if not query:
            return

        def job() -> str:
            lines = self.service.history_lines(query=query, limit=40)
            if not lines:
                return f"No history entry matches {query!r}."
            return "\n".join(line.describe() for line in lines)

        self._run(f"Searching the history for {query!r}…", job)

    # -- sources ----------------------------------------------------------

    def _on_database(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.window, title="Choose a SQLite database",
            filetypes=[("SQLite database", "*.db *.sqlite *.sqlite3"), ("All files", "*.*")],
        )
        if not path:
            return

        def job() -> str:
            from gui.sql_import import SqlImportService

            service = SqlImportService()
            report = service.inspect(path)
            return report.describe() if hasattr(report, "describe") else str(report)

        self._run(f"Inspecting {path}…", job)

    def _on_internet(self) -> None:
        if not self._require_web("Learning from the internet"):
            return
        urls = simpledialog.askstring(
            "Learn from the internet",
            "One or more URLs, separated by commas:",
            parent=self.window,
        )
        if not urls:
            return
        parsed = [u.strip() for u in urls.replace("\n", ",").split(",") if u.strip()]
        if not parsed:
            return
        if not messagebox.askyesno(
            "Confirm the web path",
            "Importing from the internet is denied by default. Approve fetching "
            f"{len(parsed)} URL(s) now?",
            parent=self.window,
        ):
            return

        def job() -> str:
            outcomes = self.service.import_urls(parsed, approved=True)
            if not outcomes:
                return "The web path was not approved; nothing was imported."
            return "\n".join(
                f"{'ok' if o.ok else 'failed'}  {o.source}: {o.message}" for o in outcomes
            )

        self._run(f"Importing {len(parsed)} URL(s)…", job)

    def _on_gemini(self) -> None:
        if not self._require_web("Asking a cloud model (Gemini)"):
            return
        provider = self._choose(
            "Learn from a cloud model",
            "Which cloud provider?",
            values=self.service.cloud_choices(),
            initial=str(self.service.setting("cloud_provider")),
        )
        if not provider:
            return
        topic = self._choose(
            "Learn from Gemini",
            "The learning topic (pick one or type your own):",
            values=self.service.topic_choices(),
        )
        if not topic:
            return

        def job() -> str:
            questions = self.service.generate_questions(topic, count=5, provider=provider)
            if not questions:
                return (
                    f"{provider} could not generate questions (set the provider's API key "
                    "and keep the internet switch on)."
                )
            examples = self.service.ask_many(questions, provider=provider, topic=topic)
            return "\n".join(e.describe() for e in examples) + (
                "\n\nThese are pending; approve them before export."
            )

        self._run(f"Asking {provider} about {topic!r}…", job)

    def _on_local_model(self) -> None:
        mouth = self._choose(
            "Learn from a local model",
            "Which local model?",
            values=self.service.mouth_choices(),
            initial="ultron",
        )
        if not mouth:
            return
        topic = self._choose(
            "Learning topic",
            "The learning topic (pick one or type your own):",
            values=self.service.topic_choices(),
        )
        if not topic:
            return

        def job() -> str:
            provider = f"local:{mouth}"
            from core.training_center.providers import LocalModelLearningProvider

            questions = LocalModelLearningProvider(mouth).generate(topic, count=5)
            if not questions:
                return f"The local model {mouth!r} could not generate questions (is it installed?)."
            examples = [
                self.service.ask(question, provider=provider, topic=topic)
                for question in questions
            ]
            return "\n".join(e.describe() for e in examples)

        self._run(f"Asking the local model {mouth!r} about {topic!r}…", job)

    def _on_documents(self) -> None:
        paths = filedialog.askopenfilenames(
            parent=self.window, title="Choose documents to import",
            filetypes=[
                ("Documents", "*.txt *.md *.markdown *.csv *.json *.jsonl *.xml *.html *.pdf *.docx *.docm"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return

        def job() -> str:
            report = self.service.import_files(list(paths))
            return report.summary()

        self._run(f"Importing {len(paths)} document(s)…", job)

    def _on_books(self) -> None:
        folder = filedialog.askdirectory(parent=self.window, title="Choose a books folder")
        if not folder:
            return

        def job() -> str:
            report = self.service.import_folder(folder)
            return report.summary()

        self._run(f"Importing the collection at {folder}…", job)

    def _on_characters(self) -> None:
        character = self._choose(
            "Learn with a character",
            "Which character?",
            values=self.service.character_choices(),
            initial="hakeem",
        )
        if not character:
            return
        if character == "cloud" and not self._require_web("Learning with the Cloud character"):
            return
        topic = self._choose(
            "Learning topic",
            "The learning topic (pick one or type your own):",
            values=self.service.topic_choices(),
        )
        if not topic:
            return

        def job() -> str:
            from core.training_center.providers import CharacterLearningProvider

            provider = CharacterLearningProvider(character)
            questions = provider.generate(topic, count=5)
            if not questions:
                return f"The character {character!r} could not generate questions."
            examples = [
                self.service.ask(question, provider=f"character:{character}", topic=topic)
                for question in questions
            ]
            return "\n".join(e.describe() for e in examples)

        self._run(f"Learning with {character!r} about {topic!r}…", job)

    # -- knowledge --------------------------------------------------------

    def _on_knowledge_ingest(self) -> None:
        folder = filedialog.askdirectory(parent=self.window, title="Choose a folder to fold in")
        if not folder:
            return

        def job() -> str:
            report = self.service.import_folder(folder)
            return report.summary()

        self._run(f"Folding {folder} into the knowledge base…", job)

    def _on_knowledge_search(self) -> None:
        query = simpledialog.askstring("Search the knowledge base", "The query:", parent=self.window)
        if not query:
            return

        def job() -> str:
            passages = self.service.knowledge_search(query)
            if not passages:
                return f"No passage matched {query!r}."
            return "\n\n".join(f"[{p.citation}]\n{p.text}" for p in passages)

        self._run(f"Searching for {query!r}…", job)

    def _on_reindex(self) -> None:
        def job() -> str:
            summary = self.service.reindex()
            return f"Index rebuilt: {summary}"

        self._run("Rebuilding the search index…", job)

    def _on_export_knowledge(self) -> None:
        folder = filedialog.askdirectory(parent=self.window, title="Back up the knowledge base to…")
        if not folder:
            return
        self._run(
            "Backing up the knowledge base…",
            lambda: self.service.export_knowledge(folder),
        )

    # -- Q&A --------------------------------------------------------------

    def _on_ask(self) -> None:
        question = simpledialog.askstring("Ask a question", "The question:", parent=self.window)
        if not question:
            return
        provider = self._choose(
            "Provider",
            "Which provider should answer?",
            values=self.service.provider_choices(),
            initial=str(self.service.setting("default_provider")),
        )
        if not provider:
            return
        provider = provider.strip()
        if not self._provider_ready(provider):
            return

        def job() -> str:
            example = self.service.ask(question, provider=provider)
            return example.describe() + "\n\nThis is pending; approve it before export."

        self._run(f"Asking {provider}…", job)

    def _on_generate(self) -> None:
        topic = self._choose(
            "Generate questions",
            "The learning topic (pick one or type your own):",
            values=self.service.topic_choices(),
        )
        if not topic:
            return
        provider = self._choose(
            "Provider",
            "Which provider should generate the questions?",
            values=self.service.provider_choices(),
            initial=str(self.service.setting("default_provider")),
        )
        if not provider:
            return
        provider = provider.strip()
        if not self._provider_ready(provider):
            return

        def job() -> str:
            questions = self.service.generate_questions(topic, count=5, provider=provider)
            if not questions:
                return f"{provider} could not generate questions on {topic!r}."
            examples = self.service.ask_many(questions, provider=provider, topic=topic)
            return "\n".join(e.describe() for e in examples)

        self._run(f"Generating questions on {topic!r}…", job)

    def _provider_ready(self, provider: str) -> bool:
        """Make sure a provider that needs the internet has the switch on.

        A cloud provider (a name in the cloud catalog, or ``cloud``) is refused
        unless the internet switch is on; the local, retrieval, and manual
        providers need nothing and are always ready.
        """
        cloud = False
        try:
            from core.cloud_model import provider_names

            cloud = provider == "cloud" or provider in provider_names()
        except Exception:  # pragma: no cover - defensive
            cloud = False
        if cloud and not self._require_web(f"Asking the cloud provider {provider!r}"):
            return False
        return True

    def _on_review(self) -> None:
        self._run(
            "Reading the review queue…",
            lambda: self.service.review_summary().describe()
            + "\n\n"
            + "\n".join(e.describe() for e in self.service.examples())
            + "\n\nUse the CLI `review --approve <id>` (or the buttons below) to decide.",
        )

    def _on_export_dataset(self) -> None:
        dataset_id = self._choose(
            "Export approved answers",
            "The dataset id to build from the approved examples:",
            values=self.service.dataset_choices(),
            initial="learned",
        )
        if not dataset_id:
            return

        def job() -> str:
            row = self.service.build_dataset_from_approved(dataset_id)
            return row.describe()

        self._run(f"Building {dataset_id!r} from approved examples…", job)

    # -- datasets ---------------------------------------------------------

    def _on_datasets(self) -> None:
        def job() -> str:
            rows = self.service.dataset_rows()
            if not rows:
                return "No datasets yet."
            return "\n".join(row.describe() for row in rows)

        self._run("Listing the datasets…", job)

    def _on_build_dataset(self) -> None:
        dataset_id = self._choose(
            "Build a dataset",
            "The dataset id to build (type a new name, or pick an existing one):",
            values=self.service.dataset_choices(),
            initial="learned",
        )
        if not dataset_id:
            return

        def job() -> str:
            row = self.service.build_dataset_from_approved(dataset_id)
            return row.describe()

        self._run(f"Building {dataset_id!r}…", job)

    def _on_validate_dataset(self) -> None:
        dataset_id = self._choose(
            "Validate a dataset",
            "Which dataset?",
            values=self.service.dataset_choices(),
        )
        if not dataset_id:
            return
        self._run(f"Validating {dataset_id!r}…", lambda: self.service.validate_dataset(dataset_id))

    def _on_export_dataset_file(self) -> None:
        dataset_id = self._choose(
            "Export a dataset",
            "Which dataset?",
            values=self.service.dataset_choices(),
        )
        if not dataset_id:
            return
        folder = filedialog.askdirectory(parent=self.window, title="Export the dataset to…")
        if not folder:
            return

        def job() -> str:
            report = self.service.export_dataset(dataset_id, folder)
            return report.describe()

        self._run(f"Exporting {dataset_id!r}…", job)

    # -- jobs -------------------------------------------------------------

    def _on_jobs(self) -> None:
        def job() -> str:
            rows = self.service.job_rows()
            if not rows:
                return "No training jobs yet."
            return "\n".join(row.describe() for row in rows)

        self._run("Listing the training jobs…", job)

    def _on_feasibility(self) -> None:
        dataset_id = self._choose(
            "Check feasibility",
            "Which dataset?",
            values=self.service.dataset_choices(),
        )
        if not dataset_id:
            return

        def job() -> str:
            model = _inprocess_model()
            config = self.service.build_config(
                model_id="fayd-inprocess-model", dataset_id=dataset_id,
                dataset_version=0, preset=str(self.service.setting("default_preset")),
            )
            report = self.service.feasibility(model, dataset_id, config)
            return report.describe()

        self._run(f"Checking feasibility for {dataset_id!r}…", job)

    def _on_start_job(self) -> None:
        dataset_id = self._choose(
            "Start a training job",
            "Which dataset?",
            values=self.service.dataset_choices(),
        )
        if not dataset_id:
            return
        preset = str(self.service.setting("default_preset"))
        if not messagebox.askyesno(
            "Start training",
            f"Start a governed training run on {dataset_id!r} with the {preset!r} preset? "
            "It runs in the background and can be cancelled.",
            parent=self.window,
        ):
            return

        def job() -> str:
            model = _inprocess_model()
            config = self.service.build_config(
                model_id="fayd-inprocess-model", dataset_id=dataset_id,
                dataset_version=0, preset=preset,
            )
            handle = self._start_streamed_job(model, dataset_id, config)
            return self._describe_handle(handle)

        self._run(f"Training on {dataset_id!r}…", job)

    def _on_cancel_job(self) -> None:
        run_id = self._choose(
            "Cancel a job",
            "Which run?",
            values=self.service.run_choices(),
        )
        if not run_id:
            return
        cancelled = self.service.cancel_job(run_id)
        self._add_message(
            "ok" if cancelled else "error",
            "Cancelled." if cancelled else f"No running job with id {run_id!r}.",
        )

    def _on_resume_job(self) -> None:
        run_id = self._choose(
            "Resume a job",
            "Which run should we resume from?",
            values=self.service.run_choices(),
        )
        if not run_id:
            return
        dataset_id = self._choose(
            "Dataset",
            "Which dataset should the resumed run continue on?",
            values=self.service.dataset_choices(),
        )
        if not dataset_id:
            return

        def job() -> str:
            adapter = self.service.jobs.resume_from_adapter(run_id)
            if not adapter:
                return f"The run {run_id!r} left no adapter to resume from."
            model = _inprocess_model()
            config = self.service.build_config(
                model_id="fayd-inprocess-model", dataset_id=dataset_id,
                dataset_version=0, preset=str(self.service.setting("default_preset")),
            )
            handle = self._start_streamed_job(
                model, dataset_id, config, resume_from=adapter
            )
            return self._describe_handle(handle)

        self._run(f"Resuming from {run_id!r}…", job)

    def _on_retry_job(self) -> None:
        run_id = self._choose(
            "Retry a job",
            "Which run should we retry?",
            values=self.service.run_choices(),
        )
        if not run_id:
            return

        def job() -> str:
            view = self.service.jobs.get(run_id)
            if view is None:
                return f"No training run {run_id!r} was found."
            model = _inprocess_model()
            config = self.service.build_config(
                model_id=view.model_id, dataset_id=view.dataset_id,
                dataset_version=view.dataset_version, preset=str(self.service.setting("default_preset")),
            )
            handle = self._start_streamed_job(model, view.dataset_id, config)
            return self._describe_handle(handle)

        self._run(f"Retrying {run_id!r}…", job)

    # -- fine-tuning ------------------------------------------------------

    def _on_compat(self) -> None:
        def job() -> str:
            from ml.training.compatibility import validate_compatibility

            model = _inprocess_model()
            verdict = validate_compatibility(model)
            return verdict.describe() if hasattr(verdict, "describe") else str(verdict)

        self._run("Checking method compatibility…", job)

    def _on_presets(self) -> None:
        from ml.training.config import preset_summaries

        lines = tuple(f"{name}: {summary}" for name, summary in preset_summaries())
        self._report("assistant", "Training presets (CPU-first):", lines)

    def _on_adapters(self) -> None:
        def job() -> str:
            from ml.training.coordinator import TrainingCoordinator

            coordinator = TrainingCoordinator()
            adapters = coordinator.registry.adapters()
            if not adapters:
                return "No adapters have been trained yet."
            active = coordinator.registry.active_id()
            lines = [f"active: {active or 'none'}"]
            lines.extend(
                f"{a.adapter_id} [{a.status}] {a.dataset_id}@v{a.dataset_version}"
                for a in adapters
            )
            return "\n".join(lines)

        self._run("Reading the adapter registry…", job)

    # -- pretraining ------------------------------------------------------

    def _on_pretrain_status(self) -> None:
        def job() -> str:
            from core.paths import default_checkpoint
            from ml.llm.state import load_step_count

            checkpoint = default_checkpoint()
            if not checkpoint.exists():
                return (
                    "No in-project checkpoint exists yet. Train one with the base "
                    "pipeline (`python -m ml`) or continue from a dataset here."
                )
            steps = load_step_count(checkpoint)
            size = checkpoint.stat().st_size
            return f"Checkpoint: {checkpoint}\n  steps: {steps}\n  size: {size / 1024:.1f} KB"

        self._run("Reading the checkpoint status…", job)

    def _on_estimate(self) -> None:
        def job() -> str:
            model = _inprocess_model()
            params = None
            try:
                params = sum(int(p.size) for p in model.parameters())
            except Exception:
                params = None
            estimate = self.service.estimate(parameter_count=params, rank=8, batch_size=4, grad_accum=2)
            return estimate.describe()

        self._run("Estimating the resources…", job)

    def _on_continue_training(self) -> None:
        dataset_id = self._choose(
            "Continue training",
            "Which dataset should the continued run train on?",
            values=self.service.dataset_choices(),
        )
        if not dataset_id:
            return
        if not messagebox.askyesno(
            "Continue training",
            f"Continue training the in-project checkpoint on {dataset_id!r}? It runs in "
            "the background.",
            parent=self.window,
        ):
            return

        def job() -> str:
            model = _inprocess_model()
            config = self.service.build_config(
                model_id="fayd-inprocess-model", dataset_id=dataset_id,
                dataset_version=0, preset=str(self.service.setting("default_preset")),
            )
            handle = self._start_streamed_job(model, dataset_id, config)
            return self._describe_handle(handle)

        self._run(f"Continuing training on {dataset_id!r}…", job)

    # -- the language model (every training surface in one place) ----------

    def _on_lm_status(self) -> None:
        self._run("Reading the model and its corpus…", self.service.lm_status)

    def _on_lm_corpus(self) -> None:
        self._run("Reading the training corpus…", self.service.lm_corpus)

    def _on_lm_settings(self) -> None:
        self._add_message("assistant", self.service.lm_settings_text())

    def _on_lm_set_setting(self) -> None:
        key = self._choose(
            "Change a training setting",
            "Which option?",
            values=self.service.lm_settings_keys(),
        )
        if not key:
            return
        key = key.strip()
        if key not in self.service.lm_settings_keys():
            self._add_message("error", f"Unknown option {key!r}.")
            return
        current = getattr(self.service.lm_settings(), key, "")
        value = self._choose(
            "Value",
            f"New value for {key}:",
            values=self.service.lm_settings_values(key),
            initial=str(current),
        )
        if value == "":
            return
        parsed: object = value
        lowered = value.strip().lower()
        if lowered in ("true", "false"):
            parsed = lowered == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        try:
            self.service.lm_set_setting(key, parsed)
        except KeyError as error:
            self._add_message("error", str(error))
            return
        self._add_message("ok", f"{key} = {parsed}\n\n{self.service.lm_settings_text()}")

    def _on_lm_train(self) -> None:
        self._start_lm_train(fresh=False, shards=None)

    def _on_lm_train_fresh(self) -> None:
        self._start_lm_train(fresh=True, shards=None)

    def _on_lm_train_sharded(self) -> None:
        value = self._choose(
            "Continue the next corpus shard",
            "How many shards should the corpus be split into?",
            values=("2", "4", "8", "16"),
            initial="4",
        )
        if not value:
            return
        try:
            shards = int(value)
        except ValueError:
            self._add_message("error", f"Not a number: {value!r}")
            return
        self._start_lm_train(fresh=False, shards=shards)

    def _start_lm_train(self, *, fresh: bool, shards: int | None) -> None:
        settings = self.service.lm_settings()
        label = "from scratch" if fresh else "continuing the checkpoint"
        if shards:
            label += f", next of {shards} shards"
        if not messagebox.askyesno(
            "Train the language model",
            f"Train the project's own language model ({settings.steps} steps, "
            f"{settings.preset}, {label})?\n\nIt runs on a worker thread and streams "
            "its progress into this chat area.",
            parent=self.window,
        ):
            return

        def job() -> str:
            outcome = self.service.lm_train(
                fresh=fresh,
                shards=shards,
                on_line=self._stream_line,
                on_step=self._lm_step,
            )
            return outcome.describe()

        self._run(f"Training the language model ({label})…", job)

    def _lm_step(self, step: int, loss: float, total: int) -> None:
        """Queue one live step line from the training thread (Tk-safe)."""
        percent = (step / total * 100) if total else 0.0
        self._stream_line(f"step {step}/{total} ({percent:.0f}%) · loss {loss:.4f}")

    def _on_lm_release(self) -> None:
        if not messagebox.askyesno(
            "Train everything for a release",
            "Train every learning layer — the base pipeline, the petroleum tasks, and "
            "the language model — and write the release manifest?\n\nIt is a long run.",
            parent=self.window,
        ):
            return

        def job() -> str:
            outcome = self.service.lm_release(on_line=self._stream_line)
            return outcome.describe()

        self._run("Training every learning layer for a release…", job)

    def _on_lm_evaluate(self) -> None:
        def job() -> str:
            return self.service.lm_evaluate(on_line=self._stream_line).describe()

        self._run("Measuring the model on held-out data…", job)

    def _on_lm_generate(self) -> None:
        prompt = self._choose(
            "Sample from the model",
            "The prompt to continue from:",
            initial="porosity",
        )
        if not prompt:
            return

        def job() -> str:
            text = self.service.lm_generate(prompt)
            return f"{prompt}{text}"

        self._run(f"Sampling from the model with {prompt!r}…", job)

    def _on_lm_install_pack(self) -> None:
        key = self._choose(
            "Install the committed weights",
            "Which weights-pack entry?",
            values=self.service.lm_pack_choices(),
            initial="fayd-llm",
        )
        if not key:
            return
        key = key.strip()
        if not messagebox.askyesno(
            "Install weights",
            f"Install the pack entry {key!r} into the folder the trainer reads? "
            "An existing checkpoint is kept as a backup.",
            parent=self.window,
        ):
            return
        self._run(
            f"Installing the {key!r} weights…",
            lambda: self.service.lm_install_pack(key).describe(),
        )

    def _on_lm_import_weights(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.window, title="Choose a .npz weight file",
            filetypes=[("NumPy weights", "*.npz"), ("All files", "*.*")],
        )
        if not path:
            return
        self._run(
            f"Importing weights from {path}…",
            lambda: self.service.lm_import_weights(path).describe(),
        )

    def _on_lm_fetch_weights(self) -> None:
        url = simpledialog.askstring(
            "Fetch weights", "The HTTPS URL of the .npz weight file:", parent=self.window
        )
        if not url:
            return
        if not messagebox.askyesno(
            "Fetch weights",
            f"Fetch weights from\n\n{url}\n\nThis leaves this machine and needs approval.",
            parent=self.window,
        ):
            return
        self._run(
            "Fetching weights…",
            lambda: self.service.lm_fetch_weights(url, approved=True).describe(),
        )

    def _on_lm_build_learned(self) -> None:
        tier = self._choose(
            "Train a larger capacity tier",
            "Which tier?",
            values=self.service.lm_tier_choices(),
            initial="learned-large",
        )
        if not tier:
            return
        if not messagebox.askyesno(
            "Train a larger tier",
            f"Train the {tier!r} tier? It is a long run into the pack folder.",
            parent=self.window,
        ):
            return

        def job() -> str:
            return self.service.lm_build_learned(tier, on_line=self._stream_line).describe()

        self._run(f"Training the {tier!r} tier…", job)

    def _on_lm_learned(self) -> None:
        self._run("Listing the learned tiers…", self.service.lm_learned_text)

    def _on_lm_plan(self) -> None:
        self._add_message("assistant", self.service.language_model.plan())

    # -- evaluation -------------------------------------------------------

    def _on_evaluate(self) -> None:
        run_id = self._choose(
            "Evaluate a model",
            "Which run should we evaluate?",
            values=self.service.run_choices(),
        )
        if not run_id:
            return

        def job() -> str:
            report = self.service.evaluate_run(run_id)
            return report.describe()

        self._run(f"Evaluating {run_id!r}…", job)

    def _on_compare(self) -> None:
        self._add_message(
            "note",
            "Compare two evaluation reports: run `python -m cli train evaluate <run_id>` for each "
            "run, then read the two reports side by side here — every metric is on the held-out "
            "test split, so the comparison is honest.",
        )

    def _on_gate(self) -> None:
        def job() -> str:
            from core.training_center.gate import describe

            return describe()

        self._run("Running the held-out gate…", job)

    # -- knowledge base ---------------------------------------------------

    def _on_kb_build(self) -> None:
        if not self._require_web("Building the bulk knowledge base"):
            return
        target = simpledialog.askstring(
            "Build the knowledge base", "Target size in MB:", parent=self.window, initialvalue="100"
        )
        if not target:
            return
        try:
            target_mb = float(target)
        except ValueError:
            self._add_message("error", "The target must be a number of megabytes.")
            return
        if not messagebox.askyesno(
            "Build the knowledge base",
            f"Build the bulk knowledge base up to {target_mb:g} MB from public sources? "
            "This fetches from the internet and is denied by default.",
            parent=self.window,
        ):
            return

        def job() -> str:
            report = self.service.build_knowledge_base(
                target_mb=target_mb, progress=self._stream_line
            )
            return report.describe() if hasattr(report, "describe") else str(report)

        self._run(f"Building the knowledge base to {target_mb:g} MB…", job)

    def _on_kb_stats(self) -> None:
        def job() -> str:
            stats = self.service.knowledge_stats()
            return "\n".join(f"{key}: {value}" for key, value in sorted(stats.items()))

        self._run("Reading the knowledge-base status…", job)

    # -- pipelines --------------------------------------------------------

    def _on_pipeline_list(self) -> None:
        def job() -> str:
            pipelines = self.service.pipeline_list()
            if not pipelines:
                return "No saved pipelines. Create one with `pipeline --create <id>` in the CLI."
            return "\n".join(pipeline.describe() for pipeline in pipelines)

        self._run("Listing the pipelines…", job)

    def _on_pipeline_run(self) -> None:
        pipeline_id = self._choose(
            "Run a pipeline",
            "Which pipeline?",
            values=self.service.pipeline_choices(),
        )
        if not pipeline_id:
            return
        if not messagebox.askyesno(
            "Run the pipeline",
            f"Run {pipeline_id!r}? A governed stage (training) is refused unless approved, "
            "and a stage that needs the web is refused too.",
            parent=self.window,
        ):
            return
        if self._pipeline_needs_web(pipeline_id) and not self._require_web(
            f"The pipeline {pipeline_id!r} downloads URLs"
        ):
            return

        def job() -> str:
            report = self.service.run_pipeline(
                pipeline_id, approved=True, on_stage=lambda outcome: self._stream_line(outcome.describe())
            )
            return report.describe()

        self._run(f"Running {pipeline_id!r}…", job)

    def _pipeline_needs_web(self, pipeline_id: str) -> bool:
        """Whether a saved pipeline has a stage that reaches the internet."""
        try:
            pipeline = self.service.pipelines.get(pipeline_id)
        except Exception:  # pragma: no cover - defensive
            return False
        if pipeline is None:
            return False
        return any(stage.stage == "import_urls" and stage.enabled for stage in pipeline.stages)

    # -- resources --------------------------------------------------------

    def _on_resources(self) -> None:
        self._run("Reading this machine's resources…", lambda: self.service.resources().describe())

    # -- settings ---------------------------------------------------------

    def _on_settings(self) -> None:
        self._add_message("assistant", self.service.settings_text())

    def _on_set_setting(self) -> None:
        key = self._choose(
            "Change a setting",
            "Which setting?",
            values=self.service.setting_keys(),
        )
        if not key:
            return
        key = key.strip()
        if key not in self.service.setting_keys():
            self._add_message("error", f"Unknown setting {key!r}.")
            return
        value = self._choose(
            "Value",
            f"New value for {key}:",
            values=self.service.setting_value_choices(key),
            initial=str(self.service.settings().get(key, "")),
        )
        if value == "":
            return
        parsed: object = value
        if value.lower() in ("true", "false"):
            parsed = value.lower() == "true"
        elif value.isdigit():
            parsed = int(value)
        try:
            self.service.set_setting(key, parsed)
        except KeyError as error:
            self._add_message("error", str(error))
            return
        self._add_message("ok", f"{key} = {parsed}")
        if key == "web":
            self._refresh_web_toggle()

    def _on_toggle_web(self) -> None:
        """Flip the internet switch from the Settings section."""
        now = self.service.toggle_web()
        self._refresh_web_toggle()
        self._add_message("ok" if now else "note", self.service.web_status())

    def _on_catalog(self) -> None:
        self._add_message("assistant", self.service.catalog_text())

    # -- the composer -----------------------------------------------------

    def _on_command(self) -> None:
        question = self._command_var.get().strip()
        if not question:
            return
        self._command_var.set("")
        self._add_message("user", question)
        provider = str(self.service.setting("default_provider"))

        def job() -> str:
            example = self.service.ask(question, provider=provider)
            return example.describe() + "\n\nThis is pending; approve it before export."

        self._run(f"Asking {provider}…", job)

    # -- lifecycle --------------------------------------------------------

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Close the help window this one opened (if any) so it does not outlive its
        # parent; a failure there must never stop this window closing.
        help_window = getattr(self, "_help_window", None)
        if help_window is not None:
            with best_effort("close the Training Center's help window"):
                help_window._close()
            self._help_window = None
        if self._pump_after is not None:
            try:
                self.window.after_cancel(self._pump_after)
            except tk.TclError:
                pass
            self._pump_after = None
        try:
            self.window.destroy()
        finally:
            if self._on_close is not None:
                self._on_close()


def _inprocess_model() -> Any:
    """The in-project transformer the window trains: the checkpoint, or a fresh one."""
    from core.paths import default_checkpoint
    from ml.llm.transformer import TransformerConfig, TransformerLM

    checkpoint = default_checkpoint()
    if checkpoint.exists():
        return TransformerLM.load(checkpoint)
    return TransformerLM(TransformerConfig.preset("small"))


__all__ = [
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "MIN_HEIGHT",
    "MIN_WIDTH",
    "SIDEBAR_WIDTH",
    "TrainingCenterDialog",
]
