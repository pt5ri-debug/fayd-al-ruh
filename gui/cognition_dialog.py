"""The Autonomous Cognitive System window: mind, learning, training, characters.

Reached from the sidebar's **Autonomous mind** entry (and from the composer's
＋ menu), this one window drives the whole autonomous layer through a single
:class:`~gui.cognition.CognitiveService`, so the window and
``python -m cli cognition`` share one proactive mind, one learning system, one
version log, and one evaluator -- no logic is reimplemented in either front end.

The window is built the way every other window in the project is built: the same
:class:`~gui.app.Theme` tokens, the shared
:class:`~gui.sidebar.SidebarColumn` on the left, the shared
:class:`~gui.chat_composer.ChatComposer` at the bottom, the three
:class:`~gui.window_controls` title-bar buttons, and a worker thread whose result
is handed back through a queue the UI thread drains (Tk is not thread-safe).

The left sidebar is a stack of collapsible sections, one per area of the task:

* **Proactive Mind** -- run a thinking cycle, review the ideas, see the drives,
  and run one honest end-to-end demonstration.
* **Cognitive Learning** -- learn a statement, recall from every memory, and
  record an experience with its outcome.
* **Gradual Training** -- the staged trainer: run one stage, advance gradually,
  see how far training has come, read the version log, and roll back. Training is
  a governed action, so the run button asks for confirmation first.
* **Character Evaluation** -- the eight characters scored on four dimensions
  (training structure, memory, traits, development headroom), with each
  character's strengths, shortcomings, and next steps.
* **Cognitive Indicators** -- the ten indicators, measured.

Everything is honest: a governed action states why it was refused, a missing
model names the real next step, and a character's score is computed from its real
configuration and learning store -- never from a fixed table.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable
from typing import Any

from cognition.evaluation import INDICATORS, LABELS
from cognition.knowledge_depth import DEFAULT_HARVEST_LIMIT
from cognition.runner import ResourceMode
from cognition.training import TrainingStage
from core.cognition_catalog import (
    CATEGORY_SUMMARIES,
    CATEGORY_SUMMARIES_AR,
    capability,
)

from gui.app import RADIUS_LARGE, RoundedBox
from gui.character_evaluation import DIMENSION_TITLES, EvaluationReport
from gui.chat_composer import ChatComposer, Tooltip
from gui.cognition import CognitiveService, OperationReport
from gui.sidebar import SidebarColumn
from gui.typography import resolve
from gui.window_controls import fit_geometry, install_window_controls

TITLE = "Autonomous Cognitive System"
SUBTITLE = (
    "The proactive mind, cognitive learning, and gradual training — over the "
    "project's own knowledge graph, with the eight characters evaluated."
)
HONESTY_LINE = (
    "Nothing here widens the model's authority: a generated idea is never a fact, "
    "a teacher is never trusted blindly, and a trained version is adopted only when "
    "the evaluation shows it did not regress — otherwise it rolls back."
)

#: The window's opening size and the smallest it may be resized to.
DEFAULT_WIDTH = 1240
DEFAULT_HEIGHT = 760
MIN_WIDTH = 980
MIN_HEIGHT = 640
SIDEBAR_WIDTH = 300

#: The conversation column is capped, like the chat window's, so it reads centred.
_COLUMN_MAX = 880

#: A transcript card taller than this many pixels scrolls inside itself.
_MAX_CARD_HEIGHT = 420


class CognitionDialog:
    """The one window for the autonomous cognitive system."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        service: CognitiveService,
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
        self._pump_after: str | None = None
        self._cards: list[tk.Text] = []
        self._results: queue.Queue[
            tuple[Callable[[Any, Exception | None], None], Any, Exception | None]
        ] = queue.Queue()
        # The self-study session emits each turn from its worker thread; the UI
        # thread drains this queue and draws them, so the questions and answers
        # appear on screen *as they happen* (Tk is not thread-safe).
        self._turns: queue.Queue[Any] = queue.Queue()
        self._turn_pump: str | None = None

        self._statement_var = tk.StringVar(value="")
        self._command_var = tk.StringVar(value="")
        self._study_count_var = tk.StringVar(value="200")

        self.window = tk.Toplevel(parent)
        self.window.title(f"{TITLE} — Fayd Al-Ruh")
        self.window.configure(bg=theme.base)
        self.window.geometry(self._initial_geometry())
        self.window.minsize(MIN_WIDTH, MIN_HEIGHT)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)

        self._build()
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
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=820,
        ).pack(anchor="w", pady=(2, 0))
        self._spinner = tk.Label(
            inner, text="", bg=theme.titlebar, fg=theme.primary,
            font=self._type.font("2xs", weight="bold"),
        )
        self._spinner.pack(side="right", padx=(0, 12))
        self._window_controls = install_window_controls(
            self.window, inner, theme=theme, typography=self._type, on_close=self._close
        )
        tk.Frame(self.window, bg=theme.titlebar_edge, height=1).pack(fill="x")

    def _button(self, parent: tk.Misc, text: str, command: Any) -> tk.Button:
        return tk.Button(
            parent, text=text, command=command, relief="flat", borderwidth=0,
            highlightthickness=0, bg=self.theme.surface_raised, fg=self.theme.foreground,
            activebackground=self.theme.interactive_active, activeforeground=self.theme.contrast,
            font=self._type.font("2xs", weight="bold"), padx=10, pady=4, cursor="hand2",
            anchor="w",
        )

    # -- explaining a control before it is pressed -------------------------

    def _hint(self, parent: tk.Misc, text: str) -> tk.Label:
        """A one-line note under a control, so its purpose is readable at rest."""
        label = tk.Label(
            parent, text=text, bg=self.theme.surface, fg=self.theme.text_dim,
            font=self._type.font("2xs"), justify="left", anchor="w",
            wraplength=SIDEBAR_WIDTH - 48,
        )
        label.pack(fill="x", padx=(20, 12), pady=(0, 6))
        return label

    def _section_note(self, section: tk.Frame, category: str) -> None:
        """The one-sentence description of a whole section, drawn under its title.

        Both languages are shown: the English summary a reader already knows, and
        the Arabic one beside it, so the panel is usable whichever language the
        reader prefers. A section with no declared summary draws nothing.
        """
        english = CATEGORY_SUMMARIES.get(category, "")
        arabic = CATEGORY_SUMMARIES_AR.get(category, "")
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
        """A button whose label, hint, and tooltip all come from the catalogue.

        The catalogue (:mod:`core.cognition_catalog`) is the single source of
        truth, so the window can never describe a control differently from the CLI
        or the docs -- and the hover tooltip carries the full description,
        including what the action returns and the headless command that runs it.
        """
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

    def _build_sidebar(self, parent: tk.Frame) -> None:
        column = SidebarColumn(
            parent, theme=self.theme, typography=self._type, width=SIDEBAR_WIDTH,
            scope="cognition",
        )
        self._sidebar_column = column
        self._sidebar = column.body
        theme = self.theme

        brand = tk.Frame(column.body, bg=theme.surface)
        brand.pack(fill="x", padx=14, pady=(14, 4))
        dot = tk.Canvas(brand, width=24, height=24, bg=theme.surface, highlightthickness=0)
        dot.create_oval(1, 1, 23, 23, fill=theme.primary, outline="")
        dot.create_text(
            12, 13, text="ع", fill=theme.primary_foreground,
            font=self._type.font("sm", weight="bold"),
        )
        dot.pack(side="left")
        tk.Label(
            brand, text="Cognitive System", bg=theme.surface, fg=theme.contrast,
            font=self._type.font("body", weight="bold"),
        ).pack(side="left", padx=(8, 0))

        # A short instruction so a first-time reader knows the panel explains
        # itself: every button has a note under it and a full tooltip on hover.
        tk.Label(
            column.body,
            text=(
                "Every button is explained: a note under it says what it does, and "
                "hovering shows what it returns and the exact command it runs. The "
                "button labelled \"What does each item do?\" prints the whole list."
            ),
            bg=theme.surface, fg=theme.text_dim, font=self._type.font("2xs"),
            justify="left", anchor="w", wraplength=SIDEBAR_WIDTH - 40,
        ).pack(fill="x", padx=14, pady=(0, 8))

        self._build_mind_section(column)
        self._build_study_section(column)
        self._build_depth_section(column)
        self._build_metacognition_section(column)
        self._build_learning_section(column)
        self._build_consolidation_section(column)
        self._build_training_section(column)
        self._build_characters_section(column)
        self._build_indicators_section(column)
        self._build_system_section(column)

    def _build_mind_section(self, column: SidebarColumn) -> None:
        section = column.add("mind", "Proactive Mind", icon="🧠")
        self._section_note(section, "mind")
        self._cap_button(section, "think", self._on_think)
        self._cap_button(section, "ideas", self._on_ideas)
        self._cap_button(section, "drives", self._on_drives)
        self._cap_button(section, "prove", self._on_prove)

    def _build_study_section(self, column: SidebarColumn) -> None:
        section = column.add("study", "Self-Study (ask & learn)", icon="📖")
        self._section_note(section, "study")
        row = tk.Frame(section, bg=self.theme.surface)
        row.pack(fill="x", padx=12, pady=(2, 0))
        tk.Label(
            row, text="Questions per session:", bg=self.theme.surface,
            fg=self.theme.text_secondary, font=self._type.font("2xs"),
        ).pack(side="left")
        tk.Entry(
            row, textvariable=self._study_count_var, width=6, justify="center",
            bg=self.theme.surface_raised, fg=self.theme.foreground, relief="flat",
            highlightthickness=0, insertbackground=self.theme.foreground,
            font=self._type.font("2xs"),
        ).pack(side="left", padx=(6, 0))
        self._hint(section, "How many questions one session asks (200 by default).")
        self._study_button = self._cap_button(section, "study", self._on_study)
        self._cap_button(section, "bank_size", self._on_bank_size)
        self._cap_button(section, "next_questions", self._on_next_questions)
        self._cap_button(section, "last_report", self._on_last_report)

    def _build_depth_section(self, column: SidebarColumn) -> None:
        section = column.add("depth", "Knowledge Depth (widest reach)", icon="🌐")
        self._section_note(section, "depth")
        self._cap_button(section, "depth", self._on_depth)
        self._cap_button(section, "coverage", self._on_coverage)
        self._cap_button(section, "harvest", self._on_harvest)
        self._cap_button(section, "bridges", self._on_bridges)
        self._cap_button(section, "arabic", self._on_arabic)

    def _build_metacognition_section(self, column: SidebarColumn) -> None:
        section = column.add("metacognition", "Metacognition (self-model)", icon="🪞")
        self._section_note(section, "metacognition")
        self._cap_button(section, "metacognition", self._on_metacognition)
        self._cap_button(section, "plan", self._on_plan)

    def _build_consolidation_section(self, column: SidebarColumn) -> None:
        section = column.add("consolidation", "Consolidation & Review", icon="🌙")
        self._section_note(section, "consolidation")
        self._cap_button(section, "consolidate", self._on_consolidate)
        self._cap_button(section, "patterns", self._on_patterns)
        self._cap_button(section, "review", self._on_review)
        self._cap_button(section, "skills", self._on_skills)

    def _build_learning_section(self, column: SidebarColumn) -> None:
        section = column.add("learning", "Cognitive Learning", icon="📚")
        self._section_note(section, "learning")
        wrap = tk.Frame(section, bg=self.theme.surface)
        wrap.pack(fill="x", padx=12, pady=(4, 0))
        tk.Label(
            wrap, text="Statement to learn or to recall by", bg=self.theme.surface,
            fg=self.theme.text_secondary, font=self._type.font("2xs"), anchor="w",
        ).pack(fill="x")
        entry = tk.Entry(
            wrap, textvariable=self._statement_var, bg=self.theme.base,
            fg=self.theme.foreground, relief="flat", highlightthickness=1,
            highlightbackground=self.theme.border_subtle, highlightcolor=self.theme.primary,
            insertbackground=self.theme.foreground, font=self._type.font("xs"),
        )
        entry.pack(fill="x", ipady=4, pady=(2, 4))
        entry.bind("<Return>", lambda _e: self._on_learn())
        self._cap_button(section, "learn", self._on_learn)
        self._cap_button(section, "recall", self._on_recall)
        self._cap_button(section, "learning_stats", self._on_learning_stats)

    def _build_training_section(self, column: SidebarColumn) -> None:
        section = column.add("training", "Gradual Training", icon="🎓")
        self._section_note(section, "training")
        theme = self.theme
        row = tk.Frame(section, bg=theme.surface)
        row.pack(fill="x", padx=12, pady=(4, 0))
        tk.Label(
            row, text="Stage", bg=theme.surface, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w",
        ).pack(side="left")
        self._stage_var = tk.StringVar(value=TrainingStage.SMOKE.value)
        stage_menu = tk.OptionMenu(row, self._stage_var, *[s.value for s in TrainingStage])
        stage_menu.configure(
            bg=theme.surface_raised, fg=theme.foreground, relief="flat",
            highlightthickness=0, font=self._type.font("2xs"), activebackground=theme.interactive_active,
        )
        stage_menu["menu"].configure(bg=theme.surface, fg=theme.foreground)
        stage_menu.pack(side="right")
        self._hint(
            section,
            "smoke (seconds) < quick (a minute) < standard (minutes) < full (long).",
        )
        self._cap_button(section, "train", self._on_train)
        self._cap_button(section, "train_gradual", self._on_train_gradual)
        self._cap_button(section, "train_autonomous", self._on_train_autonomous)
        self._cap_button(section, "progression", self._on_progression)
        self._cap_button(section, "curriculum", self._on_curriculum)
        self._cap_button(section, "versions", self._on_versions)
        self._cap_button(section, "rollback", self._on_rollback)

    def _build_characters_section(self, column: SidebarColumn) -> None:
        section = column.add("characters", "Character Evaluation", icon="🧑")
        self._section_note(section, "characters")
        self._cap_button(section, "characters", self._on_characters)

    def _build_indicators_section(self, column: SidebarColumn) -> None:
        section = column.add("indicators", "Cognitive Indicators", icon="📊")
        self._section_note(section, "indicators")
        self._cap_button(section, "evaluate", self._on_evaluate)

    def _build_system_section(self, column: SidebarColumn) -> None:
        section = column.add("system", "Resource Mode", icon="⚙")
        self._section_note(section, "system")
        theme = self.theme
        row = tk.Frame(section, bg=theme.surface)
        row.pack(fill="x", padx=12, pady=(4, 0))
        self._mode_var = tk.StringVar(value=self.service.mode().value)
        mode_menu = tk.OptionMenu(row, self._mode_var, *[m.value for m in ResourceMode])
        mode_menu.configure(
            bg=theme.surface_raised, fg=theme.foreground, relief="flat",
            highlightthickness=0, font=self._type.font("2xs"), activebackground=theme.interactive_active,
        )
        mode_menu["menu"].configure(bg=theme.surface, fg=theme.foreground)
        mode_menu.pack(fill="x")
        self._hint(section, "light < balanced < intensive (more cycles, never unbounded).")
        self._cap_button(section, "mode", self._on_mode)
        self._cap_button(section, "tick", self._on_tick)
        self._cap_button(section, "catalog", self._on_catalog)

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
        """The shared rounded composer card, used as a command line for the panel."""
        composer = ChatComposer(
            parent, theme=self.theme, typography=self._type,
            placeholder="Type a statement to learn, or a question to teach the student…",
            on_submit=self._on_command, outer_bg=self.theme.base, with_attach=False,
            variable=self._command_var,
        )
        composer.wrap.pack(side="bottom", fill="x", padx=self._column_pad(), pady=(8, 12))
        self._composer = composer
        self._command_entry = composer.entry
        self._command_button = self._button(composer.right_tools, "Run", self._on_command)
        composer.add_tool(self._command_button)
        cap = capability("command")
        if cap is not None:
            Tooltip(
                self._command_button,
                f"{cap.title}\n{cap.summary}\n\nReturns: {cap.outcome}\n{cap.cli}",
                self.theme,
                self._type,
            )

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
        """Add one message card and return its text widget."""
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
        """Size a card to its content, capped so a long answer scrolls itself."""
        try:
            widget.update_idletasks()
            lines = int(widget.index("end-1c").split(".")[0])
        except tk.TclError:
            return
        height = max(1, min(lines, 40))
        widget.configure(height=height)

    def _scroll_to_end(self) -> None:
        self._transcript.update_idletasks()
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._canvas.yview_moveto(1.0)

    # -- background work --------------------------------------------------

    def _start_work(
        self, job: Callable[[], Any], finish: Callable[[Any, Exception | None], None]
    ) -> None:
        if self._busy:
            return
        self._busy = True
        self._spinner.configure(text="working…")

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
        try:
            finish, result, error = self._results.get_nowait()
        except queue.Empty:
            if self._busy and not self._closed:
                self._schedule_pump()
            return
        self._busy = False
        self._spinner.configure(text="")
        finish(result, error)

    def _report(self, role: str, title: str, lines: tuple[str, ...] = (), detail: str = "") -> None:
        body = title
        if lines:
            body += "\n" + "\n".join(f"• {line}" for line in lines)
        if detail:
            body += f"\n\n{detail}"
        self._add_message(role, body)

    def _finish_report(self, result: OperationReport | None, error: Exception | None) -> None:
        if error is not None:
            self._add_message("error", f"{type(error).__name__}: {error}")
            return
        assert result is not None
        role = "ok" if result.ok else "error"
        self._report(role, result.title, result.lines, result.detail)

    # -- overview ---------------------------------------------------------

    def _refresh_overview(self) -> None:
        def job() -> dict[str, Any]:
            return self.service.status()

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("error", f"Could not read the mind's status: {error}")
                return
            status = result
            self._add_message(
                "note",
                "Autonomous cognitive system ready.\n"
                f"• cycles run: {status.get('cycles', 0)}\n"
                f"• interval: {status.get('interval_seconds', '?')}s, "
                f"{status.get('ideas_per_cycle', '?')} idea(s) per cycle\n"
                f"• paused: {status.get('paused', False)}",
            )

        self._start_work(job, finish)

    # -- knowledge depth --------------------------------------------------

    def _on_depth(self) -> None:
        self._start_work(self.service.depth, self._finish_report)

    def _on_coverage(self) -> None:
        def job() -> tuple[str, ...]:
            return self.service.coverage()

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("error", str(error))
                return
            self._report("assistant", "Coverage by discipline and course", tuple(result))

        self._start_work(job, finish)

    def _on_harvest(self) -> None:
        if not self._confirm(
            "Harvest relations from the ingested text?",
            "This reads typed relations from the bulk knowledge base (the files you "
            "imported) and folds them into the graph through the bounded learner. "
            "Nothing overrides a seeded relation, and both endpoints must already be "
            "known concepts. Continue?",
        ):
            return
        self._start_work(
            lambda: self.service.harvest(
                limit=DEFAULT_HARVEST_LIMIT, write=True, approved=True
            ),
            self._finish_report,
        )

    def _on_bridges(self) -> None:
        def job() -> tuple[str, ...]:
            return self.service.bridges(limit=6)

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("error", str(error))
                return
            bridges = result
            if not bridges:
                self._add_message(
                    "note",
                    "No bridge found within the hop bound: the two disciplines the "
                    "graph connects first are not linked by a short chain.",
                )
                return
            self._report("assistant", "Bridges between disciplines", tuple(bridges))

        self._start_work(job, finish)

    def _on_arabic(self) -> None:
        self._start_work(self.service.arabic, self._finish_report)

    # -- the capability catalogue -----------------------------------------

    def _on_catalog(self) -> None:
        def job() -> tuple[str, ...]:
            return self.service.catalog()

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("error", str(error))
                return
            self._add_message(
                "assistant",
                "What each item in this window does\n\n" + "\n".join(result),
            )

        self._start_work(job, finish)

    # -- proactive mind ---------------------------------------------------

    def _on_think(self) -> None:
        self._start_work(self.service.think, self._finish_report)

    # -- self-study -------------------------------------------------------

    def _on_bank_size(self) -> None:
        def job() -> int:
            return self.service.question_bank_size()

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("error", str(error))
                return
            self._add_message(
                "assistant",
                f"The mind's question bank holds {result} distinct questions, built "
                "from the graph's own relations and the project's authored topics.",
            )

        self._start_work(job, finish)

    def _on_next_questions(self) -> None:
        def job() -> tuple[str, ...]:
            return self.service.next_questions(count=20)

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("error", str(error))
                return
            questions = result
            if not questions:
                self._add_message("note", "No questions available.")
                return
            self._report("assistant", "The next questions the mind would ask", tuple(questions))

        self._start_work(job, finish)

    def _on_last_report(self) -> None:
        report = self.service.last_study_report()
        if report is None:
            self._add_message("note", "No self-study session has been run yet.")
            return
        self._add_message("assistant", report.describe())

    def _on_study(self) -> None:
        """Start a self-study session and stream the questions and answers live."""
        if self._busy:
            return
        try:
            count = max(1, int(self._study_count_var.get().strip() or "200"))
        except ValueError:
            count = 200
        self._add_message(
            "user",
            f"Start asking questions ({count}) — the mind will study its own "
            "questions and report how much it gained.",
        )
        self._study_button.configure(state="disabled")
        self._start_work(
            lambda: self.service.study(
                count=count,
                on_turn=self._turns.put,
                measure_indicators=True,
                approved=True,
            ),
            self._finish_study,
        )
        self._schedule_turn_pump()

    def _schedule_turn_pump(self) -> None:
        if self._closed:
            return
        self._turn_pump = self.window.after(120, self._pump_turns)

    def _pump_turns(self) -> None:
        """Draw the questions and answers that arrived since the last pump."""
        self._turn_pump = None
        drew = 0
        while True:
            try:
                turn = self._turns.get_nowait()
            except queue.Empty:
                break
            self._add_message(
                "note" if not turn.answered else "assistant",
                f"Q{turn.index}: {turn.question}\nA: {turn.answer}",
            )
            drew += 1
            if drew >= 12:  # bounded per pump, so the UI never stalls
                break
        if self._busy and not self._closed:
            self._schedule_turn_pump()

    def _finish_study(self, result: Any, error: Exception | None) -> None:
        # Drain any remaining turns before the report, so the stream is complete.
        self._drain_turns()
        self._study_button.configure(state="normal")
        if error is not None:
            self._add_message("error", f"{type(error).__name__}: {error}")
            return
        report = result
        role = "ok" if report.ok else "error"
        self._report(role, report.title, report.lines)

    def _drain_turns(self) -> None:
        while True:
            try:
                turn = self._turns.get_nowait()
            except queue.Empty:
                return
            self._add_message(
                "note" if not turn.answered else "assistant",
                f"Q{turn.index}: {turn.question}\nA: {turn.answer}",
            )

    def _on_ideas(self) -> None:
        def job() -> tuple[Any, ...]:
            return self.service.ideas()

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("error", str(error))
                return
            rows = result
            if not rows:
                self._add_message("note", "No ideas awaiting review. Run a thinking cycle first.")
                return
            for row in rows:
                self._add_message("assistant", row.describe())

        self._start_work(job, finish)

    def _on_drives(self) -> None:
        def job() -> tuple[str, ...]:
            return self.service.drives()

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("error", str(error))
                return
            drives = result
            if not drives:
                self._add_message("note", "Nothing to pursue: the graph is complete for now.")
                return
            self._report("assistant", "What the mind would pursue next", tuple(drives))

        self._start_work(job, finish)

    def _on_prove(self) -> None:
        def job() -> tuple[OperationReport, ...]:
            return self.service.prove()

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("error", str(error))
                return
            for report in result:
                self._report("ok" if report.ok else "error", report.title, report.lines, report.detail)

        self._start_work(job, finish)

    # -- metacognition ----------------------------------------------------

    def _on_metacognition(self) -> None:
        self._start_work(self.service.metacognition, self._finish_report)

    def _on_plan(self) -> None:
        self._start_work(self.service.plan_next, self._finish_report)

    # -- learning ---------------------------------------------------------

    def _on_learn(self) -> None:
        statement = self._statement_var.get().strip()
        if not statement:
            self._add_message("error", "Type a statement to learn first.")
            return
        self._add_message("user", statement)
        self._statement_var.set("")
        self._start_work(lambda: self.service.learn(statement), self._finish_report)

    def _on_recall(self) -> None:
        query = self._statement_var.get().strip() or "permeability"
        self._add_message("user", f"recall: {query}")

        def job() -> str:
            return self.service.recall(query).describe()

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("error", str(error))
                return
            self._add_message("assistant", result)

        self._start_work(job, finish)

    def _on_learning_stats(self) -> None:
        def job() -> str:
            return self.service.learning_stats()

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("error", str(error))
                return
            self._add_message("assistant", f"Learning system: {result}")

        self._start_work(job, finish)

    # -- consolidation ----------------------------------------------------

    def _on_consolidate(self) -> None:
        self._start_work(self.service.consolidate, self._finish_report)

    def _on_patterns(self) -> None:
        def job() -> tuple[str, ...]:
            return self.service.patterns()

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("error", str(error))
                return
            patterns = result
            if not patterns:
                self._add_message(
                    "note",
                    "No patterns yet. Record experiences, then consolidate.",
                )
                return
            self._report("assistant", "Patterns extracted from experience", tuple(patterns))

        self._start_work(job, finish)

    def _on_review(self) -> None:
        self._start_work(lambda: self.service.review_due(remembered=True), self._finish_report)

    def _on_skills(self) -> None:
        def job() -> tuple[str, ...]:
            return self.service.skills()

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("error", str(error))
                return
            skills = result
            if not skills:
                self._add_message("note", "No skills practised yet.")
                return
            self._report("assistant", "Practised skills and their proficiency", tuple(skills))

        self._start_work(job, finish)

    # -- training ---------------------------------------------------------

    def _on_train(self) -> None:
        stage = TrainingStage(self._stage_var.get())
        if not self._confirm(
            "Run a training stage?",
            f"This runs the '{stage.value}' stage and writes a candidate model version.\n"
            "It is adopted only when the evaluation shows it did not regress; "
            "otherwise the previous version stays in use.",
        ):
            return
        self._start_work(
            lambda: self.service.train(stage=stage, approved=True), self._finish_report
        )

    def _on_train_gradual(self) -> None:
        up_to = TrainingStage(self._stage_var.get())
        if not self._confirm(
            "Advance gradual training?",
            f"This advances stage by stage up to '{up_to.value}', stopping at the first "
            "stage the evaluation does not accept. It is rollback-safe: the base model "
            "is never modified in place.",
        ):
            return
        self._start_work(
            lambda: self.service.run_gradual(up_to=up_to, approved=True), self._finish_report
        )

    def _on_train_autonomous(self) -> None:
        if not self._confirm(
            "Run the adaptive curriculum?",
            "This advances the training stages on its own: it raises the stage when a "
            "run is accepted, changes the learning rate when a stage plateaus, and "
            "stops at an explicit effort budget. It is rollback-safe: the base model "
            "is never modified in place, and a version is adopted only through the gate.",
        ):
            return
        self._start_work(
            lambda: self.service.run_autonomous(approved=True), self._finish_report
        )

    def _on_curriculum(self) -> None:
        def job() -> tuple[str, ...]:
            return self.service.curriculum_memory()

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("error", str(error))
                return
            attempts = result
            if not attempts:
                self._add_message("note", "No training attempts recorded yet.")
                return
            self._report("assistant", "Adaptive curriculum's training memory", tuple(attempts))

        self._start_work(job, finish)

    def _on_progression(self) -> None:
        def job() -> str:
            return self.service.progression().describe()

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("error", str(error))
                return
            self._add_message("assistant", result)

        self._start_work(job, finish)

    def _on_versions(self) -> None:
        def job() -> str:
            return self.service.versions_report()

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("error", str(error))
                return
            self._add_message("assistant", result)

        self._start_work(job, finish)

    def _on_rollback(self) -> None:
        if not self._confirm(
            "Roll back?",
            "This returns the model to the last accepted version. The base model is "
            "never modified, so this is always safe.",
        ):
            return
        self._start_work(self.service.rollback, self._finish_report)

    # -- characters -------------------------------------------------------

    def _on_characters(self) -> None:
        def job() -> EvaluationReport:
            return self.service.evaluate_characters()

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("error", str(error))
                return
            report: EvaluationReport = result
            self._add_message("note", report.describe())
            for character in report.characters:
                lines = [f"{d.title}: {d.percent}%" for d in character.dimensions]
                lines.append("")
                lines.append("Strengths:")
                lines.extend(f"  + {s}" for s in character.strengths or ("(none observed yet)",))
                lines.append("Shortcomings:")
                lines.extend(f"  - {s}" for s in character.shortcomings or ("(none)",))
                lines.append("Next steps:")
                lines.extend(f"  → {s}" for s in character.next_steps or ("(none)",))
                flag = "  (private records)" if character.private else ""
                self._add_message(
                    "assistant",
                    f"{character.name} — overall {round(character.overall * 100)}%{flag}\n"
                    + "\n".join(lines),
                )

        self._start_work(job, finish)

    # -- indicators -------------------------------------------------------

    def _on_evaluate(self) -> None:
        def job() -> str:
            report = self.service.evaluate()
            lines = [f"overall {report.overall:.3f}"]
            for key in INDICATORS:
                note = report.notes.get(key, "")
                suffix = f"  — {note}" if note else ""
                lines.append(f"{LABELS[key]}: {report.scores[key]:.3f}{suffix}")
            return "\n".join(lines)

        def finish(result: Any, error: Exception | None) -> None:
            if error is not None:
                self._add_message("error", str(error))
                return
            self._add_message("assistant", f"Cognitive indicators\n{result}")

        self._start_work(job, finish)

    # -- system -----------------------------------------------------------

    def _on_mode(self) -> None:
        mode = ResourceMode(self._mode_var.get())
        self._start_work(lambda: self.service.set_mode(mode), self._finish_report)

    def _on_tick(self) -> None:
        self._start_work(self.service.tick, self._finish_report)

    # -- command line -----------------------------------------------------

    def _on_command(self) -> None:
        text = self._command_var.get().strip()
        if not text:
            return
        self._add_message("user", text)
        self._command_var.set("")
        self._start_work(lambda: self.service.teach(text), self._finish_report)

    # -- confirmation -----------------------------------------------------

    def _confirm(self, title: str, message: str) -> bool:
        """A governed action asks first. Returns True only on an explicit yes."""
        from tkinter import messagebox

        try:
            return bool(messagebox.askyesno(title, message, parent=self.window))
        except tk.TclError:
            return False

    # -- lifecycle --------------------------------------------------------

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pump_after is not None:
            try:
                self.window.after_cancel(self._pump_after)
            except tk.TclError:
                pass
            self._pump_after = None
        if self._turn_pump is not None:
            try:
                self.window.after_cancel(self._turn_pump)
            except tk.TclError:
                pass
            self._turn_pump = None
        try:
            self.window.destroy()
        finally:
            if self._on_close is not None:
                self._on_close()


#: Re-exported so a caller can reach the dimension titles without importing the
#: evaluation engine directly.
__all__ = [
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "DIMENSION_TITLES",
    "MIN_HEIGHT",
    "MIN_WIDTH",
    "SIDEBAR_WIDTH",
    "CognitionDialog",
]
