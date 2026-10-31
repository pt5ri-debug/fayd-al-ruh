"""The Characters window: one page per character, with APPLY and hot reload.

The sidebar's **Characters** section opens this. The left column lists the seven
characters and the project's own model; choosing one shows its **complete, editable
configuration**, organised into the sections the task asks for:

* **General information** -- name, description, role, activation;
* **Personality** -- traits, behaviour, communication and reasoning style,
  language, formality, emotional expression, system instructions, knowledge domain,
  objectives;
* **Language model** -- the assigned model, its parameters, a connection test, and
  an enable/disable switch;
* **Voice settings** -- the TTS engine, the voice, the language, the speed, the
  pitch and volume where the engine supports them, and a test playback;
* **Learning settings** -- learning on/off, learning from the model, from the
  internet, the verification and confidence thresholds, automatic or supervised,
  and the knowledge-sharing permission;
* **Memory and learning management** -- the character's own statistics, its
  history, a search over its records, an export/backup, and a reset that must be
  explicitly confirmed.

A clearly visible **APPLY** button validates the whole draft, saves it, activates
what it can now, and **rolls back** on failure -- so a bad edit can never leave the
character half-configured. What cannot be applied live (a model or voice swap) is
named as needing a restart rather than pretended.

The window is built the way every other window in the project is built: the same
:class:`~gui.app.Theme` tokens, the shared :class:`~gui.chat_composer.ChatComposer`
for the search box, the shared :class:`~gui.window_controls.WindowControls` trio, a
:class:`~gui.sidebar.SidebarColumn` on the left, and a worker thread whose result
is handed back through a queue the UI thread drains. The logic lives in
:class:`gui.character_manager.CharacterManagerService`, so the whole workflow is
tested with no display; this module is only the view.

Tk is imported lazily with the rest of the window, so the CLI never touches it.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from tkinter import messagebox
from typing import Any

from core.character_config import (
    CharacterConfig,
    ConfigError,
)
from core.learning_isolation import MAIN_MODEL_OWNER, is_private

from gui.app import RoundedBox
from gui.character_manager import CharacterManagerService
from gui.sidebar import SidebarColumn
from gui.typography import resolve
from gui.window_controls import fit_geometry, install_window_controls

#: The transcript reads as a centred column, capped like the chat window's.
_COLUMN_MAX = 900

#: The sidebar is this wide; the settings column takes the rest.
SIDEBAR_WIDTH = 300

DEFAULT_WIDTH = 1200
DEFAULT_HEIGHT = 760
MIN_WIDTH = 980
MIN_HEIGHT = 640


class CharacterManagerDialog:
    """The Characters window: pick a character, edit it, APPLY."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        service: CharacterManagerService,
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
        # The editor's live variables, rebuilt when a character is chosen.
        self._vars: dict[str, tk.Variable] = {}
        self._current = ""
        self._dirty = False
        self._results: queue.Queue[tuple[Callable[[Any, Exception | None], None], Any, Exception | None]] = (
            queue.Queue()
        )

        self.window = tk.Toplevel(parent)
        self.window.title("Characters — Fayd Al-Ruh")
        self.window.configure(bg=theme.base)
        self.window.geometry(self._initial_geometry())
        self.window.minsize(MIN_WIDTH, MIN_HEIGHT)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)

        self._build()
        self._select(self.service.keys()[0])

    # -- layout -----------------------------------------------------------

    def _initial_geometry(self) -> str:
        return fit_geometry(self.window, DEFAULT_WIDTH, DEFAULT_HEIGHT, min_height=MIN_HEIGHT)

    def _build(self) -> None:
        self._build_header()
        theme = self.theme
        body = tk.Frame(self.window, bg=theme.base)
        body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=0, minsize=SIDEBAR_WIDTH)
        body.columnconfigure(1, weight=1)
        self._build_sidebar(body)
        self._build_editor(body)

    def _build_header(self) -> None:
        theme = self.theme
        bar = tk.Frame(self.window, bg=theme.titlebar)
        bar.pack(fill="x")
        inner = tk.Frame(bar, bg=theme.titlebar)
        inner.pack(fill="x", padx=16, pady=(12, 8))
        titles = tk.Frame(inner, bg=theme.titlebar)
        titles.pack(side="left", anchor="w")
        tk.Label(
            titles, text="Characters", bg=theme.titlebar, fg=theme.contrast,
            font=self._type.font("heading3", weight="bold"), anchor="w",
        ).pack(anchor="w")
        tk.Label(
            titles,
            text="Seven characters and the project's own model — each with its own "
            "personality, model, voice, and isolated learning store.",
            bg=theme.titlebar, fg=theme.text_secondary, font=self._type.font("2xs"),
            anchor="w", justify="left", wraplength=820,
        ).pack(anchor="w", pady=(2, 0))
        self._window_controls = install_window_controls(
            self.window, inner, theme=theme, typography=self._type, on_close=self._close
        )
        tk.Frame(self.window, bg=theme.titlebar_edge, height=1).pack(fill="x")

    def _button(
        self, parent: tk.Misc, text: str, command: Any, *, small: bool = False, primary: bool = False
    ) -> tk.Button:
        theme = self.theme
        background = theme.primary if primary else theme.surface_raised
        foreground = theme.primary_foreground if primary else theme.foreground
        return tk.Button(
            parent, text=text, command=command, relief="flat", borderwidth=0,
            highlightthickness=0, bg=background, fg=foreground,
            activebackground=theme.interactive_active, activeforeground=theme.contrast,
            font=self._type.font("2xs", weight="bold"), padx=10, pady=4 if small else 7,
            cursor="hand2", anchor="w" if small else "center",
        )

    # -- sidebar (the character list) -------------------------------------

    def _build_sidebar(self, parent: tk.Frame) -> None:
        column = SidebarColumn(
            parent,
            theme=self.theme,
            typography=self._type,
            width=SIDEBAR_WIDTH,
            scope="characters",
            state_root=getattr(self.service, "_data_root", None),
        )
        self._sidebar_column = column
        self._sidebar = column.body
        self._sidebar_canvas = column.canvas
        self._sidebar_scrollbar = column.scrollbar

        picker = column.add("characters", "Characters", icon="🧑")
        self._picker_host = tk.Frame(picker, bg=self.theme.surface)
        self._picker_host.pack(fill="x", padx=10, pady=(2, 6))
        self._character_buttons: dict[str, tk.Button] = {}
        for row in self.service.summaries():
            label = row.name
            if row.private:
                label += "  🔒"
            if row.key == MAIN_MODEL_OWNER:
                label += "  (main)"
            button = self._button(
                self._picker_host, label, partial(self._select, row.key), small=True
            )
            button.pack(fill="x", pady=1)
            self._character_buttons[row.key] = button

        # The knowledge-sharing policy is stated, not hidden: the main model may
        # learn from six characters, and never from Natasha.
        sharing = column.add("sharing", "Knowledge sharing", icon="🔗")
        host = tk.Frame(sharing, bg=self.theme.surface)
        host.pack(fill="x", padx=12, pady=(2, 6))
        for line in self._sharing_lines():
            tk.Label(
                host, text=line, bg=self.theme.surface, fg=self.theme.text_secondary,
                font=self._type.font("2xs"), anchor="w", justify="left", wraplength=SIDEBAR_WIDTH - 40,
            ).pack(anchor="w", pady=1)

    def _sharing_lines(self) -> tuple[str, ...]:
        shareable = ", ".join(self.service.shareable_characters())
        private = ", ".join(self.service.private_characters())
        return (
            "The main model may learn from:",
            f"  {shareable}",
            "Never shared:",
            f"  {private}",
        )

    # -- editor -----------------------------------------------------------

    def _build_editor(self, parent: tk.Frame) -> None:
        theme = self.theme
        col = tk.Frame(parent, bg=theme.base)
        col.grid(row=0, column=1, sticky="nsew")

        self._build_action_bar(col)

        container = tk.Frame(col, bg=theme.base)
        container.pack(fill="both", expand=True)
        self._canvas = tk.Canvas(container, bg=theme.base, highlightthickness=0)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._scrollbar = tk.Scrollbar(container, command=self._canvas.yview)
        self._scrollbar.pack(side="right", fill="y")
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        self._form = tk.Frame(self._canvas, bg=theme.base)
        self._window = self._canvas.create_window((0, 0), window=self._form, anchor="nw")
        self._form.bind("<Configure>", self._on_form_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        for widget in (self._canvas, self._form):
            widget.bind("<MouseWheel>", self._on_mousewheel)
            widget.bind("<Button-4>", self._on_mousewheel)
            widget.bind("<Button-5>", self._on_mousewheel)

    def _build_action_bar(self, parent: tk.Frame) -> None:
        """The title line plus the clearly visible APPLY button."""
        theme = self.theme
        bar = tk.Frame(parent, bg=theme.base)
        bar.pack(fill="x", padx=18, pady=(12, 4))
        titles = tk.Frame(bar, bg=theme.base)
        titles.pack(side="left", anchor="w")
        self._title = tk.Label(
            titles, text="", bg=theme.base, fg=theme.contrast,
            font=self._type.font("heading4", weight="bold"), anchor="w",
        )
        self._title.pack(anchor="w")
        self._subtitle = tk.Label(
            titles, text="", bg=theme.base, fg=theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=700,
        )
        self._subtitle.pack(anchor="w", pady=(1, 0))

        self._apply_button = self._button(bar, "APPLY", self._on_apply, primary=True)
        self._apply_button.pack(side="right")
        self._reset_button = self._button(bar, "Reset to default", self._on_reset, small=True)
        self._reset_button.pack(side="right", padx=(0, 8))

        self._status = tk.Label(
            parent, text="", bg=theme.base, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=820,
        )
        self._status.pack(fill="x", padx=18, pady=(0, 6))

    # -- selecting a character --------------------------------------------

    def _select(self, key: str) -> None:
        self._current = key
        self._dirty = False
        for owner, button in self._character_buttons.items():
            button.configure(
                bg=self.theme.primary if owner == key else self.theme.surface_raised,
                fg=self.theme.primary_foreground if owner == key else self.theme.foreground,
            )
        self._rebuild_form()

    def _rebuild_form(self) -> None:
        for child in self._form.winfo_children():
            child.destroy()
        self._vars.clear()
        config = self.service.config(self._current)
        self._title.configure(text=config.general.name)
        summary = self.service.summary(self._current)
        self._subtitle.configure(
            text=f"{config.general.role or '(no role)'} — {summary.records} learning record(s)"
        )
        self._status.configure(text=config.describe())
        self._form.columnconfigure(0, weight=1)

        self._build_general(config)
        self._build_personality(config)
        self._build_model(config)
        self._build_voice(config)
        self._build_learning(config)
        self._build_management()

    def _section(self, title: str) -> tk.Frame:
        """A titled card the caller fills with rows."""
        theme = self.theme
        card = RoundedBox(
            self._form, bg=theme.base, fill=theme.surface, outline=theme.border_subtle,
            corner=12, parent_bg=theme.base,
        )
        card.pack(fill="x", padx=18, pady=(6, 2))
        tk.Label(
            card.inner, text=title, bg=theme.surface, fg=theme.contrast,
            font=self._type.font("sm", weight="bold"), anchor="w",
        ).pack(anchor="w", padx=14, pady=(10, 2))
        body = tk.Frame(card.inner, bg=theme.surface)
        body.pack(fill="x", padx=14, pady=(0, 12))
        return body

    def _row(self, parent: tk.Frame, label: str) -> tk.Frame:
        theme = self.theme
        row = tk.Frame(parent, bg=theme.surface)
        row.pack(fill="x", pady=2)
        tk.Label(
            row, text=label, bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs"), width=22, anchor="w",
        ).pack(side="left")
        return row

    def _entry(self, parent: tk.Frame, label: str, key: str, value: str) -> tk.Entry:
        theme = self.theme
        row = self._row(parent, label)
        var = tk.StringVar(value=value)
        self._vars[key] = var
        entry = tk.Entry(
            row, textvariable=var, bg=theme.base, fg=theme.foreground,
            insertbackground=theme.foreground, relief="flat", highlightthickness=0,
            font=self._type.font("body"),
        )
        entry.pack(side="left", fill="x", expand=True, ipady=3)
        var.trace_add("write", lambda *_: self._mark_dirty())
        return entry

    def _text(self, parent: tk.Frame, label: str, key: str, value: str, *, height: int = 3) -> tk.Text:
        theme = self.theme
        row = self._row(parent, label)
        widget = tk.Text(
            row, height=height, wrap="word", relief="flat", borderwidth=0,
            highlightthickness=0, bg=theme.base, fg=theme.foreground,
            font=self._type.font("body"), padx=6, pady=3,
        )
        widget.insert("1.0", value)
        widget.pack(side="left", fill="x", expand=True)
        self._vars[key] = widget  # type: ignore[assignment]
        widget.bind("<KeyRelease>", lambda _e: self._mark_dirty())
        return widget

    def _check(self, parent: tk.Frame, label: str, key: str, value: bool) -> None:
        theme = self.theme
        var = tk.BooleanVar(value=value)
        self._vars[key] = var
        box = tk.Checkbutton(
            parent, text=label, variable=var, bg=theme.surface, fg=theme.foreground,
            activebackground=theme.surface, activeforeground=theme.contrast,
            selectcolor=theme.base, font=self._type.font("2xs"), anchor="w",
            highlightthickness=0, borderwidth=0,
        )
        box.pack(anchor="w", pady=1)
        var.trace_add("write", lambda *_: self._mark_dirty())

    def _choice(
        self, parent: tk.Frame, label: str, key: str, value: str, options: tuple[str, ...]
    ) -> None:
        theme = self.theme
        row = self._row(parent, label)
        var = tk.StringVar(value=value or (options[0] if options else ""))
        self._vars[key] = var
        menu = tk.OptionMenu(row, var, *(options or ("",)))
        menu.configure(
            bg=theme.surface_raised, fg=theme.foreground, activebackground=theme.interactive_active,
            activeforeground=theme.contrast, relief="flat", highlightthickness=0,
            font=self._type.font("2xs"), anchor="w",
        )
        menu["menu"].configure(bg=theme.surface_raised, fg=theme.foreground, font=self._type.font("2xs"))
        menu.pack(side="left", fill="x", expand=True)
        var.trace_add("write", lambda *_: self._mark_dirty())

    # -- sections ---------------------------------------------------------

    def _build_general(self, config: CharacterConfig) -> None:
        body = self._section("A. General information")
        self._entry(body, "Name", "general.name", config.general.name)
        self._text(body, "Description", "general.description", config.general.description, height=2)
        self._entry(body, "Role", "general.role", config.general.role)
        self._check(body, "Active", "general.active", config.general.active)

    def _build_personality(self, config: CharacterConfig) -> None:
        body = self._section("B. Personality")
        personality = config.personality
        self._text(body, "Personality traits (one per line)", "personality.traits", "\n".join(personality.traits))
        self._text(
            body, "Behaviour (one per line)", "personality.behaviour", "\n".join(personality.behaviour)
        )
        self._text(body, "Rules (one per line)", "personality.rules", "\n".join(personality.rules))
        self._entry(body, "Communication style", "personality.communication_style", personality.communication_style)
        self._entry(body, "Reasoning style", "personality.reasoning_style", personality.reasoning_style)
        self._choice(
            body, "Preferred language", "personality.preferred_language",
            personality.preferred_language, self.service.languages(),
        )
        self._entry(body, "Formality (0..1)", "personality.formality", str(personality.formality))
        self._entry(
            body, "Emotional expression (0..1)", "personality.emotional_expression",
            str(personality.emotional_expression),
        )
        self._entry(body, "Preferred answer length", "personality.response_length", personality.response_length)
        self._text(body, "System instructions", "personality.instructions", personality.instructions, height=2)
        self._entry(body, "Knowledge domain", "personality.knowledge_domain", personality.knowledge_domain)
        self._text(body, "Objectives (one per line)", "personality.objectives", "\n".join(personality.objectives))

    def _build_model(self, config: CharacterConfig) -> None:
        body = self._section("C. Language model")
        choices = self.service.model_choices(self._current)
        keys = tuple(key for key, _title, _ok in choices) or ("",)
        self._choice(body, "Assigned model", "model.key", config.model.key, keys)
        for key, title, available in choices[:6]:
            mark = "available" if available else "not downloaded"
            tk.Label(
                body, text=f"    {key} — {title} ({mark})", bg=self.theme.surface,
                fg=self.theme.text_secondary, font=self._type.font("2xs"), anchor="w",
            ).pack(anchor="w")
        self._entry(body, "Provider", "model.provider", config.model.provider)
        self._entry(body, "Model identifier / path", "model.model", config.model.model)
        self._entry(body, "Temperature", "model.temperature", str(config.model.temperature))
        self._entry(body, "Context length", "model.context_length", str(config.model.context_length))
        self._entry(body, "Max output tokens", "model.max_tokens", str(config.model.max_tokens))
        self._check(body, "Model enabled", "model.active", config.model.active)
        row = tk.Frame(body, bg=self.theme.surface)
        row.pack(fill="x", pady=(4, 0))
        self._button(row, "Test the model connection", self._on_test_model, small=True).pack(anchor="w")

    def _build_voice(self, config: CharacterConfig) -> None:
        body = self._section("D. Voice settings")
        voices = self.service.voice_choices()
        keys = tuple(key for key, _label, _source in voices) or ("",)
        self._choice(
            body, "TTS engine", "voice.engine", config.voice.engine,
            ("", *self._engine_keys()),
        )
        self._choice(body, "Voice model / voice", "voice.voice", config.voice.voice, keys)
        self._entry(body, "Language", "voice.language", config.voice.language)
        self._entry(body, "Speaking speed", "voice.speed", str(config.voice.speed))
        self._entry(body, "Pitch (if supported)", "voice.pitch", str(config.voice.pitch))
        self._entry(body, "Volume", "voice.volume", str(config.voice.volume))
        self._check(body, "Voice enabled", "voice.active", config.voice.active)
        note = self.service.voice_support(config.voice.engine) if config.voice.engine else ""
        if note:
            tk.Label(
                body, text=note, bg=self.theme.surface, fg=self.theme.text_secondary,
                font=self._type.font("2xs"), anchor="w", justify="left", wraplength=680,
            ).pack(anchor="w", pady=(2, 0))
        row = tk.Frame(body, bg=self.theme.surface)
        row.pack(fill="x", pady=(4, 0))
        self._button(row, "Test the voice", self._on_test_voice, small=True).pack(anchor="w")

    def _engine_keys(self) -> tuple[str, ...]:
        return tuple(key for key, _title in self.service.voice_sources())

    def _build_learning(self, config: CharacterConfig) -> None:
        body = self._section("E. Learning settings")
        learning = config.learning
        self._check(body, "Learning enabled", "learning.enabled", learning.enabled)
        self._check(
            body, "Learn from the character's language model", "learning.learn_from_model",
            learning.learn_from_model,
        )
        self._check(
            body, "Learn from the internet", "learning.learn_from_internet",
            learning.learn_from_internet,
        )
        self._entry(
            body, "Verification threshold", "learning.verification_threshold",
            str(learning.verification_threshold),
        )
        self._entry(
            body, "Confidence threshold", "learning.confidence_threshold",
            str(learning.confidence_threshold),
        )
        self._choice(body, "Mode", "learning.mode", learning.mode, self.service.learning_modes())
        if self._current == MAIN_MODEL_OWNER:
            tk.Label(
                body, text="    (the main model's own learning; sharing is not applicable)",
                bg=self.theme.surface, fg=self.theme.text_secondary, font=self._type.font("2xs"),
                anchor="w",
            ).pack(anchor="w")
        elif is_private(self._current):
            tk.Label(
                body, text=f"    {self.service.share_reason(self._current)}",
                bg=self.theme.surface, fg=self.theme.text_secondary, font=self._type.font("2xs"),
                anchor="w", justify="left", wraplength=680,
            ).pack(anchor="w")
        else:
            self._check(
                body, "Share knowledge with the main model", "learning.share_with_main_model",
                learning.share_with_main_model,
            )

    def _build_management(self) -> None:
        body = self._section("F. Memory and learning management")
        stats = self.service.stats(self._current)
        tk.Label(
            body, text=stats.summary(), bg=self.theme.surface, fg=self.theme.foreground,
            font=self._type.font("2xs"), anchor="w", justify="left",
        ).pack(anchor="w", pady=(0, 4))

        # A fill indicator: how much of the character's own memory is in use. It is
        # read from the character's real facts and exchanges, so it never overstates
        # what it holds.
        facts, exchanges, ratio = self.service.memory_fill(self._current)
        tk.Label(
            body,
            text=f"Memory fill: {round(ratio * 100)}% ({facts} fact(s), {exchanges} exchange(s))",
            bg=self.theme.surface, fg=self.theme.text_secondary,
            font=self._type.font("2xs"), anchor="w",
        ).pack(anchor="w", pady=(0, 4))

        search_row = tk.Frame(body, bg=self.theme.surface)
        search_row.pack(fill="x", pady=2)
        self._search_var = tk.StringVar(value="")
        tk.Entry(
            search_row, textvariable=self._search_var, bg=self.theme.base, fg=self.theme.foreground,
            insertbackground=self.theme.foreground, relief="flat", highlightthickness=0,
            font=self._type.font("body"),
        ).pack(side="left", fill="x", expand=True, ipady=3)
        self._button(search_row, "Search", self._on_search, small=True).pack(side="left", padx=(6, 0))

        # What this character *remembers* (the durable facts it carries into its
        # next answer) -- distinct from the learning records below, which are the
        # audit trail of what it answered. This is where the user teaches a fact.
        tk.Label(
            body, text="What this character remembers", bg=self.theme.surface,
            fg=self.theme.text_secondary, font=self._type.font("2xs"), anchor="w",
        ).pack(anchor="w", pady=(2, 0))
        self._facts_host = tk.Frame(body, bg=self.theme.surface)
        self._facts_host.pack(fill="x", pady=(0, 2))
        remember_row = tk.Frame(body, bg=self.theme.surface)
        remember_row.pack(fill="x", pady=2)
        self._fact_var = tk.StringVar(value="")
        tk.Entry(
            remember_row, textvariable=self._fact_var, bg=self.theme.base,
            fg=self.theme.foreground, insertbackground=self.theme.foreground, relief="flat",
            highlightthickness=0, font=self._type.font("body"),
        ).pack(side="left", fill="x", expand=True, ipady=3)
        self._button(remember_row, "Remember", self._on_remember, small=True).pack(
            side="left", padx=(6, 0)
        )
        self._button(remember_row, "Forget all", self._on_forget_facts, small=True).pack(
            side="left", padx=(6, 0)
        )
        self._button(
            remember_row, "Seed knowledge", self._on_seed, small=True
        ).pack(side="left", padx=(6, 0))
        self._show_facts(self.service.facts(self._current))

        self._records_host = tk.Frame(body, bg=self.theme.surface)
        self._records_host.pack(fill="x", pady=(4, 4))
        self._show_records(self.service.records(self._current, limit=20))

        actions = tk.Frame(body, bg=self.theme.surface)
        actions.pack(fill="x", pady=(2, 0))
        self._button(actions, "Export / backup…", self._on_export, small=True).pack(side="left")
        self._button(actions, "Reset learning data…", self._on_reset_learning, small=True).pack(
            side="left", padx=(6, 0)
        )

    def _show_facts(self, facts: Any) -> None:
        for child in self._facts_host.winfo_children():
            child.destroy()
        if not facts:
            tk.Label(
                self._facts_host, text="Nothing yet — teach it a fact below.", bg=self.theme.surface,
                fg=self.theme.text_secondary, font=self._type.font("2xs"), anchor="w",
            ).pack(anchor="w")
            return
        for fact in facts:
            tk.Label(
                self._facts_host, text=f"- {fact}", bg=self.theme.surface, fg=self.theme.foreground,
                font=self._type.font("2xs"), anchor="w", justify="left", wraplength=760,
            ).pack(anchor="w", pady=1)

    def _on_remember(self) -> None:
        """Teach the chosen character one durable fact."""
        text = self._fact_var.get().strip()
        if not text:
            self._status.configure(text="Type a fact to remember first.")
            return
        try:
            stored = self.service.remember_fact(self._current, text)
        except (ValueError, OSError) as error:
            self._status.configure(text=f"Could not remember that: {error}")
            return
        self._fact_var.set("")
        self._show_facts(self.service.facts(self._current))
        if stored:
            self._status.configure(text=f"{self._current} will remember that.")
        else:
            self._status.configure(text=f"{self._current} already knew that.")

    def _on_forget_facts(self) -> None:
        """Clear the chosen character's memory (facts and exchanges)."""
        self.service.forget_facts(self._current, confirm=True)
        self._show_facts(self.service.facts(self._current))
        self._status.configure(text=f"{self._current}'s memory was cleared.")

    def _on_seed(self) -> None:
        """Seed the character's memory from the project's own authored knowledge."""
        try:
            outcome = self.service.seed(self._current, limit=40, conversations=True)
        except (ValueError, OSError) as error:
            self._status.configure(text=f"Could not seed: {error}")
            return
        self._show_facts(self.service.facts(self._current))
        added = getattr(outcome, "added", 0)
        self._status.configure(
            text=f"{self._current}: +{added} item(s) seeded from the project's knowledge."
        )

    def _show_records(self, records: Any) -> None:
        for child in self._records_host.winfo_children():
            child.destroy()
        if not records:
            tk.Label(
                self._records_host, text="No learning records yet.", bg=self.theme.surface,
                fg=self.theme.text_secondary, font=self._type.font("2xs"), anchor="w",
            ).pack(anchor="w")
            return
        for record in records:
            tk.Label(
                self._records_host, text=self.service.describe_record(record), bg=self.theme.surface,
                fg=self.theme.foreground, font=self._type.font("2xs"), anchor="w",
                justify="left", wraplength=760,
            ).pack(anchor="w", pady=1)

    # -- reading the form -------------------------------------------------

    def _mark_dirty(self) -> None:
        self._dirty = True

    def _collect(self) -> CharacterConfig:
        """Read the form into a configuration, or raise :class:`ConfigError`."""
        config = self.service.config(self._current)
        v = self._vars

        def text(key: str) -> str:
            widget = v[key]
            if isinstance(widget, tk.Text):
                return widget.get("1.0", "end").strip()
            return str(widget.get()).strip()

        def lines(key: str) -> tuple[str, ...]:
            return tuple(line.strip() for line in text(key).splitlines() if line.strip())

        general = replace(
            config.general,
            name=text("general.name"),
            description=text("general.description"),
            role=text("general.role"),
            active=bool(v["general.active"].get()),
        )
        personality = replace(
            config.personality,
            traits=lines("personality.traits"),
            behaviour=lines("personality.behaviour"),
            rules=lines("personality.rules"),
            communication_style=text("personality.communication_style"),
            reasoning_style=text("personality.reasoning_style"),
            preferred_language=text("personality.preferred_language"),
            formality=_number(text("personality.formality"), "formality"),
            emotional_expression=_number(text("personality.emotional_expression"), "emotional expression"),
            response_length=text("personality.response_length"),
            instructions=text("personality.instructions"),
            knowledge_domain=text("personality.knowledge_domain"),
            objectives=lines("personality.objectives"),
        )
        model = replace(
            config.model,
            key=text("model.key"),
            provider=text("model.provider"),
            model=text("model.model"),
            temperature=_number(text("model.temperature"), "temperature"),
            context_length=int(_number(text("model.context_length"), "context length")),
            max_tokens=int(_number(text("model.max_tokens"), "max tokens")),
            active=bool(v["model.active"].get()),
        )
        voice = replace(
            config.voice,
            engine=text("voice.engine"),
            voice=text("voice.voice"),
            language=text("voice.language"),
            speed=_number(text("voice.speed"), "speed"),
            pitch=_number(text("voice.pitch"), "pitch"),
            volume=_number(text("voice.volume"), "volume"),
            active=bool(v["voice.active"].get()),
        )
        learning = replace(
            config.learning,
            enabled=bool(v["learning.enabled"].get()),
            learn_from_model=bool(v["learning.learn_from_model"].get()),
            learn_from_internet=bool(v["learning.learn_from_internet"].get()),
            mode=text("learning.mode"),
            verification_threshold=_number(text("learning.verification_threshold"), "verification threshold"),
            confidence_threshold=_number(text("learning.confidence_threshold"), "confidence threshold"),
            share_with_main_model=(
                bool(v["learning.share_with_main_model"].get())
                if "learning.share_with_main_model" in v
                else config.learning.share_with_main_model
            ),
        )
        return replace(
            config,
            general=general,
            personality=personality,
            model=model,
            voice=voice,
            learning=learning,
        )

    # -- actions ----------------------------------------------------------

    def _on_apply(self) -> None:
        """Validate, save, activate -- and roll back on failure."""
        try:
            draft = self._collect()
        except (ConfigError, ValueError) as error:
            self._status.configure(text=f"Not applied: {error}")
            return
        result = self.service.apply(draft)
        if result.ok:
            self._dirty = False
            # Rebuild the page so it shows the saved values, then restore the APPLY
            # report -- which names what activated now and what needs a restart --
            # because that confirmation is the point of pressing APPLY.
            report = result.describe()
            self._rebuild_form()
            self._status.configure(text=report)
        else:
            self._status.configure(text=result.describe())

    def _on_reset(self) -> None:
        self.service.reset(self._current)
        self._rebuild_form()
        self._status.configure(text=f"{self._current} is back to the shipped default.")

    def _on_test_model(self) -> None:
        """Report what the chosen model is and whether it is available here."""
        try:
            draft = self._collect()
        except (ConfigError, ValueError) as error:
            self._add_note(f"Not tested: {error}")
            return
        choices = {key: (title, ok) for key, title, ok in self.service.model_choices(self._current)}
        key = draft.model.key
        if key and key in choices:
            title, ok = choices[key]
            state = "available" if ok else "not downloaded here"
            self._add_note(f"{title}: {state}.")
        elif key:
            self._add_note(f"{key} is not in this character's model catalog.")
        else:
            self._add_note("The character will use its mouth's own default model.")

    def _on_test_voice(self) -> None:
        """Render a short phrase with the chosen voice, or say why it cannot.

        Playing a phrase really synthesises audio (a neural engine can take a
        second), so it runs on the worker thread like every other slow action in
        the project, and the result is reported on the UI thread.
        """
        from core.reading import TextReader

        try:
            draft = self._collect()
        except (ConfigError, ValueError) as error:
            self._add_note(f"Not tested: {error}")
            return
        reader = TextReader()
        if not reader.available():
            self._add_note("No speech engine is installed here, so the voice cannot be tested yet.")
            return
        name = draft.general.name

        def job() -> str:
            # Apply the whole Voice section -- the voice, its language, and its
            # speed/pitch/volume -- so the test plays what APPLY would really use,
            # not just the voice's timbre.
            from core.character_engine import apply_character_voice

            apply_character_voice(reader, draft)
            reader.speak(f"This is {name} speaking.")
            return f"Played a test phrase with {reader.describe()}."

        self._start_work(job, self._render_note)

    def _render_note(self, result: Any, error: Exception | None) -> None:
        """Report a worker result on the UI thread."""
        if error is not None:
            self._add_note(f"The voice could not be played: {error}")
            return
        self._add_note(str(result))

    def _on_search(self) -> None:
        query = self._search_var.get().strip()
        if not query:
            self._show_records(self.service.records(self._current, limit=20))
            return
        self._show_records(self.service.search(self._current, query, limit=20))

    def _on_export(self) -> None:
        try:
            path = self.service.export(self._current)
        except OSError as error:
            self._add_note(f"Could not export: {error}")
            return
        self._add_note(f"Exported to {path}.")

    def _on_reset_learning(self) -> None:
        if not messagebox.askyesno(
            "Reset learning data",
            f"Forget every learning record for {self._current}? This cannot be undone.",
            parent=self.window,
        ):
            return
        self.service.reset_learning(self._current, confirm=True)
        self._rebuild_form()
        self._status.configure(text=f"{self._current}'s learning records were reset.")

    def _add_note(self, text: str) -> None:
        self._status.configure(text=text)

    # -- geometry helpers -------------------------------------------------

    def _on_form_configure(self, _event: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        delta = -1 * (event.delta // 120) if getattr(event, "delta", 0) else (
            -1 if getattr(event, "num", 0) == 4 else 1
        )
        self._canvas.yview_scroll(delta, "units")

    # -- background work --------------------------------------------------

    def _start_work(
        self, job: Callable[[], Any], finish: Callable[[Any, Exception | None], None]
    ) -> None:
        self._busy = True

        def run() -> None:
            try:
                result = job()
                self._results.put((finish, result, None))
            except Exception as error:
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
        finish(result, error)

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
        try:
            self.window.destroy()
        finally:
            if self._on_close is not None:
                self._on_close()


def _number(value: str, label: str) -> float:
    """Parse a float from a form field, with an honest error for a bad value."""
    try:
        return float(value)
    except ValueError as error:
        raise ConfigError("general", "", f"{label} must be a number, got {value!r}.") from error


__all__ = [
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "MIN_HEIGHT",
    "MIN_WIDTH",
    "SIDEBAR_WIDTH",
    "CharacterManagerDialog",
]
