"""The Romance & Companion Studio window: local companions, dating, roleplay.

Reached from the chat composer's **Companion** control (beside **Workplace** and
**Code Studio**) and from the sidebar, this is a standalone professional studio
page for the project's local AI companions. Everything it shows is a thin view
over :class:`gui.romance.RomanceStudioService`, so the whole workflow is testable
with no display and the CLI (``python -m cli romance``) shares one database with
the window. Tk is imported lazily with the rest of the window, so the CLI never
touches it.

Layout:

* **Left sidebar** (the shared :class:`~gui.sidebar.SidebarColumn`) -- the
  character library, the conversation list, the scenario library, the conversation
  modes, the voice picker, and the safety/local-only controls.
* **Centre** -- the selected character's header, the conversation transcript, the
  message composer, the mode selector, the voice controls, and the live
  relationship indicator.
* **Right panel** -- the character's profile, personality traits, interests, the
  relationship state and its milestones, the important memories, and the voice
  configuration; the panel collapses to give the transcript the full width.

The page is bilingual-ready and states plainly, outside the fiction, that these
are fictional AI companions (the character's own words stay in character).
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gui.dialog_base import ThemedDialog
from gui.romance import RomanceStudioService
from gui.sidebar import SidebarColumn
from gui.typography import resolve
from gui.window_controls import fit_geometry

TITLE = "Romance & Companion Studio"
SUBTITLE = "استوديو الرومانسية والرفقة — local AI companions, dating, and roleplay"

#: The window's opening size, clamped to the desktop's own work area.
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 800
MIN_WIDTH = 1000
MIN_HEIGHT = 660

#: The application's honest disclosure (outside the fiction).
DISCLOSURE = (
    "These are fictional AI companions, not real people. Everything stays on this "
    "machine: no cloud inference and no remote conversation service. Adult content "
    "is off by default and every fictional participant must be an adult (18+)."
)

_COLUMN_MAX = 860
_MAX_BUBBLE_LINES = 40


class RomanceStudioDialog(ThemedDialog):
    """The Romance & Companion Studio page."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        service: RomanceStudioService,
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
        self._list_type = resolve(parent, ui_family=font, mono_family=mono, overrides=list_overrides)
        self._on_close = on_close
        self._closed = False
        self._busy = False
        self._right_open = True
        self._selected_character = ""
        self._selected_conversation = ""
        self._character_ids: list[str] = []
        self._archive_ids: list[str] = []
        self._conversation_ids: list[str] = []
        self._example_names: list[str] = []
        self._library_rows: list[Any] = []
        self._mode_keys: list[str] = []
        self._model_keys: list[str] = []
        self._model_labels: dict[str, str] = {}
        self._voice_options: list[Any] = []
        self._scenario_rows: list[Any] = []
        self._memory_ids: list[str] = []
        self._editors: list[Any] = []
        self._settings_dialog: Any = None
        self._hero_stats: dict[str, tk.Label] = {}
        self._avatar: Any = None
        self._row_frames: list[tuple[tk.Frame, tk.Text]] = []
        self._results: queue.Queue[tuple[Callable[[Any, Exception | None], None], Any, Exception | None]] = queue.Queue()

        self.window = tk.Toplevel(parent)
        self.window.title(TITLE)
        self.window.configure(bg=theme.base)
        self.window.geometry(
            fit_geometry(self.window, DEFAULT_WIDTH, DEFAULT_HEIGHT, min_height=MIN_HEIGHT)
        )
        self.window.minsize(MIN_WIDTH, MIN_HEIGHT)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._themed_build()

    def _after_build(self) -> None:
        self._add_message("note", DISCLOSURE)
        self._refresh_all()

    # -- layout -----------------------------------------------------------

    def _build(self) -> None:
        self._build_header()
        theme = self.theme
        body = tk.Frame(self.window, bg=theme.base)
        body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=0, minsize=320)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, weight=0, minsize=300)
        self._body = body
        self._build_sidebar(body)
        self._build_centre(body)
        self._build_right(body)

    def _build_header(self) -> None:
        from gui.components import DialogHeader

        header = DialogHeader(
            self.window, title=TITLE, subtitle=SUBTITLE, theme=self.theme,
            typography=self._type, on_close=self._close, extra=self._header_extra,
        )
        self._spinner = header.spinner
        self._window_controls = header.controls

    def _header_extra(self, inner: tk.Frame) -> None:
        from gui.components import button

        theme = self.theme

        # The live stat strip sits between the title and the window controls.
        stats = tk.Frame(inner, bg=theme.titlebar)
        stats.pack(side="left", anchor="w", padx=(22, 0))
        self._hero_stats = {}
        for key, label in (("characters", "Companions"), ("conversations", "Chats"),
                           ("memories", "Memories"), ("scenarios", "Scenarios")):
            cell = tk.Frame(stats, bg=theme.titlebar)
            cell.pack(side="left", padx=(0, 16))
            value = tk.Label(cell, text="0", bg=theme.titlebar, fg=theme.primary,
                             font=self._type.font("heading4", weight="bold"), anchor="w")
            value.pack(anchor="w")
            tk.Label(cell, text=label, bg=theme.titlebar, fg=theme.text_dim,
                     font=self._type.font("2xs"), anchor="w").pack(anchor="w")
            self._hero_stats[key] = value

        button(inner, "New companion…", self._create_character, theme=theme,
               typography=self._type, variant="primary", small=True).pack(side="right", padx=(6, 0))
        button(inner, "Settings…", self._open_settings, theme=theme,
               typography=self._type, variant="secondary", small=True).pack(side="right", padx=6)
        button(inner, "Right panel", self._toggle_right, theme=theme,
               typography=self._type, variant="secondary", small=True).pack(side="right", padx=6)

    # -- sidebar ----------------------------------------------------------

    def _build_sidebar(self, parent: tk.Frame) -> None:
        column = SidebarColumn(
            parent, theme=self.theme, typography=self._type, scope="romance", width=330,
        )
        self._sidebar_column = column
        self._sidebar = column.body
        theme = self.theme

        brand = tk.Frame(column.body, bg=theme.surface)
        brand.pack(fill="x", padx=14, pady=(14, 6))
        dot = tk.Canvas(brand, width=30, height=30, bg=theme.surface, highlightthickness=0)
        dot.create_oval(1, 1, 29, 29, fill=theme.primary, outline="")
        dot.create_text(15, 16, text="❤", fill=theme.primary_foreground,
                        font=self._type.font("sm", weight="bold"))
        dot.pack(side="left")
        titles = tk.Frame(brand, bg=theme.surface)
        titles.pack(side="left", padx=(9, 0))
        tk.Label(titles, text="Companion Studio", bg=theme.surface, fg=theme.contrast,
                 font=self._type.font("body", weight="bold"), anchor="w").pack(anchor="w")
        tk.Label(titles, text="Pick a companion and talk", bg=theme.surface, fg=theme.text_dim,
                 font=self._type.font("2xs"), anchor="w").pack(anchor="w")

        # Characters.
        chars = column.add("characters", "My Companions", icon="👤", open=True)
        search_row = tk.Frame(chars, bg=theme.surface)
        search_row.pack(fill="x", pady=(0, 4))
        self._char_search_var = tk.StringVar()
        search = tk.Entry(
            search_row, textvariable=self._char_search_var, relief="flat", borderwidth=1,
            highlightthickness=0, bg=theme.surface_raised, fg=theme.foreground,
            insertbackground=theme.foreground, font=self._type.font("2xs"),
        )
        search.pack(fill="x", ipady=3)
        search.bind("<KeyRelease>", lambda _e: self._refresh_characters())
        self._favorites_only = tk.BooleanVar(value=False)
        tk.Checkbutton(
            chars, text="Favorites only", variable=self._favorites_only, bg=theme.surface,
            fg=theme.text_dim, activebackground=theme.surface, activeforeground=theme.foreground,
            selectcolor=theme.base, font=self._type.font("2xs"), anchor="w",
            highlightthickness=0, borderwidth=0, command=self._refresh_characters,
        ).pack(fill="x")
        self._char_list = self._listbox(chars, height=8, on_select=self._on_character_selected)
        row = tk.Frame(chars, bg=theme.surface)
        row.pack(fill="x", pady=(4, 0))
        self._button(row, "New…", self._create_character, variant="primary")
        self._button(row, "Edit…", self._edit_character)
        self._button(row, "Duplicate", self._duplicate_character)
        row1 = tk.Frame(chars, bg=theme.surface)
        row1.pack(fill="x")
        self._button(row1, "★ Favorite", self._toggle_favorite)
        self._button(row1, "Archive", self._archive_character)
        self._button(row1, "Delete…", self._delete_character, variant="danger")
        row1b = tk.Frame(chars, bg=theme.surface)
        row1b.pack(fill="x")
        self._button(row1b, "Export…", self._export_character)
        self._button(row1b, "Import…", self._import_character)

        # Character Library (ready-made, fully-specified companions).
        discover = column.add("discover", "Character Library", icon="📚", open=True)
        tk.Label(
            discover, text="Ready-made companions — pick an archetype, then add one.",
            bg=theme.surface, fg=theme.text_dim, font=self._type.font("2xs"),
            anchor="w", justify="left", wraplength=290,
        ).pack(fill="x", pady=(0, 4))
        self._library_category_var = tk.StringVar(value="")
        self._library_category_menu = tk.OptionMenu(discover, self._library_category_var, "")
        self._style_option(self._library_category_menu)
        self._library_category_menu.pack(fill="x", pady=2)
        self._library_category_var.trace_add("write", lambda *_: self._refresh_library())
        self._library_search_var = tk.StringVar()
        lib_search = tk.Entry(
            discover, textvariable=self._library_search_var, relief="flat", borderwidth=1,
            highlightthickness=0, bg=theme.surface_raised, fg=theme.foreground,
            insertbackground=theme.foreground, font=self._type.font("2xs"),
        )
        lib_search.pack(fill="x", ipady=3, pady=(2, 4))
        lib_search.bind("<KeyRelease>", lambda _e: self._refresh_library())
        self._example_list = self._listbox(discover, height=7, on_select=self._on_library_selected)
        self._library_preview = tk.Label(
            discover, text="", bg=theme.surface_raised, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=290,
            padx=8, pady=6,
        )
        self._library_preview.pack(fill="x", pady=(2, 4))
        lib_row = tk.Frame(discover, bg=theme.surface)
        lib_row.pack(fill="x")
        self._button(lib_row, "Add the chosen", self._add_example, variant="primary")
        self._button(lib_row, "Add the whole group", self._add_library_category)

        archive = column.add("archive", "Archived", icon="🗄")
        self._archive_list = self._listbox(archive, height=4, on_select=self._on_character_selected)
        self._button(archive, "Restore", self._restore_character)

        # Conversations.
        convs = column.add("conversations", "Recent Conversations", icon="💬", open=True)
        self._conv_list = self._listbox(convs, height=6, on_select=self._on_conversation_selected)
        row2 = tk.Frame(convs, bg=theme.surface)
        row2.pack(fill="x", pady=(4, 0))
        self._button(row2, "New chat", self._new_conversation, variant="primary")
        self._button(row2, "Delete chat", self._delete_conversation)

        # Modes.
        modes = column.add("modes", "Conversation Mode", icon="🎭", open=True)
        self._mode_var = tk.StringVar(value="")
        self._mode_menu = tk.OptionMenu(modes, self._mode_var, "")
        self._style_option(self._mode_menu)
        self._mode_menu.pack(fill="x", pady=2)
        self._mode_var.trace_add("write", lambda *_: self._on_mode_changed())
        self._mode_note = tk.Label(
            modes, text="", bg=theme.surface, fg=theme.text_dim, font=self._type.font("2xs"),
            anchor="w", justify="left", wraplength=260,
        )
        self._mode_note.pack(fill="x", pady=(2, 0))

        # Language model (assigned per character).
        model = column.add("model", "Language Model", icon="🧩")
        self._model_note = tk.Label(
            model, text="", bg=theme.surface, fg=theme.text_dim, font=self._type.font("2xs"),
            anchor="w", justify="left", wraplength=260,
        )
        self._model_note.pack(fill="x", pady=(0, 4))
        self._model_var = tk.StringVar(value="")
        self._model_menu = tk.OptionMenu(model, self._model_var, "")
        self._style_option(self._model_menu)
        self._model_menu.pack(fill="x", pady=2)
        self._button(model, "Assign to the character", self._assign_model)

        # Scenarios.
        scenarios = column.add("scenarios", "Dating Scenarios & Roleplay", icon="📖")
        self._scenario_list = self._listbox(scenarios, height=5, on_select=self._on_scenario_selected)
        row_s = tk.Frame(scenarios, bg=theme.surface)
        row_s.pack(fill="x", pady=(4, 0))
        self._button(row_s, "New scenario…", self._create_scenario)
        self._button(row_s, "Start", self._start_scenario, variant="primary")
        row_s2 = tk.Frame(scenarios, bg=theme.surface)
        row_s2.pack(fill="x")
        self._button(row_s2, "Add character", self._add_scenario_participant)

        # Voice.
        voice = column.add("voice", "Voice", icon="🔊")
        self._voice_list = self._listbox(voice, height=5, on_select=self._on_voice_selected)
        row3 = tk.Frame(voice, bg=theme.surface)
        row3.pack(fill="x", pady=(4, 0))
        self._button(row3, "Speak last reply", self._speak_last)
        self._button(row3, "Stop", self._stop_speaking)
        self._voice_engine_var = tk.StringVar(value="auto")
        self._voice_engine_menu = tk.OptionMenu(voice, self._voice_engine_var, "")
        self._style_option(self._voice_engine_menu)
        self._voice_engine_menu.pack(fill="x", pady=(4, 0))
        self._voice_engine_var.trace_add("write", lambda *_: self._on_voice_engine_changed())

        # Settings.
        settings = column.add("settings", "Studio Settings", icon="⚙")
        self._button(settings, "Open settings…", self._open_settings)
        self._button(settings, "Studio summary", self._show_status)

        # Safety.
        safety = column.add("safety", "Privacy & Safety", icon="🔒", open=True)
        self._safety_label = tk.Label(
            safety, text="", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=260,
        )
        self._safety_label.pack(fill="x", pady=2)
        row4 = tk.Frame(safety, bg=theme.surface)
        row4.pack(fill="x", pady=(4, 0))
        self._button(row4, "Adult mode", self._toggle_adult)
        self._button(row4, "Confirm 18+", self._confirm_eligibility)
        self._boundary_entry = tk.Entry(
            safety, relief="flat", borderwidth=1, highlightthickness=0,
            bg=theme.surface_raised, fg=theme.foreground,
            insertbackground=theme.foreground, font=self._type.font("2xs"),
        )
        self._boundary_entry.pack(fill="x", pady=(4, 0), ipady=3)
        row5 = tk.Frame(safety, bg=theme.surface)
        row5.pack(fill="x")
        self._button(row5, "Add boundary", self._add_boundary)

    # -- centre -----------------------------------------------------------

    def _build_centre(self, parent: tk.Frame) -> None:
        theme = self.theme
        col = tk.Frame(parent, bg=theme.base)
        col.grid(row=0, column=1, sticky="nsew")
        self._centre = col

        # Character banner: avatar + name + stage, on a raised card.
        head = tk.Frame(col, bg=theme.surface)
        head.pack(fill="x")
        head_inner = tk.Frame(head, bg=theme.surface)
        head_inner.pack(fill="x", padx=16, pady=10)
        self._avatar = tk.Canvas(head_inner, width=48, height=48, bg=theme.surface, highlightthickness=0)
        self._avatar.pack(side="left")
        titles = tk.Frame(head_inner, bg=theme.surface)
        titles.pack(side="left", padx=(12, 0), anchor="w")
        self._char_title = tk.Label(
            titles, text="No companion selected", bg=theme.surface, fg=theme.contrast,
            font=self._type.font("heading4", weight="bold"), anchor="w",
        )
        self._char_title.pack(anchor="w")
        self._char_subtitle = tk.Label(
            titles, text="Pick a companion from the library on the left, or add one.",
            bg=theme.surface, fg=theme.text_dim, font=self._type.font("2xs"), anchor="w",
            justify="left", wraplength=520,
        )
        self._char_subtitle.pack(anchor="w")
        self._stage_label = tk.Label(
            head_inner, text="", bg=theme.surface, fg=theme.primary,
            font=self._type.font("sm", weight="bold"), anchor="e",
        )
        self._stage_label.pack(side="right")
        tk.Frame(col, bg=theme.border_subtle, height=1).pack(fill="x")

        container = tk.Frame(col, bg=theme.base)
        container.pack(fill="both", expand=True)
        self._canvas = tk.Canvas(container, bg=theme.base, highlightthickness=0)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._scrollbar = tk.Scrollbar(container, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        self._scrollbar.pack(side="right", fill="y")
        self._transcript = tk.Frame(self._canvas, bg=theme.base)
        self._window = self._canvas.create_window((0, 0), window=self._transcript, anchor="nw")
        self._transcript.bind("<Configure>", self._on_transcript_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        for widget in (self._canvas, self._transcript):
            widget.bind("<MouseWheel>", self._on_mousewheel)
            widget.bind("<Button-4>", self._on_mousewheel)
            widget.bind("<Button-5>", self._on_mousewheel)

        # Composer card.
        composer = tk.Frame(col, bg=theme.base)
        composer.pack(fill="x", padx=16, pady=(6, 4))
        card = tk.Frame(composer, bg=theme.surface_raised, highlightthickness=1,
                        highlightbackground=theme.border_subtle)
        card.pack(fill="x")
        self._entry = tk.Text(
            card, height=2, wrap="word", relief="flat", borderwidth=0,
            highlightthickness=0, bg=theme.surface_raised, fg=theme.foreground,
            insertbackground=theme.foreground, font=self._type.font("body"), padx=12, pady=10,
        )
        self._entry.pack(side="left", fill="x", expand=True)
        self._entry.bind("<Return>", self._on_return)
        self._entry.bind("<Shift-Return>", lambda _e: None)
        self._send_button = self._button(composer, "Send ➤", self._send, anchor="center", variant="primary")
        self._send_button.pack(side="left", padx=(8, 0))
        self._status_line = tk.Label(
            col, text="", bg=theme.base, fg=theme.text_dim, font=self._type.font("2xs"), anchor="w",
        )
        self._status_line.pack(fill="x", padx=16, pady=(0, 8))

    # -- right panel ------------------------------------------------------

    def _build_right(self, parent: tk.Frame) -> None:
        theme = self.theme
        panel = tk.Frame(parent, bg=theme.surface)
        panel.grid(row=0, column=2, sticky="nsew")
        self._right_panel = panel
        self._right_column = SidebarColumn(
            panel, theme=self.theme, typography=self._type, scope="romance_info", width=310,
        )

        profile = self._right_column.add("profile", "Profile", icon="🪪", open=True)
        self._profile_text = self._panel_text(profile, height=8)
        self._button(profile, "Edit the character…", self._edit_character)

        traits = self._right_column.add("traits", "Personality & Interests", icon="✨", open=True)
        self._traits_text = self._panel_text(traits, height=8)

        relationship = self._right_column.add("relationship", "Relationship", icon="💞", open=True)
        self._relationship_text = self._panel_text(relationship, height=6)
        self._relationship_bar = tk.Canvas(
            relationship, height=14, bg=theme.surface, highlightthickness=0,
        )
        self._relationship_bar.pack(fill="x", pady=(4, 2))
        rel_row = tk.Frame(relationship, bg=theme.surface)
        rel_row.pack(fill="x", pady=(2, 0))
        self._button(rel_row, "Edit…", self._edit_relationship)
        self._button(rel_row, "Reset", self._reset_relationship)

        milestones = self._right_column.add("milestones", "Milestones", icon="🏁")
        self._milestones_text = self._panel_text(milestones, height=6)

        memories = self._right_column.add("memories", "Important Memories", icon="🧠")
        self._memory_list = self._listbox(memories, height=6, on_select=lambda _i: None)
        mem_row = tk.Frame(memories, bg=theme.surface)
        mem_row.pack(fill="x", pady=(2, 0))
        self._button(mem_row, "Remember…", self._add_memory)
        self._button(mem_row, "Forget", self._forget_memory)
        mem_row2 = tk.Frame(memories, bg=theme.surface)
        mem_row2.pack(fill="x")
        self._button(mem_row2, "Export…", self._export_memories)

        voice = self._right_column.add("voice_config", "Voice Configuration", icon="🎙")
        self._voice_text = self._panel_text(voice, height=5)
        self._button(voice, "Turn voice on/off", self._toggle_voice)

        scenario = self._right_column.add("scenario", "Active Scenario", icon="🎬")
        self._scenario_text = self._panel_text(scenario, height=5)

    def _relationship_closeness(self, snapshot: Any) -> float:
        """A single 0..1 figure for the progress bar: the mean of the dimensions."""
        names = ("familiarity", "trust", "closeness", "comfort", "affection",
                 "shared_experiences")
        values = [float(getattr(snapshot, name, 0.0) or 0.0) for name in names]
        return sum(values) / len(values) if values else 0.0

    def _draw_relationship_bar(self, value: float) -> None:
        """Paint a slim progress bar for the relationship's closeness (0..1)."""
        canvas = getattr(self, "_relationship_bar", None)
        if canvas is None:
            return
        theme = self.theme
        try:
            canvas.delete("all")
            width = max(canvas.winfo_width(), 200)
        except tk.TclError:
            return
        clamped = max(0.0, min(1.0, float(value)))
        canvas.create_rectangle(0, 4, width, 12, fill=theme.surface_raised, outline="")
        filled = int(width * clamped)
        if filled > 0:
            canvas.create_rectangle(0, 4, filled, 12, fill=theme.primary, outline="")

    # -- small widget helpers --------------------------------------------

    def _button(
        self, parent: tk.Misc, text: str, command: Any, *, anchor: str = "w",
        small: bool = True, variant: str = "secondary",
    ) -> tk.Button:
        from gui.components import button

        return button(
            parent, text, command, theme=self.theme, typography=self._type,
            variant=variant, small=small, anchor=anchor,  # type: ignore[arg-type]
        )

    def _listbox(self, parent: tk.Misc, *, height: int, on_select: Callable[[int], None]) -> tk.Listbox:
        theme = self.theme
        frame = tk.Frame(parent, bg=theme.surface)
        frame.pack(fill="x", pady=2)
        widget = tk.Listbox(
            frame, height=height, bg=theme.surface_raised, fg=theme.foreground,
            selectbackground=theme.primary, selectforeground=theme.primary_foreground,
            relief="flat", borderwidth=0, highlightthickness=0,
            font=self._list_type.font("2xs"), activestyle="none",
        )
        widget.pack(side="left", fill="x", expand=True)
        scroll = tk.Scrollbar(frame, command=widget.yview)
        widget.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        widget.bind("<<ListboxSelect>>", lambda _e: self._dispatch_select(widget, on_select))
        return widget

    def _dispatch_select(self, widget: tk.Listbox, on_select: Callable[[int], None]) -> None:
        selection = widget.curselection()
        if selection:
            on_select(int(selection[0]))

    def _style_option(self, menu: tk.OptionMenu) -> None:
        theme = self.theme
        menu.configure(
            bg=theme.surface_raised, fg=theme.foreground, activebackground=theme.interactive_active,
            activeforeground=theme.foreground, relief="flat", highlightthickness=0,
            borderwidth=0, font=self._type.font("2xs"), anchor="w", padx=8,
        )
        menu["menu"].configure(
            bg=theme.surface_raised, fg=theme.foreground, activebackground=theme.primary,
            activeforeground=theme.primary_foreground, font=self._type.font("2xs"),
        )

    def _panel_text(self, parent: tk.Misc, *, height: int) -> tk.Text:
        theme = self.theme
        widget = tk.Text(
            parent, height=height, wrap="word", relief="flat", borderwidth=0, highlightthickness=0,
            bg=theme.surface, fg=theme.foreground, font=self._type.font("2xs"), padx=0, pady=2,
            cursor="arrow",
        )
        widget.pack(fill="x")
        widget.configure(state="disabled")
        return widget

    def _set_panel(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text or "(nothing yet)")
        widget.configure(state="disabled")

    def _character_subtitle(self, character: Any) -> str:
        """A one-line readable summary under the character's name."""
        bits = [
            f"{character.fictional_age}" if character.fictional_age else "",
            character.gender,
            character.nationality,
            " · ".join(character.languages) if character.languages else "",
            character.relationship_style,
        ]
        line = " · ".join(b for b in bits if b)
        return line or "A local companion"

    def _draw_avatar(self, name: str) -> None:
        """Paint a simple circular avatar with the character's initial."""
        canvas = getattr(self, "_avatar", None)
        if canvas is None:
            return
        theme = self.theme
        try:
            canvas.delete("all")
        except tk.TclError:
            return
        canvas.create_oval(2, 2, 46, 46, fill=theme.primary, outline="")
        initial = (name.strip()[:1] or "❤").upper()
        canvas.create_text(24, 25, text=initial, fill=theme.primary_foreground,
                           font=self._type.font("heading4", weight="bold"))

    # -- geometry ---------------------------------------------------------

    def _toggle_right(self) -> None:
        self._right_open = not self._right_open
        if self._right_open:
            self._right_panel.grid(row=0, column=2, sticky="nsew")
            self._body.columnconfigure(2, weight=0, minsize=300)
        else:
            self._right_panel.grid_remove()
            self._body.columnconfigure(2, weight=0, minsize=0)
        self._reflow()

    def _column_pad(self) -> int:
        try:
            width = self._transcript.winfo_width() or self._canvas.winfo_width()
        except (tk.TclError, AttributeError):
            width = _COLUMN_MAX
        return 16 if width <= _COLUMN_MAX else max(16, (width - _COLUMN_MAX) // 2)

    def _on_transcript_configure(self, _event: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)
        self._reflow()

    def _on_mousewheel(self, event: tk.Event) -> None:
        delta = -1 * (event.delta // 120) if getattr(event, "delta", 0) else (
            -1 if getattr(event, "num", 0) == 4 else 1
        )
        self._canvas.yview_scroll(delta, "units")

    def _reflow(self) -> None:
        pad = self._column_pad()
        for row, _widget in self._row_frames:
            try:
                row.pack_configure(padx=(pad, pad))
            except tk.TclError:
                continue

    # -- transcript -------------------------------------------------------

    def _add_message(self, role: str, text: str, *, title: str = "") -> tk.Text:
        theme = self.theme
        pad = self._column_pad()
        is_user = role == "user"
        row = tk.Frame(self._transcript, bg=theme.base)
        row.pack(fill="x", padx=(pad, pad), pady=6)
        head = tk.Frame(row, bg=theme.base)
        head.pack(fill="x", pady=(0, 3))
        label = title or {"user": "You", "note": "Note"}.get(role, "Companion")
        colour = theme.primary if is_user else theme.text_secondary
        tk.Label(
            head, text=label, bg=theme.base, fg=colour,
            font=self._type.font("xs", weight="bold"), anchor="w",
        ).pack(side="right" if is_user else "left")
        widget = tk.Text(
            row, height=1, wrap="word", relief="flat", borderwidth=0, highlightthickness=0,
            bg=theme.user_bubble if is_user else theme.surface,
            fg=theme.user_bubble_fg if is_user else theme.foreground,
            font=self._type.font("body"), padx=12, pady=8, cursor="xterm", takefocus=1,
        )
        widget.insert("1.0", text)
        widget.pack(anchor="e" if is_user else "w", fill="x")
        self._row_frames.append((row, widget))
        try:
            from gui import clipboard

            clipboard.make_read_only(widget)
        except Exception:
            widget.configure(state="disabled")
        self._fit_height(widget)
        self._scroll_to_bottom()
        return widget

    def _fit_height(self, widget: tk.Text) -> None:
        widget.update_idletasks()
        measured = widget.count("1.0", "end", "displaylines")
        lines = int(measured[0]) if measured else 1
        widget.configure(height=min(max(lines, 1), _MAX_BUBBLE_LINES))

    def _scroll_to_bottom(self) -> None:
        self._canvas.update_idletasks()
        self._canvas.yview_moveto(1.0)

    def _clear_transcript(self) -> None:
        for row, _widget in self._row_frames:
            row.destroy()
        self._row_frames.clear()

    # -- refresh ----------------------------------------------------------

    def _refresh_all(self) -> None:
        self._refresh_characters()
        self._refresh_library()
        self._refresh_conversations()
        self._refresh_modes()
        self._refresh_models()
        self._refresh_scenarios()
        self._refresh_voices()
        self._refresh_safety()
        self._refresh_character_panels()
        self._refresh_hero()

    def _refresh_hero(self) -> None:
        """The header's live stat strip, read from the Studio's own counts."""
        try:
            status = self.service.status()
        except Exception:
            return
        values = {
            "characters": status.character_count,
            "conversations": status.conversation_count,
            "memories": status.memory_count,
            "scenarios": status.scenario_count,
        }
        for key, label in getattr(self, "_hero_stats", {}).items():
            try:
                label.configure(text=str(values.get(key, 0)))
            except tk.TclError:
                continue

    def _refresh_characters(self) -> None:
        search = self._char_search_var.get().strip() if hasattr(self, "_char_search_var") else ""
        favorite = True if getattr(self, "_favorites_only", None) is not None and self._favorites_only.get() else None
        self._char_list.delete(0, "end")
        self._character_ids = []
        for row in self.service.character_rows(search=search, favorite=favorite):
            self._character_ids.append(row.character_id)
            self._char_list.insert("end", row.label())
        self._archive_list.delete(0, "end")
        self._archive_ids = []
        for row in self.service.character_rows(include_archived=True):
            if row.archived:
                self._archive_ids.append(row.character_id)
                self._archive_list.insert("end", row.label())
        self._refresh_library()
        self._refresh_hero()

    def _refresh_library(self) -> None:
        """Fill the archetype picker and the ready-made character list."""
        # The archetype menu (rebuilt only when its contents change).
        categories = self.service.library_categories()
        self._library_category_keys = [""] + [key for key, _t, _i, _s in categories]
        labels = ["All archetypes"] + [f"{icon} {title}" for _k, title, icon, _s in categories]
        menu = self._library_category_menu["menu"]
        menu.delete(0, "end")
        for key, label in zip(self._library_category_keys, labels, strict=True):
            menu.add_command(label=label, command=lambda k=key: self._library_category_var.set(k))

        search = self._library_search_var.get().strip() if hasattr(self, "_library_search_var") else ""
        current = self._library_category_var.get()
        rows = self.service.library_rows(category=current, search=search)
        self._library_rows = list(rows)
        self._example_names = [row.name for row in rows]
        self._example_list.delete(0, "end")
        for row in rows:
            self._example_list.insert("end", row.label())
        if not rows:
            self._library_preview.configure(text="No companion matches that filter.")

    def _on_library_selected(self, index: int) -> None:
        rows = getattr(self, "_library_rows", [])
        if index < len(rows):
            self._library_preview.configure(text=rows[index].describe())

    def _add_library_category(self) -> None:
        key = self._library_category_var.get()
        try:
            created = self.service.seed_library_category(key)
        except Exception as error:
            self._set_status(str(error))
            return
        self._refresh_characters()
        scope = "this archetype" if key else "every archetype"
        self._set_status(f"Added {created} ready-made companion(s) from {scope}.")

    def _add_example(self) -> None:
        selection = self._example_list.curselection()
        if not selection or selection[0] >= len(self._example_names):
            self._set_status("Choose a companion from the library to add.")
            return
        name = self._example_names[selection[0]]
        try:
            self.service.add_library_character(name)
        except Exception as error:
            self._set_status(str(error))
            return
        self._refresh_characters()
        self._set_status(f"Added {name} to your companions.")

    def _refresh_conversations(self) -> None:
        self._conv_list.delete(0, "end")
        self._conversation_ids = []
        if self._selected_character:
            rows = self.service.conversations_for(self._selected_character)
        else:
            rows = self.service.recent_conversations(limit=20)
        for row in rows:
            self._conversation_ids.append(row.conversation_id)
            self._conv_list.insert("end", f"{row.title} ({row.message_count})")

    def _refresh_modes(self) -> None:
        modes = self.service.modes()
        self._mode_keys = [key for key, _title, _summary in modes]
        labels = [f"{title}" for _key, title, _summary in modes]
        menu = self._mode_menu["menu"]
        menu.delete(0, "end")
        for key, title in zip(self._mode_keys, labels, strict=True):
            menu.add_command(label=title, command=lambda k=key: self._mode_var.set(k))
        if not self._mode_var.get() and self._mode_keys:
            self._mode_var.set(self._mode_keys[0])
        self._update_mode_note()

    def _update_mode_note(self) -> None:
        mode = self._mode_var.get()
        summary = next((s for k, _t, s in self.service.modes() if k == mode), "")
        self._mode_note.configure(text=summary)

    def _refresh_models(self) -> None:
        """Fill the model picker from the models this machine really has."""
        choices = self.service.model_choices()
        menu = self._model_menu["menu"]
        menu.delete(0, "end")
        self._model_keys = [choice.key for choice in choices]
        self._model_labels = {choice.key: choice.label() for choice in choices}
        for choice in choices:
            menu.add_command(label=choice.label(), command=lambda k=choice.key: self._model_var.set(k))
        if not choices:
            self._model_note.configure(
                text="No language model is installed. Open the Model Manager to download one."
            )
            self._model_var.set("")
            return
        ready = sum(1 for c in choices if c.available)
        self._model_note.configure(text=f"{ready} of {len(choices)} models are installed and ready.")
        # Preselect the character's own model, else the first ready one, else the first.
        current = self.service.model_key(self._selected_character) if self._selected_character else ""
        pick = current or next((c.key for c in choices if c.available), choices[0].key)
        self._model_var.set(self._model_labels.get(pick, pick))

    def _selected_model_key(self) -> str:
        shown = self._model_var.get()
        for key, label in getattr(self, "_model_labels", {}).items():
            if label == shown:
                return key
        return shown if shown in getattr(self, "_model_keys", []) else ""

    def _refresh_scenarios(self) -> None:
        self._scenario_list.delete(0, "end")
        self._scenario_rows = list(self.service.scenarios())
        for row in self._scenario_rows:
            self._scenario_list.insert("end", f"{row.title} [{row.category_label}]")

    def _refresh_voices(self) -> None:
        self._voice_list.delete(0, "end")
        self._voice_options = list(self.service.voices())
        for option in self._voice_options:
            self._voice_list.insert("end", f"{option.label} ({option.source})")
        menu = self._voice_engine_menu["menu"]
        menu.delete(0, "end")
        for engine in self.service.voice_engines():
            menu.add_command(label=engine, command=lambda e=engine: self._voice_engine_var.set(e))
        if self._selected_character:
            try:
                engine = self.service.voice_profile(self._selected_character).engine or "auto"
                self._voice_engine_var.set(engine)
            except Exception:
                pass

    def _refresh_safety(self) -> None:
        status = self.service.safety_status()
        adult = "on" if status.get("adult_mode") else "off"
        local = "local only" if status.get("local_only") else "REMOTE ALLOWED"
        self._safety_label.configure(text=f"Adult mode: {adult}\nInference: {local}")

    def _refresh_character_panels(self) -> None:
        if not self._selected_character:
            for widget in (
                self._profile_text, self._traits_text, self._relationship_text,
                self._milestones_text, self._voice_text, self._scenario_text,
            ):
                self._set_panel(widget, "")
            self._memory_list.delete(0, "end")
            self._char_title.configure(text="No companion selected")
            self._stage_label.configure(text="")
            if hasattr(self, "_char_subtitle"):
                self._char_subtitle.configure(
                    text="Pick a companion from the library on the left, or add one."
                )
            self._draw_avatar("")
            return
        try:
            bundle = self.service.character(self._selected_character)
        except Exception:
            return
        character = bundle.character
        self._char_title.configure(text=character.name)
        if hasattr(self, "_char_subtitle"):
            self._char_subtitle.configure(text=self._character_subtitle(character))
        self._draw_avatar(character.name)
        model_label = ""
        if character.model_key:
            model_label = self._model_labels.get(character.model_key, character.model_key) \
                if hasattr(self, "_model_labels") else character.model_key
        profile = [
            f"Name: {character.name}",
            f"Nickname: {character.nickname}" if character.nickname else "",
            f"Age: {character.fictional_age}" if character.fictional_age else "",
            f"Gender: {character.gender}" if character.gender else "",
            f"Nationality: {character.nationality}" if character.nationality else "",
            f"Languages: {', '.join(character.languages)}" if character.languages else "",
            f"Relationship style: {character.relationship_style}" if character.relationship_style else "",
            f"Mode: {character.default_mode}",
            f"Model: {model_label or '(Studio default)'}",
            "Adult eligible: yes" if character.adult_eligible else "Adult eligible: no",
        ]
        self._set_panel(self._profile_text, "\n".join(p for p in profile if p))
        dials = self.service.personality(self._selected_character)
        traits = [
            f"{name.replace('_', ' ')}: {float(dials.get(name, 0.5)):.2f}"
            for name in (
                "introversion", "playfulness", "warmth", "expressiveness", "confidence",
                "humour", "initiative", "affection", "romantic_intensity", "formality",
            )
        ]
        interests = bundle.interests()
        if interests:
            traits.append("Interests: " + ", ".join(interests))
        boundaries = self.service.boundaries(self._selected_character)
        if boundaries:
            traits.append("Boundaries: " + ", ".join(b.value for b in boundaries))
        self._set_panel(self._traits_text, "\n".join(traits))
        try:
            snapshot = self.service.relationship(self._selected_character)
            self._stage_label.configure(text=snapshot.stage_title)
            self._set_panel(self._relationship_text, snapshot.describe())
            self._draw_relationship_bar(self._relationship_closeness(snapshot))
            events = self.service.relationship_history(self._selected_character)
            self._set_panel(
                self._milestones_text,
                "\n".join(f"• {getattr(e, 'title', '') or e.kind}" for e in events[-12:]),
            )
        except Exception:
            self._set_panel(self._relationship_text, "")
            self._set_panel(self._milestones_text, "")
            self._draw_relationship_bar(0.0)
        try:
            memories = self.service.memories(self._selected_character)
            self._memory_ids = [m.record_id for m in memories]
            self._memory_list.delete(0, "end")
            for memory in memories:
                self._memory_list.insert("end", f"[{memory.kind}] {memory.content}")
        except Exception:
            self._memory_ids = []
            self._memory_list.delete(0, "end")
        try:
            view = self.service.voice_profile(self._selected_character)
            self._set_panel(self._voice_text, view.describe())
        except Exception:
            self._set_panel(self._voice_text, "")

    # -- selection handlers ----------------------------------------------

    def _on_character_selected(self, index: int) -> None:
        if index >= len(self._character_ids):
            return
        self._selected_character = self._character_ids[index]
        self._selected_conversation = ""
        self._clear_transcript()
        self._refresh_conversations()
        self._refresh_models()
        self._refresh_character_panels()
        self._refresh_voices()
        # Start the mode selector on the character's own default mode.
        try:
            default_mode = self.service.character(self._selected_character).character.default_mode
        except Exception:
            default_mode = ""
        if default_mode:
            self._mode_var.set(default_mode)
        self._set_status(f"Selected {self._char_title.cget('text')}.")

    def _on_conversation_selected(self, index: int) -> None:
        if index >= len(self._conversation_ids):
            return
        self._selected_conversation = self._conversation_ids[index]
        self._load_conversation()

    def _on_scenario_selected(self, index: int) -> None:
        rows = getattr(self, "_scenario_rows", [])
        if index < len(rows):
            row = rows[index]
            participants = self.service.scenario_participants(row.scenario_id)
            names = ", ".join(
                self.service.character(p.character_id).name
                for p in participants
                if p.character_id
            )
            self._set_panel(
                self._scenario_text,
                f"{row.title}\nCategory: {row.category_label}\n"
                f"Level: {row.content_level}\nParticipants: {row.participant_count}"
                + (f" ({names})" if names else ""),
            )

    def _on_voice_selected(self, index: int) -> None:
        options = getattr(self, "_voice_options", [])
        if index >= len(options) or not self._selected_character:
            return
        option = options[index]
        try:
            self.service.set_voice(
                self._selected_character, engine=option.engine, voice_key=option.key,
                language=option.language, enabled=True,
            )
            self._refresh_character_panels()
            self._set_status(f"Voice set to {option.label}.")
        except Exception as error:
            self._set_status(f"Could not set the voice: {error}")

    def _on_voice_engine_changed(self) -> None:
        engine = self._voice_engine_var.get()
        if not engine or not self._selected_character:
            return
        try:
            self.service.set_voice(self._selected_character, engine=engine)
            self._refresh_character_panels()
        except Exception as error:
            self._set_status(f"Could not set the voice engine: {error}")

    def _on_mode_changed(self) -> None:
        self._update_mode_note()
        mode = self._mode_var.get()
        if not mode:
            return
        if self._selected_conversation:
            try:
                self.service.set_mode(self._selected_conversation, mode)
                self._set_status(f"Mode set to {mode}.")
            except Exception as error:
                self._set_status(f"Could not change the mode: {error}")

    def _load_conversation(self) -> None:
        self._clear_transcript()
        if not self._selected_conversation:
            return
        try:
            record, turns = self.service.resume(self._selected_conversation)
        except Exception as error:
            self._set_status(f"Could not open the conversation: {error}")
            return
        self._mode_var.set(record.mode)
        for turn in turns:
            role = "user" if turn.role == "user" else "assistant"
            title = "You" if role == "user" else (getattr(turn, "speaker", "") or "Companion")
            self._add_message(role, turn.content, title=title)

    # -- actions ----------------------------------------------------------

    def _new_conversation(self) -> None:
        if not self._selected_character:
            self._set_status("Select a character first.")
            return
        mode = self._mode_var.get() or "general"
        conversation = self.service.start_conversation(self._selected_character, mode=mode)
        self._selected_conversation = conversation.record_id
        self._clear_transcript()
        self._refresh_conversations()
        self._set_status("New conversation started.")

    def _send(self) -> None:
        text = self._entry.get("1.0", "end").strip()
        if not text:
            return
        if not self._selected_character:
            self._set_status("Select a character first.")
            return
        if not self._selected_conversation:
            self._new_conversation()
        self._entry.delete("1.0", "end")
        self._add_message("user", text, title="You")
        mode = self._mode_var.get() or ""
        conversation_id = self._selected_conversation
        self._run_background(
            "Thinking…",
            lambda: self.service.send(conversation_id, text, mode=mode),
            self._on_reply,
        )

    def _on_return(self, event: tk.Event) -> str:
        if int(getattr(event, "state", 0)) & 0x0001:  # Shift held: insert a newline
            return ""
        self._send()
        return "break"

    def _on_reply(self, result: Any, error: Exception | None) -> None:
        if error is not None:
            self._add_message("note", f"The turn failed: {error}")
            self._set_status("The turn failed.")
            return
        if getattr(result, "refused", False):
            self._add_message("note", f"Refused: {result.reason}")
            self._set_status("Refused by the safety floor.")
            return
        if not getattr(result, "ok", False):
            self._add_message("note", f"No answer: {result.error}")
            self._set_status("No model answered — see the note above.")
            return
        speaker = result.speakers[0] if result.speakers else "Companion"
        self._add_message("assistant", result.reply, title=speaker)
        self._set_status("")
        self._refresh_character_panels()

    def _speak_last(self) -> None:
        if not self._selected_character:
            self._set_status("Select a character first.")
            return
        text = ""
        for _row, widget in reversed(self._row_frames):
            text = widget.get("1.0", "end").strip()
            if text:
                break
        if not text:
            self._set_status("There is nothing to read yet.")
            return
        result = self.service.speak(self._selected_character, text)
        self._set_status("Speaking…" if result else "No voice engine is installed.")

    def _stop_speaking(self) -> None:
        try:
            self.service.voice.stop()
            self._set_status("Stopped.")
        except Exception as error:
            self._set_status(str(error))

    def _toggle_favorite(self) -> None:
        if not self._selected_character:
            return
        rows = self.service.character_rows()
        current = next((r for r in rows if r.character_id == self._selected_character), None)
        self.service.set_favorite(self._selected_character, not (current and current.favorite))
        self._refresh_characters()

    def _archive_character(self) -> None:
        if not self._selected_character:
            return
        self.service.archive_character(self._selected_character)
        self._refresh_all()

    def _restore_character(self) -> None:
        selection = self._archive_list.curselection()
        if not selection:
            return
        rows = [r for r in self.service.character_rows(include_archived=True) if r.archived]
        if selection[0] < len(rows):
            self.service.restore_character(rows[selection[0]].character_id)
            self._refresh_all()

    def _toggle_adult(self) -> None:
        status = self.service.safety_status()
        if status.get("adult_mode"):
            self.service.set_adult_mode(False)
            self._set_status("Adult mode is off.")
        else:
            from tkinter import messagebox

            if not messagebox.askyesno(
                "Adult content",
                "Adult content is off by default. Do you confirm you are an adult and "
                "want to enable it? Every fictional participant must be 18 or older.",
                parent=self.window,
            ):
                return
            try:
                self.service.set_adult_mode(True, confirmed=True)
                self._set_status("Adult mode is on.")
            except Exception as error:
                self._set_status(f"Adult mode could not be enabled: {error}")
        self._refresh_safety()

    def _confirm_eligibility(self) -> None:
        """Record the selected character's adult-eligibility confirmation."""
        if not self._selected_character:
            self._set_status("Select a character first.")
            return
        try:
            self.service.confirm_eligibility(self._selected_character)
            self._refresh_character_panels()
            self._set_status("Confirmed: this character is an adult (18+).")
        except Exception as error:
            self._set_status(f"Could not confirm eligibility: {error}")

    def _add_boundary(self) -> None:
        value = self._boundary_entry.get().strip()
        if not self._selected_character:
            self._set_status("Select a character first.")
            return
        if not value:
            return
        self._boundary_entry.delete(0, "end")
        try:
            self.service.add_boundary(self._selected_character, value)
            self._refresh_character_panels()
            self._set_status("Boundary added.")
        except Exception as error:
            self._set_status(f"Could not add the boundary: {error}")

    # -- character management --------------------------------------------

    def _create_character(self) -> None:
        self._open_editor("")

    def _edit_character(self) -> None:
        if not self._selected_character:
            self._set_status("Select a character first.")
            return
        self._open_editor(self._selected_character)

    def _open_editor(self, character_id: str) -> None:
        from gui.romance_editor import CharacterEditorDialog

        dialog = CharacterEditorDialog(
            self.window, self.service, character_id=character_id,
            theme=self.theme, font=self.font, mono=self.mono,
            overrides=self._overrides, list_overrides=self._list_overrides,
            on_saved=self._on_editor_saved,
        )
        self._editors = getattr(self, "_editors", [])
        self._editors.append(dialog)

    def _on_editor_saved(self, character_id: str) -> None:
        if character_id:
            self._selected_character = character_id
        self._refresh_all()

    def _duplicate_character(self) -> None:
        if not self._selected_character:
            self._set_status("Select a character first.")
            return
        try:
            bundle = self.service.duplicate_character(self._selected_character)
            self._selected_character = bundle.id
            self._refresh_all()
            self._set_status(f"Duplicated as {bundle.name}.")
        except Exception as error:
            self._set_status(f"Could not duplicate: {error}")

    def _delete_character(self) -> None:
        if not self._selected_character:
            self._set_status("Select a character first.")
            return
        choice = _ask_retention(self.window, self.theme, self._type)
        if choice is None:
            return
        try:
            self.service.delete_character(self._selected_character, choice=choice)
        except Exception as error:
            self._set_status(f"Could not delete: {error}")
            return
        self._selected_character = ""
        self._refresh_all()
        self._set_status("Deleted.")

    def _export_character(self) -> None:
        if not self._selected_character:
            self._set_status("Select a character first.")
            return
        from tkinter import filedialog

        path = filedialog.asksaveasfilename(
            parent=self.window, title="Export the character",
            defaultextension=".json", filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.service.export_character(self._selected_character, path)
            self._set_status(f"Exported to {path}.")
        except Exception as error:
            self._set_status(f"Could not export: {error}")

    def _import_character(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            parent=self.window, title="Import a character",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            bundle = self.service.import_character(path)
            self._selected_character = bundle.id
            self._refresh_all()
            self._set_status(f"Imported {bundle.name}.")
        except Exception as error:
            self._set_status(f"Could not import: {error}")

    # -- models, relationship, memory, scenarios, voice, settings --------

    def _assign_model(self) -> None:
        if not self._selected_character:
            self._set_status("Select a character first.")
            return
        key = self._selected_model_key()
        if not key:
            self._set_status("No language model is available to assign.")
            return
        try:
            self.service.assign_model(self._selected_character, key)
            self._refresh_character_panels()
            self._set_status("Model assigned to the character.")
        except Exception as error:
            self._set_status(f"Could not assign the model: {error}")

    def _edit_relationship(self) -> None:
        if not self._selected_character:
            self._set_status("Select a character first.")
            return
        values = _prompt_dimensions(
            self.window, self.theme, self._type,
            self.service.relationship_dimensions(),
            self.service.relationship(self._selected_character).dimensions,
        )
        if values is None:
            return
        try:
            self.service.set_relationship(self._selected_character, **values)
            self._refresh_character_panels()
            self._set_status("Relationship updated.")
        except Exception as error:
            self._set_status(f"Could not update the relationship: {error}")

    def _reset_relationship(self) -> None:
        if not self._selected_character:
            return
        from tkinter import messagebox

        if not messagebox.askyesno(
            "Reset the relationship", "Return the relationship to its starting point?", parent=self.window
        ):
            return
        try:
            self.service.reset_relationship(self._selected_character)
            self._refresh_character_panels()
            self._set_status("Relationship reset.")
        except Exception as error:
            self._set_status(f"Could not reset: {error}")

    def _add_memory(self) -> None:
        if not self._selected_character:
            self._set_status("Select a character first.")
            return
        from core.romance.memory import KIND_LABELS, MEMORY_KINDS

        values = [(kind, KIND_LABELS.get(kind, kind)) for kind in MEMORY_KINDS]
        content = _prompt_memory(self.window, self.theme, self._type, values)
        if content is None:
            return
        kind, text = content
        try:
            self.service.add_memory(self._selected_character, kind, text)
            self._refresh_character_panels()
            self._set_status("Memory saved.")
        except Exception as error:
            self._set_status(f"Could not save the memory: {error}")

    def _forget_memory(self) -> None:
        selection = self._memory_list.curselection()
        if not selection or selection[0] >= len(getattr(self, "_memory_ids", [])):
            self._set_status("Choose a memory to forget.")
            return
        try:
            self.service.forget_memory(self._memory_ids[selection[0]])
            self._refresh_character_panels()
            self._set_status("Memory forgotten.")
        except Exception as error:
            self._set_status(f"Could not forget: {error}")

    def _export_memories(self) -> None:
        if not self._selected_character:
            return
        import json
        from tkinter import filedialog

        path = filedialog.asksaveasfilename(
            parent=self.window, title="Export the memories",
            defaultextension=".json", filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            export = self.service.export_memories(self._selected_character)
            Path(path).write_text(json.dumps(export.to_primitive(), ensure_ascii=False, indent=2), encoding="utf-8")
            self._set_status(f"Exported the memories to {path}.")
        except Exception as error:
            self._set_status(f"Could not export: {error}")

    def _create_scenario(self) -> None:
        values = _prompt_scenario(self.window, self.theme, self._type, self.service.scenario_categories())
        if values is None:
            return
        title, category, location, level = values
        try:
            self.service.create_scenario(
                title, category=category, location=location, content_level=level,
            )
            self._refresh_scenarios()
            self._set_status(f"Created the scenario {title}.")
        except Exception as error:
            self._set_status(f"Could not create the scenario: {error}")

    def _start_scenario(self) -> None:
        selection = self._scenario_list.curselection()
        rows = getattr(self, "_scenario_rows", [])
        if not selection or selection[0] >= len(rows):
            self._set_status("Choose a scenario to start.")
            return
        scenario_id = rows[selection[0]].scenario_id
        try:
            session = self.service.start_scenario(scenario_id, character_id=self._selected_character)
            self._set_status(f"Scenario started (session {getattr(session, 'record_id', '')}).")
            self._on_scenario_selected(selection[0])
        except Exception as error:
            self._set_status(f"Could not start the scenario: {error}")

    def _add_scenario_participant(self) -> None:
        selection = self._scenario_list.curselection()
        rows = getattr(self, "_scenario_rows", [])
        if not selection or selection[0] >= len(rows):
            self._set_status("Choose a scenario first.")
            return
        if not self._selected_character:
            self._set_status("Select a character to add to the scenario.")
            return
        try:
            self.service.add_participant(rows[selection[0]].scenario_id, self._selected_character)
            self._refresh_scenarios()
            self._on_scenario_selected(selection[0])
            self._set_status("Character added to the scenario.")
        except Exception as error:
            self._set_status(f"Could not add the character: {error}")

    def _toggle_voice(self) -> None:
        if not self._selected_character:
            self._set_status("Select a character first.")
            return
        try:
            view = self.service.voice_profile(self._selected_character)
            self.service.set_voice(self._selected_character, enabled=not view.enabled)
            self._refresh_character_panels()
            self._set_status("Voice turned on." if not view.enabled else "Voice turned off.")
        except Exception as error:
            self._set_status(f"Could not change the voice: {error}")

    def _open_settings(self) -> None:
        from gui.romance_settings import StudioSettingsDialog

        self._settings_dialog = StudioSettingsDialog(
            self.window, self.service, theme=self.theme, font=self.font, mono=self.mono,
            overrides=self._overrides, list_overrides=self._list_overrides,
            on_applied=self._refresh_all,
        )

    def _show_status(self) -> None:
        status = self.service.status()
        self._add_message("note", status.describe(), title="Studio summary")

    def _delete_conversation(self) -> None:
        if not self._selected_conversation:
            self._set_status("Select a conversation first.")
            return
        from tkinter import messagebox

        if not messagebox.askyesno(
            "Delete the conversation", "Delete this conversation and its messages?", parent=self.window
        ):
            return
        try:
            self.service.delete_conversation(self._selected_conversation)
        except Exception as error:
            self._set_status(f"Could not delete the conversation: {error}")
            return
        self._selected_conversation = ""
        self._clear_transcript()
        self._refresh_conversations()
        self._set_status("Conversation deleted.")

    def _set_status(self, text: str) -> None:
        try:
            self._status_line.configure(text=text)
        except tk.TclError:
            pass

    # -- background work --------------------------------------------------

    def _run_background(
        self,
        message: str,
        job: Callable[[], Any],
        finish: Callable[[Any, Exception | None], None],
    ) -> None:
        self._busy = True
        self._set_status(message)
        self._spinner.pack(side="right", padx=8)
        self._spinner.start(12)

        def run() -> None:
            try:
                self._results.put((finish, job(), None))
            except Exception as error:
                self._results.put((finish, None, error))

        threading.Thread(target=run, daemon=True).start()
        self.window.after(60, self._pump_results)

    def _pump_results(self) -> None:
        try:
            finish, result, error = self._results.get_nowait()
        except queue.Empty:
            if self._busy and not self._closed:
                self.window.after(60, self._pump_results)
            return
        self._busy = False
        self._spinner.stop()
        self._spinner.pack_forget()
        finish(result, error)

    # -- close ------------------------------------------------------------

    def retheme(self, theme: Any) -> None:
        """Restyle this window (and the editor/settings windows it holds open)."""
        super().retheme(theme)
        for dialog in self._child_dialogs():
            handler = getattr(dialog, "retheme", None)
            if callable(handler):
                try:
                    handler(theme)
                except Exception:
                    pass

    def retype(self, overrides: Any = None, list_overrides: Any = None) -> None:
        """Rebuild this window with new typography (and its child windows)."""
        super().retype(overrides, list_overrides)
        for dialog in self._child_dialogs():
            handler = getattr(dialog, "retype", None)
            if callable(handler):
                try:
                    handler(overrides, list_overrides)
                except Exception:
                    pass

    def _child_dialogs(self) -> list[Any]:
        """The editor/settings windows this Studio currently holds open."""
        dialogs = [d for d in getattr(self, "_editors", []) if getattr(d, "_closed", False) is False]
        settings = getattr(self, "_settings_dialog", None)
        if settings is not None and getattr(settings, "_closed", False) is False:
            dialogs.append(settings)
        return dialogs

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for dialog in self._child_dialogs():
            handler = getattr(dialog, "_close", None)
            if callable(handler):
                try:
                    handler()
                except Exception:
                    pass
        try:
            self.service.voice.stop()
        except Exception:
            pass
        try:
            self.window.destroy()
        finally:
            if self._on_close is not None:
                self._on_close()


def _prompt_scenario(
    parent: tk.Misc, theme: Any, typography: Any, categories: tuple[tuple[str, str], ...]
) -> tuple[str, str, str, str] | None:
    """Ask for a scenario's title, category, location, and content level."""
    dialog = tk.Toplevel(parent)
    dialog.title("New scenario")
    dialog.configure(bg=theme.base)
    dialog.transient(parent)  # type: ignore[call-overload]
    dialog.grab_set()
    result: dict[str, tuple[str, str, str, str] | None] = {"value": None}
    title_var = tk.StringVar()
    category_var = tk.StringVar(value=categories[0][1] if categories else "")
    location_var = tk.StringVar()
    level_var = tk.StringVar(value="general")
    rows = (
        ("Title", title_var, None),
        ("Category", category_var, [label for _v, label in categories]),
        ("Location (fictional)", location_var, None),
        ("Content level", level_var, ["general", "romantic", "adult"]),
    )
    for index, (label, variable, values) in enumerate(rows):
        tk.Label(dialog, text=label, bg=theme.base, fg=theme.foreground,
                 font=typography.font("sm")).grid(row=index, column=0, sticky="w", padx=12, pady=(8 if index == 0 else 2, 2))
        if values:
            from tkinter import ttk

            ttk.Combobox(dialog, textvariable=variable, values=values, state="readonly").grid(
                row=index, column=1, padx=12, pady=(8 if index == 0 else 2, 2)
            )
        else:
            tk.Entry(dialog, textvariable=variable, bg=theme.surface, fg=theme.foreground,
                     insertbackground=theme.foreground, relief="flat").grid(
                row=index, column=1, padx=12, pady=(8 if index == 0 else 2, 2)
            )

    def confirm() -> None:
        title = title_var.get().strip()
        if title:
            category = next((v for v, label in categories if label == category_var.get()), category_var.get())
            result["value"] = (title, category or "first_meeting", location_var.get().strip(), level_var.get() or "general")
        dialog.destroy()

    row = tk.Frame(dialog, bg=theme.base)
    row.grid(row=len(rows), column=0, columnspan=2, pady=12)
    tk.Button(row, text="Create", command=confirm, bg=theme.primary, fg=theme.primary_foreground,
              relief="flat", font=typography.font("sm")).pack(side="left", padx=6)
    tk.Button(row, text="Cancel", command=dialog.destroy, bg=theme.surface_raised, fg=theme.foreground,
              relief="flat", font=typography.font("sm")).pack(side="left", padx=6)
    dialog.wait_window()
    return result["value"]


def _prompt_dimensions(
    parent: tk.Misc, theme: Any, typography: Any,
    dimensions: tuple[str, ...], current: dict[str, float],
) -> dict[str, float] | None:
    """Ask for each relationship dimension's value on a 0..1 scale."""
    dialog = tk.Toplevel(parent)
    dialog.title("Edit the relationship")
    dialog.configure(bg=theme.base)
    dialog.transient(parent)  # type: ignore[call-overload]
    dialog.grab_set()
    result: dict[str, dict[str, float] | None] = {"value": None}
    variables: dict[str, tk.DoubleVar] = {}
    for index, name in enumerate(dimensions):
        tk.Label(dialog, text=name.replace("_", " ").title(), bg=theme.base, fg=theme.foreground,
                 font=typography.font("sm")).grid(row=index, column=0, sticky="w", padx=12, pady=(8 if index == 0 else 2, 2))
        variable = tk.DoubleVar(value=float(current.get(name, 0.0)))
        variables[name] = variable
        from tkinter import ttk

        ttk.Scale(dialog, from_=0.0, to=1.0, variable=variable, orient="horizontal", length=220).grid(
            row=index, column=1, padx=12, pady=(8 if index == 0 else 2, 2)
        )

    def confirm() -> None:
        result["value"] = {name: round(float(var.get()), 4) for name, var in variables.items()}
        dialog.destroy()

    row = tk.Frame(dialog, bg=theme.base)
    row.grid(row=len(dimensions), column=0, columnspan=2, pady=12)
    tk.Button(row, text="Save", command=confirm, bg=theme.primary, fg=theme.primary_foreground,
              relief="flat", font=typography.font("sm")).pack(side="left", padx=6)
    tk.Button(row, text="Cancel", command=dialog.destroy, bg=theme.surface_raised, fg=theme.foreground,
              relief="flat", font=typography.font("sm")).pack(side="left", padx=6)
    dialog.wait_window()
    return result["value"]


def _prompt_memory(
    parent: tk.Misc, theme: Any, typography: Any, kinds: list[tuple[str, str]]
) -> tuple[str, str] | None:
    """Ask for a memory's kind and content."""
    dialog = tk.Toplevel(parent)
    dialog.title("Remember something")
    dialog.configure(bg=theme.base)
    dialog.transient(parent)  # type: ignore[call-overload]
    dialog.grab_set()
    result: dict[str, tuple[str, str] | None] = {"value": None}
    kind_var = tk.StringVar(value=kinds[0][1] if kinds else "")
    tk.Label(dialog, text="Kind", bg=theme.base, fg=theme.foreground,
             font=typography.font("sm")).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 2))
    from tkinter import ttk

    ttk.Combobox(dialog, textvariable=kind_var, values=[label for _k, label in kinds], state="readonly").grid(
        row=0, column=1, padx=12, pady=(12, 2)
    )
    tk.Label(dialog, text="Content", bg=theme.base, fg=theme.foreground,
             font=typography.font("sm")).grid(row=1, column=0, sticky="nw", padx=12, pady=2)
    text = tk.Text(dialog, width=44, height=5, wrap="word", bg=theme.surface, fg=theme.foreground,
                   insertbackground=theme.foreground, relief="flat")
    text.grid(row=1, column=1, padx=12, pady=2)

    def confirm() -> None:
        content = text.get("1.0", "end").strip()
        if content:
            kind = next((k for k, label in kinds if label == kind_var.get()), kind_var.get())
            result["value"] = (kind or "semantic", content)
        dialog.destroy()

    row = tk.Frame(dialog, bg=theme.base)
    row.grid(row=2, column=0, columnspan=2, pady=12)
    tk.Button(row, text="Save", command=confirm, bg=theme.primary, fg=theme.primary_foreground,
              relief="flat", font=typography.font("sm")).pack(side="left", padx=6)
    tk.Button(row, text="Cancel", command=dialog.destroy, bg=theme.surface_raised, fg=theme.foreground,
              relief="flat", font=typography.font("sm")).pack(side="left", padx=6)
    dialog.wait_window()
    return result["value"]


def _ask_retention(parent: tk.Misc, theme: Any, typography: Any) -> str | None:
    """Ask how a character's data should be handled, returning its retention choice."""
    from gui.romance_editor import _ask_retention as ask

    return ask(parent, theme, typography)
