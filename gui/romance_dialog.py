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
from typing import Any

from gui.dialog_base import ThemedDialog
from gui.romance import RomanceStudioService
from gui.sidebar import SidebarColumn
from gui.typography import resolve

TITLE = "Romance & Companion Studio"
SUBTITLE = "استوديو الرومانسية والرفقة — local AI companions, dating, and roleplay"

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
        self._conversation_ids: list[str] = []
        self._example_names: list[str] = []
        self._mode_keys: list[str] = []
        self._row_frames: list[tuple[tk.Frame, tk.Text]] = []
        self._results: queue.Queue[tuple[Callable[[Any, Exception | None], None], Any, Exception | None]] = queue.Queue()

        self.window = tk.Toplevel(parent)
        self.window.title(TITLE)
        self.window.configure(bg=theme.base)
        self.window.geometry("1280x800")
        self.window.minsize(1000, 660)
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
            typography=self._type, on_close=self._close,
            extra=self._header_extra,
        )
        self._spinner = header.spinner
        self._window_controls = header.controls

    def _header_extra(self, inner: tk.Frame) -> None:
        from gui.components import button

        button(
            inner, "Right panel", self._toggle_right, theme=self.theme,
            typography=self._type, variant="secondary", small=True,
        ).pack(side="right", padx=6)

    # -- sidebar ----------------------------------------------------------

    def _build_sidebar(self, parent: tk.Frame) -> None:
        column = SidebarColumn(
            parent, theme=self.theme, typography=self._type, scope="romance", width=320,
        )
        self._sidebar_column = column
        self._sidebar = column.body
        theme = self.theme

        brand = tk.Frame(column.body, bg=theme.surface)
        brand.pack(fill="x", padx=14, pady=(14, 4))
        dot = tk.Canvas(brand, width=26, height=26, bg=theme.surface, highlightthickness=0)
        dot.create_oval(1, 1, 25, 25, fill=theme.primary, outline="")
        dot.create_text(13, 14, text="❤", fill=theme.primary_foreground,
                        font=self._type.font("sm", weight="bold"))
        dot.pack(side="left")
        tk.Label(brand, text="Companion Studio", bg=theme.surface, fg=theme.contrast,
                 font=self._type.font("body", weight="bold")).pack(side="left", padx=(8, 0))

        # Characters.
        chars = column.add("characters", "Characters", icon="👤", open=True)
        self._char_list = self._listbox(chars, height=8, on_select=self._on_character_selected)
        row = tk.Frame(chars, bg=theme.surface)
        row.pack(fill="x", pady=(4, 0))
        self._button(row, "New…", self._create_character)
        self._button(row, "Favorite", self._toggle_favorite)
        self._button(row, "Archive", self._archive_character)

        archive = column.add("archive", "Archived", icon="🗄")
        self._archive_list = self._listbox(archive, height=4, on_select=self._on_character_selected)
        self._button(archive, "Restore", self._restore_character)

        # Discover: the clearly-labelled example companions this build ships.
        discover = column.add("discover", "Discover Characters", icon="🔎")
        self._example_list = self._listbox(
            discover, height=4, on_select=lambda _i: None
        )
        self._button(discover, "Add the chosen example", self._add_example)

        # Conversations.
        convs = column.add("conversations", "Recent Conversations", icon="💬", open=True)
        self._conv_list = self._listbox(convs, height=6, on_select=self._on_conversation_selected)
        row2 = tk.Frame(convs, bg=theme.surface)
        row2.pack(fill="x", pady=(4, 0))
        self._button(row2, "New chat", self._new_conversation)

        # Modes.
        modes = column.add("modes", "Conversation Mode", icon="🎭")
        self._mode_var = tk.StringVar(value="")
        self._mode_menu = tk.OptionMenu(modes, self._mode_var, "")
        self._style_option(self._mode_menu)
        self._mode_menu.pack(fill="x", pady=2)
        self._mode_var.trace_add("write", lambda *_: self._on_mode_changed())

        # Scenarios.
        scenarios = column.add("scenarios", "Dating Scenarios & Roleplay", icon="📖")
        self._scenario_list = self._listbox(scenarios, height=5, on_select=self._on_scenario_selected)
        self._button(scenarios, "New scenario…", self._create_scenario)

        # Voice.
        voice = column.add("voice", "Voice", icon="🔊")
        self._voice_list = self._listbox(voice, height=5, on_select=self._on_voice_selected)
        row3 = tk.Frame(voice, bg=theme.surface)
        row3.pack(fill="x", pady=(4, 0))
        self._button(row3, "Speak last reply", self._speak_last)
        self._button(row3, "Stop", self._stop_speaking)

        # Safety.
        safety = column.add("safety", "Privacy & Safety", icon="🔒")
        self._safety_label = tk.Label(
            safety, text="", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=250,
        )
        self._safety_label.pack(fill="x", pady=2)
        row4 = tk.Frame(safety, bg=theme.surface)
        row4.pack(fill="x", pady=(4, 0))
        self._button(row4, "Adult mode", self._toggle_adult)

    # -- centre -----------------------------------------------------------

    def _build_centre(self, parent: tk.Frame) -> None:
        theme = self.theme
        col = tk.Frame(parent, bg=theme.base)
        col.grid(row=0, column=1, sticky="nsew")
        self._centre = col

        head = tk.Frame(col, bg=theme.base)
        head.pack(fill="x", padx=16, pady=(10, 4))
        self._char_title = tk.Label(
            head, text="No character selected", bg=theme.base, fg=theme.contrast,
            font=self._type.font("heading4", weight="bold"), anchor="w",
        )
        self._char_title.pack(side="left")
        self._stage_label = tk.Label(
            head, text="", bg=theme.base, fg=theme.primary,
            font=self._type.font("sm", weight="bold"), anchor="e",
        )
        self._stage_label.pack(side="right")

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

        composer = tk.Frame(col, bg=theme.base)
        composer.pack(fill="x", padx=16, pady=(4, 12))
        self._entry = tk.Text(
            composer, height=2, wrap="word", relief="flat", borderwidth=1,
            highlightthickness=0, bg=theme.surface, fg=theme.foreground,
            insertbackground=theme.foreground, font=self._type.font("body"), padx=10, pady=8,
        )
        self._entry.pack(side="left", fill="x", expand=True)
        self._entry.bind("<Return>", self._on_return)
        self._entry.bind("<Shift-Return>", lambda _e: None)
        self._send_button = self._button(composer, "Send", self._send, anchor="center")
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
            panel, theme=self.theme, typography=self._type, scope="romance_info", width=300,
        )

        profile = self._right_column.add("profile", "Profile", icon="🪪", open=True)
        self._profile_text = self._panel_text(profile, height=8)

        traits = self._right_column.add("traits", "Personality & Interests", icon="✨", open=True)
        self._traits_text = self._panel_text(traits, height=8)

        relationship = self._right_column.add("relationship", "Relationship", icon="💞", open=True)
        self._relationship_text = self._panel_text(relationship, height=6)

        milestones = self._right_column.add("milestones", "Milestones", icon="🏁")
        self._milestones_text = self._panel_text(milestones, height=6)

        memories = self._right_column.add("memories", "Important Memories", icon="🧠")
        self._memories_text = self._panel_text(memories, height=8)

        voice = self._right_column.add("voice_config", "Voice Configuration", icon="🎙")
        self._voice_text = self._panel_text(voice, height=5)

        scenario = self._right_column.add("scenario", "Active Scenario", icon="🎬")
        self._scenario_text = self._panel_text(scenario, height=5)

    # -- small widget helpers --------------------------------------------

    def _button(
        self, parent: tk.Misc, text: str, command: Any, *, anchor: str = "w", small: bool = True
    ) -> tk.Button:
        from gui.components import button

        return button(
            parent, text, command, theme=self.theme, typography=self._type,
            variant="secondary", small=small, anchor=anchor,  # type: ignore[arg-type]
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
        row = tk.Frame(self._transcript, bg=theme.base)
        row.pack(fill="x", padx=(pad, pad), pady=6)
        head = tk.Frame(row, bg=theme.base)
        head.pack(fill="x", pady=(0, 4))
        label = title or {"user": "You", "note": "Note"}.get(role, "Companion")
        colour = theme.primary if role == "user" else theme.text_dim
        tk.Label(
            head, text=label, bg=theme.base, fg=colour,
            font=self._type.font("xs", weight="bold"), anchor="w",
        ).pack(side="left")
        widget = tk.Text(
            row, height=1, wrap="word", relief="flat", borderwidth=0, highlightthickness=0,
            bg=theme.base, fg=theme.foreground, font=self._type.font("body"),
            padx=0, pady=0, cursor="xterm", takefocus=1,
        )
        widget.insert("1.0", text)
        widget.pack(anchor="w", fill="x")
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
        self._refresh_conversations()
        self._refresh_modes()
        self._refresh_scenarios()
        self._refresh_voices()
        self._refresh_safety()
        self._refresh_character_panels()

    def _refresh_characters(self) -> None:
        self._char_list.delete(0, "end")
        self._character_ids = []
        for row in self.service.character_rows():
            self._character_ids.append(row.character_id)
            self._char_list.insert("end", row.label())
        self._archive_list.delete(0, "end")
        for row in self.service.character_rows(include_archived=True):
            if row.archived:
                self._archive_list.insert("end", row.label())
        self._refresh_examples()

    def _refresh_examples(self) -> None:
        self._example_list.delete(0, "end")
        self._example_names = []
        for draft, present in self.service.discover_examples():
            self._example_names.append(draft.name)
            mark = "✓ " if present else "＋ "
            self._example_list.insert("end", f"{mark}{draft.name} — {draft.nickname or draft.gender}")

    def _add_example(self) -> None:
        selection = self._example_list.curselection()
        if not selection or selection[0] >= len(self._example_names):
            self._set_status("Choose an example to add.")
            return
        name = self._example_names[selection[0]]
        try:
            self.service.add_example(name)
        except Exception as error:
            self._set_status(str(error))
            return
        self._refresh_characters()
        self._set_status(f"Added the example companion {name}.")

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

    def _refresh_scenarios(self) -> None:
        self._scenario_list.delete(0, "end")
        for row in self.service.scenarios():
            self._scenario_list.insert("end", f"{row.title} [{row.category_label}]")

    def _refresh_voices(self) -> None:
        self._voice_list.delete(0, "end")
        for option in self.service.voices():
            self._voice_list.insert("end", f"{option.label} ({option.source})")

    def _refresh_safety(self) -> None:
        status = self.service.safety_status()
        adult = "on" if status.get("adult_mode") else "off"
        local = "local only" if status.get("local_only") else "REMOTE ALLOWED"
        self._safety_label.configure(text=f"Adult mode: {adult}\nInference: {local}")

    def _refresh_character_panels(self) -> None:
        if not self._selected_character:
            for widget in (
                self._profile_text, self._traits_text, self._relationship_text,
                self._milestones_text, self._memories_text, self._voice_text,
                self._scenario_text,
            ):
                self._set_panel(widget, "")
            self._char_title.configure(text="No character selected")
            self._stage_label.configure(text="")
            return
        try:
            bundle = self.service.character(self._selected_character)
        except Exception:
            return
        character = bundle.character
        self._char_title.configure(text=character.name)
        profile = [
            f"Name: {character.name}",
            f"Nickname: {character.nickname}" if character.nickname else "",
            f"Age: {character.fictional_age}" if character.fictional_age else "",
            f"Gender: {character.gender}" if character.gender else "",
            f"Nationality: {character.nationality}" if character.nationality else "",
            f"Languages: {', '.join(character.languages)}" if character.languages else "",
            f"Relationship style: {character.relationship_style}" if character.relationship_style else "",
            f"Mode: {character.default_mode}",
        ]
        self._set_panel(self._profile_text, "\n".join(p for p in profile if p))
        personality = bundle.personality
        traits = []
        if personality is not None:
            for name in ("introversion", "playfulness", "warmth", "expressiveness", "confidence", "humor", "initiative", "affection", "romantic_intensity"):
                value = getattr(personality, name, None)
                if value is not None:
                    traits.append(f"{name.replace('_', ' ')}: {float(value):.2f}")
        interests = bundle.interests()
        if interests:
            traits.append("Interests: " + ", ".join(interests))
        self._set_panel(self._traits_text, "\n".join(traits))
        try:
            snapshot = self.service.relationship(self._selected_character)
            self._stage_label.configure(text=snapshot.stage_title)
            self._set_panel(self._relationship_text, snapshot.describe())
            events = self.service.relationship_history(self._selected_character)
            self._set_panel(
                self._milestones_text,
                "\n".join(f"• {getattr(e, 'title', '') or e.kind}" for e in events[-12:]),
            )
        except Exception:
            self._set_panel(self._relationship_text, "")
            self._set_panel(self._milestones_text, "")
        try:
            memories = self.service.memories(self._selected_character)
            self._set_panel(
                self._memories_text,
                "\n".join(f"• [{m.kind}] {m.content}" for m in memories[-12:]),
            )
        except Exception:
            self._set_panel(self._memories_text, "")
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
        self._refresh_character_panels()
        self._set_status(f"Selected {self._char_title.cget('text')}.")

    def _on_conversation_selected(self, index: int) -> None:
        if index >= len(self._conversation_ids):
            return
        self._selected_conversation = self._conversation_ids[index]
        self._load_conversation()

    def _on_scenario_selected(self, index: int) -> None:
        rows = self.service.scenarios()
        if index < len(rows):
            row = rows[index]
            self._set_panel(
                self._scenario_text,
                f"{row.title}\nCategory: {row.category_label}\nLevel: {row.content_level}\nParticipants: {row.participant_count}",
            )

    def _on_voice_selected(self, index: int) -> None:
        options = self.service.voices()
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

    def _on_mode_changed(self) -> None:
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
            try:
                self.service.set_adult_mode(True, confirmed=True)
                self._set_status("Adult mode is on.")
            except Exception as error:
                self._set_status(f"Adult mode needs an adult character: {error}")
        self._refresh_safety()

    def _create_character(self) -> None:
        values = _prompt_character(self.window, self.theme, self._type)
        if values is None:
            return
        name, age = values
        try:
            bundle = self.service.create_character(name, fictional_age=age)
            self._selected_character = bundle.id
            self._refresh_all()
            self._set_status(f"Created {bundle.name}.")
        except Exception as error:
            self._set_status(f"Could not create the character: {error}")

    def _create_scenario(self) -> None:
        title = _prompt_text(self.window, self.theme, self._type, "Scenario title")
        if not title:
            return
        try:
            scenario = self.service.create_scenario(
                title, category="first_meeting", location="",
            )
            self._refresh_scenarios()
            self._set_status(f"Created scenario {scenario.record_id}.")
        except Exception as error:
            self._set_status(f"Could not create the scenario: {error}")

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

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.service.voice.stop()
        except Exception:
            pass
        try:
            self.window.destroy()
        finally:
            if self._on_close is not None:
                self._on_close()


def _prompt_character(
    parent: tk.Misc, theme: Any, typography: Any
) -> tuple[str, int | None] | None:
    """A small modal asking for a character's name and (adult) fictional age."""
    dialog = tk.Toplevel(parent)
    dialog.title("New character")
    dialog.configure(bg=theme.base)
    dialog.transient(parent)  # type: ignore[call-overload]
    dialog.grab_set()
    result: dict[str, tuple[str, int | None] | None] = {"value": None}
    name_var = tk.StringVar()
    age_var = tk.StringVar()
    tk.Label(dialog, text="Name", bg=theme.base, fg=theme.foreground,
             font=typography.font("sm")).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 2))
    tk.Entry(dialog, textvariable=name_var, bg=theme.surface, fg=theme.foreground,
             insertbackground=theme.foreground, relief="flat").grid(row=0, column=1, padx=12, pady=(12, 2))
    tk.Label(dialog, text="Fictional age (adults only)", bg=theme.base, fg=theme.foreground,
             font=typography.font("sm")).grid(row=1, column=0, sticky="w", padx=12, pady=2)
    tk.Entry(dialog, textvariable=age_var, bg=theme.surface, fg=theme.foreground,
             insertbackground=theme.foreground, relief="flat").grid(row=1, column=1, padx=12, pady=2)

    def confirm() -> None:
        name = name_var.get().strip()
        age: int | None = None
        if age_var.get().strip():
            try:
                age = int(age_var.get().strip())
            except ValueError:
                age = None
        if name:
            result["value"] = (name, age)
        dialog.destroy()

    row = tk.Frame(dialog, bg=theme.base)
    row.grid(row=2, column=0, columnspan=2, pady=12)
    tk.Button(row, text="Create", command=confirm, bg=theme.primary, fg=theme.primary_foreground,
              relief="flat", font=typography.font("sm")).pack(side="left", padx=6)
    tk.Button(row, text="Cancel", command=dialog.destroy, bg=theme.surface_raised, fg=theme.foreground,
              relief="flat", font=typography.font("sm")).pack(side="left", padx=6)
    dialog.wait_window()
    return result["value"]


def _prompt_text(parent: tk.Misc, theme: Any, typography: Any, label: str) -> str:
    dialog = tk.Toplevel(parent)
    dialog.title(label)
    dialog.configure(bg=theme.base)
    dialog.transient(parent)  # type: ignore[call-overload]
    dialog.grab_set()
    result: dict[str, str] = {"value": ""}
    variable = tk.StringVar()
    tk.Label(dialog, text=label, bg=theme.base, fg=theme.foreground,
             font=typography.font("sm")).pack(padx=12, pady=(12, 2))
    entry = tk.Entry(dialog, textvariable=variable, bg=theme.surface, fg=theme.foreground,
                     insertbackground=theme.foreground, relief="flat", width=36)
    entry.pack(padx=12, pady=2)
    entry.focus_set()

    def confirm() -> None:
        result["value"] = variable.get().strip()
        dialog.destroy()

    tk.Button(dialog, text="OK", command=confirm, bg=theme.primary, fg=theme.primary_foreground,
              relief="flat", font=typography.font("sm")).pack(pady=12)
    dialog.wait_window()
    return result["value"]
