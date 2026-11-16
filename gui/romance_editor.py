"""The character editor: create or edit one companion in full.

A dedicated window, opened from the Studio's **New…** / **Edit…** controls, that
writes a character's whole profile through :class:`gui.romance.RomanceStudioService`
-- identity, personality dials, interests, boundaries, the language model, the
voice engine, and the adult-eligibility confirmation. It is a thin form over the
service, so the CLI and the window can never disagree about what a character is,
and Tk is imported lazily with the rest of the window layer.

Every field is a real control: a text box, a dropdown of the models this machine
really has, a 0..1 slider for each personality dial, and an add/remove list for
interests and boundaries. **Save** creates the character (or updates it), so no
button on this page is decorative.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import filedialog, ttk
from typing import Any

from core.romance.model import ROLE_USER

from gui.dialog_base import ThemedDialog
from gui.romance import RomanceStudioService
from gui.typography import resolve

TITLE_NEW = "New companion"
TITLE_EDIT = "Edit companion"
TITLE_NEW_PERSONA = "New persona (you)"
TITLE_EDIT_PERSONA = "Edit your persona"


def _dial_handler(label: tk.Label, var: tk.DoubleVar) -> Callable[[str], None]:
    """The scale handler that repaints a dial's numeric read-out."""

    def handler(_value: str) -> None:
        label.configure(text=f"{var.get():.2f}")

    return handler


#: The personality dials, in a readable order, with their labels and poles.
DIAL_ROWS: tuple[tuple[str, str, str, str], ...] = (
    ("introversion", "Introversion / extroversion", "outgoing", "reserved"),
    ("playfulness", "Seriousness / playfulness", "playful", "serious"),
    ("warmth", "Warmth / reserve", "warm", "reserved"),
    ("expressiveness", "Emotional expressiveness", "expressive", "contained"),
    ("confidence", "Confidence", "assured", "modest"),
    ("humour", "Humour", "funny", "dry"),
    ("initiative", "Conversation initiative", "leads", "follows"),
    ("affection", "Affectionate style", "affectionate", "reserved"),
    ("romantic_intensity", "Romantic intensity", "ardent", "calm"),
    ("formality", "Language formality", "formal", "casual"),
)

#: The answer-length choices, with their labels.
RESPONSE_LENGTH_CHOICES: tuple[tuple[str, str], ...] = (
    ("brief", "Brief"),
    ("balanced", "Balanced"),
    ("detailed", "Detailed"),
)

#: The *reach* dials: how the character leads, and how far it may take a romantic
#: or sensual scene with a consenting adult. Each is tone guidance only -- the
#: safety floor (adults only, consent, the honesty that she is fictional) holds at
#: every value.
REACH_DIAL_ROWS: tuple[tuple[str, str, str, str], ...] = (
    ("boldness", "Boldness / daring", "bold", "guarded"),
    ("passion", "Passion / intensity", "ardent", "composed"),
    ("sensuality", "Sensuality / openness of wording", "direct", "reserved"),
    ("escalation", "Escalation / lets the register build", "builds", "steady"),
)

#: The kinship choices: the relationship degree a character holds toward the user.
KINSHIP_CHOICES: tuple[tuple[str, str], ...] = (
    ("", "— none —"),
    ("mother", "Mother"),
    ("father", "Father"),
    ("parent", "Parent"),
    ("son", "Son"),
    ("daughter", "Daughter"),
    ("child", "Child"),
    ("sister", "Sister"),
    ("brother", "Brother"),
    ("sibling", "Sibling"),
    ("wife", "Wife"),
    ("husband", "Husband"),
    ("partner", "Partner"),
    ("fiancé(e)", "Fiancé(e)"),
    ("girlfriend", "Girlfriend"),
    ("boyfriend", "Boyfriend"),
    ("lover", "Lover"),
    ("grandmother", "Grandmother"),
    ("grandfather", "Grandfather"),
    ("aunt", "Aunt"),
    ("uncle", "Uncle"),
    ("cousin", "Cousin"),
    ("niece", "Niece"),
    ("nephew", "Nephew"),
    ("friend", "Friend"),
    ("best friend", "Best friend"),
    ("colleague", "Colleague"),
    ("neighbour", "Neighbour"),
    ("teacher", "Teacher"),
    ("student", "Student"),
    ("mentor", "Mentor"),
    ("companion", "Companion"),
    ("stranger", "Stranger"),
)

_RETENTION_CHOICES: tuple[tuple[str, str], ...] = (
    ("keep", "Keep the conversations and memories"),
    ("purge", "Delete the conversations and memories"),
    ("anonymize", "Keep them, but detach them from the name"),
)


class CharacterEditorDialog(ThemedDialog):
    """Create or edit one companion character."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        service: RomanceStudioService,
        *,
        character_id: str = "",
        role: str = "companion",
        theme: Any,
        font: str,
        mono: str,
        overrides: object | None = None,
        list_overrides: object | None = None,
        on_close: Callable[[], None] | None = None,
        on_saved: Callable[[str], None] | None = None,
    ) -> None:
        self.service = service
        self.character_id = character_id
        #: ``companion`` (who the user talks to) or ``user`` (the user's own
        #: persona, who the user talks *as*). The same window serves both, because
        #: a persona carries exactly the same attributes a companion does.
        self.role = role
        self.theme = theme
        self.font = font
        self.mono = mono
        self._type = resolve(parent, ui_family=font, mono_family=mono, overrides=overrides)
        self._overrides = overrides
        self._list_overrides = list_overrides
        self._list_type = resolve(parent, ui_family=font, mono_family=mono, overrides=list_overrides)
        self._on_close = on_close
        self._on_saved = on_saved
        self._closed = False
        self._editing = bool(character_id)
        self._interest_ids: list[str] = []
        self._boundary_ids: list[str] = []
        self._dial_scales: dict[str, tk.DoubleVar] = {}
        self._vars: dict[str, tk.StringVar] = {}
        self._avatar_photo: Any = None

        self.window = tk.Toplevel(parent)
        self.window.title(self._title())
        self.window.configure(bg=theme.base)
        self.window.geometry("820x760")
        self.window.minsize(680, 560)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._themed_build()

    def _title(self) -> str:
        if self.role == ROLE_USER:
            return TITLE_EDIT_PERSONA if self._editing else TITLE_NEW_PERSONA
        return TITLE_EDIT if self._editing else TITLE_NEW

    # -- build ------------------------------------------------------------

    def _build(self) -> None:
        from gui.components import DialogHeader

        header = DialogHeader(
            self.window,
            title=self._title(),
            subtitle=self._subtitle(),
            theme=self.theme, typography=self._type, on_close=self._close,
            extra=self._header_extra,
        )
        self._spinner = header.spinner

        body = tk.Frame(self.window, bg=self.theme.base)
        body.pack(fill="both", expand=True)
        self._canvas = tk.Canvas(body, bg=self.theme.base, highlightthickness=0)
        self._canvas.pack(side="left", fill="both", expand=True)
        scroll = tk.Scrollbar(body, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self._form = tk.Frame(self._canvas, bg=self.theme.base)
        self._window = self._canvas.create_window((0, 0), window=self._form, anchor="nw")
        self._form.bind("<Configure>", lambda _e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", lambda e: self._canvas.itemconfigure(self._window, width=e.width))
        for widget in (self._canvas, self._form):
            widget.bind("<MouseWheel>", self._on_wheel)
            widget.bind("<Button-4>", self._on_wheel)
            widget.bind("<Button-5>", self._on_wheel)

        self._build_avatar()
        self._build_identity()
        self._build_about()
        self._build_personality()
        self._build_reach()
        self._build_interests()
        self._build_boundaries()
        self._build_model_voice()
        self._build_adult()
        self._build_footer()

    def _subtitle(self) -> str:
        if self.role == ROLE_USER:
            return (
                "You — the person the companions talk with. Your name, your story, "
                "your voice, and your own avatar."
            )
        return "A fictional AI companion — independent identity, memory, and voice."

    def _header_extra(self, inner: tk.Frame) -> None:
        pass

    def _after_build(self) -> None:
        self._load_character()

    # -- sections ---------------------------------------------------------

    def _section(self, title: str, summary: str = "") -> tk.Frame:
        theme = self.theme
        outer = tk.Frame(self._form, bg=theme.surface)
        outer.pack(fill="x", padx=16, pady=(10, 0))
        tk.Label(
            outer, text=title, bg=theme.surface, fg=theme.contrast,
            font=self._type.font("sm", weight="bold"), anchor="w",
        ).pack(fill="x", padx=14, pady=(12, 0))
        if summary:
            tk.Label(
                outer, text=summary, bg=theme.surface, fg=theme.text_dim,
                font=self._type.font("2xs"), anchor="w", justify="left", wraplength=680,
            ).pack(fill="x", padx=14, pady=(2, 0))
        inner = tk.Frame(outer, bg=theme.surface)
        inner.pack(fill="x", padx=14, pady=(8, 12))
        return inner

    def _row(self, parent: tk.Frame, label: str, *, hint: str = "") -> tk.Frame:
        theme = self.theme
        row = tk.Frame(parent, bg=theme.surface)
        row.pack(fill="x", pady=3)
        head = tk.Frame(row, bg=theme.surface)
        head.pack(fill="x")
        tk.Label(
            head, text=label, bg=theme.surface, fg=theme.foreground,
            font=self._type.font("xs", weight="bold"), anchor="w",
        ).pack(side="left")
        if hint:
            tk.Label(
                head, text=hint, bg=theme.surface, fg=theme.text_dim,
                font=self._type.font("2xs"), anchor="e",
            ).pack(side="right")
        field = tk.Frame(row, bg=theme.surface)
        field.pack(fill="x", pady=(2, 0))
        return field

    def _entry(self, parent: tk.Frame, key: str, *, initial: str = "", width: int = 40) -> tk.Entry:
        var = self._vars.setdefault(key, tk.StringVar(value=initial))
        entry = tk.Entry(
            parent, textvariable=var, width=width, relief="flat", borderwidth=1,
            highlightthickness=0, bg=self.theme.surface_raised, fg=self.theme.foreground,
            insertbackground=self.theme.foreground, font=self._type.font("sm"),
        )
        entry.pack(fill="x", ipady=4, padx=1)
        return entry

    def _textarea(self, parent: tk.Frame, key: str, *, height: int = 3) -> tk.Text:
        theme = self.theme
        widget = tk.Text(
            parent, height=height, wrap="word", relief="flat", borderwidth=1,
            highlightthickness=0, bg=theme.surface_raised, fg=theme.foreground,
            insertbackground=theme.foreground, font=self._type.font("sm"), padx=6, pady=4,
        )
        widget.pack(fill="x")
        self._text_widgets = getattr(self, "_text_widgets", {})
        self._text_widgets[key] = widget
        return widget

    def _combo(self, parent: tk.Frame, key: str, rows: tuple[tuple[str, str], ...], *, initial: str = "") -> ttk.Combobox:
        var = self._vars.setdefault(key, tk.StringVar(value=initial))
        labels = [label for _value, label in rows]
        box = ttk.Combobox(parent, textvariable=var, values=labels, state="readonly", font=self._type.font("sm"))
        box.pack(fill="x")
        self._combo_rows = getattr(self, "_combo_rows", {})
        self._combo_rows[key] = rows
        return box

    def _combo_value(self, key: str) -> str:
        """The value behind the label currently shown in a readonly combobox."""
        rows = getattr(self, "_combo_rows", {}).get(key, ())
        shown = self._vars[key].get()
        for value, label in rows:
            if label == shown:
                return value
        return shown

    def _combo_set(self, key: str, value: str) -> None:
        rows = getattr(self, "_combo_rows", {}).get(key, ())
        for row_value, label in rows:
            if row_value == value:
                self._vars[key].set(label)
                return
        if rows:
            self._vars[key].set(rows[0][1])

    def _check(self, parent: tk.Frame, key: str, text: str, *, initial: bool = False) -> tk.BooleanVar:
        theme = self.theme
        var = tk.BooleanVar(value=initial)
        tk.Checkbutton(
            parent, text=text, variable=var, bg=theme.surface, fg=theme.foreground,
            activebackground=theme.surface, activeforeground=theme.foreground,
            selectcolor=theme.base, font=self._type.font("sm"), anchor="w",
            highlightthickness=0, borderwidth=0,
        ).pack(fill="x")
        self._check_vars = getattr(self, "_check_vars", {})
        self._check_vars[key] = var
        return var

    def _dial(self, parent: tk.Frame, key: str, label: str, low: str, high: str) -> None:
        theme = self.theme
        row = tk.Frame(parent, bg=theme.surface)
        row.pack(fill="x", pady=2)
        tk.Label(
            row, text=label, bg=theme.surface, fg=theme.foreground,
            font=self._type.font("xs"), anchor="w", width=28,
        ).pack(side="left")
        var = tk.DoubleVar(value=0.5)
        self._dial_scales[key] = var
        value_label = tk.Label(
            row, text="0.50", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs"), width=5, anchor="e",
        )
        value_label.pack(side="right")
        scale = ttk.Scale(
            row, from_=0.0, to=1.0, variable=var, orient="horizontal",
            command=_dial_handler(value_label, var),
        )
        scale.pack(side="left", fill="x", expand=True, padx=(0, 8))
        tk.Label(row, text=low, bg=theme.surface, fg=theme.text_dim, font=self._type.font("2xs")).pack(side="left")
        tk.Label(row, text=" · ", bg=theme.surface, fg=theme.text_dim, font=self._type.font("2xs")).pack(side="left")
        tk.Label(row, text=high, bg=theme.surface, fg=theme.text_dim, font=self._type.font("2xs")).pack(side="left")

    # -- the individual sections -----------------------------------------

    def _build_avatar(self) -> None:
        s = self._section(
            "Avatar",
            "Each character has its own picture: upload one, or generate one from "
            "the character's own description. Clearing it falls back to the "
            "initial-letter badge.",
        )
        row = tk.Frame(s, bg=self.theme.surface)
        row.pack(fill="x", pady=(0, 6))
        self._avatar_canvas = tk.Canvas(
            row, width=72, height=72, bg=self.theme.surface_raised,
            highlightthickness=0,
        )
        self._avatar_canvas.pack(side="left")
        controls = tk.Frame(row, bg=self.theme.surface)
        controls.pack(side="left", fill="x", expand=True, padx=(10, 0))
        self._button(controls, "Upload a picture…", self._upload_avatar).pack(anchor="w")
        gen_row = tk.Frame(controls, bg=self.theme.surface)
        gen_row.pack(anchor="w", fill="x", pady=(6, 0))
        self._button(gen_row, "Generate", self._generate_avatar).pack(side="left")
        self._combo(
            gen_row,
            "avatar_backend",
            self.service.avatar_backends(),
            initial="procedural",
        )
        self._button(controls, "Clear", self._clear_avatar).pack(anchor="w", pady=(6, 0))
        self._avatar_note = tk.Label(
            controls, text="", bg=self.theme.surface, fg=self.theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=560,
        )
        self._avatar_note.pack(fill="x", pady=(6, 0))

    def _build_identity(self) -> None:
        s = self._section("Identity", "Who this character is (all fictional).")
        self._entry(self._row(s, "Display name"), "name")
        self._entry(self._row(s, "Nickname", hint="optional"), "nickname")
        if self.role == ROLE_USER:
            self._active_var = self._check(
                s, "persona_active",
                "Speak as this persona (the companions address you by this name).",
            )
        self._entry(self._row(s, "Fictional age", hint="adults only (18+)"), "age", width=8)
        self._entry(self._row(s, "Gender", hint="optional"), "gender")
        self._entry(self._row(s, "Nationality / culture", hint="optional"), "nationality")
        self._entry(self._row(s, "Languages", hint="comma-separated"), "languages")
        self._combo(
            self._row(
                s,
                "Kinship",
                hint=(
                    "who you are to the companions"
                    if self.role == ROLE_USER
                    else "who this character is to you"
                ),
            ),
            "kinship",
            KINSHIP_CHOICES,
        )
        self._entry(self._row(s, "Preferred relationship style", hint="optional"), "relationship_style")

    def _build_about(self) -> None:
        s = self._section(
            "About",
            "Your story and appearance."
            if self.role == ROLE_USER
            else "The character's biography and appearance (its own story).",
        )
        self._textarea(self._row(s, "Biography"), "biography", height=3)
        self._textarea(self._row(s, "Appearance"), "appearance", height=2)

    def _build_personality(self) -> None:
        s = self._section(
            "Personality", "Tone guidance for how the character speaks — never a widening of what it may claim."
        )
        for key, label, low, high in DIAL_ROWS:
            self._dial(s, key, label, low, high)
        self._combo(self._row(s, "Preferred answer length"), "response_length", RESPONSE_LENGTH_CHOICES, initial="Balanced")
        self._entry(self._row(s, "Arabic dialect", hint="optional"), "dialect")
        self._textarea(self._row(s, "Extra instructions", hint="free guidance"), "instructions", height=2)

    def _build_reach(self) -> None:
        s = self._section(
            "Romantic reach",
            "How this character leads, and how far it may take a scene with a "
            "consenting adult. Tone only — adults only, consent, and the honesty "
            "that the character is fictional hold at every value.",
        )
        for key, label, low, high in REACH_DIAL_ROWS:
            self._dial(s, key, label, low, high)

    def _build_interests(self) -> None:
        s = self._section("Interests & hobbies", "What the character likes to talk about.")
        self._interest_list = tk.Listbox(
            s, height=4, bg=self.theme.surface_raised, fg=self.theme.foreground,
            selectbackground=self.theme.primary, selectforeground=self.theme.primary_foreground,
            relief="flat", borderwidth=0, highlightthickness=0,
            font=self._list_type.font("2xs"), activestyle="none",
        )
        self._interest_list.pack(fill="x", pady=(0, 4))
        add_row = tk.Frame(s, bg=self.theme.surface)
        add_row.pack(fill="x")
        self._interest_entry = tk.Entry(
            add_row, relief="flat", borderwidth=1, highlightthickness=0,
            bg=self.theme.surface_raised, fg=self.theme.foreground,
            insertbackground=self.theme.foreground, font=self._type.font("sm"),
        )
        self._interest_entry.pack(side="left", fill="x", expand=True, ipady=3)
        self._button(add_row, "Add", self._add_interest).pack(side="left", padx=(6, 0))
        self._button(add_row, "Remove", self._remove_interest).pack(side="left", padx=(6, 0))

    def _build_boundaries(self) -> None:
        s = self._section("Conversation boundaries", "Opt-outs the character must always respect.")
        self._boundary_list = tk.Listbox(
            s, height=3, bg=self.theme.surface_raised, fg=self.theme.foreground,
            selectbackground=self.theme.primary, selectforeground=self.theme.primary_foreground,
            relief="flat", borderwidth=0, highlightthickness=0,
            font=self._list_type.font("2xs"), activestyle="none",
        )
        self._boundary_list.pack(fill="x", pady=(0, 4))
        add_row = tk.Frame(s, bg=self.theme.surface)
        add_row.pack(fill="x")
        self._boundary_entry = tk.Entry(
            add_row, relief="flat", borderwidth=1, highlightthickness=0,
            bg=self.theme.surface_raised, fg=self.theme.foreground,
            insertbackground=self.theme.foreground, font=self._type.font("sm"),
        )
        self._boundary_entry.pack(side="left", fill="x", expand=True, ipady=3)
        self._button(add_row, "Add", self._add_boundary).pack(side="left", padx=(6, 0))
        self._button(add_row, "Remove", self._remove_boundary).pack(side="left", padx=(6, 0))

    def _build_model_voice(self) -> None:
        s = self._section(
            "Language model & voice",
            "Chosen from the models this machine really has; a missing one says so.",
        )
        self._model_note = tk.Label(
            s, text="", bg=self.theme.surface, fg=self.theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=680,
        )
        self._model_note.pack(fill="x", pady=(0, 4))
        choices = [(choice.key, choice.label()) for choice in self.service.model_choices()]
        self._combo(self._row(s, "Language model"), "model_key", tuple(choices) or (("", "(no model installed)"),))
        self._combo(
            self._row(s, "Voice engine"),
            "voice_engine",
            tuple((engine, engine) for engine in self.service.voice_engines()),
            initial="auto",
        )
        self._combo(
            self._row(s, "Default conversation mode"),
            "default_mode",
            tuple((k, title) for k, title, _s in self.service.modes()),
        )

    def _build_adult(self) -> None:
        s = self._section(
            "Adult content",
            "Adult mode is off by default. Confirming requires a fictional age of 18 or older.",
        )
        self._adult_var = self._check(
            s, "adult_eligible", "This character is an adult (18+) and may take part in adult scenes.",
        )

    def _build_footer(self) -> None:
        theme = self.theme
        bar = tk.Frame(self._form, bg=theme.base)
        bar.pack(fill="x", padx=16, pady=(12, 16))
        self._button(bar, "Save", self._save, variant="primary").pack(side="left")
        self._button(bar, "Cancel", self._close).pack(side="left", padx=(8, 0))
        if self._editing:
            self._button(bar, "Duplicate", self._duplicate).pack(side="right")
            self._button(bar, "Export…", self._export).pack(side="right", padx=(0, 8))
            self._button(bar, "Delete…", self._delete, variant="danger").pack(side="right", padx=(0, 8))
        self._status_label = tk.Label(
            self._form, text="", bg=theme.base, fg=theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=760,
        )
        self._status_label.pack(fill="x", padx=16, pady=(0, 10))

    # -- loading ----------------------------------------------------------

    def _load_character(self) -> None:
        if not self._editing:
            self._combo_set("default_mode", "general")
            self._combo_set("response_length", "balanced")
            self._combo_set("voice_engine", "auto")
            self._combo_set("kinship", "")
            self._combo_set("avatar_backend", "procedural")
            self._set_model_note()
            self._refresh_avatar()
            return
        try:
            bundle = self.service.character(self.character_id)
        except Exception as error:
            self._status(str(error))
            return
        character = bundle.character
        self.role = getattr(character, "role", self.role) or self.role
        self.window.title(self._title())
        self._vars["name"].set(character.name)
        self._vars["nickname"].set(character.nickname)
        self._vars["age"].set("" if character.fictional_age is None else str(character.fictional_age))
        self._vars["gender"].set(character.gender)
        self._vars["nationality"].set(character.nationality)
        self._vars["languages"].set(", ".join(character.languages))
        self._combo_set("kinship", character.kinship or "")
        self._vars["relationship_style"].set(character.relationship_style)
        self._text_widgets["biography"].delete("1.0", "end")
        self._text_widgets["biography"].insert("1.0", character.biography)
        self._text_widgets["appearance"].delete("1.0", "end")
        self._text_widgets["appearance"].insert("1.0", character.appearance)
        self._combo_set("default_mode", character.default_mode or "general")

        personality = self.service.personality(self.character_id)
        for key, _label, _low, _high in (*DIAL_ROWS, *REACH_DIAL_ROWS):
            self._dial_scales[key].set(float(personality.get(key, 0.5)))
        self._combo_set("response_length", personality.get("response_length", "balanced"))
        self._vars["dialect"].set(personality.get("dialect", ""))
        self._text_widgets["instructions"].delete("1.0", "end")
        self._text_widgets["instructions"].insert("1.0", personality.get("instructions", ""))

        voice = bundle.voice
        self._combo_set("voice_engine", (voice.engine if voice else "") or "auto")
        self._combo_set("model_key", character.model_key or "")
        self._adult_var.set(bool(character.adult_eligible))
        active_var = getattr(self, "_active_var", None)
        if active_var is not None:
            active_var.set(bool(getattr(character, "is_active", False)))
        self._refresh_interests()
        self._refresh_boundaries()
        self._refresh_avatar()
        self._set_model_note()

    def _set_model_note(self) -> None:
        choices = self.service.model_choices()
        if not choices:
            self._model_note.configure(text="No language model is installed. Download one from the Model Manager to chat.")
            return
        ready = sum(1 for c in choices if c.available)
        self._model_note.configure(text=f"{ready} of {len(choices)} models are installed and ready here.")

    def _refresh_interests(self) -> None:
        self._interest_list.delete(0, "end")
        self._interest_ids = []
        try:
            bundle = self.service.character(self.character_id)
        except Exception:
            return
        for item in bundle.profile_items:
            self._interest_ids.append(item.record_id)
            self._interest_list.insert("end", item.label)

    def _refresh_boundaries(self) -> None:
        self._boundary_list.delete(0, "end")
        self._boundary_ids = []
        for boundary in self.service.boundaries(self.character_id):
            self._boundary_ids.append(boundary.record_id)
            self._boundary_list.insert("end", boundary.value)

    # -- list actions -----------------------------------------------------

    def _add_interest(self) -> None:
        label = self._interest_entry.get().strip()
        if not label or not self._editing:
            if not self._editing:
                self._status("Save the character first, then add interests.")
            return
        self._interest_entry.delete(0, "end")
        try:
            self.service.add_interest(self.character_id, label)
            self._refresh_interests()
        except Exception as error:
            self._status(str(error))

    def _remove_interest(self) -> None:
        selection = self._interest_list.curselection()
        if not selection or selection[0] >= len(self._interest_ids):
            return
        self.service.remove_interest(self.character_id, self._interest_ids[selection[0]])
        self._refresh_interests()

    def _add_boundary(self) -> None:
        value = self._boundary_entry.get().strip()
        if not value or not self._editing:
            if not self._editing:
                self._status("Save the character first, then add boundaries.")
            return
        self._boundary_entry.delete(0, "end")
        try:
            self.service.add_boundary(self.character_id, value)
            self._refresh_boundaries()
        except Exception as error:
            self._status(str(error))

    def _remove_boundary(self) -> None:
        selection = self._boundary_list.curselection()
        if not selection or selection[0] >= len(self._boundary_ids):
            return
        self.service.remove_boundary(self._boundary_ids[selection[0]])
        self._refresh_boundaries()

    # -- save / manage ----------------------------------------------------

    def _gather(self) -> dict[str, Any]:
        name = self._vars["name"].get().strip()
        if not name:
            raise ValueError("a character needs a name")
        age: int | None = None
        if self._vars["age"].get().strip():
            try:
                age = int(self._vars["age"].get().strip())
            except ValueError as error:
                raise ValueError("the fictional age must be a whole number") from error
        languages = tuple(
            part.strip() for part in self._vars["languages"].get().split(",") if part.strip()
        )
        personality: dict[str, Any] = {key: float(var.get()) for key, var in self._dial_scales.items()}
        personality["response_length"] = self._combo_value("response_length") or "balanced"
        personality["dialect"] = self._vars["dialect"].get().strip()
        personality["instructions"] = self._text_widgets["instructions"].get("1.0", "end").strip()
        return {
            "name": name,
            "nickname": self._vars["nickname"].get().strip(),
            "fictional_age": age,
            "gender": self._vars["gender"].get().strip(),
            "nationality": self._vars["nationality"].get().strip(),
            "languages": languages,
            "kinship": self._combo_value("kinship").strip(),
            "relationship_style": self._vars["relationship_style"].get().strip(),
            "biography": self._text_widgets["biography"].get("1.0", "end").strip(),
            "appearance": self._text_widgets["appearance"].get("1.0", "end").strip(),
            "default_mode": self._combo_value("default_mode") or "general",
            "model_key": self._combo_value("model_key") or "",
            "voice_engine": self._combo_value("voice_engine") or "auto",
            "personality": personality,
            "adult_eligible": bool(self._adult_var.get()),
            "persona_active": self._persona_active(),
        }

    def _persona_active(self) -> bool:
        """Whether the editor's "speak as this" switch is set (companions: never)."""
        variable = getattr(self, "_active_var", None)
        return bool(variable.get()) if variable is not None else False

    def _save(self) -> None:
        try:
            values = self._gather()
        except ValueError as error:
            self._status(str(error))
            return
        try:
            if self._editing:
                self.service.update_character(
                    self.character_id,
                    name=values["name"], nickname=values["nickname"],
                    fictional_age=values["fictional_age"], gender=values["gender"],
                    nationality=values["nationality"], languages=values["languages"],
                    kinship=values["kinship"],
                    relationship_style=values["relationship_style"],
                    biography=values["biography"], appearance=values["appearance"],
                    default_mode=values["default_mode"], model_key=values["model_key"],
                )
                self.service.set_personality(self.character_id, values["personality"])
                self.service.set_voice(self.character_id, engine=values["voice_engine"])
                character_id = self.character_id
            else:
                bundle = self.service.create_character(
                    values["name"], role=self.role,
                    fictional_age=values["fictional_age"],
                    nickname=values["nickname"], gender=values["gender"],
                    nationality=values["nationality"], languages=values["languages"],
                    appearance=values["appearance"],
                    kinship=values["kinship"],
                    relationship_style=values["relationship_style"],
                    default_mode=values["default_mode"], model_key=values["model_key"],
                    biography=values["biography"], personality=values["personality"],
                )
                character_id = bundle.id
                self.character_id = character_id
                self._editing = True
            # A persona may be made the one the user speaks as; the switch is one
            # call, so exactly one persona is ever active.
            if self.role == ROLE_USER and values.get("persona_active"):
                try:
                    self.service.activate_persona(character_id)
                except Exception as error:
                    self._status(f"The persona was saved but not activated: {error}")
            # Adult eligibility is a separate, explicit confirmation the safety
            # floor owns: it is only recorded when the age really is 18+.
            if values["adult_eligible"]:
                try:
                    self.service.confirm_eligibility(character_id)
                except Exception as error:
                    self._status(f"Adult eligibility was not recorded: {error}")
        except Exception as error:
            self._status(f"Could not save: {error}")
            return
        self._status("Saved.")
        if self._on_saved is not None:
            self._on_saved(character_id)
        self._refresh_interests()
        self._refresh_boundaries()

    def _duplicate(self) -> None:
        try:
            bundle = self.service.duplicate_character(self.character_id)
        except Exception as error:
            self._status(str(error))
            return
        if self._on_saved is not None:
            self._on_saved(bundle.id)
        self._status(f"Duplicated as {bundle.name}.")

    def _export(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.window, title="Export the character",
            defaultextension=".json", filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.service.export_character(self.character_id, path)
        except Exception as error:
            self._status(str(error))
            return
        self._status(f"Exported to {path}.")

    def _delete(self) -> None:
        choice = _ask_retention(self.window, self.theme, self._type)
        if choice is None:
            return
        try:
            self.service.delete_character(self.character_id, choice=choice)
        except Exception as error:
            self._status(str(error))
            return
        if self._on_saved is not None:
            self._on_saved("")
        self._status("Deleted.")
        self._close()

    # -- small helpers ----------------------------------------------------

    def _button(self, parent: tk.Misc, text: str, command: Any, *, variant: str = "secondary") -> tk.Button:
        from gui.components import button

        return button(
            parent, text, command, theme=self.theme, typography=self._type,
            variant=variant, small=True,
        )

    # -- avatar -----------------------------------------------------------

    def _upload_avatar(self) -> None:
        if not self.character_id:
            self._status("Save the character first, then set its avatar.")
            return
        path = filedialog.askopenfilename(
            parent=self.window, title="Choose a picture",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.gif *.bmp *.webp *.ppm"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            result = self.service.set_avatar_from_file(self.character_id, path)
        except Exception as error:
            self._status(f"Could not set the avatar: {error}")
            return
        self._avatar_note.configure(text=result.describe())
        self._refresh_avatar()

    def _generate_avatar(self) -> None:
        if not self.character_id:
            self._status("Save the character first, then generate its avatar.")
            return
        backend = self._combo_value("avatar_backend") or "procedural"
        try:
            result = self.service.generate_avatar(self.character_id, backend=backend)
        except Exception as error:
            self._status(f"Could not generate the avatar: {error}")
            return
        self._avatar_note.configure(text=result.describe())
        self._refresh_avatar()

    def _clear_avatar(self) -> None:
        if not self.character_id:
            return
        try:
            self.service.clear_avatar(self.character_id)
        except Exception as error:
            self._status(str(error))
            return
        self._avatar_note.configure(text="Avatar cleared.")
        self._refresh_avatar()

    def _refresh_avatar(self) -> None:
        """Draw the character's current picture, or its initial badge."""
        canvas = getattr(self, "_avatar_canvas", None)
        if canvas is None:
            return
        try:
            canvas.delete("all")
        except tk.TclError:
            return
        path = self.service.avatar_path(self.character_id) if self.character_id else None
        if path is not None:
            try:
                self._avatar_photo = tk.PhotoImage(file=str(path))
                canvas.create_image(36, 36, image=self._avatar_photo)
                return
            except Exception:
                self._avatar_photo = None
        name_var = self._vars.get("name")
        name = name_var.get().strip() if name_var is not None else ""
        canvas.create_oval(2, 2, 70, 70, fill=self.theme.primary, outline="")
        canvas.create_text(
            36, 37, text=(name[:1] or "❤").upper(),
            fill=self.theme.primary_foreground, font=self._type.font("heading3", weight="bold"),
        )

    def _status(self, text: str) -> None:
        try:
            self._status_label.configure(text=text)
        except (AttributeError, tk.TclError):
            pass

    def _on_wheel(self, event: tk.Event) -> None:
        delta = -1 * (event.delta // 120) if getattr(event, "delta", 0) else (
            -1 if getattr(event, "num", 0) == 4 else 1
        )
        self._canvas.yview_scroll(delta, "units")

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.window.destroy()
        finally:
            if self._on_close is not None:
                self._on_close()


def _ask_retention(parent: tk.Misc, theme: Any, typography: Any) -> str | None:
    """Ask how a character's data should be handled, returning its retention choice."""
    dialog = tk.Toplevel(parent)
    dialog.title("Delete the character")
    dialog.configure(bg=theme.base)
    dialog.transient(parent)  # type: ignore[call-overload]
    dialog.grab_set()
    result: dict[str, str | None] = {"value": None}
    tk.Label(
        dialog, text="What should happen to this character's conversations and memories?",
        bg=theme.base, fg=theme.foreground, font=typography.font("sm"),
        wraplength=420, justify="left",
    ).pack(padx=14, pady=(14, 6))
    var = tk.StringVar(value="keep")
    for value, label in _RETENTION_CHOICES:
        tk.Radiobutton(
            dialog, text=label, value=value, variable=var, bg=theme.base, fg=theme.foreground,
            activebackground=theme.base, activeforeground=theme.foreground, selectcolor=theme.surface,
            font=typography.font("sm"), anchor="w", highlightthickness=0, borderwidth=0,
        ).pack(fill="x", padx=14, anchor="w")

    def confirm() -> None:
        result["value"] = var.get()
        dialog.destroy()

    row = tk.Frame(dialog, bg=theme.base)
    row.pack(pady=14)
    tk.Button(row, text="Delete", command=confirm, bg=theme.danger, fg=theme.primary_foreground,
              relief="flat", font=typography.font("sm")).pack(side="left", padx=6)
    tk.Button(row, text="Cancel", command=dialog.destroy, bg=theme.surface_raised, fg=theme.foreground,
              relief="flat", font=typography.font("sm")).pack(side="left", padx=6)
    dialog.wait_window()
    return result["value"]


__all__ = ["DIAL_ROWS", "RESPONSE_LENGTH_CHOICES", "CharacterEditorDialog"]
