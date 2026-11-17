"""The Persona Manager and the Character Manager: full control in one window.

The Studio's sidebar edits a character in a small form and its avatar in the
editor. The two **managers** are the dedicated windows the user asked for: one
place that lists every character of a kind -- the user's own **personas** (*My
Persona Manager*) or the **companions** (*Character Manager*) -- with the
character's whole profile on the right and every attribute editable, its
**avatar** set, generated, or cleared, and the ready-made library beside it so a
new persona or companion can be chosen and start chatting at once.

It is deliberately one class over two roles, exactly like
:class:`gui.romance_editor.CharacterEditorDialog`: a persona carries the same
fields as a companion, so one window serves both and the two can never drift
apart. The window is a *view* over :class:`gui.romance.RomanceStudioService`, so
the window, the editor, and the CLI can never disagree about what a character is.

Built the way every other window is built: the shared :class:`~gui.themes.Theme`
tokens, :class:`gui.components.DialogHeader`/``button``, the shared
:class:`gui.sidebar.SidebarColumn`, a worker-thread/queue the UI thread drains,
and :class:`gui.dialog_base.ThemedDialog` so a theme or font change restyles it in
place. Tk is imported with the rest of the window layer.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import filedialog, ttk
from typing import Any

from core.romance.model import ROLE_COMPANION, ROLE_USER

from gui.dialog_base import ThemedDialog
from gui.romance import RomanceStudioService
from gui.romance_editor import (
    DIAL_ROWS,
    KINSHIP_CHOICES,
    REACH_DIAL_ROWS,
    RESPONSE_LENGTH_CHOICES,
    _ask_retention,
    _dial_handler,
)
from gui.sidebar import SidebarColumn
from gui.typography import resolve

TITLE_PERSONA_MANAGER = "My Persona Manager"
TITLE_CHARACTER_MANAGER = "Character Manager"

#: A short, honest line under each manager's title.
_SUBTITLE = {
    ROLE_USER: "Every persona you can be — edit the full profile, set the picture, "
    "or pick a ready-made one and speak as it.",
    ROLE_COMPANION: "Every companion you can talk to — edit the full profile, set the "
    "picture, or pick a ready-made one and start chatting.",
}

#: The library section's title and hint, per role.
_LIBRARY_TITLE = {
    ROLE_USER: "Ready-made personas",
    ROLE_COMPANION: "Ready-made companions",
}
_LIBRARY_HINT = {
    ROLE_USER: "Pick one and it becomes the person the companions address by name.",
    ROLE_COMPANION: "Pick one to bring it into the Studio and talk to it.",
}


class CharacterManagerDialog(ThemedDialog):
    """Full control over one kind of character, and its avatar, in one window."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        service: RomanceStudioService,
        *,
        role: str = ROLE_COMPANION,
        theme: Any,
        font: str,
        mono: str,
        overrides: object | None = None,
        list_overrides: object | None = None,
        on_close: Callable[[], None] | None = None,
        on_changed: Callable[[str], None] | None = None,
    ) -> None:
        self.service = service
        self.role = role if role in (ROLE_USER, ROLE_COMPANION) else ROLE_COMPANION
        self.theme = theme
        self.font = font
        self.mono = mono
        self._type = resolve(parent, ui_family=font, mono_family=mono, overrides=overrides)
        self._overrides = overrides
        self._list_overrides = list_overrides
        self._list_type = resolve(parent, ui_family=font, mono_family=mono, overrides=list_overrides)
        self._on_close = on_close
        self._on_changed = on_changed
        self._closed = False

        self._selected = ""
        self._ids: list[str] = []
        self._library_ids: list[str] = []
        self._library_rows: list[Any] = []
        self._interest_ids: list[str] = []
        self._boundary_ids: list[str] = []
        self._dial_scales: dict[str, tk.DoubleVar] = {}
        self._vars: dict[str, tk.StringVar] = {}
        self._text_widgets: dict[str, tk.Text] = {}
        self._combo_rows: dict[str, tuple[tuple[str, str], ...]] = {}
        self._check_vars: dict[str, tk.BooleanVar] = {}
        self._avatar_photo: Any = None

        self.window = tk.Toplevel(parent)
        self.window.title(self._title())
        self.window.configure(bg=theme.base)
        self.window.geometry("1080x760")
        self.window.minsize(820, 560)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)
        self._themed_build()

    # -- identity ---------------------------------------------------------

    def _title(self) -> str:
        return TITLE_PERSONA_MANAGER if self.role == ROLE_USER else TITLE_CHARACTER_MANAGER

    def _plural(self) -> str:
        return "persona" if self.role == ROLE_USER else "companion"

    # -- build ------------------------------------------------------------

    def _build(self) -> None:
        from gui.components import DialogHeader

        header = DialogHeader(
            self.window,
            title=self._title(),
            subtitle=_SUBTITLE.get(self.role, ""),
            theme=self.theme, typography=self._type, on_close=self._close,
        )
        self._spinner = header.spinner

        body = tk.Frame(self.window, bg=self.theme.base)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=0, minsize=320)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        sidebar_parent = tk.Frame(body, bg=self.theme.surface)
        sidebar_parent.grid(row=0, column=0, sticky="nsew")
        # The wrapper must give its single child the whole cell, or the column
        # collapses to its natural height and its lower controls (the *Open the
        # body builder…* button among them) are clipped off the bottom.
        sidebar_parent.rowconfigure(0, weight=1)
        sidebar_parent.columnconfigure(0, weight=1)
        self._column = SidebarColumn(
            sidebar_parent, theme=self.theme, typography=self._type,
            scope=f"romance-{self._plural()}-manager", width=320,
        )
        self._build_sidebar(self._column)

        right = tk.Frame(body, bg=self.theme.base)
        right.grid(row=0, column=1, sticky="nsew")
        self._build_profile(right)

        self._status = tk.Label(
            self.window, text="", bg=self.theme.base, fg=self.theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=1000,
        )
        self._status.pack(fill="x", padx=16, pady=(0, 8))

    def _after_build(self) -> None:
        self._refresh_list()
        self._refresh_library()
        if self._ids:
            self._select(self._ids[0])

    # -- sidebar ----------------------------------------------------------

    def _build_sidebar(self, column: SidebarColumn) -> None:
        theme = self.theme

        listing = column.add(
            "list", f"My {'Personas' if self.role == ROLE_USER else 'Companions'}",
            icon="🙋" if self.role == ROLE_USER else "👤", open=True,
        )
        self._search_var = tk.StringVar()
        search = tk.Entry(
            listing, textvariable=self._search_var, relief="flat", borderwidth=1,
            highlightthickness=0, bg=theme.surface_raised, fg=theme.foreground,
            insertbackground=theme.foreground, font=self._type.font("2xs"),
        )
        search.pack(fill="x", ipady=3, pady=(0, 4))
        search.bind("<KeyRelease>", lambda _e: self._refresh_list())
        self._list = self._listbox(listing, height=9, on_select=self._on_list_selected)
        row = tk.Frame(listing, bg=theme.surface)
        row.pack(fill="x", pady=(4, 0))
        self._button(row, "New…", self._create, variant="primary")
        self._button(row, "Edit the profile…", self._edit_in_editor)
        row2 = tk.Frame(listing, bg=theme.surface)
        row2.pack(fill="x")
        self._button(row2, "Duplicate", self._duplicate)
        if self.role == ROLE_USER:
            self._button(row2, "Speak as this", self._activate)
        row3 = tk.Frame(listing, bg=theme.surface)
        row3.pack(fill="x")
        self._button(row3, "Export…", self._export)
        self._button(row3, "Import…", self._import)
        self._button(row3, "Delete…", self._delete, variant="danger")

        library = column.add(
            "library", _LIBRARY_TITLE.get(self.role, "Library"), icon="📚", open=True,
        )
        tk.Label(
            library, text=_LIBRARY_HINT.get(self.role, ""),
            bg=theme.surface, fg=theme.text_dim, font=self._type.font("2xs"),
            anchor="w", justify="left", wraplength=280,
        ).pack(fill="x", pady=(0, 4))
        self._library_category_var = tk.StringVar(value="")
        self._library_category_menu = tk.OptionMenu(library, self._library_category_var, "")
        self._style_option(self._library_category_menu)
        self._library_category_menu.pack(fill="x", pady=2)
        self._library_category_var.trace_add("write", lambda *_: self._refresh_library())
        self._library_search_var = tk.StringVar()
        lib_search = tk.Entry(
            library, textvariable=self._library_search_var, relief="flat", borderwidth=1,
            highlightthickness=0, bg=theme.surface_raised, fg=theme.foreground,
            insertbackground=theme.foreground, font=self._type.font("2xs"),
        )
        lib_search.pack(fill="x", ipady=3, pady=(2, 4))
        lib_search.bind("<KeyRelease>", lambda _e: self._refresh_library())
        self._library_list = self._listbox(library, height=8, on_select=self._on_library_selected)
        self._library_preview = tk.Label(
            library, text="", bg=theme.surface_raised, fg=theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=280,
            padx=8, pady=6,
        )
        self._library_preview.pack(fill="x", pady=(2, 4))
        lib_row = tk.Frame(library, bg=theme.surface)
        lib_row.pack(fill="x")
        self._button(lib_row, "Add the chosen", self._add_library, variant="primary")
        self._button(lib_row, "Add the whole group", self._add_library_group)

        body = column.add("body", "Body & shape", icon="🧍", open=True)
        tk.Label(
            body,
            text=(
                "Build the character's whole body — pick a template, then shape every "
                "measurement, and use the render as the picture."
            ),
            bg=theme.surface,
            fg=theme.text_dim,
            font=self._type.font("2xs"),
            anchor="w",
            justify="left",
            wraplength=280,
        ).pack(fill="x", pady=(0, 4))
        self._button(
            body, "Open the body builder…", self._open_body_studio, variant="primary",
        )

    # -- the profile panel (right) ----------------------------------------

    def _build_profile(self, parent: tk.Frame) -> None:
        theme = self.theme
        head = tk.Frame(parent, bg=theme.base)
        head.pack(fill="x", padx=16, pady=(12, 0))
        self._avatar_canvas = tk.Canvas(
            head, width=76, height=76, bg=theme.surface_raised, highlightthickness=0,
        )
        self._avatar_canvas.pack(side="left")
        titles = tk.Frame(head, bg=theme.base)
        titles.pack(side="left", fill="x", expand=True, padx=(12, 0))
        self._name_label = tk.Label(
            titles, text="", bg=theme.base, fg=theme.contrast,
            font=self._type.font("heading3", weight="bold"), anchor="w",
        )
        self._name_label.pack(anchor="w")
        self._subtitle_label = tk.Label(
            titles, text="", bg=theme.base, fg=theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=600,
        )
        self._subtitle_label.pack(anchor="w")

        avatar_row = tk.Frame(titles, bg=theme.base)
        avatar_row.pack(anchor="w", fill="x", pady=(6, 0))
        self._button(avatar_row, "Upload a picture…", self._upload_avatar)
        self._button(avatar_row, "Generate", self._generate_avatar)
        self._button(avatar_row, "Clear picture", self._clear_avatar)
        backend_row = tk.Frame(titles, bg=theme.base)
        backend_row.pack(anchor="w", fill="x", pady=(4, 0))
        tk.Label(
            backend_row, text="Generator:", bg=theme.base, fg=theme.text_dim,
            font=self._type.font("2xs"),
        ).pack(side="left", padx=(0, 6))
        self._combo(
            backend_row, "avatar_backend", self.service.avatar_backends(), width=30,
        ).pack(side="left")

        canvas_holder = tk.Frame(parent, bg=theme.base)
        canvas_holder.pack(fill="both", expand=True, padx=16, pady=(8, 0))
        self._canvas = tk.Canvas(canvas_holder, bg=theme.base, highlightthickness=0)
        self._canvas.pack(side="left", fill="both", expand=True)
        scroll = tk.Scrollbar(canvas_holder, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self._form = tk.Frame(self._canvas, bg=theme.base)
        self._window = self._canvas.create_window((0, 0), window=self._form, anchor="nw")
        self._form.bind(
            "<Configure>", lambda _e: self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        )
        self._canvas.bind(
            "<Configure>", lambda e: self._canvas.itemconfigure(self._window, width=e.width)
        )
        for widget in (self._canvas, self._form):
            widget.bind("<MouseWheel>", self._on_wheel)
            widget.bind("<Button-4>", self._on_wheel)
            widget.bind("<Button-5>", self._on_wheel)

        self._build_identity()
        self._build_about()
        self._build_personality()
        self._build_reach()
        self._build_interests()
        self._build_boundaries()
        self._build_model_voice()
        self._build_adult()
        self._build_footer()

    def _section(self, title: str, summary: str = "") -> tk.Frame:
        from gui.components import Section

        return Section(
            self._form, title=title, theme=self.theme, typography=self._type, card=False,
        ).body

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

    def _entry(self, parent: tk.Frame, key: str, *, width: int = 40) -> tk.Entry:
        var = self._vars.setdefault(key, tk.StringVar(value=""))
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
        self._text_widgets[key] = widget
        return widget

    def _combo(
        self, parent: tk.Frame, key: str, rows: tuple[tuple[str, str], ...], *, width: int = 40,
    ) -> ttk.Combobox:
        var = self._vars.setdefault(key, tk.StringVar(value=""))
        labels = [label for _value, label in rows]
        box = ttk.Combobox(
            parent, textvariable=var, values=labels, state="readonly",
            font=self._type.font("sm"), width=width,
        )
        box.pack(fill="x")
        self._combo_rows[key] = rows
        return box

    def _combo_value(self, key: str) -> str:
        rows = self._combo_rows.get(key, ())
        shown = self._vars[key].get()
        for value, label in rows:
            if label == shown:
                return value
        return shown

    def _combo_set(self, key: str, value: str) -> None:
        rows = self._combo_rows.get(key, ())
        for row_value, label in rows:
            if row_value == value:
                self._vars[key].set(label)
                return
        if rows:
            self._vars[key].set(rows[0][1])

    def _check(self, parent: tk.Frame, key: str, text: str) -> tk.BooleanVar:
        theme = self.theme
        var = self._check_vars.setdefault(key, tk.BooleanVar(value=False))
        tk.Checkbutton(
            parent, text=text, variable=var, bg=theme.surface, fg=theme.foreground,
            activebackground=theme.surface, activeforeground=theme.foreground,
            selectcolor=theme.base, font=self._type.font("sm"), anchor="w",
            highlightthickness=0, borderwidth=0,
        ).pack(fill="x")
        return var

    def _dial(self, parent: tk.Frame, key: str, label: str, low: str, high: str) -> None:
        theme = self.theme
        row = tk.Frame(parent, bg=theme.surface)
        row.pack(fill="x", pady=2)
        tk.Label(
            row, text=label, bg=theme.surface, fg=theme.foreground,
            font=self._type.font("xs"), anchor="w", width=30,
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

    # -- the profile's own sections ---------------------------------------

    def _build_identity(self) -> None:
        s = self._section("Identity", "Who this character is (all fictional).")
        self._entry(self._row(s, "Display name"), "name")
        self._entry(self._row(s, "Nickname", hint="optional"), "nickname")
        if self.role == ROLE_USER:
            self._check(
                s, "persona_active",
                "Speak as this persona (the companions address you by this name).",
            )
        self._entry(self._row(s, "Fictional age", hint="adults only (18+)"), "age", width=8)
        self._entry(self._row(s, "Gender", hint="optional"), "gender")
        self._entry(self._row(s, "Nationality / culture", hint="optional"), "nationality")
        self._entry(self._row(s, "Languages", hint="comma-separated"), "languages")
        self._combo(
            self._row(
                s, "Kinship",
                hint=(
                    "who you are to the companions"
                    if self.role == ROLE_USER
                    else "who this character is to you"
                ),
            ),
            "kinship", KINSHIP_CHOICES,
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
            "Personality",
            "Tone guidance for how the character speaks — never a widening of what it may claim.",
        )
        for key, label, low, high in DIAL_ROWS:
            self._dial(s, key, label, low, high)
        self._combo(
            self._row(s, "Preferred answer length"), "response_length",
            RESPONSE_LENGTH_CHOICES,
        )
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
        self._interest_list = self._listbox(s, height=4, on_select=lambda _i: None, framed=False)
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
        self._boundary_list = self._listbox(s, height=3, on_select=lambda _i: None, framed=False)
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
        self._combo(
            self._row(s, "Language model"), "model_key",
            tuple(choices) or (("", "(no model installed)"),),
        )
        self._combo(
            self._row(s, "Voice engine"), "voice_engine",
            tuple((engine, engine) for engine in self.service.voice_engines()),
        )
        self._combo(
            self._row(s, "Default conversation mode"), "default_mode",
            tuple((k, title) for k, title, _s in self.service.modes()),
        )

    def _build_adult(self) -> None:
        s = self._section(
            "Adult content",
            "Adult mode is off by default. Confirming requires a fictional age of 18 or older.",
        )
        self._check(
            s, "adult_eligible",
            "This character is an adult (18+) and may take part in adult scenes.",
        )

    def _build_footer(self) -> None:
        theme = self.theme
        bar = tk.Frame(self._form, bg=theme.base)
        bar.pack(fill="x", padx=16, pady=(12, 16))
        self._button(bar, "Save the profile", self._save, variant="primary").pack(side="left")
        self._button(bar, "Reload", self._reload).pack(side="left", padx=(8, 0))
        if self.role == ROLE_USER:
            self._button(bar, "Speak as this", self._activate).pack(side="right")

    # -- small widgets ----------------------------------------------------

    def _button(self, parent: tk.Misc, text: str, command: Any, *, variant: str = "secondary") -> tk.Button:
        """A themed sidebar button, **placed** in its row.

        The button is packed here so a caller that does not chain its own
        ``.pack(...)`` still shows the control: an unplaced Tk widget has no
        geometry manager, so it is never drawn and can never be clicked -- which
        is exactly how the whole manager sidebar (New…, Add the chosen, the body
        builder…) ended up invisible. A caller that packs it again only re-applies
        the geometry (``pack`` is idempotent), so both call styles work.
        """
        from gui.components import button

        widget = button(
            parent, text, command, theme=self.theme, typography=self._type,
            variant=variant, small=True,
        )
        if not widget.winfo_manager():
            widget.pack(side="left", padx=(0, 6))
        return widget

    def _listbox(
        self, parent: tk.Misc, *, height: int, on_select: Callable[[int], None],
        framed: bool = True,
    ) -> tk.Listbox:
        theme = self.theme
        holder: tk.Misc = parent
        if framed:
            holder = tk.Frame(parent, bg=theme.surface)
            holder.pack(fill="x", pady=2)
        widget = tk.Listbox(
            holder, height=height, bg=theme.surface_raised, fg=theme.foreground,
            selectbackground=theme.primary, selectforeground=theme.primary_foreground,
            relief="flat", borderwidth=0, highlightthickness=0,
            font=self._list_type.font("2xs"), activestyle="none",
        )
        widget.pack(side="left", fill="x", expand=True)
        scroll = tk.Scrollbar(holder, command=widget.yview)
        widget.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        widget.bind("<<ListboxSelect>>", lambda _e: self._dispatch_select(widget, on_select))
        return widget

    def _dispatch_select(self, widget: tk.Listbox, on_select: Callable[[int], None]) -> None:
        if getattr(self, "_suppress_select", False):
            return
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

    # -- the character list ------------------------------------------------

    def _rows(self) -> list[Any]:
        search = self._search_var.get().strip() if hasattr(self, "_search_var") else ""
        if self.role == ROLE_USER:
            return self.service.persona_rows(search=search)
        return self.service.character_rows(search=search)

    def _refresh_list(self) -> None:
        widget = getattr(self, "_list", None)
        if widget is None:
            return
        previous = getattr(self, "_suppress_select", False)
        self._suppress_select = True
        try:
            widget.delete(0, "end")
            self._ids = []
            for row in self._rows():
                self._ids.append(row.character_id)
                widget.insert("end", row.label())
        finally:
            self._suppress_select = previous
        if self._selected and self._selected in self._ids:
            self._mark_selected(self._ids.index(self._selected))

    def _mark_selected(self, index: int) -> None:
        widget = getattr(self, "_list", None)
        if widget is None:
            return
        previous = getattr(self, "_suppress_select", False)
        self._suppress_select = True
        try:
            widget.selection_clear(0, "end")
            widget.selection_set(index)
            widget.see(index)
        except tk.TclError:
            pass
        finally:
            self._suppress_select = previous

    def _on_list_selected(self, index: int) -> None:
        if index < len(self._ids):
            self._select(self._ids[index])

    def _select(self, character_id: str) -> None:
        self._selected = character_id
        self._load(character_id)
        if self._selected in self._ids:
            self._mark_selected(self._ids.index(self._selected))

    def _selected_id(self) -> str:
        return self._selected

    # -- loading the profile ----------------------------------------------

    def _load(self, character_id: str) -> None:
        try:
            bundle = self.service.character(character_id)
        except Exception as error:
            self._set_status(str(error))
            return
        character = bundle.character
        self._vars["name"].set(character.name)
        self._vars["nickname"].set(character.nickname)
        self._vars["age"].set("" if character.fictional_age is None else str(character.fictional_age))
        self._vars["gender"].set(character.gender)
        self._vars["nationality"].set(character.nationality)
        self._vars["languages"].set(", ".join(character.languages))
        self._combo_set("kinship", character.kinship or "")
        self._vars["relationship_style"].set(character.relationship_style)
        self._set_text("biography", character.biography)
        self._set_text("appearance", character.appearance)
        self._combo_set("default_mode", character.default_mode or "general")

        personality = self.service.personality(character_id)
        for key, _label, _low, _high in (*DIAL_ROWS, *REACH_DIAL_ROWS):
            self._dial_scales[key].set(float(personality.get(key, 0.5)))
        self._combo_set("response_length", personality.get("response_length", "balanced"))
        self._vars["dialect"].set(personality.get("dialect", ""))
        self._set_text("instructions", personality.get("instructions", ""))

        voice = bundle.voice
        self._combo_set("voice_engine", (voice.engine if voice else "") or "auto")
        self._combo_set("model_key", character.model_key or "")
        self._check_vars["adult_eligible"].set(bool(character.adult_eligible))
        active_var = self._check_vars.get("persona_active")
        if active_var is not None:
            active_var.set(bool(getattr(character, "is_active", False)))

        self._name_label.configure(text=character.name or "(unnamed)")
        self._subtitle_label.configure(text=self._subtitle_for(character))
        self._set_model_note()
        self._refresh_interests()
        self._refresh_boundaries()
        self._refresh_avatar()

    def _subtitle_for(self, character: Any) -> str:
        bits = [
            f"{character.fictional_age}" if character.fictional_age else "",
            character.gender,
            character.nationality,
            " · ".join(character.languages) if character.languages else "",
            character.relationship_style,
        ]
        line = " · ".join(b for b in bits if b)
        active = "speaking as this" if getattr(character, "is_active", False) else ""
        return " · ".join(b for b in (line, active) if b) or "A local character"

    def _set_text(self, key: str, value: str) -> None:
        widget = self._text_widgets.get(key)
        if widget is None:
            return
        widget.delete("1.0", "end")
        widget.insert("1.0", value or "")

    def _set_model_note(self) -> None:
        choices = self.service.model_choices()
        if not choices:
            self._model_note.configure(
                text="No language model is installed. Download one from the Model Manager to chat."
            )
            return
        ready = sum(1 for c in choices if c.available)
        self._model_note.configure(text=f"{ready} of {len(choices)} models are installed and ready here.")

    def _refresh_interests(self) -> None:
        widget = getattr(self, "_interest_list", None)
        if widget is None:
            return
        widget.delete(0, "end")
        self._interest_ids = []
        try:
            bundle = self.service.character(self._selected)
        except Exception:
            return
        for item in bundle.profile_items:
            self._interest_ids.append(item.record_id)
            widget.insert("end", item.label)

    def _refresh_boundaries(self) -> None:
        widget = getattr(self, "_boundary_list", None)
        if widget is None:
            return
        widget.delete(0, "end")
        self._boundary_ids = []
        for boundary in self.service.boundaries(self._selected):
            self._boundary_ids.append(boundary.record_id)
            widget.insert("end", boundary.value)

    # -- avatar -----------------------------------------------------------

    def _refresh_avatar(self) -> None:
        canvas = getattr(self, "_avatar_canvas", None)
        if canvas is None:
            return
        try:
            canvas.delete("all")
        except tk.TclError:
            return
        path = self.service.avatar_path(self._selected) if self._selected else None
        if path is not None:
            try:
                self._avatar_photo = tk.PhotoImage(file=str(path))
                canvas.create_image(38, 38, image=self._avatar_photo)
                return
            except Exception:
                self._avatar_photo = None
        name = self._vars.get("name")
        initial = (name.get().strip()[:1] if name is not None else "") or "❤"
        canvas.create_oval(2, 2, 74, 74, fill=self.theme.primary, outline="")
        canvas.create_text(
            38, 39, text=initial.upper(), fill=self.theme.primary_foreground,
            font=self._type.font("heading3", weight="bold"),
        )

    def _upload_avatar(self) -> None:
        character_id = self._selected_id()
        if not character_id:
            self._set_status(f"Select a {self._plural()} first.")
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
            result = self.service.set_avatar_from_file(character_id, path)
        except Exception as error:
            self._set_status(f"Could not set the picture: {error}")
            return
        self._set_status(result.describe())
        self._refresh_avatar()
        self._refresh_list()
        self._notify(character_id)

    def _generate_avatar(self) -> None:
        character_id = self._selected_id()
        if not character_id:
            self._set_status(f"Select a {self._plural()} first.")
            return
        backend = self._combo_value("avatar_backend") or "procedural"
        try:
            result = self.service.generate_avatar(character_id, backend=backend)
        except Exception as error:
            self._set_status(f"Could not generate the picture: {error}")
            return
        self._set_status(result.describe())
        self._refresh_avatar()
        self._refresh_list()
        self._notify(character_id)

    def _clear_avatar(self) -> None:
        character_id = self._selected_id()
        if not character_id:
            return
        try:
            self.service.clear_avatar(character_id)
        except Exception as error:
            self._set_status(str(error))
            return
        self._set_status("Picture cleared.")
        self._refresh_avatar()
        self._refresh_list()
        self._notify(character_id)

    # -- profile list actions ---------------------------------------------

    def _add_interest(self) -> None:
        label = self._interest_entry.get().strip()
        if not label or not self._selected:
            return
        self._interest_entry.delete(0, "end")
        try:
            self.service.add_interest(self._selected, label)
            self._refresh_interests()
        except Exception as error:
            self._set_status(str(error))

    def _remove_interest(self) -> None:
        selection = self._interest_list.curselection()
        if not selection or selection[0] >= len(self._interest_ids):
            return
        self.service.remove_interest(self._selected, self._interest_ids[selection[0]])
        self._refresh_interests()

    def _add_boundary(self) -> None:
        value = self._boundary_entry.get().strip()
        if not value or not self._selected:
            return
        self._boundary_entry.delete(0, "end")
        try:
            self.service.add_boundary(self._selected, value)
            self._refresh_boundaries()
        except Exception as error:
            self._set_status(str(error))

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
            "adult_eligible": bool(self._check_vars["adult_eligible"].get()),
            "persona_active": self._persona_active(),
        }

    def _persona_active(self) -> bool:
        variable = self._check_vars.get("persona_active")
        return bool(variable.get()) if variable is not None else False

    def _save(self) -> None:
        if not self._selected:
            self._set_status(f"Select a {self._plural()} first.")
            return
        try:
            values = self._gather()
        except ValueError as error:
            self._set_status(str(error))
            return
        character_id = self._selected
        try:
            self.service.update_character(
                character_id,
                name=values["name"], nickname=values["nickname"],
                fictional_age=values["fictional_age"], gender=values["gender"],
                nationality=values["nationality"], languages=values["languages"],
                kinship=values["kinship"], relationship_style=values["relationship_style"],
                biography=values["biography"], appearance=values["appearance"],
                default_mode=values["default_mode"], model_key=values["model_key"],
            )
            self.service.set_personality(character_id, values["personality"])
            self.service.set_voice(character_id, engine=values["voice_engine"])
            if self.role == ROLE_USER and values.get("persona_active"):
                self.service.activate_persona(character_id)
            if values["adult_eligible"]:
                self.service.confirm_eligibility(character_id)
        except Exception as error:
            self._set_status(f"Could not save: {error}")
            return
        self._set_status("Saved.")
        self._refresh_list()
        self._load(character_id)
        self._notify(character_id)

    def _reload(self) -> None:
        if self._selected:
            self._load(self._selected)
            self._set_status("Reloaded from the database.")

    def _create(self) -> None:
        from gui.romance_editor import CharacterEditorDialog

        dialog = CharacterEditorDialog(
            self.window, self.service, character_id="", role=self.role,
            theme=self.theme, font=self.font, mono=self.mono,
            overrides=self._overrides, list_overrides=self._list_overrides,
            on_saved=self._on_created,
        )
        self._children = getattr(self, "_children", [])
        self._children.append(dialog)

    def _on_created(self, character_id: str) -> None:
        self._selected = character_id
        self._refresh_list()
        if character_id:
            self._select(character_id)
        self._notify(character_id)

    def _open_body_studio(self) -> None:
        """Open the body builder for the selected character (one window per manager)."""
        character_id = self._selected_id()
        if not character_id:
            self._set_status(f"Select a {self._plural()} first.")
            return
        from gui.romance_body import BodyStudioDialog

        existing = getattr(self, "_body_studio", None)
        if existing is not None and not getattr(existing, "_closed", True):
            try:
                existing.window.lift()
                return
            except Exception:
                pass
        dialog = BodyStudioDialog(
            self.window, self.service, character_id, role=self.role,
            theme=self.theme, font=self.font, mono=self.mono,
            overrides=self._overrides, list_overrides=self._list_overrides,
            on_changed=self._on_body_changed,
        )
        self._body_studio = dialog
        self._children = getattr(self, "_children", [])
        self._children.append(dialog)

    def _on_body_changed(self, character_id: str) -> None:
        """A body render was stored as the picture: refresh this window's view."""
        self._refresh_avatar()
        self._refresh_list()
        self._notify(character_id)

    def _edit_in_editor(self) -> None:
        if not self._selected:
            self._set_status(f"Select a {self._plural()} first.")
            return
        from gui.romance_editor import CharacterEditorDialog

        dialog = CharacterEditorDialog(
            self.window, self.service, character_id=self._selected, role=self.role,
            theme=self.theme, font=self.font, mono=self.mono,
            overrides=self._overrides, list_overrides=self._list_overrides,
            on_saved=self._on_created,
        )
        self._children = getattr(self, "_children", [])
        self._children.append(dialog)

    def _duplicate(self) -> None:
        if not self._selected:
            self._set_status(f"Select a {self._plural()} first.")
            return
        try:
            bundle = self.service.duplicate_character(self._selected)
        except Exception as error:
            self._set_status(f"Could not duplicate: {error}")
            return
        self._selected = bundle.id
        self._refresh_list()
        self._select(bundle.id)
        self._set_status(f"Duplicated as {bundle.name}.")
        self._notify(bundle.id)

    def _activate(self) -> None:
        if not self._selected:
            self._set_status("Select a persona first.")
            return
        try:
            self.service.activate_persona(self._selected)
        except Exception as error:
            self._set_status(f"Could not activate the persona: {error}")
            return
        self._refresh_list()
        self._load(self._selected)
        self._set_status(f"Now speaking as {self.service.active_persona_name()}.")
        self._notify(self._selected)

    def _export(self) -> None:
        if not self._selected:
            self._set_status(f"Select a {self._plural()} first.")
            return
        path = filedialog.asksaveasfilename(
            parent=self.window, title=f"Export the {self._plural()}",
            defaultextension=".json", filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.service.export_character(self._selected, path)
        except Exception as error:
            self._set_status(str(error))
            return
        self._set_status(f"Exported to {path}.")

    def _import(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.window, title=f"Import a {self._plural()}",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            bundle = self.service.import_character(path)
        except Exception as error:
            self._set_status(f"Could not import: {error}")
            return
        self._selected = bundle.id
        self._refresh_list()
        self._select(bundle.id)
        self._set_status(f"Imported {bundle.name}.")
        self._notify(bundle.id)

    def _delete(self) -> None:
        if not self._selected:
            self._set_status(f"Select a {self._plural()} first.")
            return
        choice = _ask_retention(self.window, self.theme, self._type)
        if choice is None:
            return
        try:
            self.service.delete_character(self._selected, choice=choice)
        except Exception as error:
            self._set_status(str(error))
            return
        self._selected = ""
        self._refresh_list()
        if self._ids:
            self._select(self._ids[0])
        else:
            self._clear_profile()
        self._set_status("Deleted.")
        self._notify("")

    def _clear_profile(self) -> None:
        self._name_label.configure(text="")
        self._subtitle_label.configure(text="")
        for key, var in self._vars.items():
            if key in ("voice_engine", "response_length", "default_mode"):
                continue
            var.set("")
        self._set_text("biography", "")
        self._set_text("appearance", "")
        self._set_text("instructions", "")
        for key in ("adult_eligible", "persona_active"):
            flag = self._check_vars.get(key)
            if flag is not None:
                flag.set(False)
        self._refresh_interests()
        self._refresh_boundaries()
        self._refresh_avatar()

    # -- the ready-made library -------------------------------------------

    def _refresh_library(self) -> None:
        menu = getattr(self, "_library_category_menu", None)
        widget = getattr(self, "_library_list", None)
        if menu is None or widget is None:
            return
        if self.role == ROLE_USER:
            categories = self.service.persona_library_categories()
        else:
            categories = self.service.library_categories()
        keys = [""] + [key for key, _t, _i, _s in categories]
        labels = ["All archetypes"] + [title for _k, title, _i, _s in categories]
        self._library_keys = keys
        submenu = menu["menu"]
        submenu.delete(0, "end")
        for key, label in zip(keys, labels, strict=True):
            submenu.add_command(label=label, command=lambda k=key: self._library_category_var.set(k))
        if self._library_category_var.get() not in keys:
            self._library_category_var.set("")

        search = self._library_search_var.get().strip() if hasattr(self, "_library_search_var") else ""
        current = self._library_category_var.get()
        rows = self.service.library_rows(category=current, search=search, role=self.role)
        self._library_rows = list(rows)
        previous = getattr(self, "_suppress_select", False)
        self._suppress_select = True
        try:
            widget.delete(0, "end")
            for row in rows:
                widget.insert("end", row.label())
        finally:
            self._suppress_select = previous
        if not rows:
            self._library_preview.configure(text="No character matches that filter.")

    def _on_library_selected(self, index: int) -> None:
        rows = getattr(self, "_library_rows", [])
        if index >= len(rows):
            return
        row = rows[index]
        self._library_preview.configure(text=row.describe())
        if row.chat_ready:
            self._library_preview.configure(
                text=row.describe() + f"\n\nAlready in your library ({row.character_id})."
            )

    def _add_library(self) -> None:
        selection = self._library_list.curselection()
        rows = getattr(self, "_library_rows", [])
        if not selection or selection[0] >= len(rows):
            self._set_status("Choose a character from the library first.")
            return
        name = rows[selection[0]].name
        try:
            if self.role == ROLE_USER:
                bundle = self.service.add_persona_library_character(name)
            else:
                bundle = self.service.add_library_character(name)
        except Exception as error:
            self._set_status(str(error))
            return
        self._selected = bundle.id
        self._refresh_list()
        self._refresh_library()
        self._select(bundle.id)
        verb = "now speaking as" if self.role == ROLE_USER else "added"
        self._set_status(f"{bundle.name} {verb}.")
        self._notify(bundle.id)

    def _add_library_group(self) -> None:
        key = self._library_category_var.get()
        try:
            if self.role == ROLE_USER:
                count = self.service.seed_persona_library_category(key)
            else:
                count = self.service.seed_library_category(key)
        except Exception as error:
            self._set_status(str(error))
            return
        self._refresh_list()
        self._refresh_library()
        if self._ids and not self._selected:
            self._select(self._ids[0])
        self._set_status(f"Added {count} character(s).")
        self._notify(self._selected)

    # -- helpers ----------------------------------------------------------

    def _notify(self, character_id: str) -> None:
        if self._on_changed is not None:
            self._on_changed(character_id)

    def _set_status(self, text: str) -> None:
        try:
            self._status.configure(text=text)
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
        for child in getattr(self, "_children", []):
            try:
                child._close()
            except Exception:
                pass
        try:
            self.window.destroy()
        finally:
            if self._on_close is not None:
                self._on_close()


#: The window's two names, so a caller reads them the same way the sidebar does.
MANAGER_TITLES = {
    ROLE_USER: TITLE_PERSONA_MANAGER,
    ROLE_COMPANION: TITLE_CHARACTER_MANAGER,
}


__all__ = [
    "MANAGER_TITLES",
    "TITLE_CHARACTER_MANAGER",
    "TITLE_PERSONA_MANAGER",
    "CharacterManagerDialog",
]
