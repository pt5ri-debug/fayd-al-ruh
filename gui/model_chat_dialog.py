"""A dedicated chat window for one answering model, with its own identity.

The composer's model row switches the *main* conversation's mouth; this window is
the other half the user asked for: pressing a model's chip opens **that model's
own chat**, with three things that belong to it alone, down the left side:

1. the **character's logo and name** -- shown as a badge and changeable (pick any
   picture, or rename the character);
2. the **reading voice** its answers are spoken with -- exclusive to this model,
   and remembered until the user changes it again;
3. the **language model** it answers with -- chosen from the project's full
   catalog (every GGUF model any mouth has downloaded, plus the trained
   checkpoint), not only the mouth's own engine.

The window is built the way every other window in the project is built -- the
same :class:`~gui.app.Theme` tokens, :class:`~gui.chat_composer.RoundedBox`, the
shared :class:`~gui.chat_composer.ChatComposer`, the shared
:class:`~gui.window_controls.WindowControls` trio, a centred conversation column,
and a worker thread whose result is handed back through a queue the UI thread
drains (Tk is not thread-safe). The logic lives in
:class:`gui.model_chat.ModelChatService`, so the whole workflow is tested with no
display; this module is only the view.

Tk is imported lazily with the rest of the window, so the CLI never touches it.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Callable
from functools import partial
from tkinter import filedialog
from tkinter import font as tkfont
from typing import Any

from core.reading import clip_is_silent

from gui import markdown as md
from gui import rich_text
from gui.app import RADIUS_LARGE, RoundedBox
from gui.chat_composer import ChatComposer
from gui.image_preview import load_scaled
from gui.message_footer import MessageFooter, strip_markdown
from gui.model_chat import ModelChatService, mouth_title
from gui.typography import resolve
from gui.window_controls import fit_geometry, install_window_controls

#: The transcript reads as a centred column, capped like the chat window's.
_COLUMN_MAX = 860

#: The logo badge's side length, in pixels (the sidebar's identity block).
LOGO_SIZE = 84

#: The sidebar is this wide; the conversation column takes the rest.
SIDEBAR_WIDTH = 300

#: The window's opening size. The height is deliberately 90% of the window's
#: original 720px, so the seven model windows sit a little shorter on screen; it is
#: capped to the usable desktop so it still fits a small display, and the user can
#: grow it to the work area with the title bar's maximize control. The minimum size
#: is unchanged.
DEFAULT_WIDTH = 1180
DEFAULT_HEIGHT = 648
MIN_WIDTH = 920
MIN_HEIGHT = 620


class ModelChatDialog:
    """One mouth's own window: its logo, its voice, its model, and its chat."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        service: ModelChatService,
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
        self._logo_photo: tk.PhotoImage | None = None
        # Keeps each small avatar's image alive with its own widget (Tk holds only
        # the image name, so a dropped PhotoImage would paint blank).
        self._avatar_photos: dict[tk.Canvas, tk.PhotoImage] = {}
        self._recording: Any = None
        self._pump_after: str | None = None
        self._rows: list[tuple[str, str, tk.Frame, tk.Text, RoundedBox | None]] = []
        # The footers under the messages, so the one reading aloud shows "Stop".
        self._footers: list[MessageFooter] = []
        self._playing_footer: MessageFooter | None = None

        # Worker threads hand results back through a queue the UI thread drains on
        # its own clock: calling `after` from a worker thread races with the main
        # loop (Tk is not thread-safe), the pattern the rest of the project uses.
        self._results: queue.Queue[
            tuple[Callable[[Any, Exception | None], None], Any, Exception | None]
        ] = queue.Queue()

        self._question_var = tk.StringVar(value="")

        self.window = tk.Toplevel(parent)
        self.window.title(f"{service.name} — {mouth_title(service.mouth)}")
        self.window.configure(bg=theme.base)
        self.window.geometry(self._initial_geometry())
        self.window.minsize(MIN_WIDTH, MIN_HEIGHT)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)

        self._build()
        self._refresh_identity()
        self._refresh_voice_status()
        self._refresh_model_status()

    # -- layout -----------------------------------------------------------

    def _initial_geometry(self) -> str:
        """The opening geometry, kept inside the usable desktop (screen minus taskbar)."""
        return fit_geometry(self.window, DEFAULT_WIDTH, DEFAULT_HEIGHT, min_height=MIN_HEIGHT)

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
        self._header_title = tk.Label(
            titles, text=self.service.name, bg=theme.titlebar, fg=theme.contrast,
            font=self._type.font("heading3", weight="bold"), anchor="w",
        )
        self._header_title.pack(anchor="w")
        tk.Label(
            titles, text=f"{mouth_title(self.service.mouth)} — {self.service.blurb()}",
            bg=theme.titlebar, fg=theme.text_secondary, font=self._type.font("2xs"),
            anchor="w", justify="left", wraplength=760,
        ).pack(anchor="w", pady=(2, 0))
        self._window_controls = install_window_controls(
            self.window, inner, theme=theme, typography=self._type, on_close=self._close
        )
        tk.Frame(self.window, bg=theme.titlebar_edge, height=1).pack(fill="x")

    def _section_label(self, parent: tk.Misc, text: str) -> None:
        tk.Label(
            parent, text=text.upper(), bg=self.theme.surface, fg=self.theme.text_dim,
            font=self._type.font("2xs", weight="bold"), anchor="w",
        ).pack(fill="x", padx=14, pady=(12, 2))

    def _button(self, parent: tk.Misc, text: str, command: Any, *, small: bool = False) -> tk.Button:
        return tk.Button(
            parent, text=text, command=command, relief="flat", borderwidth=0,
            highlightthickness=0, bg=self.theme.surface_raised, fg=self.theme.foreground,
            activebackground=self.theme.interactive_active, activeforeground=self.theme.contrast,
            font=self._type.font("2xs", weight="bold"), padx=10, pady=4 if small else 6,
            cursor="hand2", anchor="w",
        )

    # -- sidebar ----------------------------------------------------------

    def _build_sidebar(self, parent: tk.Frame) -> None:
        theme = self.theme
        shell = tk.Frame(parent, bg=theme.surface, width=SIDEBAR_WIDTH)
        shell.grid(row=0, column=0, sticky="nsew")
        shell.grid_propagate(False)

        canvas = tk.Canvas(shell, bg=theme.surface, highlightthickness=0, width=SIDEBAR_WIDTH)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar = tk.Scrollbar(shell, command=canvas.yview)
        # The sidebar scrollbar is always visible (not shown only on overflow), so
        # the left column reads as one scrollable list and the model/voice options
        # are never a dead end on a short window.
        scrollbar.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scrollbar.set)
        self._sidebar_canvas = canvas
        self._sidebar_scrollbar = scrollbar

        sidebar = tk.Frame(canvas, bg=theme.surface)
        self._sidebar = sidebar
        window = canvas.create_window((0, 0), window=sidebar, anchor="nw")
        self._sidebar_window = window
        sidebar.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(window, width=e.width),
        )
        for widget in (canvas, sidebar):
            widget.bind("<MouseWheel>", self._on_sidebar_wheel)
            widget.bind("<Button-4>", self._on_sidebar_wheel)
            widget.bind("<Button-5>", self._on_sidebar_wheel)

        self._build_identity(sidebar)
        self._build_voice_section(sidebar)
        self._build_model_section(sidebar)

    # -- identity (logo + name) ------------------------------------------

    def _build_identity(self, sidebar: tk.Frame) -> None:
        theme = self.theme
        self._section_label(sidebar, "Character")
        block = tk.Frame(sidebar, bg=theme.surface)
        block.pack(fill="x", padx=14, pady=(0, 4))

        self._logo_canvas = tk.Canvas(
            block, width=LOGO_SIZE, height=LOGO_SIZE, bg=theme.surface,
            highlightthickness=0, cursor="hand2",
        )
        self._logo_canvas.pack(anchor="w")
        self._logo_canvas.bind("<Button-1>", lambda _e: self._on_change_logo())

        self._name_var = tk.StringVar(value=self.service.name)
        self._name_entry = tk.Entry(
            block, textvariable=self._name_var, bg=theme.base, fg=theme.foreground,
            insertbackground=theme.foreground, relief="flat", highlightthickness=0,
            font=self._type.font("body", weight="bold"), width=18,
        )
        self._name_entry.pack(fill="x", pady=(6, 4))
        self._name_entry.bind("<Return>", lambda _e: self._on_rename())
        self._name_entry.bind("<FocusOut>", lambda _e: self._on_rename())

        actions = tk.Frame(block, bg=theme.surface)
        actions.pack(fill="x")
        self._logo_button = self._button(
            actions, "Change the logo…", self._on_change_logo, small=True
        )
        self._logo_button.pack(fill="x", pady=(0, 4))
        self._logo_reset_button = self._button(
            actions, "Reset to the default badge", self._on_reset_logo, small=True
        )
        self._logo_reset_button.pack(fill="x")

    def _draw_default_badge(self) -> None:
        """Paint the fallback badge: a filled disc with the name's first letter."""
        canvas = self._logo_canvas
        canvas.delete("all")
        theme = self.theme
        canvas.create_oval(2, 2, LOGO_SIZE - 2, LOGO_SIZE - 2, fill=theme.primary, outline="")
        initial = (self.service.name.strip()[:1] or "ف").upper()
        canvas.create_text(
            LOGO_SIZE // 2, LOGO_SIZE // 2 + 1, text=initial,
            fill=theme.primary_foreground, font=self._type.font("hero", weight="bold"),
        )

    def _refresh_identity(self) -> None:
        """Show the saved logo, or the default badge when none is set."""
        path = self.service.logo_path()
        if path is None:
            self._logo_photo = None
            self._draw_default_badge()
        else:
            photo = load_scaled(path, LOGO_SIZE)
            if photo is None:
                self._logo_photo = None
                self._draw_default_badge()
            else:
                self._logo_photo = photo
                self._logo_canvas.delete("all")
                self._logo_canvas.create_image(
                    LOGO_SIZE // 2, LOGO_SIZE // 2, image=photo
                )
        if self._name_var.get() != self.service.name:
            self._name_var.set(self.service.name)
        self._header_title.configure(text=self.service.name)
        self.window.title(f"{self.service.name} — {mouth_title(self.service.mouth)}")

    def _on_rename(self) -> None:
        wanted = self._name_var.get().strip()
        if wanted == self.service.name:
            return
        self.service.set_name(wanted)
        self._refresh_identity()

    def _on_change_logo(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.window,
            title=f"Choose a logo for {self.service.name}",
            filetypes=[
                ("Images", "*.png *.gif *.jpg *.jpeg *.webp *.ppm *.pgm"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self.service.set_logo(path)
        except ValueError as error:
            self._add_message("note", str(error))
            return
        self._refresh_identity()
        self._add_message("note", f"Logo updated for {self.service.name}.")

    def _on_reset_logo(self) -> None:
        self.service.clear_logo()
        self._refresh_identity()

    # -- voice ------------------------------------------------------------

    def _build_voice_section(self, sidebar: tk.Frame) -> None:
        theme = self.theme
        self._section_label(sidebar, "Reading voice")
        self._voice_var = tk.StringVar(value=self.service.voice_key())
        holder = tk.Frame(sidebar, bg=theme.surface)
        holder.pack(fill="x", padx=10)

        for section in self.service.voice_sections():
            self._voice_group(holder, section.title, indent=4)
            for header, group in section.groups:
                self._voice_group(holder, header, indent=10, dim=True)
                for voice in group:
                    suffix = f"  ({voice.gender})" if getattr(voice, "gender", "") else ""
                    label = voice.label() if hasattr(voice, "label") else voice.label_en
                    self._voice_radio(holder, f"{label}{suffix}", voice.key, indent=10)

        self._voice_note = tk.Label(
            sidebar, text="", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left", wraplength=SIDEBAR_WIDTH - 40,
        )
        self._voice_note.pack(fill="x", padx=14, pady=(4, 2))
        self._button(
            sidebar, "▷   Test the reading voice", self._on_test_voice, small=True
        ).pack(fill="x", padx=12, pady=(0, 4))

    def _voice_group(
        self, parent: tk.Misc, text: str, *, indent: int, dim: bool = False
    ) -> None:
        tk.Label(
            parent, text=text, bg=self.theme.surface,
            fg=self.theme.text_dim if dim else self.theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", padx=indent,
        ).pack(fill="x", pady=(4, 0))

    def _voice_radio(self, parent: tk.Misc, label: str, key: str, *, indent: int) -> None:
        tk.Radiobutton(
            parent, text=label, value=key, variable=self._voice_var,
            command=partial(self._on_voice_change, key),
            bg=self.theme.surface, fg=self.theme.foreground,
            selectcolor=self.theme.surface, activebackground=self.theme.surface,
            activeforeground=self.theme.foreground, font=self._type.font("xs"),
            anchor="w", padx=6, highlightthickness=0, bd=0, cursor="hand2",
        ).pack(fill="x", padx=indent, pady=0)

    def _on_voice_change(self, key: str) -> None:
        """Remember this mouth's voice -- exclusive to it -- and persist the choice."""
        try:
            chosen = self.service.set_voice(key)
        except ValueError as error:
            self._add_message("note", str(error))
            self._voice_var.set(self.service.voice_key())
            return
        self._voice_var.set(chosen)
        self._refresh_voice_status()
        self._add_message("note", f"Reading voice set to {self.service.voice_label(chosen)}.")

    def _refresh_voice_status(self) -> None:
        key = self.service.voice_key()
        label = self.service.voice_label(key)
        if self.service.reading_available():
            self._voice_note.configure(
                text=f"Answers here are read aloud with “{label}”. "
                "This choice belongs to this model only."
            )
        else:
            self._voice_note.configure(
                text="No speech engine is installed here, so nothing can be read aloud yet. "
                "The voice choice is still remembered."
            )

    def _on_test_voice(self) -> None:
        if self._busy:
            return
        self._start_work(
            lambda: self.service.read_aloud(
                f"This is {self.service.name} reading with the chosen voice."
            ),
            self._render_speech,
        )

    def _render_speech(self, clip: Any, error: Exception | None) -> None:
        if error is not None:
            self._add_message("note", f"Could not read aloud: {error}")
            return
        if clip_is_silent(clip):
            self._add_message(
                "note", "The engine produced no audio; install a speech engine to hear it."
            )

    # -- model ------------------------------------------------------------

    def _build_model_section(self, sidebar: tk.Frame) -> None:
        theme = self.theme
        self._section_label(sidebar, "Language model")
        holder = tk.Frame(sidebar, bg=theme.surface)
        holder.pack(fill="both", expand=True, padx=12)

        self._model_list = tk.Listbox(
            holder, bg=theme.base, fg=theme.foreground, relief="flat",
            highlightthickness=0, borderwidth=0, selectbackground=theme.interactive_active,
            selectforeground=theme.contrast, font=self._type.font("xs"), activestyle="none",
            exportselection=False, height=8, width=32,
        )
        self._model_list.pack(fill="both", expand=True)
        self._model_list.bind("<<ListboxSelect>>", self._on_model_select)
        self._model_keys: list[str] = []
        self._populate_models()

        actions = tk.Frame(sidebar, bg=theme.surface)
        actions.pack(fill="x", padx=12, pady=(6, 0))
        self._add_model_button = self._button(
            actions, "＋  Add a language model…", self._on_add_model, small=True
        )
        self._add_model_button.pack(fill="x", pady=(0, 4))
        self._download_model_button = self._button(
            actions, "⇩  Download the chosen model", self._on_download_model, small=True
        )
        self._download_model_button.pack(fill="x", pady=(0, 4))
        self._remove_model_button = self._button(
            actions, "✕  Forget this model", self._on_remove_model, small=True
        )
        self._remove_model_button.pack(fill="x")
        # The retrieval mouth (Jarvis, from the sources) has no language model to
        # add or forget, so those two actions are removed rather than shown dead.
        if not self._can_add_model():
            self._add_model_button.pack_forget()
            self._remove_model_button.pack_forget()

        self._model_note = tk.Label(
            sidebar, text="", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left",
            wraplength=SIDEBAR_WIDTH - 40,
        )
        self._model_note.pack(fill="x", padx=14, pady=(4, 12))

    def _populate_models(self) -> None:
        """Fill the model list from what this window can offer, marking the choice."""
        self._model_list.delete(0, "end")
        self._model_keys = []
        current = self.service.model_key()
        for index, model in enumerate(self.service.model_choices()):
            self._model_list.insert("end", model.label())
            self._model_keys.append(model.key)
            if current and model.key == current:
                self._model_list.selection_clear(0, "end")
                self._model_list.selection_set(index)
                self._model_list.see(index)
        self._refresh_model_buttons()

    def _refresh_model_buttons(self) -> None:
        """Enable each model action only when it applies to the chosen model."""
        key = self.service.model_key()
        try:
            self._download_model_button.configure(
                state="normal" if self._can_download(key) else "disabled"
            )
            self._remove_model_button.configure(
                state="normal" if self._can_remove(key) else "disabled"
            )
        except (AttributeError, tk.TclError):
            pass

    def _can_download(self, key: str) -> bool:
        """True when the service can fetch the chosen model's weights."""
        check = getattr(self.service, "can_download", None)
        return bool(check(key)) if callable(check) else False

    def _can_remove(self, key: str) -> bool:
        """True when the chosen model is one the user added here."""
        check = getattr(self.service, "can_remove", None)
        return bool(check(key)) if callable(check) else False

    def _can_add_model(self) -> bool:
        """True when this mouth runs a language model a file can be added to."""
        check = getattr(self.service, "can_add_model", None)
        return bool(check()) if callable(check) else True

    def _on_model_select(self, _event: tk.Event) -> None:
        selection = self._model_list.curselection()
        if not selection:
            return
        # Tk's `selection_set` adds to the selection, so the *most recent* index is
        # the one the user just chose (the same rule the Course window follows).
        index = max(selection)
        if index >= len(self._model_keys):
            return
        key = self._model_keys[index]
        if key == self.service.model_key():
            return
        try:
            self.service.set_model(key)
        except (ValueError, OSError) as error:
            self._add_message("note", f"Could not switch model: {error}")
            return
        self._refresh_model_status()
        self._refresh_model_buttons()
        self._add_message("note", f"{self.service.name} will answer with {self._model_label(key)}.")

    def _model_label(self, key: str) -> str:
        """A readable title for a model key, from what this window offers."""
        for model in self.service.model_choices():
            if model.key == key:
                return model.title
        return key

    def _on_add_model(self) -> None:
        """Add a GGUF file on this machine and start using it in this window."""
        path = filedialog.askopenfilename(
            parent=self.window,
            title=f"Choose a GGUF model for {self.service.name}",
            filetypes=[("GGUF models", "*.gguf"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            key = self.service.add_model(path)
        except (ValueError, OSError) as error:
            self._add_message("note", f"Could not add the model: {error}")
            return
        self._populate_models()
        self._refresh_model_status()
        self._add_message(
            "note",
            f"Added {self._model_label(key)} and made it active here. "
            "Ask a question to use it.",
        )

    def _on_remove_model(self) -> None:
        """Forget the chosen model when the user added it here."""
        key = self.service.model_key()
        if not self._can_remove(key):
            self._add_message(
                "note", "Only a model you added here can be forgotten; pick another one instead."
            )
            return
        title = self._model_label(key)
        self.service.remove_model(key)
        self._populate_models()
        self._refresh_model_status()
        self._add_message("note", f"Forgot {title}.")

    def _on_download_model(self) -> None:
        """Fetch the chosen model's weights, on a worker thread, with approval."""
        if self._busy:
            return
        key = self.service.model_key()
        if not self._can_download(key):
            self._add_message("note", "The chosen model has nothing to download.")
            return
        title = self._model_label(key)
        command = getattr(self.service, "download_command", lambda _key: "")(key)
        hint = f" (or run `{command}`)" if command else ""
        self._add_message("note", f"Downloading {title}… this can take a while{hint}.")
        self._start_work(lambda: self.service.download_model(key, approved=True), self._render_download)

    def _render_download(self, path: Any, error: Exception | None) -> None:
        if error is not None:
            self._add_message("note", f"Could not download the model: {error}")
            return
        self._populate_models()
        self._refresh_model_status()
        self._add_message("note", f"Model ready at {path}. Ask a question to use it.")

    def _refresh_model_status(self) -> None:
        status = self.service.model_status()
        status_fn = getattr(self.service, "activation_status", None)
        activation = status_fn() if callable(status_fn) else ""
        self._model_note.configure(text=f"{status}\n{activation}".strip())
        if not self.service.model_ready():
            self._model_note.configure(fg=self.theme.text_dim)
        else:
            self._model_note.configure(fg=self.theme.text_secondary)
        self._refresh_model_buttons()
        # The lookup switch follows the live mouth: switching to a language-model
        # mouth greys it, switching back to the sources re-enables it.
        self._refresh_web_toggle()

    # -- conversation -----------------------------------------------------

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
        # The chat scrollbar is always visible, so the conversation always reads as
        # a scrollable area even before the first message overflows it.
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
        self._add_message("note", self.service.blurb())

    def _build_composer(self, parent: tk.Frame) -> None:
        """The shared rounded composer card, with an optional microphone."""
        composer = ChatComposer(
            parent, theme=self.theme, typography=self._type,
            placeholder=f"Ask {self.service.name}…", on_submit=self._on_send,
            outer_bg=self.theme.base, with_attach=False, variable=self._question_var,
        )
        composer.wrap.pack(side="bottom", fill="x", padx=self._column_pad(), pady=(8, 12))
        self._composer = composer
        self._entry = composer.entry
        # The microphone: dictation through the project's own recorder, exactly as
        # the main chat window offers it.
        self._mic_button = composer.add_mic(self._toggle_mic, tip="Dictate with the microphone")
        if not self.service.microphone_available():
            self._mic_button.configure(state="disabled")
        # The internet-lookup switch, beside the microphone: one click turns web
        # search on or off for *this* window. Only the grounded sources mouth can
        # search, so a language-model window shows it greyed rather than lying.
        self._web_toggle = composer.add_web_toggle(
            self._on_web_toggled,
            initial=bool(getattr(self.service, "web_enabled", lambda: False)()),
            available=bool(getattr(self.service, "web_available", lambda: False)()),
            tip=self.service.web_status()
            if hasattr(self.service, "web_status") else "Internet lookups: on or off",
        )
        composer.focus()

    def _on_web_toggled(self, enabled: bool) -> None:
        """Turn this window's internet lookups on or off from the switch."""
        setter = getattr(self.service, "set_web", None)
        if setter is not None:
            setter(enabled)
        self._refresh_web_toggle()

    def _refresh_web_toggle(self) -> None:
        """Keep the switch in step with the live session."""
        toggle = getattr(self, "_web_toggle", None)
        if toggle is None:
            return
        available_fn = getattr(self.service, "web_available", None)
        enabled_fn = getattr(self.service, "web_enabled", None)
        toggle.set_state(
            bool(enabled_fn()) if enabled_fn is not None else False,
            available=bool(available_fn()) if available_fn is not None else False,
        )

    # -- geometry helpers --------------------------------------------------

    def _column_pad(self) -> int:
        try:
            width = self._transcript.winfo_width() or self._canvas.winfo_width()
        except (tk.TclError, AttributeError):
            width = _COLUMN_MAX
        if width <= _COLUMN_MAX:
            return 16
        return max(16, (width - _COLUMN_MAX) // 2)

    def _available_column(self) -> int:
        try:
            width = self._canvas.winfo_width()
        except (tk.TclError, AttributeError):
            return _COLUMN_MAX
        if width <= 2:
            return _COLUMN_MAX
        return max(160, min(_COLUMN_MAX, width - 2 * self._column_pad()))

    def _bubble_fonts(self) -> tuple[tkfont.Font, tkfont.Font]:
        return (
            tkfont.Font(font=self._type.font("body")),
            tkfont.Font(font=self._type.mono("code")),
        )

    def _bubble_content_width(self, text: str, *, role: str) -> int:
        normal, mono = self._bubble_fonts()
        widest = 0
        for line in md.iter_styled_lines(text):
            font = mono if line.in_code else normal
            widest = max(widest, font.measure(line.text))
        if role == "user":
            return max(80, min(widest, int(_COLUMN_MAX * 0.84) - 52))
        return max(120, min(widest, self._available_column()))

    def _bubble_text_width(self, text: str, *, role: str) -> int:
        normal, _mono = self._bubble_fonts()
        char_px = max(1, normal.measure("0"))
        # Round *down*: a Text width is whole characters, so rounding up can ask
        # for a few pixels more than the column and force a horizontal scroll.
        return max(1, self._bubble_content_width(text, role=role) // char_px)

    def _bubble_card_width(self, text: str) -> int:
        return self._bubble_content_width(text, role="user") + 52

    # -- transcript -------------------------------------------------------

    def _add_message(
        self, role: str, text: str, *, title: str = "", mono_body: bool = False
    ) -> tk.Text:
        theme = self.theme
        pad = self._column_pad()
        row = tk.Frame(self._transcript, bg=theme.base)
        row.pack(fill="x", padx=(pad, pad), pady=6)
        card: RoundedBox | None = None

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
            card.configure(width=self._bubble_card_width(text))
            widget.configure(width=self._bubble_text_width(text, role="user"))
        else:
            head = tk.Frame(row, bg=theme.base)
            head.pack(fill="x", pady=(0, 4))
            if role in ("assistant", "model"):
                self._avatar(head)
                padx = (6, 0)
            else:
                padx = (0, 0)
            tk.Label(
                head, text=title or (self.service.name if role != "note" else "Note"),
                bg=theme.base, fg=theme.text_dim, font=self._type.font("xs", weight="bold"),
                anchor="w",
            ).pack(side="left", padx=padx)
            widget = tk.Text(
                row, height=1, wrap="word", relief="flat", borderwidth=0, highlightthickness=0,
                bg=theme.base, fg=theme.foreground,
                font=self._type.mono("code") if mono_body else self._type.font("body"),
                padx=0, pady=0, cursor="xterm", takefocus=1,
            )
            rich_text.apply_body_spacing(widget, self._type)
            if not mono_body:
                self._insert_rich(widget, text)
            else:
                widget.insert("1.0", text)
            widget.pack(anchor="w")
            widget.configure(width=self._bubble_text_width(text, role=role))

        self._rows.append((role, text, row, widget, card))
        self._make_read_only(widget)
        for wheel in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(wheel, self._on_bubble_wheel, add="+")
        self._fit_height(widget)
        # Every message carries the same footer the main window draws -- Copy and a
        # Read aloud / Stop toggle -- so a per-model window reads like the chat window.
        footer = MessageFooter(
            row, text=strip_markdown(text), widget=widget, theme=theme,
            typography=self._type, on_toggle=self._toggle_footer_reading,
        )
        self._footers.append(footer)
        self._scroll_to_bottom()
        return widget

    def _toggle_footer_reading(self, footer: MessageFooter) -> None:
        """Read one message aloud, or stop the reading that footer started."""
        if self._playing_footer is not None and self._playing_footer is not footer:
            self.service.stop_reading()
            self._playing_footer.set_playing(False)
            self._playing_footer = None
        if footer.playing:
            self.service.stop_reading()
            footer.set_playing(False)
            self._playing_footer = None
            return
        if not self.service.reading_available():
            self._add_message(
                "note",
                "No speech engine is installed here, so nothing can be read aloud yet.",
            )
            return
        if self._busy:
            return
        self._playing_footer = footer
        footer.set_playing(True)
        self._start_work(
            partial(self.service.start_reading, footer.text),
            self._render_footer_reading,
        )

    def _render_footer_reading(self, clip: Any, error: Exception | None) -> None:
        if self._playing_footer is not None:
            self._playing_footer.set_playing(False)
            self._playing_footer = None
        if error is not None:
            self._add_message("note", f"Could not read aloud: {error}")
        elif clip_is_silent(clip):
            self._add_message(
                "note", "The engine produced no audio; install a speech engine to hear it."
            )

    def _avatar(self, parent: tk.Misc) -> None:
        """A small round badge: the saved logo when there is one, else an initial."""
        size = 24
        badge = tk.Canvas(parent, width=size, height=size, bg=self.theme.base, highlightthickness=0)
        path = self.service.logo_path()
        photo = load_scaled(path, size) if path is not None else None
        if photo is not None:
            # Keep a reference alive with the widget: Tk holds only the image
            # *name*, so a dropped PhotoImage would paint as a blank canvas.
            self._avatar_photos[badge] = photo
            badge.create_image(size // 2, size // 2, image=photo)
        else:
            badge.create_oval(1, 1, size - 1, size - 1, fill=self.theme.primary, outline="")
            badge.create_text(
                size // 2, size // 2 + 1,
                text=(self.service.name.strip()[:1] or "ف").upper(),
                fill=self.theme.primary_foreground,
                font=self._type.font("sm", weight="bold"),
            )
        badge.pack(side="left")

    def _insert_rich(self, widget: tk.Text, text: str) -> None:
        rich_text.insert_markdown(widget, text, theme=self.theme, typography=self._type)

    def _fit_height(self, widget: tk.Text) -> None:
        """Size a message's card to its whole content, so nothing is clipped.

        The card grows to hold every displayed line (the transcript scrolls when
        the conversation is longer than the window), so a long answer is fully
        readable in its own message rather than hidden behind an inner scrollbar.
        """
        widget.update_idletasks()
        measured = widget.count("1.0", "end", "displaylines")
        lines = int(measured[0]) if measured else 1
        widget.configure(height=max(lines, 1))

    @staticmethod
    def _make_read_only(widget: tk.Text) -> None:
        def block(event: tk.Event) -> str:
            if int(event.state) & 0x0004 or int(event.state) & 0x0008:
                return ""
            if event.keysym in {"Left", "Right", "Up", "Down", "Home", "End", "Prior", "Next"}:
                return ""
            return "break"

        for sequence in ("<Key>", "<<Paste>>", "<<Cut>>", "<<Clear>>"):
            widget.bind(sequence, block)

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
        for _role, _text, row, _widget, _card in self._rows:
            row.pack_configure(padx=(pad, pad))
        self._remetric_rows()

    def _remetric_rows(self) -> None:
        for role, text, _row, widget, card in self._rows:
            try:
                if card is not None:
                    card.configure(width=self._bubble_card_width(text))
                widget.configure(width=self._bubble_text_width(text, role=role))
            except tk.TclError:
                continue
            self._fit_height(widget)

    def _toggle_scrollbar(self) -> None:
        """Keep the transcript's scroll region honest after the content changes.

        The chat scrollbar is always visible, so this no longer shows or hides it;
        it only keeps the canvas's scroll region current, which is what lets the
        thumb reflect the real overflow. It is kept as a named step because every
        content change already calls it.
        """
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_mousewheel(self, event: tk.Event) -> None:
        delta = -1 * (event.delta // 120) if getattr(event, "delta", 0) else (
            -1 if getattr(event, "num", 0) == 4 else 1
        )
        self._canvas.yview_scroll(delta, "units")
        self._toggle_scrollbar()

    def _on_sidebar_wheel(self, event: tk.Event) -> None:
        delta = -1 * (event.delta // 120) if getattr(event, "delta", 0) else (
            -1 if getattr(event, "num", 0) == 4 else 1
        )
        self._sidebar_canvas.yview_scroll(delta, "units")

    def _scroll_to_bottom(self) -> None:
        self._canvas.update_idletasks()
        self._canvas.yview_moveto(1.0)
        self._toggle_scrollbar()

    # -- talking ----------------------------------------------------------

    def _on_send(self) -> None:
        if self._busy:
            return
        message = self._composer.text().strip()
        if not message:
            return
        self._composer.clear()
        self._add_message("user", message)
        self._start_work(lambda: self.service.ask(message), self._render_reply)

    def _render_reply(self, reply: Any, error: Exception | None) -> None:
        if error is not None:
            self._add_message("note", f"Could not answer: {error}")
            return
        text = getattr(reply, "text", "") or ""
        hint = getattr(reply, "hint", "") or ""
        if hint:
            text = f"{text}\n\n{hint}" if text else hint
        self._add_message("assistant", text or "(no answer)")
        self._refresh_model_status()

    # -- microphone -------------------------------------------------------

    def _toggle_mic(self) -> None:
        if self._busy:
            return
        if self._recording is not None:
            self._stop_mic()
        else:
            self._start_mic()

    def _start_mic(self) -> None:
        try:
            self._recording = self.service.start_recording()
        except (OSError, RuntimeError, ValueError) as error:
            self._add_message("note", f"The microphone is unavailable: {error}")
            return
        self._composer.set_mic_recording(True)
        self._add_message("note", "Recording… click the microphone again to stop.")

    def _stop_mic(self) -> None:
        recording = self._recording
        self._recording = None
        self._composer.set_mic_recording(False)
        if recording is None:
            return
        self._start_work(lambda: self.service.transcribe(recording), self._render_transcript)

    def _render_transcript(self, text: Any, error: Exception | None) -> None:
        if error is not None:
            self._add_message("note", f"Could not transcribe: {error}")
            return
        heard = (text or "").strip()
        if not heard:
            self._add_message("note", "Nothing was recognised in the recording.")
            return
        self._composer.set_text(heard)
        self._add_message("note", f"Heard: {heard}")

    # -- background work --------------------------------------------------

    def _start_work(
        self,
        job: Callable[[], Any],
        finish: Callable[[Any, Exception | None], None],
    ) -> None:
        """Run ``job`` on a worker thread, then hand its result back on the UI thread."""
        self._busy = True
        self._set_send_enabled(False)

        def run() -> None:
            try:
                result = job()
                self._results.put((finish, result, None))
            except Exception as error:
                self._results.put((finish, None, error))

        threading.Thread(target=run, daemon=True).start()
        self._schedule_pump()

    def _schedule_pump(self) -> None:
        """Queue one UI-thread drain of the worker queue, remembering its id.

        The id is kept so a closed window can cancel a pending drain; a stray
        ``after`` on a destroyed interpreter raises ``invalid command name``.
        """
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
        self._set_send_enabled(True)
        finish(result, error)

    def _set_send_enabled(self, enabled: bool) -> None:
        try:
            self._composer.set_send_enabled(enabled)
        except (AttributeError, tk.TclError):
            pass

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
        if self._recording is not None:
            try:
                self.service.stop_recording()
            except Exception:
                pass
            self._recording = None
        # End a reading in progress and drop every footer's bound command, so the
        # destroyed window is not held by a cycle only the cyclic collector could
        # break (which can run on a worker thread and deadlock on a Tcl call).
        try:
            self.service.stop_reading()
        except Exception:
            pass
        for footer in self._footers:
            footer.clear_commands()
        self._footers.clear()
        self._playing_footer = None
        try:
            self.window.destroy()
        finally:
            if self._on_close is not None:
                self._on_close()


__all__ = [
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "LOGO_SIZE",
    "MIN_HEIGHT",
    "MIN_WIDTH",
    "SIDEBAR_WIDTH",
    "ModelChatDialog",
]
