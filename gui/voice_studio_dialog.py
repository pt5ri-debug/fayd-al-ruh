"""A dedicated studio window for one speech engine: read a document aloud.

The composer's model row carries one chip per answering *model*
(:mod:`gui.model_chat_dialog`); this is its sibling the user asked for -- one chip
per **speech engine**, each opening that engine's own studio. Down the left side,
in the order asked for:

1. the **engine's logo and name** -- shown as a badge and changeable (pick any
   picture, or rename the studio);
2. the **reading voice** -- drawn from **only** the voices this engine really
   reports here, exclusive to this studio, and remembered until the user changes it
   again;
3. the **mood** (an emotional state: happy, sad, excited, rapturous, in love,
   angry, and the rest of the catalog) -- applied to the voice the moment it is
   chosen, and also exclusive to this studio.

Below them the studio holds a **document** and reads it aloud: *Open a document…*
loads it through the project's own reader (text, Word, PDF, and the optional OCR
path for a scanned file), the text appears in the transcript, and **Read aloud**
plays it through this engine with the chosen voice and mood. A microphone control
sits in the composer, so a dictated note can be read back too.

An **Engine setup** block closes the sidebar: it says honestly whether the engine
and its voices are installed here, and offers the one governed action that fixes
what is missing (*Install the engine*, a fetch-voice picker, or *Download the
voices*). This is what makes the studio play its *own* voice rather than fall back
to the system default: without the engine's voice a local studio has nothing it can
speak, and the block is how the user gets it.

The window is built the way every other window in the project is built -- the same
:class:`~gui.app.Theme` tokens, :class:`~gui.chat_composer.RoundedBox`, the shared
:class:`~gui.chat_composer.ChatComposer`, the shared
:class:`~gui.window_controls.WindowControls` trio, a centred column, and a worker
thread whose result is handed back through a queue the UI thread drains (Tk is not
thread-safe). The logic lives in :class:`gui.voice_studio.VoiceStudioService`, so
the whole workflow is tested with no display; this module is only the view.

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

from gui import rich_text
from gui.app import RoundedBox
from gui.chat_composer import ChatComposer
from gui.image_preview import load_scaled
from gui.message_footer import MessageFooter, strip_markdown
from gui.typography import resolve
from gui.voice_studio import DOCUMENT_SUFFIXES, VoiceStudioService
from gui.window_controls import fit_geometry, install_window_controls

#: The transcript reads as a centred column, capped like the chat window's.
_COLUMN_MAX = 860

#: The logo badge's side length, in pixels (the sidebar's identity block).
LOGO_SIZE = 84

#: The sidebar is this wide; the conversation column takes the rest.
SIDEBAR_WIDTH = 320

#: The window's opening size, capped to the usable desktop so it fits a small
#: display, and growable to the work area with the title bar's maximize control.
DEFAULT_WIDTH = 1180
DEFAULT_HEIGHT = 700
MIN_WIDTH = 940
MIN_HEIGHT = 620


class VoiceStudioDialog:
    """One speech engine's own window: its logo, its voice, its mood, its document."""

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        service: VoiceStudioService,
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
        self._avatar_photos: dict[tk.Canvas, tk.PhotoImage] = {}
        self._recording: Any = None
        self._clone_recording: Any = None
        self._pump_after: str | None = None
        self._rows: list[tuple[str, str, tk.Frame, tk.Text, RoundedBox | None]] = []
        # The footers under the messages, so the one reading aloud is painted as
        # "Stop" and every other as "Read aloud" (one reading at a time).
        self._footers: list[MessageFooter] = []
        self._playing_footer: MessageFooter | None = None

        self._results: queue.Queue[
            tuple[Callable[[Any, Exception | None], None], Any, Exception | None]
        ] = queue.Queue()

        self._question_var = tk.StringVar(value="")

        self.window = tk.Toplevel(parent)
        self.window.title(f"{service.name} — {service.title}")
        self.window.configure(bg=theme.base)
        self.window.geometry(self._initial_geometry())
        self.window.minsize(MIN_WIDTH, MIN_HEIGHT)
        self.window.transient(parent)
        self.window.protocol("WM_DELETE_WINDOW", self._close)

        self._build()
        self._refresh_identity()
        self._refresh_voice_status()
        self._refresh_mood_status()
        self._refresh_document_status()
        self._refresh_setup_status()

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
            titles, text=f"{self.service.title} — {self.service.blurb()}",
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
        self._build_own_voice_section(sidebar)
        self._build_mood_section(sidebar)
        self._build_document_section(sidebar)
        self._build_setup_section(sidebar)

    # -- identity (logo + name) ------------------------------------------

    def _build_identity(self, sidebar: tk.Frame) -> None:
        theme = self.theme
        self._section_label(sidebar, "Engine")
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
        initial = (self.service.name.strip()[:1] or "ص").upper()
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
                self._logo_canvas.create_image(LOGO_SIZE // 2, LOGO_SIZE // 2, image=photo)
        if self._name_var.get() != self.service.name:
            self._name_var.set(self.service.name)
        self._header_title.configure(text=self.service.name)
        self.window.title(f"{self.service.name} — {self.service.title}")

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
        self._voice_host = sidebar
        holder = tk.Frame(sidebar, bg=theme.surface)
        holder.pack(fill="x", padx=10)
        self._voice_holder = holder

        groups = self.service.voice_groups()
        for group in groups:
            self._voice_group(holder, group.label, indent=4)
            for voice in group.voices:
                suffix = f"  ({voice.gender})" if getattr(voice, "gender", "") else ""
                label = voice.label() if hasattr(voice, "label") else voice.label_en
                self._voice_radio(holder, f"{label}{suffix}", voice.key, indent=10)
        # The synthetic catalog is settings for the *system* studio only; a local
        # model studio is never offered it.
        for _key, label_en, voices in self.service.catalog_voices():
            self._voice_group(holder, label_en, indent=4)
            for voice in voices:
                self._voice_radio(holder, voice.label_en, voice.key, indent=10)

        if not groups and not self.service.catalog_voices():
            tk.Label(
                holder,
                text="No voice from this engine is installed here yet.",
                bg=theme.surface, fg=theme.text_dim, font=self._type.font("2xs"),
                anchor="w", justify="left", wraplength=SIDEBAR_WIDTH - 40,
            ).pack(fill="x", padx=10, pady=(2, 0))

        self._voice_note = tk.Label(
            sidebar, text="", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left",
            wraplength=SIDEBAR_WIDTH - 40,
        )
        self._voice_note.pack(fill="x", padx=14, pady=(4, 2))
        self._button(
            sidebar, "▷   Test the reading voice", self._on_test_voice, small=True
        ).pack(fill="x", padx=12, pady=(0, 4))

    def _voice_group(self, parent: tk.Misc, text: str, *, indent: int) -> None:
        tk.Label(
            parent, text=text, bg=self.theme.surface, fg=self.theme.text_secondary,
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
        """Remember this studio's voice -- exclusive to it -- and persist the choice."""
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
                text=f"Documents here are read with “{label}”. "
                "This choice belongs to this engine only."
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
            lambda: self.service.test_voice(),
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
            return
        self._add_message("note", "Played the sample reading.")

    # -- the user's own voice (VoiceTut clones from a clip) ---------------

    def _build_own_voice_section(self, sidebar: tk.Frame) -> None:
        """VoiceTut's own capability: speak *your* voice, from a clip.

        VoiceTut is a zero-shot cloning model, so it can read documents in the
        user's own voice once a short recording is given to it -- the "merge my
        voice into the model" the user asked for, done the way the model works.
        Two buttons: record a clip with the microphone, or import an audio file.
        A third removes an imported voice. The section is built only for the
        engine that can clone (see :meth:`VoiceStudioService.can_import_voices`).
        """
        if not self.service.can_import_voices():
            return
        theme = self.theme
        self._section_label(sidebar, "My voice (clone)")
        actions = tk.Frame(sidebar, bg=theme.surface)
        actions.pack(fill="x", padx=12)

        self._clone_button = self._button(
            actions, "🎙   Clone my voice (record)", self._on_clone_voice, small=True
        )
        self._clone_button.pack(fill="x", pady=(0, 4))
        self._import_button = self._button(
            actions, "＋   Import a voice file…", self._on_import_voice, small=True
        )
        self._import_button.pack(fill="x", pady=(0, 4))

        # Removing an imported voice: a menubutton of the ones the user added, so
        # the list is always what is really registered here.
        self._remove_menu = tk.Menubutton(
            actions, text="✕   Remove an imported voice", bg=theme.surface_raised,
            fg=theme.foreground, activebackground=theme.interactive_active,
            activeforeground=theme.contrast, relief="flat", borderwidth=0,
            highlightthickness=0, anchor="w", font=self._type.font("2xs", weight="bold"),
            cursor="hand2", padx=10, pady=4,
        )
        self._remove_menu.pack(fill="x")

        self._own_voice_note = tk.Label(
            sidebar, text="", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left",
            wraplength=SIDEBAR_WIDTH - 40,
        )
        self._own_voice_note.pack(fill="x", padx=14, pady=(4, 2))
        self._refresh_own_voice()

    def _refresh_own_voice(self) -> None:
        """Rebuild the remove menu and the note from the real imported voices."""
        if not hasattr(self, "_own_voice_note"):
            return
        names = self.service.imported_voice_names()
        menu = tk.Menu(
            self._remove_menu, tearoff=0, bg=self.theme.surface_raised,
            fg=self.theme.foreground, activebackground=self.theme.interactive_active,
            activeforeground=self.theme.contrast, font=self._type.font("xs"),
        )
        if names:
            for name in names:
                menu.add_command(
                    label=name, command=partial(self._on_remove_voice, name)
                )
            self._remove_menu.configure(state="normal")
        else:
            menu.add_command(label="(no imported voice yet)", state="disabled")
            self._remove_menu.configure(state="disabled")
        self._remove_menu.configure(menu=menu)
        if names:
            self._own_voice_note.configure(
                text="Imported voices: " + ", ".join(names)
                + ". Each is a reference clip this engine clones, and it is read with "
                "when you pick it in the voice list above."
            )
        else:
            self._own_voice_note.configure(
                text="Record a short clip or import an audio file (.wav, .mp3, .m4a, "
                ".flac, .ogg) and this engine reads documents in that voice. (An .onnx "
                "voice belongs to the Piper studio, not this one.)"
            )

    def _on_clone_voice(self) -> None:
        """Record the user's voice, or stop the recording and register it.

        A toggle like the microphone: the first click starts recording, the second
        stops it and registers the clip as a new voice of this engine.
        """
        if getattr(self, "_clone_recording", None) is not None:
            self._stop_clone_recording()
            return
        if self._busy:
            return
        try:
            recording = self.service.start_recording()
        except (OSError, RuntimeError, ValueError) as error:
            self._add_message("note", f"The microphone is unavailable: {error}")
            return
        self._clone_recording = recording
        self._composer.set_mic_recording(True)
        self._add_message(
            "note",
            "Recording your voice… click “Clone my voice” again (or the microphone) "
            "to stop; the clip becomes a voice this engine can read with.",
        )

    def _stop_clone_recording(self) -> None:
        """Finish the clone recording and register it."""
        recording = getattr(self, "_clone_recording", None)
        self._clone_recording = None
        self._composer.set_mic_recording(False)
        if recording is None:
            return
        self._start_work(
            lambda: self.service.clone_voice(recording.result(), name="My voice"),
            self._render_own_voice,
        )

    def _on_import_voice(self) -> None:
        """Import an audio file as a new voice of this engine."""
        if self._busy:
            return
        from core.audiocpp_speech import IMPORT_PICKER_SUFFIXES

        pattern = " ".join(f"*{suffix}" for suffix in IMPORT_PICKER_SUFFIXES)
        path = filedialog.askopenfilename(
            parent=self.window,
            title=f"Choose an audio clip for {self.service.name} to clone",
            filetypes=[
                ("Audio clips", pattern),
                ("WAV audio", "*.wav"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._start_work(
            partial(self.service.import_voice, path),
            self._render_own_voice,
        )

    def _on_remove_voice(self, name: str) -> None:
        if self._busy:
            return
        self._start_work(
            partial(self.service.remove_voice, name),
            self._render_own_voice_removed,
        )

    def _render_own_voice(self, voice: Any, error: Exception | None) -> None:
        if error is not None:
            self._add_message("note", f"Could not add the voice: {error}")
            return
        self._rebuild_voice_section()
        self._refresh_voice_status()
        self._refresh_own_voice()
        self._add_message(
            "note",
            f"Added “{voice.name}” as a voice of {self.service.name}; it is selected now.",
        )

    def _render_own_voice_removed(self, removed: Any, error: Exception | None) -> None:
        if error is not None:
            self._add_message("note", f"Could not remove the voice: {error}")
            return
        self._rebuild_voice_section()
        self._refresh_voice_status()
        self._refresh_own_voice()
        self._add_message(
            "note",
            "Removed the voice." if removed else "That voice was not found.",
        )

    # -- mood -------------------------------------------------------------

    def _build_mood_section(self, sidebar: tk.Frame) -> None:
        theme = self.theme
        self._section_label(sidebar, "Mood / state")
        holder = tk.Frame(sidebar, bg=theme.surface)
        holder.pack(fill="x", padx=10)

        # A menubutton of the *readable* labels; choosing one applies that state's
        # settings at once and remembers the choice for this engine alone.
        self._mood_menu = tk.Menubutton(
            holder, text="", bg=theme.surface_raised, fg=theme.foreground,
            activebackground=theme.interactive_active, activeforeground=theme.contrast,
            relief="flat", borderwidth=0, highlightthickness=0, anchor="w",
            font=self._type.font("xs"), cursor="hand2", padx=8, pady=4,
        )
        menu = tk.Menu(
            self._mood_menu, tearoff=0, bg=theme.surface_raised, fg=theme.foreground,
            activebackground=theme.interactive_active, activeforeground=theme.contrast,
            font=self._type.font("xs"),
        )
        for mood in self.service.moods():
            menu.add_command(
                label=mood.label(), command=partial(self._on_mood_change, mood.key)
            )
        self._mood_menu.configure(menu=menu)
        self._mood_menu.pack(fill="x")

        self._mood_note = tk.Label(
            sidebar, text="", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left",
            wraplength=SIDEBAR_WIDTH - 40,
        )
        self._mood_note.pack(fill="x", padx=14, pady=(4, 2))

    def _on_mood_change(self, key: str) -> None:
        """Apply the chosen mood's settings to this studio at once."""
        try:
            chosen = self.service.set_mood(key)
        except ValueError as error:
            self._add_message("note", str(error))
            return
        self._refresh_mood_status()
        self._add_message(
            "note", f"Mood set to {self.service.mood_label(chosen)} — applied at once."
        )

    def _refresh_mood_status(self) -> None:
        key = self.service.mood_key()
        self._mood_menu.configure(text=f"▾  {self.service.mood_label(key)}")
        self._mood_note.configure(
            text=f"Current mood: {self.service.mood_label(key)}. "
            + self.service.mood_support_note()
        )

    # -- document ---------------------------------------------------------

    def _build_document_section(self, sidebar: tk.Frame) -> None:
        self._section_label(sidebar, "Document")
        actions = tk.Frame(sidebar, bg=self.theme.surface)
        actions.pack(fill="x", padx=12)
        self._open_button = self._button(
            actions, "Open a document…", self._on_open_document, small=True
        )
        self._open_button.pack(fill="x", pady=(0, 4))
        self._read_button = self._button(
            actions, "▶   Read the document aloud", self._on_read_aloud, small=True
        )
        self._read_button.pack(fill="x", pady=(0, 4))
        self._save_button = self._button(
            actions, "Save the speech as WAV…", self._on_save_speech, small=True
        )
        self._save_button.pack(fill="x")

        self._document_note = tk.Label(
            sidebar, text="", bg=self.theme.surface, fg=self.theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left",
            wraplength=SIDEBAR_WIDTH - 40,
        )
        self._document_note.pack(fill="x", padx=14, pady=(4, 12))

    # -- engine setup (install / download) --------------------------------

    def _build_setup_section(self, sidebar: tk.Frame) -> None:
        """The honest "what is missing here, and how to get it" block.

        This is the block that answers the reported problem directly: a studio whose
        engine or voices are not installed says so and offers the one governed action
        that fixes it -- *Install the engine* (a Python extra, the system package, or
        audio.cpp's binary) or *Download the voices* (the model/voice weights). Both
        run on the worker thread, and the voices are re-listed when they succeed.
        """
        theme = self.theme
        self._section_label(sidebar, "Engine setup")
        holder = tk.Frame(sidebar, bg=theme.surface)
        holder.pack(fill="x", padx=12)

        self._setup_note = tk.Label(
            holder, text="", bg=theme.surface, fg=theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left",
            wraplength=SIDEBAR_WIDTH - 40,
        )
        self._setup_note.pack(fill="x", pady=(0, 6))

        self._setup_install_button = self._button(
            holder, "⬇   Install the engine", self._on_install_engine, small=True
        )
        self._setup_download_button = self._button(
            holder, "⬇   Download the voices", self._on_download_voices, small=True
        )
        self._setup_voice_var = tk.StringVar(value="")
        self._setup_voice_menu = tk.Menubutton(
            holder, text="", bg=theme.surface_raised, fg=theme.foreground,
            activebackground=theme.interactive_active, activeforeground=theme.contrast,
            relief="flat", borderwidth=0, highlightthickness=0, anchor="w",
            font=self._type.font("2xs"), cursor="hand2", padx=8, pady=4,
        )
        self._setup_voice_menu.configure(menu=tk.Menu(self._setup_voice_menu, tearoff=0))

    def _refresh_setup_status(self) -> None:
        """Re-read the engine's state and rebuild the setup controls to match."""
        status = self.service.setup_status()
        self._setup_note.configure(text=status.describe())
        ready = status.ready
        self._setup_install_button.configure(state="disabled" if ready or not status.can_install_runtime else "normal")
        self._setup_download_button.configure(state="disabled" if ready or not status.can_download_voices else "normal")
        # The install button is only packed when it is the action that helps.
        self._setup_install_button.pack_forget()
        self._setup_download_button.pack_forget()
        self._setup_voice_menu.pack_forget()
        if not ready and status.can_install_runtime:
            self._setup_install_button.pack(fill="x", pady=(0, 4))
        if not ready and status.can_download_voices:
            fetchable = self.service.fetchable_voices()
            if len(fetchable) > 1:
                menu = tk.Menu(self._setup_voice_menu, tearoff=0)
                for key, title in fetchable:
                    menu.add_command(
                        label=title, command=partial(self._on_choose_fetch_voice, key)
                    )
                self._setup_voice_menu.configure(menu=menu)
                if not self._setup_voice_var.get() and fetchable:
                    self._setup_voice_var.set(fetchable[0][0])
                self._setup_voice_menu.configure(
                    text=f"▾  {self._fetch_voice_title(self._setup_voice_var.get())}"
                )
                self._setup_voice_menu.pack(fill="x", pady=(0, 4))
            self._setup_download_button.pack(fill="x")

    def _fetch_voice_title(self, key: str) -> str:
        for candidate, title in self.service.fetchable_voices():
            if candidate == key:
                return title
        return key or "the default voice"

    def _on_choose_fetch_voice(self, key: str) -> None:
        self._setup_voice_var.set(key)
        self._setup_voice_menu.configure(text=f"▾  {self._fetch_voice_title(key)}")

    def _on_install_engine(self) -> None:
        if self._busy:
            return
        self._start_work(self.service.install_runtime, self._render_setup)

    def _on_download_voices(self) -> None:
        if self._busy:
            return
        key = self._setup_voice_var.get()
        self._start_work(partial(self.service.download_voices, key), self._render_setup)

    def _render_setup(self, report: Any, error: Exception | None) -> None:
        if error is not None:
            self._add_message("note", f"Setup failed: {error}")
            return
        self._add_message("note", report.message)
        # A successful install/download changes what the studio can offer, so the
        # voice list and the status are rebuilt from the engine's real state.
        self._rebuild_voice_section()
        self._refresh_voice_status()
        self._refresh_setup_status()

    def _rebuild_voice_section(self) -> None:
        """Redraw the voice radios after an install, without reopening the window."""
        try:
            self._voice_holder.destroy()
        except (AttributeError, tk.TclError):
            return
        holder = tk.Frame(self._voice_host, bg=self.theme.surface)
        holder.pack(fill="x", padx=10, before=self._voice_note)
        self._voice_holder = holder
        for group in self.service.voice_groups():
            self._voice_group(holder, group.label, indent=4)
            for voice in group.voices:
                suffix = f"  ({voice.gender})" if getattr(voice, "gender", "") else ""
                label = voice.label() if hasattr(voice, "label") else voice.label_en
                self._voice_radio(holder, f"{label}{suffix}", voice.key, indent=10)
        for _key, label_en, voices in self.service.catalog_voices():
            self._voice_group(holder, label_en, indent=4)
            for voice in voices:
                self._voice_radio(holder, voice.label_en, voice.key, indent=10)
        self._voice_var.set(self.service.voice_key())

    def _refresh_document_status(self) -> None:
        document = self.service.document
        if document is None:
            self._document_note.configure(text="No document is open yet.")
        else:
            self._document_note.configure(text=document.summary())

    def _on_open_document(self) -> None:
        pattern = " ".join(f"*{suffix}" for suffix in DOCUMENT_SUFFIXES)
        path = filedialog.askopenfilename(
            parent=self.window,
            title=f"Choose a document for {self.service.name} to read",
            filetypes=[
                ("Documents", pattern),
                ("Text", "*.txt *.md"),
                ("Word", "*.docx *.docm"),
                ("PDF", "*.pdf"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._start_work(
            partial(self.service.load_document, path),
            self._render_document,
        )

    def _render_document(self, document: Any, error: Exception | None) -> None:
        if error is not None:
            self._add_message("note", f"Could not open the document: {error}")
            return
        self._refresh_document_status()
        self._add_message("note", f"Opened {document.summary()}")
        text = self.service.document_text()
        if text:
            self._add_message("document", text)

    def _on_read_aloud(self) -> None:
        """Read the document aloud, or stop a reading already in progress.

        This is the toggle the user asked for: the button reads while idle and
        stops while a reading runs, alternating with every click. Stopping is
        immediate (the clip playing is ended and the worker's loop breaks at the
        next chunk), so a second click never has to wait for the document to end.
        """
        if self.service.reading():
            self.service.stop_reading()
            self._set_reading_button(False)
            return
        if self.service.document is None:
            self._add_message("note", "Open a document first, then read it aloud.")
            return
        if self._busy:
            return
        self._set_reading_button(True)
        self._start_work(self.service.start_reading, self._render_reading)

    def _set_reading_button(self, reading: bool) -> None:
        """Paint the read button as ``▶ Read`` (idle) or ``⏹ Stop`` (reading)."""
        with _suppress_tcl():
            self._read_button.configure(
                text="⏹   Stop reading" if reading else "▶   Read the document aloud"
            )

    def _render_reading(self, report: Any, error: Exception | None) -> None:
        self._set_reading_button(False)
        if self._playing_footer is not None:
            self._playing_footer.set_playing(False)
            self._playing_footer = None
        if error is not None:
            self._add_message("note", f"Could not read the document: {error}")
            return
        self._add_message("note", report.summary())

    def _on_save_speech(self) -> None:
        if self.service.document is None:
            self._add_message("note", "Open a document first, then save its speech.")
            return
        path = filedialog.asksaveasfilename(
            parent=self.window,
            title="Save the speech as a WAV file",
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
        )
        if not path:
            return
        self._start_work(
            partial(self.service.save_speech, path),
            self._render_saved,
        )

    def _render_saved(self, path: Any, error: Exception | None) -> None:
        if error is not None:
            self._add_message("note", f"Could not save the speech: {error}")
            return
        self._add_message("note", f"Saved the speech to {path}")

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
        self._add_message("note", self.service.reading_status())

    def _build_composer(self, parent: tk.Frame) -> None:
        """The shared rounded composer card, with the microphone."""
        composer = ChatComposer(
            parent, theme=self.theme, typography=self._type,
            placeholder=f"Type text for {self.service.name} to read aloud…",
            on_submit=self._on_send, outer_bg=self.theme.base, with_attach=False,
            variable=self._question_var,
        )
        composer.wrap.pack(side="bottom", fill="x", padx=self._column_pad(), pady=(8, 12))
        self._composer = composer
        self._entry = composer.entry
        # The microphone: dictation through the project's own recorder, exactly as
        # the main chat window offers it.
        self._mic_button = composer.add_mic(self._toggle_mic, tip="Dictate with the microphone")
        if not self.service.microphone_available():
            self._mic_button.configure(state="disabled")
        composer.focus()

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
        normal, _mono = self._bubble_fonts()
        widest = 0
        for line in text.splitlines() or [""]:
            widest = max(widest, normal.measure(line))
        return max(120, min(widest, self._available_column()))

    def _bubble_text_width(self, text: str, *, role: str) -> int:
        normal, _mono = self._bubble_fonts()
        char_px = max(1, normal.measure("0"))
        return max(1, self._bubble_content_width(text, role=role) // char_px)

    # -- transcript -------------------------------------------------------

    def _add_message(
        self, role: str, text: str, *, title: str = "", mono_body: bool = False
    ) -> tk.Text:
        theme = self.theme
        pad = self._column_pad()
        row = tk.Frame(self._transcript, bg=theme.base)
        row.pack(fill="x", padx=(pad, pad), pady=6)
        card: RoundedBox | None = None

        head = tk.Frame(row, bg=theme.base)
        head.pack(fill="x", pady=(0, 4))
        if role in ("assistant", "document"):
            self._avatar(head)
            padx = (6, 0)
        else:
            padx = (0, 0)
        labels = {"note": "Note", "document": self.service.name, "user": "You"}
        tk.Label(
            head, text=title or labels.get(role, self.service.name),
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
        if mono_body:
            widget.insert("1.0", text)
        else:
            rich_text.insert_markdown(widget, text, theme=self.theme, typography=self._type)
        widget.pack(anchor="w")
        widget.configure(width=self._bubble_text_width(text, role=role))

        self._rows.append((role, text, row, widget, card))
        self._make_read_only(widget)
        for wheel in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(wheel, self._on_bubble_wheel, add="+")
        self._fit_height(widget)
        # Every message carries the same footer the main window draws -- Copy and a
        # Read aloud / Stop toggle -- so a studio reads exactly like the chat window.
        footer = MessageFooter(
            row, text=strip_markdown(text), widget=widget, theme=theme,
            typography=self._type, on_toggle=self._toggle_footer_reading,
        )
        self._footers.append(footer)
        self._scroll_to_bottom()
        return widget

    def _toggle_footer_reading(self, footer: MessageFooter) -> None:
        """Read one message aloud, or stop the reading that footer started.

        One reading runs at a time: starting a new one stops the previous, and the
        footer of the message being read is the only one painted ``⏹ Stop``.
        """
        if self._playing_footer is not None and self._playing_footer is not footer:
            self.service.stop_reading()
            self._playing_footer.set_playing(False)
            self._playing_footer = None
        # A document read in progress is one reading too; end it before starting
        # another, so only ever one thing speaks.
        if self.service.reading():
            self.service.stop_reading()
            self._set_reading_button(False)
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
            self._render_reading,
        )

    def _avatar(self, parent: tk.Misc) -> None:
        """A small round badge: the saved logo when there is one, else an initial."""
        size = 24
        badge = tk.Canvas(parent, width=size, height=size, bg=self.theme.base, highlightthickness=0)
        path = self.service.logo_path()
        photo = load_scaled(path, size) if path is not None else None
        if photo is not None:
            self._avatar_photos[badge] = photo
            badge.create_image(size // 2, size // 2, image=photo)
        else:
            badge.create_oval(1, 1, size - 1, size - 1, fill=self.theme.primary, outline="")
            badge.create_text(
                size // 2, size // 2 + 1,
                text=(self.service.name.strip()[:1] or "ص").upper(),
                fill=self.theme.primary_foreground,
                font=self._type.font("sm", weight="bold"),
            )
        badge.pack(side="left")

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

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)
        self._reflow()

    def _reflow(self) -> None:
        pad = self._column_pad()
        for _role, _text, row, _widget, _card in self._rows:
            row.pack_configure(padx=(pad, pad))
        self._remetric_rows()

    def _remetric_rows(self) -> None:
        for role, text, _row, widget, _card in self._rows:
            try:
                widget.configure(width=self._bubble_text_width(text, role=role))
            except tk.TclError:
                continue
            self._fit_height(widget)

    def _on_mousewheel(self, event: tk.Event) -> None:
        delta = -1 * (event.delta // 120) if getattr(event, "delta", 0) else (
            -1 if getattr(event, "num", 0) == 4 else 1
        )
        self._canvas.yview_scroll(delta, "units")

    def _on_sidebar_wheel(self, event: tk.Event) -> None:
        delta = -1 * (event.delta // 120) if getattr(event, "delta", 0) else (
            -1 if getattr(event, "num", 0) == 4 else 1
        )
        self._sidebar_canvas.yview_scroll(delta, "units")

    def _scroll_to_bottom(self) -> None:
        self._canvas.update_idletasks()
        self._canvas.yview_moveto(1.0)

    # -- talking ----------------------------------------------------------

    def _on_send(self) -> None:
        """Read the typed text aloud through this studio's voice and mood."""
        if self._busy:
            return
        message = self._composer.text().strip()
        if not message:
            return
        self._composer.clear()
        self._add_message("user", message)
        self._start_work(partial(self.service.read_aloud, message), self._render_reading)

    # -- microphone -------------------------------------------------------

    def _toggle_mic(self) -> None:
        if self._busy:
            return
        # A clone recording and a dictation recording share the one microphone, so
        # the microphone button finishes whichever is running.
        if self._clone_recording is not None:
            self._stop_clone_recording()
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
        self._start_work(lambda: self.service.transcribe(recording.result()), self._render_transcript)

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
        if self._clone_recording is not None:
            try:
                self.service.stop_recording()
            except Exception:
                pass
            self._clone_recording = None
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


def _suppress_tcl():
    """A context manager that swallows a Tk error (a destroyed widget's call)."""
    import contextlib

    return contextlib.suppress(tk.TclError)


__all__ = [
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "LOGO_SIZE",
    "MIN_HEIGHT",
    "MIN_WIDTH",
    "SIDEBAR_WIDTH",
    "VoiceStudioDialog",
]
