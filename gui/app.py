"""The Tkinter window: a ChatGPT-style desktop front end for Fayd Al-Ruh.

Tkinter ships with CPython, so the GUI adds no dependency. Long work (an answer
or a sidebar command) runs on a worker thread and posts results back through a
queue; Tk is only ever touched from the main thread, as Tk requires.

The layout follows ChatGPT closely: a collapsible left sidebar (logo, New chat,
search, conversation history grouped by date, tools, settings), a slim header
that names the active **Fayd Al-Ruh Model** and the model selector, a centred
transcript with Markdown-rendered messages, and a rounded composer pinned to the
bottom. The window has **one** operating mode -- the project's own model, named
by the header -- and the model selector on the right chooses which answering
character (Jarvis with sources + web, Ultron, Vision, Banner, Natasha, Hakeem, or
Cloud) speaks in the same conversation; see :mod:`gui.modes`.

The window is a thin shell over :class:`gui.service.ChatService`; it holds no
answer logic of its own. This module is imported lazily by the CLI, so importing
it never opens a window and never breaks the command-line path.
"""

from __future__ import annotations

import contextlib
import io
import queue
import threading
import time
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tkinter import filedialog, simpledialog
from tkinter import font as tkfont
from typing import Any, Literal

from ai import chat as chat_module
from companion.companion_agent import CompanionError
from core import optional_deps, pdf_ocr, speech
from core.banner import BannerUnavailable
from core.best_effort import best_effort
from core.cloud_model import CloudUnavailable
from core.paths import data_dir
from core.ultron import UltronUnavailable
from core.version import __version__
from core.vision_model import VisionUnavailable
from core.voices import (
    SOURCE_AUDIOCPP,
    SOURCE_PIPER,
    SOURCE_QWEN,
    SOURCE_SYSTEM,
)
from hakeem.hakeem_agent import HakeemError

from gui import chat_composer, rich_text
from gui import markdown as md
from gui.actions import GuiAction, grouped_visible_actions
from gui.attachments import Attachment, AttachmentNotice
from gui.chat_composer import (
    ChatComposer,
    CircleButton,
    RoundedBox,
)
from gui.chat_composer import (
    Tooltip as _Tooltip,
)
from gui.modes import (
    FAYD_MODEL_MODE,
    HistoryItem,
    ModeSpec,
    group_by_date,
    search_items,
)
from gui.options import build_parser
from gui.service import ChatReply, ChatService
from gui.sidebar import SidebarSection
from gui.typography import Typography, for_root
from gui.window_controls import install_window_controls

#: The answering mouths the header's model selector offers, in order. The label
#: is what the user reads; the key is the value :class:`ChatService` expects.
#: Ultron, Vision, and Banner are mouths *inside* this chat, exactly like the other
#: ones, so a conversation with any of them happens in the project's own window,
#: not a separate one. The project's own trained model -- the **Fayd Al-Ruh
#: Model** -- is *not* listed here: it is the window's default mouth (named by the
#: header's mode control), so the selector is the switch to the other characters.
MOUTH_CHOICES: tuple[tuple[str, str], ...] = (
    (chat_module.MOUTH_SOURCES, "Jarvis (Internal Sources + Web)"),
    (chat_module.MOUTH_ULTRON, "Ultron (general local model)"),
    (chat_module.MOUTH_VISION, "Vision (petroleum local model)"),
    (chat_module.MOUTH_BANNER, "Banner (presentation local model)"),
    (chat_module.MOUTH_CLOUD, "Cloud (remote model: ChatGPT, Claude, Gemini)"),
    (chat_module.MOUTH_NATASHA, "Natasha (companion: memory, tone)"),
    (chat_module.MOUTH_HAKEEM, "Hakeem (scholar of religion and history)"),
)
MOUTH_LABELS: dict[str, str] = {
    **dict(MOUTH_CHOICES),
    # The project's own trained model is the window's default mouth, named by the
    # header's mode control rather than listed in the selector, but it still needs
    # a readable label for the header and the status line.
    chat_module.MOUTH_MODEL: FAYD_MODEL_MODE,
}

#: The seven model switches the composer's bottom row offers, left to right, in
#: the order they were asked for: the mouth key, the model's own name, and what it
#: is for. The row is the quick switch beside the input, and the header's model
#: selector on the right offers the same seven characters -- both drive the one
#: :meth:`FaydGUI._switch_mouth`, so the two can never disagree about which model
#: answers. The project's own trained model is the window's default (returned to
#: by the header's mode control), so it is not listed in either switch.
MODEL_BUTTONS: tuple[tuple[str, str, str], ...] = (
    (chat_module.MOUTH_SOURCES, "Jarvis", "Sources"),
    (chat_module.MOUTH_ULTRON, "Ultron", "General"),
    (chat_module.MOUTH_VISION, "Vision", "Petroleum"),
    (chat_module.MOUTH_BANNER, "Banner", "Instructor"),
    (chat_module.MOUTH_NATASHA, "Natasha", "Personality"),
    (chat_module.MOUTH_HAKEEM, "Hakeem", "Religion"),
    (chat_module.MOUTH_CLOUD, "Cloud", "Network"),
)

#: The four **speech-engine** studios, as the composer row's trailing group. They
#: sit at the far right of the same row as the model buttons and are ordered right
#: to left in the order asked for: ``(source key, button name, what it is for)``.
#: Pressing one opens that engine's own studio window, where a document is read
#: aloud with a voice and a mood that belong to that engine alone. The names are
#: the engine titles the user asked for.
VOICE_ENGINE_BUTTONS: tuple[tuple[str, str, str], ...] = (
    (SOURCE_SYSTEM, "Installed Voices", "Arabic + English"),
    (SOURCE_AUDIOCPP, "VoiceTut-TTS", "Local Arabic"),
    (SOURCE_PIPER, "Piper", "Arabic + English"),
    (SOURCE_QWEN, "Qwen3-TTS", "English"),
)

# ------------------------------------------------------- OpenHands design tokens
#
# Colours, radii and spacing mirror the OpenHands chat UI (the OpenHands
# frontend's ``src/tailwind.css``): the cool-grey scale, a gold primary accent,
# and HeroUI's small/large radii. Keeping the same tokens is what makes the
# window read like OpenHands rather than a generic form.

#: The small↔large radius pair OpenHands uses (HeroUI ``radius.small``/``large``).
#: They live in :mod:`gui.chat_composer` (the shared composer), so the window,
#: the dialogs, and the panel cannot drift apart; they are re-exported here.
RADIUS_SMALL = chat_composer.RADIUS_SMALL
RADIUS_LARGE = chat_composer.RADIUS_LARGE

#: The send control is a disc, so its side length is a single diameter in pixels.
SEND_BUTTON_SIZE = chat_composer.SEND_BUTTON_SIZE

#: Below this width the sidebar starts hidden (a drawer), like ChatGPT on mobile.
_COMPACT_WIDTH = 900
#: The sidebar's expanded width and its collapsed rail width.
_SIDEBAR_WIDTH = 288
_RAIL_WIDTH = 0


@dataclass(frozen=True)
class Theme:
    """One appearance: every colour the window draws with.

    Field names follow OpenHands' semantic tokens (``surface``, ``tertiary``,
    ``border_subtle`` …) so the mapping back to the web UI stays obvious.
    """

    name: str
    # Surfaces, from deepest to most raised.
    surface_deep: str      # context menus, deep insets
    base: str              # app shell background
    surface: str           # cards, composer, sidebar
    surface_raised: str    # hover/selected tint inside a panel
    # Text, from most to least prominent.
    contrast: str          # strongest contrast (assistant text, titles)
    foreground: str        # primary readable text
    text_secondary: str    # secondary text
    text_dim: str          # placeholders, very muted labels
    # Lines and state.
    border: str            # standard 1px stroke
    border_subtle: str     # within-panel separators, list dividers
    primary: str           # gold accent — send button, brand, focus
    primary_foreground: str  # text/icon on the primary fill
    interactive_hover: str   # hover on the base surface
    interactive_active: str  # selected/active on the base surface
    user_bubble: str       # user message fill
    user_bubble_fg: str    # user message text
    code_bg: str           # fenced/inline code background
    code_fg: str           # fenced code text
    quote_bar: str         # blockquote accent
    titlebar: str          # every window's title bar (a dark grey band)
    titlebar_edge: str     # the hairline under a window's title bar
    success: str
    danger: str
    disabled: str


DARK_THEME = Theme(
    name="dark",
    surface_deep="#05070A",     # cool-grey-975
    base="#0B0E14",             # cool-grey-950
    surface="#21252F",          # cool-grey-925
    surface_raised="#2C313F",   # cool-grey-900
    contrast="#F7F9FC",         # cool-grey-50
    foreground="#C3CDDC",       # cool-grey-300
    text_secondary="#A3B0C4",   # cool-grey-400
    text_dim="#7E8A9E",         # cool-grey-500
    border="#4B5468",           # cool-grey-700
    border_subtle="#383F50",    # cool-grey-800
    primary="#C9B974",          # gold
    primary_foreground="#0B0E14",
    interactive_hover="#2C313F",  # cool-grey-900
    interactive_active="#383F50",  # cool-grey-800
    user_bubble="#2C313F",        # cool-grey-900, like OpenHands' bg-tertiary
    user_bubble_fg="#C3CDDC",
    code_bg="#151A24",            # a touch deeper than the base
    code_fg="#DCE3EE",
    quote_bar="#4B5468",
    titlebar="#181C25",         # a dark grey band, a step above the base
    titlebar_edge="#2C313F",    # cool-grey-900, the hairline under it
    success="#A5E75E",
    danger="#E76A5E",
    disabled="#383F50",
)

LIGHT_THEME = Theme(
    name="light",
    surface_deep="#EEF2F7",     # cool-grey-100
    base="#F7F9FC",             # cool-grey-50
    surface="#FFFFFF",
    surface_raised="#EEF2F7",   # cool-grey-100
    contrast="#0B0E14",         # cool-grey-950
    foreground="#2C313F",       # cool-grey-900
    text_secondary="#4B5468",   # cool-grey-700
    text_dim="#7E8A9E",         # cool-grey-500
    border="#DCE3EE",           # cool-grey-200
    border_subtle="#EEF2F7",    # cool-grey-100
    primary="#8A6E1F",          # darkened gold, AA on the light base
    primary_foreground="#FFFFFF",
    interactive_hover="#EEF2F7",  # cool-grey-100
    interactive_active="#DCE3EE",  # cool-grey-200
    user_bubble="#EEF2F7",        # cool-grey-100
    user_bubble_fg="#0B0E14",
    code_bg="#F0F3F8",
    code_fg="#1B2029",
    quote_bar="#C4CEDD",
    titlebar="#DCE3EE",         # a light grey band in the light appearance
    titlebar_edge="#C4CEDD",    # the hairline under it
    success="#3F7D1E",
    danger="#C0473C",
    disabled="#DCE3EE",
)

THEMES = {theme.name: theme for theme in (DARK_THEME, LIGHT_THEME)}

#: The window's saved appearance lives in the user's data directory.
_THEME_NAMESPACE = "gui"
_THEME_KEY = "appearance"

# A paragraph-cap for a single message; longer notes scroll inside their card.
_MAX_BUBBLE_LINES = 40

#: The chat column's maximum width, so long lines stay readable and the
#: transcript reads like a centred conversation (OpenHands caps it the same way).
_COLUMN_MAX = 820


def _pick_font() -> str:
    """The proportional family the window draws with (OpenHands' UI stack).

    Kept as a module-level function because the dialogs and the tests call it;
    the choice itself now lives in :mod:`gui.typography`.
    """
    return for_root(None).ui_family


def _pick_mono() -> str:
    """The monospace family the window draws code with (OpenHands' code stack)."""
    return for_root(None).mono_family


def load_theme_name() -> str:
    """The saved theme name, defaulting to dark; never raises on a bad store."""
    from storage.json_store import JsonStore

    try:
        saved = JsonStore(data_dir()).load(_THEME_NAMESPACE, _THEME_KEY)
    except Exception:  # a broken store must not stop the window opening
        return DARK_THEME.name
    name = saved.get("theme") if isinstance(saved, dict) else None
    return name if name in THEMES else DARK_THEME.name


def save_theme_name(name: str) -> None:
    """Persist the chosen theme; a store error is recorded, not fatal."""
    from storage.json_store import JsonStore

    if name not in THEMES:
        return
    with best_effort("save the theme"):
        JsonStore(data_dir()).save(_THEME_NAMESPACE, _THEME_KEY, {"theme": name})


_VOICE_KEY = "reading_voice"
_SPEAK_KEY = "auto_speak"


def load_reading_voice() -> str:
    """The saved reading-voice key, defaulting to the catalog default.

    A saved ``system:<engine>:<name>`` or ``model:<engine>:<name>`` key is accepted
    as-is (it names a real installed voice or a voice of a local model); the voice is
    resolved against what the OS or the model reports when it is used, so a voice
    that was uninstalled falls back honestly rather than erroring.
    """
    from core.voices import (
        DEFAULT_VOICE_KEY,
        VOICE_CATALOG,
        is_model_voice_key,
        is_system_voice_key,
    )
    from storage.json_store import JsonStore

    valid = {voice.key for voice in VOICE_CATALOG}
    try:
        saved = JsonStore(data_dir()).load(_THEME_NAMESPACE, _VOICE_KEY)
    except Exception:
        return DEFAULT_VOICE_KEY
    key = saved.get("voice") if isinstance(saved, dict) else None
    if not isinstance(key, str):
        return DEFAULT_VOICE_KEY
    if key in valid or is_system_voice_key(key) or is_model_voice_key(key):
        return key
    return DEFAULT_VOICE_KEY


def save_reading_voice(key: str) -> None:
    """Persist the chosen reading voice; a store error is recorded, not fatal."""
    from storage.json_store import JsonStore

    with best_effort("save the reading voice"):
        JsonStore(data_dir()).save(_THEME_NAMESPACE, _VOICE_KEY, {"voice": key})


def load_auto_speak() -> bool:
    """Whether new assistant answers should be read aloud automatically."""
    from storage.json_store import JsonStore

    try:
        saved = JsonStore(data_dir()).load(_THEME_NAMESPACE, _SPEAK_KEY)
    except Exception:
        return False
    enabled = saved.get("enabled") if isinstance(saved, dict) else None
    return enabled if isinstance(enabled, bool) else False


def save_auto_speak(enabled: bool) -> None:
    """Persist the auto-speak choice; a store error is recorded, not fatal."""
    from storage.json_store import JsonStore

    with best_effort("save the auto-speak choice"):
        JsonStore(data_dir()).save(_THEME_NAMESPACE, _SPEAK_KEY, {"enabled": bool(enabled)})


_SECTIONS_KEY = "sidebar_sections"


def load_sidebar_sections(root: Path | None = None) -> dict[str, bool]:
    """The saved open/closed state of the sidebar's collapsible sections.

    A section the user closed stays closed on the next launch, so the sidebar
    remembers the shape they arranged it into. ``root`` defaults to the app data
    directory; a caller that owns its own data root (the window, through the
    service) passes it so the choice is isolated with the rest of that store. A
    missing or broken record reads as empty, and a value that is not a real bool
    is dropped, so a corrupt store can never leave a section undrawable.
    """
    from storage.json_store import JsonStore

    try:
        saved = JsonStore(root if root is not None else data_dir()).load(
            _THEME_NAMESPACE, _SECTIONS_KEY
        )
    except Exception:
        return {}
    raw = saved.get("sections") if isinstance(saved, dict) else None
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items() if isinstance(value, bool)}


def save_sidebar_sections(states: dict[str, bool], root: Path | None = None) -> None:
    """Persist the sidebar sections' open/closed state; a store error is recorded."""
    from storage.json_store import JsonStore

    payload = {str(key): bool(value) for key, value in states.items()}
    with best_effort("save the sidebar sections"):
        JsonStore(root if root is not None else data_dir()).save(
            _THEME_NAMESPACE, _SECTIONS_KEY, {"sections": payload}
        )


def _strip_markdown(text: str) -> str:
    """Turn rendered Markdown back into plain prose, so a reader speaks words.

    A speech engine should not read ``**``, backticks, or a list bullet aloud, so
    the markers are removed while the words and the code are kept. The rule lives
    in :func:`gui.message_footer.strip_markdown` so every window that reads aloud
    strips the same way; this name is kept for the main window's own callers.
    """
    from gui.message_footer import strip_markdown

    return strip_markdown(text)


class LineStream(io.TextIOBase):
    """A text stream that forwards each complete line to a queue.

    Training entry points print their live progress to stderr; the GUI redirects
    stderr through this stream so each line reaches the transcript as it is
    written, instead of arriving only after the run ends.
    """

    def __init__(self, lines: queue.Queue[str]) -> None:
        self._lines = lines
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._lines.put(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self._lines.put(self._buffer)
        self._buffer = ""


@dataclass
class _Row:
    """One transcript row and the widgets that make it up."""

    role: str
    widget: tk.Text
    row: tk.Frame
    text: str
    #: The full rendered text (``text`` plus any ``extra``), used to size the bubble.
    body: str = ""


@dataclass(frozen=True)
class _SpokenClip:
    """A rendered clip awaiting playback, tagged with the click that asked for it.

    The token lets the window drop a render that a newer click (or a stop) has
    superseded, so a second click never starts the previous message talking.
    """

    text: str
    clip: Any
    token: int


@dataclass(frozen=True)
class _Transcript:
    """The text the microphone produced, ready to place in the composer."""

    text: str


class FaydGUI:
    """The application window and its wiring to :class:`ChatService`."""

    #: The general-purpose composer hint (the reference wording), kept as a class
    #: constant so a caller or a test can name it without a live window.
    COMPOSER_PLACEHOLDER = "What do you want to build?"

    #: An optional override for the service a dedicated model window drives
    #: (:meth:`_model_chat_service`). A test sets this to build a lighter service
    #: and open the window without a full session; ``None`` builds the real one.
    model_chat_service_factory: Callable[[str], Any] | None = None

    def __init__(
        self,
        root: tk.Tk,
        *,
        service: ChatService | None = None,
        model_checkpoint: str | None = None,
        no_web: bool = False,
        web: bool | None = None,
        session: str | None = None,
        theme: str | None = None,
        operating_mode: str | None = None,
    ) -> None:
        self.root = root
        self.service = service or ChatService(
            no_web=no_web,
            web=web,
            model_checkpoint=model_checkpoint,
            session=session,
            operating_mode=operating_mode,
        )
        self._results: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._lines: queue.Queue[str] = queue.Queue()
        self._busy = False
        self._streaming_label: tk.Text | None = None
        # Speech jobs (render / transcribe) run off the busy path, so the composer
        # stays usable while a clip renders or a recording is transcribed.
        self._speech_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        # Live output is coalesced and rendered at a bounded rate: a fast training
        # loop can emit thousands of lines, and reconfiguring on every one would
        # freeze the window.
        self._pending: list[str] = []
        self._last_render = 0.0
        self._rows: list[_Row] = []
        # One type scale for the whole window (OpenHands' sizes, DPI-corrected).
        self._type: Typography = for_root(root)
        self._font = self._type.ui_family
        self._mono = self._type.mono_family
        self._thinking: tk.Label | None = None
        self._thinking_row: tk.Widget | None = None
        self._thinking_phase = 0
        # Sidebar visibility. Wide windows show it; a compact window hides it as
        # a drawer that the header's menu button toggles.
        self._sidebar_visible = True
        self._auto_hidden = False
        self._settings_open = False
        self._voices_open = False
        # The sidebar's collapsible sections: which are open, and the widget for
        # each, so a toggle can persist the choice and a rebuild can restore it.
        # The state is stored under the service's own data root, so the window and
        # the rest of that store agree, and a test service is isolated.
        self._section_states: dict[str, bool] = load_sidebar_sections(
            getattr(self.service, "data_root", None)
        )
        self._sections: dict[str, SidebarSection] = {}
        # The Settings section's contents are built once, on first open, so the
        # Cloud pickers are not rebuilt on every open (they keep their state).
        self._settings_built = False
        # The settings-panel radio only exists once the panel is expanded; the
        # mouth can still be switched from the header before that, so the
        # variable is created here and the radio simply attaches to it.
        self._source_var = tk.StringVar(value=self.service.source)
        self._action_buttons: list[tk.Widget] = []
        self._retry_text = ""
        self._pending_message: str | None = None
        self._confirm_bar: tk.Frame | None = None
        self._composer_lift = 0
        self._history_query = tk.StringVar()

        # The theme is read from the user's saved preference unless one was passed
        # in; both toggle later through _toggle_theme.
        name = theme if theme in THEMES else load_theme_name()
        self._theme = THEMES[name]

        # Reading aloud: the saved voice is authoritative, so the window and the
        # service agree from the first message. Auto-speak is opt-in.
        self._voice_key = load_reading_voice()
        try:
            self.service.set_voice(self._voice_key)
        except ValueError:
            self._voice_key = self.service.voice_key()
        self._auto_speak = tk.BooleanVar(value=load_auto_speak())
        self._speak_var = tk.StringVar(value=self._voice_key)
        self._speaking = False
        # The playback in progress (a stoppable clip) and the message it belongs
        # to, so a second click on that message's speaker stops the reading and a
        # new click elsewhere stops the old reading before starting the new one.
        self._playback: Any = None
        # The per-message speaker control currently showing "Stop", so its label
        # can be restored when the reading ends or is superseded.
        self._speak_control: tk.Label | None = None
        # A counter that tags each "read aloud" request; a render whose token is
        # stale (superseded by a newer click or a stop) is discarded on arrival.
        self._speak_token = 0
        # Microphone state: the recording in progress, its button, and the
        # transcript note shown while recording.
        self._recording: Any = None
        self._mic_button: tk.Button | None = None
        self._listening = False

        # The transcript is selectable and copyable; Delete clears it, Ctrl+C
        # copies the selection, Ctrl+A selects the whole transcript, and a
        # right-click offers the same as a menu.
        root.bind("<Control-c>", self._on_copy)
        root.bind("<Control-C>", self._on_copy)
        root.bind("<Control-a>", self._on_select_all)
        root.bind("<Control-A>", self._on_select_all)
        root.bind("<Delete>", self._on_clear_transcript)
        root.bind("<Button-3>", self._on_right_click)
        root.bind("<Control-n>", self._on_new_chat)
        root.bind("<Control-s>", self._on_save)
        self._clipboard_menu = tk.Menu(root, tearoff=0)
        self._clipboard_menu.add_command(label="Copy", command=self._on_copy)
        self._clipboard_menu.add_command(
            label="Copy conversation", command=self._on_copy_conversation
        )
        self._clipboard_menu.add_command(label="Select all", command=self._on_select_all)
        self._clipboard_menu.add_separator()
        self._clipboard_menu.add_command(label="Clear transcript", command=self._clear_transcript)

        root.title("Fayd Al-Ruh")
        root.geometry("1180x780")
        root.minsize(420, 520)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        root.bind("<Configure>", self._on_root_configure)

    def _on_close(self) -> None:
        """Stop any reading or recording before the window is destroyed."""
        self._stop_speaking()
        self._cancel_recording()
        self.root.destroy()

    # ------------------------------------------------------------------ layout

    @property
    def _bubbles(self) -> list[tk.Text]:
        """The message text widgets, in order (kept for tests and callers)."""
        return [entry.widget for entry in self._rows]

    @property
    def _session_buttons(self) -> list[tk.Widget]:
        """The sidebar history widgets, in order (kept for tests and callers)."""
        return self._history_buttons

    def _build_ui(self) -> None:
        """Build (or rebuild) the whole window for the current theme.

        Called again by :meth:`_toggle_theme`; the transcript is snapshotted
        first and restored, so switching appearance keeps every message.
        """
        snapshot = self._snapshot_transcript()
        self._rows = []
        self._pending = []
        self._streaming_label = None
        self._thinking_row = None
        self._thinking = None
        self._confirm_bar = None
        # A rebuild (a theme switch) re-creates the sidebar from scratch, so the
        # sections and the built-once Settings contents start fresh.
        self._sections = {}
        self._settings_built = False
        for child in self.root.winfo_children():
            if isinstance(child, tk.Menu):  # the clipboard menu outlives a rebuild
                continue
            child.destroy()
        self.root.configure(bg=self._theme.base)

        shell = tk.Frame(self.root, bg=self._theme.base)
        shell.pack(fill="both", expand=True)
        self._build_sidebar(shell)
        self._build_main(shell)
        self._apply_mode_label()
        for role, text in snapshot:
            self._add_bubble(role, text)
        if not snapshot:
            self._show_hero()
        self._set_send_enabled(not self._busy)
        if not getattr(self, "_pumping", False):
            self._pumping = True
            self._pump()

    def _build_sidebar(self, shell: tk.Frame) -> None:
        """The left sidebar: brand, New chat, search, and grouped sections.

        The column is a short list of **collapsible sections** rather than one
        flat scroll of buttons, the way a graphics or design program arranges its
        tools: the chat history, the reading voice, the answering-assistant
        actions (grouped by theme), the speech models, the training tools, the
        system tools, and the settings -- each with its own header, each closed or
        opened by a click, and each remembering that choice. Nothing is listed
        twice: the reading voice lives in exactly one section, and the settings
        section holds only what no other section already offers.
        """
        sidebar = tk.Frame(shell, bg=self._theme.surface, width=_SIDEBAR_WIDTH)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        self._sidebar = sidebar
        tk.Frame(sidebar, bg=self._theme.border_subtle, width=1).pack(side="right", fill="y")

        self._brand(sidebar)

        new_chat = tk.Button(
            sidebar, text="＋   New chat", command=self._on_new_chat,
            bg=self._theme.surface_raised, fg=self._theme.contrast,
            activebackground=self._theme.interactive_active,
            activeforeground=self._theme.contrast,
            font=self._type.font("sm", weight="bold"), relief="flat", borderwidth=0,
            anchor="w", padx=14, pady=8, cursor="hand2",
        )
        new_chat.bind("<Enter>", lambda _e: new_chat.configure(bg=self._theme.interactive_active))
        new_chat.bind("<Leave>", lambda _e: new_chat.configure(bg=self._theme.surface_raised))
        new_chat.pack(fill="x", padx=12, pady=(2, 6))

        self._search_entry = tk.Entry(
            sidebar, textvariable=self._history_query, bg=self._theme.surface_raised,
            fg=self._theme.foreground, insertbackground=self._theme.primary,
            relief="flat", font=self._type.font("sm"), highlightthickness=0,
        )
        self._search_entry.pack(fill="x", padx=12, pady=(0, 8), ipady=6)
        self._search_entry.insert(0, "")
        self._history_query.trace_add("write", lambda *_: self.refresh_history())
        self._search_placeholder()

        # Everything below the header scrolls behind a scrollbar that appears
        # only when it actually overflows.
        body = tk.Frame(sidebar, bg=self._theme.surface)
        body.pack(side="top", fill="both", expand=True)
        self._side_canvas = tk.Canvas(body, bg=self._theme.surface, highlightthickness=0, width=1)
        self._side_canvas.pack(side="left", fill="both", expand=True)
        self._side_scrollbar = tk.Scrollbar(body, command=self._side_canvas.yview)
        self._side_canvas.configure(yscrollcommand=self._side_scrollbar.set)
        panel = tk.Frame(self._side_canvas, bg=self._theme.surface)
        window = self._side_canvas.create_window((0, 0), window=panel, anchor="nw")
        panel.bind("<Configure>", self._on_side_panel_configure)
        self._side_canvas.bind("<Configure>", lambda e: self._side_canvas.itemconfigure(window, width=e.width))
        for widget in (self._side_canvas, panel):
            widget.bind("<MouseWheel>", self._on_sidebar_wheel)
            widget.bind("<Button-4>", self._on_sidebar_wheel)
            widget.bind("<Button-5>", self._on_sidebar_wheel)
        self._side_panel = panel

        # 1. The chat history, open by default -- it is what a user returns to.
        chats = self._add_section(panel, key="chats", title="Chats", icon="🗂", default_open=True)
        self._history_panel = tk.Frame(chats.body, bg=self._theme.surface)
        self._history_panel.pack(fill="x")
        self._history_buttons: list[tk.Widget] = []
        self.refresh_history()

        # 2. The reading voice, the *one* place a voice is chosen (the settings
        # section no longer repeats it). Its list is built when the section opens,
        # so it always reflects what is installed right now.
        voices = self._add_section(
            panel, key="voices", title="Reading Voice", icon="🗣", default_open=False
        )
        self._voices_panel = tk.Frame(voices.body, bg=self._theme.surface)
        self._voices_panel.pack(fill="x")
        self._voices_open = voices.is_open

        # 3. The answering-assistant actions, grouped by theme. Each group is its
        # own section, so the column is a handful of readable headers instead of
        # one long list of buttons. The buttons are built now (so a search or a
        # test sees them all) but stay out of sight until their section is opened.
        self._action_buttons.clear()
        for group, members in grouped_visible_actions():
            section = self._add_section(
                panel,
                key=f"group:{group.key}",
                title=group.title,
                icon=group.icon,
                default_open=False,
            )
            for action in members:
                button = self._sidebar_button(
                    section.body, action.sidebar_label, partial(self._on_action, action), small=True
                )
                button.pack(fill="x", padx=12, pady=1)
                # The sidebar shows the short label; the full one is the tooltip,
                # so the complete explanation is still one hover away.
                if action.sidebar_label != action.label:
                    _Tooltip(button, action.label, self._theme, self._type)
                self._action_buttons.append(button)

        # 4. Settings: the cloud provider/model pickers and the appearance toggle.
        # Nothing here repeats another section (the voices used to be listed here
        # as well, which is exactly the duplication that is now gone).
        settings = self._add_section(
            panel, key="settings", title="Settings", icon="⚙", default_open=False
        )
        self._settings_panel = tk.Frame(settings.body, bg=self._theme.surface)
        self._settings_panel.pack(fill="x")
        self._settings_open = settings.is_open

        self._status = tk.Label(
            sidebar, text=self._status_text(), bg=self._theme.surface, fg=self._theme.text_dim,
            font=self._type.font("2xs"), anchor="w", justify="left", padx=18, pady=10,
        )
        self._status.pack(side="bottom", fill="x")

        if self._is_compact():
            sidebar.pack_forget()
            self._sidebar_visible = False

    def _add_section(
        self,
        panel: tk.Misc,
        *,
        key: str,
        title: str,
        icon: str,
        default_open: bool,
    ) -> SidebarSection:
        """Build one collapsible sidebar section, restoring its saved state."""
        section = SidebarSection(
            panel,
            key=key,
            title=title,
            icon=icon,
            theme=self._theme,
            typography=self._type,
            open=self._section_states.get(key, default_open),
            on_toggle=self._on_section_toggle,
        )
        self._sections[key] = section
        return section

    def _open_section(self, key: str) -> None:
        """Open a section programmatically (idempotent), if it exists.

        The section headers own the real click-to-collapse behaviour; this is the
        "show me this panel" entry point the window's own handlers use, so opening
        an already-open section is a no-op and a missing one is ignored rather
        than raising. It opens **without persisting**, because showing a panel
        because the program asked for it must not rewrite the layout the user
        arranged -- only the header's own click persists.
        """
        section = self._sections.get(key)
        if section is not None:
            section.set_open(True, persist=False)

    def _on_section_toggle(self, key: str, is_open: bool) -> None:
        """Remember a section's open/closed choice and keep the scrollbar honest."""
        self._section_states[key] = is_open
        save_sidebar_sections(self._section_states, getattr(self.service, "data_root", None))
        if key == "voices":
            self._voices_open = is_open
            # The voice list is the one menu that must reflect what is installed
            # *right now*, so it is (re)built the moment the section is opened --
            # a plain header click is how a user reaches it, and it must not open
            # to an empty panel.
            if is_open:
                self._rebuild_voice_section()
        elif key == "settings":
            self._settings_open = is_open
            # The settings contents are built on first open (they touch the cloud
            # pickers), so a plain header click must build them too -- otherwise the
            # section would open empty.
            if is_open and not self._settings_built:
                self._build_settings_panel(self._settings_panel)
                self._settings_built = True
        self.root.after_idle(self._toggle_sidebar_scrollbar)

    def _search_placeholder(self) -> None:
        """Show a greyed hint in the search box while it is empty."""
        if self._history_query.get():
            return
        self._search_entry.configure(fg=self._theme.text_dim)
        self._search_entry.insert(0, "Search chats")

        def clear(_event: tk.Event) -> None:
            if self._history_query.get() == "Search chats":
                self._search_entry.delete(0, "end")
                self._search_entry.configure(fg=self._theme.foreground)

        def restore(_event: tk.Event) -> None:
            if not self._history_query.get():
                self._search_entry.configure(fg=self._theme.text_dim)
                self._search_entry.insert(0, "Search chats")

        self._search_entry.bind("<FocusIn>", clear)
        self._search_entry.bind("<FocusOut>", restore)

    def _on_side_panel_configure(self, _event: tk.Event) -> None:
        self._side_canvas.configure(scrollregion=self._side_canvas.bbox("all"))
        self._toggle_sidebar_scrollbar()

    def _toggle_sidebar_scrollbar(self) -> None:
        """Show the sidebar scrollbar only when its content overflows."""
        first, last = self._side_canvas.yview()
        if first <= 0.0 and last >= 1.0:
            self._side_scrollbar.pack_forget()
        else:
            self._side_scrollbar.pack(side="right", fill="y")

    def _on_sidebar_wheel(self, event: tk.Event) -> str:
        delta = -1 * (event.delta // 120) if getattr(event, "delta", 0) else (
            -1 if getattr(event, "num", 0) == 4 else 1
        )
        self._side_canvas.yview_scroll(delta, "units")
        self._toggle_sidebar_scrollbar()
        return "break"

    # ------------------------------------------------------------- appearance

    def _toggle_theme(self) -> str:
        """Switch between dark and light, persist the choice, and rebuild."""
        self._theme = LIGHT_THEME if self._theme.name == "dark" else DARK_THEME
        save_theme_name(self._theme.name)
        self._rebuild_preserving_transcript()
        return "break"

    def _rebuild_preserving_transcript(self) -> None:
        transcript = self._snapshot_transcript()
        self._build_ui()
        # _build_ui restores the transcript itself; nothing more to do.
        if transcript and self._rows:
            self._scroll_to_bottom()

    def _snapshot_transcript(self) -> list[tuple[str, str]]:
        """The visible messages as (role, text), so a rebuild can restore them."""
        return [(entry.role, entry.text) for entry in self._rows]

    def _brand(self, sidebar: tk.Frame) -> None:
        header = tk.Frame(sidebar, bg=self._theme.surface)
        header.pack(fill="x", padx=18, pady=(18, 10))
        dot = tk.Canvas(header, width=30, height=30, bg=self._theme.surface, highlightthickness=0)
        dot.create_oval(1, 1, 29, 29, fill=self._theme.primary, outline="")
        dot.create_text(15, 16, text="ف", fill=self._theme.primary_foreground, font=self._type.font("sm", weight="bold"))
        dot.pack(side="left")
        titles = tk.Frame(header, bg=self._theme.surface)
        titles.pack(side="left", padx=(10, 0))
        tk.Label(
            titles, text="Fayd Al-Ruh", bg=self._theme.surface, fg=self._theme.contrast,
            font=self._type.font("heading3", weight="bold"), anchor="w",
        ).pack(anchor="w")
        tk.Label(
            titles, text=f"v{__version__}  ·  petroleum mind", bg=self._theme.surface,
            fg=self._theme.text_dim, font=self._type.font("2xs"), anchor="w",
        ).pack(anchor="w")

    def _sidebar_button(
        self,
        parent: tk.Misc,
        text: str,
        command: Callable[[], Any],
        *,
        small: bool = False,
        active: bool = False,
    ) -> tk.Label:
        """A flat, hoverable label that behaves like a button."""
        base = self._theme.surface_raised if active else self._theme.surface
        button = tk.Label(
            parent, text=text, bg=base, fg=self._theme.foreground,
            font=self._type.font("xs" if small else "body"), anchor="w", padx=12,
            pady=(4 if small else 9),
            justify="left", wraplength=232 if small else 0, cursor="hand2",
        )
        button.bind("<Enter>", lambda _e: button.configure(bg=self._theme.interactive_hover))
        button.bind("<Leave>", lambda _e: button.configure(bg=base))
        button.bind("<Button-1>", lambda _e: command())
        return button

    # ----------------------------------------------------------------- history

    def refresh_history(self) -> None:
        """Rebuild the Chats list from the store, grouped by date, with search."""
        panel = getattr(self, "_history_panel", None)
        if panel is None:
            return
        for widget in self._history_buttons:
            widget.destroy()
        self._history_buttons = []
        try:
            items = self.service.saved_session_summaries()
        except Exception:  # a broken store must not blank the sidebar
            items = ()
        query = self._history_query.get()
        if query == "Search chats":
            query = ""
        items = search_items(items, query)
        if not items:
            hint = tk.Label(
                self._history_panel,
                text="No matching chats." if query else "No saved chats yet.",
                bg=self._theme.surface, fg=self._theme.text_dim, font=self._type.font("xs"),
                anchor="w", padx=18, pady=2,
            )
            hint.pack(fill="x")
            self._history_buttons.append(hint)
            return
        for bucket, group in group_by_date(items):
            header = tk.Label(
                self._history_panel, text=bucket, bg=self._theme.surface,
                fg=self._theme.text_dim, font=self._type.font("2xs", weight="bold"), anchor="w", padx=18,
            )
            header.pack(fill="x", pady=(6, 1))
            self._history_buttons.append(header)
            for item in group:
                button = self._history_button(self._history_panel, item)
                button.pack(fill="x", padx=10, pady=1)
                self._history_buttons.append(button)

    def _history_button(self, parent: tk.Misc, item: HistoryItem) -> tk.Label:
        label = f"{item.label}   ✓" if item.is_current else item.label
        return self._sidebar_button(
            parent, label, lambda: self._on_resume_session(item), small=True, active=item.is_current
        )

    def _render_tools(self) -> None:
        """Refresh the sidebar's tool list -- which the sidebar no longer shows.

        The sidebar used to carry a **Tools** section listing the sandbox tools.
        It is gone, so there is nothing to paint; the guard keeps the call sites
        (``_set_mode``) working without a panel to draw into. The tools themselves
        are unchanged and still run in Jarvics Mode.
        """
        panel = getattr(self, "_tools_panel", None)
        if panel is None or not panel.winfo_exists():
            return
        for child in panel.winfo_children():
            child.destroy()
        if not self.service.tool_names:
            tk.Label(
                panel, text="No tools available.", bg=self._theme.surface,
                fg=self._theme.text_dim, font=self._type.font("xs"), anchor="w", padx=18, pady=2,
            ).pack(fill="x")
            return
        enabled = self.service.tools_enabled
        colour = self._theme.foreground if enabled else self._theme.text_dim
        for name in self.service.tool_names:
            tk.Label(
                panel, text=f"• {name}", bg=self._theme.surface, fg=colour,
                font=self._type.font("xs"), anchor="w", padx=20, pady=1,
            ).pack(fill="x")
        note = (
            "Available in Jarvics Mode."
            if enabled
            else "Switch to Jarvics Mode to use them."
        )
        tk.Label(
            panel, text=note, bg=self._theme.surface, fg=self._theme.text_dim,
            font=self._type.font("2xs"), anchor="w", padx=20, pady=2, wraplength=232, justify="left",
        ).pack(fill="x")

    def _toggle_settings(self) -> None:
        """Ensure the sidebar's **Settings** section is open (and built).

        The section header owns the real open/close click; this method is the
        programmatic "show the settings" entry point the window and its tests use,
        so it is idempotent (opening an already-open section is a no-op) and builds
        the section's contents on first use. The section holds only what no other
        section offers -- the Cloud provider and model pickers and the appearance
        switch, plus the two session controls. The reading voice and the assistant
        actions are deliberately *not* repeated here: they have their own sections,
        which is what removes the duplication the old flat sidebar carried.
        """
        self._open_section("settings")
        if not self._settings_built:
            self._build_settings_panel(self._settings_panel)
            self._settings_built = True

    def _build_settings_panel(self, panel: tk.Misc) -> None:
        """Build the Settings section's contents (once, on first open)."""
        # The "Answering mouth" radio list is hidden on purpose: the header's model
        # selector is the one visible mouth switch. The variable still tracks the
        # live mouth (the header sets it), so nothing else has to change.
        self._source_var.set(self.service.source)

        self._build_cloud_provider_picker(panel)

        self._sidebar_button(
            panel, "⤓   Save conversation", self._on_save, small=True
        ).pack(fill="x", padx=12, pady=(6, 0))
        self._sidebar_button(
            panel, "⌫   Clear transcript", self._clear_transcript, small=True
        ).pack(fill="x", padx=12, pady=(2, 0))

        # Appearance: one switch, with the same one-line note the old Appearance
        # block carried, kept inside Settings so the sidebar has a single home for
        # every preference.
        row = tk.Frame(panel, bg=self._theme.surface)
        row.pack(fill="x", padx=12, pady=(10, 2))
        right = "light mode  ☀" if self._theme.name == "dark" else "dark mode  ☾"
        self._sidebar_button(row, right, self._toggle_theme, small=True).pack(
            fill="x", pady=(0, 4)
        )
        tk.Label(
            row, text="Theme and mode are saved for next launch.", bg=self._theme.surface,
            fg=self._theme.text_dim, font=self._type.font("2xs"), anchor="w", padx=12,
        ).pack(fill="x")

    def _build_cloud_provider_picker(self, panel: tk.Misc) -> None:
        """The Cloud provider picker: choose Gemini *or* Groq (or any provider).

        Shown at the top of the settings panel so a user who picks the Cloud mouth
        can also choose *which* remote service answers -- Gemini or Groq for the free
        tier, or any other provider. The choice is remembered, so the next answer and
        the next launch use it without a flag. It is a plain either/or menu: the
        current provider is the one shown, and picking another one stores it at once.
        """
        from core.cloud_model import provider_summaries

        provider = self.service.cloud_provider()
        self._cloud_provider_var = tk.StringVar(value=provider)
        row = tk.Frame(panel, bg=self._theme.surface)
        row.pack(fill="x", padx=14, pady=(6, 0))
        tk.Label(
            row, text="Cloud provider", bg=self._theme.surface,
            fg=self._theme.text_secondary, font=self._type.font("2xs"),
            anchor="w", padx=4,
        ).pack(side="left")
        menu = tk.OptionMenu(
            row, self._cloud_provider_var,
            *[key for key, *_rest in provider_summaries()],
            command=self._on_cloud_provider_change,
        )
        menu.configure(
            bg=self._theme.surface, fg=self._theme.foreground, font=self._type.font("xs"),
            activebackground=self._theme.interactive_hover, activeforeground=self._theme.foreground,
            highlightthickness=0, bd=0, relief="flat", cursor="hand2", anchor="w",
        )
        menu["menu"].configure(
            bg=self._theme.surface, fg=self._theme.foreground, font=self._type.font("xs"),
        )
        menu.pack(side="right", padx=4)
        self._cloud_provider_note = tk.Label(
            panel, text=self._cloud_provider_line(provider),
            bg=self._theme.surface, fg=self._theme.text_dim,
            font=self._type.font("2xs"), anchor="w", padx=20, pady=2,
            wraplength=232, justify="left",
        )
        self._cloud_provider_note.pack(fill="x")

        model_row = tk.Frame(panel, bg=self._theme.surface)
        model_row.pack(fill="x", padx=14, pady=(4, 0))
        tk.Label(
            model_row, text="Model", bg=self._theme.surface,
            fg=self._theme.text_secondary, font=self._type.font("2xs"),
            anchor="w", padx=4,
        ).pack(side="left")
        self._cloud_model_var = tk.StringVar(value=self.service.cloud_model())
        # A real dropdown, styled like the provider menu above it: ttk's default
        # Combobox draws an unthemed white box that reads as a text field, not a
        # menu, in this dark palette. An OptionMenu carries the theme's own colours,
        # so the model list is visibly a list.
        self._cloud_model_picker = tk.OptionMenu(
            model_row, self._cloud_model_var,
            *self.service.cloud_model_choices(),
            command=self._set_cloud_model,
        )
        self._cloud_model_picker.configure(
            bg=self._theme.surface, fg=self._theme.foreground, font=self._type.font("xs"),
            activebackground=self._theme.interactive_hover, activeforeground=self._theme.foreground,
            highlightthickness=0, bd=0, relief="flat", cursor="hand2", anchor="w",
        )
        self._cloud_model_picker["menu"].configure(
            bg=self._theme.surface, fg=self._theme.foreground, font=self._type.font("xs"),
        )
        self._cloud_model_picker.pack(side="left", fill="x", expand=True, padx=(2, 4))
        self._add_cloud_model_other_entry()

        # A one-line, always-visible answer to "which of these are free?": the
        # free-tier providers need no card, so a user who does not want to pay
        # should not have to hunt for them in the dropdown.
        free_note = tk.Label(
            panel,
            text="Free (no card): " + ", ".join(
                title for _key, title, _env, _signup in self.service.free_cloud_providers()
            ),
            bg=self._theme.surface, fg=self._theme.text_dim,
            font=self._type.font("2xs"), anchor="w", padx=20, pady=2,
            wraplength=232, justify="left",
        )
        free_note.pack(fill="x")
        free_button = tk.Button(
            panel, text="Free providers and how to start…", command=self._show_free_cloud_providers,
            bg=self._theme.surface, fg=self._theme.primary,
            activebackground=self._theme.interactive_hover, activeforeground=self._theme.primary,
            font=self._type.font("2xs"), relief="flat", borderwidth=0,
            anchor="w", padx=18, pady=2, cursor="hand2",
        )
        free_button.pack(fill="x")

    def _show_free_cloud_providers(self) -> None:
        """Post the free-tier providers and their signup links into the transcript.

        The window's shortcut for a user who does not want to pay: the no-card
        providers, the environment variable to set, where to create the key, and
        the exact commands to start talking to one -- the same list the CLI's
        ``cloud free`` prints.
        """
        rows = self.service.free_cloud_providers()
        lines = ["**Free cloud providers (no card needed)**", ""]
        for key, title, env, signup in rows:
            lines.append(f"- **{title}** (`{key}`) — set `{env}` · key: {signup}")
        lines.append("- **Ollama** / **LM Studio** — run on this machine, no key, no internet")
        lines.append("")
        example_key, example_title, example_env, example_signup = rows[0]
        lines.append("Two steps (example: " + example_title + "):")
        lines.append(f"1. Create a key: {example_signup}")
        lines.append(
            f"2. Set `{example_env}`, then choose **{example_key}** above, then ask."
        )
        lines.append("")
        lines.append(
            "Nothing is sent anywhere until you choose the Cloud mouth, and the key "
            "is read from the environment — never written to the repository."
        )
        self._add_bubble("assistant", "\n".join(lines))

    def _add_cloud_model_other_entry(self) -> None:
        """Offer an "Other…" row, so a model the catalog does not list can be set.

        The dropdown is the common case; a provider that renames a model faster
        than this catalog is updated still needs a way in, so the last entry asks
        for a name and stores it (the same path the CLI's ``--model`` uses).
        """
        menu = self._cloud_model_picker["menu"]
        menu.add_separator()
        menu.add_command(label="Other…", command=self._prompt_cloud_model)

    def _prompt_cloud_model(self) -> None:
        """Ask for a model name and store it (the dropdown's "Other…" row)."""
        current = self.service.cloud_model()
        typed = simpledialog.askstring(
            "Cloud model", "Model name to ask the provider for:", initialvalue=current, parent=self.root
        )
        if not typed:
            return
        try:
            stored = self.service.set_cloud_model(typed.strip())
        except ValueError as error:
            self._add_bubble("note", f"Could not set the Cloud model: {error}")
            return
        # Keep the custom name in the menu so it stays selectable.
        menu = self._cloud_model_picker["menu"]
        labels = [menu.entrycget(i, "label") for i in range(menu.index("end") + 1)]
        if stored not in labels:
            menu.insert_command(0, label=stored, command=partial(self._set_cloud_model, stored))
        self._cloud_model_var.set(stored)
        self._add_bubble("note", f"Cloud will ask for model {stored!r}.")

    def _refresh_cloud_model_choices(self) -> None:
        """Re-point the model dropdown at the current provider's lineup.

        Called when the provider changes, so the menu always lists *that*
        provider's models (Gemini's for Gemini, Groq's for Groq) instead of the
        previous provider's.
        """
        if not hasattr(self, "_cloud_model_picker"):
            return
        menu = self._cloud_model_picker["menu"]
        menu.delete(0, "end")
        for model in self.service.cloud_model_choices():
            menu.add_command(label=model, command=partial(self._set_cloud_model, model))
        self._add_cloud_model_other_entry()
        self._cloud_model_var.set(self.service.cloud_model())

    def _set_cloud_model(self, model: str) -> None:
        """Set the shown model and store it (the dropdown's per-entry callback)."""
        self._cloud_model_var.set(model)
        self._on_cloud_model_change()

    def _on_cloud_model_change(self) -> None:
        """Store the picked Cloud model, so a retired default is fixed once.

        A provider renames and shuts models down; when the built-in default stops
        working, choosing a working name here (e.g. ``gemini-3.8-flash``) makes every
        later answer use it, and the window and the CLI agree.
        """
        typed = self._cloud_model_var.get().strip()
        try:
            stored = self.service.set_cloud_model(typed)
        except ValueError as error:
            self._add_bubble("note", f"Could not set the Cloud model: {error}")
            return
        self._cloud_model_var.set(stored)
        self._add_bubble("note", f"Cloud will ask for model {stored!r}.")

    def _cloud_provider_line(self, key: str) -> str:
        """An honest one-line note about a provider: key needed, free tier, or local."""
        import os

        from core.cloud_model import resolve_provider

        spec = resolve_provider(key)
        if not spec.needs_key:
            return f"{spec.title}: on this machine, no key needed."
        has_key = bool(os.environ.get(spec.api_key_env))
        tier = "free tier" if spec.free else "paid"
        if has_key:
            return f"{spec.title}: key found ({spec.api_key_env}), {tier}."
        return f"{spec.title}: {tier}. Set {spec.api_key_env} to use it."

    def _on_cloud_provider_change(self, key: str) -> None:
        """Store the chosen Cloud provider and refresh the note and the header."""
        try:
            stored = self.service.set_cloud_provider(key)
        except ValueError as error:
            self._add_bubble("note", f"Could not change the Cloud provider: {error}")
            self._cloud_provider_var.set(self.service.cloud_provider())
            return
        if hasattr(self, "_cloud_provider_note"):
            self._cloud_provider_note.configure(text=self._cloud_provider_line(stored))
        self._refresh_cloud_model_choices()
        self._source_menu_var_reset()
        self._add_bubble("note", f"Cloud will answer from {self._cloud_provider_line(stored)}")

    def _build_voice_picker(self, panel: tk.Misc) -> None:
        """The reading-voice list plus the auto-speak toggle.

        The voices are grouped by the **source** they came from -- the OS's own
        installed voices, the local VoiceTut-TTS model, Piper, Qwen3-TTS, and the
        synthetic catalog -- each under the name of its source, with an honest note
        for any source that contributed no voice here. This is the *one* voice menu
        in the sidebar (the Settings section no longer repeats it).
        """
        self._build_voice_sections(panel)
        tk.Checkbutton(
            panel, text="Read new answers aloud automatically", variable=self._auto_speak,
            command=self._on_auto_speak_toggle, bg=self._theme.surface, fg=self._theme.foreground,
            selectcolor=self._theme.surface, activebackground=self._theme.surface,
            activeforeground=self._theme.foreground, font=self._type.font("xs"),
            anchor="w", padx=6, highlightthickness=0, bd=0, cursor="hand2",
        ).pack(fill="x", padx=16, pady=(6, 0))
        self._sidebar_button(
            panel, "▷   Test the reading voice", self._test_voice, small=True
        ).pack(fill="x", padx=12, pady=(4, 0))
        if not self.service.reading_available():
            self._sidebar_button(
                panel, "⤓   Install a speech engine…", self._on_install_speech,
                small=True,
            ).pack(fill="x", padx=12, pady=(2, 0))

    def _build_voice_sections(self, panel: tk.Misc) -> None:
        """The voice groups themselves, used by the reading-voice section.

        Every voice sits under the name of the **source** it came from --
        *Installed Voices*, *VoiceTut-TTS*, *Piper*, *Qwen3-TTS*, and the
        synthetic catalog -- and each source shows only the voices its engine
        really reports here, so a listed voice is always one that can actually
        speak (an engine that is not installed contributes no voices, and the
        synthetic catalog is dropped entirely when the system engine itself is
        absent, since those settings would produce silence).
        """
        for section in self.service.voice_sections():
            self._voice_group_label(panel, section.title, indent=22)
            for header, group in section.groups:
                self._voice_group_label(panel, header, indent=28, dim=True)
                for voice in group:
                    suffix = f"  ({voice.gender})" if getattr(voice, "gender", "") else ""
                    label = voice.label() if hasattr(voice, "label") else voice.label_en
                    self._voice_radio(panel, f"{label}{suffix}", voice.key, indent=28)

        for note in self.service.voice_source_notes():
            tk.Label(
                panel, text=note, bg=self._theme.surface, fg=self._theme.text_dim,
                font=self._type.font("2xs"), anchor="w", padx=20, pady=2,
                wraplength=232, justify="left",
            ).pack(fill="x")

    def _voice_group_label(
        self, panel: tk.Misc, text: str, *, indent: int, dim: bool = False
    ) -> None:
        """A group header inside a voice menu (the two menus share this too)."""
        tk.Label(
            panel, text=text, bg=self._theme.surface,
            fg=self._theme.text_dim if dim else self._theme.text_secondary,
            font=self._type.font("2xs"), anchor="w", padx=indent,
        ).pack(fill="x", pady=(4, 0))

    def _voice_radio(self, panel: tk.Misc, label: str, key: str, *, indent: int) -> None:
        """One selectable voice, wired to the same handler in both menus."""
        tk.Radiobutton(
            panel, text=label, value=key, variable=self._speak_var,
            command=partial(self._on_voice_change, key),
            bg=self._theme.surface, fg=self._theme.foreground,
            selectcolor=self._theme.surface, activebackground=self._theme.surface,
            activeforeground=self._theme.foreground, font=self._type.font("xs"),
            anchor="w", padx=6, highlightthickness=0, bd=0, cursor="hand2",
        ).pack(fill="x", padx=indent, pady=0)

    def _toggle_voices(self) -> None:
        """Ensure the sidebar's **Reading Voice** section is open, and fill it.

        The section header owns the real open/close click; this is the programmatic
        "show the reading voice" entry point, so it is idempotent. The voice list is
        (re)built each time it is opened, so it always reflects what is installed
        right now; this is the one place a reading voice is chosen.
        """
        self._open_section("voices")
        self._rebuild_voice_section()

    def _rebuild_voice_section(self) -> None:
        """Fill the Reading Voice section from the engines' live voice lists."""
        panel = getattr(self, "_voices_panel", None)
        if panel is None:
            return
        for child in panel.winfo_children():
            child.destroy()
        # A stored voice that is no longer installed must not leave the list with
        # nothing selected, so the live key is re-applied to the variable first.
        self._speak_var.set(self._voice_key)
        self._build_voice_picker(panel)

    def _on_install_speech(self) -> None:
        """Install a real speech engine (a system package), on the worker thread."""
        from core.speech import speech_install_plan

        plan = speech_install_plan()
        # The result message carries the plan's hint when it cannot install, so only
        # the "starting" line is announced here to avoid printing the hint twice.
        if plan.can_install:
            self._add_bubble("note", f"Installing a speech engine: {' '.join(plan.command)}")
        self._start_work(self._install_speech_worker)

    def _install_speech_worker(self):
        """Run the install and return the result for :meth:`_render_result`."""
        from core.speech import install_speech_engine

        return install_speech_engine()

    def _refresh_voice_menus(self) -> None:
        """Rebuild the open Reading Voice section so newly installed voices appear.

        The voice list is the only menu that shows the live engines, so this only
        has to rebuild it; a closed section picks the new voices up when it is next
        opened.
        """
        section = self._sections.get("voices")
        if section is not None and section.is_open:
            self._rebuild_voice_section()

    def _test_voice(self) -> None:
        """Speak a short sample so the user can hear the chosen voice."""
        sample = "مرحبا، هذه قراءة تجريبية بصوت فيض الروح."
        self._speak_text(sample)

    # -------------------------------------------------------------------- main

    def _build_main(self, shell: tk.Frame) -> None:
        main = tk.Frame(shell, bg=self._theme.base)
        main.pack(side="left", fill="both", expand=True)
        self._main_frame = main

        self._build_header(main)

        container = tk.Frame(main, bg=self._theme.base)
        container.pack(fill="both", expand=True)
        self._canvas = tk.Canvas(container, bg=self._theme.base, highlightthickness=0)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._scrollbar = tk.Scrollbar(container, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        self._transcript = tk.Frame(self._canvas, bg=self._theme.base)
        self._row_frames: list[tk.Frame] = []
        self._window = self._canvas.create_window((0, 0), window=self._transcript, anchor="nw")
        self._transcript.bind("<Configure>", self._on_transcript_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        for widget in (self._canvas, self._transcript):
            widget.bind("<MouseWheel>", self._on_mousewheel)
            widget.bind("<Button-4>", self._on_mousewheel)
            widget.bind("<Button-5>", self._on_mousewheel)

        self._empty: tk.Widget | None = None
        self._build_composer(main)

    def _build_header(self, main: tk.Frame) -> None:
        """The slim title bar: mode switch, model selector, and menu controls.

        The bar is the window's own title band, in the same dark grey every
        dialog uses, and it carries the same three controls on its right -- so
        the main window and every secondary one read as one program.
        """
        bar = tk.Frame(main, bg=self._theme.titlebar)
        bar.pack(fill="x")
        inner = tk.Frame(bar, bg=self._theme.titlebar)
        inner.pack(fill="x", padx=12, pady=8)

        # Left: the sidebar toggle (also the mobile drawer handle) and the title.
        self._sidebar_toggle = self._icon_button(
            inner, "☰", self._toggle_sidebar, tip="Show or hide the sidebar"
        )
        self._sidebar_toggle.pack(side="left")

        # The mode control: the project's one model, shown by name. The window no
        # longer switches between two operating modes -- the Fayd Al-Ruh Model is
        # the chat, and Jarvis (sources + web) is chosen from the model selector on
        # the right -- so this is a plain button naming the model in force, and
        # pressing it returns to that model from any other character.
        self._mode_var = tk.StringVar(value=self.service.operating_mode.label)
        self._mode_menu = tk.Button(
            inner, textvariable=self._mode_var, command=self._on_mode_menu,
            bg=self._theme.surface_raised, fg=self._theme.contrast,
            activebackground=self._theme.interactive_active, activeforeground=self._theme.contrast,
            font=self._type.font("sm", weight="bold"), padx=10, pady=4,
            relief="flat", borderwidth=0, cursor="hand2",
        )
        self._mode_menu.pack(side="left", padx=(8, 0))
        _Tooltip(self._mode_menu, "Answer with the project's own trained model", self._theme, self._type)
        # The one-line mode description, shown small beside the name. It is kept
        # short so the header never outgrows a narrow window (the model selector
        # and the three window controls share the same bar).
        self._mode_label = tk.Label(
            inner, text="", bg=self._theme.titlebar, fg=self._theme.text_secondary,
            font=self._type.font("xs"), anchor="w",
        )
        self._mode_label.pack(side="left", padx=(10, 0))

        # Right: the model selector (the window's real mouth switch) and theme.
        self._source_var_menu = tk.StringVar(value=self._source_choice_label())
        self._source_menu = tk.OptionMenu(
            inner, self._source_var_menu,
            *[label for _key, label in MOUTH_CHOICES], command=self._on_source_menu,
        )
        self._source_menu.configure(
            bg=self._theme.titlebar, fg=self._theme.foreground, activebackground=self._theme.surface_raised,
            activeforeground=self._theme.contrast, relief="flat", highlightthickness=0,
            font=self._type.font("xs"), padx=6, pady=3, bd=0, cursor="hand2",
        )
        self._source_menu["menu"].configure(bg=self._theme.surface, fg=self._theme.foreground)
        self._source_menu.pack(side="right")
        self._icon_button(inner, "☾" if self._theme.name == "dark" else "☀", self._toggle_theme, tip="Toggle theme").pack(
            side="right", padx=(0, 6)
        )

        # The same minimize / maximize / close trio every dialog carries, in the
        # same place, so the main window is closed and grown the same way.
        self._window_controls = install_window_controls(
            self.root, inner, theme=self._theme, typography=self._type,
            on_close=self._on_close,
        )

        tk.Frame(main, bg=self._theme.titlebar_edge, height=1).pack(fill="x")

    def _icon_button(self, parent: tk.Misc, glyph: str, command: Callable[[], Any], *, tip: str = "") -> tk.Button:
        button = tk.Button(
            parent, text=glyph, command=command, bg=self._theme.titlebar, fg=self._theme.foreground,
            activebackground=self._theme.surface_raised, activeforeground=self._theme.contrast,
            font=self._type.font("sm"), relief="flat", borderwidth=0, padx=8, pady=2, cursor="hand2",
        )
        if tip:
            _Tooltip(button, tip, self._theme, self._type)
        return button

    def _column_pad(self) -> int:
        """Side padding that centres the transcript in a readable column."""
        canvas = getattr(self, "_canvas", None)
        width = canvas.winfo_width() if canvas is not None else 0
        if width <= 1:
            return 20
        return max(16, (width - _COLUMN_MAX) // 2)

    def _bubble_fonts(self) -> tuple[tkfont.Font, tkfont.Font]:
        """The proportional and mono fonts a bubble measures its text with.

        Measured at the same sizes the body and code tags draw with, so the
        width a bubble reserves matches the text it will hold.
        """
        return (
            tkfont.Font(font=self._type.font("body")),
            tkfont.Font(font=self._type.mono("code")),
        )

    def _available_column(self) -> int:
        """The transcript column's pixel width, capped for readability."""
        canvas = getattr(self, "_canvas", None)
        width = canvas.winfo_width() if canvas is not None else 0
        if width <= 1:
            return _COLUMN_MAX
        return max(160, min(_COLUMN_MAX, width - 2 * self._column_pad()))

    def _bubble_content_width(self, text: str, *, role: str) -> int:
        """The pixel width a bubble's text may occupy: fit the widest line, capped.

        The height is set separately by :meth:`_fit_height`; together the two make a
        bubble grow with the amount of text it holds instead of being a fixed box.
        A user bubble is right-aligned and a little narrower than the column; an
        assistant bubble may use the full column but never overflows it.
        """
        normal, mono = self._bubble_fonts()
        widest = 0
        for line in md.iter_styled_lines(text):
            font = mono if line.in_code else normal
            widest = max(widest, font.measure(line.text))
        if role == "user":
            return max(80, min(widest, int(_COLUMN_MAX * 0.84) - 52))
        pad = 6 if role != "note" else 0
        return max(120, min(widest, self._available_column() - 2 * pad))

    def _bubble_text_width(self, text: str, *, role: str) -> int:
        """The ``Text`` width, in characters, that yields the content width."""
        normal, _mono = self._bubble_fonts()
        char_px = max(1, normal.measure("0"))
        content = self._bubble_content_width(text, role=role)
        return max(1, -(-content // char_px))

    def _bubble_card_width(self, text: str) -> int:
        """The pixel width of a user bubble's rounded card, chrome included."""
        return self._bubble_content_width(text, role="user") + 52

    def _show_hero(self) -> None:
        """The welcome/empty state, shown whenever the transcript is empty."""
        if self._empty is not None:
            return
        self._empty = tk.Frame(self._transcript, bg=self._theme.base)
        self._empty.pack(pady=(64, 0), padx=24)
        badge = tk.Canvas(
            self._empty, width=64, height=64, bg=self._theme.base, highlightthickness=0
        )
        badge.create_oval(1, 1, 63, 63, fill=self._theme.primary, outline="")
        badge.create_text(32, 34, text="ف", fill=self._theme.primary_foreground, font=self._type.font("hero", weight="bold"))
        badge.pack()
        tk.Label(
            self._empty, text="Fayd Al-Ruh",
            bg=self._theme.base, fg=self._theme.contrast, font=self._type.font("hero", weight="bold"),
        ).pack(pady=(14, 0))
        tk.Label(
            self._empty, text="Ask anything about petroleum engineering",
            bg=self._theme.base, fg=self._theme.foreground, font=self._type.font("sm"),
        ).pack(pady=(6, 0))
        tk.Label(
            self._empty, text=self._hero_subtitle(),
            bg=self._theme.base, fg=self._theme.text_secondary, font=self._type.font("body"),
            justify="center", wraplength=560,
        ).pack(pady=(10, 0))
        tk.Label(
            self._empty,
            text=(
                "Answering mouths: the “Fayd Al-Ruh Model” answers from its own "
                "trained knowledge · “Jarvis” cites the project's sources (and the "
                "web) · “Ultron” answers anything, locally · “Vision” is the "
                "petroleum local model · “Banner” drafts presentation outlines and "
                "slide content · “Natasha”, “Hakeem”, and “Cloud” each answer in "
                "their own character, with their own memory and learning"
            ),
            bg=self._theme.base, fg=self._theme.text_dim, font=self._type.font("xs"), justify="center",
            wraplength=560,
        ).pack(pady=(16, 0))

    def _hero_subtitle(self) -> str:
        return (
            "Fayd Al-Ruh Model: reservoir rock, drilling, production, petrophysics … "
            "answers from the project's knowledge and its own trained model, and from "
            "the web when you turn it on."
        )

    # -------------------------------------------------------------- transcript

    def _add_bubble(self, role: str, text: str, *, extra: str = "") -> tk.Text:
        """Add one message row and return its selectable read-only text widget."""
        pad = self._column_pad()
        row = tk.Frame(self._transcript, bg=self._theme.base)
        row.pack(fill="x", padx=(pad, pad), pady=6)
        self._row_frames.append(row)
        self._hide_empty()

        if role == "user":
            card = RoundedBox(
                row, bg=self._theme.user_bubble, fill=self._theme.user_bubble,
                outline=self._theme.user_bubble, corner=RADIUS_LARGE,
                parent_bg=self._theme.base,
            )
            card.pack(side="right", anchor="e")
            widget = tk.Text(
                card.inner, height=1, wrap="word", relief="flat", borderwidth=0,
                highlightthickness=0, bg=self._theme.user_bubble, fg=self._theme.user_bubble_fg,
                font=self._type.font("body"), padx=14, pady=8, cursor="xterm", takefocus=1, width=0,
            )
            widget.insert("1.0", text)
            self._make_read_only(widget)
            widget.pack(fill="both", expand=True)
            card.configure(width=self._bubble_card_width(text))
            self._fit_height(widget)
            body = text
        else:
            bg = self._theme.base
            head = tk.Frame(row, bg=bg)
            head.pack(fill="x", pady=(0, 4))
            if role == "note":
                accent, fg = self._theme.text_dim, self._theme.text_secondary
            else:
                self._avatar(head)
                accent, fg = self._theme.text_dim, self._theme.foreground
            tk.Label(
                head, text=self._role_label(role), bg=bg, fg=accent,
                font=self._type.font("xs", weight="bold"), anchor="w",
            ).pack(side="left", padx=(6 if role != "note" else 0, 0))

            widget = tk.Text(
                row, height=1, wrap="word", relief="flat", borderwidth=0,
                highlightthickness=0, bg=bg, fg=fg, font=self._type.font("body"),
                padx=(6 if role != "note" else 0), pady=0, cursor="xterm", takefocus=1,
            )
            rich_text.apply_body_spacing(widget, self._type)
            body = f"{text}\n\n{extra}" if extra else text
            self._insert_rich(widget, body)
            self._make_read_only(widget)
            widget.configure(width=self._bubble_text_width(body, role=role))
            widget.pack(anchor="w")
            self._fit_height(widget)

        for wheel in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(wheel, self._on_bubble_wheel, add="+")

        footer = tk.Frame(row, bg=self._theme.base)
        footer.pack(fill="x", pady=(2, 0))
        if role != "user":
            self._copy_label(footer, "⧉ Copy", widget)
            retry = tk.Label(
                footer, text="↻ Retry", bg=self._theme.base, fg=self._theme.text_dim,
                font=self._type.font("2xs"), cursor="hand2",
            )
            retry.pack(side="left", padx=(10, 0))
            retry.bind("<Button-1>", lambda _e: self._on_retry())
            retry.bind("<Enter>", lambda _e: retry.configure(fg=self._theme.foreground))
            retry.bind("<Leave>", lambda _e: retry.configure(fg=self._theme.text_dim))
        else:
            self._copy_label(footer, "⧉ Copy", widget)
        self._speak_label(footer, body, side="left")

        self._rows.append(_Row(role=role, widget=widget, row=row, text=text, body=body))
        self._scroll_to_bottom()
        return widget

    def _copy_label(self, footer: tk.Frame, text: str, widget: tk.Text) -> None:
        copy = tk.Label(
            footer, text=text, bg=self._theme.base, fg=self._theme.text_dim,
            font=self._type.font("2xs"), cursor="hand2",
        )
        copy.pack(side="right" if widget.cget("bg") == self._theme.user_bubble else "left")
        copy.bind("<Button-1>", lambda _e: self._copy_widget(widget))
        copy.bind("<Enter>", lambda _e: copy.configure(fg=self._theme.foreground))
        copy.bind("<Leave>", lambda _e: copy.configure(fg=self._theme.text_dim))

    # ------------------------------------------------------------- reading

    def _speak_label(
        self, footer: tk.Frame, text: str, *, side: Literal["left", "right", "top", "bottom"] = "left"
    ) -> None:
        """Add the per-message read-aloud toggle (🔊 Read aloud / ⏹ Stop).

        Clicking it once starts reading the message with the chosen voice; a
        second click stops the reading. Only one message is read at a time, so a
        click on another message's control stops the first before starting.
        """
        if not text.strip():
            return
        label = tk.Label(
            footer, text="🔊 Read aloud", bg=self._theme.base, fg=self._theme.text_dim,
            font=self._type.font("2xs"), cursor="hand2",
        )
        label.pack(side=side, padx=(10, 0))
        label.bind("<Button-1>", lambda _e: self._toggle_speak(label, text))
        label.bind("<Enter>", lambda _e: label.configure(fg=self._theme.foreground))
        label.bind("<Leave>", lambda _e: label.configure(fg=self._theme.text_dim))

    def _toggle_speak(self, label: tk.Label, text: str) -> None:
        """Start reading ``text``, or stop the reading if this message is playing.

        The active control is remembered as soon as the reading starts, so a
        second click stops it even while the clip is still being rendered.
        """
        if self._speak_control is label:
            self._stop_speaking()
            return
        self._speak_text(text, label=label)

    def _speak_text(self, text: str, *, label: tk.Label | None = None) -> None:
        """Read ``text`` aloud in the background, reporting honestly if it cannot.

        A reading already in progress is stopped first, so starting a new one (or
        auto-speak after a fresh answer) never leaves two voices talking at once.
        """
        clean = _strip_markdown(text).strip()
        if not clean:
            return
        self._stop_speaking()
        if not self.service.reading_available():
            from core.speech import speech_install_plan

            self._add_bubble(
                "note",
                "No speech engine is installed, so I cannot speak. "
                + speech_install_plan().hint
                + " Or run `python -m cli speak --install-engine`.",
            )
            return
        self._speak_token += 1
        token = self._speak_token
        self._speaking = True
        if label is not None:
            label.configure(text="⏹ Stop")
            self._speak_control = label
        self._status.configure(text="Speaking…")
        self._start_speech_job(
            lambda: _SpokenClip(clean, self.service.render_aloud(clean), token)
        )

    def _start_speech_job(self, job) -> None:
        """Run a short speech job (render/transcribe) off the busy path.

        The composer and the transcript stay responsive, and the result is handed
        back through the pump on the Tk thread; a render carries its own token in
        the returned :class:`_SpokenClip`, so a newer click can discard a
        superseded clip when it arrives.
        """

        def run() -> None:
            try:
                self._speech_queue.put(("ok", job()))
            except Exception as error:  # surfaced in the transcript, never crashes Tk
                self._speech_queue.put(("error", error))

        threading.Thread(target=run, daemon=True).start()

    def _play_rendered(self, payload: _SpokenClip) -> None:
        """Play a rendered clip, unless a newer click has superseded it."""
        if payload.token != self._speak_token:
            # A stop or a newer click happened while this clip was rendering.
            return
        if payload.clip.is_silent:
            # The engine produced no audio at all. Report it honestly rather than
            # showing a "Reading aloud (0.0 s)" note as though it had spoken, and
            # point at the real next step (a different voice, or an engine install).
            self._speaking = False
            self._reset_speak_control()
            self._add_bubble(
                "note",
                "That voice produced no audio on this machine (the engine returned "
                "silence). Try another voice, or install a speech engine from the "
                "Voices menu.",
            )
            return
        playback = self.service.play_clip(payload.clip)
        if playback is None:
            self._add_bubble(
                "note",
                "No audio player is installed (install aplay, afplay, or sox), "
                "so the reading could not be played.",
            )
            self._speaking = False
            self._reset_speak_control()
            return
        self._playback = playback
        duration = payload.clip.duration_seconds
        self._add_bubble(
            "note",
            f"🔊 Reading aloud with the {self._voice_label()} voice "
            f"({duration:.1f} s) — click Stop to end it.",
        )
        self._watch_playback(playback)

    def _watch_playback(self, playback: Any) -> None:
        """Poll a playback on the Tk thread; clear the state when it ends."""

        def poll() -> None:
            if self._playback is not playback:
                return
            if playback.playing:
                self.root.after(200, poll)
                return
            self._finish_playback()

        self.root.after(200, poll)

    def _finish_playback(self) -> None:
        """The clip ended on its own: clear the playing state and restore the label."""
        playback = self._playback
        self._playback = None
        self._speaking = False
        self._reset_speak_control()
        self._update_status()
        if playback is not None:
            with contextlib.suppress(Exception):
                playback.wait()

    def _stop_speaking(self) -> None:
        """Stop the reading in progress, if any, and restore its control.

        The token is bumped so a clip still rendering on the worker thread is
        discarded when it arrives, rather than starting to talk after the stop.
        """
        self._speak_token += 1
        playback = self._playback
        self._playback = None
        self._speaking = False
        self._reset_speak_control()
        if playback is not None:
            with contextlib.suppress(Exception):
                playback.stop()

    def _reset_speak_control(self) -> None:
        """Restore the speaker control that was showing "Stop" to "Read aloud"."""
        if self._speak_control is not None:
            with contextlib.suppress(tk.TclError):
                self._speak_control.configure(text="🔊 Read aloud")
            self._speak_control = None

    # -------------------------------------------------------------- listening

    def _toggle_mic(self) -> None:
        """Start a recording, or stop it and transcribe with the local Qwen3-ASR."""
        if self._recording is not None:
            self._finish_recording()
            return
        if not self.service.microphone_available():
            self._add_bubble("note", self.service.asr_status())
            return
        if self.service.recognizer_name() == "transcript":
            self._add_bubble("note", self.service.asr_status())
            return
        try:
            self._recording = self.service.start_recording()
        except ValueError as error:
            self._add_bubble("note", f"Could not start recording: {error}")
            self._recording = None
            return
        self._set_mic_recording(True)
        self._status.configure(text="Listening… click the microphone again to stop.")
        self._add_bubble("note", "🎤 Listening… click the microphone again to stop and transcribe.")

    def _on_web_toggled(self, enabled: bool) -> None:
        """Turn internet lookups on or off from the composer's switch.

        The switch is a *view* of the session's real state: it is repainted from
        what the service reports after the change, so a lookup that cannot be
        configured (or was refused) never leaves the control reading "on".
        """
        now = self.service.set_web(enabled)
        self._refresh_web_toggle()
        if enabled and not now:
            self._add_bubble("note", self.service.web_status())
            return
        self._status.configure(text=self.service.web_status())

    def _refresh_web_toggle(self) -> None:
        """Keep the switch in step with the live session (after any change)."""
        toggle = getattr(self, "_web_toggle", None)
        if toggle is None:
            return
        toggle.set_state(
            self.service.web_enabled(),
            available=self.service.web_available(),
        )

    def _finish_recording(self) -> None:
        """Stop the recording and transcribe it in the background."""
        recording = self.service.stop_recording()
        self._recording = None
        self._set_mic_recording(False)
        if recording is None:
            return
        self._status.configure(text="Transcribing…")
        self._start_speech_job(lambda: _Transcript(self._transcribe(recording)))

    def _transcribe(self, recording: Any) -> str:
        """Finish the recording and transcribe its clip (runs on the worker thread)."""
        clip = recording.result()
        return self.service.transcribe(clip)

    def _cancel_recording(self) -> None:
        """Abandon a recording in progress (a new chat, or a resume)."""
        self.service.stop_recording()
        self._recording = None
        self._set_mic_recording(False)

    def _set_mic_recording(self, recording: bool) -> None:
        """Reflect the recording state on the microphone button."""
        self._listening = recording
        with contextlib.suppress(tk.TclError):
            self._composer.set_mic_recording(recording)

    def _voice_label(self) -> str:
        from core.voices import (
            find_model_voice,
            find_system_voice,
            find_voice,
            is_model_voice_key,
            is_system_voice_key,
        )

        if is_system_voice_key(self._voice_key):
            installed = find_system_voice(self._voice_key)
            if installed is None:
                return self._voice_key
            return f"{installed.category_label()} “{installed.label()}”"
        if is_model_voice_key(self._voice_key):
            model_voice = find_model_voice(self._voice_key)
            if model_voice is None:
                return self._voice_key
            return f"{model_voice.category_label()} “{model_voice.label()}”"
        try:
            catalog_voice = find_voice(self._voice_key)
        except ValueError:
            return self._voice_key
        return f"{catalog_voice.category_label()} “{catalog_voice.label()}”"

    def _on_voice_change(self, key: str) -> None:
        """Apply the chosen reading voice, persist it, and say which one is active."""
        try:
            voice = self.service.set_voice(key)
        except ValueError as error:
            self._add_bubble("note", f"Could not use that voice: {error}")
            return
        self._voice_key = voice.key
        self._speak_var.set(voice.key)
        save_reading_voice(voice.key)
        self._add_bubble("note", f"Reading voice set to {voice.category_label()} “{voice.label()}”.")

    def _on_auto_speak_toggle(self) -> None:
        """Persist the auto-speak choice and confirm it."""
        enabled = bool(self._auto_speak.get())
        save_auto_speak(enabled)
        self._add_bubble(
            "note",
            "Auto-speak is on: new answers will be read aloud." if enabled
            else "Auto-speak is off.",
        )

    def _insert_rich(self, widget: tk.Text, text: str) -> None:
        """Render Markdown into a message's Text widget (shared renderer)."""
        rich_text.insert_markdown(widget, text, theme=self._theme, typography=self._type)

    def _avatar(self, parent: tk.Misc) -> None:
        """A small gold disc with the brand letter, beside an assistant message."""
        badge = tk.Canvas(parent, width=24, height=24, bg=self._theme.base, highlightthickness=0)
        badge.create_oval(1, 1, 23, 23, fill=self._theme.primary, outline="")
        badge.create_text(12, 13, text="ف", fill=self._theme.primary_foreground, font=self._type.font("sm", weight="bold"))
        badge.pack(side="left")

    def _reflow(self) -> None:
        """Recentre rows and resize bubbles after the window width changes."""
        pad = self._column_pad()
        for row in self._row_frames:
            row.pack_configure(padx=(pad, pad))
        for entry in self._rows:
            if entry.role == "user":
                box = entry.widget.master.master if entry.widget.master is not None else None
                if isinstance(box, RoundedBox):
                    box.configure(width=self._bubble_card_width(entry.body))
                self._fit_height(entry.widget)
            else:
                entry.widget.configure(width=self._bubble_text_width(entry.body, role=entry.role))
                self._fit_height(entry.widget)

    @staticmethod
    def _role_label(role: str) -> str:
        if role == "user":
            return "You"
        if role == "note":
            return "Status"
        return "Fayd Al-Ruh"

    def _copy_widget(self, widget: tk.Text) -> None:
        self._copy_text(widget.get("1.0", "end-1c"))

    @staticmethod
    def _make_read_only(widget: tk.Text) -> None:
        """Keep a Text editable-looking but read-only, yet still selectable/copyable."""
        def block(event: tk.Event) -> str:
            if int(event.state) & 0x0004 or int(event.state) & 0x0008:  # Control / Command
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

    def _fit_height(self, widget: tk.Text) -> None:
        """Size a read-only Text to its wrapped content, up to a scroll cap."""
        widget.update_idletasks()
        measured = widget.count("1.0", "end", "displaylines")
        lines = int(measured[0]) if measured else 1
        widget.configure(height=min(max(lines, 1), _MAX_BUBBLE_LINES))

    def _bubble_texts(self) -> list[str]:
        return [entry.text for entry in self._rows]

    def _hide_empty(self) -> None:
        empty = getattr(self, "_empty", None)
        if empty is not None:
            empty.destroy()
            self._empty = None

    def _on_right_click(self, event: tk.Event) -> None:
        try:
            self._clipboard_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._clipboard_menu.grab_release()

    def _copy_text(self, text: str) -> None:
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update_idletasks()

    def _selected_text(self) -> str:
        widget = self.root.focus_get()
        if isinstance(widget, tk.Text):
            try:
                if widget.tag_ranges("sel"):
                    return widget.get("sel.first", "sel.last")
            except tk.TclError:
                pass
        for entry in self._rows:
            try:
                if entry.widget.tag_ranges("sel"):
                    return entry.widget.get("sel.first", "sel.last")
            except tk.TclError:
                continue
        return ""

    def _on_copy(self, _event: tk.Event | None = None) -> str:
        selection = self._selected_text()
        if not selection:
            return ""
        self._copy_text(selection)
        return "break"

    def _on_copy_conversation(self) -> None:
        self._copy_text("\n\n".join(self._bubble_texts()))

    def _on_select_all(self, _event: tk.Event | None = None) -> str:
        if self.root.focus_get() is self._entry:
            return ""
        for entry in self._rows:
            entry.widget.tag_add("sel", "1.0", "end-1c")
        return "break"

    def _on_clear_transcript(self, _event: tk.Event | None = None) -> str:
        if self.root.focus_get() is self._entry:
            return ""
        self._clear_transcript()
        return "break"

    def _clear_transcript(self) -> None:
        for entry in list(self._rows):
            entry.row.destroy()
        self._rows.clear()
        self._row_frames.clear()
        self._show_hero()

    def _on_transcript_configure(self, _event: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._toggle_scrollbar()

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)
        self._reflow()
        if self._empty is not None:
            self._empty.pack_configure(padx=self._column_pad())

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

    # ------------------------------------------------------------- composer

    def _on_composer_configure(self, _event: tk.Event | None = None) -> None:
        """Lift the composer by half the input box's height.

        The lift is extra bottom padding under the composer, so it moves the
        input box (and everything above it) up by half the box's own height
        without changing the layout that already pins the composer to the bottom.

        The height used is the box's *requested* height, not its allocated one:
        the lift changes the space the parent gives the box, so measuring the
        allocated height would feed the lift back into its own input and oscillate
        (each ``<Configure>`` shrinking the box, which shrinks the lift, which
        grows the box). The requested height is the height of the box's own
        contents and is independent of the lift, so the loop cannot close.
        """
        box_height = self._composer_box.winfo_reqheight()
        if box_height <= 1:
            return
        lift = box_height // 2
        if lift != self._composer_lift:
            self._composer_lift = lift
            self._composer_wrap.pack_configure(pady=(8, 14 + lift))

    def _build_composer(self, main: tk.Frame) -> None:
        composer = tk.Frame(main, bg=self._theme.base)
        composer.pack(fill="x", padx=self._column_pad(), pady=(8, 14))
        self._composer_wrap = composer
        # Raise the whole composer (input box, chips, buttons) by half the input
        # box's own height, measured once it has a real size.
        composer.bind("<Configure>", self._on_composer_configure)

        self._confirm_bar = tk.Frame(composer, bg=self._theme.surface_raised)
        self._confirm_label = tk.Label(
            self._confirm_bar, text="", bg=self._theme.surface_raised, fg=self._theme.foreground,
            font=self._type.font("xs"), anchor="w", justify="left", wraplength=_COLUMN_MAX - 200,
        )
        self._confirm_label.pack(side="left", padx=10, pady=6)
        for label, approve in (("Run", True), ("Cancel", False)):
            button = tk.Button(
                self._confirm_bar, text=label,
                command=partial(self._resolve_confirmation, approve),
                bg=self._theme.primary if approve else self._theme.surface,
                fg=self._theme.primary_foreground if approve else self._theme.foreground,
                relief="flat", borderwidth=0, font=self._type.font("xs", weight="bold"), padx=10, pady=2,
                cursor="hand2",
            )
            button.pack(side="right", padx=(0, 6), pady=4)

        self._attachments_row = tk.Frame(composer, bg=self._theme.base)
        self._attachment_chips: list[tk.Widget] = []
        self._attachments_row.pack(side="top", fill="x", pady=(0, 6))

        # The shared chat composer: the wide, rounded panel with the input on top
        # and the toolbars beneath it. The window keeps its own attribute names
        # (``_composer_box``, ``_entry``, ``_send``, ``_mic_button``) pointed at
        # the panel's widgets, so the send handler, the microphone flow, and the
        # tests all keep working unchanged.
        self._composer = ChatComposer(
            composer, theme=self._theme, typography=self._type,
            placeholder=self.COMPOSER_PLACEHOLDER, on_submit=self._on_send,
            outer_bg=self._theme.base, with_model_bar=True,
        )
        self._composer.wrap.pack(side="top", fill="x")
        self._composer_box = self._composer.box
        # The box's own height settles after the outer frame's last configure, so
        # listen to the box too, or the lift would be computed from a height the
        # box no longer has. (The model row now sits *below* the box, so it no
        # longer contributes to the box's own height -- only to the lift's input.)
        self._composer_box.bind("<Configure>", self._on_composer_configure, add="+")
        self._entry = self._composer.entry
        self._send = self._composer.send_button

        self._attach_button = self._composer.attach_button
        assert self._attach_button is not None  # the window builds with_attach=True
        self._attach_button.configure(command=self._on_attach_clicked)
        self._composer.set_tooltip(self._attach_button, "Attach a Word (.docx) or text file")

        # The microphone: one click starts recording (through the local Qwen3-ASR
        # model when installed), a second click stops and transcribes into the
        # composer. Kept beside Send, in the composer, so voice is a first-class
        # input rather than a separate window.
        self._mic_button = self._composer.add_mic(
            self._toggle_mic, "Dictate with the microphone (Qwen3-ASR)"
        )

        # The internet-lookup switch, beside the microphone: one click turns web
        # search on or off for *this* conversation. It starts from the session's
        # real state (off by default) and shows whatever the service reports, so a
        # refused or unavailable lookup is never drawn as "on".
        self._web_toggle = self._composer.add_web_toggle(
            self._on_web_toggled,
            initial=self.service.web_enabled(),
            available=self.service.web_available(),
            tip=self.service.web_status(),
        )
        self._refresh_web_toggle()

        # The two compact selectors of the composer's bottom-left toolbar. They are
        # *views* of the real controls, not a second source of truth: the Code
        # Studio pill opens the Studio (its only real "code mode"), and the Chat
        # Studio pill offers the same answering models the header's selector does,
        # so nothing here invents a model the app does not have.
        self._code_mode_selector = self._composer.add_mode_selector(
            icon="⌘", label="Code Studio",
            options=[("Open the Code Studio…", self._open_code_studio)],
            tip="Code Studio: edit and merge the project's own code",
        )
        self._chat_mode_selector = self._composer.add_mode_selector(
            icon="💬", label=self._source_choice_label(),
            options=[(text, partial(self._switch_mouth, key)) for key, text in MOUTH_CHOICES],
            tip="The answering model for this conversation",
        )

        # The model-switch row *below* the card, outside it: one chip per
        # answering model, in the order asked for, so the mouth is one click from
        # the composer without cluttering the input panel itself. Clicking a chip
        # opens that model's *own* chat window (``_open_model_chat``), with its
        # logo, its exclusive reading voice, and its language model; the header's
        # selector still switches *this* conversation's mouth, so both remain
        # available.
        self._model_bar = self._composer.model_bar()
        for key, name, hint in MODEL_BUTTONS:
            self._model_bar.add(
                key, name=name, hint=hint,
                command=partial(self._open_model_chat, key),
                tip=f"Open {name}'s own chat window ({hint.lower()})",
            )
        # The speech-engine studios are the row's *trailing* group, pinned to the
        # far right, right to left in the order asked for, with a gap before them so
        # they never touch the model buttons. Each opens its own studio window.
        for key, name, hint in VOICE_ENGINE_BUTTONS:
            self._model_bar.add(
                key, name=name, hint=hint, trailing=True,
                command=partial(self._open_voice_studio, key),
                tip=f"Open the {name} studio ({hint.lower()})",
            )
        self._sync_model_buttons()

        self._render_attachment_chips()
        self._composer.focus()

    def _sync_model_buttons(self) -> None:
        """Accent the chip of the mouth that is actually answering.

        The row is a *view* of the live mouth, never a second source of truth: the
        header selector and the sidebar actions update it too, because they all
        end in :meth:`_switch_mouth`.
        """
        bar = getattr(self, "_model_bar", None)
        if bar is not None:
            bar.set_active(self.service.source)

    # ------------------------------------------------------------- behavior

    def _apply_mode_label(self) -> None:
        """Reflect the current model in the header and the composer pill.

        The window has one operating mode -- the **Fayd Al-Ruh Model** -- so the
        header names the project's own model, and the composer's Chat pill shows
        the *answering mouth* in force (the trained model by default, or whichever
        character the selector chose).
        """
        spec = self.service.operating_mode
        self._mode_var.set(spec.label)
        self._mode_label.configure(text=f"{spec.label} · {spec.description}")
        self._mode_menu.configure(fg=self._theme.contrast)
        self._chat_mode_selector.set_label(self._source_choice_label())

    def _source_choice_label(self) -> str:
        return MOUTH_LABELS.get(self.service.source, MOUTH_LABELS[chat_module.MOUTH_SOURCES])

    def _status_text(self) -> str:
        return (
            f"Session  {self.service.session or 'unsaved'}\n"
            f"{self.service.operating_mode.label} · Ctrl+N new · Ctrl+S save"
        )

    def _on_mode_menu(self) -> None:
        """The header's model control: answer with the project's own model again.

        The window has one operating mode -- the **Fayd Al-Ruh Model** -- and the
        model selector on the right switches to the other characters (Jarvis,
        Ultron, Vision, Banner, Natasha, Hakeem, Cloud). Pressing this control
        returns the conversation to the trained model from whichever character is
        answering, so the model is always one click away.
        """
        self._switch_mouth(chat_module.MOUTH_MODEL)

    def _set_mode(self, spec: ModeSpec) -> None:
        previous = self.service.operating_mode
        if spec.key == previous.key:
            return
        self.service.set_operating_mode(spec.key)
        self._apply_mode_label()
        self._render_tools()
        self._add_bubble(
            "note",
            f"Switched to {spec.label}. "
            + (
                "Sandboxed tools are now available for this conversation."
                if spec.allows_tools
                else "Answers stay conversational; tools are off."
            ),
        )
        self._update_status()

    def _on_source_menu(self, label: str) -> None:
        """The header's model selector: swap the answering mouth in this chat."""
        want = next((key for key, text in MOUTH_CHOICES if text == label), None)
        if want is not None:
            self._switch_mouth(want)

    def _source_menu_var_reset(self) -> None:
        """Keep the model selector's shown value in step with the real mouth."""
        self._source_var_menu.set(self._source_choice_label())

    def _switch_mouth(self, want: str) -> None:
        """Swap the mouth the *same* chat answers from, reporting a failure plainly.

        All seven mouths -- sources, the trained model, Ultron, Vision, Banner,
        Cloud, and Natasha -- answer in this conversation; only ``mode.assistant``
        changes, so the transcript, the session, and the controls are untouched. A
        mouth that cannot load leaves the current one in place and shows the real
        next step.
        """
        if want == self.service.source:
            return
        try:
            if want == chat_module.MOUTH_MODEL:
                self.service.use_model()
            elif want == chat_module.MOUTH_ULTRON:
                self.service.use_ultron()
            elif want == chat_module.MOUTH_VISION:
                self.service.use_vision()
            elif want == chat_module.MOUTH_BANNER:
                self.service.use_banner()
            elif want == chat_module.MOUTH_CLOUD:
                self.service.use_cloud()
            elif want == chat_module.MOUTH_NATASHA:
                self.service.use_natasha()
            elif want == chat_module.MOUTH_HAKEEM:
                self.service.use_hakeem()
            else:
                self.service.use_sources()
        except (
            FileNotFoundError,
            OSError,
            ValueError,
            UltronUnavailable,
            VisionUnavailable,
            BannerUnavailable,
            CloudUnavailable,
            CompanionError,
            HakeemError,
        ) as error:
            self._source_var.set(self.service.source)
            self._source_menu_var_reset()
            self._sync_model_buttons()
            self._add_bubble("note", chat_module.switch_failure_message(want, error))
            return
        self._source_var.set(self.service.source)
        self._source_menu_var_reset()
        self._sync_model_buttons()
        self._apply_mode_label()
        self._add_bubble("note", chat_module.mouth_switched_message(want))

    def _on_enter(self, _event: tk.Event) -> str:
        self._on_send()
        return "break"

    def _on_send(self) -> None:
        if self._busy:
            return
        message = self._composer.text().strip()
        if not message:
            return
        self._stop_speaking()
        self._cancel_recording()
        self._composer.clear()
        self._add_bubble("user", message)

        # In Jarvics Mode a destructive plan asks first; everything else sends now.
        preview = self.service.preview_execution(message) if self.service.tools_enabled else None
        if preview is not None and preview.kind == "tool" and preview.needs_confirmation:
            self._ask_confirmation(message, preview)
            return
        self._dispatch(message)

    def _ask_confirmation(self, message: str, preview) -> None:
        """Show the plan and wait for the user before a mutating tool runs."""
        self._pending_message = message
        lines = "\n".join(f"• {step['tool']}: {step['arguments']}" for step in preview.steps)
        self._confirm_label.configure(
            text=f"This will change files in the sandbox:\n{lines}\nRun it?"
        )
        if self._confirm_bar is not None:
            self._confirm_bar.pack(side="top", fill="x", pady=(0, 6), before=self._attachments_row)
        self._scroll_to_bottom()

    def _resolve_confirmation(self, approve: bool) -> None:
        message = self._pending_message
        self._pending_message = None
        if self._confirm_bar is not None:
            self._confirm_bar.pack_forget()
        if not message:
            return
        if approve:
            self._dispatch(message)
        else:
            self._add_bubble("note", "Cancelled — the tool was not run.")

    def _dispatch(self, message: str) -> None:
        self._retry_text = message
        self._show_thinking("Thinking" if not self.service.tools_enabled else "Working")
        self._start_work(lambda: self.service.ask(message))

    def _on_retry(self) -> None:
        """Re-send the last message; the previous answer stays in the transcript."""
        if self._busy or not self._retry_text:
            return
        self._add_bubble("note", f"Retrying: {self._retry_text}")
        self._dispatch(self._retry_text)

    def _on_new_chat(self, _event: tk.Event | None = None) -> str:
        if self._busy:
            return "break"
        self._stop_speaking()
        self._cancel_recording()
        conversation_id = self.service.new_conversation()
        self._hide_thinking()
        self._reset_attachments()
        self._clear_transcript()
        self._add_bubble("note", f"Started a new conversation ({conversation_id}).")
        self._update_status()
        self.refresh_history()
        return "break"

    def _on_save(self, _event: tk.Event | None = None) -> str:
        if self._busy:
            return "break"
        self._add_bubble("note", self.service.save())
        self._update_status()
        self.refresh_history()
        return "break"

    def _on_resume_session(self, item: HistoryItem) -> None:
        if self._busy or item.is_current:
            return
        if self.service.resume(item.session_id):
            self._stop_speaking()
            self._cancel_recording()
            self._reset_attachments()
            self._add_bubble("note", f"Resumed conversation {item.session_id}.")
            self._update_status()
            self.refresh_history()
        else:
            self._add_bubble("note", f"Could not resume conversation {item.session_id}.")
            self.refresh_history()

    # ---------------------------------------------------------- sidebar toggle

    def _is_compact(self) -> bool:
        width = self.root.winfo_width()
        if width <= 1:
            return False  # not realised yet; keep the sidebar visible
        return width < _COMPACT_WIDTH

    def _toggle_sidebar(self) -> None:
        """Show or hide the sidebar (a drawer on a compact window)."""
        if self._sidebar_visible:
            self._sidebar.pack_forget()
            self._sidebar_visible = False
            self._auto_hidden = False
        else:
            self._sidebar.pack(side="left", fill="y", before=self._main_frame)
            self._sidebar_visible = True
            self._auto_hidden = False

    def _on_root_configure(self, event: tk.Event) -> None:
        """Collapse the sidebar to a drawer on a compact window, and restore it."""
        if event.widget is not self.root:
            return
        if self._is_compact() and self._sidebar_visible:
            self._sidebar.pack_forget()
            self._sidebar_visible = False
            self._auto_hidden = True
        elif not self._is_compact() and not self._sidebar_visible and getattr(self, "_auto_hidden", False):
            self._sidebar.pack(side="left", fill="y", before=self._main_frame)
            self._sidebar_visible = True
            self._auto_hidden = False

    # ------------------------------------------------------------ attachments

    def _on_attach_clicked(self) -> None:
        """Open the ＋ menu: attach a file to this chat, or open the Import Center.

        Attaching keeps the existing behaviour (the file's text joins this
        conversation's evidence). The Import Center is the *persistent* path: it
        writes into the knowledge base through the ingestion pipeline, so the
        imported documents are available to every future chat.
        """
        menu = self._build_attach_menu()
        assert self._attach_button is not None
        try:
            menu.tk_popup(
                self._attach_button.winfo_rootx(),
                self._attach_button.winfo_rooty() + self._attach_button.winfo_height(),
            )
        finally:
            menu.grab_release()

    def _build_attach_menu(self) -> tk.Menu:
        """Build the ＋ menu (kept separate so tests can read its entries)."""
        theme = self._theme
        menu = tk.Menu(
            self.root, tearoff=0, bg=theme.surface, fg=theme.foreground,
            activebackground=theme.interactive_active, activeforeground=theme.contrast,
            font=self._type.font("sm"), borderwidth=0,
        )
        menu.add_command(label="Attach a file to this chat…", command=self._attach_one_file)
        menu.add_command(
            label="Attach a scanned PDF (OCR)…", command=lambda: self._attach_one_file(ocr=True)
        )
        menu.add_separator()
        ocr_ready = optional_deps.is_available(optional_deps.FEATURE_OCR)
        menu.add_command(
            label="OCR support is installed" if ocr_ready else "Install OCR support (scanned PDFs)…",
            command=self._on_install_ocr,
            state="disabled" if ocr_ready else "normal",
        )
        menu.add_separator()
        menu.add_command(
            label="Open Knowledge Import Center…", command=self._open_import_center
        )
        menu.add_command(
            label="Import a SQL database (SQLite)…", command=self._open_sql_import
        )
        menu.add_command(
            label="Generate an image from text…", command=self._open_image_gen
        )
        menu.add_command(
            label="Edit an image from text…", command=self._open_image_edit
        )
        menu.add_separator()
        menu.add_command(
            label="Presentation Studio — build a PowerPoint deck…",
            command=self._open_presentation,
        )
        menu.add_separator()
        menu.add_command(
            label="Doc's Studio — read any document (incl. scanned)…",
            command=self._open_docs_studio,
        )
        menu.add_command(
            label="Scene Studio — build, pose, and render a 3D figure…",
            command=self._open_scene_studio,
        )
        menu.add_separator()
        menu.add_command(
            label="Code Studio — edit and merge the project's own code…",
            command=self._open_code_studio,
        )
        return menu

    def _attach_one_file(self, *, ocr: bool = False) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Attach a scanned PDF (OCR)" if ocr else "Attach a document",
            filetypes=[
                ("Documents", "*.txt *.md *.markdown *.csv *.json *.pdf *.docx *.docm"),
                ("Text files", "*.txt *.md *.markdown"),
                ("PDF", "*.pdf"),
                ("Word documents", "*.docx *.docm"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._on_attach(path, ocr=ocr)

    def _open_import_center(self) -> None:
        """Open the Knowledge Import Center as a separate window."""
        from gui.import_dialog import ImportCenterDialog

        # Keep a reference so the window is not garbage-collected while open.
        self._import_dialog = ImportCenterDialog(
            self.root,
            self.service,
            theme=self._theme,
            font=self._font,
            mono=self._mono,
            on_close=lambda: setattr(self, "_import_dialog", None),
        )

    def _open_sql_import(self) -> None:
        """Open the SQL Database Import screen as a separate window."""
        from gui.sql_import_dialog import SqlImportDialog

        self._sql_import_dialog = SqlImportDialog(
            self.root,
            self.service,
            theme=self._theme,
            font=self._font,
            mono=self._mono,
            on_close=lambda: setattr(self, "_sql_import_dialog", None),
        )

    def _open_model_chat(self, mouth: str) -> None:
        """Open (or focus) the dedicated chat window for one answering model.

        The composer's model row opens this: each model gets its own conversation,
        with its logo, its exclusive reading voice, and its language model. One
        window per model is kept and reused, so clicking the same chip twice
        raises the window already open rather than starting a second one.
        """
        dialogs = getattr(self, "_model_chat_dialogs", None)
        if dialogs is None:
            dialogs = {}
            self._model_chat_dialogs = dialogs
        existing = dialogs.get(mouth)
        if existing is not None:
            try:
                existing.window.deiconify()
                existing.window.lift()
                return
            except tk.TclError:  # the window was closed out from under us
                dialogs.pop(mouth, None)

        from gui.model_chat_dialog import ModelChatDialog

        dialogs[mouth] = ModelChatDialog(
            self.root,
            self._model_chat_service(mouth),
            theme=self._theme,
            font=self._font,
            mono=self._mono,
            on_close=partial(self._forget_model_chat, mouth),
        )

    def _model_chat_service(self, mouth: str):
        """The service a dedicated model window drives.

        A real :class:`gui.model_chat.ModelChatService` by default (its own
        session, its own voice, its own model). ``model_chat_service_factory`` may
        be set to build a lighter one, so a test can open the window without a
        full session.
        """
        factory = getattr(self, "model_chat_service_factory", None)
        if factory is not None:
            return factory(mouth)
        from gui.model_chat import ModelChatService

        return ModelChatService(mouth, data_root=data_dir())

    def _forget_model_chat(self, mouth: str) -> None:
        """Drop a closed model window's reference, so it is not held alive."""
        dialogs = getattr(self, "_model_chat_dialogs", None)
        if dialogs is not None:
            dialogs.pop(mouth, None)

    def _open_voice_studio(self, source: str) -> None:
        """Open (or focus) the dedicated studio window for one speech engine.

        The composer's trailing row opens this: each speech engine gets its own
        window where a document is loaded and read aloud, with a voice and a mood
        exclusive to that engine. One window per engine is kept and reused, so
        clicking the same button twice raises the window already open.
        """
        dialogs = getattr(self, "_voice_studio_dialogs", None)
        if dialogs is None:
            dialogs = {}
            self._voice_studio_dialogs = dialogs
        existing = dialogs.get(source)
        if existing is not None:
            try:
                existing.window.deiconify()
                existing.window.lift()
                return
            except tk.TclError:  # the window was closed out from under us
                dialogs.pop(source, None)

        from gui.voice_studio_dialog import VoiceStudioDialog

        dialogs[source] = VoiceStudioDialog(
            self.root,
            self._voice_studio_service(source),
            theme=self._theme,
            font=self._font,
            mono=self._mono,
            on_close=partial(self._forget_voice_studio, source),
        )

    def _voice_studio_service(self, source: str):
        """The service a dedicated speech-engine studio drives.

        A real :class:`gui.voice_studio.VoiceStudioService` by default (its own
        reader, its own voice, its own mood). ``voice_studio_service_factory`` may
        be set to build a lighter one, so a test can open the window without a
        full speech engine.
        """
        factory = getattr(self, "voice_studio_service_factory", None)
        if factory is not None:
            return factory(source)
        from gui.voice_studio import VoiceStudioService

        return VoiceStudioService(source)

    def _forget_voice_studio(self, source: str) -> None:
        """Drop a closed studio window's reference, so it is not held alive."""
        dialogs = getattr(self, "_voice_studio_dialogs", None)
        if dialogs is not None:
            dialogs.pop(source, None)

    def _open_characters(self) -> None:
        """Open (or focus) the Characters management window.

        The sidebar's **Characters** entry opens this: one window that lists the
        seven characters and the project's own model, and shows the chosen one's
        full editable configuration with an APPLY button. One window is kept and
        reused, so clicking the entry twice raises the window already open.
        """
        existing = getattr(self, "_characters_dialog", None)
        if existing is not None:
            try:
                existing.window.deiconify()
                existing.window.lift()
                return
            except tk.TclError:  # the window was closed out from under us
                self._characters_dialog = None

        from gui.character_dialog import CharacterManagerDialog

        self._characters_dialog = CharacterManagerDialog(
            self.root,
            self._character_service(),
            theme=self._theme,
            font=self._font,
            mono=self._mono,
            on_close=self._forget_characters,
        )

    def _character_service(self):
        """The service the Characters window drives.

        A real :class:`gui.character_manager.CharacterManagerService` by default.
        ``character_service_factory`` may be set to build a lighter one, so a test
        can open the window without a full data directory.
        """
        factory = getattr(self, "character_service_factory", None)
        if factory is not None:
            return factory()
        from gui.character_manager import CharacterManagerService

        return CharacterManagerService(data_root=data_dir())

    def _forget_characters(self) -> None:
        """Drop the closed Characters window's reference, so it is not held alive."""
        self._characters_dialog = None

    def _open_image_gen(self) -> None:
        """Open the text-to-image screen as a separate window."""
        from gui.image_gen_dialog import ImageGenDialog

        self._image_gen_dialog = ImageGenDialog(
            self.root,
            self.service.image_gen,
            theme=self._theme,
            font=self._font,
            mono=self._mono,
            on_close=lambda: setattr(self, "_image_gen_dialog", None),
        )

    def _open_image_edit(self) -> None:
        """Open the image-editing screen as a separate window."""
        from gui.image_edit_dialog import ImageEditDialog

        self._image_edit_dialog = ImageEditDialog(
            self.root,
            self.service.image_edit,
            theme=self._theme,
            font=self._font,
            mono=self._mono,
            on_close=lambda: setattr(self, "_image_edit_dialog", None),
        )

    def _open_presentation(self) -> None:
        """Open Presentation Studio: build a PowerPoint deck from a discussion.

        The Studio drives the same ``gui.presentation.PresentationService`` the
        CLI's ``presentation`` command drives, and its discussion asks *this*
        window's own assistant, so there is no second AI and no second store.
        """
        from gui.presentation_dialog import PresentationDialog

        existing = getattr(self, "_presentation_dialog", None)
        if existing is not None:
            existing.window.deiconify()
            existing.window.lift()
            return
        self._presentation_dialog = PresentationDialog(
            self.root,
            self.service.presentation,
            theme=self._theme,
            font=self._font,
            mono=self._mono,
            on_close=lambda: setattr(self, "_presentation_dialog", None),
            on_open_final_editor=self._open_presentation_finalize,
            model_status=self.service.presentation_model_status,
            ensure_author=self.service.presentation_author,
            on_open_course=self._open_presentation_course,
        )

    def _open_presentation_course(self, course_key: str) -> None:
        """Open the Course window for the deck's curriculum course.

        Reached from the Studio once the deck's own content matches a course
        strongly enough. It is a sibling window that drives the *same*
        ``PresentationService``, so the coverage it shows is the deck's real one.
        The window can hand concepts back to the Studio and jump to a slide; it
        never edits the deck itself. When a Course window is already open it is
        rebuilt for the new course rather than duplicated.
        """
        from gui.presentation_course_dialog import PresentationCourseDialog

        dialog = getattr(self, "_presentation_course_dialog", None)
        if dialog is not None:
            dialog.window.destroy()
        self._presentation_course_dialog = PresentationCourseDialog(
            self.root,
            self.service.presentation,
            project_id=self._presentation_project_id(),
            course_key=course_key,
            theme=self._theme,
            font=self._font,
            mono=self._mono,
            on_close=lambda: setattr(self, "_presentation_course_dialog", None),
            on_use_concepts=self._use_presentation_course_concepts,
            on_open_slide=self._reveal_presentation_slide,
        )

    def _use_presentation_course_concepts(self, text: str) -> None:
        """Copy curriculum concepts into the Studio's own question box."""
        dialog = getattr(self, "_presentation_dialog", None)
        use = getattr(dialog, "use_course_concepts", None)
        if callable(use):
            use(text)

    def _reveal_presentation_slide(self, number: int) -> None:
        """Show the slide that covers a concept, in the Studio behind."""
        dialog = getattr(self, "_presentation_dialog", None)
        reveal = getattr(dialog, "reveal_slide_number", None)
        if callable(reveal):
            reveal(number)

    def _open_presentation_finalize(self) -> None:
        """Open the final-editing window for the Studio's approved deck.

        Reached from the Studio's **Approve the content** step. It is a sibling
        window that drives the *same* ``PresentationService``, so the two views
        never disagree. When a final editor is already open for the same project
        it is raised rather than duplicated.
        """
        from gui.presentation_finalize_dialog import PresentationFinalizeDialog

        dialog = getattr(self, "_presentation_finalize_dialog", None)
        project_id = self._presentation_project_id()
        if dialog is not None and getattr(dialog, "_project_id", "") == project_id:
            dialog.window.deiconify()
            dialog.window.lift()
            return
        if dialog is not None:
            dialog.window.destroy()
        self._presentation_finalize_dialog = PresentationFinalizeDialog(
            self.root,
            self.service.presentation,
            project_id=project_id,
            theme=self._theme,
            font=self._font,
            mono=self._mono,
            on_close=lambda: setattr(self, "_presentation_finalize_dialog", None),
            on_edit=self._refresh_presentation_dialog,
        )

    def _presentation_project_id(self) -> str:
        """The project the Studio currently has open (the final editor's target)."""
        dialog = getattr(self, "_presentation_dialog", None)
        return str(getattr(dialog, "_project_id", "") or "")

    def _refresh_presentation_dialog(self) -> None:
        """Refresh the Studio behind the final editor after an edit there."""
        dialog = getattr(self, "_presentation_dialog", None)
        refresh = getattr(dialog, "refresh", None)
        if callable(refresh):
            refresh(keep_note=True)

    def _open_docs_studio(self) -> None:
        """Open Doc's Studio: read any document, then ask about it.

        The Studio reads with its *own* local vision model (a Qwen3-VL GGUF loaded
        in-process), so a scanned PDF or an image can be read as well as a text
        file. It also borrows *this* window's assistant for the grounded path, so
        there is no second AI and no second knowledge base.
        """
        from gui.docs_dialog import DocsDialog

        existing = getattr(self, "_docs_studio_dialog", None)
        if existing is not None:
            existing.window.deiconify()
            existing.window.lift()
            return
        service = self.service.docs
        service.set_ask(self.service.docs_assistant_answer)
        self._docs_studio_dialog = DocsDialog(
            self.root,
            service,
            theme=self._theme,
            font=self._font,
            mono=self._mono,
            on_close=lambda: setattr(self, "_docs_studio_dialog", None),
        )

    def _open_scene_studio(self) -> None:
        """Open the Scene Studio: build, pose, and render a 3D figure locally.

        The Studio renders with the project's own CPU software rasteriser and
        builds its figures and props in code, so it needs no GPU, no model
        download, and no network. It drives the *same* service the CLI's
        ``scene3d`` command drives, so both see one scene.
        """
        from gui.scene3d_dialog import SceneStudioDialog

        existing = getattr(self, "_scene_studio_dialog", None)
        if existing is not None:
            existing.window.deiconify()
            existing.window.lift()
            return
        self._scene_studio_dialog = SceneStudioDialog(
            self.root,
            self.service.scene3d,
            theme=self._theme,
            font=self._font,
            mono=self._mono,
            on_close=lambda: setattr(self, "_scene_studio_dialog", None),
        )

    def _open_code_studio(self) -> None:
        """Open the Code Studio: edit, test, and merge the project's own code.

        The Studio works in isolated git worktrees, so nothing it changes reaches
        the running project until the user presses Merge there.
        """
        from gui.code_studio import CodeStudioService
        from gui.code_studio_dialog import CodeStudioDialog

        existing = getattr(self, "_code_studio_dialog", None)
        if existing is not None:
            existing.window.deiconify()
            existing.window.lift()
            return
        from core.paths import executable_dir

        service = CodeStudioService(executable_dir())
        self._code_studio_dialog = CodeStudioDialog(
            self.root,
            service,
            theme=self._theme,
            font=self._font,
            mono=self._mono,
            on_close=lambda: setattr(self, "_code_studio_dialog", None),
        )

    def _select_ultron_mouth(self) -> None:
        """Switch this chat to the Ultron mouth from the sidebar action.

        Ultron answers in the *same* conversation -- no separate window -- so this
        just drives the header's model selector. When the optional runtime or a
        model file is missing, the honest next step is shown in the transcript
        instead of opening a mouth that could not answer.
        """
        self._switch_mouth(chat_module.MOUTH_ULTRON)

    def _select_vision_mouth(self) -> None:
        """Switch this chat to the Vision mouth from the sidebar action.

        Vision answers in the *same* conversation -- no separate window -- so this
        just drives the header's model selector. When the optional runtime or a
        model file is missing, the honest next step is shown in the transcript
        instead of opening a mouth that could not answer.
        """
        self._switch_mouth(chat_module.MOUTH_VISION)

    def _select_banner_mouth(self) -> None:
        """Switch this chat to the Banner mouth from the sidebar action.

        Banner answers in the *same* conversation -- no separate window -- so this
        just drives the header's model selector. When the optional runtime or a
        model file is missing, the honest next step is shown in the transcript
        instead of opening a mouth that could not answer.
        """
        self._switch_mouth(chat_module.MOUTH_BANNER)

    def _select_cloud_mouth(self) -> None:
        """Switch this chat to the Cloud mouth from the sidebar action.

        Cloud answers in the *same* conversation -- no separate window -- so this
        just drives the header's model selector. When no API key is set or the
        endpoint refuses, the honest next step is shown in the transcript instead
        of opening a mouth that could not answer. Selecting this mouth is the
        user's explicit approval to send their words to the remote model.
        """
        self._switch_mouth(chat_module.MOUTH_CLOUD)

    def _on_install_ocr(self) -> None:
        """Install OCR support from within the window.

        OCR has two halves: the Python packages (pip installs them) and the
        Tesseract *engine* (an OS package pip cannot install). This installs
        whichever half is missing and reports the outcome, then re-checks the
        feature so the ＋ menu label reflects the new state.
        """
        if self._busy:
            return
        if optional_deps.is_available(optional_deps.FEATURE_OCR):
            self._add_bubble("note", "OCR support is already installed.")
            return
        if optional_deps.missing_imports(optional_deps.FEATURE_OCR):
            if optional_deps.install_command(optional_deps.FEATURE_OCR) is None:
                self._add_bubble("note", optional_deps.install_hint(optional_deps.FEATURE_OCR))
                return
            self._add_bubble("note", "Installing OCR support… this can take a moment.")
        else:
            # The packages are present; the engine is what is missing.
            from core.pdf_ocr import DEFAULT_OCR_LANGUAGES, ocr_install_plan

            plan = ocr_install_plan(DEFAULT_OCR_LANGUAGES)
            if plan.can_install:
                self._add_bubble(
                    "note",
                    "Installing the Tesseract OCR engine and its "
                    f"{'/'.join(DEFAULT_OCR_LANGUAGES)} language data: "
                    f"{' '.join(plan.command)}",
                )
        self._start_work(self._install_ocr_worker)

    def _install_ocr_worker(self) -> Any:
        """Install the OCR Python packages, then the Tesseract engine + language data.

        Returns an :class:`~core.optional_deps.InstallResult` when pip ran (its
        message already names the engine step when the engine is still missing),
        or an :class:`~core.pdf_ocr.OcrInstallResult` when only the engine was
        left to install. The engine install always fetches English *and* Arabic
        language data, so a scanned Arabic page is readable without a second step.
        """
        from core.pdf_ocr import DEFAULT_OCR_LANGUAGES, install_ocr_engine

        if optional_deps.missing_imports(optional_deps.FEATURE_OCR):
            result = optional_deps.install(optional_deps.FEATURE_OCR)
            if not result.ok:
                return result
        if optional_deps.status(optional_deps.FEATURE_OCR).system_ok:
            return optional_deps.InstallResult(
                key=optional_deps.FEATURE_OCR, ok=True, message="OCR support is installed."
            )
        return install_ocr_engine(languages=DEFAULT_OCR_LANGUAGES)

    def _render_install_result(self, result: optional_deps.InstallResult) -> None:
        self._add_bubble("note", result.message)
        if result.ok:
            self._add_bubble(
                "note",
                "You can now attach a scanned PDF with “Attach a scanned PDF (OCR)…”.",
            )

    def _render_ocr_install(self, result: pdf_ocr.OcrInstallResult) -> None:
        """Show the outcome of installing the Tesseract engine (a system package)."""
        self._add_bubble("note", result.message)
        if result.ok:
            self._add_bubble(
                "note",
                "You can now attach a scanned PDF with “Attach a scanned PDF (OCR)…”.",
            )

    def _render_speech_install(self, result: speech.SpeechInstallResult) -> None:
        """Show the outcome of installing a speech engine (a system package)."""
        self._add_bubble("note", result.message)
        if result.ok:
            self.service.refresh_reading()
            self._refresh_voice_menus()
            self._add_bubble("note", "You can now use “Read aloud” with the chosen voice.")

    def _on_attach(self, path: str, *, ocr: bool = False) -> AttachmentNotice:
        notice = self.service.attach_file(path, ocr=ocr)
        if not notice.ok:
            self._add_bubble("note", f"Could not attach {notice.filename}: {notice.message}")
            return notice
        summary = notice.message
        if notice.preview:
            summary = f"{summary}\n\nPreview:\n{notice.preview}"
        self._add_bubble("note", summary)
        self._render_attachment_chips()
        return notice

    def _render_attachment_chips(self) -> None:
        if not hasattr(self, "_attachments_row"):
            return
        for widget in self._attachment_chips:
            widget.destroy()
        self._attachment_chips = []
        attachments = self.service.attachments
        self._attachments_row.pack_configure(pady=(0, 6) if attachments else 0)
        for attachment in attachments:
            self._attachment_chips.append(self._attachment_chip(attachment))

    def _attachment_chip(self, attachment: Attachment) -> tk.Widget:
        chip = tk.Frame(self._attachments_row, bg=self._theme.surface_raised)
        chip.pack(side="left", padx=(0, 6), pady=2)
        tk.Label(
            chip, text=f"📄 {attachment.filename}", bg=self._theme.surface_raised,
            fg=self._theme.foreground, font=self._type.font("xs"), padx=8, pady=3,
        ).pack(side="left")
        close = tk.Label(
            chip, text="✕", bg=self._theme.surface_raised, fg=self._theme.text_dim,
            font=self._type.font("xs", weight="bold"), padx=8, pady=3, cursor="hand2",
        )
        close.pack(side="left")
        close.bind("<Enter>", lambda _e: close.configure(fg=self._theme.primary))
        close.bind("<Leave>", lambda _e: close.configure(fg=self._theme.text_dim))
        close.bind("<Button-1>", lambda _e: self._detach(attachment))
        return chip

    def _detach(self, attachment: Attachment) -> None:
        removed = self.service.remove_attachment(attachment)
        self._render_attachment_chips()
        self._add_bubble(
            "note",
            f"Detached {attachment.filename}."
            if removed
            else f"Detached {attachment.filename} (its text was already gone).",
        )

    def _reset_attachments(self) -> None:
        self.service.clear_attachments()
        self._render_attachment_chips()

    def _update_status(self) -> None:
        self._status.configure(text=self._status_text())

    # ---------------------------------------------------------------- actions

    def _on_action(self, action: GuiAction) -> None:
        if self._busy:
            return
        # The Import Center has a real window, so the sidebar opens it instead of
        # running the headless prompt (which has no terminal to read from here).
        if action.key == "import-center":
            self._open_import_center()
            return
        # The Characters entry opens the character management window.
        if action.key in {"characters", "characters-shareable"}:
            self._open_characters()
            return
        # The text-to-image screen likewise has its own window.
        if action.key == "image":
            self._open_image_gen()
            return
        # The Code Studio has its own window (and its own service), so the sidebar
        # opens it rather than running the headless status command.
        if action.key == "code-studio":
            self._open_code_studio()
            return
        # Presentation Studio likewise has its own window and service.
        if action.key == "presentation":
            self._open_presentation()
            return
        # The Scene Studio has its own window, so the sidebar opens it rather than
        # running the headless status command.
        if action.key == "scene-studio":
            self._open_scene_studio()
            return
        # Ultron and Vision answer in this same chat (the header's model selector),
        # so the sidebar entry switches the mouth rather than opening a window.
        if action.key == "ultron":
            self._select_ultron_mouth()
            return
        if action.key == "vision":
            self._select_vision_mouth()
            return
        if action.key == "banner":
            self._select_banner_mouth()
            return
        if action.key == "cloud":
            self._select_cloud_mouth()
            return
        argv = action.argv
        if action.needs_input:
            question = self._prompt_for_question(action.label)
            if not question:
                return
            argv = (*argv, question)
        self._add_bubble("note", f"Running: {action.label} …")
        self._streaming_label = self._add_bubble("note", f"{action.label} — live output:")
        self._start_work(lambda: self._run_action(action, argv))

    def _prompt_for_question(self, title: str) -> str:
        return (simpledialog.askstring(title, "Question:", parent=self.root) or "").strip()

    def _run_action(self, action: GuiAction, argv: tuple[str, ...]) -> str:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(LineStream(self._lines)):
            action.command(argv)
        output = buffer.getvalue().strip()
        return output or f"{action.label} finished."

    def _append_progress_line(self, text: str) -> None:
        self._pending.append(text)

    def _flush_progress(self, *, force: bool = False) -> None:
        if not self._pending:
            return
        now = time.monotonic()
        if not force and now - self._last_render < 0.15:
            return
        text = "\n".join(self._pending)
        self._pending.clear()
        self._last_render = now
        if self._streaming_label is None:
            self._add_bubble("note", text)
        else:
            self._streaming_label.insert("end", f"\n{text}")
            self._fit_height(self._streaming_label)
        self._scroll_to_bottom()

    # ------------------------------------------------------- thinking indicator

    def _build_thinking(self) -> None:
        row = tk.Frame(self._transcript, bg=self._theme.base)
        self._thinking_row = row
        self._thinking = tk.Label(
            row, text="Thinking·", bg=self._theme.base, fg=self._theme.text_secondary,
            font=self._type.font("sm", slant="italic"),
        )
        self._thinking.pack(side="left")
        self._thinking_base = "Thinking"
        self._thinking_phase = 1

    def _show_thinking(self, label: str) -> None:
        if self._thinking is None or self._thinking_row is None:
            self._build_thinking()
        self._thinking_base = label
        self._thinking_phase = 1
        if self._thinking is not None:
            self._thinking.configure(text=f"{label}·")
        if self._thinking_row is not None:
            self._thinking_row.pack(fill="x", padx=(self._column_pad(), self._column_pad()), pady=6)
        self._hide_empty()
        self._scroll_to_bottom()
        self.root.after(380, self._animate_thinking)

    def _animate_thinking(self) -> None:
        if self._thinking is None or self._thinking_row is None:
            return
        if not self._thinking_row.winfo_ismapped():
            return
        self._thinking_phase = self._thinking_phase % 3 + 1
        self._thinking.configure(text=f"{self._thinking_base}{'·' * self._thinking_phase}")
        self.root.after(380, self._animate_thinking)

    def _hide_thinking(self) -> None:
        if self._thinking_row is not None:
            self._thinking_row.pack_forget()

    # -------------------------------------------------------------- threading

    def _start_work(self, job) -> None:
        self._busy = True
        self._set_send_enabled(False)
        self._status.configure(text="Working…\nThis can take a moment.")

        def run() -> None:
            try:
                self._results.put(("ok", job()))
            except Exception as error:  # surfaced in the transcript, never crashes Tk
                self._results.put(("error", error))

        threading.Thread(target=run, daemon=True).start()

    def _set_send_enabled(self, enabled: bool) -> None:
        """Open or close the composer's own Send gate (a busy screen closes it)."""
        self._composer.set_send_enabled(enabled)

    def _pump(self) -> None:
        self._drain_lines(cap=500)
        self._flush_progress()
        self._drain_speech()
        try:
            kind, payload = self._results.get_nowait()
        except queue.Empty:
            kind = None
        if kind is not None:
            self._drain_lines(cap=None)
            self._flush_progress(force=True)
            self._hide_thinking()
            if kind == "ok":
                self._render_result(payload)
            else:
                self._add_bubble("assistant", f"Something went wrong: {payload}")
            self._busy = False
            self._streaming_label = None
            self._set_send_enabled(True)
            self._update_status()
            self.refresh_history()
        self.root.after(100, self._pump)

    def _drain_speech(self) -> None:
        """Hand a finished speech job (render/transcribe) back on the Tk thread."""
        try:
            kind, payload = self._speech_queue.get_nowait()
        except queue.Empty:
            return
        if kind != "ok":
            self._speaking = False
            self._reset_speak_control()
            self._add_bubble("assistant", f"Speech failed: {payload}")
            self._update_status()
            return
        if isinstance(payload, _SpokenClip):
            self._play_rendered(payload)
        elif isinstance(payload, _Transcript):
            self._show_transcript(payload.text)

    def _show_transcript(self, text: str) -> None:
        """Place a microphone transcript in the composer, ready to send or edit."""
        if not text:
            self._add_bubble("note", "I could not make out any words — please try again.")
            self._update_status()
            return
        self._composer.set_text(text)
        self._composer.focus()
        self._status.configure(text="Transcribed with " + self.service.recognizer_name() + ".")
        self._add_bubble("note", f"🎤 Heard: {text}")

    def _drain_lines(self, *, cap: int | None) -> None:
        drained = 0
        while cap is None or drained < cap:
            try:
                self._append_progress_line(self._lines.get_nowait())
                drained += 1
            except queue.Empty:
                break

    def _render_result(self, payload: Any) -> None:
        if isinstance(payload, optional_deps.InstallResult):
            self._render_install_result(payload)
        elif isinstance(payload, pdf_ocr.OcrInstallResult):
            self._render_ocr_install(payload)
        elif isinstance(payload, speech.SpeechInstallResult):
            self._render_speech_install(payload)
        elif isinstance(payload, ChatReply):
            if payload.steps:
                self._add_bubble("assistant", self._render_steps(payload))
            else:
                self._add_bubble("assistant", payload.text)
            if payload.hint:
                self._add_bubble("note", payload.hint)
            if self._auto_speak.get() and payload.text.strip():
                self._speak_text(payload.text)
        else:
            self._add_bubble("assistant", str(payload))
        self._composer.focus()

    @staticmethod
    def _render_steps(payload: ChatReply) -> str:
        """Render a tool run's trace so results and errors are both visible."""
        lines: list[str] = []
        for step in payload.steps:
            status = "ok" if step.get("ok") else "error"
            detail = step.get("output") if step.get("ok") else step.get("error")
            lines.append(f"- **{step.get('tool', '?')}** [{status}]: {detail}")
        return "\n".join(lines) if lines else payload.text


def main(argv: list[str] | None = None) -> int:
    """Launch the GUI. Returns a non-zero code when no display is available."""
    args = build_parser().parse_args(argv)
    try:
        root = tk.Tk()
    except tk.TclError as error:
        print(f"The graphical interface needs a display: {error}")
        print("Use the command line instead: python -m cli chat")
        return 1

    service = ChatService(
        no_web=args.no_web,
        web=True if getattr(args, "web", False) else None,
        root=Path(args.root) if args.root else None,
        read_write=args.read_write,
        model_checkpoint=args.model_checkpoint,
        ultron_model=args.ultron_model,
        vision_model=args.vision_model,
        banner_model=args.banner_model,
        cloud_provider=args.cloud_provider,
        cloud_model=args.cloud_model,
        session=args.session,
        operating_mode=args.mode,
        tts_engine=args.tts,
        asr_engine=args.asr,
    )
    if args.mind:
        try:
            service.use_model()
        except (FileNotFoundError, OSError, ValueError) as error:
            print(f"Could not load the trained model: {error}")
            return 1
    if args.ultron:
        try:
            service.use_ultron()
        except (UltronUnavailable, OSError, ValueError) as error:
            print(f"Could not load Ultron: {error}")
            return 1
    if args.vision:
        try:
            service.use_vision()
        except (VisionUnavailable, OSError, ValueError) as error:
            print(f"Could not load Vision: {error}")
            return 1
    if args.banner:
        try:
            service.use_banner()
        except (BannerUnavailable, OSError, ValueError) as error:
            print(f"Could not load Banner: {error}")
            return 1
    if args.cloud:
        try:
            service.use_cloud()
        except (CloudUnavailable, OSError, ValueError) as error:
            print(f"Could not reach the cloud model: {error}")
            return 1

    FaydGUI(root, service=service)
    root.mainloop()
    return 0


# ``RoundedBox``/``CircleButton``/``RADIUS_LARGE``/``SEND_BUTTON_SIZE`` are
# re-exported for the dialogs and the tests, which build on the window's own
# widget vocabulary; they are defined once in :mod:`gui.chat_composer`.
__all__ = [
    "RADIUS_LARGE",
    "RADIUS_SMALL",
    "SEND_BUTTON_SIZE",
    "CircleButton",
    "FaydGUI",
    "RoundedBox",
    "main",
]
