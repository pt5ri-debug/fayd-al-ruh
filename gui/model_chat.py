"""One dedicated chat session per answering mouth, with its own voice and model.

The main window offers every mouth inside *one* conversation (the header's model
selector and the composer's model row). This module is the other half: a
**separate** session for a single mouth, so a user can talk to Ultron in one
window and to Hakeem in another without the two transcripts sharing a
conversation. It is the engine behind :mod:`gui.model_chat_dialog`.

What belongs to the *mouth* -- and is therefore remembered here, per mouth:

* its **character name** and its **logo** (see :mod:`core.mouth_profile`);
* the **reading voice** its answers are read aloud with, exclusive to it;
* the **language model** it answers with, chosen from the project's full catalog
  (see :mod:`core.language_models`).

The class is Tk-free on purpose, so the whole workflow -- choosing a model,
choosing a voice, asking a question, transcribing a recording -- is exercised
headless with no display, no model, and no microphone. The window is a thin view
over it. A ``service`` may be injected (a test's own), so nothing here needs a
real session to be tested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai import mouths as mouths_module
from companion.companion_agent import CompanionError
from core import custom_models
from core import language_models as catalog
from core import mouth_profile as profiles
from core.banner import BannerUnavailable
from core.best_effort import best_effort
from core.character_engine import apply_character_voice
from core.cloud_model import CloudUnavailable
from core.paths import data_dir
from core.reading import TextReader, batch_chunks, clip_is_silent, split_for_speech
from core.ultron import UltronUnavailable
from core.vision_model import VisionUnavailable
from core.voices import (
    ModelVoice,
    SystemVoice,
    Voice,
    find_model_voice,
    find_system_voice,
    find_voice,
    is_model_voice_key,
    is_system_voice_key,
)
from hakeem.hakeem_agent import HakeemError

from gui.service import ChatReply, ChatService

#: The errors activating a mouth can raise. Each carries the real next step
#: (install the runtime, set a key, download the weights), so a dedicated window
#: that cannot bring its mouth up reports it instead of failing to open.
_ACTIVATION_ERRORS = (
    FileNotFoundError,
    OSError,
    ValueError,
    UltronUnavailable,
    VisionUnavailable,
    BannerUnavailable,
    CloudUnavailable,
    CompanionError,
    HakeemError,
)

#: A short, honest line per mouth explaining what it is, shown in its window.
MOUTH_BLURBS: dict[str, str] = {
    catalog.MOUTH_SOURCES: (
        "Answers from the project's own knowledge, with citations. No language "
        "model is involved, so there is nothing to choose here."
    ),
    catalog.MOUTH_MODEL: (
        "Answers from the small language model trained on this project's own "
        "corpus. Its words are its own; they carry no citations."
    ),
    catalog.MOUTH_ULTRON: (
        "A real, general local model. Ask about anything; every answer runs on "
        "this machine and carries no citations."
    ),
    catalog.MOUTH_VISION: (
        "A real, petroleum-engineering local model (reservoir, drilling, "
        "production, PVT). Runs on this machine; no citations."
    ),
    catalog.MOUTH_BANNER: (
        "The presentation local model: it turns a discussion into an outline and "
        "an outline into slide content. Runs on this machine; no citations."
    ),
    catalog.MOUTH_CLOUD: (
        "A remote model reached over the network. Choosing it is your explicit "
        "approval to send your words off this machine; no citations."
    ),
    catalog.MOUTH_NATASHA: (
        "The companion: a stable personality, a lasting memory, and a changing "
        "tone, through the local model. No citations."
    ),
    catalog.MOUTH_HAKEEM: (
        "The scholar of religion and history: he compares the Abrahamic faiths and "
        "discusses the ancient civilisations, through the local model. No citations."
    ),
}


def mouth_title(mouth: str) -> str:
    """The human name of a mouth, for a window title and a section header."""
    return catalog.ENGINE_LABELS.get(mouth, mouth)


class ModelChatService:
    """One mouth's own conversation, voice, and model choice.

    It wraps a :class:`~gui.service.ChatService` (its own session) and the mouth's
    saved :class:`~core.mouth_profile.MouthProfile`, and keeps a :class:`~core.reading.TextReader`
    of its own so the voice is exclusive to this mouth. Nothing is loaded until a
    question is actually asked, so opening the window is cheap.
    """

    def __init__(
        self,
        mouth: str,
        *,
        service: ChatService | None = None,
        data_root: Path | None = None,
        reader: TextReader | None = None,
    ) -> None:
        if mouth not in catalog.MOUTHS:
            raise ValueError(f"Unknown mouth {mouth!r}.")
        self.mouth = mouth
        self._data_root = data_root if data_root is not None else data_dir()
        self.service = service if service is not None else ChatService(data_root=self._data_root)
        self.profile = profiles.load_profile(mouth)
        # The exclusive reading voice: this mouth's own TextReader, built with the
        # voice saved for it. The shared reader (the main window's) is untouched, so
        # choosing a voice here never changes the main chat's.
        self._reader = reader if reader is not None else TextReader()
        saved_voice = self.profile.voice
        if saved_voice:
            self._apply_voice(saved_voice)
        else:
            # No per-window voice yet: seed it from the character's own Voice
            # section, so a voice chosen in the Characters window is in force here
            # rather than only stored. A voice this machine does not have resolves
            # to nothing and the reader keeps its default, honestly.
            self._apply_character_voice()
        self._recording: Any = None
        # The reading in progress, so a second click can stop it (the toggle the
        # user asked for): the text being read and the clip playing right now.
        self._reading_text: str = ""
        self._playback: Any = None
        # Apply the saved model choice before the mouth answers, so the first
        # question already uses the model the user picked. Guarded so a caller's
        # lightweight stand-in service (a test's) still works.
        if self.profile.model and hasattr(self.service, "set_mouth_model"):
            self.service.set_mouth_model(mouth, self.profile.model)
        # The mouth the window is for must be the one answering *here*, so a
        # question asked in this window is answered by this model rather than by
        # the grounded default. Best-effort: a mouth that cannot load (its runtime
        # or weights are missing) reports the real next step in ``activation_status``
        # instead of failing to open the window.
        self._activation_error: str = ""
        self._activate()

    # ------------------------------------------------------------- activation

    def _activate(self) -> None:
        """Make this window's mouth the one answering its own session.

        The dedicated window owns its own conversation, so making its mouth active
        never touches the main window's mouth. A failure is recorded, not raised:
        the window still opens and reports the real next step, so the user can add
        or download a model from it.
        """
        activate = getattr(self.service, "activate_mouth", None)
        if activate is None:
            return
        try:
            activate(self.mouth)
            self._activation_error = ""
        except _ACTIVATION_ERRORS as error:
            self._activation_error = str(error)
        except Exception as error:  # a stand-in service may raise anything
            self._activation_error = str(error)

    def active(self) -> bool:
        """True when this window's mouth is the one answering its session."""
        check = getattr(self.service, "mouth_is_active", None)
        if check is None:
            return True
        try:
            return bool(check(self.mouth))
        except Exception:
            return True

    def activation_status(self) -> str:
        """An honest line about whether this mouth is answering here, and the next step."""
        if self.active():
            return f"{self.name} is answering here."
        if self._activation_error:
            return f"{self.name} cannot answer yet: {self._activation_error}"
        return f"{self.name} is not answering here yet."

    # --------------------------------------------------------------- identity

    @property
    def name(self) -> str:
        """The character's display name (the user may have renamed it)."""
        return self.profile.name

    def blurb(self) -> str:
        """A short, honest explanation of what this mouth is."""
        return MOUTH_BLURBS.get(self.mouth, "")

    def set_name(self, name: str) -> str:
        """Rename this mouth's character, persist it, and return the new name."""
        self.profile = profiles.set_name(self.mouth, name)
        return self.profile.name

    def logo_path(self) -> Path | None:
        """The saved logo file, or ``None`` when the default badge should be drawn."""
        return self.profile.logo_path()

    def set_logo(self, source: str | Path) -> Path | None:
        """Copy ``source`` in as this mouth's logo and return the stored file."""
        self.profile = profiles.save_logo(self.mouth, source)
        return self.profile.logo_path()

    def clear_logo(self) -> None:
        """Remove this mouth's logo, back to the default badge."""
        self.profile = profiles.clear_logo(self.mouth)

    # ------------------------------------------------------------------ voice

    def voice_key(self) -> str:
        """The voice key this mouth reads with (its own, not the shared one)."""
        return self._reader.voice.key

    @property
    def character(self):
        """This mouth's live character assistant, or ``None`` when it is not loaded.

        Read *without* building it, so a caller (an APPLY hot-applying the
        character's model settings) can reach the loaded mouth without pulling a
        model into memory. Returns the character wrapper when the session was built
        with characters on, and the raw mouth otherwise.
        """
        switcher = getattr(self.service, "switcher", None)
        lookup = getattr(switcher, "assistant_for", None)
        if callable(lookup):
            return lookup(self.mouth)
        return None

    def set_voice(self, key: str) -> str:
        """Choose this mouth's reading voice, persist it, and return its key.

        Raises :class:`ValueError` for an unknown key, so the window reports a bad
        choice instead of silently reading with something else.
        """
        voice = self._reader.set_voice(key)
        self.profile = profiles.set_voice(self.mouth, voice.key)
        return voice.key

    def voice_sections(self):
        """The available voices grouped by the source each came from, for a menu."""
        from core.voices import voice_sections

        return voice_sections()

    def voice_label(self, key: str) -> str:
        """A readable label for a voice key, resolved against what is installed."""
        voice: Voice | SystemVoice | ModelVoice | None = None
        if is_system_voice_key(key):
            voice = find_system_voice(key)
        elif is_model_voice_key(key):
            voice = find_model_voice(key)
        else:
            try:
                voice = find_voice(key)
            except ValueError:
                return key
        if voice is None:
            return key
        return f"{voice.category_label()} “{voice.label()}”"

    def reading_available(self) -> bool:
        """True when a real speech engine is installed here."""
        return self._reader.available()

    def read_aloud(self, text: str):
        """Read ``text`` aloud with this mouth's own voice and return the clip."""
        return self._reader.speak(text)

    # -------------------------------------------------- stoppable reading

    def reading(self) -> bool:
        """True while a reading is in progress (so the control shows ``Stop``)."""
        return self._reading_text != ""

    def start_reading(self, text: str):
        """Read ``text`` aloud, blocking, so a second click can stop it.

        The window runs this on a worker thread: the text is split into
        sentence-sized chunks, grouped into batches, and each batch is rendered in
        **one** engine call -- a local engine (audio.cpp) loads its 1.35 GB model per
        call, so a paragraph-per-call read spent its time loading weights and looked
        frozen. The loop checks after every clip whether :meth:`stop_reading` was
        called, so the reading ends at the next boundary and the sound stops at once.
        Returns the clip that was spoken (the first one) so the caller keeps the old
        return shape.
        """
        body = (text or "").strip()
        if not body:
            raise ValueError("There is nothing to read.")
        chunks = split_for_speech(body)
        self._reading_text = body
        first: Any = None
        try:
            for group in batch_chunks(chunks):
                if not self.reading():
                    break
                clips = self._render_group(group)
                for clip in clips:
                    if not self.reading():
                        break
                    if first is None:
                        first = clip
                    if clip_is_silent(clip):
                        return first
                    self._playback = self._reader.play(clip)
                    if self._playback is not None:
                        self._playback.wait()
                    self._playback = None
            return first
        finally:
            self._reading_text = ""
            self._playback = None

    def _render_group(self, group: tuple[str, ...]) -> list[Any]:
        """Render a batch, loading a local engine's model **once** for the batch.

        ``render_many`` is the batched path; a reader double that only offers
        ``render`` is called chunk by chunk, so the test seam stays small.
        """
        many = getattr(self._reader, "render_many", None)
        if callable(many):
            return list(many(list(group)))
        return [self._reader.render(chunk) for chunk in group]

    def stop_reading(self) -> bool:
        """Ask the reading in progress to stop; ``True`` when one was running."""
        if not self.reading():
            return False
        self._reading_text = ""
        playback = self._playback
        if playback is not None:
            with best_effort("stop the clip that is playing"):
                playback.stop()
        return True

    def save_speech(self, text: str, path: str | Path) -> Path:
        """Render ``text`` to a WAV file with this mouth's voice."""
        return self._reader.save(text, path)

    def _apply_voice(self, key: str) -> None:
        """Apply a saved voice, ignoring one that is no longer installed."""
        try:
            self._reader.set_voice(key)
        except ValueError:
            self.profile = profiles.save_profile(self.profile.with_voice(""))

    def _apply_character_voice(self) -> str:
        """Seed this window's voice from the character's own Voice section.

        Returns the voice key that took effect, or "" when the character asks for
        none or the voice it names is not installed here. The per-window choice
        (``set_voice``) still wins once the user makes one: this only fills the gap
        so a voice, language, and speed/pitch/volume chosen in the Characters window
        are in force.
        """
        try:
            from core.character_registry import default_registry

            config = default_registry().load(self.mouth)
        except (ValueError, OSError):
            return ""
        choice = apply_character_voice(self._reader, config)
        return self._reader.voice.key if choice.ok else ""

    def reload_character_voice(self) -> bool:
        """Re-apply the character's Voice section to this window's reader.

        Called when the Characters window's APPLY changes the voice: returns True
        when this window's reader took the change, so the manager can report the
        voice as activated honestly. A window the user has pointed at a voice of its
        own keeps that choice -- only the character's own voice is re-applied here
        when the window has no per-window override.
        """
        if self.profile.voice:
            return False
        return bool(self._apply_character_voice())

    # ------------------------------------------------------------------ model

    def model_choices(self):
        """Every language model this window can offer, with its real availability."""
        return catalog.models_for_choice(self.mouth)

    def model_key(self) -> str:
        """The chosen model key (``""`` means the mouth's own default)."""
        getter = getattr(self.service, "mouth_model", None)
        return getter(self.mouth) if getter is not None else self.profile.model

    def set_model(self, key: str) -> str:
        """Point this mouth at ``key``, persist it, and return the stored key.

        The mouth is rebuilt at once (through :meth:`service.set_mouth_model`) when
        it is the one answering, and this window re-activates it so the next
        question uses the chosen model and it is the active one here. A model that
        is not offered here is still accepted (the user may know a file the catalog
        does not name yet), but a *known* one that is not downloaded is reported by
        :meth:`model_ready`. The choice is saved first and the load is best-effort,
        so choosing a model whose runtime or weights are missing still remembers it
        and reports the real next step rather than losing the choice.
        """
        self.profile = profiles.set_model(self.mouth, key)
        setter = getattr(self.service, "set_mouth_model", None)
        if setter is not None:
            try:
                setter(self.mouth, key)
            except _ACTIVATION_ERRORS:
                pass  # saved; ``activation_status`` reports why it cannot answer yet
            except Exception:
                pass
        self._activate()
        return key

    def add_model(self, path: str | Path, *, title: str = "") -> str:
        """Add a GGUF file on this machine as a selectable model and use it here.

        The file is remembered in the custom-model registry (see
        :mod:`core.custom_models`) under a ``custom:`` key, offered to this window's
        picker, and immediately made this window's active model, so it is ready to
        chat with at once. Raises :class:`ValueError` for a missing or non-GGUF
        file, so the window reports a bad choice rather than storing it.
        """
        if not self.can_add_model():
            raise ValueError(
                f"{self.name} answers from cited passages and has no language model "
                "to add."
            )
        entry = custom_models.add_model(path, title=title)
        self.set_model(entry.key)
        return entry.key

    def can_add_model(self) -> bool:
        """True when this mouth runs a language model that a file can be added to.

        The retrieval mouth (Jarvis, from the sources) has no language model at
        all, so its window offers no *Add a language model…* action rather than a
        button that would store a choice nothing uses.
        """
        return self.mouth in mouths_module.MODEL_MOUTHS

    def remove_model(self, key: str) -> bool:
        """Forget a model the user added here; ``True`` when one was removed.

        Only a ``custom:`` entry can be removed -- a catalog model is forgotten by
        pointing the window at another one. When the removed model was the active
        choice, the window falls back to the mouth's own default so the next
        question still answers.
        """
        if not custom_models.is_custom_key(key):
            return False
        removed = custom_models.remove_model(key)
        if removed and self.model_key() == key:
            self.set_model("")
        return removed

    def can_remove(self, key: str) -> bool:
        """True when ``key`` is a model the user added here (so it can be forgotten)."""
        return custom_models.is_custom_key(key)

    def can_download(self, key: str | None = None) -> bool:
        """True when the chosen (or named) model has weights this build can fetch."""
        choice = self._choice(key)
        return catalog.can_download(choice)

    def download_command(self, key: str | None = None) -> str:
        """The governed command that fetches the chosen model, or ``""``."""
        choice = self._choice(key)
        return choice.download_command if choice is not None else ""

    def download_model(self, key: str | None = None, *, approved: bool = True) -> Path:
        """Fetch the chosen model's weights through the owning engine's own gate.

        ``approved=True`` is the user's explicit click, exactly like every other
        model download in the project: the fetch still runs through the engine's
        permission and human-approval gates (see :func:`core.language_models.download_model`),
        so nothing is fetched that the user did not ask for. Raises the engine's own
        availability error, or :class:`ValueError` when there is nothing to fetch.
        """
        chosen = key or self.model_key() or catalog.default_language_model_key(self.mouth) or ""
        return catalog.download_model(self.mouth, chosen, approved=approved)

    def _choice(self, key: str | None = None):
        """The catalog entry for ``key`` (or the chosen model), or ``None``."""
        wanted = key if key is not None else self.model_key()
        if not wanted:
            wanted = catalog.default_language_model_key(self.mouth) or ""
        return catalog.resolve_choice(self.mouth, wanted) if wanted else None

    def model_ready(self) -> bool:
        """True when the chosen model (or the mouth's default) could answer now."""
        key = self.model_key()
        if not key:
            default = catalog.default_language_model_key(self.mouth)
            choice = catalog.find_language_model(self.mouth, default)
        else:
            choice = catalog.resolve_choice(self.mouth, key)
        return bool(choice is not None and choice.available)

    def model_status(self) -> str:
        """An honest one-line status of the chosen model and the real next step."""
        key = self.model_key()
        choice = catalog.resolve_choice(self.mouth, key) if key else None
        if choice is None:
            default = catalog.default_language_model_key(self.mouth)
            choice = catalog.find_language_model(self.mouth, default)
        if choice is None:
            return "No selectable model for this mouth."
        return choice.describe()

    def describe(self) -> str:
        """A one-line description of the mouth, its model, and its voice."""
        return f"{self.name}: {self.model_status()} Voice: {self.voice_label(self.voice_key())}."

    # --------------------------------------------------------------- talking

    def ask(self, message: str) -> ChatReply:
        """Send ``message`` to this mouth and return the rendered reply."""
        return self.service.ask(message)

    def new_conversation(self) -> str:
        """Start a fresh conversation for this mouth and return its id."""
        return self.service.new_conversation()

    def save(self) -> str:
        """Persist this mouth's conversation."""
        return self.service.save()

    def session(self) -> str | None:
        return self.service.session

    # ------------------------------------------------- internet lookup toggle

    def web_available(self) -> bool:
        """True when this window's mouth can search the internet.

        Only the grounded *sources* mouth searches; a language-model mouth answers
        from its own weights, so its window honestly has no lookup switch. The
        answer is keyed on *this window's own mouth*, not on whichever mouth the
        wrapped session happens to have active at the moment.
        """
        if self.mouth != catalog.MOUTH_SOURCES:
            return False
        check = getattr(self.service, "web_available", None)
        return bool(check()) if check is not None else False

    def web_enabled(self) -> bool:
        """True when this window will actually search the internet right now."""
        check = getattr(self.service, "web_enabled", None)
        return bool(check()) if check is not None else False

    def set_web(self, enabled: bool) -> bool:
        """Turn internet lookups on or off; returns the state now in force.

        Delegates to the window's own session, so the switch here changes *this*
        window's lookups and persists the choice for the next session. A mouth
        that cannot search refuses to be switched on.
        """
        if enabled and not self.web_available():
            return False
        setter = getattr(self.service, "set_web", None)
        return bool(setter(enabled)) if setter is not None else False

    def web_status(self) -> str:
        """An honest one-line status for the switch's tooltip."""
        status = getattr(self.service, "web_status", None)
        if status is not None:
            return str(status())
        if not self.web_available():
            return "Internet lookups are not available for this model."
        return "Internet lookups: on or off"

    # ----------------------------------------------------------- microphone

    def microphone_available(self) -> bool:
        return self.service.microphone_available()

    def asr_status(self) -> str:
        return self.service.asr_status()

    def recognizer_name(self) -> str:
        return self.service.recognizer_name()

    def start_recording(self):
        """Begin a recording through the shared microphone recorder."""
        self._recording = self.service.start_recording()
        return self._recording

    def stop_recording(self):
        """Stop the recording (if any) and return it."""
        recording = self.service.stop_recording()
        self._recording = None
        return recording

    def transcribe(self, recording: Any) -> str:
        """Finish a recording and turn it into text."""
        return self.service.transcribe(recording.result())


__all__ = ["MOUTH_BLURBS", "ModelChatService", "mouth_title"]
