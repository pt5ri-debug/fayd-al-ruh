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

from core import language_models as catalog
from core import mouth_profile as profiles
from core.paths import data_dir
from core.reading import TextReader
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

from gui.service import ChatReply, ChatService

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
        self._recording: Any = None
        # Apply the saved model choice before the mouth answers, so the first
        # question already uses the model the user picked. Guarded so a caller's
        # lightweight stand-in service (a test's) still works.
        if self.profile.model and hasattr(self.service, "set_mouth_model"):
            self.service.set_mouth_model(mouth, self.profile.model)

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

    def save_speech(self, text: str, path: str | Path) -> Path:
        """Render ``text`` to a WAV file with this mouth's voice."""
        return self._reader.save(text, path)

    def _apply_voice(self, key: str) -> None:
        """Apply a saved voice, ignoring one that is no longer installed."""
        try:
            self._reader.set_voice(key)
        except ValueError:
            self.profile = profiles.save_profile(self.profile.with_voice(""))

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

        The mouth is rebuilt at once when it is the one answering, so the next
        question uses the chosen model. A model that is not offered here is still
        accepted (the user may know a file the catalog does not name yet), but a
        *known* one that is not downloaded is reported by :meth:`model_ready`.
        """
        setter = getattr(self.service, "set_mouth_model", None)
        if setter is not None:
            setter(self.mouth, key)
        self.profile = profiles.set_model(self.mouth, key)
        return key

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
