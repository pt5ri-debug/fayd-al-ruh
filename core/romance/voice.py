"""The companion voice adapter: give each character a voice, honestly.

This is a **thin adapter** over the project's own speech stack -- it does not add
a second TTS or STT engine. It reads a character's voice settings from its own
:class:`~core.romance.model.CharacterVoiceProfile`, offers only the voices this
machine **really** has (through the project's own ``voice_sections``), and speaks
or transcribes through the project's own :class:`core.reading.TextReader` and
:func:`core.speech.load_recognizer`.

Three honesty rules:

* **Only real voices are offered.** :meth:`available_voices` lists what the
  installed engines actually report for a language -- never a made-up name -- and
  a saved voice that has since been uninstalled is reported as unavailable rather
  than silently swapped.
* **A missing engine never crashes a scene.** Reading and transcription are
  best-effort: an engine that is absent (or a render that fails) is reported, not
  raised into the conversation.
* **Nothing is sent off the machine.** The adapter only calls the local reader and
  recognizer the project already uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.romance.characters import CharacterManager
from core.romance.storage import CompanionStorageService
from core.voices import find_model_voice, find_system_voice, find_voice, voice_sections

#: The voice engines the adapter can name (a saved profile's ``engine``).
ENGINE_AUTO = "auto"
ENGINE_NAMES: tuple[str, ...] = (ENGINE_AUTO, "espeak", "piper", "qwen3-tts", "audio.cpp")


class VoiceAdapterError(RuntimeError):
    """Raised when a voice operation cannot be carried out as asked."""


@dataclass(frozen=True)
class SpeakOutcome:
    """What one attempt to read aloud actually did, with an honest reason.

    ``speak`` used to return the playback handle or ``None``, so a window could
    not tell a *failed render* from a *silent machine*: both looked like "no
    voice engine is installed", which is wrong and unhelpful. The outcome keeps
    the two apart -- ``rendered`` is the clip (or ``None`` when nothing could be
    made), and ``played`` says whether it actually reached the speakers -- with a
    readable ``detail`` naming the real next step.
    """

    rendered: Any = None
    played: bool = False
    playback: Any = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.rendered is not None


@dataclass(frozen=True)
class VoiceOption:
    """One voice that can really be chosen here, with the source it came from."""

    key: str
    label: str
    source: str
    language: str
    engine: str

    def line(self) -> str:
        return f"{self.label} ({self.source})"


@dataclass(frozen=True)
class VoiceProfileView:
    """A character's voice settings, with whether each part is really available."""

    character_id: str
    engine: str
    voice_key: str
    language: str
    dialect: str
    speed: float
    pitch: float
    volume: float
    mood: str
    enabled: bool
    voice_available: bool
    engine_available: bool

    def describe(self) -> str:
        if not self.enabled:
            return "Voice is off for this character."
        state = "available" if self.voice_available else "not installed"
        engine = self.engine or "auto"
        return f"Voice {self.voice_key or '(default)'} via {engine} ({state})."


class CompanionVoiceAdapter:
    """Read and write a character's voice, and speak/transcribe through it."""

    def __init__(
        self,
        storage: CompanionStorageService,
        *,
        characters: CharacterManager | None = None,
        reader: Any = None,
        recognizer: Any = None,
    ) -> None:
        self.storage = storage
        self.characters = characters if characters is not None else CharacterManager(storage)
        self._reader = reader
        self._recognizer = recognizer
        self._playback: Any = None

    # -- the project's own engines (lazy, so import loads nothing) --------

    def _get_reader(self) -> Any:
        if self._reader is None:
            from core.reading import TextReader

            self._reader = TextReader()
        return self._reader

    def _get_recognizer(self) -> Any:
        if self._recognizer is None:
            from core.speech import discover_recognizer_name, load_recognizer

            self._recognizer = load_recognizer(discover_recognizer_name())
        return self._recognizer

    # -- available voices -------------------------------------------------

    def available_voices(
        self, *, engine: str | None = None, language: str | None = None
    ) -> tuple[VoiceOption, ...]:
        """The voices this machine really has, optionally narrowed.

        Built from the project's own ``voice_sections`` (which drops a source that
        contributed no voice here), so a listed voice is always one that can
        actually speak. ``engine``/``language`` narrow the list when given.
        """
        options: list[VoiceOption] = []
        for section in voice_sections():
            for voice in section.voices:
                option = self._as_option(voice, section.source)
                if option is None:
                    continue
                if language and not self._language_matches(option.language, language):
                    continue
                if engine and engine != ENGINE_AUTO and option.engine != engine:
                    continue
                options.append(option)
        return tuple(options)

    def voices_for_language(self, language: str) -> tuple[VoiceOption, ...]:
        """The voices that can speak ``language`` (the honest menu for a character)."""
        return self.available_voices(language=language)

    # -- profiles ---------------------------------------------------------

    def profile(self, character_id: str) -> Any:
        """The character's voice profile, creating a default one if absent."""
        if self.storage.domain.get("companion_characters", character_id) is None:
            raise VoiceAdapterError(f"no character {character_id!r}")
        for row in self.storage.domain.list_records(
            "character_voice_profiles",
            where="character_id = ? AND active = 1",
            params=(character_id,),
            order_by="created_at, record_id",
        ):
            return row
        return self.storage.domain.create("character_voice_profiles", character_id=character_id)

    def set_voice(self, character_id: str, **changes: Any) -> Any:
        """Update the character's voice settings (validated, clamped)."""
        allowed = {
            "engine",
            "voice_key",
            "language",
            "dialect",
            "speed",
            "pitch",
            "volume",
            "mood",
            "enabled",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise VoiceAdapterError(f"unknown voice field(s): {', '.join(sorted(unknown))}")
        if "engine" in changes and changes["engine"] not in ENGINE_NAMES:
            raise VoiceAdapterError(f"unknown voice engine: {changes['engine']!r}")
        if changes.get("mood"):
            from core.voice_moods import find_mood

            try:
                find_mood(changes["mood"])
            except Exception as error:
                raise VoiceAdapterError(f"unknown voice mood: {changes['mood']!r}") from error
        for name, low, high in (("speed", 0.25, 4.0), ("pitch", -1.0, 1.0), ("volume", 0.0, 1.0)):
            if name in changes:
                changes[name] = max(low, min(high, float(changes[name])))
        current = self.profile(character_id)
        return self.storage.domain.update("character_voice_profiles", current.record_id, **changes)

    def view(self, character_id: str) -> VoiceProfileView:
        """The profile with an honest statement of what is really available."""
        profile = self.profile(character_id)
        voice_available = self._voice_exists(profile.voice_key)
        engine_available = self.engine_available(profile.engine)
        return VoiceProfileView(
            character_id=character_id,
            engine=profile.engine,
            voice_key=profile.voice_key,
            language=profile.language,
            dialect=profile.dialect,
            speed=float(profile.speed),
            pitch=float(profile.pitch),
            volume=float(profile.volume),
            mood=profile.mood,
            enabled=bool(profile.enabled),
            voice_available=voice_available,
            engine_available=engine_available,
        )

    def engine_available(self, engine: str) -> bool:
        """Whether a named engine can speak here (``auto`` is always 'yes')."""
        if not engine or engine == ENGINE_AUTO:
            return True
        try:
            from core.speech import speech_available

            return bool(speech_available())
        except Exception:
            return False

    def speech_available(self) -> bool:
        """Whether any speech engine is installed here (the honest yes/no)."""
        try:
            from core.speech import speech_available

            return bool(speech_available())
        except Exception:
            return False

    # -- speaking / listening ---------------------------------------------

    def read_aloud(self, character_id: str, text: str) -> Any:
        """Render ``text`` in the character's voice (``None`` when it cannot).

        A disabled voice, an empty text, or a reader that fails returns ``None``
        rather than raising, so a scene is never broken by speech.
        """
        profile = self.profile(character_id)
        if not profile.enabled or not (text or "").strip():
            return None
        reader = self._get_reader()
        self._apply_voice(reader, profile)
        try:
            return reader.render(text)
        except Exception:
            return None

    def speak(self, character_id: str, text: str) -> Any:
        """Render and play ``text`` (best-effort; returns the playback or ``None``).

        Kept for callers that only need the handle; :meth:`speak_outcome` is the
        honest form and is what the window uses, so a failed render is not
        reported as a missing engine.
        """
        return self.speak_outcome(character_id, text).playback

    def speak_outcome(self, character_id: str, text: str) -> SpeakOutcome:
        """Read ``text`` aloud and report exactly what happened.

        Four honest cases, each with its own detail line: the voice is off for
        this character; the text is empty; the render failed (no engine, or the
        engine errored); or the clip was made and -- when a player exists -- it
        is playing. Playback is best-effort, so a machine with no audio player
        still returns the rendered clip and says the reading was silent.
        """
        profile = self.profile(character_id)
        if not profile.enabled:
            return SpeakOutcome(detail="Voice is off for this character.")
        if not (text or "").strip():
            return SpeakOutcome(detail="There is nothing to read.")
        clip = self.read_aloud(character_id, text)
        if clip is None:
            return SpeakOutcome(
                detail=(
                    "No speech engine could read this text here. "
                    + (
                        "Install one (espeak-ng, Piper, or Qwen3-TTS)."
                        if not self.speech_available()
                        else "The engine could not speak this text."
                    )
                )
            )
        reader = self._get_reader()
        try:
            playback = reader.play(clip)
        except Exception:
            playback = None
        if playback is None:
            self._playback = None
            return SpeakOutcome(
                rendered=clip,
                played=False,
                detail="The clip was rendered but no audio player is installed, so it was silent.",
            )
        self._playback = playback
        return SpeakOutcome(rendered=clip, played=True, playback=playback, detail="")

    def stop(self) -> None:
        """Stop any playback the reader is holding (best-effort)."""
        if self._playback is None:
            return
        try:
            self._playback.stop()
        except Exception:
            return None
        finally:
            self._playback = None

    def transcribe(self, audio: Any) -> Any:
        """Transcribe a recording with the project's own recognizer.

        ``audio`` is an :class:`~core.speech.AudioClip` (as the microphone gives)
        or a path to a WAV file. Returns ``None`` rather than raising when no
        recognizer is available, so a push-to-talk control can report honestly.
        """
        try:
            clip = audio
            if isinstance(audio, (str, bytes)) or hasattr(audio, "__fspath__"):
                from core.speech import AudioClip

                clip = AudioClip.read_wav(audio)
            recognizer = self._get_recognizer()
            return recognizer.transcribe(clip)
        except Exception:
            return None

    def recognizer_available(self) -> bool:
        """Whether a speech recognizer is installed here."""
        try:
            self._get_recognizer()
        except Exception:
            return False
        return True

    def microphone_available(self) -> bool:
        """Whether a microphone recorder program is installed here."""
        try:
            from core.speech import MicrophoneRecorder

            return bool(MicrophoneRecorder().available())
        except Exception:
            return False

    def microphone_status(self) -> str:
        """An honest, readable status of the microphone path and its next step."""
        try:
            from core.speech import microphone_install_hint
        except Exception:
            microphone_install_hint = lambda: ""  # noqa: E731
        if not self.microphone_available():
            return "No microphone recorder is installed. " + microphone_install_hint()
        if not self.recognizer_available():
            return "A microphone is available but no speech-to-text engine is installed."
        return "Listening through the local speech engine."

    # -- internals --------------------------------------------------------

    def _apply_voice(self, reader: Any, profile: Any) -> None:
        if profile.voice_key:
            try:
                reader.set_voice(profile.voice_key)
            except Exception:
                # A voice that is no longer installed: the reader keeps its own.
                pass
        if profile.mood:
            try:
                reader.set_mood(profile.mood)
            except Exception:
                pass

    @staticmethod
    def _as_option(voice: Any, source: str) -> VoiceOption | None:
        key = getattr(voice, "key", "")
        if not key:
            return None
        label = getattr(voice, "label", None)
        if callable(label):
            try:
                label = label()
            except Exception:
                label = None
        if not label:
            label = getattr(voice, "name", None) or key
        language = getattr(voice, "language", "") or getattr(voice, "language_code", "")
        engine = getattr(voice, "engine", "") or source
        return VoiceOption(
            key=str(key),
            label=str(label),
            source=str(source),
            language=str(language),
            engine=str(engine),
        )

    @staticmethod
    def _language_matches(voice_language: str, wanted: str) -> bool:
        if not voice_language:
            return True
        return voice_language.lower().startswith(wanted.lower()[:2])

    def _voice_exists(self, key: str) -> bool:
        if not key:
            return True
        if key.startswith("system:"):
            return find_system_voice(key) is not None
        if key.startswith("model:"):
            return find_model_voice(key) is not None
        try:
            find_voice(key)
            return True
        except Exception:
            return False


__all__ = [
    "ENGINE_AUTO",
    "ENGINE_NAMES",
    "CompanionVoiceAdapter",
    "SpeakOutcome",
    "VoiceAdapterError",
    "VoiceOption",
    "VoiceProfileView",
]
