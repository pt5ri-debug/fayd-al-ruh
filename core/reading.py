"""Read the chat's text aloud: text-to-speech with a chosen voice.

This is the "make the written words audible" layer. It ties three existing pieces
together -- a :class:`~core.voices.Voice` from the catalog, a
:class:`~core.speech.SpeechSynthesizer` (a real system engine when one is
installed), and a :class:`~core.speech.Speaker` that plays the clip -- behind one
small, testable object, :class:`TextReader`.

The project's honesty rule applies here too: if no speech engine is installed the
reader still *runs*, but it says so. :meth:`TextReader.available` is False, the
synthesizer's name is ``'silent'``, and :meth:`TextReader.describe` explains that
no real voice can be produced on this machine rather than pretending otherwise.
"""

from __future__ import annotations

from pathlib import Path

from core.speech import (
    AudioClip,
    Playback,
    SilentSynthesizer,
    Speaker,
    SpeechSynthesizer,
    load_synthesizer,
    refresh_windows_speech_caches,
    speech_install_plan,
    windows_voice_report,
)
from core.voices import SystemVoice, Voice, default_voice, find_system_voice, find_voice, is_system_voice_key


class TextReader:
    """Reads text aloud through the system speech engine, with a chosen voice.

    ``engine`` is a synthesizer name (``'auto'`` picks the best installed system
    engine, else the silent stand-in). ``voice`` is a catalog key or
    :class:`~core.voices.Voice` -- or a ``system:<engine>:<name>`` key or
    :class:`~core.voices.SystemVoice` naming a voice the OS really reports, which
    is preferred when it exists; ``language`` pins the spoken language and
    ``None`` lets the engine decide from the text, so Arabic is read as Arabic.
    """

    def __init__(
        self,
        *,
        engine: str = "auto",
        voice: str | Voice | SystemVoice | None = None,
        language: str | None = None,
        speaker: Speaker | None = None,
        synthesizer: SpeechSynthesizer | None = None,
    ) -> None:
        self.engine = engine
        self.language = language
        self._speaker = speaker
        self._voice = _as_voice(voice)
        self._synthesizer = synthesizer
        if synthesizer is not None:
            self._engine = synthesizer.name

    # ------------------------------------------------------------------ voice

    @property
    def voice(self) -> Voice | SystemVoice:
        """The voice currently used to read."""
        return self._voice

    def set_voice(self, voice: str | Voice | SystemVoice) -> Voice | SystemVoice:
        """Choose a different voice and rebuild the engine for it."""
        self._voice = _as_voice(voice)
        self._synthesizer = None  # rebuilt lazily with the new voice
        return self._voice

    def system_voices(self) -> tuple[SystemVoice, ...]:
        """The *real* voices installed on this machine (empty when none are known).

        The menus prefer these over the synthetic catalog: they are what the
        machine will actually speak with, by their own names.
        """
        from core.speech import system_voices

        return system_voices()

    # ------------------------------------------------------------- synthesizer

    @property
    def synthesizer(self) -> SpeechSynthesizer:
        """The engine that turns text into audio, built once and cached."""
        if self._synthesizer is None:
            self._synthesizer = load_synthesizer(
                self.engine, voice=self._voice, language=self.language
            )
            self._engine = self._synthesizer.name
        return self._synthesizer

    @property
    def engine_name(self) -> str:
        """The resolved engine name (``'silent'`` when nothing real is installed)."""
        return self.synthesizer.name

    def available(self) -> bool:
        """True when a real speech engine is installed (so audio is not silent)."""
        return self.synthesizer.name != SilentSynthesizer.name

    def refresh(self) -> None:
        """Drop the cached engine and speaker, so a newly installed engine is found.

        The synthesizer is built once and cached; after the user installs a speech
        engine -- or adds an Arabic voice in Windows Settings -- mid-session this
        re-discovers it instead of keeping the silent one or an English voice.
        """
        self._synthesizer = None
        self._speaker = None
        refresh_windows_speech_caches()

    @property
    def speaker(self) -> Speaker:
        if self._speaker is None:
            self._speaker = Speaker()
        return self._speaker

    def can_play(self) -> bool:
        """True when the machine can actually play the rendered audio."""
        return self.speaker.available()

    # -------------------------------------------------------------- reading

    def render(self, text: str) -> AudioClip:
        """Synthesize ``text`` and return the clip without playing it."""
        return self.synthesizer.synthesize(text)

    def speak(self, text: str) -> AudioClip:
        """Read ``text`` aloud and return the clip that was spoken.

        The clip is always rendered. Playback is best-effort: when the machine has
        no audio player the clip is returned without playing, so the caller can
        still save it or show its duration rather than failing.
        """
        clip = self.render(text)
        if self.speaker.available():
            self.speaker.play(clip)
        return clip

    def play(self, clip: AudioClip) -> Playback | None:
        """Play an already-rendered ``clip`` in the background.

        Returns a stoppable :class:`~core.speech.Playback` so a caller can end the
        reading (a second click on "Read aloud"), or ``None`` when the machine has
        no audio player -- the clip is then simply not played, matching
        :meth:`speak`'s best-effort rule rather than raising.
        """
        if not self.speaker.available():
            return None
        return self.speaker.start(clip)

    def save(self, text: str, path: str | Path) -> Path:
        """Render ``text`` and write it to a WAV file, returning the path."""
        clip = self.render(text)
        target = Path(path)
        clip.write_wav(target)
        return target

    def describe(self) -> str:
        """An honest, human-readable status of this reader."""
        if not self.available():
            return (
                "No speech engine is installed, so text is read silently. "
                + speech_install_plan().hint
            )
        played = "played through the speakers" if self.can_play() else "saved but not played (no audio player)"
        line = (
            f"Reading with the {self._voice.category_label()} voice “{self._voice.label()}” "
            f"through {self.engine_name}; audio is {played}."
        )
        note = windows_voice_report()
        if note:
            line = f"{line} {note}"
        return line


def _as_voice(value: str | Voice | SystemVoice | None) -> Voice | SystemVoice:
    if value is None:
        return default_voice()
    if isinstance(value, (Voice, SystemVoice)):
        return value
    key = str(value)
    if is_system_voice_key(key):
        resolved = find_system_voice(key)
        if resolved is not None:
            return resolved
        # The voice is no longer installed; fall back to the catalog default so a
        # saved choice never asks the engine for a name it would reject.
        return default_voice()
    return find_voice(key)


__all__ = ["TextReader"]
