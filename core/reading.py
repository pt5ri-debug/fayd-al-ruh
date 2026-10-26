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
from core.voice_moods import Mood, Prosody, resolve_mood
from core.voices import (
    ModelVoice,
    SystemVoice,
    Voice,
    default_voice,
    find_model_voice,
    find_system_voice,
    find_voice,
    is_model_voice_key,
    is_system_voice_key,
)


class TextReader:
    """Reads text aloud through the system speech engine, with a chosen voice.

    ``engine`` is a synthesizer name (``'auto'`` picks the best installed system
    engine, else the silent stand-in). ``voice`` is a catalog key or
    :class:`~core.voices.Voice` -- or a ``system:<engine>:<name>`` key or
    :class:`~core.voices.SystemVoice` naming a voice the OS really reports, which
    is preferred when it exists; ``language`` pins the spoken language and
    ``None`` lets the engine decide from the text, so Arabic is read as Arabic.
    ``mood`` is an optional emotional state (a
    :class:`~core.voice_moods.Mood` or its key) applied to the delivery, so the
    same voice can read happily, sadly, excitedly, and so on.
    """

    def __init__(
        self,
        *,
        engine: str = "auto",
        voice: str | Voice | SystemVoice | ModelVoice | None = None,
        language: str | None = None,
        speaker: Speaker | None = None,
        synthesizer: SpeechSynthesizer | None = None,
        mood: str | Mood | None = None,
    ) -> None:
        self.engine = engine
        self.language = language
        self._speaker = speaker
        self._voice = _as_voice(voice)
        self._synthesizer = synthesizer
        self._mood: Mood | None = _as_mood(mood)
        if synthesizer is not None:
            self._engine = synthesizer.name

    # ------------------------------------------------------------------ voice

    @property
    def voice(self) -> Voice | SystemVoice | ModelVoice:
        """The voice currently used to read."""
        return self._voice

    def set_voice(self, voice: str | Voice | SystemVoice | ModelVoice) -> Voice | SystemVoice | ModelVoice:
        """Choose a different voice and rebuild the engine for it."""
        self._voice = _as_voice(voice)
        self._synthesizer = None  # rebuilt lazily with the new voice
        return self._voice

    # ------------------------------------------------------------------- mood

    @property
    def mood(self) -> Mood | None:
        """The emotional state applied to the reading, or ``None`` for the voice's own."""
        return self._mood

    def set_mood(self, mood: str | Mood | None) -> Mood | None:
        """Choose the emotional state and rebuild the engine for it."""
        chosen = _as_mood(mood)
        self._mood = chosen
        self._synthesizer = None  # the mood is applied when the engine is rebuilt
        return chosen

    def _prosody(self) -> Prosody | None:
        """The prosody the current mood carries, or ``None`` for the neutral one."""
        if self._mood is None or self._mood.neutral:
            return None
        return self._mood.prosody

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
        """The engine that turns text into audio, built once and cached.

        A studio names its engine explicitly (``piper``, ``audio.cpp``, ...), and
        ``load_synthesizer`` raises for a named engine that is not installed rather
        than silently substituting silence. That is right for a *reading* call, but
        a studio must still open and report honestly, so the failure is caught here:
        the reader falls back to the silent engine, :meth:`available` stays False,
        and :meth:`describe` names the real next step instead of raising.
        """
        if self._synthesizer is None:
            try:
                self._synthesizer = load_synthesizer(
                    self.engine,
                    voice=self._voice,
                    language=self.language,
                    prosody=self._prosody(),
                )
            except ValueError as error:
                self._engine_error = str(error)
                self._synthesizer = SilentSynthesizer()
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
        mood = (
            f" in a {self._mood.label().lower()} mood"
            if self._mood is not None and not self._mood.neutral
            else ""
        )
        line = (
            f"Reading with the {self._voice.category_label()} voice “{self._voice.label()}” "
            f"through {self.engine_name}{mood}; audio is {played}."
        )
        note = windows_voice_report()
        if note:
            line = f"{line} {note}"
        return line


def _as_mood(value: str | Mood | None) -> Mood | None:
    """Coerce a mood key or object into a :class:`~core.voice_moods.Mood`.

    ``None`` means "the voice's own reading", which is kept as ``None`` rather than
    forced to the neutral mood, so a caller can tell "no mood chosen" from "the
    neutral mood chosen". An unknown key falls back to the neutral mood rather than
    raising, so a saved mood that left the catalog never breaks a reader.
    """
    if value is None:
        return None
    if isinstance(value, Mood):
        return value
    return resolve_mood(str(value))


def _as_voice(value: str | Voice | SystemVoice | ModelVoice | None) -> Voice | SystemVoice | ModelVoice:
    if value is None:
        return default_voice()
    if isinstance(value, (Voice, SystemVoice, ModelVoice)):
        return value
    key = str(value)
    if is_system_voice_key(key):
        resolved = find_system_voice(key)
        if resolved is not None:
            return resolved
        # The voice is no longer installed; fall back to the catalog default so a
        # saved choice never asks the engine for a name it would reject.
        return default_voice()
    if is_model_voice_key(key):
        resolved_model = find_model_voice(key)
        if resolved_model is not None:
            return resolved_model
        # The model (or that speaker) is no longer installed here.
        return default_voice()
    return find_voice(key)


def clip_is_silent(clip: object) -> bool:
    """True when a rendered clip carries no audio at all.

    A real :class:`~core.speech.AudioClip` exposes ``is_silent``; a caller's own
    test double may expose ``silent``. Both are honoured, because reading only one
    of the two names silently reported "spoke" for a clip that was in fact empty --
    the exact "engine produced no audio" case the windows mean to name.
    """
    if clip is None:
        return True
    value = getattr(clip, "is_silent", None)
    if value is None:
        value = getattr(clip, "silent", None)
    return bool(value)


def split_for_speech(text: str, *, max_chars: int = 600) -> tuple[str, ...]:
    """Split ``text`` into paragraph-sized chunks a synthesizer can read in turn.

    A paragraph longer than ``max_chars`` is split on sentence boundaries, so no
    single call is enormous; the chunks keep their order, so the reading is the
    text's own sequence. Shared by every window that reads aloud, so a document,
    a message, and a chat reply are all broken into speech the same way.
    """
    chunks: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue
        piece = ""
        for sentence in _sentences(paragraph):
            if piece and len(piece) + len(sentence) + 1 > max_chars:
                chunks.append(piece)
                piece = sentence
            else:
                piece = f"{piece} {sentence}".strip()
        if piece:
            chunks.extend(_hard_split(piece, max_chars))
    return tuple(chunks)


def _hard_split(text: str, max_chars: int) -> tuple[str, ...]:
    """Break a chunk that is still too long (no sentence marks) into pieces."""
    if len(text) <= max_chars:
        return (text,)
    return tuple(
        text[index : index + max_chars] for index in range(0, len(text), max_chars)
    )


def _sentences(paragraph: str) -> tuple[str, ...]:
    """Split a paragraph into sentences on the common enders, keeping the marks."""
    sentences: list[str] = []
    current = ""
    for character in paragraph:
        current += character
        if character in ".!?؟。":
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return tuple(sentences)


__all__ = ["TextReader", "clip_is_silent", "split_for_speech"]
