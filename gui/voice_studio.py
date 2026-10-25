"""One studio per speech engine: read a document aloud, with a voice and a mood.

The composer's model row opens each *answering model*'s own chat window
(:mod:`gui.model_chat_dialog`). This module is the sibling the user asked for: one
window per **speech engine**, where a document is loaded and read aloud, with three
things that belong to that engine alone:

1. the **engine's logo and name** -- shown as a badge and changeable (pick any
   picture, or rename the studio);
2. the **voice** its documents are read with -- exclusive to that engine, and
   remembered until the user changes it again, drawn from **only** the voices that
   engine really reports here (never another engine's);
3. the **mood** (an emotional state -- happy, sad, excited, rapturous, in love,
   angry, and so on) applied to that voice the moment it is chosen.

The document is loaded through the project's own reader (text, Word, PDF, and the
optional OCR path), shown as text, and read aloud in chunks so a long document
plays continuously.

The class is Tk-free on purpose, so the whole workflow -- choosing a voice, choosing
a mood, loading a document, reading it, renaming the studio, replacing its logo --
is exercised headless with no display, no model, and no audio hardware. The window
is a thin view over it. A ``reader`` may be injected (a test's own), so nothing here
needs a real speech engine to be tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core import voice_studio_profile as profiles
from core.document_reader import DocumentText, read_document
from core.reading import TextReader
from core.voice_moods import Mood, mood_keys, resolve_mood
from core.voices import (
    SOURCE_AUDIOCPP,
    SOURCE_PIPER,
    SOURCE_QWEN,
    SOURCE_SYSTEM,
    Voice,
    find_model_voice,
    find_system_voice,
    find_voice,
    group_model_voices,
    group_system_voices,
    is_model_voice_key,
    is_system_voice_key,
    model_voices,
    system_voices,
    voice_categories,
)

#: The engine name each studio source maps to, so a chosen voice can be routed to
#: the engine that owns it (see :func:`core.speech._engine_for_voice`). The system
#: studio uses ``auto`` -- it speaks with the voices the OS installed.
ENGINE_FOR_SOURCE: dict[str, str] = {
    SOURCE_SYSTEM: "auto",
    SOURCE_AUDIOCPP: "audio.cpp",
    SOURCE_PIPER: "piper",
    SOURCE_QWEN: "qwen3-tts",
}

#: The file suffixes the document picker offers, drawn from the reader's own set.
DOCUMENT_SUFFIXES: tuple[str, ...] = (
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl",
    ".docx", ".docm", ".pdf",
)

#: How many characters of a document are shown in the window (the whole text is
#: still read aloud). A huge document must not make the window unresponsive.
DEFAULT_PREVIEW_CHARS = 20_000


@dataclass(frozen=True)
class VoiceGroup:
    """One labelled group of voices in a studio's picker.

    ``label`` is the group header (a language for the real and model voices, a
    category for the synthetic catalog) and ``voices`` are the voices under it.
    """

    label: str
    voices: tuple[Any, ...]


@dataclass
class LoadedDocument:
    """A document opened in a studio: its path, its text, and where it came from."""

    path: Path
    text: str
    title: str = ""
    kind: str = "text"
    truncated: bool = False
    scanned: bool = False
    page_count: int = 0

    @property
    def filename(self) -> str:
        return self.path.name

    def summary(self) -> str:
        """An honest one-line description of the loaded document."""
        parts = [self.filename]
        if self.kind:
            parts.append(f"({self.kind})")
        parts.append(f"{len(self.text.split())} words")
        if self.page_count:
            parts.append(f"{self.page_count} page(s)")
        if self.scanned:
            parts.append("scanned pages were OCR'd")
        if self.truncated:
            parts.append("capped at the reader's limit")
        return " · ".join(parts)


@dataclass
class ReadReport:
    """The outcome of reading a document aloud, honestly reported."""

    ok: bool
    chunks: int = 0
    characters: int = 0
    seconds: float = 0.0
    message: str = ""

    def summary(self) -> str:
        if not self.ok:
            return self.message or "Nothing was read."
        return f"Read {self.characters} characters in {self.chunks} part(s)."


def _voice_key_of(voice: Any) -> str:
    """The stable key of any voice kind (a catalog, system, or model voice)."""
    return getattr(voice, "key", "")


def _voice_label(voice: Any) -> str:
    """A readable label for any voice kind."""
    label = getattr(voice, "label", None)
    if callable(label):
        return str(label())
    return str(getattr(voice, "label_en", getattr(voice, "name", "")))


class VoiceStudioService:
    """One speech engine's own studio: its logo, its voices, its moods, its document.

    It owns a :class:`~core.reading.TextReader` built for the engine's synthesizer
    and this studio's saved voice and mood, and the studio's saved
    :class:`~core.voice_studio_profile.VoiceStudioProfile`. Nothing is loaded until a
    document is actually read, so opening the window is cheap.
    """

    def __init__(
        self,
        source: str,
        *,
        reader: Any = None,
        data_root: Path | None = None,
    ) -> None:
        if source not in profiles.source_names():
            raise ValueError(
                f"Unknown speech-engine studio {source!r}; "
                f"choose one of {', '.join(profiles.source_names())}."
            )
        self.source = source
        self.engine = ENGINE_FOR_SOURCE.get(source, "auto")
        self.profile = profiles.load_profile(source)
        # The studio's own reader, built with the engine and the saved voice/mood, so
        # the voice and mood are exclusive to this studio and never touch the main
        # chat's reader.
        self._reader = reader if reader is not None else TextReader(engine=self.engine)
        saved_voice = self.profile.voice
        if saved_voice:
            self._apply_voice(saved_voice)
        self._reader.set_mood(profiles.resolve_mood(source))
        self.document: LoadedDocument | None = None
        self._last_report: ReadReport | None = None
        self._recording: Any = None

    # --------------------------------------------------------------- identity

    @property
    def title(self) -> str:
        """The engine's full title (its button and window label)."""
        return profiles.studio_title(self.source)

    @property
    def name(self) -> str:
        """The studio's display name (the user may have renamed it)."""
        return self.profile.name

    def set_name(self, name: str) -> str:
        """Rename this studio, persist it, and return the new name."""
        self.profile = profiles.set_name(self.source, name)
        return self.profile.name

    def logo_path(self) -> Path | None:
        """The saved logo file, or ``None`` when the default badge should be drawn."""
        return self.profile.logo_path()

    def set_logo(self, source_path: str | Path) -> Path | None:
        """Copy ``source_path`` in as this studio's logo and return the stored file."""
        self.profile = profiles.save_logo(self.source, source_path)
        return self.profile.logo_path()

    def clear_logo(self) -> None:
        """Remove this studio's logo, back to the default badge."""
        self.profile = profiles.clear_logo(self.source)

    def blurb(self) -> str:
        """A short, honest line about what this studio speaks with."""
        engine = self._reader.engine_name if self._reader.available() else "no engine installed"
        return f"Reads documents aloud through {self.title} ({engine})."

    # ------------------------------------------------------------------ voices

    def voices(self) -> tuple[Any, ...]:
        """Every voice this engine really reports here -- and only its own.

        The *Installed Voices* studio lists the voices the OS installed; each local
        model studio lists only that model's own voices. A voice that belongs to
        another engine is never offered, which is the whole point of a per-engine
        studio.
        """
        if self.source == SOURCE_SYSTEM:
            return tuple(system_voices())
        return tuple(voice for voice in model_voices() if voice.source == self.source)

    def voice_groups(self) -> tuple[VoiceGroup, ...]:
        """The studio's voices grouped by language (Arabic first), for its picker."""
        voices = self.voices()
        groups: tuple[tuple[str, tuple[Any, ...]], ...]
        if self.source == SOURCE_SYSTEM:
            groups = group_system_voices(voices)
        else:
            groups = group_model_voices(voices)
        return tuple(VoiceGroup(label, tuple(group)) for label, group in groups)

    def catalog_voices(self) -> tuple[tuple[str, str, tuple[Voice, ...]], ...]:
        """The synthetic catalog, offered only by the *Installed Voices* studio.

        The catalog is a set of *settings* for whichever engine discovery picks, so
        it belongs to the system studio alone; the local model studios speak with
        their own voices and are never given these.
        """
        if self.source != SOURCE_SYSTEM:
            return ()
        return tuple(
            (key, label_en, tuple(voice for voice in self._catalog() if voice.category == key))
            for key, label_en, _label_ar in voice_categories()
        )

    @staticmethod
    def _catalog() -> tuple[Voice, ...]:
        from core.voices import VOICE_CATALOG

        return VOICE_CATALOG

    def voice_keys(self) -> tuple[str, ...]:
        """Every voice key this studio offers, in menu order."""
        keys = [_voice_key_of(voice) for voice in self.voices()]
        for _key, _label, group in self.catalog_voices():
            keys.extend(_voice_key_of(voice) for voice in group)
        return tuple(key for key in keys if key)

    def voice_key(self) -> str:
        """The chosen voice key (its own, not another studio's)."""
        return self._reader.voice.key

    def set_voice(self, key: str) -> str:
        """Choose this studio's reading voice, persist it, and return its key.

        Raises :class:`ValueError` for a key this studio does not offer, so a
        window cannot silently read with another engine's voice.
        """
        if key not in self.voice_keys():
            raise ValueError(f"{self.title} does not offer the voice {key!r}.")
        voice = self._reader.set_voice(key)
        self.profile = profiles.set_voice(self.source, voice.key)
        return voice.key

    def voice_label(self, key: str) -> str:
        """A readable label for a voice key, resolved against what is installed."""
        voice = self._resolve_voice(key)
        if voice is None:
            return key
        return f"{voice.category_label()} “{_voice_label(voice)}”"

    @staticmethod
    def _resolve_voice(key: str) -> Any:
        if is_system_voice_key(key):
            return find_system_voice(key)
        if is_model_voice_key(key):
            return find_model_voice(key)
        try:
            return find_voice(key)
        except ValueError:
            return None

    def _apply_voice(self, key: str) -> None:
        """Apply a saved voice, ignoring one that is no longer installed."""
        try:
            self._reader.set_voice(key)
        except ValueError:
            self.profile = profiles.save_profile(self.profile.with_voice(""))

    # ------------------------------------------------------------------- moods

    def moods(self) -> tuple[Mood, ...]:
        """Every selectable mood, in catalog order."""
        from core.voice_moods import MOODS

        return MOODS

    def mood_key(self) -> str:
        """The chosen mood key (its own, not another studio's)."""
        mood = self._reader.mood
        return mood.key if mood is not None else profiles.DEFAULT_MOOD_KEY

    def set_mood(self, key: str) -> str:
        """Choose this studio's mood, apply it at once, persist it, and return its key.

        Raises :class:`ValueError` for an unknown key, so a window reports a bad
        choice instead of applying something else.
        """
        if key not in mood_keys():
            raise ValueError(f"Unknown mood {key!r}.")
        mood = resolve_mood(key)
        self._reader.set_mood(mood)
        self.profile = profiles.set_mood(self.source, mood.key)
        return mood.key

    def mood_label(self, key: str) -> str:
        """A readable label for a mood key."""
        return resolve_mood(key).label()

    def mood_support_note(self) -> str:
        """An honest note about how much of a mood this studio's engine can apply."""
        from core.voice_moods import support_note

        return f"{self.title}: {support_note(self._reader.engine_name)}"

    # --------------------------------------------------------------- reading

    def reading_available(self) -> bool:
        """True when a real speech engine is installed here."""
        return self._reader.available()

    def reading_status(self) -> str:
        """An honest, human-readable status of this studio's reader."""
        return self._reader.describe()

    def test_voice(self, text: str = "") -> Any:
        """Read a short line aloud so the user can hear the voice and mood."""
        sample = text.strip() or f"This is {self.name} reading with the chosen voice and mood."
        return self._reader.speak(sample)

    # -------------------------------------------------------------- documents

    def load_document(self, path: str | Path, *, allow_ocr: bool = True) -> LoadedDocument:
        """Open a document, read its text, and remember it for reading aloud.

        The project's own reader does the work (text, Word, PDF, and the optional
        OCR path for a scanned PDF); a file the reader refuses raises
        :class:`core.document_reader.DocumentReadError` with the real reason, which
        the window reports rather than showing an empty document.
        """
        source = Path(path).expanduser()
        result: DocumentText = read_document(source, allow_ocr=allow_ocr)
        self.document = LoadedDocument(
            path=source,
            text=result.text,
            title=result.title,
            kind=result.kind,
            truncated=result.truncated,
            scanned=result.scanned,
            page_count=len(result.page_numbers),
        )
        return self.document

    def document_text(self, *, limit: int = DEFAULT_PREVIEW_CHARS) -> str:
        """The loaded document's text, capped for display (``""`` when none is loaded)."""
        if self.document is None:
            return ""
        text = self.document.text
        if limit and len(text) > limit:
            return text[:limit] + "\n\n… (the rest is still read aloud)"
        return text

    def read_aloud(self, text: str | None = None) -> ReadReport:
        """Read the loaded document (or ``text``) aloud with this studio's voice/mood.

        A long document is read in paragraph-sized chunks so playback is continuous
        and a failure part-way is reported honestly rather than silently truncated.
        """
        body = (text if text is not None else (self.document.text if self.document else "")).strip()
        if not body:
            report = ReadReport(ok=False, message="There is no document to read.")
            self._last_report = report
            return report
        if not self.reading_available():
            report = ReadReport(
                ok=False,
                message="No speech engine is installed here, so nothing can be read aloud.",
            )
            self._last_report = report
            return report
        chunks = _split_for_speech(body)
        read = 0
        seconds = 0.0
        for chunk in chunks:
            clip = self._reader.speak(chunk)
            if getattr(clip, "silent", False):
                report = ReadReport(
                    ok=False,
                    chunks=read,
                    characters=sum(len(c) for c in chunks[:read]),
                    seconds=seconds,
                    message="The engine produced no audio; install a speech engine to hear it.",
                )
                self._last_report = report
                return report
            read += 1
            seconds += float(getattr(clip, "duration_seconds", 0.0) or 0.0)
        report = ReadReport(
            ok=True,
            chunks=len(chunks),
            characters=len(body),
            seconds=seconds,
        )
        self._last_report = report
        return report

    def last_report(self) -> ReadReport | None:
        """The most recent read's outcome, or ``None`` when nothing was read yet."""
        return self._last_report

    def save_speech(self, path: str | Path, text: str | None = None) -> Path:
        """Render the loaded document (or ``text``) to a WAV file."""
        body = (text if text is not None else (self.document.text if self.document else "")).strip()
        return self._reader.save(body, path)

    def close_document(self) -> None:
        """Forget the loaded document."""
        self.document = None

    # ------------------------------------------------------------ microphone

    def microphone_available(self) -> bool:
        """True when a recorder *and* a speech-to-text engine are installed here."""
        from core.speech import MicrophoneRecorder

        return MicrophoneRecorder().available() and self._recognizer_available()

    def asr_status(self) -> str:
        """An honest, human-readable status of the microphone path."""
        from core.speech import MicrophoneRecorder, microphone_install_hint

        if not MicrophoneRecorder().available():
            return (
                "No microphone recorder is installed, so the microphone cannot record. "
                + microphone_install_hint()
            )
        if not self._recognizer_available():
            from core.qwen_speech import qwen_asr_status

            return "No speech-to-text engine is installed. " + qwen_asr_status().hint
        return "Listening through the project's own speech-to-text engine."

    def start_recording(self) -> Any:
        """Begin a microphone recording (raises when no recorder is installed)."""
        from core.speech import MicrophoneRecorder

        if self._recording is not None and getattr(self._recording, "recording", False):
            return self._recording
        self._recording = MicrophoneRecorder(seconds=120).start()
        return self._recording

    def stop_recording(self) -> Any:
        """Stop the current recording (if any) and return it, or ``None``."""
        recording = self._recording
        self._recording = None
        if recording is not None:
            recording.stop()
        return recording

    def transcribe(self, clip: Any) -> str:
        """Turn a recorded clip into text, raising when no recognizer is available."""
        from core.speech import load_recognizer

        name = _discover_recognizer_name()
        if name == "transcript":
            raise ValueError(self.asr_status())
        recognizer = load_recognizer(name)
        return recognizer.transcribe(clip).strip()

    @staticmethod
    def _recognizer_available() -> bool:
        return _discover_recognizer_name() != "transcript"

    def describe(self) -> str:
        """A one-line description of the studio, its voice, and its mood."""
        label = self.name if self.name != self.title else self.title
        return (
            f"{label}: voice {self.voice_label(self.voice_key())}, "
            f"mood {self.mood_label(self.mood_key())}."
        )


def _discover_recognizer_name() -> str:
    """The ASR engine discovery would pick (``'transcript'`` when there is none)."""
    from core.speech import discover_recognizer_name

    return discover_recognizer_name()


def _split_for_speech(text: str, *, max_chars: int = 600) -> tuple[str, ...]:
    """Split ``text`` into paragraph-sized chunks a synthesizer can read in turn.

    A paragraph longer than ``max_chars`` is split on sentence boundaries, so no
    single call is enormous; the chunks keep their order, so the reading is the
    document's own sequence.
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


__all__ = [
    "DEFAULT_PREVIEW_CHARS",
    "DOCUMENT_SUFFIXES",
    "ENGINE_FOR_SOURCE",
    "LoadedDocument",
    "ReadReport",
    "VoiceGroup",
    "VoiceStudioService",
]
