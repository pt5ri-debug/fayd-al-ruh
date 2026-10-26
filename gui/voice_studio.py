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

from core import reading
from core import voice_studio_profile as profiles
from core.audiocpp_speech import AUDIOCPP_ENGINE_NAME
from core.best_effort import best_effort
from core.document_reader import DocumentText, read_document
from core.reading import TextReader, clip_is_silent, split_for_speech
from core.voice_moods import Mood, mood_keys, resolve_mood
from core.voices import (
    SOURCE_AUDIOCPP,
    SOURCE_PIPER,
    SOURCE_QWEN,
    SOURCE_SYSTEM,
    Voice,
    default_voice,
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


@dataclass(frozen=True)
class SetupStatus:
    """Exactly what is present and what is missing for one engine's studio.

    Two independent halves, because they fail for different reasons and are fixed
    differently: the **runtime** (a Python extra or a native binary) and the
    **voices** (a model file, a reference clip, or the OS's own voices). A studio
    is ``ready`` only when both are present, and each missing half carries its own
    real next step in ``hint``.
    """

    source: str
    runtime_ok: bool
    voices_ok: bool
    hint: str
    can_install_runtime: bool = False
    can_download_voices: bool = False

    @property
    def ready(self) -> bool:
        """True when this engine can really read text aloud here."""
        return self.runtime_ok and self.voices_ok

    def describe(self) -> str:
        """An honest one-line status naming the missing half and the next step."""
        if self.ready:
            return "Ready: the engine and at least one voice are installed."
        missing = []
        if not self.runtime_ok:
            missing.append("the engine")
        if not self.voices_ok:
            missing.append("its voices")
        return f"Not ready here: {', and '.join(missing)} missing. {self.hint}"


@dataclass
class SetupReport:
    """The outcome of an attempted install or download, ready to show the user."""

    ok: bool
    message: str = ""
    details: str = ""


def _voice_key_of(voice: Any) -> str:
    """The stable key of any voice kind (a catalog, system, or model voice)."""
    return getattr(voice, "key", "")


def _is_catalog_key(key: str) -> bool:
    """True for a synthetic-catalog key (not a ``system:`` or ``model:`` voice).

    The synthetic catalog is a set of *settings* for whichever engine discovery
    picks, so a local engine cannot be handed one; the prefix test is the same one
    :func:`core.voices.is_system_voice_key`/:func:`is_model_voice_key` use.
    """
    return not (is_system_voice_key(key) or is_model_voice_key(key))


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
        # chat's reader. The audio.cpp studio reads through the engine's **resident
        # server** when the build ships one, because a document is read paragraph by
        # paragraph and every separate process would reload the whole GGUF each time.
        self._reader = (
            reader
            if reader is not None
            else TextReader(
                engine=self.engine,
                server=self.engine == AUDIOCPP_ENGINE_NAME,
            )
        )
        # Pick the studio's starting voice *before* anything else. Without this the
        # reader kept its own default -- the synthetic catalog's ``woman-1`` -- and a
        # local engine cannot speak a catalog voice: audio.cpp has no reference clip
        # to clone, Piper has no matching ONNX file, and Qwen3-TTS has no speaker
        # name, so each one failed and fell back to the system engine (the Windows
        # default voice) instead of the voice the studio offered.
        self._apply_initial_voice()
        self._reader.set_mood(profiles.resolve_mood(source))
        self.document: LoadedDocument | None = None
        self._last_report: ReadReport | None = None
        self._recording: Any = None
        # The reading in progress, so a second click can stop it (the toggle the
        # user asked for): the chunk index reached and the clip playing right now.
        self._playback: Any = None
        self._reading_text: str = ""
        self._reading_index: int = 0

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

    # ------------------------------------------------- the user's own voices

    def can_import_voices(self) -> bool:
        """True for the VoiceTut studio, which can clone a voice from a clip.

        Only a *zero-shot cloning* engine can take the user's own voice: it speaks
        from a reference clip, so importing one (or recording one) is meaningful.
        Piper needs an ONNX model and Qwen3-TTS has a fixed speaker set, so neither
        offers this -- the buttons are hidden rather than shown and refused.
        """
        return self.source == SOURCE_AUDIOCPP

    def import_voice(
        self, path: str | Path | Any, *, name: str = "", transcript: str = ""
    ) -> Any:
        """Import an audio file (or a recorded clip) as a new voice of this engine.

        Delegates to :func:`core.audiocpp_speech.import_voice`, so the window, the
        CLI, and the registry agree. Raises :class:`ValueError` for a file the
        engine cannot clone from (an ``.onnx`` voice, a bad format, silence), so the
        window reports the real reason instead of adding a dead voice.
        """
        from core.audiocpp_speech import import_voice as _import

        voice = _import(path, name=name, transcript=transcript)
        self._ensure_transcript(voice.name, self._clip_path(voice.clip), transcript)
        self.refresh()
        self._select_voice_if_offered(voice.name)
        return voice

    def clone_voice(
        self, recording: str | Path | Any, *, name: str = "", transcript: str = ""
    ) -> Any:
        """Register a recording of the user's own voice, then select it.

        ``recording`` is a file path or the microphone's own
        :class:`~core.speech.AudioClip` (what :meth:`start_recording` yields). When
        no transcript was given, the speech-to-text engine's own reading of the clip
        is used, because a cloning voice needs to know what the clip says -- a voice
        recorded without one is registered but cannot be spoken.
        """
        from core.audiocpp_speech import clone_voice as _clone

        label = name or "My voice"
        voice = _clone(recording, name=label, transcript=transcript)
        self._ensure_transcript(label, recording, transcript)
        self.refresh()
        self._select_voice_if_offered(voice.name)
        return voice

    def _clip_path(self, clip: str) -> Path:
        """The full path of an imported voice's reference clip."""
        from core.audiocpp_speech import voices_dir

        return voices_dir() / clip

    def _ensure_transcript(self, name: str, clip: Any, transcript: str) -> None:
        """Transcribe ``clip`` when the voice has no transcript, so it can be spoken.

        Best-effort on purpose: a machine with no speech-to-text engine still gets
        the voice, and the studio says which next step fills the transcript in.
        """
        if transcript.strip():
            return
        if not self._recognizer_available():
            return
        with best_effort("transcribe a cloned voice so the engine can speak it"):
            from core.audiocpp_speech import set_imported_voice_transcript

            text = self.transcribe(clip)
            set_imported_voice_transcript(name, text)

    def remove_voice(self, name: str) -> bool:
        """Remove an imported voice (and its clip); ``True`` when one was removed."""
        from core.audiocpp_speech import remove_imported_voice

        removed = remove_imported_voice(name)
        if removed:
            self.refresh()
        return removed

    def imported_voice_names(self) -> tuple[str, ...]:
        """The names of the voices the user added here, in registry order."""
        if not self.can_import_voices():
            return ()
        from core.audiocpp_speech import load_imported_voices

        return tuple(voice.name for voice in load_imported_voices())

    def _select_voice_if_offered(self, name: str) -> None:
        """Select a freshly added voice by its model key, when the studio offers it."""
        key = f"model:{self.engine}:{name}"
        if key in self.voice_keys():
            with best_effort("select the voice that was just added"):
                self.set_voice(key)

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

    def _apply_initial_voice(self) -> None:
        """Start on the engine's own voice: the saved one, else its first real voice.

        A saved voice is honoured only while the engine still offers it. Otherwise
        the studio starts on the first voice the engine *really* reports here, so a
        local engine is handed one of its own voices (with its reference clip, ONNX
        file, or speaker name) instead of the synthetic catalog default it cannot
        speak.

        The synthetic catalog is *settings for the system engine*, so it is the one
        engine those keys are valid for: the system studio keeps the project's own
        documented default catalog voice (:func:`core.voices.default_voice`) rather
        than switching to an arbitrary catalog entry, and a studio with no voice at
        all leaves the reader on its own default (nothing better to choose).
        """
        available = self.voice_keys()
        saved = self.profile.voice
        if saved and saved in available:
            self._apply_voice(saved)
            return
        real = tuple(key for key in available if not _is_catalog_key(key))
        if real:
            self._apply_voice(real[0])
            return
        if self.source == SOURCE_SYSTEM:
            fallback = default_voice().key
            if fallback in available:
                self._apply_voice(fallback)

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

    # ------------------------------------------------------------------ setup

    def setup_status(self) -> SetupStatus:
        """Report exactly what this engine is missing here, and the real next step.

        The two halves are kept apart on purpose: the runtime (a Python extra, or
        audio.cpp's native binary) is installed differently from the voices (a model
        file, a reference clip, or the OS's own installed voices), so a user is
        never told to "install the engine" when the engine is present and only a
        voice is missing -- which is exactly the confusion this studio now avoids.
        """
        if self.source == SOURCE_SYSTEM:
            from core.speech import speech_available, speech_install_plan

            voices = bool(system_voices())
            available = speech_available()
            hint = "Ready." if available and voices else speech_install_plan().hint
            return SetupStatus(
                source=self.source,
                runtime_ok=available,
                voices_ok=voices or available,
                hint=hint,
                can_install_runtime=not available and speech_install_plan().can_install,
                can_download_voices=False,
            )
        if self.source == SOURCE_PIPER:
            from core.piper_speech import piper_status

            piper = piper_status()
            return SetupStatus(
                source=self.source,
                runtime_ok=piper.extra_installed,
                voices_ok=piper.model_cached,
                hint=piper.hint,
                can_install_runtime=not piper.extra_installed,
                can_download_voices=piper.extra_installed and not piper.model_cached,
            )
        if self.source == SOURCE_QWEN:
            from core.qwen_speech import qwen_tts_status

            qwen = qwen_tts_status()
            return SetupStatus(
                source=self.source,
                runtime_ok=qwen.extra_installed,
                voices_ok=qwen.model_cached,
                hint=qwen.hint,
                can_install_runtime=not qwen.extra_installed,
                can_download_voices=qwen.extra_installed and not qwen.model_cached,
            )
        # audio.cpp: the engine is a native binary and the voices are reference clips.
        from core.audiocpp_speech import audiocpp_status

        audio = audiocpp_status()
        return SetupStatus(
            source=self.source,
            runtime_ok=audio.engine_installed,
            voices_ok=audio.voices > 0,
            hint=audio.hint,
            can_install_runtime=not audio.engine_installed,
            can_download_voices=audio.engine_installed and audio.voices == 0,
        )

    def setup_status_text(self) -> str:
        """A short, human-readable line for the window's setup block."""
        return self.setup_status().describe()

    def fetchable_voices(self) -> tuple[tuple[str, str], ...]:
        """The voices this engine can fetch, as ``(key, title)`` rows.

        Empty for the system studio (its voices come from the OS, not a download) and
        for an engine whose runtime is not installed yet -- you cannot download a
        voice for an engine that cannot run it. This is what the window's download
        picker is built from, so it never offers something the engine cannot use.
        """
        status = self.setup_status()
        if not status.runtime_ok:
            return ()
        if self.source == SOURCE_PIPER:
            from core.piper_speech import spec_summaries

            return tuple((row[0], row[1]) for row in spec_summaries())
        if self.source == SOURCE_QWEN:
            from core.qwen_speech import tts_spec_summaries

            return tuple((row[0], row[1]) for row in tts_spec_summaries())
        if self.source == SOURCE_AUDIOCPP:
            from core.audiocpp_speech import spec_summaries

            return tuple((row[0], row[1]) for row in spec_summaries())
        return ()

    def install_runtime(self) -> SetupReport:
        """Install this engine's runtime (a Python extra, or audio.cpp's binary).

        Governed exactly like every other external action in the project: the model
        and engine downloads run only through the permission and human-approval
        gates, so this passes ``approved=True`` because the user clicked the button
        (the explicit approval). Never raises; the failure is reported.
        """
        try:
            if self.source == SOURCE_SYSTEM:
                from core.speech import install_speech_engine

                speech = install_speech_engine()
                return SetupReport(ok=speech.ok, message=speech.message)
            if self.source == SOURCE_PIPER:
                from core import optional_deps

                piper = optional_deps.install(optional_deps.FEATURE_PIPER)
                return SetupReport(ok=piper.ok, message=piper.message)
            if self.source == SOURCE_QWEN:
                from core import optional_deps

                qwen = optional_deps.install(optional_deps.FEATURE_QWEN_TTS)
                return SetupReport(ok=qwen.ok, message=qwen.message)
            from core.audiocpp_speech import install_engine

            path = install_engine(approved=True)
            return SetupReport(ok=True, message=f"Installed the audio.cpp engine in {path}.")
        except Exception as error:  # a governed action must report, never crash the window
            return SetupReport(ok=False, message=f"Could not install the engine: {error}")
        finally:
            self.refresh()

    def download_voices(self, voice_key: str = "") -> SetupReport:
        """Download this engine's model/voice weights (a governed external action).

        ``voice_key`` selects one fetchable voice for Piper/Qwen3-TTS/audio.cpp; an
        empty key takes the engine's own default. Runs only through the permission
        and human-approval gates (``approved=True`` is the click), and never raises.
        """
        status = self.setup_status()
        if not status.runtime_ok:
            return SetupReport(
                ok=False,
                message=f"Install the engine first, then its voices. {status.hint}",
            )
        try:
            if self.source == SOURCE_PIPER:
                from core.piper_speech import download_model as download_piper_voice

                path = download_piper_voice(voice_key or None, approved=True)
                self.refresh()
                return SetupReport(ok=True, message=f"Downloaded the Piper voice into {path}.")
            if self.source == SOURCE_QWEN:
                from core.qwen_speech import download_model as download_qwen_model

                path = download_qwen_model("tts", voice_key or None, approved=True)
                self.refresh()
                return SetupReport(ok=True, message=f"Downloaded the Qwen3-TTS model into {path}.")
            if self.source == SOURCE_AUDIOCPP:
                from core.audiocpp_speech import download_model as download_voicetut
                from core.audiocpp_speech import download_speakers

                model = download_voicetut(voice_key or None, approved=True)
                download_speakers(approved=True)
                self.refresh()
                return SetupReport(
                    ok=True,
                    message=f"Downloaded the VoiceTut-TTS model and its voices into {model}.",
                )
            return SetupReport(
                ok=False,
                message="This studio's voices come from the operating system, not a download.",
            )
        except Exception as error:
            return SetupReport(ok=False, message=f"Could not download the voices: {error}")

    def refresh(self) -> None:
        """Re-discover the engine and voices after an install, and re-apply the voice.

        A newly installed engine (or a newly downloaded voice) is picked up here, so
        the studio starts reading with it without reopening the window. The saved
        voice is kept when it is still offered, and the first real voice is used when
        it is not -- the same rule the studio opens with. ``TextReader`` re-discovers
        its engine; an injected reader (the test seam) may not implement that, which
        is not a failure here -- the voice is still re-applied.
        """
        refresh = getattr(self._reader, "refresh", None)
        if callable(refresh):
            with best_effort("refresh the reading engine after a setup change"):
                refresh()
        self._apply_initial_voice()

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
        for group in _batch_chunks(chunks):
            try:
                clips = self._render_group(group)
            except Exception as error:  # a governed read reports, never crashes
                report = ReadReport(
                    ok=False, chunks=read, characters=sum(len(c) for c in chunks[:read]),
                    seconds=seconds, message=f"The engine could not read the document: {error}",
                )
                self._last_report = report
                return report
            for clip in clips:
                if clip_is_silent(clip):
                    report = ReadReport(
                        ok=False,
                        chunks=read,
                        characters=sum(len(c) for c in chunks[:read]),
                        seconds=seconds,
                        message="The engine produced no audio; install a speech engine to hear it.",
                    )
                    self._last_report = report
                    return report
                playback = self._reader.play(clip)
                if playback is not None:
                    playback.wait()
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

    # ---------------------------------------------------- stoppable reading

    def reading(self) -> bool:
        """True while a reading is in progress (so the button shows ``Stop``)."""
        return self._reading_text != ""

    def start_reading(self, text: str | None = None) -> ReadReport:
        """Read the loaded document aloud, blocking, so a click can stop it.

        This is the toggle's ``start`` half: it reads the whole body chunk by chunk
        and *plays* each chunk in the background, then waits for it, checking after
        every chunk whether :meth:`stop_reading` was called (the window runs this on
        a worker thread, so the UI stays live). A stop ends the reading at once and
        is reported honestly as stopped rather than finished.

        Unlike :meth:`read_aloud` -- which renders and speaks each chunk and is kept
        for the CLI and the tests -- this path is the one the window's toggle uses,
        because only it can be interrupted between chunks.
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
        self._reading_text = body
        self._reading_index = 0
        read = 0
        seconds = 0.0
        try:
            for group in _batch_chunks(chunks):
                if not self.reading():
                    break
                try:
                    clips = self._render_group(group)
                except Exception as error:  # a governed read reports, never crashes
                    report = ReadReport(
                        ok=False, chunks=read, characters=sum(len(c) for c in chunks[:read]),
                        seconds=seconds, message=f"The engine could not read the document: {error}",
                    )
                    self._last_report = report
                    return report
                for clip in clips:
                    if not self.reading():
                        break
                    if clip_is_silent(clip):
                        report = ReadReport(
                            ok=False, chunks=read, characters=sum(len(c) for c in chunks[:read]),
                            seconds=seconds,
                            message="The engine produced no audio; install a speech engine to hear it.",
                        )
                        self._last_report = report
                        return report
                    self._reading_index = read
                    self._playback = self._reader.play(clip)
                    if self._playback is not None:
                        self._playback.wait()
                    self._playback = None
                    read += 1
                    seconds += float(getattr(clip, "duration_seconds", 0.0) or 0.0)
            # `reading()` is still true only when the loop ran to the end: a stop
            # clears the flag, so this is "completed", not "stopped".
            completed = self.reading()
            report = ReadReport(
                ok=True, chunks=read, characters=len(body), seconds=seconds,
                message="" if completed else "Reading stopped.",
            )
            self._last_report = report
            return report
        finally:
            self._reading_text = ""
            self._reading_index = 0
            self._playback = None

    def _render_group(self, group: tuple[str, ...]) -> list[Any]:
        """Render a batch of chunks, loading a local engine's model **once**.

        ``render_many`` is the batched path (audio.cpp loads its 1.35 GB GGUF once
        for the whole batch); a reader double that only offers ``render`` is called
        chunk by chunk, so the test seam stays small and the result is identical.
        """
        many = getattr(self._reader, "render_many", None)
        if callable(many):
            return list(many(list(group)))
        return [self._reader.render(chunk) for chunk in group]

    def stop_reading(self) -> bool:
        """Ask the reading in progress to stop; ``True`` when one was running.

        The stop takes effect at the next chunk boundary and, because the playing
        clip is stopped here, the sound ends at once. The window calls this on the
        UI thread while the worker thread is inside :meth:`start_reading`.
        """
        if not self.reading():
            return False
        self._reading_text = ""
        playback = self._playback
        if playback is not None:
            with best_effort("stop the clip that is playing"):
                playback.stop()
        return True

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
    """Split text into speech chunks (the shared :func:`core.reading.split_for_speech`).

    Kept as a thin alias so the studio and the tests share one implementation with
    every other window that reads aloud.
    """
    return split_for_speech(text, max_chars=max_chars)


#: How many characters of chunk text one engine call may carry (the shared cap).
BATCH_MAX_CHARS = reading.BATCH_MAX_CHARS

#: How many chunks one engine call may carry (the shared cap).
BATCH_MAX_CHUNKS = reading.BATCH_MAX_CHUNKS


def _batch_chunks(chunks: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Group speech chunks into engine-call batches (the shared :func:`core.reading.batch_chunks`).

    Kept as a thin alias so the studio and the tests share one implementation with
    every other window that reads aloud.
    """
    return reading.batch_chunks(chunks)


__all__ = [
    "DEFAULT_PREVIEW_CHARS",
    "DOCUMENT_SUFFIXES",
    "ENGINE_FOR_SOURCE",
    "LoadedDocument",
    "ReadReport",
    "SetupReport",
    "SetupStatus",
    "VoiceGroup",
    "VoiceStudioService",
]
