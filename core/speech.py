"""Speech I/O for the assistant: Arabic ASR and TTS behind a small interface.

Voice is a *front end* to the grounded assistant, not a new reasoning layer. This
module defines the contracts and the plumbing so a real speech engine can be
dropped in without touching the cognitive core:

* :class:`SpeechRecognizer` (ASR) turns audio into text;
* :class:`SpeechSynthesizer` (TTS) turns text into audio;
* :class:`AudioClip` is a tiny, dependency-free PCM buffer, so the whole pipeline
  can be tested and demoed without any audio hardware or third-party package.

Engines are discovered lazily by name (``piper``, ``whisper``, ...). When none is
installed the project still exposes a working *text* voice loop -- the recognizer
returns a supplied transcript and the synthesizer produces a silent clip -- so the
conversation flow is exercised end to end on a fresh clone.

This module is the **one** place every chat interface gets its speech from: the
GUI, Jarvis, ``ai.voice``, ``ai.speak``, and every future mouth call
:func:`load_recognizer` / :func:`load_synthesizer` (or ``'auto'``). The two
**Qwen3** models -- :mod:`core.qwen_speech`'s Qwen3-ASR and Qwen3-TTS -- are wired
in here as named engines (``'qwen3-asr'`` / ``'qwen3-tts'``), so adding them once
makes them available to every interface, present and future, with no per-window
work. ``auto`` prefers the Qwen3 model for ASR (it covers Arabic) and a system
engine for TTS (tiny, installed, and Arabic-capable); choosing ``qwen3-tts``
explicitly still reads Arabic through the system engine rather than failing.
"""

from __future__ import annotations

import array
import contextlib
import platform
import re
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from core.arabic_speech import prepare_for_speech
from core.model_interface import CognitiveModel, ModelRequest
from core.voice_moods import Prosody, apply_mood
from core.voices import (
    ModelVoice,
    SystemVoice,
    Voice,
    default_voice,
    detect_language,
    find_model_voice,
    find_system_voice,
    find_voice,
    is_model_voice_key,
    is_system_voice_key,
)

SAMPLE_RATE = 16_000

#: A voice the caller can choose: a synthetic catalog :class:`Voice`, a
#: :class:`SystemVoice` naming a voice the OS really reports, or a
#: :class:`ModelVoice` naming a voice of a **local model** the project runs. The
#: engine treats a ``SystemVoice`` as an exact name to select, a ``Voice`` as
#: settings to map, and a ``ModelVoice`` as a name for the synthesizer that owns its
#: engine (so the reader can route it there rather than to the default).
SelectableVoice = Voice | SystemVoice | ModelVoice


#: The voice kinds an engine selects by **name** rather than by mapping catalog
#: settings: a :class:`~core.voices.SystemVoice` (a voice the OS installed) and a
#: :class:`~core.voices.ModelVoice` (one inside a local model). Neither carries a
#: catalog ``variant``/``pitch``/``speed``, so every engine that maps those onto its
#: own knobs must leave a named voice alone. One tuple, used through
#: ``isinstance`` -- which also narrows the union for the type checker -- so adding a
#: voice kind cannot leave one of those engines behind.
NAMED_VOICE_TYPES = (SystemVoice, ModelVoice)


def _pcm_to_mono16(frames: bytes, channels: int, width: int) -> tuple[int, ...]:
    """Decode raw PCM ``frames`` into mono 16-bit samples.

    ``channels`` and ``width`` come straight from the WAV header. 8-bit WAV is
    unsigned, 16/32-bit are signed little-endian; 24-bit is not produced by the
    engines used here and raises so a caller sees an honest failure rather than a
    misread clip. Several channels are averaged to mono.
    """
    if width == 1:
        samples = [(value - 128) << 8 for value in frames]
    elif width == 2:
        buffer = array.array("h")
        buffer.frombytes(frames[: len(frames) - (len(frames) % 2)])
        samples = list(buffer)
    elif width == 4:
        buffer = array.array("i")
        buffer.frombytes(frames[: len(frames) - (len(frames) % 4)])
        samples = [value >> 16 for value in buffer]
    else:
        raise ValueError(f"Unsupported WAV sample width: {width} bytes.")
    if channels <= 1:
        return tuple(samples)
    mono: list[int] = []
    for start in range(0, len(samples) - channels + 1, channels):
        mono.append(sum(samples[start : start + channels]) // channels)
    return tuple(mono)


@dataclass(frozen=True)
class AudioClip:
    """A mono PCM buffer, 16-bit samples at :data:`SAMPLE_RATE`."""

    samples: tuple[int, ...]
    sample_rate: int = SAMPLE_RATE

    def __post_init__(self) -> None:
        if self.sample_rate < 1:
            raise ValueError("Sample rate must be positive.")
        if any(not -32768 <= sample <= 32767 for sample in self.samples):
            raise ValueError("PCM samples must be signed 16-bit values.")

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def is_silent(self) -> bool:
        return not any(self.samples)

    def to_bytes(self) -> bytes:
        import array

        buffer = array.array("h", self.samples)
        return buffer.tobytes()

    @classmethod
    def silence(cls, duration_seconds: float, *, sample_rate: int = SAMPLE_RATE) -> AudioClip:
        if duration_seconds < 0:
            raise ValueError("Silence duration cannot be negative.")
        return cls(tuple([0] * int(duration_seconds * sample_rate)), sample_rate)

    @classmethod
    def from_bytes(cls, payload: bytes, *, sample_rate: int = SAMPLE_RATE) -> AudioClip:
        import array

        samples = array.array("h")
        samples.frombytes(payload[: len(payload) - (len(payload) % 2)])
        return cls(tuple(samples), sample_rate)

    def write_wav(self, path) -> None:
        """Write a 16-bit mono WAV file (no third-party dependency)."""
        import wave

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(target), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes(self.to_bytes())

    @classmethod
    def read_wav(cls, path) -> AudioClip:
        """Read a WAV file of any common shape, down-mixing it to mono 16-bit.

        A speech engine does not always hand back 16-bit mono: the Windows OneCore
        synthesizer can emit 8-bit or stereo audio, and refusing it silently turned
        a real reading into silence. The decode below accepts 8/16/32-bit samples
        and any channel count, so the clip is read instead of discarded.
        """
        import wave

        with wave.open(str(Path(path)), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
        return cls(_pcm_to_mono16(frames, channels, width), sample_rate=rate)


@runtime_checkable
class SpeechRecognizer(Protocol):
    """Turns an audio clip into text (Arabic ASR)."""

    name: str

    def transcribe(self, clip: AudioClip) -> str: ...


@runtime_checkable
class SpeechSynthesizer(Protocol):
    """Turns text into audio (Arabic TTS)."""

    name: str

    def synthesize(self, text: str) -> AudioClip: ...


class SilentSynthesizer:
    """A TTS stand-in that produces a silent clip sized to the text length.

    Keeps the voice loop runnable and testable when no engine is installed; the
    duration estimate is deliberately crude (about 12 characters per second).
    """

    name = "silent"

    def synthesize(self, text: str) -> AudioClip:
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")
        seconds = max(0.2, len(text) / 12)
        return AudioClip.silence(seconds)


# System TTS engines tried in order: (name, executable, argv builder, voice-option
# builder). Each argv builder returns the argv that renders ``text`` to the WAV
# file ``path``; the voice-option builder returns the extra engine flags for a
# chosen :class:`~core.voices.Voice` (empty when the engine has no such control).
# These are ordinary programs already present on many desktops, so no pip package
# and no network access is needed; when none exists the silent synthesizer is used.
def _espeak_argv(executable: str, text: str, path: Path) -> list[str]:
    return [executable, "-w", str(path), text]


def _flite_argv(executable: str, text: str, path: Path) -> list[str]:
    return [executable, "-t", text, "-o", str(path)]


def _say_argv(executable: str, text: str, path: Path) -> list[str]:
    # macOS: ``say -o file.wav`` renders a WAVE file.
    return [executable, "-o", str(path), text]


def _espeak_voice_options(voice: SelectableVoice | None, language: str) -> tuple[str, ...]:
    """The ``espeak-ng`` flags that select a catalog voice.

    The catalog voice's own :meth:`~core.voices.Voice.espeak_options` carry the
    timbre, pitch, and speed *and* the two settings that keep the reading natural
    rather than mechanical: the intonation range (``-P``) and the word gap (``-g``).

    ``espeak-ng`` has no named voices, so a :class:`~core.voices.SystemVoice` (a
    voice the OS installed) or a :class:`~core.voices.ModelVoice` (a voice of a
    local model) cannot be selected by name here; the language is still honoured,
    which is the honest best it can do.
    """
    if voice is None:
        return ()
    if isinstance(voice, NAMED_VOICE_TYPES):
        return ()
    return voice.espeak_options(language)


#: Cache of ``language -> installed macOS voice name`` (``None`` when none), so the
#: voice list is read at most once per language.
_macos_voice_cache: dict[str, str | None] = {}


def _macos_language_voice(language: str) -> str | None:
    """The name of an installed macOS ``say`` voice for ``language``, or ``None``.

    ``say`` lists its voices as ``Name   <locale>   # example``; the first voice
    whose locale matches ``language`` (e.g. ``ar``) is returned. ``None`` means no
    such voice is installed here, so the caller leaves the engine default in place
    rather than passing a name ``say`` would reject.
    """
    if language in _macos_voice_cache:
        return _macos_voice_cache[language]
    name: str | None = None
    try:
        completed = subprocess.run(
            ["say", "-v", "?"], capture_output=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        completed = None
    if completed is not None and completed.returncode == 0:
        for line in completed.stdout.decode("utf-8", "replace").splitlines():
            match = re.match(r"^(.+?)\s+([a-z]{2})_[A-Z]{2}\s", line)
            if match and match.group(2) == language:
                name = match.group(1).strip()
                break
    _macos_voice_cache[language] = name
    return name


def _say_voice_options(voice: SelectableVoice | None, language: str) -> tuple[str, ...]:
    """The ``say`` flags for a chosen voice and language.

    ``say`` has no gender variant, so only the speaking rate carries over from the
    catalog. It also has no automatic language detection: without ``-v`` it reads
    with the system default voice, which mangles Arabic. A :class:`~core.voices.SystemVoice`
    is selected by its own exact name (``-v <name>``), which is how the real
    installed voice the user picked is used; otherwise an installed voice for the
    language is chosen, and if none is installed the engine default is left in
    place.
    """
    options: list[str] = []
    if isinstance(voice, SystemVoice):
        return ("-v", voice.name)
    if isinstance(voice, ModelVoice):
        # ``say`` cannot speak a local model's voice; leave it to the engine default.
        return ()
    if language and language != "en":
        name = _macos_language_voice(language)
        if name:
            options += ["-v", name]
    if voice is not None:
        options += ["-r", str(voice.speed)]
    return tuple(options)


def _flite_voice_options(_voice: Voice | None, _language: str) -> tuple[str, ...]:
    """``flite`` exposes no voice controls here, so a chosen voice is engine-default."""
    return ()


#: Windows ships its own speech engine (SAPI, reached through ``System.Speech``);
#: this is the name the project gives it, and the shell that drives it.
_WINDOWS_SAPI = "windows-sapi"
#: The WinRT speech engine (``Windows.Media.SpeechSynthesis``). On Windows 10/11
#: the voices added under Settings -> Speech -> Add voices are *OneCore* voices,
#: which the legacy ``System.Speech`` engine cannot list at all; this engine can.
_WINDOWS_ONE_CORE = "windows-onecore"
_WINDOWS_SHELL = "powershell"

#: How long a Windows synthesis run may take. A long Arabic answer is a long
#: utterance, and the old 60 s cap could cut it off mid-sentence.
_WINDOWS_TTS_TIMEOUT = 180


def _ps_quote(value: str) -> str:
    """Escape ``value`` for a single-quoted Windows PowerShell string literal."""
    return value.replace("'", "''")


def _voice_gender(voice: SelectableVoice | None) -> str | None:
    """``"Male"``/``"Female"`` for a chosen voice, or ``None`` when unspecified.

    A :class:`~core.voices.SystemVoice` and a :class:`~core.voices.ModelVoice` both
    carry the gender they were reported with, which is exact; a catalog
    :class:`~core.voices.Voice` derives it from its variant letter.
    """
    if voice is None:
        return None
    if isinstance(voice, NAMED_VOICE_TYPES):
        gender = voice.gender.strip().capitalize()
        return gender if gender in ("Male", "Female") else None
    return "Male" if voice.variant[:1].lower() == "m" else "Female"


def _sapi_rate(voice: SelectableVoice | None, prosody: Prosody | None = None) -> int:
    """Map a catalog voice's speed onto SAPI's ``-10..10`` rate (175 wpm is neutral).

    A chosen :class:`~core.voice_moods.Prosody` shifts the rate: its ``speed_scale``
    multiplies the voice's own pace before the mapping, so a mood that hurries or
    slows the delivery does so on top of the voice the user picked.
    """
    if voice is None and prosody is None:
        return 0
    if isinstance(voice, NAMED_VOICE_TYPES):
        # An installed voice keeps its own neutral rate; only a mood may move it.
        base = 0
    else:
        speed = voice.speed if voice is not None else 175
        if prosody is not None:
            speed = round(speed * prosody.speed_scale)
        base = max(-10, min(10, round((speed - 175) / 20)))
    if prosody is not None:
        base = max(-10, min(10, base + round((prosody.speed_scale - 1.0) * 10)))
    return base


def _sapi_select_voice(voice: SelectableVoice | None, language: str | None = None) -> str:
    """A PowerShell fragment that picks a SAPI voice for the language and gender.

    SAPI voices are named and installed by the OS. A :class:`~core.voices.SystemVoice`
    is selected by its **exact name** -- the whole point of listing the real
    voices, so the one the user picked is the one that speaks. A catalog
    :class:`~core.voices.Voice` cannot name an exact timbre, so it maps as far as
    SAPI allows: a non-English ``language`` first narrows the candidates to voices
    whose culture matches (so Arabic text is read by an Arabic voice when one is
    installed), then the variant's letter picks a male or female voice and the
    trailing number picks among them. When no voice of that language or gender
    exists the engine keeps its default rather than failing.
    """
    if isinstance(voice, SystemVoice):
        return f"$s.SelectVoice('{_ps_quote(voice.name)}');"
    if isinstance(voice, ModelVoice):
        # A local model's voice is not a SAPI voice; keep the engine default rather
        # than asking SAPI for a name it does not have.
        return ""
    if voice is None and not (language and language != "en"):
        return ""
    script = ["$v=@($s.GetInstalledVoices()|?{$_.Enabled});"]
    if language and language != "en":
        script.append(
            "$c=@($v|?{$_.VoiceInfo.Culture.TwoLetterISOLanguageName -eq '"
            + _ps_quote(language)
            + "'});if($c.Count -gt 0){$v=$c};"
        )
    if voice is not None:
        gender = _voice_gender(voice)
        index = _variant_index(voice)
        script.append(
            f"$g=@($v|?{{$_.VoiceInfo.Gender -eq '{gender}'}});"
            "if($g.Count -eq 0){$g=$v};"
            f"if($g.Count -gt 0){{$s.SelectVoice($g[{index} % $g.Count].VoiceInfo.Name)}};"
        )
    else:
        script.append("if($v.Count -gt 0){$s.SelectVoice($v[0].VoiceInfo.Name)};")
    return "".join(script)


def _sapi_language_probe_script(code: str) -> str:
    """PowerShell that prints ``yes`` when SAPI has an enabled voice for ``code``."""
    return (
        "$ErrorActionPreference='Continue';"
        "try{"
        "Add-Type -AssemblyName System.Speech;"
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "if(@($s.GetInstalledVoices()|?{$_.Enabled -and "
        f"$_.VoiceInfo.Culture.TwoLetterISOLanguageName -eq '{_ps_quote(code)}'}}).Count -gt 0)"
        "{'yes'}"
        "}catch{}"
    )


def _sapi_serves_language(language: str) -> bool:
    """True when a SAPI voice for ``language`` is installed and enabled here.

    Asks SAPI itself, because the voice store it *can* see is the only thing that
    decides whether handing it the text would produce the right voice or a mangled
    reading by a mismatched one.
    """
    code = language.strip().lower().split("-")[0][:2]
    try:
        completed = subprocess.run(
            [
                _WINDOWS_SHELL,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                _sapi_language_probe_script(code),
            ],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and b"yes" in completed.stdout


def _windows_sapi_script(
    text_path: Path,
    target: Path,
    voice: SelectableVoice | None,
    language: str | None = None,
    prosody: Prosody | None = None,
) -> str:
    """The one-line PowerShell that renders a text file to the WAV file ``target``.

    ``language`` (e.g. ``"ar"``) selects a SAPI voice whose culture matches, so
    Arabic is read as Arabic; ``None`` leaves the engine's own choice in place. The
    text is read from ``text_path`` as UTF-8 rather than embedded in the command,
    so a long Arabic answer is neither truncated by the command-line limit nor
    dependent on how the shell would have encoded it. ``prosody`` (a chosen mood)
    shifts the rate, the one knob SAPI exposes.
    """
    script = [
        "$ErrorActionPreference='Stop';",
        "Add-Type -AssemblyName System.Speech;",
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;",
        f"$s.Rate={_sapi_rate(voice, prosody)};",
    ]
    selection = _sapi_select_voice(voice, language)
    if selection:
        script.append(selection)
    script.append(
        f"$text=[System.IO.File]::ReadAllText('{_ps_quote(str(text_path))}',"
        "[System.Text.Encoding]::UTF8);"
    )
    script.append(f"$s.SetOutputToWaveFile('{_ps_quote(str(target))}');")
    script.append("$s.Speak($text);")
    script.append("$s.Dispose()")
    return " ".join(script)


def _variant_index(voice: SelectableVoice | None) -> int:
    """The catalog variant's trailing number as a zero-based index (``m3`` -> 2)."""
    if voice is None:
        return 0
    if isinstance(voice, NAMED_VOICE_TYPES):
        return 0
    digits = "".join(character for character in voice.variant if character.isdigit())
    return max(0, int(digits) - 1) if digits else 0


def _onecore_rate(voice: SelectableVoice | None, prosody: Prosody | None = None) -> float:
    """Map a catalog voice's speed onto a WinRT SSML ``prosody`` rate multiplier.

    175 wpm is the neutral reference the SAPI path uses; the multiplier is clamped
    so an extreme catalog entry cannot ask the engine for an unreadable speed. A
    real installed voice keeps the engine's own neutral rate, but a chosen
    :class:`~core.voice_moods.Prosody` moves it -- that is the only way a mood
    changes an installed voice's pace.
    """
    if voice is None and prosody is None:
        return 1.0
    if isinstance(voice, NAMED_VOICE_TYPES):
        base = 1.0
    else:
        base = round((voice.speed if voice is not None else 175) / 175, 2)
    if prosody is not None:
        base *= prosody.speed_scale
    return max(0.5, min(2.0, round(base, 2)))


def _onecore_pitch(voice: SelectableVoice | None, prosody: Prosody | None = None) -> str:
    """Map a catalog voice's pitch onto an SSML ``prosody`` pitch offset.

    The WinRT/SAPI engines carry their own sentence intonation, so the melody
    cannot be set directly; what can be set is the voice's register. A catalog
    voice's pitch (0..99, neutral 50) becomes a percentage offset from neutral, so
    the four voices in a category stay clearly distinct instead of all sharing the
    engine's one default register. An installed voice keeps its own register, but a
    chosen :class:`~core.voice_moods.Prosody` shifts it, so a sad or joyful reading
    still reads as such on an installed voice.
    """
    if voice is None and prosody is None:
        return "0%"
    if isinstance(voice, NAMED_VOICE_TYPES):
        offset = 0
    else:
        offset = (voice.pitch - 50) if voice is not None else 0
    if prosody is not None:
        offset += prosody.pitch_delta
    return f"{max(-50, min(50, offset)):+d}%"


def _windows_onecore_script(
    text_path: Path,
    target: Path,
    voice: SelectableVoice | None,
    language: str | None,
    prosody: Prosody | None = None,
) -> str:
    """PowerShell that renders ``text_path`` to the WAV ``target`` with a WinRT voice.

    Why this exists at all: on Windows 10/11 the voices a user adds under
    *Settings -> Time & Language -> Speech -> Add voices* are **OneCore** voices.
    The legacy ``System.Speech`` engine the project used first cannot see them --
    ``GetInstalledVoices()`` simply omits them -- so an Arabic voice that the OS
    offers to every other app was invisible here and Arabic came out mangled by an
    English voice. ``Windows.Media.SpeechSynthesis`` *can* see every installed
    voice, so this script is what actually reads Arabic correctly.

    ``text_path`` is read as UTF-8 so a long Arabic answer is neither truncated by
    the command-line limit nor mangled by console encoding. A chosen **installed**
    voice is spoken through the documented path -- set the ``Voice`` property and
    synthesise plain text -- because the WinRT engine takes the voice from any SSML
    it is given, so a bare ``xml:lang`` there overrode the property and read every
    Arabic entry with the language's *default* voice (a woman's entry and a man's
    entry both came out as the same male voice). Plain text leaves the property
    authoritative, and an installed voice keeps its own neutral rate, so no SSML is
    needed. A catalog voice still uses SSML, where the rate matters, with the picked
    voice named in the markup *and* its own language in ``xml:lang`` so the markup
    cannot fall back to the language default either. For a catalog voice the SSML
    also carries the voice's ``rate`` and ``pitch``, so the four voices of a
    category keep distinct speeds and registers instead of one engine default.
    """
    language = language or "en"
    gender = _voice_gender(voice)
    index = _variant_index(voice)
    rate = _onecore_rate(voice, prosody)
    pitch = _onecore_pitch(voice, prosody)
    # An installed voice is spoken as plain text (the Voice property is authoritative
    # and its rate is already neutral); a catalog voice needs SSML for the rate. A
    # chosen mood also needs SSML on an installed voice, because the rate/pitch it
    # changes live only in the markup.
    use_ssml = prosody is not None or not isinstance(voice, NAMED_VOICE_TYPES)
    selection: list[str] = [
        "$voices=@([Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices);",
        "$pick=$null;",
    ]
    if isinstance(voice, SystemVoice):
        # The exact installed voice the user picked, by the name the OS reports.
        selection.append(
            "$pick=@($voices|?{$_.DisplayName -eq '"
            + _ps_quote(voice.name)
            + "'})|Select-Object -First 1;"
        )
    else:
        if language and language != "en":
            # Narrow to the spoken language first, so Arabic is read by an Arabic
            # voice; when none is installed the engine default is kept (honest).
            selection.append(
                "$m=@($voices|?{$_.Language -like '" + _ps_quote(language) + "*'});"
                "if($m.Count -gt 0){$voices=$m};"
            )
        if gender is not None:
            selection.append(
                f"$g=@($voices|?{{$_.Gender.ToString() -eq '{gender}'}});"
                "if($g.Count -gt 0){$voices=$g};"
            )
        selection.append(f"if($voices.Count -gt 0){{$pick=$voices[{index} % $voices.Count]}};")
    script = [
        "$ErrorActionPreference='Stop';",
        "Add-Type -AssemblyName System.Runtime.WindowsRuntime;",
        "[void][Windows.Media.SpeechSynthesis.SpeechSynthesizer,Windows.Media.SpeechSynthesis,"
        "ContentType=WindowsRuntime];",
        "[void][Windows.Media.SpeechSynthesis.SpeechSynthesisStream,Windows.Media.SpeechSynthesis,"
        "ContentType=WindowsRuntime];",
        "$asTask=([System.WindowsRuntimeSystemExtensions].GetMethods()|?"
        "{$_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and "
        "$_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'})[0];",
        "function Await($t,$r){$m=$asTask.MakeGenericMethod($r);"
        "$n=$m.Invoke($null,@($t));$n.Wait(-1)|Out-Null;$n.Result};",
        f"$text=[System.IO.File]::ReadAllText('{_ps_quote(str(text_path))}',"
        "[System.Text.Encoding]::UTF8);",
        "$speak=[Windows.Media.SpeechSynthesis.SpeechSynthesizer]::new();",
        *selection,
        "if($pick){$speak.Voice=$pick};",
    ]
    if use_ssml:
        script += [
            "$esc=[System.Security.SecurityElement]::Escape($text);",
            "$lang='" + _ps_quote(language) + "';$open='';$close='';",
            "if($pick){$lang=$pick.Language;"
            "$open='<voice name=\"'+[System.Security.SecurityElement]::Escape($pick.DisplayName)+'\">';"
            "$close='</voice>'};",
            "$ssml='<speak version=\"1.0\" "
            "xmlns=\"http://www.w3.org/2001/10/synthesis\" "
            "xml:lang=\"'+$lang+'\">'+$open+'<prosody rate=\""
            f"{rate}"
            "\" pitch=\""
            f"{pitch}"
            "\">'+$esc+'</prosody>'+$close+'</speak>';",
            "try{$stream=Await ($speak.SynthesizeSsmlToStreamAsync($ssml)) "
            "([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])}"
            "catch{$stream=Await ($speak.SynthesizeTextToStreamAsync($text)) "
            "([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])};",
        ]
    else:
        # Plain text: nothing in markup can override the Voice property we just set.
        script.append(
            "$stream=Await ($speak.SynthesizeTextToStreamAsync($text)) "
            "([Windows.Media.SpeechSynthesis.SpeechSynthesisStream]);"
        )
    script += [
        "$src=[System.IO.WindowsRuntimeStreamExtensions]::AsStreamForRead($stream);",
        f"$out=[System.IO.File]::Create('{_ps_quote(str(target))}');",
        "$src.CopyTo($out);$out.Dispose();$src.Dispose();$stream.Dispose();$speak.Dispose();",
    ]
    return " ".join(script)


def _windows_recorder_script(
    target: Path, marker: Path, sample_rate: int, seconds: float
) -> str:
    """PowerShell that records the microphone to the WAV ``target`` until stopped.

    Why this exists at all: Windows ships no ``arecord``/``sox``, so a Windows user
    who clicks the microphone icon had **nothing** to capture with -- the button
    simply reported "install arecord or sox", a Linux-only instruction. Windows
    does have its own capture through the MCI ``waveaudio`` device (``winmm.dll``),
    reachable from PowerShell with no extra install, so the icon records out of the
    box the same way read-aloud already does.

    ``marker`` is the stop signal: the caller creates that file (a second click on
    the icon) and the loop below then stops and **saves** the clip. A marker file
    is used rather than a signal because MCI finalises the WAV in the ``save``
    step -- killing the process (what ``SIGINT`` does to ``arecord``) would leave
    an unusable file. ``seconds`` is the same safety bound ``arecord -d`` has, so
    a forgotten recording cannot grow without end.
    """
    alias = "faydmic"
    deadline = f"(Get-Date).AddSeconds({float(seconds):.3f})"
    return " ".join(
        [
            "$ErrorActionPreference='Stop';",
            # One P/Invoke into winmm.dll; the MCI command language then opens the
            # default recording device, records, and writes a WAV on `save`.
            "Add-Type -MemberDefinition "
            "'[DllImport(\"winmm.dll\", CharSet=CharSet.Unicode)] "
            "public static extern int mciSendString(string c, System.Text.StringBuilder r, "
            "int l, System.IntPtr h);' -Name WinMM -Namespace Fayd;",
            f"$target='{_ps_quote(str(target))}';",
            f"$marker='{_ps_quote(str(marker))}';",
            f"$alias='{alias}';",
            f"$rate={int(sample_rate)};",
            f"$stopAt={deadline};",
            "[Fayd.WinMM]::mciSendString('open new type waveaudio alias '+$alias,"
            "$null,0,[IntPtr]::Zero)|Out-Null;",
            # Ask for mono 16-bit at our sample rate; a driver that refuses keeps its
            # own format, and AudioClip.read_wav decodes whatever comes back.
            "[Fayd.WinMM]::mciSendString('set '+$alias+' time format ms bitspersample 16 "
            "channels 1 samplespersec '+$rate,$null,0,[IntPtr]::Zero)|Out-Null;",
            "[Fayd.WinMM]::mciSendString('record '+$alias,$null,0,[IntPtr]::Zero)|Out-Null;",
            "while(-not (Test-Path -LiteralPath $marker) -and (Get-Date) -lt $stopAt)"
            "{Start-Sleep -Milliseconds 100};",
            "[Fayd.WinMM]::mciSendString('stop '+$alias,$null,0,[IntPtr]::Zero)|Out-Null;",
            # MCI wants the path quoted (`save <alias> "<file>"`); a temp folder can
            # contain spaces, so the quotes are built explicitly.
            "$save='save '+$alias+' \"'+$target+'\"';",
            "[Fayd.WinMM]::mciSendString($save,$null,0,[IntPtr]::Zero)|Out-Null;",
            "[Fayd.WinMM]::mciSendString('close '+$alias,$null,0,[IntPtr]::Zero)|Out-Null",
        ]
    )


def _windows_shell_available() -> bool:
    """True when Windows PowerShell is on PATH (the shell both Windows paths drive)."""
    return platform.system().lower() == "windows" and shutil.which(_WINDOWS_SHELL) is not None


def windows_speech_available() -> bool:
    """True when Windows' built-in speech engine can be driven here (SAPI/PowerShell)."""
    return _windows_shell_available()


#: Cache of the WinRT/OneCore availability probe (``None`` = not probed yet).
_onecore_cache: list[bool | None] = [None]


def refresh_windows_speech_caches() -> None:
    """Forget the cached Windows engine/voice probes, so a new voice is seen.

    Windows only reveals a voice added through *Settings -> Add voices* after the
    caches are dropped; :meth:`core.reading.TextReader.refresh` calls this, which is
    how the window's *Install a speech engine* flow picks up a newly added voice
    without a restart.
    """
    _onecore_cache[0] = None
    _windows_voices_cache[0] = None
    _system_voices_cache[0] = None


def windows_onecore_available(*, refresh: bool = False) -> bool:
    """True when the WinRT speech engine can be reached here (Windows 10/11).

    The legacy ``System.Speech`` engine cannot list the OneCore voices Windows
    installs through *Settings -> Add voices*, which is exactly how an Arabic
    voice is added on Windows 11. This probes ``Windows.Media.SpeechSynthesis``
    once and caches the verdict, so a chat window does not run PowerShell on every
    answer. A machine where the probe fails simply keeps the SAPI path.
    """
    if not windows_speech_available():
        return False
    if _onecore_cache[0] is not None and not refresh:
        return bool(_onecore_cache[0])
    # Deliberately tolerant: the WinRT projection can emit a non-terminating
    # warning (a missing optional assembly, a PowerShell 7 host, a locked store),
    # and with ``$ErrorActionPreference='Stop'`` any of those would abort the probe
    # and hide the OneCore voices -- including the Arabic one added through
    # Settings -> Add voices. A try/catch that only prints on success means a
    # transient warning degrades to "SAPI only" rather than a false negative.
    script = (
        "$ErrorActionPreference='Continue';"
        "try{"
        "Add-Type -AssemblyName System.Runtime.WindowsRuntime;"
        "[void][Windows.Media.SpeechSynthesis.SpeechSynthesizer,Windows.Media.SpeechSynthesis,"
        "ContentType=WindowsRuntime];"
        "if([Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices.Count -gt 0){'ok'}"
        "}catch{}"
    )
    try:
        completed = subprocess.run(
            [_WINDOWS_SHELL, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        _onecore_cache[0] = False
        return False
    available = completed.returncode == 0 and b"ok" in completed.stdout
    _onecore_cache[0] = available
    return available


@dataclass(frozen=True)
class WindowsVoice:
    """One voice the Windows speech engine reports (name, language tag, gender).

    ``engine`` names the Windows engine that reported it (``windows-onecore`` or
    ``windows-sapi``). The two engines keep *separate* voice stores, so the same
    machine can offer a voice to one and not the other; recording which engine saw
    it is what lets the synthesizer be pointed at the right one.
    """

    name: str
    language: str
    gender: str
    engine: str = ""

    @property
    def arabic(self) -> bool:
        """True when this voice speaks Arabic (a language tag such as ``ar-SA``)."""
        return self.language.strip().lower().startswith("ar")


#: Cache of the installed Windows voices (``None`` = not listed yet).
_windows_voices_cache: list[tuple[WindowsVoice, ...] | None] = [None]


def _parse_voice_lines(output: bytes, *, engine: str = "") -> tuple[WindowsVoice, ...]:
    """Parse ``name|language|gender`` lines emitted by a listing script.

    The gender is optional -- a listing that omits it (or leaves it blank) still
    yields a usable voice, because the name and the language are what matter for
    selecting and grouping it. A line without a name is skipped.
    """
    voices: list[WindowsVoice] = []
    for line in output.decode("utf-8", "replace").splitlines():
        parts = [part.strip() for part in line.strip().split("|")]
        if len(parts) >= 2 and parts[0]:
            gender = parts[2] if len(parts) > 2 else ""
            voices.append(WindowsVoice(parts[0], parts[1], gender, engine=engine))
    return tuple(voices)


def _list_windows_voices(script: str, engine: str) -> tuple[WindowsVoice, ...]:
    """Run one listing ``script`` and parse its voices, or ``()`` on any failure."""
    try:
        completed = subprocess.run(
            [_WINDOWS_SHELL, "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if completed.returncode != 0:
        return ()
    return _parse_voice_lines(completed.stdout, engine=engine)


#: The WinRT listing: every voice Windows installs, including the OneCore store.
#: Tolerant on purpose -- a non-terminating warning must not abort the listing and
#: hide the Arabic voice the user installed.
_ONECORE_LIST_SCRIPT = (
    "$ErrorActionPreference='Continue';"
    "try{"
    "Add-Type -AssemblyName System.Runtime.WindowsRuntime;"
    "[void][Windows.Media.SpeechSynthesis.SpeechSynthesizer,Windows.Media.SpeechSynthesis,"
    "ContentType=WindowsRuntime];"
    "[Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices|%{"
    "$_.DisplayName+'|'+$_.Language+'|'+$_.Gender}"
    "}catch{}"
)

#: The legacy listing: only what ``System.Speech`` can see (never OneCore voices).
_SAPI_LIST_SCRIPT = (
    "$ErrorActionPreference='Continue';"
    "try{"
    "Add-Type -AssemblyName System.Speech;"
    "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
    "$s.GetInstalledVoices()|%{$_.VoiceInfo}|%{$_.Name+'|'+$_.Culture.Name+'|'+$_.Gender}"
    "}catch{}"
)


def windows_installed_voices(*, refresh: bool = False) -> tuple[WindowsVoice, ...]:
    """The voices Windows actually offers, as the **union** of both engines.

    The two Windows engines keep separate voice stores, and neither is a superset
    of the other: the OneCore (WinRT) store holds the voices added under *Settings
    -> Speech -> Add voices* (how an Arabic voice is added on Windows 11) that SAPI
    cannot see at all, while SAPI can still expose voices the WinRT probe happens
    not to reach. So both are asked and the results merged -- OneCore first, so a
    voice present in both keeps the modern engine -- rather than picking one engine
    and silently missing whatever only the other knows. The WinRT listing is always
    attempted (its script is tolerant, so an unavailable store simply yields no
    lines) rather than gated on the availability probe, because a false-negative
    probe would otherwise hide the Arabic voice. Empty on a non-Windows machine, so
    the caller can report honestly instead of pretending.
    """
    if not windows_speech_available():
        return ()
    if _windows_voices_cache[0] is not None and not refresh:
        return _windows_voices_cache[0]
    merged: list[WindowsVoice] = []
    seen: set[tuple[str, str]] = set()
    # Always attempt the WinRT listing: the script is tolerant (a failure yields no
    # lines, not an error), and it is the only store that holds the voices added
    # through Settings -> Add voices. Gating it on the availability probe meant a
    # false-negative probe silently hid the Arabic voice; letting the listing itself
    # decide recovers it whenever the store really is reachable.
    for voice in _list_windows_voices(_ONECORE_LIST_SCRIPT, _WINDOWS_ONE_CORE):
        key = (voice.name, voice.language)
        if key not in seen:
            seen.add(key)
            merged.append(voice)
    for voice in _list_windows_voices(_SAPI_LIST_SCRIPT, _WINDOWS_SAPI):
        key = (voice.name, voice.language)
        if key not in seen:
            seen.add(key)
            merged.append(voice)
    voices = tuple(merged)
    _windows_voices_cache[0] = voices
    return voices


def windows_onecore_usable() -> bool:
    """True when the WinRT engine can be driven here.

    The availability probe is tried first, but its verdict is not trusted on its
    own: a listing that actually returned a OneCore voice is proof the engine works,
    so a false-negative probe cannot refuse the one engine that can see an Arabic
    voice added through *Settings -> Add voices*. Both calls are cached.
    """
    if windows_onecore_available():
        return True
    return any(voice.engine == _WINDOWS_ONE_CORE for voice in windows_installed_voices())


def windows_arabic_engine() -> str | None:
    """The Windows engine that can actually see an Arabic voice, or ``None``.

    Both voice stores are listed rather than the availability probe, so this
    reflects what the machine really has. It is what keeps an all-OneCore machine
    (Arabic added through *Settings -> Add voices*) from being read by SAPI, which
    cannot see that voice and would render Arabic with an English one instead --
    audibly, but mangled.
    """
    for engine in (_WINDOWS_ONE_CORE, _WINDOWS_SAPI):
        if any(voice.arabic for voice in windows_installed_voices() if voice.engine == engine):
            return engine
    return None


def windows_voice_report() -> str:
    """An honest, human-readable note about Arabic speech on this Windows machine.

    It exists because the failure this fixes was silent: a voice the OS offered to
    every other app was invisible to the engine the project used, so Arabic came
    out mangled. The report says which engine is used, whether an Arabic voice is
    actually visible to it, and -- when none is -- the exact step to install one.
    Empty on a non-Windows machine, so a caller can append it unconditionally.
    """
    if platform.system().lower() != "windows":
        return ""
    if not windows_speech_available():
        return "No Windows speech engine is available (PowerShell was not found)."
    voices = windows_installed_voices()
    arabic = [voice for voice in voices if voice.arabic]
    engines = sorted({voice.engine for voice in voices if voice.engine})
    seen = " and ".join(engines) if engines else _WINDOWS_SAPI
    if arabic:
        names = ", ".join(voice.name for voice in arabic[:3])
        return (
            f"Windows speech engine {seen} sees {len(voices)} voice(s), "
            f"including Arabic: {names}."
        )
    note = (
        f"Windows speech engine {seen} sees {len(voices)} voice(s) but no Arabic one, "
        "so Arabic would be read by a non-Arabic voice. Add one in Settings -> Time & "
        "Language -> Speech -> Add voices -> Arabic, then reopen this window."
    )
    if not any(voice.engine == _WINDOWS_ONE_CORE for voice in voices):
        note += (
            " (The WinRT/OneCore voice store listed no voices, so voices added "
            "through Settings -> Add voices may be missing from this list.)"
        )
    return note


#: Cache of the real installed voices, cross-platform (``None`` = not listed yet).
_system_voices_cache: list[tuple[SystemVoice, ...] | None] = [None]


def _macos_system_voices() -> tuple[SystemVoice, ...]:
    """The real voices macOS ``say`` reports, parsed from ``say -v ?``.

    ``say`` prints ``Name              locale    # example`` per voice; the name and
    the locale are all that is needed to list and then select a voice by name.
    """
    try:
        completed = subprocess.run(
            ["say", "-v", "?"], capture_output=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if completed.returncode != 0:
        return ()
    voices: list[SystemVoice] = []
    for line in completed.stdout.decode("utf-8", "replace").splitlines():
        match = re.match(r"^(.+?)\s+([a-z]{2}[-_][A-Z]{2})\s", line)
        if match:
            voices.append(
                SystemVoice(match.group(1).strip(), match.group(2).replace("_", "-"), engine="say")
            )
    return tuple(voices)


def system_voices(*, refresh: bool = False) -> tuple[SystemVoice, ...]:
    """The **real** voices installed on this machine, or ``()`` when none are known.

    This is the honest list a menu should prefer: the voices the OS itself reports
    by name (``Microsoft Hoda``, ``Majed``, ...), not the synthetic espeak-ng
    settings. On Windows the OneCore engine is asked first (it sees the voices
    added through *Settings -> Add voices*); macOS is read through ``say -v ?``.
    Other platforms -- and any machine where the query fails -- return ``()``, so a
    caller falls back to the synthetic catalog rather than inventing a voice.
    """
    if _system_voices_cache[0] is not None and not refresh:
        return _system_voices_cache[0]
    system = platform.system().lower()
    voices: tuple[SystemVoice, ...] = ()
    if system == "windows":
        # Each voice records the engine that reported it (the two stores differ),
        # so a chosen voice is selected through the engine that can actually see it.
        voices = tuple(
            SystemVoice(
                voice.name,
                voice.language,
                voice.gender,
                engine=voice.engine or _WINDOWS_SAPI,
            )
            for voice in windows_installed_voices(refresh=refresh)
        )
    elif system == "darwin":
        voices = _macos_system_voices()
    _system_voices_cache[0] = voices
    return voices


TTS_ENGINES: tuple[tuple[str, str, object, object], ...] = (
    ("espeak-ng", "espeak-ng", _espeak_argv, _espeak_voice_options),
    ("espeak", "espeak", _espeak_argv, _espeak_voice_options),
    ("flite", "flite", _flite_argv, _flite_voice_options),
    ("say", "say", _say_argv, _say_voice_options),
)


#: The system package a Debian/Ubuntu machine needs for a real voice engine.
SPEECH_SYSTEM_PACKAGES: dict[str, str] = {
    "debian": "espeak-ng",
    "fedora": "espeak-ng",
    "arch": "espeak-ng",
    "alpine": "espeak-ng",
    "macos": "espeak-ng",  # `brew install espeak-ng`; macOS also has its own `say`
    "windows": "espeak-ng",  # download the Windows build, or use `pyttsx3`
}


@dataclass(frozen=True)
class SpeechInstallPlan:
    """How to get a real speech engine, honest about what pip cannot do.

    A speech engine (``espeak-ng`` and friends) is an ordinary *system* program,
    not a Python package, so ``pip`` cannot install it -- the same distinction
    the OCR path makes for Tesseract. ``command`` is a ready-to-run package-manager
    command on this platform, or empty when the app does not know one (so the
    caller shows ``hint`` instead of pretending it can install).
    """

    platform: str
    command: tuple[str, ...]
    hint: str

    @property
    def can_install(self) -> bool:
        return bool(self.command)


def _package_manager_command() -> tuple[str, ...]:
    """A package-manager command for this OS, or empty when the app knows none."""
    system = platform.system().lower()
    if system == "linux":
        if shutil.which("apt-get"):
            return ("sudo", "apt-get", "install", "-y", SPEECH_SYSTEM_PACKAGES["debian"])
        if shutil.which("dnf"):
            return ("sudo", "dnf", "install", "-y", SPEECH_SYSTEM_PACKAGES["fedora"])
        if shutil.which("pacman"):
            return ("sudo", "pacman", "-S", "--noconfirm", SPEECH_SYSTEM_PACKAGES["arch"])
        if shutil.which("apk"):
            return ("sudo", "apk", "add", SPEECH_SYSTEM_PACKAGES["alpine"])
        return ()
    if system == "darwin" and shutil.which("brew"):
        return ("brew", "install", SPEECH_SYSTEM_PACKAGES["macos"])
    if system == "windows":
        if shutil.which("winget"):
            return ("winget", "install", "-e", "--id", "eSpeak-NG.eSpeak-NG")
        if shutil.which("choco"):
            return ("choco", "install", "-y", SPEECH_SYSTEM_PACKAGES["windows"])
    return ()


def speech_install_plan() -> SpeechInstallPlan:
    """The platform-aware way to install a real speech engine (usually ``espeak-ng``).

    Windows is the special case: it already ships a speech engine (SAPI), so the
    plan reports that there is nothing to install and ``can_install`` stays False,
    with a hint that says so instead of sending the user off to download a build.
    """
    system = platform.system().lower()
    if system == "windows" and windows_speech_available():
        engine = "OneCore" if windows_onecore_available() else "SAPI"
        return SpeechInstallPlan(
            system,
            (),
            f"Windows already provides a speech engine ({engine}), so read-aloud works "
            "without installing anything. For Arabic, add a voice in Settings -> Time & "
            "Language -> Speech -> Add voices -> Arabic; the OneCore engine reads it "
            "directly (the older SAPI engine cannot see voices added that way).",
        )
    command = _package_manager_command()
    if command:
        return SpeechInstallPlan(system, command, " ".join(command))
    if system == "darwin":
        return SpeechInstallPlan(
            system,
            (),
            "Install espeak-ng with Homebrew (`brew install espeak-ng`); macOS `say` "
            "already provides a voice.",
        )
    if system == "windows":
        return SpeechInstallPlan(
            system,
            (),
            "Install espeak-ng with `winget install -e --id eSpeak-NG.eSpeak-NG` (or "
            "`choco install espeak-ng`), then open a new terminal so it is on PATH; "
            "Windows' built-in speech engine is used automatically when PowerShell "
            "is available.",
        )
    return SpeechInstallPlan(
        system,
        (),
        "Install espeak-ng with your system package manager "
        "(e.g. `sudo apt install espeak-ng`).",
    )


def microphone_install_hint() -> str:
    """The platform-aware next step when no microphone recorder exists.

    The old message was ``install arecord or sox`` on every platform, which is a
    Linux-only instruction: Windows ships neither, and macOS has neither by
    default either. This names the real step per platform, and says plainly when
    there is nothing to install because the OS already provides a recorder.
    """
    system = platform.system().lower()
    if system == "windows":
        return (
            "Windows can record through its own audio device (MCI waveaudio); if that "
            "failed, check that a microphone is connected and allowed in "
            "Settings -> Privacy & security -> Microphone."
        )
    if system == "darwin":
        return (
            "Install a recorder with Homebrew (`brew install sox`) to capture audio."
        )
    if shutil.which("apt-get") or shutil.which("dnf") or shutil.which("pacman") or shutil.which("apk"):
        return (
            "Install a recorder such as `sox` or `alsa-utils` (e.g. "
            "`sudo apt install sox`) to capture audio."
        )
    return "Install a microphone recorder such as `sox` to capture audio."


@dataclass(frozen=True)
class SpeechInstallResult:
    """The outcome of an attempted speech-engine install, ready to show the user."""

    ok: bool
    command: tuple[str, ...] = ()
    returncode: int | None = None
    output: str = ""
    message: str = ""
    ran: bool = True


def speech_available() -> bool:
    """Whether any real TTS engine is installed (the same check the reader uses)."""
    return discover_synthesizer_name() != "silent"


def system_engine_available() -> bool:
    """True when a *system* TTS engine (the target of the synthetic catalog) exists.

    The synthetic :data:`core.voices.VOICE_CATALOG` is a set of settings for the
    system engine (``espeak-ng``/``espeak``/``flite``/``say``, or Windows SAPI), not
    for the local model engines, so a menu should offer those voices only when that
    engine is really present -- otherwise they would come out empty.
    """
    for _name, executable, _builder, _voice_options in TTS_ENGINES:
        if shutil.which(executable):
            return True
    return windows_speech_available()


def install_speech_engine(
    *,
    runner: SpeechCommandRunner | None = None,
) -> SpeechInstallResult:
    """Try to install a real speech engine as a system package. Never raises.

    ``pip`` cannot install ``espeak-ng`` (it is an OS package, like Tesseract), so
    this runs the platform's package manager through ``sudo`` when the app knows
    one. If it succeeds and the engine is now on PATH, ``ok`` is True. When the app
    does not know an install command (or the runner fails), ``ran``/``ok`` are
    False and ``message`` carries the manual hint instead of a false promise.
    """
    if speech_available():
        return SpeechInstallResult(
            ok=True, ran=False, message="A speech engine is already installed."
        )
    plan = speech_install_plan()
    if not plan.can_install:
        return SpeechInstallResult(ok=False, ran=False, message=plan.hint)
    run = runner or _default_speech_runner
    returncode, output = run(plan.command)
    ok = returncode == 0 and speech_available()
    if ok:
        message = f"Installed a speech engine ({SPEECH_SYSTEM_PACKAGES['debian']})."
    elif returncode == 0:
        message = (
            "The package manager reported success, but no speech engine is on PATH yet. "
            + plan.hint
        )
    else:
        message = f"The install command failed.\n{plan.hint}"
    return SpeechInstallResult(
        ok=ok,
        command=plan.command,
        returncode=returncode,
        output=output,
        message=message,
    )


def _default_speech_runner(argv: Sequence[str]) -> tuple[int, str]:
    completed = subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        check=False,
    )
    combined = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode, combined


#: Runs one install command and returns ``(returncode, combined_output)``.
SpeechCommandRunner = Callable[[Sequence[str]], tuple[int, str]]


def discover_synthesizer_name() -> str:
    """Return the best installed TTS engine, or ``'silent'``.

    The local **Piper** neural engine comes first when it is fully installed (the
    extra and a voice file), because it sounds like a person rather than the
    robotic formant synthesis of the system engines, and it speaks Arabic. The
    system engines (``espeak``, ``say``, Windows SAPI) come next: they are tiny,
    already installed, and speak Arabic. On Windows the **OneCore** WinRT engine is
    preferred over SAPI when it is reachable, because it is the only one that can
    see the Arabic voice installed through *Settings -> Add voices*. When nothing
    else exists but the local Qwen3-TTS model is fully present, that model is used
    instead, so a machine with the Qwen3 extra still gets a natural voice rather
    than silence.

    The **VoiceTut-TTS over audio.cpp** engine is deliberately *not* part of this
    automatic order. It is the highest-quality Arabic voice available here, but it
    is also much slower on a CPU (seconds per sentence), so making it the silent
    default would slow every reading on a machine that has it. It is reached
    explicitly instead: picking one of its voices in the reading-voice menu routes
    the reading to it through :func:`_engine_for_voice`, so a user who wants it gets
    it and a user who does not is unaffected.
    """
    from core.piper_speech import available_synthesizer as piper_available

    if piper_available():
        return "piper"
    for name, executable, _builder, _voice_options in TTS_ENGINES:
        if shutil.which(executable):
            return name
    if windows_speech_available():
        # One place decides (the engine that can actually see an Arabic voice wins),
        # so discovery and the Qwen3-TTS Arabic fallback can never disagree.
        return windows_engine_name()
    from core.qwen_speech import available_synthesizer

    if available_synthesizer():
        return "qwen3-tts"
    return "silent"


class CommandSynthesizer:
    """A real TTS engine driven through a system command.

    The engine is a normal program (``espeak``, ``say``, ...). We render to a
    temporary WAV, read it back as an :class:`AudioClip`, and never require a pip
    dependency. If the engine fails for any reason, synthesis falls back to a
    silent clip so a voice turn can still complete.

    ``voice`` is an optional :class:`~core.voices.Voice` from the catalog; its
    engine flags are inserted after the executable. ``language`` pins the spoken
    language, and when it is ``None`` the language is taken from the text (Arabic
    script -> ``ar``, otherwise ``en``), so one voice reads both. ``prosody`` (a
    chosen mood) is folded into the voice before its engine flags are built, so the
    timbre, pitch, speed, melody, and pauses all carry the mood.
    """

    def __init__(
        self,
        name: str,
        executable: str,
        builder,
        *,
        voice: SelectableVoice | None = None,
        language: str | None = None,
        voice_options=None,
        prosody: Prosody | None = None,
    ) -> None:
        self.name = name
        self.executable = executable
        self._builder = builder
        self.language = language
        self._voice_options = voice_options
        self.prosody = prosody
        # Fold the mood into a catalog voice *once*, here, so the engine flags
        # carry it. A named voice (an installed/system or model voice) keeps its own
        # timbre -- those engines select it by name -- and the mood's other engines
        # take the prosody directly.
        self.voice = apply_mood(voice, prosody) if prosody is not None else voice

    def _options_for(self, text: str) -> tuple[str, ...]:
        if self.voice is None or self._voice_options is None:
            return ()
        language = self.language or detect_language(text)
        return tuple(self._voice_options(self.voice, language))

    def synthesize(self, text: str) -> AudioClip:
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")
        # Speak the numbers and units as words, not as the symbols this engine
        # cannot say (``25%``, ``1500 psi``, ``10-20``). Presentation only, and a
        # no-op for English, so the engine is handed the same words a listener hears.
        text = prepare_for_speech(text, language=self.language)
        with tempfile.TemporaryDirectory(prefix="fayd-tts-") as folder:
            target = Path(folder) / "speech.wav"
            argv = self._builder(self.executable, text, target)
            options = self._options_for(text)
            if options:
                # Options sit between the executable and its own arguments, which
                # every engine here accepts.
                argv = [argv[0], *options, *argv[1:]]
            try:
                completed = subprocess.run(
                    argv,
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                return SilentSynthesizer().synthesize(text)
            if completed.returncode != 0 or not target.exists():
                return SilentSynthesizer().synthesize(text)
            try:
                return AudioClip.read_wav(target)
            except (OSError, ValueError):
                return SilentSynthesizer().synthesize(text)


class _WindowsPowerShellSynthesizer:
    """Shared plumbing for the two Windows engines (SAPI and WinRT/OneCore).

    Both render a WAV through a short PowerShell script, so the temp-folder life
    cycle, the timeout, the silent fallback, and the honest WAV read live here
    once. The text is written to a UTF-8 temp file and read back by the script
    rather than being embedded in the command line: a long Arabic answer would
    otherwise hit the ~32 KB command-line limit, and console encoding could mangle
    it. A subclass supplies :meth:`_script` and its own name.
    """

    name = _WINDOWS_SAPI
    executable = _WINDOWS_SHELL

    def __init__(
        self,
        *,
        voice: SelectableVoice | None = None,
        language: str | None = None,
        prosody: Prosody | None = None,
    ) -> None:
        self.voice = voice
        self.language = language
        self.prosody = prosody

    def _script(self, text_path: Path, target: Path, language: str) -> str:
        raise NotImplementedError

    def _fallback(self) -> _WindowsPowerShellSynthesizer | None:
        """The sibling Windows engine to try when this one produces no audio.

        The two Windows engines see *different* voice stores: SAPI cannot see the
        OneCore voices installed through *Settings -> Add voices*, so on a machine
        whose only voices are OneCore, SAPI writes an empty WAV. Trying the sibling
        turns that silence into speech instead of a blank reading.
        """
        return None

    def _language_in_sapi(self, language: str) -> bool:
        """True when SAPI has an enabled voice for ``language`` on this machine."""
        return _sapi_serves_language(language)

    def _render(self, text: str, language: str) -> AudioClip | None:
        """Render ``text`` to a clip, or ``None`` when the engine produced nothing."""
        with tempfile.TemporaryDirectory(prefix="fayd-tts-") as folder:
            target = Path(folder) / "speech.wav"
            text_path = Path(folder) / "text.txt"
            text_path.write_text(text, encoding="utf-8")
            script = self._script(text_path, target, language)
            argv = [
                self.executable,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ]
            try:
                completed = subprocess.run(
                    argv,
                    capture_output=True,
                    timeout=_WINDOWS_TTS_TIMEOUT,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                return None
            if completed.returncode != 0 or not target.exists():
                return None
            try:
                return AudioClip.read_wav(target)
            except (OSError, ValueError):
                return None

    def synthesize(self, text: str) -> AudioClip:
        if not text.strip():
            raise ValueError("Cannot synthesize empty text.")
        language = self.language or detect_language(text)
        # Speak the numbers and units as words rather than the symbols the engine
        # cannot say; a no-op for English, so only Arabic text changes.
        text = prepare_for_speech(text, language=language)
        clip = self._render(text, language)
        if clip is not None and clip.samples:
            return clip
        # No audio: the selected engine has no usable voice (all OneCore voices on
        # Windows 11 look empty to SAPI). Try the sibling engine before falling back
        # to silence, so the reading is not silently blank.
        fallback = self._fallback()
        if fallback is not None:
            other = fallback._render(text, language)
            if other is not None and other.samples:
                return other
        return SilentSynthesizer().synthesize(text)


class WindowsSapiSynthesizer(_WindowsPowerShellSynthesizer):
    """A real TTS engine using the legacy speech engine Windows ships (SAPI).

    Windows bundles SAPI, reachable from ``System.Speech`` through PowerShell, so a
    Windows user does **not** need to install ``espeak-ng`` at all. It renders
    ``text`` to a WAV with ``SetOutputToWaveFile``/``Speak`` and reads it back, so
    the same :class:`~core.reading.TextReader` works unchanged. A chosen catalog
    voice maps as far as SAPI allows (language culture + gender variant + rate);
    Arabic is read by an installed Arabic voice when one exists.

    Caveat, and why :class:`WindowsOneCoreSynthesizer` exists: this engine cannot
    see the OneCore voices Windows installs through *Settings -> Add voices*, so on
    Windows 11 an Arabic voice added that way is invisible here.
    """

    name = _WINDOWS_SAPI

    def _script(self, text_path: Path, target: Path, language: str) -> str:
        return _windows_sapi_script(text_path, target, self.voice, language, self.prosody)

    def _fallback(self) -> _WindowsPowerShellSynthesizer | None:
        # SAPI cannot see OneCore voices, and this is reached *after* SAPI produced
        # no audio -- the classic all-OneCore machine, where SAPI writes an empty WAV.
        # Try the WinRT engine whenever Windows speech is available rather than gating
        # on the availability probe: a false-negative probe is one of the ways the
        # Arabic voice goes missing, and the attempt costs nothing when SAPI already
        # returned silence. The WinRT run fails harmlessly (no audio) if it truly is
        # unavailable.
        if windows_speech_available():
            return WindowsOneCoreSynthesizer(
                voice=self.voice, language=self.language, prosody=self.prosody
            )
        return None

    def _render(self, text: str, language: str) -> AudioClip | None:
        # A SAPI voice whose culture does not match the text is worse than silence:
        # SAPI does not fail, it reads the text with the wrong voice, so Arabic comes
        # out mangled by an English voice. That is not the "no audio" case
        # `_fallback` is reached by, so hand a language SAPI cannot serve to the
        # WinRT engine -- the one that sees the Arabic voice added through Settings ->
        # Add voices -- and only then fall back to SAPI's own output.
        if language and language != "en" and not self._serves_language(language):
            if windows_speech_available():
                other = WindowsOneCoreSynthesizer(
                    voice=self.voice, language=self.language, prosody=self.prosody
                )._render(text, language)
                if other is not None and other.samples:
                    return other
        return super()._render(text, language)

    def _serves_language(self, language: str) -> bool:
        """True when a SAPI voice for ``language`` is installed and enabled here."""
        return _sapi_serves_language(language)


class WindowsOneCoreSynthesizer(_WindowsPowerShellSynthesizer):
    """The WinRT/OneCore TTS engine: the one that actually sees Windows 11 voices.

    Windows 10/11 keeps the voices a user adds under *Settings -> Speech -> Add
    voices* in a separate store (``Speech_OneCore``) that the legacy
    ``System.Speech`` engine does not list. ``Windows.Media.SpeechSynthesis``
    exposes every installed voice, so this engine reads Arabic with the Arabic
    voice the user installed instead of falling back to an English one and mangling
    it. It is preferred over :class:`WindowsSapiSynthesizer` whenever the WinRT
    probe succeeds; the catalog voice maps onto language, gender, and an SSML rate.
    """

    name = _WINDOWS_ONE_CORE

    def _script(self, text_path: Path, target: Path, language: str) -> str:
        return _windows_onecore_script(text_path, target, self.voice, language, self.prosody)

    def _fallback(self) -> _WindowsPowerShellSynthesizer | None:
        # The mirror of SAPI's fallback: if the WinRT engine cannot be driven after
        # all (its store is empty or the run failed), a legacy SAPI voice can still
        # read the text, so the reading is not blank when a SAPI voice exists. A
        # language SAPI cannot serve is not handed over, because SAPI would read it
        # with a mismatched voice rather than fail.
        if windows_speech_available() and (
            self.language is None
            or self.language == "en"
            or self._language_in_sapi(self.language)
        ):
            return WindowsSapiSynthesizer(
                voice=self.voice, language=self.language, prosody=self.prosody
            )
        return None


class WhisperRecognizer:
    """A real ASR engine driven through the ``whisper`` command-line program.

    ``whisper`` writes its transcript next to the audio file, so we render the
    clip to a temporary WAV, run the command, and read the ``.txt`` it produces.
    When the program is missing or fails, transcription raises ``ValueError`` so
    the caller can fall back rather than inventing speech.
    """

    name = "whisper"

    def __init__(self, executable: str = "whisper", *, model: str = "tiny") -> None:
        self.executable = executable
        self.model = model

    def transcribe(self, clip: AudioClip) -> str:
        with tempfile.TemporaryDirectory(prefix="fayd-asr-") as folder:
            audio = Path(folder) / "speech.wav"
            clip.write_wav(audio)
            argv = [
                self.executable,
                str(audio),
                "--model",
                self.model,
                "--output_format",
                "txt",
                "--output_dir",
                folder,
                "--fp16",
                "False",
            ]
            try:
                completed = subprocess.run(
                    argv,
                    capture_output=True,
                    timeout=120,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as error:
                raise ValueError(f"ASR engine {self.name!r} failed: {error}") from error
            transcript = audio.with_suffix(".txt")
            if completed.returncode != 0 or not transcript.exists():
                raise ValueError(f"ASR engine {self.name!r} produced no transcript.")
            text = transcript.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                raise ValueError(f"ASR engine {self.name!r} produced an empty transcript.")
            return text


def discover_recognizer_name() -> str:
    """Return the best installed ASR engine: ``'qwen3-asr'``, ``'whisper'``, else ``'transcript'``.

    The local Qwen3-ASR model is preferred when its extra and weights are present
    (it covers Arabic, where ``whisper`` is English-first), then the ``whisper``
    program, then the transcript stand-in -- so ``auto`` never fails and always
    picks the strongest engine that is actually installed.
    """
    from core.qwen_speech import available_recognizer

    if available_recognizer():
        return "qwen3-asr"
    return "whisper" if shutil.which("whisper") else "transcript"


class TranscriptRecognizer:
    """An ASR stand-in that echoes a supplied transcript.

    Used by the text voice loop and by tests, so the pipeline can run without
    audio hardware. It deliberately refuses to invent speech from silence.
    """

    name = "transcript"

    def __init__(self, transcript: str = "") -> None:
        self.transcript = transcript

    def transcribe(self, clip: AudioClip) -> str:
        if not self.transcript.strip():
            raise ValueError("No transcript was supplied to the recognizer.")
        return self.transcript


def load_recognizer(name: str = "transcript", **options) -> SpeechRecognizer:
    """Load a named ASR engine.

    ``name='auto'`` picks ``whisper`` when it is installed, else the transcript
    recognizer. An explicitly named engine that is unavailable raises, so a user
    who asks for it gets a clear error instead of a silent substitute.
    """
    if name == "auto":
        name = discover_recognizer_name()
    if name == "transcript":
        return TranscriptRecognizer(options.get("transcript", ""))
    if name == "whisper":
        executable = options.get("executable", "whisper")
        if shutil.which(executable):
            return WhisperRecognizer(executable, model=options.get("model", "tiny"))
        raise ValueError(
            "ASR engine 'whisper' is not installed. Use --asr auto or --asr transcript."
        )
    if name in ("qwen3-asr", "qwen"):
        from core.qwen_speech import QwenAsrRecognizer, QwenSpeechUnavailable, available_recognizer

        if not available_recognizer() and not options.get("runner"):
            from core.qwen_speech import qwen_asr_status

            raise ValueError(
                "ASR engine 'qwen3-asr' is not ready. " + qwen_asr_status().hint
            )
        try:
            return QwenAsrRecognizer(
                model_dir=options.get("model_dir"),
                runner=options.get("runner"),
                language=options.get("language"),
                device=options.get("device", "cpu"),
                root=options.get("root"),
            )
        except QwenSpeechUnavailable as error:
            raise ValueError(str(error)) from error
    raise ValueError(
        f"ASR engine {name!r} is not available in this build. "
        "Install a speech backend or use the transcript recognizer."
    )


def windows_engine_name() -> str:
    """The Windows engine that reads this machine best, by name (Arabic-aware).

    One place decides, so discovery, loading, and the Qwen3-TTS Arabic fallback all
    agree. The engine that actually lists an Arabic voice wins first, because that
    is the one that reads Arabic correctly; a false-negative WinRT probe must not
    send an Arabic reading to SAPI, which cannot see the Arabic voice added through
    *Settings -> Speech -> Add voices* and would mangle it with an English voice.
    The availability probe is the tie-breaker when no Arabic voice is installed.
    """
    engine = windows_arabic_engine()
    if engine is not None:
        return engine
    return _WINDOWS_ONE_CORE if windows_onecore_usable() else _WINDOWS_SAPI


def _windows_engine(
    *,
    voice: SelectableVoice | None = None,
    language: str | None = None,
    prosody: Prosody | None = None,
) -> SpeechSynthesizer:
    """The best Windows built-in engine for ``voice``/``language`` (:func:`windows_engine_name`)."""
    if windows_engine_name() == _WINDOWS_ONE_CORE:
        return WindowsOneCoreSynthesizer(voice=voice, language=language, prosody=prosody)
    return WindowsSapiSynthesizer(voice=voice, language=language, prosody=prosody)


def _system_synthesizer(
    *,
    voice: SelectableVoice | None = None,
    language: str | None = None,
    prosody: Prosody | None = None,
) -> SpeechSynthesizer | None:
    """The installed system TTS engine, or ``None`` when there is none.

    Used as the Arabic fallback for Qwen3-TTS (which has no Arabic). ``None`` --
    not a silent stand-in -- so the caller can decide, rather than being handed an
    engine that would produce nothing. ``prosody`` carries a chosen mood to the
    fallback too, so a mood is not lost when a language falls back.
    """
    for engine_name, executable, builder, voice_options in TTS_ENGINES:
        if shutil.which(executable):
            return CommandSynthesizer(
                engine_name,
                executable,
                builder,
                voice=voice,
                language=language,
                voice_options=voice_options,
                prosody=prosody,
            )
    if windows_speech_available():
        return _windows_engine(voice=voice, language=language, prosody=prosody)
    return None


def load_synthesizer(name: str = "silent", **options) -> SpeechSynthesizer:
    """Load a named TTS engine, optionally with a chosen catalog voice.

    ``name='auto'`` picks the first installed system engine (``espeak``,
    ``say``, ...), else the silent synthesizer. On Windows it prefers the OneCore
    engine over SAPI (see :func:`_windows_engine`). An explicitly named engine that
    is unavailable raises, so the caller never mistakes silence for speech.

    ``voice`` selects a :class:`~core.voices.Voice` from the catalog, by key or
    object; ``language`` pins the spoken language (``None`` lets the engine's text
    decide, so Arabic is read as Arabic). ``prosody`` is a chosen
    :class:`~core.voice_moods.Prosody` (a mood), applied as far as the selected
    engine really supports -- a formant engine takes pitch, speed, and melody; Piper
    takes pace and expressiveness; Qwen3-TTS takes the style instruction; the
    VoiceTut cloning engine takes little. A voice can only be *applied* by an
    engine that supports one -- the silent synthesizer ignores it, and its
    ``name`` stays ``'silent'`` so a caller can tell it apart from real speech.
    """
    if name == "auto":
        name = _engine_for_voice(options.get("voice")) or discover_synthesizer_name()
    if name == "silent":
        return SilentSynthesizer()
    voice = _resolve_voice(options.get("voice"))
    language = options.get("language")
    prosody: Prosody | None = options.get("prosody")
    if name == "audio.cpp":
        from core.audiocpp_speech import (
            AUDIOCPP_ENGINE_NAME,
            AudioCppSynthesizer,
            AudioCppUnavailable,
            audiocpp_status,
        )
        from core.audiocpp_speech import (
            available_synthesizer as audiocpp_ready,
        )

        if not audiocpp_ready(options.get("root")) and not options.get("runner"):
            raise ValueError(
                f"TTS engine {AUDIOCPP_ENGINE_NAME!r} is not ready. "
                + audiocpp_status(options.get("root")).hint
            )
        fallback = options.get("fallback")
        if fallback is None:
            # A missing engine or model must degrade to real speech, not silence: the
            # system engine reads the text instead, and the synthesizer reports it.
            fallback = _system_synthesizer(voice=voice, language=language, prosody=prosody)
        model_voice = voice if isinstance(voice, ModelVoice) else None
        try:
            return AudioCppSynthesizer(
                voice=model_voice,
                language=options.get("language"),
                spec_key=options.get("spec_key"),
                model_file=options.get("model_file"),
                runner=options.get("runner"),
                fallback=fallback,
                root=options.get("root"),
                steps=options.get("steps", 8),
                threads=options.get("threads"),
                prosody=prosody,
            )
        except AudioCppUnavailable as error:
            raise ValueError(str(error)) from error
    if name in ("piper",):
        from core.piper_speech import (
            PiperSynthesizer,
            PiperUnavailable,
        )
        from core.piper_speech import (
            available_synthesizer as piper_ready,
        )

        if not piper_ready(options.get("root")) and not options.get("runner"):
            from core.piper_speech import piper_status

            raise ValueError("TTS engine 'piper' is not ready. " + piper_status(options.get("root")).hint)
        fallback = options.get("fallback")
        if fallback is None:
            # Only one of the two languages may have a voice downloaded; a system
            # engine reads the other rather than leaving it silent. Chosen once here
            # so every caller (window, CLI, voice loop) behaves the same.
            system = _system_synthesizer(voice=voice, language=language, prosody=prosody)
            if system is not None:
                fallback = system
            else:
                # With no system engine, a language whose Piper voice was never
                # downloaded would otherwise crash the reading -- and discovery
                # picks Piper as soon as *any* voice is cached, so `auto` could
                # return an engine that fails on the other language. Degrade to
                # silence instead, exactly as a failing system engine already does,
                # so `auto` keeps its promise of a usable engine. The low-level
                # `PiperSynthesizer` still raises when built with no fallback.
                fallback = SilentSynthesizer()
        try:
            return PiperSynthesizer(
                voice=voice,
                language=language,
                spec_key=options.get("spec_key"),
                model_dir=options.get("model_dir"),
                runner=options.get("runner"),
                fallback=fallback,
                root=options.get("root"),
                prosody=prosody,
            )
        except PiperUnavailable as error:
            raise ValueError(str(error)) from error
    if name in ("qwen3-tts", "qwen"):
        from core.qwen_speech import (
            TTS_ENGINE_NAME,
            QwenSpeechUnavailable,
            QwenTtsSynthesizer,
            available_synthesizer,
        )

        if not available_synthesizer() and not options.get("runner"):
            from core.qwen_speech import qwen_tts_status

            raise ValueError("TTS engine 'qwen3-tts' is not ready. " + qwen_tts_status().hint)
        fallback = options.get("fallback")
        if fallback is None:
            # Qwen3-TTS does not speak Arabic, so a system engine reads it instead
            # of the model silently mispronouncing it. Chosen once here, so every
            # caller (window, CLI, voice loop) gets the same honest behaviour.
            system = _system_synthesizer(voice=voice, language=language, prosody=prosody)
            if system is not None:
                fallback = system
        # A chosen Qwen3-TTS voice names its own built-in speaker, so it is used
        # unless an explicit ``speaker`` was given.
        speaker = options.get("speaker")
        if speaker is None and isinstance(voice, ModelVoice) and voice.engine == TTS_ENGINE_NAME:
            speaker = voice.name
        # A chosen mood's style instruction is the one prosody Qwen3-TTS takes; an
        # explicit ``instruct`` still wins.
        instruct = options.get("instruct")
        if instruct is None and prosody is not None and prosody.instruct:
            instruct = prosody.instruct
        try:
            return QwenTtsSynthesizer(
                model_dir=options.get("model_dir"),
                runner=options.get("runner"),
                speaker=speaker,
                language=language,
                instruct=instruct,
                fallback=fallback,
                device=options.get("device", "cpu"),
                root=options.get("root"),
            )
        except QwenSpeechUnavailable as error:
            raise ValueError(str(error)) from error
    if name in (_WINDOWS_SAPI, _WINDOWS_ONE_CORE):
        if not windows_speech_available():
            raise ValueError(
                "The Windows speech engine is not available here. "
                "Use --tts auto or --tts silent."
            )
        if name == _WINDOWS_ONE_CORE:
            if not windows_onecore_usable():
                raise ValueError(
                    "The Windows OneCore speech engine is not available here. "
                    "Use --tts auto or --tts windows-sapi."
                )
            return WindowsOneCoreSynthesizer(voice=voice, language=language, prosody=prosody)
        return WindowsSapiSynthesizer(voice=voice, language=language, prosody=prosody)
    for engine_name, executable, builder, voice_options in TTS_ENGINES:
        if name == engine_name:
            if shutil.which(executable):
                return CommandSynthesizer(
                    engine_name,
                    executable,
                    builder,
                    voice=voice,
                    language=language,
                    voice_options=voice_options,
                    prosody=prosody,
                )
            raise ValueError(
                f"TTS engine {name!r} is not installed. Use --tts auto or --tts silent."
            )
    raise ValueError(
        f"TTS engine {name!r} is not available in this build. "
        "Install a speech backend or use the silent synthesizer."
    )


def _resolve_voice(value) -> SelectableVoice | None:
    """Coerce a voice key or object into a catalog ``Voice`` or a named voice.

    ``None`` returns the default catalog voice so a plain ``--tts espeak-ng``
    still speaks with a real, named timbre. A ``system:<engine>:<name>`` key is
    resolved against the voices the OS really reports, so a real installed voice is
    used exactly; when it is no longer installed the default catalog voice is
    returned rather than asking the engine for a name it would reject. A
    ``model:<engine>:<name>`` key is resolved the same way against the local models
    that are actually installed. An unknown catalog key raises, so a typo is
    reported rather than silently replaced.
    """
    if value is None:
        return default_voice()
    if isinstance(value, (Voice, SystemVoice, ModelVoice)):
        return value
    key = str(value)
    if is_system_voice_key(key):
        return find_system_voice(key) or default_voice()
    if is_model_voice_key(key):
        return find_model_voice(key) or default_voice()
    return find_voice(key)


def _engine_for_voice(value: object) -> str | None:
    """The engine a chosen named voice must be spoken by, or ``None``.

    A ``SystemVoice`` names a voice one OS engine installed, and a ``ModelVoice``
    names a voice inside one local model, so each must be handed to the engine that
    owns it. Returns ``None`` for a catalog voice (the default discovery decides) or
    for an engine that is not actually available here, so the caller falls back to
    :func:`discover_synthesizer_name`.
    """
    if isinstance(value, ModelVoice):
        if value.engine == "audio.cpp":
            from core.audiocpp_speech import (
                AUDIOCPP_ENGINE_NAME,
            )
            from core.audiocpp_speech import (
                available_synthesizer as audiocpp_ready,
            )

            if audiocpp_ready():
                return AUDIOCPP_ENGINE_NAME
            return None
        if value.engine == "piper":
            from core.piper_speech import PIPER_ENGINE_NAME
            from core.piper_speech import available_synthesizer as piper_ready

            if piper_ready():
                return PIPER_ENGINE_NAME
            return None
        if value.engine == "qwen3-tts":
            from core.qwen_speech import TTS_ENGINE_NAME
            from core.qwen_speech import available_synthesizer as qwen_ready

            if qwen_ready():
                return TTS_ENGINE_NAME
            return None
        return None
    if isinstance(value, SystemVoice):
        voice: SystemVoice | None = value
    elif isinstance(value, str) and is_model_voice_key(value):
        resolved = find_model_voice(value)
        return _engine_for_voice(resolved) if resolved is not None else None
    elif isinstance(value, str) and is_system_voice_key(value):
        voice = find_system_voice(value)
    else:
        return None
    if voice is None:
        return None
    if voice.engine == _WINDOWS_ONE_CORE and windows_onecore_usable():
        return _WINDOWS_ONE_CORE
    if voice.engine == _WINDOWS_SAPI and windows_speech_available():
        # Do not send an Arabic voice to SAPI when only OneCore can see an Arabic
        # voice: SAPI would read the text with an English voice instead.
        if voice.arabic and windows_arabic_engine() == _WINDOWS_ONE_CORE:
            return _WINDOWS_ONE_CORE
        return _WINDOWS_SAPI
    return None


class MicRecording:
    """A microphone recording in progress, stoppable before the time limit.

    :meth:`MicrophoneRecorder.start` returns one of these instead of blocking for
    a fixed number of seconds: the caller :meth:`stop` s it when the user is done
    talking (a second click on the microphone icon) and :meth:`result` returns the
    clip. ``arecord`` stops cleanly on ``SIGINT`` (it finalises the WAV header);
    ``sox`` is asked for a long capture and terminated the same way. A recorder
    that needs a *file* to stop -- the Windows MCI recorder, whose ``save`` step
    writes the WAV -- passes ``stop_path`` instead, and :meth:`stop` creates it.
    The handle also works with an injected process, so the flow is testable with
    no audio hardware.
    """

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        *,
        path: Path,
        recorder: str,
        folder: Path,
        stop_path: Path | None = None,
    ) -> None:
        self._process = process
        self._path = path
        self._recorder = recorder
        self._folder = folder
        self._stop_path = stop_path

    @property
    def recorder(self) -> str:
        return self._recorder

    @property
    def recording(self) -> bool:
        """True while the recorder is still capturing."""
        return self._process.poll() is None

    def stop(self) -> None:
        """Ask the recorder to stop and finalise the WAV (idempotent).

        ``SIGINT`` is used rather than ``terminate()`` because ``arecord`` flushes
        its WAV header on interrupt; a recorder that ignores it is killed after a
        short grace period, so ``stop`` never hangs. The Windows MCI recorder is
        stopped through its marker file instead, because it finalises the WAV in a
        ``save`` step a signal would skip.
        """
        if self._process.poll() is not None:
            return
        if self._stop_path is not None:
            with contextlib.suppress(OSError):
                self._stop_path.write_text("stop", encoding="utf-8")
        else:
            try:
                self._process.send_signal(signal.SIGINT)
            except (OSError, ValueError):
                self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)

    def result(self, *, timeout: float | None = 30.0) -> AudioClip:
        """Stop if needed and return the recorded clip.

        Raises :class:`ValueError` when no audio was captured (an unavailable
        microphone, or a clip too short to hold a word) rather than returning a
        silent clip the recognizer would reject anyway.
        """
        if self._process.poll() is None:
            self.stop()
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        try:
            if not self._path.exists() or self._path.stat().st_size <= 44:
                raise ValueError(
                    f"Microphone recorder {self._recorder!r} captured no audio."
                )
            return AudioClip.read_wav(self._path)
        finally:
            shutil.rmtree(self._folder, ignore_errors=True)


#: The Windows recorder name (MCI ``waveaudio`` through PowerShell). It is
#: discovered without ``shutil.which`` because the "program" is the shell itself.
_WINDOWS_RECORDER = "windows-mci"


class MicrophoneRecorder:
    """Records a clip from the system microphone, when a recorder exists.

    Uses ``arecord`` (Linux/ALSA), ``pw-record``/``parecord`` (PipeWire/PulseAudio),
    ``sox`` (cross-platform), or Windows' own MCI ``waveaudio`` device when none of
    those exist; when nothing is available, :meth:`available` is False and the
    caller should fall back to the text loop. No pip package and no network access
    is involved. :meth:`record` blocks for a fixed number of seconds (the CLI's
    behaviour); :meth:`start` returns a :class:`MicRecording` the caller stops when
    the user is done -- which is how the window's microphone icon toggles.
    """

    def __init__(self, *, seconds: float = 4.0, sample_rate: int = SAMPLE_RATE) -> None:
        self.seconds = seconds
        self.sample_rate = sample_rate

    @property
    def recorder(self) -> str | None:
        if shutil.which("arecord"):
            return "arecord"
        # Modern Linux desktops are PipeWire/PulseAudio first and may have no ALSA
        # ``arecord`` at all; ``pw-record``/``parecord`` capture the same microphone.
        if shutil.which("pw-record"):
            return "pw-record"
        if shutil.which("parecord"):
            return "parecord"
        if shutil.which("sox"):
            return "sox"
        # Windows ships neither arecord nor sox, so fall back to its own capture
        # (MCI waveaudio) rather than telling a Windows user to install ALSA tools.
        if _windows_shell_available():
            return _WINDOWS_RECORDER
        return None

    def available(self) -> bool:
        return self.recorder is not None

    @staticmethod
    def _argv(recorder: str, target: Path, sample_rate: int, seconds: float) -> list[str]:
        """The command that captures ``seconds`` of mono 16-bit audio to ``target``."""
        if recorder == "arecord":
            return [
                "arecord", "-q", "-f", "S16_LE", "-r", str(sample_rate),
                "-d", str(int(seconds)), str(target),
            ]
        if recorder == "pw-record":
            # ``pw-record`` infers the format from the suffix; --rate keeps the
            # sample rate the recognizers expect.
            return [
                "pw-record", "--rate", str(sample_rate), "--channels", "1",
                "--format", "s16", str(target),
            ]
        if recorder == "parecord":
            return [
                "parecord", "--rate", str(sample_rate), "--channels", "1",
                "--format", "s16le", "--file-format=wav", str(target),
            ]
        return [
            "sox", "-d", "-r", str(sample_rate), "-c", "1", "-b", "16",
            str(target), "trim", "0", str(seconds),
        ]

    def start(self) -> MicRecording:
        """Begin recording in the background; return a stoppable MicRecording.

        ``seconds`` bounds the capture (a safety limit); the caller normally stops
        it sooner. A recorder that cannot start raises, so the caller can report
        it instead of hanging.
        """
        recorder = self.recorder
        if recorder is None:
            raise ValueError("No microphone recorder is available. " + microphone_install_hint())
        folder = Path(tempfile.mkdtemp(prefix="fayd-mic-"))
        target = folder / "input.wav"
        stop_path: Path | None = None
        if recorder == _WINDOWS_RECORDER:
            # The Windows recorder stops on a marker file, not a signal, because its
            # WAV is written by a `save` step a signal would skip.
            stop_path = folder / "stop"
            script = _windows_recorder_script(target, stop_path, self.sample_rate, self.seconds)
            argv = [_WINDOWS_SHELL, "-NoProfile", "-NonInteractive", "-Command", script]
        else:
            argv = self._argv(recorder, target, self.sample_rate, self.seconds)
        try:
            process = subprocess.Popen(
                argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except OSError as error:
            shutil.rmtree(folder, ignore_errors=True)
            raise ValueError(f"Microphone recorder {recorder!r} failed: {error}") from error
        return MicRecording(
            process, path=target, recorder=recorder, folder=folder, stop_path=stop_path
        )

    def record(self) -> AudioClip:
        """Record for a fixed ``seconds`` and return the clip (blocking)."""
        return self.start().result()


class Playback:
    """A clip playing in the background, stoppable before it ends.

    :meth:`Speaker.start` returns one of these instead of blocking: the audio runs
    in a child process the caller can :meth:`stop` (so a second click on "Read
    aloud" stops the reading) while :meth:`wait` blocks until it finishes. It is
    honest about what happened -- :attr:`stopped` says whether it was cut short,
    and a real player failure is raised from :meth:`wait` rather than swallowed.
    """

    def __init__(self, process: subprocess.Popen[bytes], *, player: str, folder: Path) -> None:
        self._process = process
        self._player = player
        self._folder: Path | None = folder
        self.stopped = False

    @property
    def player(self) -> str:
        return self._player

    @property
    def playing(self) -> bool:
        """True while the audio is still being played."""
        return self._process.poll() is None

    def stop(self) -> None:
        """Stop the audio now, if it is still playing (idempotent).

        Also removes the temporary WAV, so a stopped reading leaves nothing
        behind -- the caller does not have to remember to clean up.
        """
        if self._process.poll() is None:
            self.stopped = True
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        self._cleanup()

    def wait(self, *, timeout: float | None = 120.0) -> None:
        """Block until the audio ends; raise if the player failed.

        A clip the caller stopped is *not* a failure: it returns quietly, so the
        GUI's ``stop()`` then ``wait()`` flow never reports a spurious error.
        """
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            self.stop()
            raise ValueError(f"Audio player {self._player!r} timed out.") from error
        finally:
            self._cleanup()
        if self.stopped:
            return
        if self._process.returncode != 0:
            raise ValueError(f"Audio player {self._player!r} failed to play the clip.")

    def _cleanup(self) -> None:
        if self._folder is not None:
            shutil.rmtree(self._folder, ignore_errors=True)
            self._folder = None


class Speaker:
    """Plays a clip through the system speakers, when a player exists.

    Uses ``aplay`` (Linux/ALSA), ``afplay`` (macOS), or ``sox``/``play`` if
    installed. On Windows, where none of those exist, it falls back to the
    built-in ``SoundPlayer`` through PowerShell so read-aloud is actually audible.
    :meth:`available` reports honestly, and :meth:`play` raises when no player
    exists so the caller can fall back to text. No pip package needed.

    :meth:`play` blocks until the clip ends (the CLI's behaviour); :meth:`start`
    returns a :class:`Playback` the caller can stop -- which is how a chat window
    lets a second click end the reading.
    """

    def __init__(self, *, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate

    @property
    def player(self) -> str | None:
        for executable in ("aplay", "afplay", "play", "sox"):
            if shutil.which(executable):
                return executable
        if windows_speech_available():
            return _WINDOWS_SHELL
        return None

    def available(self) -> bool:
        return self.player is not None

    @staticmethod
    def _argv(player: str, target: Path) -> list[str]:
        """The command that plays the WAV ``target`` through ``player``."""
        if player == "aplay":
            return ["aplay", "-q", str(target)]
        if player == "afplay":
            return ["afplay", str(target)]
        if player == "play":
            return ["play", "-q", str(target)]
        if player == _WINDOWS_SHELL:
            # Windows: play the WAV through the built-in SoundPlayer.
            script = (
                f"(New-Object System.Media.SoundPlayer '{_ps_quote(str(target))}').PlaySync()"
            )
            return [player, "-NoProfile", "-NonInteractive", "-Command", script]
        return ["sox", str(target), "-d"]  # sox

    def start(self, clip: AudioClip) -> Playback:
        """Begin playing ``clip`` in the background; return a stoppable Playback.

        The WAV is written under a temporary folder that outlives this call -- a
        background player reads the file after we return -- and :class:`Playback`
        removes it when the clip finishes or is stopped.
        """
        player = self.player
        if player is None:
            raise ValueError("No audio player is available (install aplay, afplay, or sox).")
        folder = Path(tempfile.mkdtemp(prefix="fayd-play-"))
        target = folder / "speech.wav"
        clip.write_wav(target)
        argv = self._argv(player, target)
        try:
            process = subprocess.Popen(
                argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except OSError as error:
            shutil.rmtree(folder, ignore_errors=True)
            raise ValueError(f"Audio player {player!r} failed: {error}") from error
        return Playback(process, player=player, folder=folder)

    def play(self, clip: AudioClip) -> None:
        """Play ``clip`` and block until it ends; raise if no player or it fails."""
        self.start(clip).wait()


@dataclass
class VoiceTurn:
    """One voice exchange: what was heard, what was answered, what was spoken."""

    heard: str
    answer_text: str
    spoken: AudioClip
    used_model: bool = False
    citations: tuple[str, ...] = field(default_factory=tuple)


class VoiceAssistant:
    """Wraps a grounded assistant with a speech recognizer and synthesizer."""

    def __init__(
        self,
        assistant,
        *,
        recognizer: SpeechRecognizer | None = None,
        synthesizer: SpeechSynthesizer | None = None,
    ) -> None:
        self.assistant = assistant
        self.recognizer = recognizer or TranscriptRecognizer()
        self.synthesizer = synthesizer or SilentSynthesizer()

    def respond_to_speech(self, clip: AudioClip, *, conversation=None, limit: int = 4) -> VoiceTurn:
        heard = self.recognizer.transcribe(clip).strip()
        if not heard:
            raise ValueError("Speech recognizer returned no text.")
        answer = self.assistant.ask(heard, conversation=conversation, limit=limit)
        spoken = self.synthesizer.synthesize(answer.text)
        return VoiceTurn(
            heard=heard,
            answer_text=answer.text,
            spoken=spoken,
            used_model=answer.used_model,
            citations=answer.citations,
        )


class AudioModel:
    """Adapter that exposes an audio clip to a :class:`CognitiveModel`.

    The clip is described as text metadata (duration, sample rate, silence) so a
    model that cannot process raw audio still receives an honest description
    rather than fabricated content.
    """

    def __init__(self, model: CognitiveModel) -> None:
        self.model = model

    def describe(self, clip: AudioClip, *, instruction: str = "Describe the audio signal.") -> str:
        request = ModelRequest(
            inputs=(),
            system_instruction=instruction,
            context=(
                f"Audio clip: duration {clip.duration_seconds:.2f} s, "
                f"sample rate {clip.sample_rate} Hz, "
                f"{'silent' if clip.is_silent else 'contains a signal'}.",
            ),
        )
        return self.model.generate(request).text


__all__ = [
    "SAMPLE_RATE",
    "SPEECH_SYSTEM_PACKAGES",
    "TTS_ENGINES",
    "AudioClip",
    "AudioModel",
    "CommandSynthesizer",
    "MicRecording",
    "MicrophoneRecorder",
    "Playback",
    "SelectableVoice",
    "SilentSynthesizer",
    "Speaker",
    "SpeechCommandRunner",
    "SpeechInstallPlan",
    "SpeechInstallResult",
    "SpeechRecognizer",
    "SpeechSynthesizer",
    "TranscriptRecognizer",
    "VoiceAssistant",
    "VoiceTurn",
    "WhisperRecognizer",
    "WindowsOneCoreSynthesizer",
    "WindowsSapiSynthesizer",
    "discover_recognizer_name",
    "discover_synthesizer_name",
    "install_speech_engine",
    "load_recognizer",
    "load_synthesizer",
    "microphone_install_hint",
    "refresh_windows_speech_caches",
    "speech_available",
    "speech_install_plan",
    "system_voices",
    "windows_arabic_engine",
    "windows_engine_name",
    "windows_installed_voices",
    "windows_onecore_available",
    "windows_onecore_usable",
    "windows_speech_available",
    "windows_voice_report",
]
