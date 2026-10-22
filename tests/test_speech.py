"""Tests for the speech layer: audio clips, recognizer/synthesizer, voice loop."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.assistant import GroundedAssistant
from core.documents import build_document_store
from core.ontology import Ontology
from core.domain_knowledge import seed_ontology
from core.curriculum_knowledge import seed_curriculum
from core.rag import RagRetriever
from core.speech import (
    AudioClip,
    AudioModel,
    CommandSynthesizer,
    MicrophoneRecorder,
    MicRecording,
    Playback,
    SilentSynthesizer,
    Speaker,
    TranscriptRecognizer,
    VoiceAssistant,
    WhisperRecognizer,
    discover_recognizer_name,
    discover_synthesizer_name,
    load_recognizer,
    load_synthesizer,
)


def _assistant():
    ontology = Ontology()
    seed_ontology(ontology)
    seed_curriculum(ontology)
    return GroundedAssistant(RagRetriever(build_document_store(), ontology=ontology))


def test_audio_clip_reports_duration_and_silence():
    clip = AudioClip.silence(1.0)
    assert clip.duration_seconds == pytest.approx(1.0)
    assert clip.is_silent
    assert AudioClip((1, -1, 2)).is_silent is False


def test_audio_clip_validates_samples_and_rate():
    with pytest.raises(ValueError):
        AudioClip((40000,))
    with pytest.raises(ValueError):
        AudioClip((0,), sample_rate=0)
    with pytest.raises(ValueError):
        AudioClip.silence(-1)


def test_audio_clip_bytes_round_trip():
    clip = AudioClip((100, -200, 300, -400))
    assert AudioClip.from_bytes(clip.to_bytes()) == clip


def test_audio_clip_wav_round_trip(tmp_path):
    clip = AudioClip(tuple(range(-100, 100)))
    path = tmp_path / "clip.wav"
    clip.write_wav(path)
    restored = AudioClip.read_wav(path)
    assert restored.samples == clip.samples
    assert restored.sample_rate == clip.sample_rate


def test_silent_synthesizer_sizes_to_text():
    clip = SilentSynthesizer().synthesize("some text")
    assert clip.is_silent
    assert clip.duration_seconds > 0
    with pytest.raises(ValueError):
        SilentSynthesizer().synthesize("  ")


def test_transcript_recognizer_returns_supplied_text():
    assert TranscriptRecognizer("hello").transcribe(AudioClip.silence(0.1)) == "hello"
    with pytest.raises(ValueError):
        TranscriptRecognizer().transcribe(AudioClip.silence(0.1))


def test_loaders_fall_back_and_reject_unknown_engines():
    assert isinstance(load_recognizer("transcript", transcript="text"), TranscriptRecognizer)
    assert isinstance(load_synthesizer("silent"), SilentSynthesizer)
    with pytest.raises(ValueError):
        load_recognizer("whisper")
    with pytest.raises(ValueError):
        load_synthesizer("not-a-real-engine")


def test_voice_assistant_runs_the_full_loop():
    voice = VoiceAssistant(
        _assistant(),
        recognizer=TranscriptRecognizer("What is the relationship between porosity and permeability?"),
        synthesizer=SilentSynthesizer(),
    )
    turn = voice.respond_to_speech(AudioClip.silence(0.2))
    assert turn.heard == "What is the relationship between porosity and permeability?"
    assert turn.answer_text
    assert turn.citations
    assert turn.spoken.is_silent


def test_voice_assistant_keeps_conversation():
    voice = VoiceAssistant(_assistant(), recognizer=TranscriptRecognizer("permeability"))
    conversation = voice.assistant.memory.start(title="voice")
    voice.respond_to_speech(AudioClip.silence(0.1), conversation=conversation)
    assert len(conversation.messages) == 2


def test_voice_assistant_rejects_empty_speech():
    voice = VoiceAssistant(_assistant(), recognizer=TranscriptRecognizer("   "))
    with pytest.raises(ValueError):
        voice.respond_to_speech(AudioClip.silence(0.1))


def test_audio_model_describes_a_clip():
    class _Echo:
        def generate(self, request):
            from core.model_interface import ModelResponse

            return ModelResponse(text="description", confidence=0.5, model_name="echo")

    description = AudioModel(_Echo()).describe(AudioClip.silence(0.5))
    assert description == "description"


# --- system engines -------------------------------------------------------


def test_discovery_returns_a_known_engine_name():
    assert discover_synthesizer_name() in {"silent", "espeak-ng", "espeak", "flite", "say", "piper"}
    assert discover_recognizer_name() in {"transcript", "whisper", "qwen3-asr"}


def test_auto_loader_always_returns_a_usable_engine():
    synth = load_synthesizer("auto")
    recog = load_recognizer("auto", transcript="hello")
    assert synth.name in {"silent", "espeak-ng", "espeak", "flite", "say", "piper"}
    assert recog.name in {"transcript", "whisper", "qwen3-asr"}
    assert synth.synthesize("hello").duration_seconds > 0


def test_auto_loader_falls_back_and_named_engine_raises():
    # 'auto' never fails: it picks the best available engine or a safe stand-in.
    assert load_synthesizer("auto").name in {"silent", "espeak-ng", "espeak", "flite", "say", "piper"}
    assert load_recognizer("auto", transcript="hi").name in {"transcript", "whisper", "qwen3-asr"}
    # an explicitly named, uninstalled engine raises rather than faking audio.
    with pytest.raises(ValueError):
        load_synthesizer("definitely-not-installed")
    with pytest.raises(ValueError):
        load_recognizer("definitely-not-installed")


def test_command_synthesizer_falls_back_when_engine_fails():
    def _builder(executable, text, path):
        return ["definitely-not-a-real-program", text]

    synth = CommandSynthesizer("fake", "definitely-not-a-real-program", _builder)
    clip = synth.synthesize("hello world")
    assert clip.is_silent  # honest fallback, never fabricated audio


def test_command_synthesizer_rejects_empty_text():
    synth = CommandSynthesizer("fake", "espeak", lambda e, t, p: [e, t])
    with pytest.raises(ValueError):
        synth.synthesize("   ")


def test_whisper_recognizer_reports_failure_without_inventing_speech():
    recognizer = WhisperRecognizer("definitely-not-a-real-program")
    with pytest.raises(ValueError):
        recognizer.transcribe(AudioClip.silence(0.5))


def test_microphone_recorder_reports_availability_honestly():
    recorder = MicrophoneRecorder()
    assert recorder.available() == (recorder.recorder is not None)


def test_speaker_reports_availability_honestly():
    speaker = Speaker()
    assert speaker.available() == (speaker.player is not None)


def test_speaker_raises_when_no_player_is_available(monkeypatch):
    import core.speech as speech

    monkeypatch.setattr(speech.shutil, "which", lambda name: None)
    with pytest.raises(ValueError):
        Speaker().play(AudioClip.silence(0.1))


class _FakeProcess:
    """A stand-in for the player/recorder child process (no real audio)."""

    def __init__(self) -> None:
        self.returncode: int | None = None  # None while running, like Popen
        self.signals: list[int] = []
        self.killed = False

    def poll(self):
        return self.returncode

    def send_signal(self, sig) -> None:
        self.signals.append(int(sig))
        self.returncode = 0

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = 0

    def wait(self, timeout=None) -> int:
        assert self.returncode is not None
        return self.returncode


def test_playback_stops_a_running_clip():
    """A Playback can stop the child process and reports it was stopped."""
    from pathlib import Path

    process = _FakeProcess()
    playback = Playback(process, player="aplay", folder=Path("/tmp/does-not-exist"))
    assert playback.playing is True
    playback.stop()
    assert playback.stopped is True
    assert process.returncode == 0  # terminated, not left running


def test_playback_wait_is_quiet_after_a_stop():
    """Stopping then waiting never reports a spurious failure."""
    from pathlib import Path

    process = _FakeProcess()
    playback = Playback(process, player="aplay", folder=Path("/tmp/does-not-exist"))
    playback.stop()
    playback.wait()  # must not raise


def test_microphone_recording_stops_and_finalises():
    """A MicRecording sends SIGINT so the recorder flushes its WAV header."""
    from pathlib import Path

    process = _FakeProcess()
    recording = MicRecording(
        process, path=Path("/tmp/nope.wav"), recorder="arecord", folder=Path("/tmp/nope")
    )
    assert recording.recording is True
    recording.stop()
    assert recording.recording is False
    assert process.signals  # SIGINT was delivered


class _FakeScriptProcess(_FakeProcess):
    """A recorder child that finishes only once its stop marker appears."""

    def __init__(self, marker: Path) -> None:
        super().__init__()
        self._marker = marker

    def wait(self, timeout=None) -> int:
        if self._marker.exists():
            self.returncode = 0
        assert self.returncode is not None, "the process was not told to stop"
        return self.returncode


def test_microphone_recording_stops_through_a_marker_file():
    """The Windows MCI recorder is stopped by a marker, not a signal.

    A signal would kill the process before its ``save`` step writes the WAV, so the
    stop must create the marker file the script polls instead.
    """
    from pathlib import Path

    with tempfile.TemporaryDirectory() as folder:
        marker = Path(folder) / "stop"
        process = _FakeScriptProcess(marker)
        recording = MicRecording(
            process,
            path=Path(folder) / "input.wav",
            recorder="windows-mci",
            folder=Path(folder),
            stop_path=marker,
        )
        assert recording.recording is True
        recording.stop()
        assert marker.exists()  # the script's own stop condition
        assert not process.signals  # a signal would skip the MCI `save`


def test_microphone_recorder_prefers_arecord_then_pipewire_then_sox(monkeypatch):
    """Discovery falls through to the modern Linux recorders before sox."""
    import core.speech as speech

    def only(*names: str):
        return lambda name: f"/usr/bin/{name}" if name in names else None

    monkeypatch.setattr(speech.shutil, "which", only("arecord"))
    assert MicrophoneRecorder().recorder == "arecord"
    monkeypatch.setattr(speech.shutil, "which", only("pw-record", "sox"))
    assert MicrophoneRecorder().recorder == "pw-record"
    monkeypatch.setattr(speech.shutil, "which", only("parecord", "sox"))
    assert MicrophoneRecorder().recorder == "parecord"
    monkeypatch.setattr(speech.shutil, "which", only("sox"))
    assert MicrophoneRecorder().recorder == "sox"


def test_microphone_recorder_uses_windows_capture_when_nothing_else_exists(monkeypatch):
    """A Windows machine with no arecord/sox still has a working recorder.

    This is the reported bug: the microphone icon did nothing on Windows because
    discovery only knew the Linux tools.
    """
    import core.speech as speech

    monkeypatch.setattr(speech.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        speech.shutil, "which", lambda name: "powershell" if name == "powershell" else None
    )
    recorder = MicrophoneRecorder()
    assert recorder.recorder == "windows-mci"
    assert recorder.available() is True
    assert MicrophoneRecorder().recorder is not None


def test_microphone_recorder_is_honestly_unavailable_without_any_recorder(monkeypatch):
    import core.speech as speech

    monkeypatch.setattr(speech.platform, "system", lambda: "Linux")
    monkeypatch.setattr(speech.shutil, "which", lambda name: None)
    assert MicrophoneRecorder().available() is False
    with pytest.raises(ValueError):
        MicrophoneRecorder().start()


def test_pipewire_and_pulse_recorder_argv_are_mono_at_our_rate():
    from pathlib import Path

    pw = MicrophoneRecorder._argv("pw-record", Path("/tmp/a.wav"), 16000, 5.0)
    assert pw[0] == "pw-record" and "--rate" in pw and "16000" in pw
    assert pw[pw.index("--channels") + 1] == "1"
    pulse = MicrophoneRecorder._argv("parecord", Path("/tmp/a.wav"), 16000, 5.0)
    assert pulse[0] == "parecord" and pulse[pulse.index("--channels") + 1] == "1"
    assert any("wav" in part for part in pulse)


def test_windows_recorder_script_records_and_saves_a_wav():
    """The script drives winmm/MCI, records, then saves the WAV on the marker."""
    from pathlib import Path

    import core.speech as speech

    script = speech._windows_recorder_script(
        Path(r"C:\Users\A B\AppData\Local\Temp\fayd-mic-1\input.wav"),
        Path(r"C:\Users\A B\AppData\Local\Temp\fayd-mic-1\stop"),
        16000,
        120.0,
    )
    assert "winmm.dll" in script  # the Windows capture API
    assert "'record '" in script  # starts capturing
    assert "save" in script  # writes the WAV the recognizer reads
    assert "Test-Path -LiteralPath $marker" in script  # stops on the second click
    assert "input.wav" in script  # the target path is passed through
    # Every PowerShell single-quoted literal must close, or the whole script is a
    # syntax error -- which is exactly how a broken `save` line hides.
    assert _single_quotes_balanced(script), script


def _single_quotes_balanced(script: str) -> bool:
    """True when every PowerShell single-quoted literal in ``script`` is closed."""
    index = 0
    while index < len(script):
        if script[index] != "'":
            index += 1
            continue
        cursor = index + 1
        while cursor < len(script):
            if script[cursor] == "'":
                if cursor + 1 < len(script) and script[cursor + 1] == "'":
                    cursor += 2  # an escaped quote inside the literal
                    continue
                break
            cursor += 1
        if cursor >= len(script):
            return False
        index = cursor + 1
    return True


def test_microphone_install_hint_is_platform_aware(monkeypatch):
    """The next step names the real platform, not a Linux-only command."""
    import core.speech as speech

    monkeypatch.setattr(speech.platform, "system", lambda: "Windows")
    windows = speech.microphone_install_hint()
    assert "arecord" not in windows and "Windows" in windows
    monkeypatch.setattr(speech.platform, "system", lambda: "Darwin")
    assert "brew" in speech.microphone_install_hint()
    monkeypatch.setattr(speech.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        speech.shutil, "which", lambda name: "/usr/bin/apt-get" if name == "apt-get" else None
    )
    assert "sox" in speech.microphone_install_hint()


def test_microphone_start_runs_the_windows_script_with_a_stop_marker(monkeypatch):
    """``start`` launches PowerShell and hands back a marker-stopped recording."""
    import core.speech as speech

    captured: dict[str, object] = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        process = _FakeProcess()
        captured["process"] = process
        return process

    monkeypatch.setattr(speech.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        speech.shutil, "which", lambda name: "powershell" if name == "powershell" else None
    )
    monkeypatch.setattr(speech.subprocess, "Popen", fake_popen)

    recording = MicrophoneRecorder(seconds=30.0).start()
    try:
        argv = captured["argv"]
        assert isinstance(argv, list) and argv[0] == "powershell"
        assert "winmm.dll" in argv[-1]
        assert recording.recorder == "windows-mci"
        stop_path = recording._stop_path
        assert stop_path is not None
        # The recorder only finishes once the marker exists -- exactly what the
        # script's own `Test-Path` loop waits for.
        process = captured["process"]
        assert isinstance(process, _FakeProcess)

        def wait(timeout=None):
            assert stop_path.exists(), "stop() must create the marker before waiting"
            process.returncode = 0
            return 0

        process.wait = wait  # type: ignore[method-assign]
        recording.stop()
        assert stop_path.exists()
    finally:
        import shutil as _shutil

        _shutil.rmtree(recording._folder, ignore_errors=True)
