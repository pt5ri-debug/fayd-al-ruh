"""Tests for the VoiceTut-TTS / audio.cpp local Arabic speech engine.

Everything here runs with **no** binary, **no** weights, and **no** network: the
runner is the injection seam, so the engine, the catalog, the gating, and the
voice routing are all exercised against a stub. The one test that needs the real
engine skips when it is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from core import audiocpp_speech as a
from core import speech, voices
from core.voices import ModelVoice

# ----------------------------------------------------------------- helpers


def _fake_install(root: Path) -> Path:
    """A minimal but real-looking install under ``root`` (no weights)."""
    cli = root / "bin" / ("audiocpp_cli.exe" if a._is_windows() else "audiocpp_cli")
    cli.parent.mkdir(parents=True, exist_ok=True)
    cli.write_text("#!/bin/sh\n", encoding="utf-8")
    model = root / "models" / "VoiceTut-TTS-GGUF" / "voicetut-tts-q8_0.gguf"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"GGUF")
    (root / "model.json").write_text(json.dumps({"model": "voicetut-q8_0"}), encoding="utf-8")
    (root / "references.json").write_text(
        json.dumps(
            [
                {
                    "audio_path": "reference_speakers/ahmed.wav",
                    "reference_text": "يعني كل ما تيجي تفكر.",
                    "speaker_name": "Ahmed",
                }
            ]
        ),
        encoding="utf-8",
    )
    directory = a.voices_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ahmed.wav").write_bytes(b"RIFF")
    return root


class _StubRunner:
    """An :class:`AudioCppRunner` that writes a tiny real WAV and records the call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def synthesize(self, text, *, language, reference, reference_text, steps):
        self.calls.append(
            {
                "text": text,
                "language": language,
                "reference": reference,
                "reference_text": reference_text,
                "steps": steps,
            }
        )
        clip = speech.AudioClip((1000, -1000, 2000, -2000), sample_rate=24_000)
        target = Path(self._dir()) / "stub.wav"
        clip.write_wav(target)
        return target

    def _dir(self) -> str:
        import tempfile

        if not hasattr(self, "_scratch"):
            self._scratch = tempfile.mkdtemp(prefix="audiocpp-stub-")
        return self._scratch


# ------------------------------------------------------------------- catalog


def test_specs_are_declarative_and_resolvable():
    assert "voicetut-q8_0" in a.spec_names()
    spec = a.resolve_spec(None)
    assert spec.key == "voicetut-q8_0"  # the recommended default
    assert a.resolve_spec("voicetut-f16").precision == "f16"
    with pytest.raises(ValueError):
        a.resolve_spec("nope")
    assert len(a.spec_summaries()) == len(a.spec_names())


def test_speakers_are_declared_with_a_gender():
    assert len(a.VOICETUT_SPEAKERS) >= 10
    for name, stem, gender in a.VOICETUT_SPEAKERS:
        assert name and stem and gender in ("male", "female")


# ------------------------------------------------------------- availability


def test_status_is_honest_when_nothing_is_installed(tmp_path):
    status = a.audiocpp_status(tmp_path / "empty")
    assert not status.engine_installed
    assert not status.model_cached
    assert not status.available
    assert "install-engine" in status.hint


def test_status_reports_the_real_next_step_at_each_stage(tmp_path):
    root = tmp_path / "data"
    # engine only -> the model is the next step
    _fake_install(root)
    (root / "models" / "VoiceTut-TTS-GGUF" / "voicetut-tts-q8_0.gguf").unlink()
    status = a.audiocpp_status(root)
    assert status.engine_installed and not status.model_cached
    assert "download" in status.hint
    # engine + model but no reference voice -> the voices are the next step
    _fake_install(root)
    (a.voices_dir(root) / "ahmed.wav").unlink()
    status = a.audiocpp_status(root)
    assert status.model_cached and status.voices == 0
    assert "--speakers" in status.hint
    # everything present
    _fake_install(root)
    status = a.audiocpp_status(root)
    assert status.available and status.voices == 1


def test_voice_list_needs_a_reference_clip(tmp_path):
    root = tmp_path / "data"
    _fake_install(root)
    listed = a.voice_list(root)
    assert [v.name for v in listed] == ["Ahmed"]
    voice = listed[0]
    assert voice.key == "model:audio.cpp:Ahmed"
    assert voice.engine == a.AUDIOCPP_ENGINE_NAME
    assert voice.language_code == "ar" and voice.arabic
    assert voice.reference is not None and voice.reference_text


def test_voice_list_is_empty_without_the_model(tmp_path):
    root = tmp_path / "data"
    _fake_install(root)
    (root / "models" / "VoiceTut-TTS-GGUF" / "voicetut-tts-q8_0.gguf").unlink()
    (root / "model.json").unlink()
    assert a.voice_list(root) == ()


def test_find_model_file_prefers_the_registry(tmp_path):
    root = tmp_path / "data"
    _fake_install(root)
    found = a.find_model_file(None, root)
    assert found is not None and found.name == "voicetut-tts-q8_0.gguf"


def test_a_hand_dropped_gguf_is_still_found(tmp_path):
    """A user who fetches the file themselves must not be told nothing is installed."""
    root = tmp_path / "data"
    loose = root / "models" / "somewhere" / "voicetut-tts-q8_0.gguf"
    loose.parent.mkdir(parents=True)
    loose.write_bytes(b"GGUF")
    assert a.find_model_file(None, root) == loose


# ------------------------------------------------------- the user's own voices


def _wav(path: Path, *, silent: bool = False) -> Path:
    """Write a real, decodable WAV (audible, or all-zero when ``silent``)."""
    samples = (0, 0, 0, 0) if silent else (1000, -1000, 2000, -2000)
    speech.AudioClip(samples, sample_rate=24_000).write_wav(path)
    return path


def test_import_voice_copies_a_wav_and_registers_it(tmp_path):
    """An imported clip becomes a voice the engine can clone, listed by name."""
    root = tmp_path / "data"
    _fake_install(root)
    clip = _wav(tmp_path / "my_clip.wav")
    voice = a.import_voice(clip, name="My voice", root=root)
    assert voice.name == "My voice"
    assert (a.voices_dir(root) / voice.clip).is_file()
    # It shows in the model's voice list beside the built-in speakers.
    names = [v.name for v in a.voice_list(root)]
    assert "My voice" in names and "Ahmed" in names
    assert tuple(v.name for v in a.load_imported_voices(root)) == ("My voice",)


def test_import_voice_refuses_an_onnx_with_the_piper_next_step(tmp_path):
    root = tmp_path / "data"
    _fake_install(root)
    onnx = tmp_path / "voice.onnx"
    onnx.write_bytes(b"ONNX")
    with pytest.raises(ValueError, match="Piper"):
        a.import_voice(onnx, root=root)


def test_import_voice_refuses_a_missing_file_and_a_bad_format(tmp_path):
    root = tmp_path / "data"
    _fake_install(root)
    with pytest.raises(ValueError, match="no file"):
        a.import_voice(tmp_path / "nope.wav", root=root)
    text = tmp_path / "notes.txt"
    text.write_text("hello")
    with pytest.raises(ValueError, match="audio clip"):
        a.import_voice(text, root=root)


def test_import_voice_refuses_a_clip_that_decodes_to_silence(tmp_path):
    root = tmp_path / "data"
    _fake_install(root)
    quiet = _wav(tmp_path / "quiet.wav", silent=True)
    with pytest.raises(ValueError, match="silence"):
        a.import_voice(quiet, root=root)
    # A refused clip leaves nothing behind in the registry.
    assert a.load_imported_voices(root) == ()


def test_import_voice_refuses_a_non_wav_with_no_converter(tmp_path, monkeypatch):
    """A non-WAV clip with no ffmpeg/sox names the real next step, never a decode error."""
    import shutil

    root = tmp_path / "data"
    _fake_install(root)
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    clip = tmp_path / "voice.mp3"
    clip.write_bytes(b"ID3")
    with pytest.raises(ValueError, match="ffmpeg"):
        a.import_voice(clip, root=root)


def test_import_voice_converts_a_non_wav_when_a_converter_exists(tmp_path, monkeypatch):
    """With ffmpeg present, another format is converted and then stored as a WAV."""
    import shutil
    import subprocess

    root = tmp_path / "data"
    _fake_install(root)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}" if name == "ffmpeg" else None)

    def _fake_run(argv, **_kwargs):
        # The converter's job is to leave a real WAV at the target path.
        _wav(Path(argv[-1]))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    clip = tmp_path / "voice.mp3"
    clip.write_bytes(b"ID3")
    voice = a.import_voice(clip, name="Converted", root=root)
    stored = a.voices_dir(root) / voice.clip
    assert stored.is_file() and stored.suffix == ".wav"


def test_clone_voice_takes_the_microphone_s_clip_directly(tmp_path):
    """The window hands the recorder's own AudioClip in, with no file round trip."""
    root = tmp_path / "data"
    _fake_install(root)
    clip = speech.AudioClip((1000, -1000, 2000, -2000), sample_rate=24_000)
    voice = a.clone_voice(clip, root=root)
    assert voice.name == "My voice"
    stored = a.voices_dir(root) / voice.clip
    assert stored.is_file()
    assert not speech.AudioClip.read_wav(stored).is_silent
    # A silent recording is refused, exactly like a silent file.
    with pytest.raises(ValueError, match="silence"):
        a.clone_voice(speech.AudioClip((0, 0, 0, 0)), name="Quiet", root=root)


def test_clone_voice_defaults_the_name_and_replaces_a_repeat(tmp_path):
    """Cloning twice under the same name keeps one row, not two."""
    root = tmp_path / "data"
    _fake_install(root)
    first = a.clone_voice(_wav(tmp_path / "a.wav"), root=root)
    assert first.name == "My voice"
    again = a.clone_voice(_wav(tmp_path / "b.wav"), root=root)
    assert again.name == "My voice"
    assert tuple(v.name for v in a.load_imported_voices(root)) == ("My voice",)


def test_remove_imported_voice_drops_the_row_and_its_clip(tmp_path):
    root = tmp_path / "data"
    _fake_install(root)
    voice = a.import_voice(_wav(tmp_path / "clip.wav"), name="Gone", root=root)
    stored = a.voices_dir(root) / voice.clip
    assert stored.is_file()
    assert a.remove_imported_voice("Gone", root=root) is True
    assert not stored.exists()
    assert a.load_imported_voices(root) == ()
    # Removing it again is a no-op that reports so.
    assert a.remove_imported_voice("Gone", root=root) is False


def test_a_voice_whose_clip_vanished_is_dropped_from_the_list(tmp_path):
    """The menu never offers a voice whose reference clip is gone."""
    root = tmp_path / "data"
    _fake_install(root)
    voice = a.import_voice(_wav(tmp_path / "clip.wav"), name="Temp", root=root)
    (a.voices_dir(root) / voice.clip).unlink()
    assert a.load_imported_voices(root) == ()
    assert "Temp" not in [v.name for v in a.voice_list(root)]


def test_a_corrupt_import_registry_is_tolerated(tmp_path):
    """A stale or unreadable registry yields no voices rather than raising."""
    root = tmp_path / "data"
    _fake_install(root)
    a.imported_voices_path(root).write_text("{not json", encoding="utf-8")
    assert a.load_imported_voices(root) == ()


# -------------------------------------------------------------------- gates


def test_normalize_language_maps_the_project_codes_onto_the_engine_s_ids():
    """The engine rejects a bare ``ar``; the project's two-letter code becomes ``arz``."""
    assert a.normalize_language("ar") == "arz"
    assert a.normalize_language("arz") == "arz"
    assert a.normalize_language("en") == "en"
    # Empty or unknown falls back to the model's own language, never an id it refuses.
    assert a.normalize_language("") == a.VOICETUT_LANGUAGE
    assert a.normalize_language("fr") == a.VOICETUT_LANGUAGE
    assert a.normalize_language(None) == a.VOICETUT_LANGUAGE


def test_downloads_and_install_are_denied_by_default(tmp_path):
    with pytest.raises(a.AudioCppUnavailable):
        a.download_model(root=tmp_path)
    with pytest.raises(a.AudioCppUnavailable):
        a.download_speakers(root=tmp_path)
    with pytest.raises(a.AudioCppUnavailable):
        a.install_engine(root=tmp_path)


# ------------------------------------------------------------------- engine


def test_synthesizer_prepares_arabic_text_for_speech():
    """audio.cpp reads Arabic symbols badly, so the text is spelled first."""
    runner = _StubRunner()
    voice = ModelVoice(name="Ahmed", engine=a.AUDIOCPP_ENGINE_NAME, reference=Path("/x.wav"))
    synth = a.AudioCppSynthesizer(voice=voice, runner=runner)
    clip = synth.synthesize("النسبة 25% والعمق 1500 psi.")
    assert not clip.is_silent and clip.sample_rate == 24_000
    sent = str(runner.calls[0]["text"])
    assert "بالمئة" in sent and "باوند لكل بوصة مربعة" in sent
    assert "%" not in sent and "psi" not in sent


def test_synthesizer_passes_the_reference_and_transcript():
    runner = _StubRunner()
    voice = ModelVoice(
        name="Sarah",
        engine=a.AUDIOCPP_ENGINE_NAME,
        reference=Path("/refs/sarah.wav"),
        reference_text="نص مرجعي",
    )
    a.AudioCppSynthesizer(voice=voice, runner=runner, steps=4).synthesize("مرحبا")
    call = runner.calls[0]
    assert call["reference"] == Path("/refs/sarah.wav")
    assert call["reference_text"] == "نص مرجعي"
    assert call["steps"] == 4
    assert call["language"] == a.VOICETUT_LANGUAGE


def test_english_text_is_left_alone():
    runner = _StubRunner()
    a.AudioCppSynthesizer(runner=runner).synthesize("Hello there")
    assert runner.calls[0]["text"] == "Hello there"


def test_synthesizer_refuses_empty_text():
    with pytest.raises(ValueError):
        a.AudioCppSynthesizer(runner=_StubRunner()).synthesize("   ")


def test_synthesizer_falls_back_when_the_engine_fails():
    """A failed engine must degrade to real speech, never to silence."""

    class _Broken:
        def synthesize(self, text, **kwargs):
            raise a.AudioCppUnavailable("boom")

    class _Fallback:
        name = "fallback"

        def __init__(self) -> None:
            self.seen: list[str] = []

        def synthesize(self, text: str) -> speech.AudioClip:
            self.seen.append(text)
            return speech.AudioClip((7, 7, 7))

    fallback = _Fallback()
    synth = a.AudioCppSynthesizer(runner=_Broken(), fallback=fallback)
    clip = synth.synthesize("مرحبا")
    assert not clip.is_silent
    assert fallback.seen == ["مرحبا"]


def test_synthesizer_reports_a_missing_engine_without_a_fallback(tmp_path):
    with pytest.raises(a.AudioCppUnavailable):
        a.AudioCppSynthesizer(root=tmp_path / "empty")


# ------------------------------------------------------------ voice routing


def test_model_voice_key_round_trips():
    voice = ModelVoice(name="Ahmed", engine=a.AUDIOCPP_ENGINE_NAME, gender="male")
    assert voice.key == "model:audio.cpp:Ahmed"
    assert voices.is_model_voice_key(voice.key)
    assert not voices.is_model_voice_key("system:sapi:Microsoft Hoda")
    assert not voices.is_model_voice_key("woman-1")


def test_model_voice_language_code_understands_egyptian_arabic():
    assert ModelVoice(name="x", engine="e", language="arz").language_code == "ar"
    assert ModelVoice(name="x", engine="e", language="ar-SA").language_code == "ar"
    assert ModelVoice(name="x", engine="e", language="en-US").language_code == "en"


def test_unknown_model_voice_key_resolves_to_the_default():
    assert speech._resolve_voice("model:audio.cpp:Nobody") == voices.default_voice()


def test_catalog_engines_ignore_a_model_voice():
    """A model voice is not a catalog voice, so espeak/SAPI must not read it."""
    voice = ModelVoice(name="Ahmed", engine=a.AUDIOCPP_ENGINE_NAME, gender="male")
    assert speech._espeak_voice_options(voice, "ar") == ()
    assert speech._sapi_select_voice(voice) == ""
    assert speech._say_voice_options(voice, "ar") == ()
    assert speech._sapi_rate(voice) == 0
    assert speech._onecore_rate(voice) == 1.0


def test_model_voice_routes_the_reader_to_its_engine(monkeypatch):
    """Selecting a model voice is what reaches audio.cpp -- it is not the default.

    Discovery is left exactly as it was (audio.cpp is never the silent default), so
    the *only* way a reading reaches this engine is by the user picking one of its
    voices; that choice is routed through ``_engine_for_voice``.
    """
    monkeypatch.setattr(a, "available_synthesizer", lambda *args, **kwargs: True)
    voice = ModelVoice(name="Ahmed", engine=a.AUDIOCPP_ENGINE_NAME)
    assert speech._engine_for_voice(voice) == a.AUDIOCPP_ENGINE_NAME
    # A model voice of an engine that is not installed routes nowhere.
    monkeypatch.setattr(a, "available_synthesizer", lambda *args, **kwargs: False)
    assert speech._engine_for_voice(voice) is None
    # And discovery still never silently becomes this engine.
    assert speech.discover_synthesizer_name() != a.AUDIOCPP_ENGINE_NAME


# -------------------------------------------------------------------- CLI


def test_cli_status_reports_and_exits_nonzero_when_missing(tmp_path, monkeypatch, capsys):
    from cli import audio as cli_audio

    monkeypatch.setattr(cli_audio, "audiocpp_status", lambda *args, **kwargs: a.audiocpp_status(tmp_path / "e"))
    assert cli_audio.main(["status"]) == 3
    assert "audio.cpp" in capsys.readouterr().out


def test_cli_download_refuses_without_approval(capsys):
    from cli import audio as cli_audio

    assert cli_audio.main(["download"]) == 4
    assert "--approve" in capsys.readouterr().out
    assert cli_audio.main(["install-engine"]) == 4


def test_cli_voices_lists_the_installed_voices(tmp_path, monkeypatch, capsys):
    from cli import audio as cli_audio

    root = tmp_path / "data"
    _fake_install(root)
    monkeypatch.setattr(cli_audio, "voice_list", lambda: a.voice_list(root))
    monkeypatch.setattr(cli_audio, "audiocpp_status", lambda *a_, **k: a.audiocpp_status(root))
    assert cli_audio.main(["voices"]) == 0
    out = capsys.readouterr().out
    assert "model:audio.cpp:Ahmed" in out


def test_cli_say_rejects_an_unknown_voice(capsys):
    from cli import audio as cli_audio

    assert cli_audio.main(["say", "hello", "--voice", "model:audio.cpp:Nobody"]) == 3
    assert "voices" in capsys.readouterr().out


def test_cli_import_clone_and_remove_round_trip(tmp_path, monkeypatch, capsys):
    """The CLI is the headless twin of the window's clone/import buttons."""
    from cli import audio as cli_audio

    root = tmp_path / "data"
    _fake_install(root)
    # Route the CLI's own calls at the temp root (the window uses the same helpers).
    monkeypatch.setattr(cli_audio, "import_voice", lambda path, **kw: a.import_voice(path, root=root, **kw))
    monkeypatch.setattr(cli_audio, "clone_voice", lambda path, **kw: a.clone_voice(path, root=root, **kw))
    monkeypatch.setattr(cli_audio, "load_imported_voices", lambda: a.load_imported_voices(root))
    monkeypatch.setattr(cli_audio, "remove_imported_voice", lambda name: a.remove_imported_voice(name, root=root))

    clip = _wav(tmp_path / "clip.wav")
    assert cli_audio.main(["import", str(clip), "--name", "Studio"]) == 0
    assert "Studio" in capsys.readouterr().out
    assert cli_audio.main(["clone", str(clip)]) == 0
    assert "My voice" in capsys.readouterr().out
    assert cli_audio.main(["my-voices"]) == 0
    listed = capsys.readouterr().out
    assert "Studio" in listed and "My voice" in listed
    assert cli_audio.main(["remove-voice", "Studio"]) == 0
    assert cli_audio.main(["remove-voice", "Studio"]) == 3


def test_cli_import_reports_a_bad_clip(tmp_path, monkeypatch, capsys):
    from cli import audio as cli_audio

    root = tmp_path / "data"
    _fake_install(root)
    monkeypatch.setattr(cli_audio, "import_voice", lambda path, **kw: a.import_voice(path, root=root, **kw))
    onnx = tmp_path / "v.onnx"
    onnx.write_bytes(b"ONNX")
    assert cli_audio.main(["import", str(onnx)]) == 2
    assert "Piper" in capsys.readouterr().out


def test_cli_my_voices_is_empty_before_anything_is_added(tmp_path, monkeypatch, capsys):
    from cli import audio as cli_audio

    monkeypatch.setattr(cli_audio, "load_imported_voices", lambda: ())
    assert cli_audio.main(["my-voices"]) == 0
    assert "No voices" in capsys.readouterr().out


# ------------------------------------------------------------- honest deps


def test_the_engine_claims_no_pip_dependency():
    """The engine is a native program and downloads with ``urllib``.

    Reporting a Python package it does not use would tell a user to run a pip
    install that cannot fix anything -- and the project's honesty rule forbids
    exactly that, so ``extra_installed`` is unconditionally true and the feature's
    import list is empty.
    """
    from core.optional_deps import FEATURE_AUDIOCPP, feature

    assert a.extra_installed() is True
    assert feature(FEATURE_AUDIOCPP).import_names == ()


def test_install_hint_names_the_engine_not_a_pip_install(tmp_path, monkeypatch):
    """With nothing installed the next step names the engine, never a pip command."""
    monkeypatch.setattr(a, "audiocpp_models_dir", lambda: tmp_path / "audio.cpp")

    hint = a.install_hint()

    assert "pip install" not in hint
    assert "cli audio" in hint


def test_feature_status_reports_the_engine_and_model(tmp_path, monkeypatch):
    """The feature is available only when the engine *and* a model are really there."""
    from core import optional_deps

    monkeypatch.setattr(a, "audiocpp_models_dir", lambda: tmp_path / "audio.cpp")
    empty = optional_deps.status(optional_deps.FEATURE_AUDIOCPP)
    assert empty.python_ok is True
    assert empty.system_ok is False, "no engine installed yet, so it is not available"

    _fake_install(tmp_path / "audio.cpp")
    assert optional_deps.status(optional_deps.FEATURE_AUDIOCPP).available is True


# --------------------------------------------------------------- real engine


def test_real_engine_renders_arabic_audio():
    """The real engine, when installed, produces non-silent Arabic speech."""
    status = a.audiocpp_status()
    if not status.available or status.voices == 0:
        pytest.skip("the audio.cpp engine, model, and voices are not all installed")
    voice = a.voice_list()[0]
    synth = a.AudioCppSynthesizer(voice=voice)
    clip = synth.synthesize("إزاي حالك النهارده؟")
    assert clip.duration_seconds > 0.2
    assert not clip.is_silent
    assert clip.sample_rate == a.AUDIOCPP_SAMPLE_RATE
