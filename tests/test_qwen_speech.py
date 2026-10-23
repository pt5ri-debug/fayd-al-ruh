"""Tests for the Qwen3 speech engines (ASR + TTS) and their wiring.

Every test injects a runner or a fake, so the suite exercises the real engine
code paths with **no** ``torch``, no ``qwen-asr``/``qwen-tts``, and no model
weights -- exactly the seam the engines expose. The optional packages are absent
on CI, and these tests must pass regardless.
"""

from __future__ import annotations

import numpy as np
import pytest

from core import qwen_speech
from core.qwen_speech import (
    ASR_CATALOG,
    QWEN_TTS_LANGUAGES,
    TTS_CATALOG,
    QwenAsrRecognizer,
    QwenSpeechUnavailable,
    QwenTtsSynthesizer,
    asr_spec_names,
    find_model_dir,
    model_is_cached,
    qwen_asr_status,
    qwen_tts_status,
    qwen_tts_supports,
    resolve_asr_spec,
    resolve_tts_spec,
    tts_spec_names,
)
from core.speech import (
    AudioClip,
    load_recognizer,
    load_synthesizer,
)


class _FakeAsrRunner:
    """Records what the engine passed it and returns a fixed transcript."""

    def __init__(self, transcript: str = "hello from qwen") -> None:
        self.transcript = transcript
        self.calls: list[tuple[int, int, str | None]] = []

    def transcribe(self, samples, sample_rate: int, *, language: str | None) -> str:
        self.calls.append((len(samples), sample_rate, language))
        return self.transcript


class _FakeTtsRunner:
    """Returns a short 24 kHz tone so resampling to 16 kHz is exercised."""

    def __init__(self, sample_rate: int = 24_000) -> None:
        self.sample_rate = sample_rate
        self.calls: list[tuple[str, str | None, str | None, str | None]] = []

    def synthesize(self, text: str, *, language, speaker, instruct):
        self.calls.append((text, language, speaker, instruct))
        samples = np.zeros(self.sample_rate // 2, dtype=np.float32)
        return samples, self.sample_rate


# --------------------------------------------------------------------- specs


def test_catalogs_expose_default_and_light_models():
    assert asr_spec_names() == ("qwen3-asr-1.7b", "qwen3-asr-0.6b")
    assert tts_spec_names() == ("qwen3-tts-1.7b", "qwen3-tts-0.6b")
    assert resolve_asr_spec(None).repo == "Qwen/Qwen3-ASR-1.7B"
    assert resolve_tts_spec(None).repo == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert resolve_asr_spec("Qwen3-ASR-0.6B").key == "qwen3-asr-0.6b"


def test_unknown_spec_name_raises():
    with pytest.raises(ValueError):
        resolve_asr_spec("nope")
    with pytest.raises(ValueError):
        resolve_tts_spec("nope")


def test_qwen_tts_language_support_states_the_arabic_gap():
    # Honesty rule: Qwen3-TTS does not speak Arabic, and the catalog says so.
    assert "Arabic" not in QWEN_TTS_LANGUAGES
    assert qwen_tts_supports("en") is True
    assert qwen_tts_supports("ar") is False


# ------------------------------------------------------------------- engines


def test_asr_recognizer_uses_the_injected_runner():
    runner = _FakeAsrRunner("  permeability  ")
    recognizer = QwenAsrRecognizer(runner=runner, language="Arabic")
    text = recognizer.transcribe(AudioClip.silence(0.5))
    assert text == "permeability"
    assert recognizer.name == "qwen3-asr"
    count, rate, language = runner.calls[0]
    assert count == 8000 and rate == 16_000 and language == "Arabic"


def test_asr_recognizer_rejects_an_empty_transcript():
    recognizer = QwenAsrRecognizer(runner=_FakeAsrRunner("   "))
    with pytest.raises(ValueError):
        recognizer.transcribe(AudioClip.silence(0.5))


def test_tts_synthesizer_resamples_to_the_project_rate():
    runner = _FakeTtsRunner(sample_rate=24_000)
    synth = QwenTtsSynthesizer(runner=runner, speaker="Aiden")
    clip = synth.synthesize("Hello there")
    assert clip.sample_rate == 16_000
    assert clip.duration_seconds == pytest.approx(0.5, abs=0.05)
    assert runner.calls[0][1] == "English"
    assert runner.calls[0][2] == "Aiden"


def test_tts_synthesizer_refuses_arabic_without_a_fallback():
    synth = QwenTtsSynthesizer(runner=_FakeTtsRunner())
    with pytest.raises(ValueError, match="cannot speak"):
        synth.synthesize("السلام عليكم")


def test_tts_synthesizer_uses_the_fallback_for_arabic():
    class _RecordingFallback:
        name = "fake-system"

        def __init__(self) -> None:
            self.seen: list[str] = []

        def synthesize(self, text: str) -> AudioClip:
            self.seen.append(text)
            return AudioClip((1, 2, 3))

    fallback = _RecordingFallback()
    synth = QwenTtsSynthesizer(runner=_FakeTtsRunner(), fallback=fallback)
    clip = synth.synthesize("السلام عليكم")
    assert fallback.seen == ["السلام عليكم"]
    assert clip.samples == (1, 2, 3)


def test_tts_synthesizer_rejects_empty_text():
    synth = QwenTtsSynthesizer(runner=_FakeTtsRunner())
    with pytest.raises(ValueError):
        synth.synthesize("   ")


# ---------------------------------------------------------------- discovery


def test_status_reports_the_next_step_when_absent(tmp_path):
    asr = qwen_asr_status(tmp_path)
    tts = qwen_tts_status(tmp_path)
    # The optional packages are absent on CI, so both report "not available" and
    # a real, actionable next step rather than pretending.
    assert asr.available is False and tts.available is False
    assert "requirements-qwen-asr.txt" in asr.hint
    assert "does not speak Arabic" in tts.hint


def test_the_two_engines_have_separate_requirements_files():
    """The engines cannot share an environment, so they must not share a file.

    ``qwen-asr`` pins ``transformers==4.57.6`` and ``qwen-tts`` pins
    ``transformers==4.57.3``. One file listing both leaves pip no common
    ``transformers``, so it backtracks to an old release that caps ``numpy<2.0``
    and then *builds* numpy 1.26.4 from source -- which fails on Windows with a
    long-path ``OSError``. Two files, one engine each, keep the resolution direct.
    """
    from core.optional_deps import (
        FEATURE_QWEN_ASR,
        FEATURE_QWEN_TTS,
        feature,
        requirements_path,
    )

    asr = feature(FEATURE_QWEN_ASR)
    tts = feature(FEATURE_QWEN_TTS)
    assert asr.requirements_file != tts.requirements_file
    for spec in (asr, tts):
        assert requirements_path(spec.key).exists(), f"{spec.requirements_file} must ship"


def test_each_requirements_file_names_only_its_own_engine():
    """A file must not pull the other engine in, or the conflict comes back."""
    from core.optional_deps import (
        FEATURE_QWEN_ASR,
        FEATURE_QWEN_TTS,
        requirements_path,
    )

    asr_text = requirements_path(FEATURE_QWEN_ASR).read_text(encoding="utf-8").lower()
    tts_text = requirements_path(FEATURE_QWEN_TTS).read_text(encoding="utf-8").lower()

    def requirements(text: str) -> list[str]:
        # Only real requirement lines: comments explain the conflict by name.
        return [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    assert "qwen-asr" in requirements(asr_text)
    assert not any("qwen-tts" in line for line in requirements(asr_text))
    assert "qwen-tts" in requirements(tts_text)
    assert not any("qwen-asr" in line for line in requirements(tts_text))


def test_the_requirements_do_not_restate_transformers():
    """A loose ``transformers`` pin is what made pip backtrack into a numpy build.

    Each engine pins the exact ``transformers`` it was built against, so restating
    a range here only widens the search space back to the numpy<2 releases.
    """
    from core.optional_deps import (
        FEATURE_QWEN_ASR,
        FEATURE_QWEN_TTS,
        requirements_path,
    )

    for key in (FEATURE_QWEN_ASR, FEATURE_QWEN_TTS):
        text = requirements_path(key).read_text(encoding="utf-8")
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert not any(line.startswith("transformers") for line in lines), (
            f"{requirements_path(key).name} must let the engine own its transformers pin"
        )


def test_model_is_cached_reads_a_model_folder(tmp_path):
    assert model_is_cached("asr", tmp_path) is False
    folder = tmp_path / "asr"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    assert model_is_cached("asr", tmp_path) is True
    assert find_model_dir("asr", tmp_path) == folder


# ------------------------------------------------------------- factory wiring


def test_speech_factory_builds_a_qwen_asr_recognizer_with_a_runner():
    runner = _FakeAsrRunner("cased hole")
    recognizer = load_recognizer("qwen3-asr", runner=runner)
    assert recognizer.name == "qwen3-asr"
    assert recognizer.transcribe(AudioClip.silence(0.2)) == "cased hole"


def test_speech_factory_builds_a_qwen_tts_synthesizer_with_a_runner():
    runner = _FakeTtsRunner()
    synth = load_synthesizer("qwen3-tts", runner=runner, speaker="Serena")
    assert synth.name == "qwen3-tts"
    assert synth.synthesize("Hello").sample_rate == 16_000


def test_named_qwen_engine_without_extra_raises_a_clear_error():
    # No runner, no extra: the factory must say what is missing, not fail silently.
    with pytest.raises(ValueError, match="not ready"):
        load_recognizer("qwen3-asr")
    with pytest.raises(ValueError, match="not ready"):
        load_synthesizer("qwen3-tts")


def test_qwen_tts_explicit_choice_reads_arabic_through_the_system_engine(monkeypatch):
    """Choosing qwen3-tts must still voice Arabic, via the system engine fallback."""
    import core.speech as speech

    monkeypatch.setattr(
        speech,
        "_system_synthesizer",
        lambda **_kw: _FakeSystemSynth(),
    )
    synth = load_synthesizer("qwen3-tts", runner=_FakeTtsRunner())
    assert synth.name == "qwen3-tts"
    clip = synth.synthesize("السلام عليكم")
    assert clip.samples == (7, 7)  # from the fake system engine


class _FakeSystemSynth:
    name = "fake-system"

    def synthesize(self, text: str) -> AudioClip:
        return AudioClip((7, 7))


def test_download_is_denied_by_default(tmp_path):
    with pytest.raises(QwenSpeechUnavailable):
        qwen_speech.download_model("asr", root=tmp_path)
    with pytest.raises(QwenSpeechUnavailable):
        qwen_speech.download_model("tts", root=tmp_path)


def test_download_rejects_an_unknown_kind(tmp_path):
    with pytest.raises(ValueError):
        qwen_speech.download_model("video", root=tmp_path)


# ------------------------------------------------------------------- extras


def test_extra_install_status_never_imports_torch_eagerly():
    # The module must import with no torch present; the check is find_spec only.
    assert qwen_speech.qwen_asr_extra_installed() in (True, False)
    assert qwen_speech.qwen_tts_extra_installed() in (True, False)
    assert set(ASR_CATALOG) and set(TTS_CATALOG)


def test_transformers_alone_does_not_count_as_asr_ready(monkeypatch):
    """`transformers` is a dependency of `qwen-asr`, not a substitute for it.

    The `[image]` extra pulls torch + transformers, so a machine can have both and
    still have no `qwen-asr`. Treating that as "ready" is what made the microphone
    fail with a raw `ModuleNotFoundError: No module named 'qwen_asr'` after the
    user had already recorded.
    """
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in ("torch", "transformers"):
            return object()
        if name == "qwen_asr":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    assert qwen_speech.qwen_asr_extra_installed() is False
    assert qwen_speech.available_recognizer() is False


def test_asr_runner_names_the_install_step_when_the_extra_is_broken(tmp_path, monkeypatch):
    """A missing `qwen_asr` must surface as the install step, not a bare ImportError."""
    import sys

    from core.qwen_speech import TransformersAsrRunner

    # Force the import to fail even on a machine that has the extra installed.
    monkeypatch.setitem(sys.modules, "qwen_asr", None)
    runner = TransformersAsrRunner(tmp_path)
    with pytest.raises(QwenSpeechUnavailable) as caught:
        runner._load()
    message = str(caught.value)
    assert "Qwen3-ASR" in message
    assert "pip install" in message
    assert "requirements-qwen-asr.txt" in message


def test_tts_runner_names_the_install_step_when_the_extra_is_broken(tmp_path, monkeypatch):
    import sys

    from core.qwen_speech import QwenTtsRunner

    monkeypatch.setitem(sys.modules, "qwen_tts", None)
    runner = QwenTtsRunner(tmp_path)
    with pytest.raises(QwenSpeechUnavailable) as caught:
        runner._load()
    message = str(caught.value)
    assert "pip install" in message
    assert "requirements-qwen-tts.txt" in message


# ------------------------------------------------------- menu / GUI wiring


def test_cli_menu_exposes_the_qwen_speech_entries():
    from cli.menu import run_qwen_speech, visible_menu

    entries = [(title, action, tuple(argv)) for title, action, argv in visible_menu()]
    assert (
        "Qwen3 speech: status of the local ASR + TTS models",
        run_qwen_speech,
        ("qwen-asr", "status"),
    ) in entries
    assert (
        "Qwen3 speech: download the local ASR model (~3.4 GB)",
        run_qwen_speech,
        ("qwen-asr", "download", "--approve"),
    ) in entries
    assert (
        "Qwen3 speech: download the local TTS model (~3.4 GB)",
        run_qwen_speech,
        ("qwen-tts", "download", "--approve"),
    ) in entries


def test_gui_actions_expose_the_qwen_speech_entries():
    from gui.actions import action_by_key, run_qwen_speech, visible_actions

    keys = {action.key for action in visible_actions()}
    assert {"qwen-speech", "qwen-speech-asr", "qwen-speech-tts"} <= keys
    action = action_by_key("qwen-speech")
    assert action.command is run_qwen_speech
    assert action.argv == ("qwen-asr", "status")


def test_cli_dispatches_the_qwen_commands(capsys):
    from cli.__main__ import main

    # `qwen-tts models` lists the catalog and needs no runtime or weights.
    assert main(["qwen-tts", "models"]) == 0
    out = capsys.readouterr().out
    assert "Qwen3-TTS 1.7B CustomVoice" in out

    # `qwen-asr status` reports honestly and exits 3 while the extra is absent.
    code = main(["qwen-asr", "status"])
    assert code == 3
    out = capsys.readouterr().out
    assert "Qwen3-ASR" in out


def test_cli_qwen_speech_help_is_available():
    import io
    from contextlib import redirect_stdout

    from cli.qwen_speech import main

    buffer = io.StringIO()
    with pytest.raises(SystemExit), redirect_stdout(buffer):
        main(["--help"])
    assert "qwen-asr" in buffer.getvalue() and "qwen-tts" in buffer.getvalue()


# --------------------------------------------------- the voice menu by source


def test_voice_list_is_empty_until_the_runtime_and_weights_are_present(tmp_path):
    """A Qwen3-TTS voice is listed only when the model can really speak here."""
    from core.qwen_speech import voice_list

    assert voice_list(tmp_path) == ()


def test_voice_list_names_the_built_in_speakers(monkeypatch, tmp_path):
    """With the runtime and weights present, each speaker is a ModelVoice."""
    from types import SimpleNamespace

    from core import qwen_speech
    from core.qwen_speech import TTS_ENGINE_NAME, voice_list

    monkeypatch.setattr(
        qwen_speech, "qwen_tts_status", lambda root=None: SimpleNamespace(available=True)
    )
    listed = voice_list(tmp_path)
    assert [v.name for v in listed] == ["Serena", "Aiden"]
    for voice in listed:
        assert voice.engine == TTS_ENGINE_NAME
        assert voice.source == "qwen3-tts"
        assert voice.language == "en" and not voice.arabic
        assert voice.key == f"model:qwen3-tts:{voice.name}"
        assert voice.title and voice.title.endswith("(English)")


def test_speech_factory_uses_the_chosen_speaker_from_a_model_voice():
    """Choosing a Qwen3-TTS voice picks that built-in speaker for the reading."""
    from core.qwen_speech import TTS_ENGINE_NAME
    from core.voices import ModelVoice

    runner = _FakeTtsRunner()
    chosen = ModelVoice(name="Aiden", engine=TTS_ENGINE_NAME, language="en")
    synth = load_synthesizer("qwen3-tts", runner=runner, voice=chosen)
    assert synth.speaker == "Aiden"
