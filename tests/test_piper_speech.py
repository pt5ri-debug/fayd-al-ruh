"""Tests for the Piper neural TTS engine (``core.piper_speech``).

The engine is exercised with an *injected runner*, so the suite needs neither the
``[piper]`` extra nor a downloaded voice: the seam is the point. A test that
actually renders audio is skipped when the runtime or a voice is absent.
"""

from __future__ import annotations

import pytest

from core.piper_speech import (
    ARABIC_VOICE_KEY,
    PIPER_ENGINE_NAME,
    PIPER_VOICES,
    PiperSynthesizer,
    PiperUnavailable,
    available_synthesizer,
    find_model_file,
    length_scale,
    noise_scale,
    noise_w_scale,
    piper_extra_installed,
    piper_status,
    resolve_spec,
    spec_for_voice,
    spec_names,
    spec_summaries,
)
from core.speech import load_synthesizer
from core.voices import CATEGORY_MAN, CATEGORY_WOMAN, find_voice, voice_keys


class _EchoRunner:
    """A runner that records what it was asked and returns a tiny waveform."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def synthesize(self, text, *, speaker, length_scale, noise_scale, noise_w_scale):
        self.calls.append(
            {
                "text": text,
                "speaker": speaker,
                "length_scale": length_scale,
                "noise_scale": noise_scale,
                "noise_w_scale": noise_w_scale,
            }
        )
        return [0.0, 0.5, -0.5, 0.25], 22_050


def test_catalog_is_well_formed_and_covers_arabic_and_english():
    keys = spec_names()
    assert len(keys) == len(set(keys))
    languages = {spec.language for spec in PIPER_VOICES}
    assert languages == {"ar", "en"}
    assert ARABIC_VOICE_KEY in keys
    # Every summary row is a 5-tuple keyed by a real voice.
    for row in spec_summaries():
        assert row[0] in keys
        assert row[2] in {"ar", "en"}
        assert row[3] > 0


def test_resolve_spec_defaults_and_rejects_unknown():
    assert resolve_spec(None).key == ARABIC_VOICE_KEY
    assert resolve_spec("en_US-amy-medium").language == "en"
    with pytest.raises(ValueError):
        resolve_spec("not-a-voice")


def test_spec_for_voice_maps_each_category_to_a_distinct_real_voice():
    # Arabic has one official voice, so every Arabic reading uses it.
    for key in voice_keys():
        assert spec_for_voice(find_voice(key), "ar").key == ARABIC_VOICE_KEY
    # English maps four voices of one category onto four different real voices.
    men = [spec_for_voice(find_voice(f"man-{i}"), "en").key for i in range(1, 5)]
    assert len(set(men)) == 4
    women = [spec_for_voice(find_voice(f"woman-{i}"), "en").key for i in range(1, 5)]
    assert len(set(women)) == 4
    assert set(men).isdisjoint(women)
    # A category is honoured: a man's voice does not become a woman's, and the
    # boys'/girls' voices use the lighter female voices (Piper ships no child
    # voices), never a man's.
    assert spec_for_voice(find_voice("man-1"), "en").category == CATEGORY_MAN
    assert spec_for_voice(find_voice("woman-1"), "en").category == CATEGORY_WOMAN
    assert spec_for_voice(find_voice("girl-1"), "en").category == CATEGORY_WOMAN
    assert spec_for_voice(find_voice("boy-1"), "en").key not in men


def test_prosody_maps_the_catalog_onto_piper_controls():
    # A slower catalog voice reads slower (a larger length_scale), and every
    # value stays inside Piper's own safe range.
    slow = find_voice("man-1")
    assert 0.7 <= length_scale(slow) <= 1.4
    assert 0.4 <= noise_scale(slow) <= 1.2
    assert 0.5 <= noise_w_scale(slow) <= 1.4
    # No voice at all still yields Piper's neutral defaults.
    assert length_scale(None) == 1.0
    assert noise_scale(None) == 0.667
    assert noise_w_scale(None) == 0.8


def test_synthesizer_uses_the_injected_runner_and_its_prosody():
    runner = _EchoRunner()
    voice = find_voice("man-1")
    synth = PiperSynthesizer(runner=runner, voice=voice)
    clip = synth.synthesize("Hello there")
    assert synth.name == PIPER_ENGINE_NAME
    assert clip.sample_rate == 16_000
    assert clip.duration_seconds > 0
    assert not clip.is_silent
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["length_scale"] == length_scale(voice)
    assert call["noise_scale"] == noise_scale(voice)


def test_synthesizer_rejects_empty_text():
    synth = PiperSynthesizer(runner=_EchoRunner(), voice=find_voice("man-1"))
    with pytest.raises(ValueError):
        synth.synthesize("   ")


class _RecordingFallback:
    name = "recording"

    def __init__(self) -> None:
        self.texts: list[str] = []

    def synthesize(self, text):
        from core.speech import AudioClip

        self.texts.append(text)
        return AudioClip.silence(0.2)


def test_fallback_reads_the_language_whose_voice_is_missing(tmp_path):
    # Only the Arabic voice is present; an English reading is handed to the
    # fallback engine instead of failing, so one voice does not silence the other.
    (tmp_path / f"{ARABIC_VOICE_KEY}.onnx").write_bytes(b"stub")
    fallback = _RecordingFallback()
    synth = PiperSynthesizer(root=tmp_path, voice=find_voice("man-1"), fallback=fallback)
    clip = synth.synthesize("This sentence is English.")
    assert fallback.texts == ["This sentence is English."]
    assert clip.duration_seconds == pytest.approx(0.2)


def test_missing_voice_without_a_fallback_raises_with_the_next_step(tmp_path):
    (tmp_path / f"{ARABIC_VOICE_KEY}.onnx").write_bytes(b"stub")
    synth = PiperSynthesizer(root=tmp_path, voice=find_voice("man-1"))
    with pytest.raises(PiperUnavailable):
        synth.synthesize("English with no English voice")


def test_synthesizer_raises_when_no_voice_is_present(tmp_path):
    # A fresh root has no voice file, so construction reports the real next step.
    with pytest.raises(PiperUnavailable):
        PiperSynthesizer(root=tmp_path)


def test_status_reports_what_is_present_and_a_next_step(tmp_path):
    status = piper_status(tmp_path)
    assert status.extra_installed == piper_extra_installed()
    assert status.model_cached is False
    assert status.hint
    assert status.available is False
    assert available_synthesizer(tmp_path) is False


def test_find_model_file_reads_a_hand_dropped_voice(tmp_path):
    voice_file = tmp_path / "en_US-amy-medium.onnx"
    voice_file.write_bytes(b"not a real model")
    assert find_model_file(None, tmp_path) == voice_file
    assert find_model_file("en_US-amy-medium", tmp_path) == voice_file
    assert find_model_file("en_US-ryan-high", tmp_path) is None


def test_load_synthesizer_refuses_piper_without_a_voice():
    # 'piper' is a real engine name now, but it must not fake audio: with no voice
    # cached it raises with the next step rather than returning silence.
    if available_synthesizer():
        pytest.skip("a Piper voice is installed on this machine")
    with pytest.raises(ValueError):
        load_synthesizer(PIPER_ENGINE_NAME)


def test_load_synthesizer_builds_piper_from_an_injected_runner():
    # The factory path is exercised with the runner seam, so no extra is needed.
    runner = _EchoRunner()
    synth = load_synthesizer(PIPER_ENGINE_NAME, runner=runner, voice=find_voice("woman-2"))
    clip = synth.synthesize("Testing the factory path.")
    assert synth.name == PIPER_ENGINE_NAME
    assert clip.duration_seconds > 0
    assert len(runner.calls) == 1


@pytest.mark.skipif(
    not available_synthesizer(),
    reason="Piper runtime or a voice is not installed",
)
def test_real_piper_renders_audio():
    # Render in whichever language has a voice installed, so the test is honest
    # about what is present rather than assuming English.
    synth = PiperSynthesizer(voice=find_voice("man-1"))
    sample = "مرحبا، هذا اختبار حقيقي." if find_model_file(ARABIC_VOICE_KEY) else "Hello, real Piper."
    clip = synth.synthesize(sample)
    assert clip.duration_seconds > 0.2
    assert not clip.is_silent
