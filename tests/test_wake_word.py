"""Tests for the wake-word gate and the audio speech gate."""

from __future__ import annotations

import pytest

from core.speech import AudioClip
from core.wake_word import DEFAULT_WAKE_WORDS, AudioSpeechGate, WakeWordDetector


def test_detects_english_wake_word_and_keeps_remainder():
    match = WakeWordDetector().detect("Jarvis, what is porosity?")
    assert match.woke
    assert match.remainder.casefold().startswith("what is porosity")


def test_longest_phrase_wins_so_hey_is_not_left_behind():
    match = WakeWordDetector().detect("hey jarvis what is permeability?")
    assert match.woke
    assert match.phrase == "hey jarvis"
    assert match.remainder.casefold() == "what is permeability"


def test_detects_arabic_wake_word():
    match = WakeWordDetector().detect("يا جارفيس ما هي المسامية")
    assert match.woke
    assert "المسامية" in match.remainder


def test_no_wake_word_leaves_the_utterance_untouched():
    match = WakeWordDetector().detect("what is porosity?")
    assert not match.woke
    assert match.phrase is None
    assert match.remainder == "what is porosity?"


def test_custom_phrases_are_supported():
    detector = WakeWordDetector(phrases=("computer",))
    assert detector.detect("computer, status").woke
    assert not detector.detect("jarvis").woke


def test_detector_requires_at_least_one_phrase():
    with pytest.raises(ValueError):
        WakeWordDetector(phrases=())


def test_default_phrases_include_english_and_arabic():
    assert any(phrase.isascii() for phrase in DEFAULT_WAKE_WORDS)
    assert any(not phrase.isascii() for phrase in DEFAULT_WAKE_WORDS)


def test_speech_gate_rejects_silence_and_short_clips():
    gate = AudioSpeechGate()
    assert not gate.has_speech(AudioClip.silence(1.0))
    assert not gate.has_speech(AudioClip((500, -500), sample_rate=16_000))


def test_speech_gate_accepts_loud_enough_clip():
    gate = AudioSpeechGate()
    loud = AudioClip((1000, -1200, 900) + (0,) * 16_000)
    assert gate.has_speech(loud)
