"""Tests for Jarvis conversation controls (mute/speak/repeat/reset/help)."""

from __future__ import annotations

import pytest

from core.jarvis_controls import ControlIntent, JarvisControls, normalize_utterance


@pytest.mark.parametrize(
    "utterance,intent",
    [
        ("stop", ControlIntent.MUTE),
        ("Stop.", ControlIntent.MUTE),
        ("stop talking", ControlIntent.MUTE),
        ("be quiet", ControlIntent.MUTE),
        ("please stop", ControlIntent.MUTE),
        ("hey jarvis stop", ControlIntent.MUTE),
        ("jarvis stop talking", ControlIntent.MUTE),
        ("speak", ControlIntent.SPEAK),
        ("unmute", ControlIntent.SPEAK),
        ("say that again", ControlIntent.REPEAT),
        ("repeat that", ControlIntent.REPEAT),
        ("please repeat", ControlIntent.REPEAT),
        ("new conversation", ControlIntent.RESET),
        ("start over", ControlIntent.RESET),
        ("save", ControlIntent.SAVE),
        ("save conversation", ControlIntent.SAVE),
        ("sessions", ControlIntent.SESSIONS),
        ("list sessions", ControlIntent.SESSIONS),
        ("help", ControlIntent.HELP),
        ("what can you do?", ControlIntent.HELP),
        ("توقف", ControlIntent.MUTE),
        ("كرر", ControlIntent.REPEAT),
        ("مساعدة", ControlIntent.HELP),
        ("احفظ", ControlIntent.SAVE),
        ("الجلسات", ControlIntent.SESSIONS),
        ("محادثة جديدة", ControlIntent.RESET),
    ],
)
def test_control_phrases_are_recognized(utterance, intent):
    match = JarvisControls().match(utterance)
    assert match is not None
    assert match.intent is intent


@pytest.mark.parametrize(
    "utterance",
    [
        "stop the pump",
        "how do I stop the pump?",
        "repeat the procedure for acidizing",
        "help me understand permeability",
        "what is a new conversation model?",
        "speak about porosity",
        "",
        "   ",
    ],
)
def test_real_questions_are_not_controls(utterance):
    assert JarvisControls().match(utterance) is None


def test_match_keeps_the_original_utterance():
    match = JarvisControls().match("Hey Jarvis, Stop!")
    assert match is not None
    assert match.utterance == "Hey Jarvis, Stop!"
    assert match.phrase == "stop"


def test_normalization_is_case_and_punctuation_insensitive():
    assert normalize_utterance("  Hey JARVIS, Stop!!  ") == "hey jarvis stop"


def test_custom_phrases_replace_the_defaults():
    controls = JarvisControls({ControlIntent.MUTE: ("halt",)})
    assert controls.match("halt").intent is ControlIntent.MUTE
    assert controls.match("stop") is None
