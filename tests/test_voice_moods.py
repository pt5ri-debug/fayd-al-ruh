"""The emotional-state catalog: the moods a reading voice can be put into.

The feature this pins: a voice can be read *how* as well as *by whom*. Each mood
carries a small, bounded prosody (a pitch offset, a speed multiplier, an
intonation-range offset, a word-gap offset, an energy multiplier, and a
natural-language instruction for an instructable engine), the numeric part folds
onto a catalog voice without leaving the catalog's own limits, and every engine
reports honestly how much of a mood it can really apply.

All of this runs with no display and no speech engine: a mood is settings, and the
catalog is the one declarative place they live.
"""

from __future__ import annotations

import pytest
from core import voice_moods as moods
from core import voices


def test_the_catalog_covers_the_states_people_ask_for() -> None:
    keys = moods.mood_keys()
    # The states named in the request, and the ones a reader expects around them.
    for expected in (
        "neutral", "happy", "sad", "excited", "rapturous", "in_love", "angry",
        "calm", "tender", "serious", "surprised", "fearful", "hopeful",
        "nostalgic", "proud", "playful", "whisper", "storytelling", "formal",
        "empathetic",
    ):
        assert expected in keys, expected
    # A generous catalog, and every key unique.
    assert len(keys) >= 20
    assert len(set(keys)) == len(keys)


def test_every_mood_has_both_languages_and_an_honest_note() -> None:
    for mood in moods.MOODS:
        assert mood.label_en.strip()
        assert mood.label_ar.strip()
        assert mood.description_en.strip()
        assert mood.description_ar.strip()
        # The note says what the mood *does*, not what the speaker feels.
        assert mood.description_en.endswith(".")


def test_neutral_changes_nothing_and_the_others_do() -> None:
    neutral = moods.find_mood("neutral")
    assert neutral.neutral
    assert neutral.prosody.neutral
    for mood in moods.MOODS:
        if mood.key == "neutral":
            continue
        assert not mood.neutral, mood.key


def test_prosody_is_bounded_and_rejects_an_extreme_entry() -> None:
    with pytest.raises(ValueError):
        moods.Prosody(pitch_delta=999)
    with pytest.raises(ValueError):
        moods.Prosody(speed_scale=9.0)
    with pytest.raises(ValueError):
        moods.Prosody(energy=0.0)
    # A well-formed prosody reports every field in its primitive form.
    prosody = moods.find_mood("happy").prosody
    primitive = prosody.to_primitive()
    assert primitive == {
        "pitch_delta": prosody.pitch_delta,
        "speed_scale": prosody.speed_scale,
        "range_delta": prosody.range_delta,
        "gap_delta": prosody.gap_delta,
        "energy": prosody.energy,
        "instruct": prosody.instruct,
    }


def test_find_mood_refuses_an_unknown_key_and_resolve_falls_back() -> None:
    with pytest.raises(ValueError):
        moods.find_mood("nope")
    assert moods.resolve_mood("nope").key == moods.DEFAULT_MOOD_KEY
    assert moods.resolve_mood(None).key == moods.DEFAULT_MOOD_KEY
    assert moods.resolve_mood("sad").key == "sad"


def test_apply_mood_shifts_a_catalog_voice_within_its_limits() -> None:
    voice = voices.find_voice("woman-1")
    happy = moods.apply_mood(voice, moods.find_mood("happy"))
    assert happy.pitch > voice.pitch
    assert happy.speed >= voice.speed
    assert happy.pitch_range > voice.pitch_range
    # The voice's identity is kept; only the delivery settings changed.
    assert happy.key == voice.key
    assert happy.variant == voice.variant


def test_apply_mood_accepts_a_bare_prosody_and_leaves_neutral_alone() -> None:
    voice = voices.find_voice("man-2")
    assert moods.apply_mood(voice, moods.Prosody()) is voice
    assert moods.apply_mood(voice, moods.find_mood("neutral")) is voice
    shifted = moods.apply_mood(voice, moods.find_mood("sad").prosody)
    assert shifted.pitch < voice.pitch


def test_apply_mood_never_leaves_the_catalog_limits() -> None:
    for key in moods.mood_keys():
        for voice_key in voices.voice_keys():
            voice = voices.find_voice(voice_key)
            shifted = moods.apply_mood(voice, moods.find_mood(key))
            assert 0 <= shifted.pitch <= voices.MAX_PITCH
            assert 1 <= shifted.speed <= voices.MAX_SPEED
            assert 0 <= shifted.pitch_range <= voices.MAX_PITCH_RANGE
            assert voices.MIN_WORD_GAP <= shifted.word_gap <= voices.MAX_WORD_GAP


def test_apply_mood_leaves_a_named_voice_alone() -> None:
    """A system or model voice keeps its own timbre; those engines take prosody directly."""
    system = voices.SystemVoice("Microsoft Hoda", "ar-SA", "Female")
    assert moods.apply_mood(system, moods.find_mood("angry")) is system
    model = voices.ModelVoice(name="ar_JO-kareem-medium", engine="piper")
    assert moods.apply_mood(model, moods.find_mood("happy")) is model


def test_support_note_is_honest_per_engine() -> None:
    # A formant engine takes the whole prosody; a cloning engine takes little.
    assert "full" in moods.support_note("espeak-ng")
    assert "limited" in moods.support_note("audio.cpp")
    assert "partial" in moods.support_note("piper")
    assert "none" in moods.support_note("silent")
    # An unknown engine claims nothing.
    assert "unknown" in moods.support_note("some-new-engine")


def test_mood_summaries_lists_every_mood_in_order() -> None:
    rows = moods.mood_summaries()
    assert len(rows) == len(moods.MOODS)
    assert [row[0] for row in rows] == list(moods.mood_keys())
    assert rows[0][1] == "Neutral"


def test_the_opposite_moods_move_the_voice_opposite_ways() -> None:
    """Happy and sad must be distinguishable, not two names for one setting."""
    voice = voices.find_voice("woman-1")
    happy = moods.apply_mood(voice, moods.find_mood("happy"))
    sad = moods.apply_mood(voice, moods.find_mood("sad"))
    assert happy.pitch > sad.pitch
    assert happy.speed > sad.speed
    assert happy.pitch_range > sad.pitch_range
