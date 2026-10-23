"""Tests for the Arabic speech-text preparation pass.

The module is dependency-free and deterministic, so these tests assert the exact
words an Arabic listener hears. The central claim under test is the one the user
reported: Arabic answers must be *spoken*, not spelled out as symbols.
"""

from __future__ import annotations

import pytest
from core.arabic_speech import number_to_arabic_words, prepare_for_speech

# ------------------------------------------------------------------- numbers


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "صفر"),
        (1, "واحد"),
        (2, "اثنان"),
        (9, "تسعة"),
        (10, "عشرة"),
        (11, "أحد عشر"),
        (19, "تسعة عشر"),
        (20, "عشرون"),
        (21, "واحد وعشرون"),
        (99, "تسعة وتسعون"),
        (100, "مئة"),
        (101, "مئة وواحد"),
        (200, "مئتان"),
        (300, "ثلاثمئة"),
        (999, "تسعمئة وتسعة وتسعون"),
    ],
)
def test_numbers_below_one_thousand(value, expected):
    assert number_to_arabic_words(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # The three Arabic counting forms: singular, dual, plural.
        (1000, "ألف"),
        (2000, "ألفان"),
        (3000, "ثلاثة آلاف"),
        (10000, "عشرة آلاف"),
        (11000, "أحد عشر ألف"),
        (25000, "خمسة وعشرون ألف"),
        (1500, "ألف وخمسمئة"),
        (1_000_000, "مليون"),
        (2_000_000, "مليونان"),
        (3_000_000, "ثلاثة ملايين"),
        (1_500_000, "مليون وخمسمئة ألف"),
    ],
)
def test_thousands_use_the_arabic_counting_forms(value, expected):
    assert number_to_arabic_words(value) == expected


def test_a_negative_number_is_spoken_with_its_sign():
    assert number_to_arabic_words(-5) == "سالب خمسة"


def test_oblique_form_is_the_genitive_a_preposition_requires():
    # ``إلى`` governs the number that follows it, so a range end takes this form.
    assert number_to_arabic_words(20, oblique=True) == "عشرين"
    assert number_to_arabic_words(2000, oblique=True) == "ألفين"
    assert number_to_arabic_words(200, oblique=True) == "مئتين"


# ------------------------------------------------------- the reported failures
#
# A bare fragment such as ``"25%"`` carries no Arabic letters, so on its own it is
# detected as English and left alone (the pass must never rewrite English). In real
# use the whole Arabic answer is handed to the reader -- it has Arabic words around
# the numbers, so the script gate opens. These tests pin ``language="ar"`` to
# exercise the conversion directly, which is exactly what the reader does when a
# voice (or the answer) is Arabic.


def test_a_percentage_keeps_its_unit():
    # The measured bug: ``25%`` was read as ``خمسة وعشرون`` -- the unit vanished.
    result = prepare_for_speech("25%", language="ar")
    assert "بالمئة" in result
    assert "خمسة وعشرون" in result


def test_a_pressure_unit_is_spelled_as_arabic_words():
    # The measured bug: ``1500 psi`` switched the engine to an English voice.
    result = prepare_for_speech("1500 psi", language="ar")
    assert "ألف وخمسمئة" in result
    assert "باوند لكل بوصة مربعة" in result
    assert "psi" not in result


def test_a_depth_unit_is_spelled_as_arabic_words():
    result = prepare_for_speech("2000 ft", language="ar")
    assert "ألفان" in result
    assert "قدم" in result
    assert "ft" not in result


def test_a_decimal_is_spoken_digit_by_digit():
    # ``3.5`` must not be read as ``three point five tenths``.
    result = prepare_for_speech("3.5 mD", language="ar")
    assert "ثلاثة فاصلة خمسة" in result
    assert "ملي دارسي" in result


def test_a_range_is_spoken_as_a_range():
    # The measured bug: ``10-20`` collapsed to gibberish with the dash lost.
    assert prepare_for_speech("10-20", language="ar") == "عشرة إلى عشرين"


def test_a_range_with_spaces_and_en_dashes_is_understood():
    assert prepare_for_speech("10 -- 20", language="ar") == "عشرة إلى عشرين"
    assert prepare_for_speech("10 – 20", language="ar") == "عشرة إلى عشرين"


def test_a_large_range_end_takes_the_genitive_form():
    # ``إلى`` governs the end of the range, so it must read ``ألفين`` not ``ألفان``.
    assert prepare_for_speech("1000-2000", language="ar") == "ألف إلى ألفين"


def test_a_bare_symbol_fragment_is_left_for_english_but_converted_for_arabic():
    # The gate is the script, so the same fragment behaves honestly either way.
    assert prepare_for_speech("25%") == "25%"
    assert prepare_for_speech("25%", language="ar") != "25%"


def test_tatweel_is_removed_so_the_word_is_not_spelled_out():
    # The kashida made the engine spell the word letter by letter.
    assert prepare_for_speech("المسامـــية") == "المسامية"


def test_the_full_reported_sentence_is_spoken_not_spelled():
    text = "المسامية تبلغ 25% والضغط 1500 psi عند عمق 2000 ft، والنفاذية 3.5 mD بين 10-20"
    result = prepare_for_speech(text)
    # Every symbol the engine cannot say has become a word it can.
    for symbol in ("%", "psi", "ft", "mD", "10-20", "3.5"):
        assert symbol not in result, f"{symbol!r} was left for the engine to mangle"
    for word in ("بالمئة", "باوند لكل بوصة مربعة", "قدم", "ملي دارسي", "عشرة إلى عشرين"):
        assert word in result


# ------------------------------------------------------------ Arabic punctuation


def test_ascii_punctuation_becomes_arabic_inside_arabic_text():
    # Clause detection follows the script, so the marks must match it.
    result = prepare_for_speech("البئر، ما الضغط؟ ثم الأنبوب; النتيجة")
    assert "?" not in result
    assert ";" not in result
    assert "؟" in result
    assert "؛" in result


# ------------------------------------------------------------------- safety


def test_english_text_is_never_touched():
    # The pass is gated to Arabic: an English answer must come back byte-identical.
    text = "The porosity is 25% and the pressure is 1500 psi, in the range 10-20."
    assert prepare_for_speech(text) == text


def test_an_english_word_is_not_read_as_a_unit():
    # ``in`` is an English preposition and also the unit for inches; a long English
    # sentence must not be rewritten into Arabic units.
    text = "The tool is in the hole and the mud is heavy."
    assert prepare_for_speech(text) == text


def test_a_latin_identifier_without_an_arabic_name_is_kept():
    # A standard body's name is not a unit, so it is left for the engine to say.
    for text in ("وفق معيار SPE", "حسب مواصفة ISO", "تصنيف API"):
        result = prepare_for_speech(text)
        assert "SPE" in result or "ISO" in result or "API" in result


def test_an_unknown_unit_is_left_alone_rather_than_guessed():
    # The pass never invents a unit: an unknown token is the engine's business.
    result = prepare_for_speech("القياس 42 zorks")
    assert "zorks" in result
    assert "42" not in result  # but the number is still spoken


def test_the_pass_is_idempotent():
    text = "المسامية 25% والضغط 1500 psi بين 10-20"
    once = prepare_for_speech(text)
    assert prepare_for_speech(once) == once


def test_empty_and_whitespace_input_are_returned_unchanged():
    assert prepare_for_speech("") == ""
    assert prepare_for_speech("   ") == "   "


def test_an_explicit_language_pins_the_pass():
    # An explicit ``en`` must win over the Arabic script in the text.
    text = "الضغط 1500"
    assert prepare_for_speech(text, language="en") == text


def test_thousands_separators_are_understood():
    assert prepare_for_speech("1,500", language="ar") == "ألف وخمسمئة"


def test_arabic_indic_digits_are_spoken():
    # Arabic-Indic digits are the same number in another script.
    assert prepare_for_speech("٢٥", language="ar") == "خمسة وعشرون"


# ------------------------------------------------------------------- wiring
#
# The pass is only worth anything if the engine actually receives its output. Each
# synthesizer must prepare the text before it renders, so an Arabic answer is
# spoken as words no matter which engine is installed.


def test_every_synthesizer_prepares_the_text_before_rendering():
    import inspect

    import core.piper_speech as piper
    import core.qwen_speech as qwen
    from core import speech

    assert "prepare_for_speech" in inspect.getsource(speech.CommandSynthesizer.synthesize)
    assert "prepare_for_speech" in inspect.getsource(
        speech._WindowsPowerShellSynthesizer.synthesize
    )
    assert "prepare_for_speech" in inspect.getsource(piper.PiperSynthesizer.synthesize)
    assert "prepare_for_speech" in inspect.getsource(qwen.QwenTtsSynthesizer.synthesize)


def test_a_command_engine_receives_the_prepared_arabic_text(tmp_path):
    # Drive the real synthesizer with a fake executable that records its argv, so
    # the claim is proven rather than assumed: the engine is handed Arabic words.
    import core.speech as speech
    from core.arabic_speech import prepare_for_speech

    captured: list[list[str]] = []

    def builder(executable: str, text: str, target):
        captured.append([executable, text, str(target)])
        # Write a valid tiny WAV so the synthesizer reads it back successfully.
        import wave

        with wave.open(str(target), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(speech.SAMPLE_RATE)
            handle.writeframes(b"\x00\x00" * 160)
        return [executable, "--write", str(target)]

    synth = speech.CommandSynthesizer("fake", "/bin/true", builder, language="ar")
    answer = "المسامية تبلغ 25% والضغط 1500 psi"
    synth.synthesize(answer)

    assert captured, "the engine was never invoked"
    handed_to_engine = captured[0][1]
    assert handed_to_engine == prepare_for_speech(answer, language="ar")
    assert "psi" not in handed_to_engine
    assert "باوند لكل بوصة مربعة" in handed_to_engine


def test_an_english_answer_reaches_the_engine_unchanged():
    # The wiring must not rewrite English either.
    import core.speech as speech

    captured: list[str] = []

    def builder(executable: str, text: str, target):
        captured.append(text)
        import wave

        with wave.open(str(target), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(speech.SAMPLE_RATE)
            handle.writeframes(b"\x00\x00" * 160)
        return [executable, "--write", str(target)]

    synth = speech.CommandSynthesizer("fake", "/bin/true", builder, language="en")
    answer = "The porosity is 25% and the pressure is 1500 psi."
    synth.synthesize(answer)
    assert captured == [answer]
