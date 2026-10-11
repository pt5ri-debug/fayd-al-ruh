"""Tests for safe, deterministic text normalization (Arabic and English)."""

from core.text_normalize import collapse_whitespace, fold_arabic, normalize_text


def test_empty_text_stays_empty():
    assert normalize_text("") == ""
    assert normalize_text("   \n\t ") == ""


def test_control_characters_are_removed_and_newlines_kept():
    raw = "a\x00b\x07c\nsecond\tline"
    out = normalize_text(raw)
    assert "\x00" not in out and "\x07" not in out
    # Tabs are whitespace and collapse to a space (as the chunker does); the
    # newline is structure and survives.
    assert out == "abc\nsecond line"


def test_line_endings_are_unified():
    assert normalize_text("a\r\nb\rc") == "a\nb\nc"


def test_trailing_spaces_are_trimmed_and_runs_of_newlines_capped():
    raw = "heading   \n\n\n\nbody   "
    assert normalize_text(raw) == "heading\n\nbody"


def test_spaces_are_collapsed_but_words_survive():
    assert normalize_text("a\t\t  b\u00a0c") == "a b c"


def test_normalization_is_idempotent():
    raw = "  Mixed   text\r\n\r\n\r\nwith   gaps  "
    once = normalize_text(raw)
    assert normalize_text(once) == once


def test_english_case_is_preserved():
    # Case folding is deliberately not applied to stored text; only query
    # normalization casefolds, so documents keep their original casing.
    assert normalize_text("Reservoir Pressure") == "Reservoir Pressure"


def test_arabic_diacritics_and_tatweel_removed_only_when_folding():
    raw = "المَعْلُومَات"  # with tashkeel
    assert normalize_text(raw) == raw  # conservative pass keeps them
    assert fold_arabic(raw) == "المعلومات"


def test_arabic_letter_variants_unified_when_folding():
    assert fold_arabic("أحمد") == "احمد"
    assert fold_arabic("مصطفى") == "مصطفي"
    assert fold_arabic("مكتبة") == "مكتبه"


def test_fold_arabic_is_idempotent():
    once = fold_arabic("إِنَّ الْحَمْدَ لِلَّهِ")
    assert fold_arabic(once) == once


def test_normalize_text_with_fold_flag_matches_fold_arabic():
    raw = "أَلْوَان"
    assert normalize_text(raw, fold_arabic_letters=True) == fold_arabic(raw)


def test_mixed_arabic_english_survives():
    out = normalize_text("Depth 2500 m — العمق ٢٥٠٠ متر")
    assert "Depth 2500 m" in out
    assert "العمق" in out


def test_collapse_whitespace_matches_the_chunker_rule():
    assert collapse_whitespace("a\n\n b\t c ") == "a b c"


def test_normalization_is_deterministic_across_runs():
    raw = "عينة  Sample\n\n\n  2500  m"
    assert normalize_text(raw, fold_arabic_letters=True) == normalize_text(
        raw, fold_arabic_letters=True
    )
