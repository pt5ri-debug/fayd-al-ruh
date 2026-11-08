"""The reading-direction preference: LTR/RTL, and what it changes here."""

from __future__ import annotations

import pytest
from gui import layout_direction as ld


def test_normalize_direction_accepts_both_vocabularies():
    assert ld.normalize_direction("rtl") == "rtl"
    assert ld.normalize_direction("RTL") == "rtl"
    assert ld.normalize_direction("ar") == "rtl"
    assert ld.normalize_direction("arabic") == "rtl"
    assert ld.normalize_direction("right") == "rtl"
    assert ld.normalize_direction("ltr") == "ltr"
    assert ld.normalize_direction("english") == "ltr"
    assert ld.normalize_direction("") == "ltr"
    assert ld.normalize_direction(None) == "ltr"
    assert ld.normalize_direction("nonsense") == "ltr"


def test_side_and_alignment_follow_the_direction():
    assert ld.side_for("ltr") == "left"
    assert ld.side_for("rtl") == "right"
    assert ld.default_alignment("ltr") == "left"
    assert ld.default_alignment("rtl") == "right"


def test_detect_direction_reads_the_dominant_script():
    assert ld.detect_direction("hello world") == "ltr"
    assert ld.detect_direction("مرحبا بالعالم") == "rtl"
    # A mostly-Arabic sentence with an English term is still RTL.
    assert ld.detect_direction("المسامية porosity تزيد النفاذية") == "rtl"
    # A mostly-English sentence with an Arabic term is still LTR.
    assert ld.detect_direction("porosity and permeability النفاذية value") == "ltr"
    # No letters at all: the fallback, never a guess.
    assert ld.detect_direction("123 + 456", fallback="rtl") == "rtl"
    assert ld.detect_direction("!!!", fallback="ltr") == "ltr"


def test_describe_names_the_side_and_the_limit():
    text = ld.describe("rtl")
    assert "right" in text
    assert "dialogs keep their own layout" in text
    assert "left" in ld.describe("ltr")


def test_load_and_save_round_trip(tmp_path):
    assert ld.load_direction(tmp_path) == ld.DEFAULT_DIRECTION
    ld.save_direction("rtl", tmp_path)
    assert ld.load_direction(tmp_path) == "rtl"
    ld.save_direction("ltr", tmp_path)
    assert ld.load_direction(tmp_path) == "ltr"


def test_a_broken_store_reads_as_the_default(tmp_path):
    store = tmp_path / "gui"
    store.mkdir(parents=True, exist_ok=True)
    (store / "layout_direction.json").write_text("{ not json", encoding="utf-8")
    assert ld.load_direction(tmp_path) == ld.DEFAULT_DIRECTION


def test_unknown_saved_value_falls_back_to_ltr(tmp_path):
    from storage.json_store import JsonStore

    JsonStore(tmp_path).save("gui", "layout_direction", {"direction": "sideways"})
    assert ld.load_direction(tmp_path) == "ltr"


@pytest.mark.parametrize("value", list(ld.DIRECTIONS))
def test_every_direction_has_a_label(value):
    assert ld.DIRECTION_LABELS[value]
