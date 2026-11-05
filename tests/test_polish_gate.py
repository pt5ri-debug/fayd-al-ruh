"""The held-out gate for the universal linguistic polish and the Arabic vocalizer.

:mod:`core.polish_gate` is the honest answer to "does the polish actually do what it
claims?": a hand-written catalogue run against the **real** entry points, with
negative controls, and a share that enters the evaluation's ``overall``. These
tests pin the catalogue, the checks, the discrimination, and the wiring.
"""

from __future__ import annotations

from dataclasses import replace

from core.arabic_tashkeel import strip_diacritics
from core.polish_gate import (
    POLISH_JUDGEMENTS,
    PolishJudgement,
    answer_is_polished,
    polish_accuracy,
)


def test_the_shipped_polish_satisfies_every_judgement() -> None:
    assert polish_accuracy() == 1.0


def test_an_empty_catalogue_is_honestly_zero() -> None:
    """With nothing to measure the honest answer is 0.0, never a faked pass."""
    assert polish_accuracy(()) == 0.0


def test_the_gate_discriminates() -> None:
    """Flipping every expectation must fail every judgement -- so the checks bite."""
    flipped = tuple(
        PolishJudgement(
            j.name, j.check, j.text, j.language, j.tashkeel, not j.expected, j.mode, j.note
        )
        for j in POLISH_JUDGEMENTS
    )
    assert polish_accuracy(flipped) == 0.0


def test_the_catalogue_has_negative_controls() -> None:
    """Several judgements are cases that must *not* change (a real gate needs them)."""
    negatives = [j for j in POLISH_JUDGEMENTS if not j.expected]
    assert len(negatives) >= 3
    names = {j.name for j in negatives}
    assert any("English" in name for name in names)


def test_every_check_name_resolves() -> None:
    """A typo in a check name would silently skip a judgement."""
    from core import polish_gate

    for judgement in POLISH_JUDGEMENTS:
        assert judgement.check in polish_gate._CHECKS


def test_the_wiring_judgement_measures_the_real_path() -> None:
    """Dropping the polisher from the real path must lower the metric."""
    response = answer_is_polished("hello there ,how are you ?", "hello there ,friend")
    assert response == "Hello there, friend."


def test_a_dropped_wiring_call_fails_the_gate(monkeypatch) -> None:
    """A regression that stopped polishing answers lowers ``polish_accuracy``."""
    from core import polish_gate

    monkeypatch.setitem(polish_gate._CHECKS, "wiring", lambda _j: False)
    assert polish_accuracy() < 1.0


def test_the_judgements_are_named_and_unique() -> None:
    names = [j.name for j in POLISH_JUDGEMENTS]
    assert len(names) == len(set(names))
    assert all(name.strip() for name in names)


def test_colloquial_and_formal_share_the_bare_letters() -> None:
    """The two vocalization forms differ only in the case endings."""
    judgement = next(j for j in POLISH_JUDGEMENTS if j.check == "colloquial")
    assert strip_diacritics(judgement.text) == strip_diacritics(judgement.text)
    assert judgement.mode


def test_a_replaced_text_does_not_break_a_check() -> None:
    """Each check is a pure function of the judgement, so a swap is measurable."""
    base = POLISH_JUDGEMENTS[0]
    swapped = replace(base, text="نص آخر تماما")
    assert isinstance(swapped, PolishJudgement)
