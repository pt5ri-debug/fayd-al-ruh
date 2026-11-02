"""The cognition catalogue: one explained entry per control, with a held-out gate.

The catalogue is the single source of truth for what every cognition control does,
so this suite pins the *catalogue* (the text, the Arabic twin, the governed flag,
the CLI command, the section) and the *gate* that keeps the window and the
catalogue from drifting apart. Everything runs headless: no display, no model, no
training run.
"""

from __future__ import annotations

import pytest
from core.cognition_catalog import (
    CAPABILITIES,
    CAPABILITY_INDEX,
    CATALOG_JUDGEMENTS,
    CATEGORIES,
    CATEGORY_SUMMARIES,
    CATEGORY_SUMMARIES_AR,
    CATEGORY_TITLES,
    WINDOW_BUTTONS,
    capabilities_for,
    capability,
    capability_groups,
    catalog_accuracy,
    catalog_lines,
    describe_catalog,
    governed_keys,
    validate_capability,
)


def test_every_capability_is_indexed() -> None:
    assert len(CAPABILITIES) == len(CAPABILITY_INDEX)
    for cap in CAPABILITIES:
        assert CAPABILITY_INDEX[cap.key] is cap


def test_every_capability_is_explained_in_both_languages() -> None:
    for cap in CAPABILITIES:
        assert cap.title, cap.key
        assert cap.summary, cap.key
        assert cap.outcome, cap.key
        assert cap.arabic_title, cap.key
        assert cap.arabic_summary, cap.key
        assert validate_capability(cap) == (), (cap.key, validate_capability(cap))


def test_every_capability_sits_in_a_declared_section() -> None:
    for cap in CAPABILITIES:
        assert cap.category in CATEGORIES, cap.key


def test_every_section_is_named_and_described_in_both_languages() -> None:
    for category in CATEGORIES:
        assert CATEGORY_TITLES.get(category), category
        assert CATEGORY_SUMMARIES.get(category), category
        assert CATEGORY_SUMMARIES_AR.get(category), category


def test_a_governed_action_names_its_confirmation() -> None:
    for cap in CAPABILITIES:
        if cap.governed:
            assert "--approve" in cap.cli, cap.key
        else:
            assert "--approve" not in cap.cli, cap.key


def test_governed_keys_are_the_governed_actions() -> None:
    assert governed_keys() == frozenset(cap.key for cap in CAPABILITIES if cap.governed)


def test_capability_lookup_is_honest_about_an_unknown_key() -> None:
    assert capability("think") is not None
    assert capability("no-such-control") is None


def test_capability_groups_keep_the_window_order() -> None:
    groups = capability_groups()
    assert [category for category, _ in groups] == list(CATEGORIES)
    for category, caps in groups:
        assert caps == capabilities_for(category)


def test_catalog_lines_are_readable_in_both_languages() -> None:
    english = catalog_lines("en")
    arabic = catalog_lines("ar")
    assert english and arabic
    assert english != arabic
    joined = "\n".join(english)
    for cap in CAPABILITIES:
        assert cap.title in joined


def test_describe_catalog_reports_every_judgement() -> None:
    text = describe_catalog()
    for judgement in CATALOG_JUDGEMENTS:
        assert judgement.description in text


# -- the held-out gate -------------------------------------------------------


def test_the_gate_passes_on_the_shipped_catalogue() -> None:
    assert catalog_accuracy() == 1.0


def test_the_gate_is_honest_with_no_judgements() -> None:
    assert catalog_accuracy(()) == 0.0


def test_the_gate_discriminates_a_broken_judgement() -> None:
    from core.cognition_catalog import CatalogJudgement

    broken = (CatalogJudgement("no_such_check", "a check that does not exist"),)
    # An unknown check cannot pass, so the gate lowers rather than silently 1.0.
    assert catalog_accuracy(broken) == 0.0


def test_every_window_button_has_an_entry_and_the_reverse() -> None:
    assert WINDOW_BUTTONS <= set(CAPABILITY_INDEX)
    assert set(CAPABILITY_INDEX) == WINDOW_BUTTONS


def test_validate_capability_reports_a_missing_summary() -> None:
    from dataclasses import replace

    cap = CAPABILITY_INDEX["think"]
    broken = replace(cap, summary="")
    assert validate_capability(broken)


def test_the_depth_section_explains_its_new_controls() -> None:
    keys = {cap.key for cap in capabilities_for("depth")}
    assert {"depth", "coverage", "harvest", "bridges", "arabic"} <= keys
    harvest = CAPABILITY_INDEX["harvest"]
    assert harvest.governed
    assert "--approve" in harvest.cli


@pytest.mark.parametrize("key", ["think", "study", "depth", "harvest", "rollback", "catalog"])
def test_a_named_capability_describes_itself(key: str) -> None:
    cap = CAPABILITY_INDEX[key]
    text = cap.describe()
    assert cap.title in text
    assert cap.summary in text
    assert cap.cli in text
    arabic = cap.describe(language="ar")
    assert cap.arabic_title in arabic
