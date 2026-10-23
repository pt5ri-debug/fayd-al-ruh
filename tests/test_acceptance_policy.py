"""The acceptance policy is the single gate for newly learned knowledge.

These tests pin the exact criteria -- the same ones that reviewed the Petroleum
Disciplines branch -- so any future change to acceptance is a visible, deliberate
change to this file rather than a quiet drift in the learner.
"""

from __future__ import annotations

from core.acceptance_policy import (
    MIN_CREDIBILITY,
    REJECT_FORMULA,
    REJECT_INCOMPLETE,
    TITLE_COVERAGE,
    clean_statement,
    is_formula_fragment,
    is_incomplete_statement,
    is_relevant_to_topic,
    is_trusted,
    rejection_reason,
    sanitize_statement,
    source_rejection_reason,
)


def _reason(**overrides):
    base = {
        "url": "https://www.spe.org/x",
        "title": "Permeability",
        "topic": "Permeability",
        "snippet": "Permeability measures how easily fluid flows through reservoir rock.",
        "credibility": 0.9,
    }
    base.update(overrides)
    return rejection_reason(**base)


# --- the happy path -------------------------------------------------------


def test_an_on_topic_trusted_result_is_accepted():
    assert _reason() is None


# --- each gate, in isolation ---------------------------------------------


def test_untrusted_domain_is_refused():
    assert _reason(url="https://randompetroleumblog.com/x") == "domain not trusted"


def test_low_credibility_is_refused():
    reason = _reason(credibility=MIN_CREDIBILITY - 0.01)
    assert reason is not None and reason.startswith("credibility")


def test_empty_snippet_is_refused():
    assert _reason(snippet="   ") == "empty snippet"


def test_formula_fragment_is_refused():
    assert _reason(snippet=r"{\displaystyle \phi} is the porosity.") == REJECT_FORMULA
    assert is_formula_fragment(r"k = \frac{q \mu}{A}")


def test_result_not_about_the_topic_is_refused():
    assert not is_relevant_to_topic("reservoir boundary", "Western Water Catchment")
    assert not is_relevant_to_topic("screen", "Blue screen of death")
    assert not is_relevant_to_topic(
        "decline curve analysis", "Economy of India"
    )


def test_a_shared_word_is_not_enough_when_the_title_is_mostly_about_something_else():
    # "screen" is shared, but the title is mostly about something else.
    assert not is_relevant_to_topic("screen", "Blue screen of death")
    assert not is_relevant_to_topic("reservoir boundary", "Western Water Catchment")


def test_title_coverage_is_symmetric():
    # Topic fully covered by the title, but the title is far larger -> refused.
    assert not is_relevant_to_topic("screen", "Blue screen of death")
    # Genuine match -> accepted.
    assert is_relevant_to_topic("gamma ray log", "Gamma ray logging")


def test_topic_match_ignores_case_and_punctuation():
    assert is_relevant_to_topic("Effective Porosity", "effective porosity")
    assert is_relevant_to_topic("well test", "well test")


# --- sense disambiguation for ambiguous head-nouns ------------------------
# "reservoir" is also an everyday word, so a water-reservoir article must not
# slip into a petroleum store. The check reads the statement itself, so the
# petroleum sense (a different context word present) is still accepted.


def test_a_water_reservoir_is_refused():
    assert not is_relevant_to_topic(
        "reservoir study",
        "Quabbin Reservoir",
        "Quabbin Reservoir is the largest inland body of water in Massachusetts.",
    )
    assert not is_relevant_to_topic(
        "reservoir", "Water reservoir", "The reservoir stores drinking water for the city."
    )


def test_the_petroleum_sense_is_still_accepted():
    # The same ambiguous word, but the statement is unmistakably petroleum.
    assert is_relevant_to_topic(
        "reservoir",
        "Reservoir",
        "A petroleum reservoir is a subsurface accumulation of hydrocarbons.",
    )
    assert is_relevant_to_topic(
        "reservoir",
        "Reservoir pressure",
        "Water drive supports reservoir pressure as the aquifer advances.",
    )


def test_the_ambiguous_word_alone_is_not_evidence_of_the_petroleum_sense():
    # "reservoir" appears in the statement, but no *other* petroleum word does.
    assert not is_relevant_to_topic(
        "reservoir", "Reservoir", "The reservoir is the largest inland body of water."
    )


def test_an_unambiguous_term_is_unaffected_by_the_sense_check():
    assert is_relevant_to_topic("gamma ray log", "Gamma ray logging")
    assert is_relevant_to_topic("Effective Porosity", "effective porosity")


def test_a_water_reservoir_reaches_the_topic_gate_through_rejection_reason():
    reason = rejection_reason(
        url="https://en.wikipedia.org/wiki/Quabbin_Reservoir",
        title="Quabbin Reservoir",
        topic="reservoir study",
        snippet="Quabbin Reservoir is the largest inland body of water in Massachusetts.",
        credibility=0.9,
    )
    assert reason is not None and "reservoir study" in reason


def test_threshold_is_documented_and_midrange():
    assert 0.0 < TITLE_COVERAGE < 1.0


# --- trust and cleaning ---------------------------------------------------


def test_is_trusted_accepts_standards_academic_and_government_hosts():
    assert is_trusted("https://www.spe.org/x")
    assert is_trusted("https://en.wikipedia.org/wiki/Permeability")
    assert is_trusted("https://www.energy.gov/x")
    assert not is_trusted("https://randompetroleumblog.com/x")


def test_clean_statement_strips_tags_urls_and_entities():
    text = 'A &amp; B <a href="#">link</a> https://www.sec.gov/whistleblower about rock.'
    cleaned = clean_statement(text)
    assert "&amp;" not in cleaned
    assert "http" not in cleaned.casefold()
    assert "<" not in cleaned
    assert "A & B" in cleaned


# --- the source gate is topic-independent ---------------------------------


def test_source_gate_ignores_topic_and_passes_a_clean_source():
    assert (
        source_rejection_reason(
            url="https://www.spe.org/x",
            snippet="Permeability measures flow through rock.",
            credibility=0.9,
        )
        is None
    )


def test_source_gate_still_refuses_untrusted_low_credibility_and_formula():
    assert (
        source_rejection_reason(
            url="https://randompetroleumblog.com/x",
            snippet="text",
            credibility=0.9,
        )
        == "domain not trusted"
    )
    assert REJECT_FORMULA == source_rejection_reason(
        url="https://www.spe.org/x",
        snippet=r"\displaystyle \phi",
        credibility=0.9,
    )
    assert source_rejection_reason(
        url="https://www.spe.org/x",
        snippet="text",
        credibility=MIN_CREDIBILITY - 0.01,
    ).startswith("credibility")


def test_incomplete_statement_is_refused():
    for fragment in (
        "into the earth's",
        "drawbacks—mud filtrate soaks into the formation",
        "Effective porosity is a measure of the rock or",
        "Porosity or void fraction is a measure of the void (i.e.",
    ):
        assert is_incomplete_statement(fragment)
    assert not is_incomplete_statement("Permeability measures how easily fluid flows through rock.")


def test_source_gate_refuses_an_incomplete_snippet():
    assert (
        source_rejection_reason(
            url="https://www.spe.org/x",
            snippet="drawbacks—mud filtrate soaks into the formation around the borehole",
            credibility=0.9,
        )
        == REJECT_INCOMPLETE
    )


def test_sanitize_statement_keeps_only_whole_sentences():
    assert sanitize_statement("Rock porosity is high. It controls flow.") == (
        "Rock porosity is high. It controls flow."
    )
    # Leading fragment dropped, the whole sentence after it kept.
    assert sanitize_statement("the earth's. A drilling rig drills wells into the ground.") == (
        "A drilling rig drills wells into the ground."
    )
    # Only a fragment: nothing survives.
    assert sanitize_statement("into the earth's") == ""


def test_source_gate_is_reused_by_the_topic_gate():
    # Same refusal, reached through either entry point.
    kwargs = {
        "url": "https://randompetroleumblog.com/x",
        "snippet": "text",
        "credibility": 0.9,
    }
    assert source_rejection_reason(**kwargs) == rejection_reason(
        title="anything", topic="anything", **kwargs
    )
