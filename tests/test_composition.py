"""Tests for the deterministic prose composer (:mod:`core.composition`).

The composer is the dry, auditable half of the project's language: it renders
evidence the system already holds into clean sentences with no model and no
external service, so its behaviour is pinned here directly.
"""

from __future__ import annotations

import pytest

from core.composition import (
    Source,
    causal_chain_sentence,
    causal_effect_sentence,
    clean,
    counterfactual_sentence,
    decision_sentences,
    ensure_terminal,
    evidence_summary_sentence,
    first_sentences,
    grounded_answer,
    reasoning_step_sentence,
    sentences,
    sources_block,
    topic_phrase,
    web_answer,
)


def test_source_requires_some_text():
    with pytest.raises(ValueError):
        Source(title=" ", text=" ")


def test_clean_collapses_whitespace_and_drops_list_markers():
    assert clean("  - Hello\n   world  ") == "Hello world"
    assert clean("1) First point") == "First point"


def test_sentences_split_without_breaking_decimals():
    parts = sentences("The gradient is 0.052 psi/ft. It rises with depth.")
    assert parts == ["The gradient is 0.052 psi/ft.", "It rises with depth."]


def test_first_sentences_limits_output():
    assert first_sentences("One. Two. Three.", 2) == "One. Two."


def test_ensure_terminal_capitalises_and_punctuates():
    assert ensure_terminal("porosity is high") == "Porosity is high."
    assert ensure_terminal("already fine.") == "Already fine."


def test_grounded_answer_reads_as_prose_without_inline_citations():
    text = grounded_answer(
        "what is porosity",
        [
            Source("Porosity", "Porosity measures void space in the rock.", "[notes:1] Porosity"),
            Source("Permeability", "Permeability controls fluid flow.", "[notes:2] Permeability"),
        ],
    )
    assert text.startswith("Here is what the available sources say about porosity.")
    assert "Porosity measures void space in the rock." in text
    assert "In addition," in text
    # Attribution is data, not prose: no citation leaks into the answer body.
    assert "Sources:" not in text
    assert "[notes:1]" not in text


def test_topic_phrase_keeps_word_order_and_drops_question_words():
    assert topic_phrase("what is the meaning of drilling oil wells?") == "drilling oil wells"
    assert topic_phrase("how does porosity relate to permeability") == "porosity relate permeability"


def test_grounded_answer_skips_bare_fragment_sources():
    text = grounded_answer(
        "what is a wellbore",
        [
            Source("Wellbore", "Wellbore.", "[a] Wellbore"),
            Source("Drilling", "A drilling rig drills wells into the earth for many purposes.", "[b] Drilling"),
        ],
    )
    assert "Wellbore." not in text
    assert "A drilling rig drills wells" in text


def test_sources_block_lists_each_citation_once_in_order():
    block = sources_block(["[a] One", "[b] Two", "[a] One"])
    assert block == "Sources:\n  - [a] One\n  - [b] Two"
    assert sources_block([]) == ""


def test_grounded_answer_without_evidence_says_so_plainly():
    text = grounded_answer("what is xyzzy", [])
    assert "not enough evidence" in text.lower()
    assert "xyzzy" in text


def test_web_answer_body_has_no_inline_attribution():
    text = web_answer(
        "what is the klinkenberg correction",
        [Source("Klinkenberg correction", "It accounts for gas slippage.", "en.wikipedia.org")],
        internal=("[notes:1] Permeability",),
    )
    assert "gas slippage" in text
    assert "internal corpus" in text.lower()
    # The source is carried by the citation list, not interpolated into a sentence.
    assert "en.wikipedia.org" not in text
    assert "(source:" not in text


def test_web_answer_without_sources_says_so_plainly():
    assert "no web source" in web_answer("what is xyzzy", []).lower()


def test_causal_chain_sentence_renders_path_and_confidence():
    text = causal_chain_sentence("rain", "wet", ["soil", "puddle"], 0.80, ["saturation"])
    assert "rain" in text and "wet" in text
    assert "rain \u2192 soil \u2192 puddle \u2192 wet" in text
    assert "0.80" in text and "saturation" in text


def test_causal_effect_sentence_shows_signed_size():
    assert "+0.60" in causal_effect_sentence("rain", "wet", 0.6)


def test_counterfactual_sentence_contrasts_confidences():
    text = counterfactual_sentence("rain", "wet", 0.9, 0.2)
    assert "0.90" in text and "0.20" in text
    assert "absence" in text


def test_reasoning_step_keeps_premise_arrow():
    text = reasoning_step_sentence("deductive", ("rain",), "wet", 0.9)
    assert "rain \u2192 wet" in text
    assert "By deductive reasoning" in text


def test_decision_sentences_show_selection_and_candidates():
    lines = decision_sentences("observe", "best", [("observe", 0.8, 0.1, 0.75)])
    assert lines[0].startswith("Selected: observe")
    assert "utility 0.80" in lines[1]


def test_decision_sentences_without_selection_state_that():
    lines = decision_sentences(None, "nothing scored", [])
    assert "nothing scored" in lines[0].lower()


def test_evidence_summary_sentence_counts_both_sides():
    text = evidence_summary_sentence("wet ground", 1, 2)
    assert "1 supporting" in text and "2 contradicting" in text
