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
    compose_model_answer,
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


def test_compose_model_answer_keeps_a_whole_sentence_and_punctuates_it():
    assert (
        compose_model_answer("the well is completed with casing. Then it flows")
        == "The well is completed with casing."
    )


def test_compose_model_answer_drops_the_trailing_incomplete_sentence():
    assert (
        compose_model_answer("Monitor and pigging the well to kicks. The procedure for Prod")
        == "Monitor and pigging the well to kicks."
    )


def test_compose_model_answer_drops_a_leading_fragment_before_a_sentence():
    assert (
        compose_model_answer("the saturation forecast the simulation. Porosity measures voids.")
        == "Porosity measures voids."
    )


def test_compose_model_answer_collapses_repeated_punctuation_and_words():
    assert compose_model_answer("Mud wh))))) The well flows.") == "Mud wh. The well flows."
    assert compose_model_answer("Mud wh)))))") == "Mud wh."
    assert compose_model_answer("the the well flows.") == "The well flows."


def test_compose_model_answer_trims_a_stuttered_separator():
    # The byte model stutters a separator (a comma, a colon). Turning the run into
    # a full stop would corrupt the model's own number or list, so the surplus
    # copies are dropped and the model's own mark is kept.
    assert compose_model_answer("the value is 2,,3 times the gradient.") == (
        "The value is 2,3 times the gradient."
    )
    assert compose_model_answer("the ratio is 3:1::1 in the mix.") == (
        "The ratio is 3:1:1 in the mix."
    )
    assert compose_model_answer("The well flows,, then stops.") == (
        "The well flows, then stops."
    )


def test_compose_model_answer_closes_a_separator_gap_without_joining_a_slash():
    # A stray space before a separator ("saturation, , are") is presentation; the
    # gap closes. A space *around* a slash is a formula ("q / A"), not a gap, so
    # it is left exactly as the model wrote it.
    assert compose_model_answer("Porosity, permeability, and saturation, , are properties.") == (
        "Porosity, permeability, and saturation, are properties."
    )
    assert compose_model_answer("The formula is q / A.") == "The formula is q / A."
    assert compose_model_answer("The value is 10 - 20 m.") == "The value is 10 - 20 m."


def test_compose_model_answer_drops_a_leading_stray_separator():
    # The byte model can start a continuation with a leftover separator (", the
    # well is cased."); it is decoration, not part of the sentence, so it is
    # dropped -- the same edge tidy the trailing decoration already gets. The
    # model's own words and any decimal are untouched.
    assert compose_model_answer(", the well is cased.") == "The well is cased."
    assert compose_model_answer("; porosity controls the flow.") == (
        "Porosity controls the flow."
    )
    assert compose_model_answer(": the mud weight is 1.2 sg.") == (
        "The mud weight is 1.2 sg."
    )
    # A leading dot or hyphen before a digit is the model's number, not a mark.
    assert compose_model_answer(".5 times the gradient.") == ".5 times the gradient."


def test_compose_model_answer_drops_a_stray_separator_before_a_full_stop():
    # A separator left right before a boundary ("flows , .") is a stutter at the
    # edge, not the model's own mark; it is dropped so the boundary does not read
    # as "flows,.". A separator the model really meant stays where it is a real
    # list ("2,,3", "saturation, , are").
    assert compose_model_answer("The well flows , . Then it stops.") == (
        "The well flows. Then it stops."
    )
    assert compose_model_answer("The well flows ; . Then it stops.") == (
        "The well flows. Then it stops."
    )
    assert compose_model_answer("The well flows : . Then it flows.") == (
        "The well flows. Then it flows."
    )


def test_compose_model_answer_trims_a_trailing_stray_separator():
    # A continuation that ends on a bare separator ("The well is cased ,") is the
    # trailing twin; the composer ends the sentence with the model's period, not
    # with "cased,.".
    assert compose_model_answer("The well is cased ,") == "The well is cased."
    assert compose_model_answer("The well is cased ;") == "The well is cased."
    assert compose_model_answer("The well is cased :") == "The well is cased."
    assert compose_model_answer("The well is cased , .") == "The well is cased."


def test_compose_model_answer_drops_a_stray_separator_after_a_boundary():
    # A separator the model leaves right after a boundary ("flows. , then") would
    # otherwise open the next sentence as ", then"; it is dropped, the boundary
    # keeps its space, and a comma the model really meant (a list) stays.
    assert compose_model_answer("The well flows. , then it stops.") == (
        "The well flows. Then it stops."
    )
    assert compose_model_answer("The well flows. ; then it stops.") == (
        "The well flows. Then it stops."
    )
    assert compose_model_answer("The well flows , and then it stops.") == (
        "The well flows, and then it stops."
    )


def test_compose_model_answer_collapses_a_run_of_the_same_full_stop():
    # Closing a gap can leave two full stops touching ("flows. ."); the model's
    # own mark is kept once, so a sentence does not end "..".
    assert compose_model_answer("The well flows. . Then it stops.") == (
        "The well flows. Then it stops."
    )
    assert compose_model_answer("The well flows! ! Then it stops.") == (
        "The well flows! Then it stops."
    )
    # Different marks are the model's own punctuation and are left alone.
    assert compose_model_answer("Is the well flowing?! The pump is running.") == (
        "Is the well flowing?! The pump is running."
    )


def test_compose_model_answer_collapses_a_repeated_phrase():
    # The byte model loops on a whole phrase, not only a single word; the repeat
    # is the model stalling, so it is shown once.
    assert (
        compose_model_answer("the same the same the rock is porous.")
        == "The same the rock is porous."
    )
    assert (
        compose_model_answer("The production forecast the production forecast for a field.")
        == "The production forecast for a field."
    )
    assert compose_model_answer("step 1 of Fluid fluid fluid flow in porous media: Force.") == (
        "Step 1 of Fluid flow in porous media: Force."
    )


def test_compose_model_answer_keeps_a_word_reused_in_a_real_sentence():
    # A word that recurs across a sentence (not back-to-back) is not a repeat.
    text = compose_model_answer("Water fills the pores of the rock and the water moves.")
    assert text == "Water fills the pores of the rock and the water moves."


def test_compose_model_answer_does_not_corrupt_a_hyphen_range_or_name():
    # A run of hyphens *between* word characters is a range ("10--20") or a
    # compound name ("5CT--spec"), not repeated decoration, so collapsing it to a
    # period would corrupt the model's own number.
    assert compose_model_answer("The value is 10--20 m.") == "The value is 10--20 m."
    assert compose_model_answer("API 5CT--spec is used.") == "API 5CT--spec is used."
    assert compose_model_answer("Range 5--10 ft.") == "Range 5--10 ft."


def test_compose_model_answer_falls_back_to_the_tidied_text():
    # No whole sentence survives, so the mouth still speaks the model's words.
    assert compose_model_answer("partway through a thoug") == "Partway through a thoug."
    assert compose_model_answer("") == ""


def test_compose_model_answer_drops_output_with_no_words():
    """A stopping run tidies to bare punctuation, which is not an answer."""
    assert compose_model_answer(")))))") == ""
    assert compose_model_answer("...") == ""
    assert compose_model_answer("   ") == ""


def test_compose_model_answer_collapses_a_long_repeat_run():
    # The byte model stalls on a phrase for several copies ("the casing the
    # casing the casing the casing"); a single regex pass only removes every
    # other copy, so the run must collapse all the way to one. The words the
    # model actually wrote around the loop are untouched.
    assert (
        compose_model_answer("schedule the casing the casing the casing the casing the lamin")
        == "Schedule the casing the lamin."
    )
    assert compose_model_answer("the well the well the well the well the well the well") == (
        "The well."
    )
    assert (
        compose_model_answer("Monitor the sample work and the stage the stage the stage the complex.")
        == "Monitor the sample work and the stage the complex."
    )


def test_compose_model_answer_collapses_a_long_repeat_run_without_touching_real_text():
    # A word reused across a real sentence (not back-to-back) still survives, and
    # a partial repeat that is not a whole loop keeps the words that are not the
    # repeated unit.
    assert (
        compose_model_answer("the safety case the case the case the case the complete.")
        == "The safety case the complete."
    )
    assert (
        compose_model_answer("Water fills the pores of the rock and the water moves.")
        == "Water fills the pores of the rock and the water moves."
    )


def test_compose_model_answer_keeps_a_word_that_survives_tidying():
    # A single surviving word is still the model's own answer.
    assert compose_model_answer("porosity") == "Porosity."


def test_compose_model_answer_collapses_a_stall_joined_by_a_conjunction():
    # The byte model also stalls on a phrase with a conjunction between the
    # copies ("the stage complex and stage complex"); the back-to-back matcher
    # never sees it, so the repeated phrase and the conjunction collapse to one.
    assert (
        compose_model_answer("Mud the screen and the stage complex and stage complex.")
        == "Mud the screen and the stage complex."
    )
    assert (
        compose_model_answer("the stress are the static and the static travel to the wellbore.")
        == "The stress are the static travel to the wellbore."
    )
    # A whole run joined by the conjunction collapses all the way to one copy.
    assert (
        compose_model_answer("the stage complex and stage complex and stage complex.")
        == "The stage complex."
    )


def test_compose_model_answer_keeps_a_real_pair_of_phrases():
    # Two different phrases joined by "and" are a real list, not a stall; they
    # share no words, so nothing is collapsed.
    assert compose_model_answer("porosity and permeability") == "Porosity and permeability."
    assert (
        compose_model_answer("The rock and the fluid and the pressure.")
        == "The rock and the fluid and the pressure."
    )


def test_compose_model_answer_preserves_decimals_and_the_models_words():
    text = compose_model_answer("the gradient is 0.052 psi/ft and the formula is q / A.")
    assert text == "The gradient is 0.052 psi/ft and the formula is q / A."


def test_compose_model_answer_keeps_every_whole_sentence_in_order():
    text = compose_model_answer("The well is cased. Then it is cemented. Finally it flows.")
    assert text == "The well is cased. Then it is cemented. Finally it flows."


def test_compose_model_answer_drops_a_dangling_final_clause():
    # "... the completion" has no verb, so the finished sentence is preferred.
    assert (
        compose_model_answer("Measure porosity and permeability. Then run the completion")
        == "Measure porosity and permeability."
    )
    # A trailing conjunction/article marks the clause as unfinished too.
    assert (
        compose_model_answer("The mud is mixed. Then pump the slurry and")
        == "The mud is mixed."
    )


def test_compose_model_answer_keeps_a_dangling_sentence_when_it_is_all_there_is():
    # Nothing cleaner survives, so the model's own (if clumsy) sentence is kept
    # rather than dropping to silence.
    assert compose_model_answer("Only pump the pad and") == "Only pump the pad and."


def test_compose_model_answer_drops_a_repeated_sentence():
    text = compose_model_answer("The well flows. The well flows. It is stable.")
    assert text == "The well flows. It is stable."


def test_compose_model_answer_drops_a_repeat_joined_by_a_conjunction():
    # "And the pump." repeats "The pump."; showing it twice never helps, so the
    # conjunction-led copy collapses into the first.
    assert compose_model_answer("The pump. The pump. And the pump.") == "The pump."
    assert compose_model_answer("It flows. Then it flows.") == "It flows."


def test_compose_model_answer_does_not_let_a_list_marker_eat_a_decimal():
    # The byte model emits numbered procedure sentences that can begin with a
    # decimal ("1.5 times the gradient"); stripping a leading "1." as a list
    # marker would corrupt the number, so the digit must survive. The lower-case
    # second sentence is capitalised (it began after a boundary), but the numbers
    # are the model's own.
    text = compose_model_answer("1.5 times the gradient. the gradient is 0.052 psi/ft.")
    assert text == "1.5 times the gradient. The gradient is 0.052 psi/ft."


def test_compose_model_answer_capitalises_a_mid_answer_lower_case_sentence():
    # The byte model runs on past a boundary without capitalising; each sentence
    # is capitalised so the answer reads as sentences, with no words changed.
    assert compose_model_answer(
        "The mud is mixed. and the slurry is pumped. the casing is run."
    ) == "The mud is mixed. And the slurry is pumped. The casing is run."
    assert compose_model_answer(
        "Porosity measures voids. permeability measures flow."
    ) == "Porosity measures voids. Permeability measures flow."


def test_compose_model_answer_does_not_capitalise_inside_a_word_or_number():
    # Only a genuine sentence opening is capitalised; a decimal, a unit, and a
    # formula are left exactly as the model wrote them.
    assert compose_model_answer(
        "The gradient is 0.052 psi/ft and the formula is q / A."
    ) == "The gradient is 0.052 psi/ft and the formula is q / A."


def test_compose_model_answer_closes_a_gap_before_punctuation():
    # The byte model leaves a space before a full stop and glues the next
    # sentence on; both are tidied to a single boundary.
    assert compose_model_answer("The well flows .Then it stops.") == (
        "The well flows. Then it stops."
    )
    assert compose_model_answer("The well flows.Then it stops.") == (
        "The well flows. Then it stops."
    )
    # A hyphenated range keeps its spacing, so the fix does not touch numbers.
    assert compose_model_answer("The value is 10 - 20 m.") == "The value is 10 - 20 m."


def test_compose_model_answer_still_drops_a_real_list_marker():
    # A numbered marker followed by a space is decoration, not a number.
    assert compose_model_answer("1. Run the casing.") == "Run the casing."
    assert compose_model_answer("2) Cement the casing.") == "Cement the casing."


def test_compose_model_answer_centres_on_the_question():
    raw = "Compute the surface facilities. The mud weight controls the wellbore pressure."
    # The sentence naming the question's words ("mud"/"weight") opens the answer.
    assert compose_model_answer(raw, question="How does mud weight work?") == (
        "The mud weight controls the wellbore pressure. Compute the surface facilities."
    )


def test_compose_model_answer_demotes_a_bare_section_label():
    raw = "Step 2 of Well integrity management: schedule the work. Cement bonds the casing."
    text = compose_model_answer(raw, question="What is a casing?")
    assert text.startswith("Cement bonds the casing.")
    assert "Step 2 of" not in text


def test_compose_model_answer_without_a_question_still_composes():
    assert compose_model_answer("The well is cased. Then it flows") == "The well is cased."


def test_compose_model_answer_reads_an_ontology_id_as_a_phrase():
    # The training prose is built from concept labels, so the model echoes an
    # identifier verbatim ("pore_pressure"); the underscore is a data-format
    # separator, not a word the model chose, so it reads as the phrase it names.
    assert (
        compose_model_answer("Pore_pressure is a concept in Reservoir simulation.")
        == "Pore pressure is a concept in Reservoir simulation."
    )
    assert (
        compose_model_answer("Voidage_replacement is a concept from Well Completions.")
        == "Voidage replacement is a concept from Well Completions."
    )
    assert compose_model_answer("co2_capture raises recovery.") == "Co2 capture raises recovery."


def test_compose_model_answer_collapses_a_looping_ontology_id():
    # An id the model loops on reads as the looping phrase it names, and collapses
    # to one copy like any other repeat.
    assert compose_model_answer("pore_pressure pore_pressure is high.") == (
        "Pore pressure is high."
    )


def test_compose_model_answer_does_not_touch_a_decimal_range_or_formula():
    # The id read-out must not disturb the model's own numbers, ranges, compound
    # names, or formulas: none of them is a whole underscore-joined token.
    assert compose_model_answer("The value is 10--20 m.") == "The value is 10--20 m."
    assert compose_model_answer("API 5CT--spec is used.") == "API 5CT--spec is used."
    assert compose_model_answer("The gradient is 0.052 psi/ft.") == "The gradient is 0.052 psi/ft."
    assert compose_model_answer("The formula is q / A.") == "The formula is q / A."
    assert compose_model_answer("The well is 10 - 20 m deep.") == "The well is 10 - 20 m deep."


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
    # A bare verb and the pronoun the question omitted are dropped, so the phrase
    # is the noun phrase the user named rather than a broken clause.
    assert topic_phrase("how does porosity relate to permeability") == "porosity permeability"
    assert topic_phrase("how do I measure porosity on a core sample?") == "porosity core sample"
    # A word that could be a noun is kept, so nothing the user named disappears.
    assert topic_phrase("how does water flooding affect oil recovery") == "water flooding oil recovery"


def test_topic_phrase_leaves_an_arabic_question_in_arabic():
    assert topic_phrase("ما هي المسامية؟") == "المسامية"
    assert topic_phrase("كيف أقيس النفاذية؟") == "أقيس النفاذية"


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
