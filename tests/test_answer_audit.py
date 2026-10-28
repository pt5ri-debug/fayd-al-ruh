"""Tests for the connected-answer-path audit (Phase 48, Gap 9).

The audit is only worth anything if it *discriminates*: it must pass the real
assistant, and it must fail when a layer stops reaching the answer object or when
a negative control invents a layer it should not. These tests pin both, plus the
honesty rule that with no path to exercise the metric is ``0.0`` rather than a
faked pass.
"""

from __future__ import annotations

import pytest
from core.answer_audit import (
    ALL_LAYERS,
    AUDIT_EXPECTATIONS,
    AUDIT_EXPECTATIONS_AR,
    LAYER_BELIEF,
    LAYER_LEAD,
    LAYER_REASONING,
    REASONING_LAYERS,
    AuditExpectation,
    answer_layers,
    answer_path_accuracy,
    audit_answer,
)
from core.assistant import build_default_assistant


@pytest.fixture(scope="module")
def assistant():
    return build_default_assistant()


def test_catalogue_only_names_known_layers():
    for expectation in AUDIT_EXPECTATIONS:
        assert set(expectation.required) <= ALL_LAYERS
        assert set(expectation.forbidden) <= ALL_LAYERS


def test_catalogue_has_positive_expectations_and_negative_controls():
    positives = [e for e in AUDIT_EXPECTATIONS if e.required]
    negatives = [e for e in AUDIT_EXPECTATIONS if e.forbidden]
    assert positives, "the audit must check layers do arrive"
    assert negatives, "the audit must include negative controls"
    # Every negative control forbids *all* reasoning layers, so a definitional
    # question cannot quietly gain one.
    for control in negatives:
        assert set(control.forbidden) == REASONING_LAYERS


def test_expectation_rejects_unknown_layer():
    with pytest.raises(ValueError):
        AuditExpectation("bad", "what is porosity?", required=("not-a-layer",))


def test_expectation_rejects_a_layer_both_required_and_forbidden():
    with pytest.raises(ValueError):
        AuditExpectation(
            "contradiction",
            "what is porosity?",
            required=("plan",),
            forbidden=("plan",),
        )


def test_expectation_requires_a_name_and_question():
    with pytest.raises(ValueError):
        AuditExpectation("", "what is porosity?")
    with pytest.raises(ValueError):
        AuditExpectation("named", "   ")


def test_answer_layers_reads_the_object_not_the_bridge(assistant):
    answer = assistant.ask("how do I measure porosity and permeability on a core sample?")
    layers = answer_layers(answer)
    assert "plan" in layers
    assert "lead" in layers
    assert "reasoning" in layers


def test_definitional_answer_carries_no_reasoning_layer(assistant):
    answer = assistant.ask("what is permeability?")
    assert not (answer_layers(answer) & REASONING_LAYERS)


def test_real_path_passes_the_whole_catalogue(assistant):
    assert answer_path_accuracy(answer_for=assistant.ask) == 1.0


def test_every_expectation_holds_on_the_real_path(assistant):
    for expectation in AUDIT_EXPECTATIONS:
        assert audit_answer(expectation, assistant.ask), expectation.name


def test_metric_is_zero_without_a_path_to_exercise():
    # The honest answer when the connected path is not measured: not a faked pass.
    assert answer_path_accuracy(answer_for=None) == 0.0


def test_metric_is_zero_for_an_empty_catalogue(assistant):
    assert answer_path_accuracy((), answer_for=assistant.ask) == 0.0


def test_metric_fails_when_a_layer_never_reaches_the_answer(assistant):
    """A bridge that produces its object but the answer drops it must fail."""

    def broken(question, **kwargs):
        answer = assistant.ask(question, **kwargs)
        answer.lead = None
        answer.plan = None
        return answer

    assert answer_path_accuracy(answer_for=broken) < 1.0


def test_metric_fails_when_a_negative_control_invents_a_layer(assistant):
    """A definitional question that gains a reasoning layer must fail its control."""

    def leaky(question, **kwargs):
        answer = assistant.ask(question, **kwargs)
        if "what is permeability" in question:
            from core.answer_plan import answer_plan

            answer.plan = answer_plan("how do I measure porosity?")
        return answer

    assert answer_path_accuracy(answer_for=leaky) < 1.0


def test_body_check_rejects_a_lead_that_is_only_on_the_object(assistant):
    """A lead object present but absent from the body must fail the body check."""
    procedural = next(
        e for e in AUDIT_EXPECTATIONS if e.name.startswith("a procedural question")
    )

    def trace_only(question, **kwargs):
        answer = assistant.ask(question, **kwargs)
        # Keep the lead object but drop it from the body the reader sees.
        answer.text = "Here is what the available sources say."
        return answer

    assert not audit_answer(procedural, trace_only)


def test_audit_fails_honestly_when_the_path_raises():
    def exploding(question, **kwargs):
        raise RuntimeError("the model blew up")

    assert answer_path_accuracy(answer_for=exploding) == 0.0


def test_audit_fails_honestly_when_the_path_returns_nothing():
    assert answer_path_accuracy(answer_for=lambda question, **kwargs: None) == 0.0


# --- Phase 49: the Arabic-parallel audit --------------------------------------


def test_arabic_catalogue_only_names_known_layers():
    for expectation in AUDIT_EXPECTATIONS_AR:
        assert set(expectation.required) <= ALL_LAYERS
        assert set(expectation.forbidden) <= ALL_LAYERS


def test_arabic_catalogue_has_positive_expectations_and_negative_controls():
    positives = [e for e in AUDIT_EXPECTATIONS_AR if e.required]
    negatives = [e for e in AUDIT_EXPECTATIONS_AR if e.forbidden]
    assert positives, "the Arabic audit must check layers do arrive"
    assert negatives, "the Arabic audit must include negative controls"
    for control in negatives:
        assert set(control.forbidden) == REASONING_LAYERS


def test_arabic_catalogue_is_a_true_parallel_of_the_english_one():
    """Each English expectation has an Arabic twin asking for the same layers."""
    assert len(AUDIT_EXPECTATIONS_AR) == len(AUDIT_EXPECTATIONS)
    english = {(e.required, e.forbidden, e.timeline) for e in AUDIT_EXPECTATIONS}
    arabic = {(e.required, e.forbidden, e.timeline) for e in AUDIT_EXPECTATIONS_AR}
    assert english == arabic


def test_arabic_catalogue_questions_are_arabic():
    for expectation in AUDIT_EXPECTATIONS_AR:
        assert any("\u0600" <= ch <= "\u06ff" for ch in expectation.question), expectation.name


def test_arabic_path_accuracy_is_one_on_the_real_assistant(assistant):
    assert answer_path_accuracy(AUDIT_EXPECTATIONS_AR, answer_for=assistant.ask) == 1.0


def test_arabic_path_accuracy_is_honest_with_no_path():
    assert answer_path_accuracy(AUDIT_EXPECTATIONS_AR, answer_for=None) == 0.0


def test_arabic_audit_fails_when_a_layer_is_dropped(assistant):
    """A leaked Arabic answer must fail, exactly as the English one does."""
    evaluative = next(e for e in AUDIT_EXPECTATIONS_AR if e.required == (LAYER_BELIEF, LAYER_LEAD, LAYER_REASONING))

    def leaky(question, **kwargs):
        answer = assistant.ask(question, **kwargs)
        answer.belief = None
        return answer

    assert answer_path_accuracy((evaluative,), answer_for=leaky) < 1.0


def test_arabic_evaluative_question_reaches_the_belief_engine():
    """The root cause Phase 49 fixed: an Arabic term must resolve to its concept."""
    from core.answer_belief import answer_belief
    from core.cognitive_controller import CognitiveController

    ontology = CognitiveController().ontology
    belief = answer_belief("هل التحميض يزيد النفاذية؟", ontology=ontology)
    assert belief is not None
    assert belief.supporting, "the Arabic evaluative question must find supporting relations"


def test_arabic_analogical_question_is_analogical():
    from core.answer_analogy import is_analogical

    assert is_analogical("هل المسامية مثل النفاذية؟")
    assert not is_analogical("ما هي المسامية؟")
