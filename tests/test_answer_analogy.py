"""The analogical layer reaches an answer: a question about likeness gets a comparison.

The project carried a whole analogical layer -- ``AnalogyEngine``,
``StructureMappingEngine``, ``Structure``, ``StructureMapping`` -- but it was
reachable only from the controller, the ``analogy`` CLI, and the evaluation
scenarios, so *"is porosity like permeability?"* got a definition back (or, worse,
the belief bridge read its ``is ... ?`` opening as a verdict about a *different*
claim). These tests pin :mod:`core.answer_analogy` -- the bridge that aligns the
two concepts' relational neighbourhoods with the project's real structure-mapping
engine -- and the held-out metric that proves it, with negative controls for a
question that warrants no comparison.
"""

from __future__ import annotations

import pytest
from core.answer_analogy import (
    ANALOGY_CUES,
    AnalogyBridgeJudgement,
    analogy_answer_accuracy,
    answer_analogy,
    is_analogical,
)
from core.answer_belief import answer_belief
from core.answer_lead import KIND_ANALOGY, answer_lead
from core.assistant import build_default_assistant

# --------------------------------------------------------------------------- #
# is_analogical: the cue gate
# --------------------------------------------------------------------------- #


def test_an_analogical_question_is_recognised_in_both_languages():
    assert is_analogical("is porosity like permeability?")
    assert is_analogical("is the density log similar to the sonic log?")
    assert is_analogical("compare porosity and permeability")
    assert is_analogical("هل المسامية تشبه النفاذية؟")


def test_a_definitional_question_is_not_analogical():
    assert not is_analogical("what is porosity?")
    assert not is_analogical("ما هي المسامية؟")
    assert not is_analogical("what is the capital of France?")


def test_a_question_ending_in_like_is_still_analogical():
    """The cue is normalised, so a trailing "like?" fires like an interior one."""
    assert is_analogical("what is porosity like?")


def test_a_verdict_question_is_not_analogical():
    """A negative control: a plain yes/no claim is not a comparison."""
    assert not is_analogical("does acid stimulation increase permeability?")


def test_the_cue_catalogue_is_non_empty():
    assert ANALOGY_CUES


# --------------------------------------------------------------------------- #
# answer_analogy: the comparison
# --------------------------------------------------------------------------- #


def test_two_analogous_concepts_are_compared():
    comparison = answer_analogy("is porosity like permeability?")
    assert comparison is not None
    assert comparison.base == "porosity"
    assert comparison.target == "permeability"
    assert comparison.meaningful is True
    assert comparison.matched


def test_two_instruments_are_compared():
    comparison = answer_analogy("is the density log similar to the sonic log?")
    assert comparison is not None
    assert comparison.meaningful is True


def test_a_property_and_an_operation_are_reported_as_not_an_analogy():
    """The verdict is the engine's, not dressed up: no shared system is said so."""
    comparison = answer_analogy("is porosity like waterflood?")
    assert comparison is not None
    assert comparison.meaningful is False


def test_the_comparison_reports_what_it_could_not_map():
    comparison = answer_analogy("is porosity like permeability?")
    assert comparison is not None
    # The target's neighbourhood holds concepts the base's does not.
    assert comparison.unmapped_target


def test_a_definitional_question_gets_no_comparison():
    """A negative control: no comparison is invented for a plain question."""
    assert answer_analogy("what is porosity?") is None


def test_an_unrelated_question_gets_no_comparison():
    assert answer_analogy("what is the capital of France?") is None


def test_a_comparison_naming_one_concept_gets_none():
    assert answer_analogy("what is porosity like?") is None


def test_a_blank_question_gets_no_comparison():
    assert answer_analogy("   ") is None


def test_the_comparison_language_follows_the_question():
    comparison = answer_analogy("هل المسامية تشبه النفاذية؟")
    assert comparison is not None
    assert comparison.language == "ar"
    assert any("\u0600" <= ch <= "\u06ff" for ch in comparison.describe())


def test_the_transfer_limit_is_enforced():
    limited = answer_analogy("is porosity like permeability?", limit=1)
    assert limited is not None
    assert len(limited.transferred) <= 1


def test_a_bad_transfer_limit_is_refused():
    with pytest.raises(ValueError):
        answer_analogy("is porosity like permeability?", limit=0)


# --------------------------------------------------------------------------- #
# answer_lead: the comparison can open the answer
# --------------------------------------------------------------------------- #


def test_a_comparison_can_open_the_answer_as_a_lead():
    comparison = answer_analogy("is porosity like permeability?")
    assert comparison is not None
    lead = answer_lead("is porosity like permeability?", analogy=comparison)
    assert lead is not None
    assert lead.kind == KIND_ANALOGY
    assert lead.heading in lead.describe()


def test_a_comparison_with_no_match_does_not_earn_a_lead():
    from core.answer_analogy import AnswerAnalogy

    empty = AnswerAnalogy(
        base="a", target="b", meaningful=False, score=0.0, matched=(), transferred=(),
        unmapped_base=(), unmapped_target=(), language="en",
    )
    assert answer_lead("is a like b?", analogy=empty) is None


def test_the_comparison_outranks_the_verdict():
    """Only one lead is built: a comparison question is not also a verdict."""
    question = "is porosity like permeability?"
    comparison = answer_analogy(question)
    belief = answer_belief(question)
    assert comparison is not None
    lead = answer_lead(question, analogy=comparison, belief=belief)
    assert lead is not None
    assert lead.kind == KIND_ANALOGY


# --------------------------------------------------------------------------- #
# The assistant: the comparison is really in the answer body
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def assistant():
    return build_default_assistant()


def test_the_assistant_carries_a_comparison_for_an_analogical_question(assistant):
    answer = assistant.ask("is porosity like permeability?")
    assert answer.analogy is not None
    assert answer.lead is not None
    assert answer.lead.kind == KIND_ANALOGY
    # The comparison is in the *body*, not only in the trace.
    assert answer.text.startswith(answer.lead.heading)


def test_an_analogical_question_earns_no_belief(assistant):
    """The belief bridge must not answer a comparison as a claim about the pair."""
    answer = assistant.ask("is porosity like permeability?")
    assert answer.belief is None


def test_a_definitional_answer_carries_no_comparison(assistant):
    """A negative control: a plain question gets no comparison."""
    answer = assistant.ask("what is porosity?")
    assert answer.analogy is None


def test_the_trace_carries_the_comparison(assistant):
    """The trace is not replaced -- the body *and* the trace carry the comparison."""
    from core.reasoning_report import SOURCE_ANALOGY

    answer = assistant.ask("is porosity like permeability?")
    assert answer.reasoning is not None
    labels = {step.label for step in answer.reasoning.steps if step.lines}
    assert SOURCE_ANALOGY in labels


# --------------------------------------------------------------------------- #
# analogy_answer_accuracy: the held-out metric, with its negative controls
# --------------------------------------------------------------------------- #


def test_analogy_answer_accuracy_is_perfect_on_the_held_out_catalogue():
    assert analogy_answer_accuracy() == 1.0


def test_analogy_answer_accuracy_is_zero_for_an_empty_catalogue():
    assert analogy_answer_accuracy(()) == 0.0


def test_analogy_answer_accuracy_fails_when_a_definitional_question_earns_one():
    """A negative control: inventing a comparison for a plain question fails."""
    judgements = (
        AnalogyBridgeJudgement("invented", "no-analogy", question="what is porosity?"),
    )
    # The real bridge earns none, so this control passes.
    assert analogy_answer_accuracy(judgements) == 1.0


def test_analogy_answer_accuracy_fails_when_the_verdict_is_wrong():
    wrong = (
        AnalogyBridgeJudgement(
            "wrong verdict",
            "meaningful",
            question="is porosity like waterflood?",
            expected=True,
        ),
    )
    assert analogy_answer_accuracy(wrong) == 0.0


# --------------------------------------------------------------------------- #
# The turn payload carries the comparison (machine-readable parity)
# --------------------------------------------------------------------------- #


def test_the_jarvis_turn_payload_carries_the_comparison(tmp_path):
    from ai.chat import turn_payload
    from core.command_router import NaturalLanguageRouter
    from core.jarvis import JarvisMode
    from core.tools import build_sandbox

    jarvis = JarvisMode(
        build_default_assistant(),
        router=NaturalLanguageRouter(build_sandbox(tmp_path)),
    )
    turn = jarvis.handle("is porosity like permeability?", speak=False)
    payload = turn_payload(turn)
    assert payload["analogy"]
    assert any("matched" in line for line in payload["analogy"])


def test_the_jarvis_turn_payload_has_no_comparison_for_a_definitional_question(tmp_path):
    from ai.chat import turn_payload
    from core.command_router import NaturalLanguageRouter
    from core.jarvis import JarvisMode
    from core.tools import build_sandbox

    jarvis = JarvisMode(
        build_default_assistant(),
        router=NaturalLanguageRouter(build_sandbox(tmp_path)),
    )
    turn = jarvis.handle("what is porosity?", speak=False)
    payload = turn_payload(turn)
    assert payload["analogy"] == []
