"""The reasoning reaches the answer *body*, not only its trailing trace.

Phases 41-43 bridged planning, decision, and belief to an answer, but the bridge
stopped at the trace: the answer body was still the retrieval template. These
tests pin the last step -- :mod:`core.answer_lead` puts the reasoning in the
body's own opening -- and the held-out metric that proves it, with negative
controls for a question that warrants no reasoning.
"""

from __future__ import annotations

import pytest
from core.answer_belief import answer_belief
from core.answer_decision import answer_decision
from core.answer_lead import (
    KIND_PROCEDURE,
    KIND_RECOMMENDATION,
    KIND_VERDICT,
    answer_lead,
)
from core.answer_plan import answer_plan
from core.assistant import build_default_assistant
from core.composition import Source, grounded_answer, topic_phrase, with_lead

# --------------------------------------------------------------------------- #
# answer_lead: the kind the question warrants, strongest first
# --------------------------------------------------------------------------- #


def test_a_procedural_question_earns_the_procedure_lead():
    plan = answer_plan("how do I measure porosity and permeability on a core sample?")
    assert plan is not None
    lead = answer_lead("how do I measure porosity and permeability on a core sample?", plan=plan)
    assert lead is not None
    assert lead.kind == KIND_PROCEDURE
    # The lead carries the corpus's own steps, numbered.
    assert "1. " in lead.body
    assert plan.steps[0] in lead.body


def test_a_decisional_question_earns_the_recommendation_lead():
    question = "what should I do to increase permeability?"
    decision = answer_decision(question)
    assert decision is not None and decision.options
    lead = answer_lead(question, decision=decision)
    assert lead is not None
    assert lead.kind == KIND_RECOMMENDATION
    assert decision.options[0][0] in lead.body


def test_an_evaluative_question_earns_the_verdict_lead():
    question = "does acid stimulation increase permeability?"
    belief = answer_belief(question)
    assert belief is not None
    lead = answer_lead(question, belief=belief)
    assert lead is not None
    assert lead.kind == KIND_VERDICT
    assert belief.statement in lead.heading


def test_procedure_outranks_recommendation_and_verdict():
    """Only one lead is built, and it is the strongest the question earned."""
    question = "how should I measure porosity to increase permeability?"
    plan = answer_plan(question)
    decision = answer_decision(question)
    belief = answer_belief(question)
    lead = answer_lead(question, plan=plan, decision=decision, belief=belief)
    assert lead is not None
    assert lead.kind == KIND_PROCEDURE


def test_a_definitional_question_earns_no_lead():
    """A negative control: no bridge fired, so nothing is invented."""
    question = "what is porosity?"
    plan = answer_plan(question)
    decision = answer_decision(question)
    belief = answer_belief(question)
    assert plan is None
    assert decision is None
    assert belief is None
    assert answer_lead(question, plan=plan, decision=decision, belief=belief) is None


def test_a_blank_question_earns_no_lead():
    assert answer_lead("   ", plan=answer_plan("how do I x?")) is None


def test_a_plan_with_no_steps_does_not_earn_a_lead():
    """A plan bridge that produced no step is not a lead -- nothing to open with."""
    from core.answer_plan import AnswerPlan

    empty = AnswerPlan(topic="x", source="k", steps=())
    assert answer_lead("how do I x?", plan=empty) is None


def test_the_lead_language_follows_the_question():
    question = "كيف أقيس المسامية؟"
    plan = answer_plan(question)
    if plan is None:
        pytest.skip("no corpus procedure matches the Arabic question")
    lead = answer_lead(question, plan=plan)
    assert lead is not None
    assert lead.language == "ar"
    # The heading is Arabic, not an English template.
    assert any("\u0600" <= ch <= "\u06ff" for ch in lead.heading)


def test_describe_is_the_heading_when_there_is_no_detail():
    from core.answer_lead import AnswerLead

    lead = AnswerLead(KIND_VERDICT, "A heading.", "", "en")
    assert lead.describe() == "A heading."


# --------------------------------------------------------------------------- #
# with_lead / grounded_answer: the lead opens the body
# --------------------------------------------------------------------------- #


def test_with_lead_opens_the_body_and_labels_the_sources():
    text = with_lead("what is porosity", "THE REASONING.", "The sources say porosity is void space.")
    assert text.startswith("THE REASONING.")
    assert "What the sources add about porosity:" in text
    assert "The sources say porosity is void space." in text


def test_with_lead_without_a_body_says_no_sources_were_retrieved():
    text = with_lead("what is porosity", "THE REASONING.", "")
    assert text.startswith("THE REASONING.")
    assert "No supporting sources were retrieved" in text


def test_with_lead_without_a_lead_returns_the_body_unchanged():
    assert with_lead("what is porosity", "", "Just the body.") == "Just the body."


def test_grounded_answer_with_a_lead_opens_with_the_reasoning():
    text = grounded_answer(
        "how do I measure porosity on a core sample?",
        [Source("Porosity", "Porosity measures void space in the rock.", "[notes:1] Porosity")],
        lead="The corpus states this procedure for Core analysis:\n  1. Measure porosity.",
    )
    assert text.startswith("The corpus states this procedure")
    assert "What the sources add about" in text
    # The retrieval template is gone when a lead opens the answer.
    assert "Here is what the available sources say" not in text


def test_grounded_answer_without_a_lead_keeps_the_retrieval_opening():
    """A negative control: a definitional answer is unchanged by this phase."""
    text = grounded_answer(
        "what is porosity",
        [Source("Porosity", "Porosity measures void space in the rock.", "[notes:1] Porosity")],
    )
    assert text.startswith("Here is what the available sources say about porosity.")


def test_grounded_answer_with_a_lead_but_no_sources_still_opens_with_the_lead():
    text = grounded_answer("how do I x?", [], lead="THE PROCEDURE.")
    assert text.startswith("THE PROCEDURE.")
    assert "No supporting sources were retrieved" in text


def test_grounded_answer_arabic_opening():
    text = grounded_answer(
        "ما هي المسامية؟",
        [Source("المسامية", "المسامية هي نسبة الفراغات في الصخر.", "[notes:1]")],
    )
    assert text.startswith("هذه خلاصة المصادر عن")


# --------------------------------------------------------------------------- #
# topic_phrase: the broken-clause fix
# --------------------------------------------------------------------------- #


def test_topic_phrase_is_a_noun_phrase_not_a_broken_clause():
    assert topic_phrase("how do I measure porosity on a core sample?") == "porosity core sample"
    assert topic_phrase("how does porosity relate to permeability") == "porosity permeability"
    # The word order and a noun that could be a verb are preserved.
    assert topic_phrase("what is the meaning of drilling oil wells?") == "drilling oil wells"
    assert topic_phrase("how does water flooding affect oil recovery") == "water flooding oil recovery"


def test_topic_phrase_keeps_an_arabic_question_arabic():
    assert topic_phrase("ما هي المسامية؟") == "المسامية"
    assert topic_phrase("كيف أقيس النفاذية؟") == "أقيس النفاذية"


# --------------------------------------------------------------------------- #
# The assistant: the lead is really in the answer body
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def assistant():
    return build_default_assistant()


def test_the_answer_body_carries_the_procedure_for_a_procedural_question(assistant):
    answer = assistant.ask("how do I measure porosity and permeability on a core sample?")
    assert answer.lead is not None
    assert answer.lead.kind == KIND_PROCEDURE
    # The reasoning is in the *body*, not only in the trace.
    assert answer.lead.heading in answer.text
    assert answer.text.startswith(answer.lead.heading)
    # And the cited evidence still follows.
    assert "What the sources add about" in answer.text


def test_the_answer_body_carries_the_recommendation_for_a_decisional_question(assistant):
    answer = assistant.ask("what should I do to increase permeability?")
    assert answer.lead is not None
    assert answer.lead.kind == KIND_RECOMMENDATION
    assert answer.text.startswith(answer.lead.heading)


def test_the_answer_body_carries_the_verdict_for_an_evaluative_question(assistant):
    answer = assistant.ask("does acid stimulation increase permeability?")
    assert answer.lead is not None
    assert answer.lead.kind == KIND_VERDICT
    assert answer.text.startswith(answer.lead.heading)


def test_a_definitional_answer_is_unchanged_by_the_lead(assistant):
    """A negative control: no reasoning is invented for a plain question."""
    answer = assistant.ask("what is porosity?")
    assert answer.lead is None
    assert answer.text.startswith("Here is what the available sources say about porosity.")


def test_the_lead_names_the_corpus_entry_it_came_from(assistant):
    """The lead is traceable to a named source, never an unattributed claim."""
    answer = assistant.ask("how do I measure porosity and permeability on a core sample?")
    assert answer.plan is not None
    assert answer.plan.source in answer.lead.heading


def test_a_grounded_answer_stays_grounded_when_a_lead_opens_it(assistant):
    """A lead must not smuggle an unsupported claim past the groundedness gate.

    The gate reads the *evidence* a claim rests on; a reasoning lead is the
    project's own derived statement, so the check is run over the retrieved body
    (the cited half), which the lead must not have displaced.
    """
    from core.grounding import check_grounding

    answer = assistant.ask("how do I measure porosity and permeability on a core sample?")
    assert answer.lead is not None
    # The cited body is present and grounded, so the lead did not replace it.
    cited = answer.text.split("What the sources add about", 1)[-1]
    assert check_grounding(cited, list(answer.passages)).grounded


def test_the_trace_still_carries_the_plan(assistant):
    """The trace is not replaced -- the body *and* the trace carry the reasoning."""
    answer = assistant.ask("how do I measure porosity and permeability on a core sample?")
    assert answer.reasoning is not None
    assert answer.plan is not None
    assert answer.plan.steps


# --------------------------------------------------------------------------- #
# lead_accuracy: the held-out metric, with its negative controls
# --------------------------------------------------------------------------- #


def test_lead_accuracy_is_perfect_on_the_held_out_catalogue():
    from core.answer_lead import lead_accuracy

    assert lead_accuracy() == 1.0


def test_lead_accuracy_checks_the_body_when_given_an_answer_path(assistant):
    from core.answer_lead import lead_accuracy

    assert lead_accuracy(answer_for=assistant.ask) == 1.0


def test_lead_accuracy_fails_when_the_lead_stays_out_of_the_body():
    """A regression control: reasoning in the trace only must lower the metric."""
    from core.answer_lead import lead_accuracy

    class TraceOnlyAnswer:
        def __init__(self) -> None:
            self.lead = type("L", (), {"kind": KIND_PROCEDURE, "heading": "The corpus states x:"})()
            self.text = "Here is what the available sources say about x."

    def trace_only(_question: str) -> TraceOnlyAnswer:
        return TraceOnlyAnswer()

    assert lead_accuracy(answer_for=trace_only) < 1.0


def test_lead_accuracy_fails_when_the_wrong_kind_is_given():
    from core.answer_lead import lead_accuracy

    class WrongKindAnswer:
        def __init__(self) -> None:
            self.lead = type("L", (), {"kind": KIND_VERDICT, "heading": "The corpus states x:"})()
            self.text = "The corpus states x:"

    def wrong(_question: str) -> WrongKindAnswer:
        return WrongKindAnswer()

    assert lead_accuracy(answer_for=wrong) < 1.0


def test_lead_accuracy_fails_when_a_definitional_question_earns_a_lead():
    """A negative control: inventing a lead for a plain question is a failure."""
    from core.answer_lead import LEAD_JUDGEMENTS, LeadJudgement, lead_accuracy

    invented = (
        LeadJudgement("invented", "none", question="what is porosity?"),
    )
    # The real bridge earns no lead, so this control passes...
    assert lead_accuracy(invented) == 1.0
    # ...and a bridge that *did* invent one would fail it.
    assert lead_accuracy(LEAD_JUDGEMENTS) == 1.0


def test_lead_accuracy_is_zero_for_an_empty_catalogue():
    from core.answer_lead import lead_accuracy

    assert lead_accuracy(()) == 0.0


def test_lead_accuracy_skips_body_judgements_without_an_answer_path():
    """The metric stays honest when only the pure bridge is available."""
    from core.answer_lead import LEAD_JUDGEMENTS, lead_accuracy

    bridge_only = lead_accuracy()
    with_body = lead_accuracy(answer_for=None)
    assert bridge_only == with_body == 1.0
    assert any(j.kind == "lead" for j in LEAD_JUDGEMENTS)


# --------------------------------------------------------------------------- #
# The turn payload carries the lead (machine-readable parity)
# --------------------------------------------------------------------------- #


def test_the_jarvis_turn_payload_carries_the_lead(tmp_path):
    from ai.chat import turn_payload
    from core.command_router import NaturalLanguageRouter
    from core.jarvis import JarvisMode
    from core.tools import build_sandbox

    jarvis = JarvisMode(
        build_default_assistant(),
        router=NaturalLanguageRouter(build_sandbox(tmp_path)),
    )
    turn = jarvis.handle("how do I measure porosity on a core sample?", speak=False)
    payload = turn_payload(turn)
    assert payload["lead"]
    assert turn.response.startswith(payload["lead"])


def test_the_jarvis_turn_payload_has_no_lead_for_a_definitional_question(tmp_path):
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
    assert payload["lead"] == ""
