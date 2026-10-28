"""Tests for Phase 51: concept identifiers rendered readably in Arabic answers.

Phase 50 made the Arabic *output path* exist, but a measured battery found the
graph's own snake_case identifiers still reaching an Arabic answer **body** --
25 across 8 questions (``shut_in``, ``injection_pattern``,
``blowout_preventer_stack``), because the reasoning layers printed a concept
*label* and a label the glossary has not reached yet is a code token, not a word.
These tests pin the fix: :func:`core.arabic_prose.arabic_render` renders a label
readably, :func:`core.arabic_prose.render_arabic_text` is the one pass over an
assembled Arabic text, the reasoning layers use it (belief, analogy, decision,
procedure), and the English path is byte-identical.
"""

from __future__ import annotations

import re

import pytest
from core.answer_analogy import answer_analogy
from core.answer_belief import answer_belief
from core.answer_decision import answer_decision
from core.answer_lead import (
    KIND_ANALOGY,
    KIND_RECOMMENDATION,
    KIND_VERDICT,
    answer_lead,
)
from core.arabic_glossary import has_arabic
from core.arabic_prose import arabic_label, arabic_render, render_arabic_text
from core.assistant import build_default_assistant
from core.cognitive_controller import CognitiveController

#: The shape a concept label takes in the graph -- what must never reach a reader.
_IDENTIFIER_RE = re.compile(
    r"(?<![A-Za-z0-9_])[a-z][a-z0-9]*(?:_[a-z0-9]+)+(?![A-Za-z0-9_])"
)

#: A measured Arabic battery: each question reaches a different reasoning layer
#: (definition, procedure, verdict, recommendation, comparison), so a leak in any
#: one of them is caught.
ARABIC_BATTERY: tuple[str, ...] = (
    "ما هي المسامية؟",
    "كيف أقيس النفاذية؟",
    "هل التحميض يزيد النفاذية؟",
    "ماذا أفعل إذا حدث اندفاع في البئر؟",
    "هل التكسير الهيدروليكي مثل التحميض؟",
    "لماذا ينخفض الضغط في المكمن مع الإنتاج؟",
    "كيف نحفر بئرًا؟",
    "كيف نمنع الانفجار؟",
    "ماذا أفعل لزيادة الإنتاج؟",
    "ما هو الإغراق المائي؟",
    "كيف نتحكم في الرمل؟",
    "ما هي خطوات اختبار الضغط؟",
)


@pytest.fixture(scope="module")
def controller() -> CognitiveController:
    return CognitiveController()


@pytest.fixture(scope="module")
def assistant():
    return build_default_assistant()


# --- the renderer ----------------------------------------------------------


def test_arabic_render_prefers_the_ontology_term(controller):
    assert arabic_render("porosity", controller.ontology) == "مسامية"
    assert arabic_render("permeability", controller.ontology) == "نفاذية"


def test_arabic_render_turns_an_unknown_identifier_into_words(controller):
    """A label the glossary has not reached yet must read as words, not a token."""
    rendered = arabic_render("not_a_real_concept", controller.ontology)
    assert "_" not in rendered
    assert rendered == "not a real concept"


def test_arabic_render_leaves_a_known_latin_identifier_alone(controller):
    """A Latin identifier the domain keeps (SPE) is not turned into words."""
    assert arabic_render("SPE", controller.ontology) == "SPE"


def test_arabic_label_still_leaves_an_unknown_identifier_unchanged(controller):
    """The training-corpus renderer is untouched: it must never invent a term."""
    assert arabic_label("not_a_real_concept", controller.ontology) == "not_a_real_concept"


# --- the pass over an assembled text ---------------------------------------


def test_render_arabic_text_renders_every_identifier(controller):
    text = "الخيارات: not_a_real_concept و another_made_up_term و third_one."
    rendered = render_arabic_text(text, controller.ontology)
    assert not _IDENTIFIER_RE.search(rendered)
    assert "not a real concept" in rendered


def test_render_arabic_text_uses_the_ontology_term(controller):
    rendered = render_arabic_text("لـshut_in:", controller.ontology)
    assert rendered == "لـالإغلاق:"


def test_render_arabic_text_handles_an_arabic_prefix(controller):
    """An Arabic prefix ("لـ") must not hide the identifier behind it."""
    rendered = render_arabic_text("لـnot_a_real_concept:", controller.ontology)
    assert not _IDENTIFIER_RE.search(rendered)
    assert "not a real concept" in rendered


def test_render_arabic_text_is_idempotent(controller):
    once = render_arabic_text("لـgas_kick و shut_in", controller.ontology)
    assert render_arabic_text(once, controller.ontology) == once


def test_render_arabic_text_leaves_plain_arabic_alone(controller):
    text = "التحفيز الحمضي يزيد النفاذية."
    assert render_arabic_text(text, controller.ontology) == text


# --- the reasoning layers --------------------------------------------------


def test_the_belief_verdict_is_arabic_for_an_arabic_question(controller):
    belief = answer_belief("هل التحميض يزيد النفاذية؟", ontology=controller.ontology)
    assert belief is not None
    assert belief.language == "ar"
    # The evidence lines are Arabic, not English identifiers.
    for line in (*belief.supporting, *belief.contradicting):
        assert not _IDENTIFIER_RE.search(line), line
        assert has_arabic(line), line


def test_the_belief_verdict_stays_english_for_an_english_question(controller):
    belief = answer_belief("does acid stimulation increase permeability?", ontology=controller.ontology)
    assert belief is not None
    assert belief.language == "en"
    assert any(line.startswith("acid_stimulation —increases→") for line in belief.supporting)


def test_the_verdict_lead_carries_no_identifier(controller):
    belief = answer_belief("هل التحميض يزيد النفاذية؟", ontology=controller.ontology)
    lead = answer_lead(
        "هل التحميض يزيد النفاذية؟", belief=belief, ontology=controller.ontology
    )
    assert lead is not None
    assert lead.kind == KIND_VERDICT
    assert not _IDENTIFIER_RE.search(lead.describe())


def test_the_recommendation_lead_carries_no_identifier(controller):
    question = "ماذا أفعل إذا حدث اندفاع في البئر؟"
    decision = answer_decision(question, ontology=controller.ontology)
    assert decision is not None
    lead = answer_lead(question, decision=decision, ontology=controller.ontology)
    assert lead is not None
    assert lead.kind == KIND_RECOMMENDATION
    assert not _IDENTIFIER_RE.search(lead.describe())
    # The goal joins the question's concepts with the Arabic connective.
    assert " و" in lead.heading


def test_the_comparison_lead_carries_no_identifier(controller):
    question = "هل التكسير الهيدروليكي مثل التحميض؟"
    comparison = answer_analogy(question, ontology=controller.ontology)
    assert comparison is not None
    lead = answer_lead(question, analogy=comparison, ontology=controller.ontology)
    assert lead is not None
    assert lead.kind == KIND_ANALOGY
    assert not _IDENTIFIER_RE.search(lead.describe())


def test_the_english_lead_is_unchanged(controller):
    """The English path keeps the graph's own identifiers, as it always did."""
    decision = answer_decision("what should I do to increase permeability?", ontology=controller.ontology)
    lead = answer_lead(
        "what should I do to increase permeability?", decision=decision, ontology=controller.ontology
    )
    assert lead is not None
    assert lead.language == "en"
    assert "acid_stimulation" in lead.body


# --- the whole answer, end to end ------------------------------------------


def test_no_identifier_reaches_an_arabic_answer_body(assistant):
    """The measured gap: a battery of Arabic questions yields no code token."""
    leaks: dict[str, str] = {}
    for question in ARABIC_BATTERY:
        answer = assistant.ask(question)
        for identifier in _IDENTIFIER_RE.findall(answer.text):
            leaks.setdefault(identifier, question)
    assert not leaks, f"identifiers leaked into Arabic answers: {leaks}"


def test_an_english_answer_keeps_its_identifiers(assistant):
    """A negative control: the English path is not rendered."""
    answer = assistant.ask("what should I do to increase permeability?")
    assert "acid_stimulation" in answer.text
