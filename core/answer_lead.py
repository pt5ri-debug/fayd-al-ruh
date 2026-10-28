"""Put the reasoning *in the answer body*, not only in its trailing trace.

Phases 41-43 bridged the planning, decision, and belief layers to an answer: a
question that asks *how* now carries the corpus's ordered procedure, one that
asks *what to do* carries the graph's ranked recommendation, and one that asks
*whether a claim holds* carries the evidence-weighed verdict. But the bridge
stopped at the **trace**: the answer *body* was still the retrieval template
("Here is what the available sources say about ..."), so the reasoning the
project had just reached the reader sat in a footer a user may never open.

This module closes that last step. :func:`answer_lead` turns the three bridges
into the **opening of the answer itself** -- a heading plus the reasoning's own
detail -- so a procedural question is answered *with the procedure*, a decisional
one with the recommendation, and an evaluative one with the verdict. The
retrieved evidence then follows underneath, labelled as what the sources add, so
the reader still sees the cited text.

It is **read-only** and **honest by construction**: it renders only the objects
the bridges already produced, it invents no step, option, or evidence, and it
returns ``None`` when no bridge fired -- so a plain definitional question keeps
the retrieval answer it had and nothing is fabricated to fill a lead.

It is **bilingual**: the lead is written in the language of the *question* (an
Arabic question gets an Arabic lead), reusing :func:`core.voices.detect_language`
so there is no second script check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.voices import detect_language

if TYPE_CHECKING:
    from core.answer_belief import AnswerBelief
    from core.answer_decision import AnswerDecision
    from core.answer_plan import AnswerPlan

#: The lead kinds, strongest first: a question that asks *how* is answered with
#: the procedure, one that asks *what to do* with the recommendation, and one
#: that asks *whether a claim holds* with the verdict. Only one lead is built --
#: the first kind the question actually warrants -- so a decisional question that
#: happens to name two concepts is not also handed a verdict.
KIND_PROCEDURE = "procedure"
KIND_RECOMMENDATION = "recommendation"
KIND_VERDICT = "verdict"

#: The lifecycle status the belief engine assigns, in Arabic. A status the map
#: does not carry is left as the engine's own word rather than guessed.
_STATUS_AR: dict[str, str] = {
    "accepted": "مقبول",
    "provisional": "مبدئي",
    "suspended": "معلَّق",
    "rejected": "مرفوض",
}


@dataclass(frozen=True)
class AnswerLead:
    """The reasoning the question warrants, as the opening of the answer."""

    kind: str
    """One of :data:`KIND_PROCEDURE`, :data:`KIND_RECOMMENDATION`, :data:`KIND_VERDICT`."""

    heading: str
    """The lead sentence that names what the reasoning is (in the question's language)."""

    body: str
    """The reasoning's own detail: the numbered steps, the ranked options, or the
    support lines. Empty when the reasoning carries no detail."""

    language: str
    """``"ar"`` or ``"en"`` -- the language the lead was written in."""

    def describe(self) -> str:
        """The lead as a heading followed by its detail, or the heading alone."""
        if not self.body.strip():
            return self.heading
        return f"{self.heading}\n{self.body}"


def _procedure_lead(plan: AnswerPlan, language: str) -> AnswerLead:
    """The plan bridge as an answer lead: the corpus's ordered steps.

    The heading names the corpus entry the procedure came from (``plan.source``),
    so a reader can trace the reasoning to the project's own curated corpus -- it
    is a *named* source, not an unattributed claim -- even when the retriever
    surfaced different passages for the question.
    """
    steps = plan.steps
    if language == "ar":
        heading = f"يحمل متن المشروع هذا الإجراء لـ{plan.topic} ({plan.source}):"
        body = "\n".join(f"  {index}. {step}" for index, step in enumerate(steps, start=1))
    else:
        heading = f"The corpus states this procedure for {plan.topic} ({plan.source}):"
        body = "\n".join(f"  {index}. {step}" for index, step in enumerate(steps, start=1))
    return AnswerLead(KIND_PROCEDURE, heading, body, language)


def _recommendation_lead(decision: AnswerDecision, language: str) -> AnswerLead:
    """The decision bridge as an answer lead: the graph's ranked options."""
    options = decision.options
    if language == "ar":
        heading = (
            f"الخيارات المرجّحة التي يحملها الرسم البياني لـ{decision.goal}:"
        )
        body = "\n".join(
            f"  - {option} ({relation}، ثقة {confidence:.2f})"
            for option, relation, confidence in options
        )
    else:
        heading = f"The graph's ranked recommendation for {decision.goal}:"
        body = "\n".join(
            f"  - {option} ({relation}, confidence {confidence:.2f})"
            for option, relation, confidence in options
        )
    return AnswerLead(KIND_RECOMMENDATION, heading, body, language)


def _verdict_lead(belief: AnswerBelief, language: str) -> AnswerLead:
    """The belief bridge as an answer lead: the evidence-weighed verdict."""
    status = belief.status
    if language == "ar":
        status = _STATUS_AR.get(status, status)
        heading = (
            f"الحكم المرجَّح بالدليل على «{belief.statement}»: "
            f"ثقة {belief.confidence:.2f} ({status})."
        )
        lines = [f"  - يدعم: {line}" for line in belief.supporting]
        lines += [f"  - يعارض: {line}" for line in belief.contradicting]
    else:
        heading = (
            f"The evidence-weighed verdict on \"{belief.statement}\": "
            f"confidence {belief.confidence:.2f} ({status})."
        )
        lines = [f"  - supports: {line}" for line in belief.supporting]
        lines += [f"  - contradicts: {line}" for line in belief.contradicting]
    return AnswerLead(KIND_VERDICT, heading, "\n".join(lines), language)


def answer_lead(
    question: str,
    *,
    plan: AnswerPlan | None = None,
    decision: AnswerDecision | None = None,
    belief: AnswerBelief | None = None,
    language: str | None = None,
) -> AnswerLead | None:
    """The lead the question warrants, or ``None`` when no bridge fired.

    The kinds are tried strongest first -- procedure, then recommendation, then
    verdict -- and only the first that applies is built, so a question is never
    handed two competing leads. ``None`` means the answer keeps its retrieval
    opening: nothing is invented to fill a lead a question did not earn.
    """
    if not question.strip():
        return None
    lang = language or detect_language(question)
    if plan is not None and plan.steps:
        return _procedure_lead(plan, lang)
    if decision is not None and decision.options:
        return _recommendation_lead(decision, lang)
    if belief is not None:
        return _verdict_lead(belief, lang)
    return None


@dataclass(frozen=True)
class LeadJudgement:
    """One held-out expectation about the answer-lead bridge.

    ``kind`` selects what is checked: ``"kind"`` that a question earns the named
    lead (and the lead's heading names its source), ``"lead"`` that the lead is
    present in the *answer body* the assistant returns, and ``"none"`` that a
    question earns no lead at all -- the negative control that proves nothing is
    invented for a definitional question.
    """

    name: str
    kind: str
    question: str
    expected: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A lead judgement needs a name.")
        if self.kind not in ("kind", "lead", "none"):
            raise ValueError(f"Unknown lead judgement kind: {self.kind}")


#: The held-out lead catalogue -- hand-written with **negative controls**: a
#: procedural question must earn the procedure, a decisional one the
#: recommendation, an evaluative one the verdict; the lead must be in the answer
#: *body* (not only the trace); and a definitional or unrelated question must
#: earn **no** lead, so nothing is fabricated to fill one.
LEAD_JUDGEMENTS: tuple[LeadJudgement, ...] = (
    LeadJudgement(
        "a procedural question earns the procedure lead",
        "kind",
        question="how do I measure porosity and permeability on a core sample?",
        expected=KIND_PROCEDURE,
    ),
    LeadJudgement(
        "a decisional question earns the recommendation lead",
        "kind",
        question="what should I do to increase permeability?",
        expected=KIND_RECOMMENDATION,
    ),
    LeadJudgement(
        "an evaluative question earns the verdict lead",
        "kind",
        question="does acid stimulation increase permeability?",
        expected=KIND_VERDICT,
    ),
    LeadJudgement(
        "the procedure lead reaches the answer body",
        "lead",
        question="how do I measure porosity and permeability on a core sample?",
        expected=KIND_PROCEDURE,
    ),
    LeadJudgement(
        "the recommendation lead reaches the answer body",
        "lead",
        question="what should I do to increase permeability?",
        expected=KIND_RECOMMENDATION,
    ),
    LeadJudgement(
        "the verdict lead reaches the answer body",
        "lead",
        question="does acid stimulation increase permeability?",
        expected=KIND_VERDICT,
    ),
    # Negative controls: no reasoning is invented for a question that asks none.
    LeadJudgement("a definitional question earns no lead", "none", question="what is porosity?"),
    LeadJudgement(
        "an unrelated question earns no lead", "none", question="what is the capital of France?"
    ),
    LeadJudgement(
        "a blank question earns no lead", "none", question="   "
    ),
)


def lead_accuracy(
    judgements: tuple[LeadJudgement, ...] = LEAD_JUDGEMENTS,
    *,
    answer_for=None,
) -> float:
    """The share of held-out lead judgements the bridge gets right.

    A judgement counts only when the bridge does exactly what it must: hand the
    question the *right* lead kind, put that lead in the **answer body** (not only
    in the trace), or -- for a negative control -- hand it **no** lead at all. So
    a bridge that invents a lead for a definitional question, that gives the wrong
    kind, or that leaves the reasoning in the trace while the body keeps the
    retrieval template, fails the relevant judgement. Returns ``0.0`` for an
    empty catalogue.

    ``answer_for`` is the seam the evaluation injects (the assistant's own
    ``ask``), so the body check exercises the real answer path without this module
    importing the assistant. With no ``answer_for`` the body judgements are
    skipped and the share is computed over the rest, so the metric is still honest
    when only the pure bridge is available.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    from core.answer_belief import answer_belief
    from core.answer_decision import answer_decision
    from core.answer_plan import answer_plan

    correct = 0
    counted = 0
    for judgement in catalogue:
        if judgement.kind == "none":
            counted += 1
            if answer_lead(judgement.question, plan=None, decision=None, belief=None) is None:
                correct += 1
            continue
        plan = answer_plan(judgement.question)
        decision = answer_decision(judgement.question)
        belief = answer_belief(judgement.question)
        lead = answer_lead(judgement.question, plan=plan, decision=decision, belief=belief)
        if judgement.kind == "kind":
            counted += 1
            if lead is not None and lead.kind == judgement.expected:
                correct += 1
            continue
        # kind == "lead": the reasoning must be in the *answer body*, not the trace.
        if answer_for is None:
            continue
        counted += 1
        try:
            answer = answer_for(judgement.question)
        except Exception:
            continue
        answer_lead_obj = getattr(answer, "lead", None)
        text = getattr(answer, "text", "")
        if (
            answer_lead_obj is not None
            and answer_lead_obj.kind == judgement.expected
            and answer_lead_obj.heading in text
            and text.startswith(answer_lead_obj.heading)
        ):
            correct += 1
    return correct / counted if counted else 0.0


__all__ = [
    "KIND_PROCEDURE",
    "KIND_RECOMMENDATION",
    "KIND_VERDICT",
    "LEAD_JUDGEMENTS",
    "AnswerLead",
    "LeadJudgement",
    "answer_lead",
    "lead_accuracy",
]
