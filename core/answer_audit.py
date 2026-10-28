"""Measure the *connected* answer path, not the isolated reasoning layer.

This is Gap 9 of ``docs/REASONING_ASSESSMENT.md``. Phases 37-47 bridged every
reasoning layer to the answer -- causality, uncertainty, the groundedness gate,
the reasoning trace, planning, decision, belief, temporal, analogy, and the
knowledge carry -- and each bridge was given its own held-out metric. But those
metrics mostly exercise the bridge **in isolation**: they call
``answer_plan``/``answer_decision``/``answer_belief``/... directly, or hand the
assistant a single question, so a layer that stops reaching the *answer object*
the user actually receives can still pass. That is exactly the regression Gap 9
warns about: an evaluation that measures the isolated module, not the connected
path.

This module is the honest answer. It asks the **real assistant** its own
questions and checks what the answer *object* actually carries -- the plan, the
recommendation, the verdict, the comparison, the timeline, the lead, the trace,
the graph facts, the inferences, the walked connections, and the causal paths --
and, where a layer is supposed to reach the **body**, that it is in the text the
reader sees rather than only in the trailing trace.

Honesty is the whole point:

* Every expectation is a *held-out* question with the layers that **must** arrive
  on the answer, plus the layers that must **not** -- the negative controls that
  prove nothing is fabricated. A definitional question ("what is porosity?") must
  carry **no** reasoning layer; an unrelated one ("what is the capital of
  France?") must carry none either.
* The check runs the real ``GroundedAssistant.ask`` when one can be built here;
  with none available the metric is computed over nothing and reported honestly
  as ``0.0`` rather than faked, so a caller can tell "the path is broken" from
  "the path was never exercised".
* A layer counts only when it is on the answer the path returned -- an isolated
  bridge that still produces its object but whose object never reaches the answer
  fails the expectation, which is the whole point.

Phase 49 adds the **Arabic-parallel** catalogue (``AUDIT_EXPECTATIONS_AR``): the
project is Arabic-first, so the connected path must deliver in Arabic too. Each
English expectation has an Arabic twin, and a regression that breaks only the
Arabic path fails there while the English catalogue stays green.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.assistant import AssistantAnswer

#: The layer names an expectation can require or forbid. They name what the
#: *answer object* carries, not what the isolated bridge can produce, so a check
#: on one of them is a check on the connected path.
LAYER_EVIDENCE = "evidence"
LAYER_PLAN = "plan"
LAYER_DECISION = "decision"
LAYER_BELIEF = "belief"
LAYER_ANALOGY = "analogy"
LAYER_TEMPORAL = "temporal"
LAYER_LEAD = "lead"
LAYER_REASONING = "reasoning"
LAYER_CONFIDENCE = "confidence"
LAYER_FACTS = "facts"
LAYER_INFERRED = "inferred"
LAYER_CONNECTIONS = "connections"
LAYER_CAUSAL = "causal"

#: The layers that are *reasoning* -- a derivation the question earned. A negative
#: control forbids all of them at once, so a definitional question cannot quietly
#: gain one.
REASONING_LAYERS: frozenset[str] = frozenset(
    {
        LAYER_PLAN,
        LAYER_DECISION,
        LAYER_BELIEF,
        LAYER_ANALOGY,
        LAYER_TEMPORAL,
        LAYER_LEAD,
    }
)

#: Every layer the audit knows how to read off an answer.
ALL_LAYERS: frozenset[str] = frozenset(
    {
        LAYER_EVIDENCE,
        LAYER_PLAN,
        LAYER_DECISION,
        LAYER_BELIEF,
        LAYER_ANALOGY,
        LAYER_TEMPORAL,
        LAYER_LEAD,
        LAYER_REASONING,
        LAYER_CONFIDENCE,
        LAYER_FACTS,
        LAYER_INFERRED,
        LAYER_CONNECTIONS,
        LAYER_CAUSAL,
    }
)


def answer_layers(answer: AssistantAnswer) -> frozenset[str]:
    """The reasoning layers one answer *object* actually carries.

    Read straight off the answer the path returned -- the optional objects
    (plan/decision/belief/analogy/temporal/lead/reasoning/confidence) and the
    non-empty tuples (facts/inferred/connections/causal) -- so a layer that the
    bridge produced but the answer dropped is simply absent. Never guesses: a
    ``None`` or an empty tuple is not a layer.
    """
    layers: set[str] = set()
    if getattr(answer, "passages", ()):
        layers.add(LAYER_EVIDENCE)
    if getattr(answer, "plan", None) is not None:
        layers.add(LAYER_PLAN)
    if getattr(answer, "decision", None) is not None:
        layers.add(LAYER_DECISION)
    if getattr(answer, "belief", None) is not None:
        layers.add(LAYER_BELIEF)
    if getattr(answer, "analogy", None) is not None:
        layers.add(LAYER_ANALOGY)
    if getattr(answer, "temporal", None) is not None:
        layers.add(LAYER_TEMPORAL)
    if getattr(answer, "lead", None) is not None:
        layers.add(LAYER_LEAD)
    if getattr(answer, "reasoning", None) is not None:
        layers.add(LAYER_REASONING)
    if getattr(answer, "confidence_report", None) is not None:
        layers.add(LAYER_CONFIDENCE)
    if getattr(answer, "relation_facts", ()):
        layers.add(LAYER_FACTS)
    if getattr(answer, "inferred", ()):
        layers.add(LAYER_INFERRED)
    if getattr(answer, "connections", ()):
        layers.add(LAYER_CONNECTIONS)
    if getattr(answer, "causal_paths", ()):
        layers.add(LAYER_CAUSAL)
    return frozenset(layers)


@dataclass(frozen=True)
class AuditExpectation:
    """One held-out expectation about what the *answer path* must deliver.

    ``required`` names the layers the answer object must carry; ``forbidden``
    names the layers it must not -- the negative controls. ``body`` asks that the
    answer's **lead** be in the text the reader sees (its heading opening the
    body), not only on the object. ``timeline`` asks the check to hand the
    assistant a short probe conversation, because a timeline needs at least two
    turns to order.
    """

    name: str
    question: str
    required: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    body: bool = False
    timeline: bool = False
    readable: bool = False
    """Ask that no snake_case concept identifier reach the answer body.

    Set on an Arabic expectation whose answer names concepts (the recommendation,
    the verdict, the comparison): the graph's own labels are snake_case, and
    before Phase 51 they reached the reader as code tokens (``shut_in``,
    ``injection_pattern``). A ``readable`` expectation fails if any survives, so a
    regression that stops rendering them is caught on the connected path, not only
    in the unit test.
    """

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("An audit expectation needs a name.")
        if not self.question.strip():
            raise ValueError("An audit expectation needs a question.")
        unknown = (set(self.required) | set(self.forbidden)) - ALL_LAYERS
        if unknown:
            raise ValueError(f"Unknown layer(s): {', '.join(sorted(unknown))}")
        if set(self.required) & set(self.forbidden):
            raise ValueError("A layer cannot be both required and forbidden.")


#: The held-out audit catalogue. Each positive expectation names the layer its
#: question earns *on the answer the path returns*; each negative control forbids
#: every reasoning layer, so nothing is fabricated to fill one.
AUDIT_EXPECTATIONS: tuple[AuditExpectation, ...] = (
    AuditExpectation(
        "a procedural question reaches the plan and the procedure lead",
        "how do I measure porosity and permeability on a core sample?",
        required=(LAYER_PLAN, LAYER_LEAD, LAYER_REASONING),
        body=True,
    ),
    AuditExpectation(
        "a decisional question reaches the recommendation",
        "what should I do to increase permeability?",
        required=(LAYER_DECISION, LAYER_LEAD, LAYER_REASONING),
        body=True,
    ),
    AuditExpectation(
        "an evaluative question reaches the belief verdict",
        "does acid stimulation increase permeability?",
        required=(LAYER_BELIEF, LAYER_LEAD, LAYER_REASONING),
        body=True,
    ),
    AuditExpectation(
        "an analogical question reaches the comparison",
        "is porosity like permeability?",
        required=(LAYER_ANALOGY, LAYER_LEAD, LAYER_REASONING),
        body=True,
    ),
    AuditExpectation(
        "a temporal question reaches the timeline",
        "what changed over the session?",
        required=(LAYER_TEMPORAL, LAYER_LEAD, LAYER_REASONING),
        body=True,
        timeline=True,
    ),
    AuditExpectation(
        "a connected question reaches the walked connections",
        "how are porosity and permeability connected?",
        required=(LAYER_CONNECTIONS, LAYER_REASONING),
    ),
    AuditExpectation(
        "a question about the graph's edges reaches the stored facts",
        "what is porosity?",
        required=(LAYER_FACTS, LAYER_REASONING),
    ),
    # Negative controls: no reasoning is invented for a question that asks none.
    AuditExpectation(
        "a definitional question carries no reasoning layer",
        "what is permeability?",
        forbidden=tuple(sorted(REASONING_LAYERS)),
    ),
    AuditExpectation(
        "an unrelated question carries no reasoning layer",
        "what is the capital of France?",
        forbidden=tuple(sorted(REASONING_LAYERS)),
    ),
    AuditExpectation(
        "a blank-adjacent question carries no reasoning layer",
        "tell me something",
        forbidden=tuple(sorted(REASONING_LAYERS)),
    ),
)

#: The **Arabic-parallel** catalogue (Phase 49). The project is Arabic-first, so
#: the connected path must deliver in Arabic too -- and until Phase 49 it did not:
#: only 39 of 495 concepts carried an Arabic alias, so an Arabic question reached
#: almost none of the graph. Each English expectation above has an Arabic twin, so
#: a regression that breaks only the Arabic path fails here while the English
#: catalogue stays green.
AUDIT_EXPECTATIONS_AR: tuple[AuditExpectation, ...] = (
    AuditExpectation(
        "سؤال إجرائي يصل إلى الخطة ورأس الإجراء",
        "كيف أقيس المسامية والنفاذية على عينة لب؟",
        required=(LAYER_PLAN, LAYER_LEAD, LAYER_REASONING),
        body=True,
    ),
    AuditExpectation(
        "سؤال قراري يصل إلى التوصية",
        "ماذا أفعل لزيادة النفاذية؟",
        required=(LAYER_DECISION, LAYER_LEAD, LAYER_REASONING),
        body=True,
    ),
    AuditExpectation(
        "سؤال تقييمي يصل إلى حكم الاعتقاد",
        "هل التحميض يزيد النفاذية؟",
        required=(LAYER_BELIEF, LAYER_LEAD, LAYER_REASONING),
        body=True,
    ),
    AuditExpectation(
        "سؤال تناظري يصل إلى المقارنة",
        "هل المسامية مثل النفاذية؟",
        required=(LAYER_ANALOGY, LAYER_LEAD, LAYER_REASONING),
        body=True,
    ),
    AuditExpectation(
        "سؤال زمني يصل إلى الخط الزمني",
        "ما الذي تغيّر خلال الجلسة؟",
        required=(LAYER_TEMPORAL, LAYER_LEAD, LAYER_REASONING),
        body=True,
        timeline=True,
    ),
    AuditExpectation(
        "سؤال عن الاتصال يصل إلى الروابط المسلوكة",
        "كيف ترتبط المسامية والنفاذية؟",
        required=(LAYER_CONNECTIONS, LAYER_REASONING),
    ),
    AuditExpectation(
        "سؤال عن حواف الرسم يصل إلى العلاقات المخزَّنة",
        "ما هي المسامية؟",
        required=(LAYER_FACTS, LAYER_REASONING),
    ),
    # Negative controls: an Arabic question that asks no reasoning must invent none.
    AuditExpectation(
        "سؤال تعريفي لا يحمل أي طبقة استدلالية",
        "ما هي النفاذية؟",
        forbidden=tuple(sorted(REASONING_LAYERS)),
    ),
    AuditExpectation(
        "سؤال غير ذي صلة لا يحمل أي طبقة استدلالية",
        "ما هي عاصمة فرنسا؟",
        forbidden=tuple(sorted(REASONING_LAYERS)),
    ),
    AuditExpectation(
        "سؤال مبهم لا يحمل أي طبقة استدلالية",
        "أخبرني بشيء",
        forbidden=tuple(sorted(REASONING_LAYERS)),
    ),
)

#: The Phase 51 catalogue: an Arabic answer that names concepts must name them
#: **readably**. Each question reaches a different layer that prints graph labels
#: (the recommendation, the verdict, the comparison), and ``readable`` asks that no
#: snake_case identifier survive in the body. Held-out from the unit tests' battery
#: -- a couple of questions differ -- so the metric is not a copy of the test.
ARABIC_READABLE_EXPECTATIONS: tuple[AuditExpectation, ...] = (
    AuditExpectation(
        "توصية عربية بلا معرّفات إنجليزية",
        "ماذا أفعل لزيادة الإنتاج؟",
        required=(LAYER_DECISION,),
        readable=True,
    ),
    AuditExpectation(
        "حكم عربي بلا معرّفات إنجليزية",
        "هل التحميض يزيد النفاذية؟",
        required=(LAYER_BELIEF,),
        readable=True,
    ),
    AuditExpectation(
        "مقارنة عربية بلا معرّفات إنجليزية",
        "هل التكسير الهيدروليكي مثل التحميض؟",
        required=(LAYER_ANALOGY,),
        readable=True,
    ),
    AuditExpectation(
        "توصية عربية عند التحكم في الآبار بلا معرّفات",
        "ماذا أفعل للتحكم في الاندفاع؟",
        required=(LAYER_DECISION,),
        readable=True,
    ),
)


def _probe_conversation():
    """A minimal four-turn conversation, so a temporal question can earn a timeline.

    The temporal bridge reads a *conversation*; a held-out question has none, so
    this supplies the shortest honest timeline. It is used only for the
    expectation that asks for one and never fabricates content: the turns name
    the question's own concepts and short replies to them.
    """
    from core.conversation import Conversation

    conversation = Conversation(id="audit-probe", title="audit probe")
    conversation.add_user("what is porosity?")
    conversation.add_assistant("porosity is the void fraction of the rock.")
    conversation.add_user("and permeability?")
    conversation.add_assistant("permeability is the rock's capacity to flow.")
    return conversation


def _lead_in_body(answer: AssistantAnswer) -> bool:
    """True when the answer's lead opens the *body*, not only the trace.

    The lead is meant to be the answer's own opening (``core.answer_lead``), so
    its heading must be present in the text and the text must start with it. An
    answer that carries the lead object but keeps the retrieval template fails.
    """
    lead = getattr(answer, "lead", None)
    if lead is None:
        return False
    text = getattr(answer, "text", "")
    return bool(lead.heading) and text.startswith(lead.heading)


def _no_identifier_in_body(answer: AssistantAnswer) -> bool:
    """Whether the answer body carries no snake_case concept identifier.

    The graph's labels are snake_case, and Phase 51 renders them readably for an
    Arabic reader (:func:`core.arabic_prose.render_arabic_text`); a body that still
    carries one is the leak the Phase 51 measurement found.
    """
    from core.arabic_prose import _IDENTIFIER_RE

    return not _IDENTIFIER_RE.search(answer.text or "")


def audit_answer(
    expectation: AuditExpectation,
    answer_for,
) -> bool:
    """Whether one expectation holds for the answer the real path returns.

    Runs ``answer_for`` (the assistant's own ``ask``), reads the layers off the
    answer object, and checks the required/forbidden sets and, when asked, that
    the lead is in the body and that no identifier leaks into it. A missing answer
    or a raised error is a failure, not a skip: the point of the audit is that the
    connected path must deliver.
    """
    kwargs: dict[str, object] = {}
    if expectation.timeline:
        kwargs["conversation"] = _probe_conversation()
    try:
        answer = answer_for(expectation.question, **kwargs)
    except Exception:
        return False
    if answer is None:
        return False
    layers = answer_layers(answer)
    if not set(expectation.required).issubset(layers):
        return False
    if set(expectation.forbidden) & layers:
        return False
    if expectation.body and not _lead_in_body(answer):
        return False
    if expectation.readable and not _no_identifier_in_body(answer):
        return False
    return True


def answer_path_accuracy(
    expectations: tuple[AuditExpectation, ...] = AUDIT_EXPECTATIONS,
    *,
    answer_for=None,
) -> float:
    """The share of held-out audit expectations the *connected path* satisfies.

    Every expectation is run against the real assistant's own ``ask`` when one is
    available. With no ``answer_for`` the path is not exercised, so the honest
    answer is ``0.0`` -- the metric reports that the connected path was not
    measured rather than claiming a pass it never earned. Returns ``0.0`` for an
    empty catalogue.
    """
    catalogue = tuple(expectations)
    if not catalogue or answer_for is None:
        return 0.0
    correct = sum(1 for expectation in catalogue if audit_answer(expectation, answer_for))
    return correct / len(catalogue)


__all__ = [
    "ALL_LAYERS",
    "ARABIC_READABLE_EXPECTATIONS",
    "AUDIT_EXPECTATIONS",
    "AUDIT_EXPECTATIONS_AR",
    "LAYER_ANALOGY",
    "LAYER_BELIEF",
    "LAYER_CAUSAL",
    "LAYER_CONFIDENCE",
    "LAYER_CONNECTIONS",
    "LAYER_DECISION",
    "LAYER_EVIDENCE",
    "LAYER_FACTS",
    "LAYER_INFERRED",
    "LAYER_LEAD",
    "LAYER_PLAN",
    "LAYER_REASONING",
    "LAYER_TEMPORAL",
    "REASONING_LAYERS",
    "AuditExpectation",
    "answer_layers",
    "answer_path_accuracy",
    "audit_answer",
]
