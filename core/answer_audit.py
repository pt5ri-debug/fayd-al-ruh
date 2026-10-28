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


def audit_answer(
    expectation: AuditExpectation,
    answer_for,
) -> bool:
    """Whether one expectation holds for the answer the real path returns.

    Runs ``answer_for`` (the assistant's own ``ask``), reads the layers off the
    answer object, and checks the required/forbidden sets and, when asked, that
    the lead is in the body. A missing answer or a raised error is a failure, not
    a skip: the point of the audit is that the connected path must deliver.
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
    "AUDIT_EXPECTATIONS",
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
