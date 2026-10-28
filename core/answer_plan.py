"""Surface the *procedure* the corpus states for a question that asks "how".

The project carries a whole planning layer -- ``Planner``, ``Action``, ``Goal``,
``PlanStep``, ``DecisionEngine`` -- but it was reachable only from the
``CognitiveController`` and the evaluation scenarios, never from an answer. So a
question like *"how do I measure porosity and permeability on a core sample?"*
got a definition back, never the ordered procedure the corpus already states
(``CorpusEntry.procedures``, 48 entries of them). That is Gap 7 of
``docs/REASONING_ASSESSMENT.md``.

This module reads the corpus's own procedure for the entry a question is about
and runs it through the project's real ``Planner`` -- ``order_steps`` -- so the
plan that reaches the reader is the one the planning engine actually produced,
with its own dependency checks (a circular or dangling dependency is refused, not
silently reordered). It is **read-only** and **honest by construction**: it only
builds a plan when the question actually *asks how* (a definitional "what is
porosity?" gets none), it never invents a step the corpus did not state, it
attributes the plan to the corpus entry it came from, and it returns ``None``
rather than a guess when no entry matches.

The plan is a *reported* artifact, not an action: nothing here executes anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.goal import Goal, PlanStep
from core.planner import Planner
from core.tokenize import content_terms

if TYPE_CHECKING:
    from core.petroleum_corpus import CorpusEntry

#: The phrases that make a question *procedural* -- a request for how to do
#: something, in either language. A plan is only produced for such a question, so
#: a definitional question never drags a procedure into its answer.
PROCEDURAL_CUES: tuple[str, ...] = (
    "how do i",
    "how do you",
    "how can i",
    "how should i",
    "how to",
    "how would i",
    "steps to",
    "step by step",
    "what are the steps",
    "procedure for",
    "the procedure",
    "the process for",
    "process to",
    "way to",
    "guide to",
    "كيف",
    "خطوات",
    "الخطوات",
    "ما هي الخطوات",
    "طريقة",
    "إجراءات",
    "اجراءات",
)


@dataclass(frozen=True)
class AnswerPlan:
    """The ordered procedure the corpus states for a question that asks how."""

    topic: str
    """The corpus entry's title, so the plan is attributed to what it came from."""

    source: str
    """The corpus entry's key, so a reader can find the entry the plan came from."""

    steps: tuple[str, ...]
    """The ordered steps, exactly the corpus's own wording, in dependency order."""

    def describe(self) -> str:
        """The plan as a heading followed by its numbered steps."""
        if not self.steps:
            return f"The procedure for {self.topic} ({self.source}): (nothing)"
        body = "\n".join(f"  {index}. {step}" for index, step in enumerate(self.steps, start=1))
        return f"The procedure for {self.topic} ({self.source}):\n{body}"


def is_procedural(question: str) -> bool:
    """Whether a question asks *how* to do something, so a plan is warranted.

    Whole-phrase and case-insensitive: the cue must appear in the question, so a
    definitional question ("what is porosity?") is never read as procedural.
    """
    lowered = question.casefold()
    return any(cue in lowered for cue in PROCEDURAL_CUES)


def _entry_terms(entry: CorpusEntry) -> set[str]:
    """Every content term an entry is about: its concepts, endpoints, and steps.

    The entry's own procedure text is included because a question that asks how to
    do something usually names the very words the procedure uses ("measure
    porosity ... on a core sample"), so the entry whose procedure matches the
    question is preferred over one that merely shares a concept.
    """
    terms: set[str] = set()
    for label, aliases in entry.concepts:
        for phrase in (label, *aliases):
            terms |= set(content_terms(phrase.replace("_", " ")))
    for source, target, _relation, _confidence in entry.facts:
        for label in (source, target):
            terms |= set(content_terms(label.replace("_", " ")))
    for step in entry.procedures:
        terms |= set(content_terms(step))
    return terms


def match_entry(question: str, *, entries: tuple[CorpusEntry, ...] | None = None) -> CorpusEntry | None:
    """The corpus entry a question is most about, or ``None`` when none matches.

    Scored by the shared content terms of the question and each entry (its
    concepts, its relation endpoints, and its own procedure text). Ties break on
    the entry key, so the same question always yields the same entry. An entry
    that states no procedure is skipped -- there is no plan to report for it --
    and a question sharing no term with any entry returns ``None``, never a guess.
    """
    from core.petroleum_corpus import entries as corpus_entries

    catalogue = entries if entries is not None else corpus_entries()
    query_terms = set(content_terms(question))
    if not query_terms:
        return None
    best: CorpusEntry | None = None
    best_score = 0
    for entry in catalogue:
        if not entry.procedures:
            continue
        score = len(_entry_terms(entry) & query_terms)
        if score > best_score or (
            score == best_score and score > 0 and best is not None and entry.key < best.key
        ):
            best = entry
            best_score = score
    return best if best_score > 0 else None


def order_procedure(
    steps: tuple[tuple[str, tuple[str, ...]], ...] | tuple[str, ...],
    *,
    goal_description: str,
) -> tuple[str, ...]:
    """Order a procedure through the project's real planner.

    ``steps`` is either the corpus's plain tuple of step texts (ordered as stated,
    each depending on the one before) or explicit ``(description, depends_on)``
    pairs. The plan is ordered by ``Planner.order_steps`` -- the planning engine's
    own topological sort -- so a circular or dangling dependency is **refused**
    with ``ValueError`` rather than silently reordered. Returns the ordered step
    texts. Nothing is executed.
    """
    plan_steps = _as_plan_steps(steps)
    goal = Goal(description=goal_description or "procedure")
    ordered = Planner().order_steps(goal, plan_steps)
    return tuple(step.description for step in ordered)


def _as_plan_steps(
    steps: tuple[tuple[str, tuple[str, ...]], ...] | tuple[str, ...],
) -> list[PlanStep]:
    """Normalise plain step texts or explicit pairs into ``PlanStep`` objects."""
    plan_steps: list[PlanStep] = []
    previous = ""
    for item in steps:
        if isinstance(item, str):
            plan_steps.append(PlanStep(item, (previous,) if previous else ()))
            previous = item
        else:
            description, depends_on = item
            plan_steps.append(PlanStep(description, tuple(depends_on)))
    return plan_steps


def answer_plan(
    question: str, *, entries: tuple[CorpusEntry, ...] | None = None
) -> AnswerPlan | None:
    """The plan for a question, or ``None`` when the question asks no procedure.

    A plan is produced only when the question is *procedural* (``is_procedural``)
    **and** an entry with a stated procedure matches it. Otherwise ``None`` -- a
    definitional or unrelated question gets no plan rather than a guess.
    """
    if not is_procedural(question):
        return None
    entry = match_entry(question, entries=entries)
    if entry is None or not entry.procedures:
        return None
    steps = order_procedure(entry.procedures, goal_description=question)
    return AnswerPlan(topic=entry.title, source=entry.key, steps=steps)


@dataclass(frozen=True)
class PlanJudgement:
    """One held-out expectation about the plan bridge.

    ``kind`` selects what is checked: ``"order"`` that a step list orders to
    ``expected``, ``"reject"`` that a bad dependency is refused, ``"no-plan"``
    that a question yields no plan, and ``"corpus"`` that a procedural question
    yields the matching entry's own steps.
    """

    name: str
    kind: str
    question: str = ""
    steps: tuple[tuple[str, tuple[str, ...]], ...] = ()
    expected: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A plan judgement needs a name.")
        if self.kind not in ("order", "reject", "no-plan", "corpus"):
            raise ValueError(f"Unknown plan judgement kind: {self.kind}")


#: The held-out plan catalogue -- hand-written with **negative controls**: a
#: definitional or unrelated question must yield no plan, a circular or dangling
#: dependency must be refused rather than reordered, and a question given out of
#: order must come back in dependency order.
PLAN_JUDGEMENTS: tuple[PlanJudgement, ...] = (
    PlanJudgement(
        "a linear procedure keeps its order",
        "order",
        steps=(("A", ()), ("B", ("A",)), ("C", ("B",))),
        expected=("A", "B", "C"),
    ),
    PlanJudgement(
        "steps given out of order are ordered",
        "order",
        steps=(("C", ("B",)), ("A", ()), ("B", ("A",))),
        expected=("A", "B", "C"),
    ),
    PlanJudgement(
        "a circular plan is refused",
        "reject",
        steps=(("A", ("B",)), ("B", ("A",))),
    ),
    PlanJudgement(
        "a dependency on an unknown step is refused",
        "reject",
        steps=(("A", ("ghost",)),),
    ),
    PlanJudgement("a definitional question gets no plan", "no-plan", question="what is porosity?"),
    PlanJudgement(
        "an unrelated question gets no plan", "no-plan", question="what is the capital of France?"
    ),
    PlanJudgement(
        "a procedural question gets the entry's own steps",
        "corpus",
        question="how do I identify the drive mechanism?",
        expected=(
            "Identify the drive from pressure and production trends.",
            "Estimate the recovery factor for that drive.",
        ),
    ),
)


def plan_accuracy(judgements: tuple[PlanJudgement, ...] = PLAN_JUDGEMENTS) -> float:
    """The share of held-out plan judgements the bridge gets right.

    A judgement counts only when the bridge does exactly what it must: order a
    plan correctly, **refuse** a circular or dangling dependency, produce no plan
    for a question that asks none, or return the matching entry's own steps. So a
    bridge that invents a step, reorders a cycle instead of refusing it, or hands
    a procedure to a definitional question fails the relevant judgement. Returns
    ``0.0`` for an empty catalogue.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    correct = 0
    for judgement in catalogue:
        try:
            if judgement.kind == "order":
                got = order_procedure(judgement.steps, goal_description="judgement")
                ok = got == judgement.expected
            elif judgement.kind == "reject":
                order_procedure(judgement.steps, goal_description="judgement")
                ok = False  # a bad dependency should have raised
            elif judgement.kind == "no-plan":
                ok = answer_plan(judgement.question) is None
            else:  # corpus
                plan = answer_plan(judgement.question)
                ok = plan is not None and plan.steps == judgement.expected
        except ValueError:
            ok = judgement.kind == "reject"
        if ok:
            correct += 1
    return correct / len(catalogue)


__all__ = [
    "PLAN_JUDGEMENTS",
    "PROCEDURAL_CUES",
    "AnswerPlan",
    "PlanJudgement",
    "answer_plan",
    "is_procedural",
    "match_entry",
    "order_procedure",
    "plan_accuracy",
]
