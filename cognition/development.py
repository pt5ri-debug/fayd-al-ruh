"""The held-out gate for the mind's development: metacognition and consolidation.

The three faculties this phase adds -- the self-model, the consolidator, and the
adaptive curriculum -- are the mind's ability to *improve itself*, so the honest
question is not "does the code run?" but "does it actually behave as claimed?".
This module is a catalogue of hand-written judgements, run against the real
objects, each with **negative controls** so an over-eager faculty fails as surely
as a broken one:

* the self-model must favour the strategy that pays off and stay neutral on an
  untried one, and must re-explore a strategy that has gone cold;
* the stagnation detector must fire on a genuinely flat, repeating history and
  **not** fire when novelty is rising or something was promoted -- the negative
  controls that stop it from crying wolf;
* the consolidator must refuse to state a rule from one anecdote and must state
  one from real support -- the negative control that stops it from inventing a
  law;
* the spaced-review box must advance on recall and reset on a miss;
* the escalation must stay inside its ceilings;
* the skill proficiency must be measured, not guessed.

``development_accuracy`` is the share of judgements the real objects satisfy, so
a regression that makes the mind over-confident, over-generalising, or unbounded
lowers the metric.
"""

from __future__ import annotations

from dataclasses import dataclass

from cognition.consolidation import Consolidator, ReviewSchedule, SkillRegistry
from cognition.metacognition import (
    COLD_AFTER_CYCLES,
    EscalationPolicy,
    SelfModel,
    StagnationDetector,
    StagnationReading,
)
from cognition.records import Idea, IdeaKind, ThoughtCycle


def _idea(*, confidence: float = 0.5, novelty: float = 0.5, kind: IdeaKind = IdeaKind.IDEA) -> Idea:
    return Idea(
        text=f"a thought with confidence {confidence} and novelty {novelty}",
        kind=kind,
        confidence=confidence,
        novelty=novelty,
    )


def _cycle(ideas: tuple[str, ...], gaps: tuple[str, ...]) -> ThoughtCycle:
    return ThoughtCycle(ideas=ideas, gaps=gaps)


@dataclass(frozen=True)
class DevelopmentJudgement:
    """One hand-written expectation about how a faculty must behave."""

    name: str
    description: str
    check: str

    def describe(self) -> str:
        return f"{self.name}: {self.description}"


def _check_self_model_favours_what_pays_off() -> bool:
    model = SelfModel()
    model.advance_cycle()
    for _ in range(4):
        model.observe("good", _idea(confidence=0.9, novelty=0.9, kind=IdeaKind.HYPOTHESIS), kept=True)
        model.observe("poor", _idea(confidence=0.1, novelty=0.1), kept=False)
    return model.bias("good") > model.bias("poor")


def _check_self_model_is_neutral_on_an_untried_strategy() -> bool:
    return SelfModel().bias("never.seen") == 0.5


def _check_self_model_re_explores_a_cold_strategy() -> bool:
    model = SelfModel()
    model.advance_cycle()
    model.observe("cold", _idea(confidence=0.1, novelty=0.1), kept=False)
    for _ in range(COLD_AFTER_CYCLES):
        model.advance_cycle()
    return model.bias("cold") == 0.5


def _check_stagnation_fires_on_a_flat_repeating_history() -> bool:
    detector = StagnationDetector(window=4, novelty_floor=0.5)
    ideas = {"i1": _idea(novelty=0.1), "i2": _idea(novelty=0.1)}
    cycles = tuple(_cycle(("i1", "i2"), ("porosity", "porosity")) for _ in range(4))
    return detector.read(cycles, ideas).stagnating


def _check_stagnation_does_not_fire_when_novelty_rises() -> bool:
    detector = StagnationDetector(window=4, novelty_floor=0.5)
    # Novelty climbs across the window, so the mind is still learning.
    ideas = {f"i{n}": _idea(novelty=0.1 + 0.2 * n) for n in range(4)}
    cycles = tuple(_cycle((f"i{n}",), ("porosity", "porosity")) for n in range(4))
    return not detector.read(cycles, ideas).stagnating


def _check_stagnation_does_not_fire_when_something_was_promoted() -> bool:
    detector = StagnationDetector(window=4, novelty_floor=0.5)
    ideas = {
        "i1": _idea(novelty=0.1),
        "i2": _idea(novelty=0.1, kind=IdeaKind.HYPOTHESIS),
    }
    cycles = tuple(_cycle(("i1", "i2"), ("porosity", "porosity")) for _ in range(4))
    return not detector.read(cycles, ideas).stagnating


def _check_escalation_is_bounded() -> bool:
    policy = EscalationPolicy(max_widen_concepts=2, max_widen_ideas=2, max_novelty_floor=0.6)
    reading = StagnationReading(True, -0.2, 0, 4, "flat")
    plan = None
    for _ in range(8):
        plan = policy.plan(reading, SelfModel())
    assert plan is not None
    return (
        plan.widen_concepts <= 2
        and plan.widen_ideas <= 2
        and plan.novelty_floor <= 0.6
    )


class _FakeLearning:
    def __init__(self) -> None:
        self._experiences: list = []
        self._facts: list[tuple[str, str]] = []

    class _Store:
        def __init__(self, items: list) -> None:
            self._items = items

        def all(self):
            return tuple(self._items)

    @property
    def experiences(self):
        return _FakeLearning._Store(self._experiences)

    @property
    def facts(self):
        return _FakeLearning._Store(self._facts)

    def record(self, expectation: str, outcome: str, *, success: bool) -> None:
        from cognition.records import Experience

        self._experiences.append(Experience(expectation=expectation, outcome=outcome, success=success))

    def learn_fact(self, fact_id, statement, **kwargs):
        self._facts.append((fact_id, statement))
        return None


def _check_a_rule_needs_real_support() -> bool:
    learning = _FakeLearning()
    learning.record("acid on tight rock", "permeability improves", success=True)
    return Consolidator(learning).consolidate().patterns == ()


def _check_repeated_support_yields_a_rule() -> bool:
    learning = _FakeLearning()
    for _ in range(3):
        learning.record("acid on tight rock", "permeability improves", success=True)
    report = Consolidator(learning).consolidate()
    return len(report.patterns) == 1 and report.patterns[0].support == 3


def _check_a_review_box_advances_and_resets() -> bool:
    schedule = ReviewSchedule()
    schedule.add("PAT-1", "when a then b")
    schedule.review("PAT-1", remembered=True)
    advanced = schedule._items["PAT-1"].box == 1
    schedule.review("PAT-1", remembered=False)
    reset = schedule._items["PAT-1"].box == 0
    return advanced and reset


def _check_skill_proficiency_is_measured() -> bool:
    registry = SkillRegistry()
    registry.practise("log a well", success=True)
    registry.practise("log a well", success=False)
    return registry.skills()[0].proficiency == 0.5


#: The held-out catalogue: each judgement is run against the real objects.
DEVELOPMENT_JUDGEMENTS: tuple[DevelopmentJudgement, ...] = (
    DevelopmentJudgement(
        "self_model_favours_what_pays_off",
        "a strategy whose ideas survive criticism is biased above one whose do not",
        "self_model_favours_what_pays_off",
    ),
    DevelopmentJudgement(
        "self_model_neutral_on_untried",
        "an untried strategy gets the neutral prior, never a lock-out",
        "self_model_neutral_on_untried",
    ),
    DevelopmentJudgement(
        "self_model_re_explores_cold",
        "a strategy that has gone cold is given the neutral prior again",
        "self_model_re_explores_cold",
    ),
    DevelopmentJudgement(
        "stagnation_fires_on_flat_history",
        "a flat, repeating history is reported as stagnation",
        "stagnation_fires_on_flat_history",
    ),
    DevelopmentJudgement(
        "stagnation_quiet_when_novelty_rises",
        "rising novelty is NOT stagnation (negative control)",
        "stagnation_quiet_when_novelty_rises",
    ),
    DevelopmentJudgement(
        "stagnation_quiet_after_a_promotion",
        "a promotion means progress, so it is NOT stagnation (negative control)",
        "stagnation_quiet_after_a_promotion",
    ),
    DevelopmentJudgement(
        "escalation_is_bounded",
        "escalation widens the search but never past its ceilings",
        "escalation_is_bounded",
    ),
    DevelopmentJudgement(
        "a_rule_needs_real_support",
        "one anecdote is never stated as a rule (negative control)",
        "a_rule_needs_real_support",
    ),
    DevelopmentJudgement(
        "repeated_support_yields_a_rule",
        "repeated support is consolidated into a rule with its count",
        "repeated_support_yields_a_rule",
    ),
    DevelopmentJudgement(
        "review_box_advances_and_resets",
        "the spaced-review box advances on recall and resets on a miss",
        "review_box_advances_and_resets",
    ),
    DevelopmentJudgement(
        "skill_proficiency_is_measured",
        "a skill's proficiency is its measured success rate, not a guess",
        "skill_proficiency_is_measured",
    ),
)

_CHECKS = {
    "self_model_favours_what_pays_off": _check_self_model_favours_what_pays_off,
    "self_model_neutral_on_untried": _check_self_model_is_neutral_on_an_untried_strategy,
    "self_model_re_explores_cold": _check_self_model_re_explores_a_cold_strategy,
    "stagnation_fires_on_flat_history": _check_stagnation_fires_on_a_flat_repeating_history,
    "stagnation_quiet_when_novelty_rises": _check_stagnation_does_not_fire_when_novelty_rises,
    "stagnation_quiet_after_a_promotion": _check_stagnation_does_not_fire_when_something_was_promoted,
    "escalation_is_bounded": _check_escalation_is_bounded,
    "a_rule_needs_real_support": _check_a_rule_needs_real_support,
    "repeated_support_yields_a_rule": _check_repeated_support_yields_a_rule,
    "review_box_advances_and_resets": _check_a_review_box_advances_and_resets,
    "skill_proficiency_is_measured": _check_skill_proficiency_is_measured,
}


def development_accuracy(judgements: tuple[DevelopmentJudgement, ...] = DEVELOPMENT_JUDGEMENTS) -> float:
    """The share of judgements the real faculties satisfy."""
    if not judgements:
        return 0.0
    passed = 0
    for judgement in judgements:
        check = _CHECKS.get(judgement.check)
        if check is None:
            continue
        try:
            if check():
                passed += 1
        except Exception:
            # A faculty that raises fails its judgement rather than the gate.
            continue
    return round(passed / len(judgements), 6)


def describe_development() -> str:
    """A readable report of every judgement and whether it held."""
    lines = ["Cognitive development gate:"]
    for judgement in DEVELOPMENT_JUDGEMENTS:
        check = _CHECKS.get(judgement.check)
        try:
            ok = bool(check()) if check is not None else False
        except Exception:
            ok = False
        lines.append(f"  [{'ok' if ok else 'FAIL'}] {judgement.describe()}")
    return "\n".join(lines)


__all__ = [
    "DEVELOPMENT_JUDGEMENTS",
    "DevelopmentJudgement",
    "describe_development",
    "development_accuracy",
]
