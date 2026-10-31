"""The maximal development of the mind: metacognition, consolidation, curriculum.

Three faculties are added to the autonomous cognitive system, and each is tested
here against the real machinery it is built on -- the project's own ontology, the
learning system's own memories, and the trainer's own version gate. Nothing loads
a model or reaches the network, and every durable store is isolated with a
temporary ``FAYD_DATA_DIR``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from cognition.consolidation import (
    REVIEW_INTERVALS_DAYS,
    Consolidator,
    Pattern,
    ReviewSchedule,
    SkillRegistry,
)
from cognition.curriculum import (
    AdaptivePolicy,
    AutonomousCurriculum,
    MultiSeedEvaluator,
    TrainingAttempt,
    TrainingMemory,
)
from cognition.engines.ideas import IdeaEngine
from cognition.metacognition import (
    COLD_AFTER_CYCLES,
    EscalationPolicy,
    FocusBuffer,
    MetacognitiveController,
    PersistentNovelty,
    SelfModel,
    StagnationDetector,
)
from cognition.proactive_mind import ProactiveMind
from cognition.records import Idea, IdeaKind, ThoughtCycle
from cognition.training import TrainingStage
from core.ontology import Ontology
from core.petroleum_corpus import seed_petroleum_corpus


@pytest.fixture(scope="module")
def ontology() -> Ontology:
    graph = Ontology()
    seed_petroleum_corpus(graph)
    return graph


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Metacognition: the self-model
# ---------------------------------------------------------------------------


def _idea(engine: str, *, confidence: float = 0.5, novelty: float = 0.5, kind=IdeaKind.IDEA) -> Idea:
    return Idea(
        text=f"a thought from {engine} number {confidence}",
        kind=kind,
        engine=engine,
        confidence=confidence,
        novelty=novelty,
    )


def test_self_model_records_what_each_strategy_produced(isolated):
    model = SelfModel()
    model.advance_cycle()
    model.observe("ideas.relation", _idea("ideas.relation"), kept=True)
    model.observe("ideas.relation", _idea("ideas.relation"), kept=False)
    record = model.records()[0]
    assert record.name == "ideas.relation"
    assert record.produced == 2
    assert record.kept == 1
    assert record.keep_rate == 0.5


def test_a_promoted_idea_is_counted_as_a_promotion(isolated):
    model = SelfModel()
    model.advance_cycle()
    model.observe(
        "imagination.counterfactual",
        _idea("imagination.counterfactual", kind=IdeaKind.HYPOTHESIS),
        kept=True,
    )
    assert model.records()[0].promoted == 1


def test_an_untried_strategy_has_a_neutral_bias(isolated):
    model = SelfModel()
    assert model.bias("never.seen") == 0.5
    assert model.yield_for("never.seen").produced == 0


def test_a_strategy_that_pays_off_is_biased_above_neutral(isolated):
    model = SelfModel()
    model.advance_cycle()
    for _ in range(4):
        model.observe("good", _idea("good", confidence=0.9, novelty=0.9, kind=IdeaKind.HYPOTHESIS), kept=True)
    assert model.bias("good") > 0.5


def test_a_cold_strategy_is_given_the_neutral_prior_again(isolated):
    model = SelfModel()
    model.advance_cycle()
    model.observe("cold", _idea("cold", confidence=0.1, novelty=0.1), kept=False)
    for _ in range(COLD_AFTER_CYCLES):
        model.advance_cycle()
    # Even a poorly-performing strategy is re-explored once it has gone cold.
    assert model.bias("cold") == 0.5


def test_the_self_model_survives_a_restart(isolated):
    model = SelfModel()
    model.advance_cycle()
    model.observe("ideas.relation", _idea("ideas.relation"), kept=True)
    model.save()
    reloaded = SelfModel.load()
    assert reloaded.cycle == 1
    assert reloaded.records()[0].produced == 1


def test_the_self_model_is_bounded_in_its_strategies(isolated):
    from cognition.metacognition import MAX_STRATEGIES

    model = SelfModel()
    for index in range(MAX_STRATEGIES + 5):
        model.observe(f"strategy-{index}", _idea(f"strategy-{index}"), kept=True)
    assert len(model.records()) <= MAX_STRATEGIES


# -- persistent novelty -----------------------------------------------------


def test_persistent_novelty_forgets_nothing_across_a_restart(isolated):
    novelty = PersistentNovelty()
    novelty.remember("porosity measures pore space")
    novelty.save()
    reloaded = PersistentNovelty.load()
    assert reloaded.count() == 1
    # A statement already thought is no longer novel to the restarted mind.
    assert reloaded.novelty("porosity measures pore space") < 1.0


def test_persistent_novelty_calls_a_new_statement_novel(isolated):
    novelty = PersistentNovelty()
    novelty.remember("porosity measures pore space")
    assert novelty.novelty("waterflood increases recovery") == 1.0


def test_persistent_novelty_is_bounded(isolated):
    novelty = PersistentNovelty(max_items=3)
    for index in range(6):
        novelty.remember(f"a distinctly worded statement number {index}")
    assert novelty.count() == 3


def test_the_idea_engine_accepts_the_durable_novelty_memory(ontology, isolated):
    # The durable memory satisfies the same interface the in-RAM tracker does.
    engine = IdeaEngine(ontology, novelty=PersistentNovelty())
    ideas = engine.generate(labels=("effective_porosity", "permeability"), limit=2)
    assert ideas


# -- focus buffer -----------------------------------------------------------


def test_focus_reinforces_and_decays():
    buffer = FocusBuffer(decay=0.5)
    buffer.reinforce("porosity")
    assert buffer.weight("porosity") == 1.0
    buffer.decay_all()
    assert 0.0 < buffer.weight("porosity") < 1.0


def test_focus_forgets_a_topic_that_fades_away():
    buffer = FocusBuffer(decay=0.1)
    buffer.reinforce("porosity")
    for _ in range(5):
        buffer.decay_all()
    assert buffer.weight("porosity") == 0.0


def test_focus_is_bounded():

    buffer = FocusBuffer(max_items=3)
    for index in range(6):
        buffer.reinforce(f"topic-{index}")
    assert len(buffer.topics(limit=100)) <= 3


def test_focus_survives_a_restart(isolated):
    buffer = FocusBuffer()
    buffer.reinforce("permeability")
    buffer.save()
    reloaded = FocusBuffer.load()
    assert "permeability" in reloaded.topics()


# -- stagnation and escalation ---------------------------------------------


def _cycle(ideas: tuple[str, ...], gaps: tuple[str, ...]) -> ThoughtCycle:
    return ThoughtCycle(ideas=ideas, gaps=gaps)


def test_stagnation_needs_more_than_one_cycle_to_judge():
    detector = StagnationDetector(window=4)
    reading = detector.read((_cycle((), ()),), {})
    assert not reading.stagnating
    assert "not enough" in reading.reason


def test_a_flat_mind_that_promotes_nothing_is_stagnating():
    detector = StagnationDetector(window=4, novelty_floor=0.5)
    ideas = {"i1": _idea("e", novelty=0.1), "i2": _idea("e", novelty=0.1)}
    cycles = tuple(_cycle(("i1", "i2"), ("porosity", "porosity")) for _ in range(4))
    reading = detector.read(cycles, ideas)
    assert reading.stagnating
    assert reading.repeated_topics > 0


def test_a_mind_that_promotes_something_is_not_stagnating():
    detector = StagnationDetector(window=4, novelty_floor=0.5)
    ideas = {
        "i1": _idea("e", novelty=0.1),
        "i2": _idea("e", novelty=0.1, kind=IdeaKind.HYPOTHESIS),
    }
    cycles = tuple(_cycle(("i1", "i2"), ("porosity", "porosity")) for _ in range(4))
    reading = detector.read(cycles, ideas)
    assert not reading.stagnating


def test_escalation_widens_the_search_within_ceilings():
    policy = EscalationPolicy(max_widen_concepts=2, max_widen_ideas=2, max_novelty_floor=0.6)
    detector = StagnationDetector(window=4, novelty_floor=0.5)
    ideas = {"i1": _idea("e", novelty=0.1)}
    cycles = tuple(_cycle(("i1",), ("porosity", "porosity")) for _ in range(4))
    reading = detector.read(cycles, ideas)
    first = policy.plan(reading, SelfModel())
    assert first is not None
    assert first.widen_concepts == 1
    for _ in range(6):
        last = policy.plan(reading, SelfModel())
    assert last is not None
    assert last.widen_concepts <= 2
    assert last.widen_ideas <= 2
    assert last.novelty_floor <= 0.6


def test_no_escalation_when_the_mind_is_progressing():
    policy = EscalationPolicy()
    from cognition.metacognition import StagnationReading

    assert policy.plan(StagnationReading(False, 0.2, 1, 0), SelfModel()) is None
    assert policy.steps == 0


# -- the controller ---------------------------------------------------------


def test_the_controller_plans_and_learns(isolated):
    controller = MetacognitiveController()
    plan = controller.plan()
    assert plan.idea_budget >= 1
    number = controller.observe_cycle(
        produced=(("ideas.relation", _idea("ideas.relation"), True),),
        topics=("porosity",),
    )
    assert number == 1
    assert controller.status()["strategies"]


def test_the_controller_persists_its_self_model(isolated):
    controller = MetacognitiveController()
    controller.observe_cycle(produced=(("ideas.relation", _idea("ideas.relation"), True),), topics=("porosity",))
    reloaded = MetacognitiveController()
    assert reloaded.self_model.cycle == 1
    assert reloaded.self_model.records()


def test_a_cycle_records_which_strategy_produced_each_idea(ontology, isolated):
    mind = ProactiveMind(ontology)
    mind.cycle(max_ideas=3)
    assert mind.metacognition.self_model.records()


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------


class _FakeLearning:
    """A stand-in learning system exposing only what the consolidator reads."""

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

    def record_experience(self, expectation: str, outcome: str, *, success: bool) -> None:
        from cognition.records import Experience

        self._experiences.append(Experience(expectation=expectation, outcome=outcome, success=success))

    def learn_fact(self, fact_id, statement, *, source=None, confidence=0.5, verified=False):
        self._facts.append((fact_id, statement))
        return None


def test_a_pattern_needs_real_support(isolated):
    learning = _FakeLearning()
    learning.record_experience("acid stimulation on tight rock", "permeability improves", success=True)
    consolidator = Consolidator(learning)
    report = consolidator.consolidate()
    # One anecdote is not a rule.
    assert report.patterns == ()


def test_repeated_success_is_consolidated_into_a_rule(isolated):
    learning = _FakeLearning()
    for _ in range(3):
        learning.record_experience("acid stimulation on tight rock", "permeability improves", success=True)
    consolidator = Consolidator(learning)
    report = consolidator.consolidate()
    assert len(report.patterns) == 1
    pattern = report.patterns[0]
    assert pattern.kind == "success_rule"
    assert pattern.support == 3
    assert pattern.confidence == 1.0


def test_a_strong_pattern_is_upgraded_into_a_learned_fact(isolated):
    learning = _FakeLearning()
    for _ in range(3):
        learning.record_experience("waterflood in a depleted reservoir", "recovery improves", success=True)
    report = Consolidator(learning).consolidate()
    assert report.upgraded
    # The upgraded fact is stored unverified -- a pattern is an inference.
    assert learning._facts


def test_a_pattern_id_is_content_addressed_and_stable():
    first = Pattern(condition="a", consequent="b", kind="success_rule", support=2, confidence=0.8)
    second = Pattern(condition="a", consequent="b", kind="success_rule", support=2, confidence=0.8)
    assert first.id == second.id
    assert first.id.startswith("PAT-")


def test_patterns_are_deterministic(isolated):
    learning = _FakeLearning()
    for _ in range(3):
        learning.record_experience("acid stimulation on tight rock", "permeability improves", success=True)
    first = [p.describe() for p in Consolidator(learning).consolidate().patterns]
    second = [p.describe() for p in Consolidator(learning).consolidate().patterns]
    assert first == second


def test_a_review_box_advances_on_recall_and_resets_on_a_miss():
    schedule = ReviewSchedule()
    item = schedule.add("PAT-1", "when a then b")
    assert item.box == 0
    schedule.review("PAT-1", remembered=True)
    assert schedule._items["PAT-1"].box == 1
    schedule.review("PAT-1", remembered=False)
    assert schedule._items["PAT-1"].box == 0
    assert schedule._items["PAT-1"].lapses == 1


def test_a_review_interval_grows_with_the_box():
    schedule = ReviewSchedule()
    schedule.add("PAT-1", "when a then b")
    now = datetime.now(UTC)
    first_due = schedule._items["PAT-1"].due_at
    schedule.review("PAT-1", remembered=True, now=now)
    assert schedule._items["PAT-1"].due_at > now
    assert schedule._items["PAT-1"].interval_days == REVIEW_INTERVALS_DAYS[1]
    assert schedule._items["PAT-1"].due_at != first_due


def test_a_review_schedule_survives_a_restart(isolated):
    schedule = ReviewSchedule()
    schedule.add("PAT-1", "when a then b")
    schedule.save()
    reloaded = ReviewSchedule.load()
    assert reloaded.count() == 1


def test_skill_proficiency_is_measured_not_guessed(isolated):
    registry = SkillRegistry()
    registry.practise("acidise a core", success=True)
    registry.practise("acidise a core", success=False)
    skill = registry.skills()[0]
    assert skill.attempts == 2
    assert skill.successes == 1
    assert skill.proficiency == 0.5


def test_skill_refinement_keeps_the_more_detailed_procedure(isolated):
    registry = SkillRegistry()
    registry.practise("log a well", success=True, steps=("a",))
    registry.practise("log a well", success=True, steps=("a", "b", "c"))
    assert registry.skills()[0].steps == ("a", "b", "c")


def test_a_skill_registry_survives_a_restart(isolated):
    registry = SkillRegistry()
    registry.practise("log a well", success=True)
    reloaded = SkillRegistry.load()
    assert reloaded.count() == 1


def test_the_consolidator_is_reachable_from_the_learning_system(ontology, isolated):
    from cognition.learning_system import CognitiveLearningSystem

    learning = CognitiveLearningSystem()
    report = learning.consolidate()
    assert report.patterns == ()


# ---------------------------------------------------------------------------
# Adaptive curriculum
# ---------------------------------------------------------------------------


def test_training_memory_records_and_persists(isolated):
    memory = TrainingMemory()
    memory.record(
        TrainingAttempt(stage="quick", accepted=True, reason="improves", learning_rate=0.001, steps=10)
    )
    reloaded = TrainingMemory.load()
    assert len(reloaded.attempts()) == 1
    assert reloaded.attempts()[0].stage == "quick"


def test_training_memory_counts_consecutive_failures():
    memory = TrainingMemory()
    memory.record(TrainingAttempt(stage="quick", accepted=True, reason="improves", learning_rate=0.001, steps=10))
    memory.record(TrainingAttempt(stage="quick", accepted=False, reason="no improvement", learning_rate=0.001, steps=10))
    memory.record(TrainingAttempt(stage="quick", accepted=False, reason="no improvement", learning_rate=0.001, steps=10))
    assert memory.consecutive_failures(TrainingStage.QUICK) == 2


class _FakeTrainer:
    """A stand-in trainer that reports the stages without loading a model."""

    def __init__(self, *, outcomes: list[bool] | None = None) -> None:
        self._outcomes = outcomes or []
        self._calls = 0
        self.accepted: list[TrainingStage] = []

    def next_stage(self):
        if self._calls >= len(self._outcomes):
            return None
        return TrainingStage.QUICK

    def run(self, *, stage, settings, progress=None):
        from types import SimpleNamespace

        accepted = self._outcomes[self._calls] if self._calls < len(self._outcomes) else False
        self._calls += 1
        self.accepted.append(stage)
        plan = SimpleNamespace(
            settings=SimpleNamespace(learning_rate=0.001, steps=10),
            stage=stage,
        )
        return SimpleNamespace(
            plan=plan,
            accepted=accepted,
            reason="improves" if accepted else "no improvement",
            version=None,
        )


def test_the_policy_advances_when_the_last_run_was_accepted():
    policy = AdaptivePolicy()
    decision = policy.decide(trainer=_FakeTrainer(outcomes=[True]), memory=TrainingMemory())
    assert decision.action == "advance"
    assert decision.stage == TrainingStage.QUICK


def test_the_policy_stops_when_every_stage_is_done():
    class _Done:
        def next_stage(self):
            return None

    decision = AdaptivePolicy().decide(trainer=_Done(), memory=TrainingMemory())
    assert decision.action == "stop"


def test_the_policy_adjusts_the_learning_rate_on_a_plateau():
    from cognition.curriculum import PLATEAU_AFTER_ATTEMPTS
    from cognition.training import STAGE_SETTINGS

    memory = TrainingMemory()
    for _ in range(PLATEAU_AFTER_ATTEMPTS):
        memory.record(
            TrainingAttempt(stage="quick", accepted=False, reason="no improvement", learning_rate=0.001, steps=10)
        )
    decision = AdaptivePolicy().decide(trainer=_FakeTrainer(outcomes=[True]), memory=memory)
    assert decision.action == "adjust"
    assert decision.settings is not None
    assert decision.settings.learning_rate != STAGE_SETTINGS[TrainingStage.QUICK].learning_rate


def test_the_policy_stops_when_the_effort_budget_is_spent():
    memory = TrainingMemory()
    for _ in range(3):
        memory.record(
            TrainingAttempt(stage="quick", accepted=False, reason="no improvement", learning_rate=0.001, steps=10)
        )
    decision = AdaptivePolicy(max_stages=3).decide(trainer=_FakeTrainer(outcomes=[True]), memory=memory)
    assert decision.action == "stop"


def test_the_autonomous_curriculum_refuses_without_approval(isolated):
    report = AutonomousCurriculum(trainer=_FakeTrainer(outcomes=[True])).run(approved=False)
    assert report.outcomes == ()
    assert "approved" in report.stopped_because


def test_the_autonomous_curriculum_runs_and_stops_itself(isolated):
    curriculum = AutonomousCurriculum(trainer=_FakeTrainer(outcomes=[True]))
    report = curriculum.run(approved=True, max_stages=3)
    assert report.outcomes
    assert report.stopped_because
    # Every decision is recorded, so the run can be inspected honestly.
    assert report.decisions


def test_the_autonomous_curriculum_records_its_attempts(isolated):
    curriculum = AutonomousCurriculum(trainer=_FakeTrainer(outcomes=[True]))
    curriculum.run(approved=True, max_stages=2)
    assert TrainingMemory.load().attempts()


def test_multi_seed_evaluation_is_honest_when_it_cannot_measure():
    reading = MultiSeedEvaluator().measure("no-such-checkpoint.npz", [])
    assert not reading.measured
    assert "too few units" in reading.reason
    assert "not available" in reading.describe()


# ---------------------------------------------------------------------------
# Integration: the whole thing through the controller
# ---------------------------------------------------------------------------


def test_the_controller_exposes_the_new_faculties(isolated):
    from core.cognitive_controller import CognitiveController

    controller = CognitiveController()
    assert controller.metacognition_status()["cycle"] == 0
    assert controller.consolidate_learning() is not None
    assert controller.autonomous_curriculum() is not None


def test_a_cycle_then_consolidation_then_plan_is_coherent(ontology, isolated):
    mind = ProactiveMind(ontology)
    mind.cycle(max_ideas=3)
    plan = mind.plan_next()
    assert plan.idea_budget >= 1
    assert isinstance(mind.metacognition.status()["strategies"], list)


def test_the_service_exposes_the_new_faculties(isolated):
    from core.cognitive_controller import CognitiveController
    from gui.cognition import CognitiveService, OperationReport

    controller = CognitiveController()
    service = CognitiveService(data_root=None, controller=controller)
    assert isinstance(service.metacognition(), OperationReport)
    assert isinstance(service.plan_next(), OperationReport)
    assert isinstance(service.consolidate(), OperationReport)
    assert isinstance(service.review_due(), OperationReport)
    assert service.patterns() == ()
    assert service.skills() == ()
    # The autonomous run is a governed action: it refuses without approval.
    refused = service.run_autonomous(approved=False)
    assert not refused.ok


# ---------------------------------------------------------------------------
# The held-out development gate
# ---------------------------------------------------------------------------


def test_the_development_gate_is_green_on_the_real_faculties():
    from cognition.development import development_accuracy

    assert development_accuracy() == 1.0


def test_the_development_gate_describes_every_judgement():
    from cognition.development import describe_development

    text = describe_development()
    assert "Cognitive development gate" in text
    assert "FAIL" not in text


def test_the_development_gate_is_empty_safe():
    from cognition.development import development_accuracy

    assert development_accuracy(()) == 0.0


def test_the_development_gate_fails_when_a_faculty_regresses(monkeypatch):
    # If the self-model stopped favouring what pays off, its judgement must fail.
    from cognition import development

    monkeypatch.setitem(
        development._CHECKS,
        "self_model_favours_what_pays_off",
        lambda: False,
    )
    assert development.development_accuracy() < 1.0
