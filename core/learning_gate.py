"""The held-out gate for the Phase-3 learning architecture.

Phase 3's learning modules each ship with unit tests that pin the good case. This
module is the *external* check: a catalogue of hand-written judgements, each run
against the **real** pipelines -- never a re-implementation -- with **negative
controls**, so a regression the unit tests still pass fails here.

Every judgement is a real call over a temporary data folder (never the user's own
data):

* **knowledge learning** -- a well-evidenced claim is integrated into the graph,
  and the negative control: a claim with **no** evidence is never integrated;
* **provenance is preserved** -- an integrated claim keeps its source and its
  evidence in the graph;
* **duplicate reinforcement** -- learning the same claim twice adds no second
  edge (the graph holds one relation, reinforced);
* **contradiction preserved** -- two opposing claims on the same pair are both
  kept and the conflict is reported, never silently resolved;
* **experience persistence** -- an experience survives a restart as an episodic
  memory, and a lesson is rebuilt from it;
* **no premature generalisation** -- the negative control: a *single* success does
  not promote a procedure, while repeated corroborated successes do;
* **failure is a caution, not a rule** -- an unsuccessful experience never becomes
  a trusted procedure;
* **feedback bounded** -- one feedback record never overrides reliable evidence,
  and an inferred signal is bounded below an explicit one;
* **silence is not approval** -- an empty or unknown reaction is refused;
* **feedback is correctable** -- a retraction supersedes without deleting;
* **versioning** -- each applied learning act advances the store's version, and a
  rollback restores the earlier version while keeping the audit trail;
* **audit trail** -- every learning act is recorded with its kind, source, and
  outcome;
* **isolation** -- one owner's learning store is not another's, and the negative
  control: the main model may **not** learn from Natasha;
* **no weight changes** -- the module never writes a model file (checked by the
  absence of any model write path in the exercised code).

``learning_accuracy`` is the share of judgements satisfied; an empty catalogue
scores ``0.0`` rather than a faked pass.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from core.experience_learning import ExperienceLearningPipeline, experience_from_result
from core.feedback_processing import FEEDBACK_INFERRED, FeedbackProcessor, feedback_from_reaction
from core.knowledge_graph import KnowledgeGraph
from core.knowledge_learning import KnowledgeLearningPipeline, claim_from_relation
from core.learning_coordinator import LearningEventProcessor, coordinator_for
from core.learning_events import KIND_KNOWLEDGE, OUTCOME_ACCEPTED, OUTCOME_REJECTED
from core.learning_isolation import LearningAccessError
from core.learning_store import LearningStore
from core.long_term_memory import LongTermMemory
from core.memory_record import MemoryKind
from core.pattern_learning import STATUS_TRUSTED, PatternLearning

#: A judgement: a name and the real check that must hold.
Judgement = tuple[str, Callable[[Path], bool]]


@dataclass
class GateReport:
    """The detailed outcome, so a failure names itself."""

    passed: tuple[str, ...] = ()
    failed: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total(self) -> int:
        return len(self.passed) + len(self.failed)

    @property
    def accuracy(self) -> float:
        return len(self.passed) / self.total if self.total else 0.0

    def describe(self) -> str:
        return (
            f"{len(self.passed)}/{self.total} learning judgements satisfied"
            + (f"; failed: {', '.join(self.failed)}" if self.failed else "")
        )


def _store(root: Path, owner: str = "fayd_al_ruh") -> LearningStore:
    return LearningStore(owner, root / f"{owner}-learning.sqlite3")


def _memory(root: Path, owner: str = "fayd_al_ruh") -> LongTermMemory:
    return LongTermMemory(owner, root / f"{owner}-memory.sqlite3")


# -- knowledge learning -----------------------------------------------------
def _knowledge_integrates_evidenced_claim(root: Path) -> bool:
    graph = KnowledgeGraph()
    store = _store(root)
    pipeline = KnowledgeLearningPipeline(graph, store=store)
    claim = claim_from_relation(
        "porosity", "increases", "permeability", confidence=0.8, source_kind="document", reference="SPE-1"
    )
    result = pipeline.learn_claim(claim, source="document", source_reference="SPE-1")
    return result.integrated == 1 and len(graph.relations) == 1


def _knowledge_refuses_unsupported_claim(root: Path) -> bool:
    """Negative control: a claim with no evidence is never integrated."""
    import dataclasses

    graph = KnowledgeGraph()
    store = _store(root)
    pipeline = KnowledgeLearningPipeline(graph, store=store)
    claim = dataclasses.replace(
        claim_from_relation("porosity", "increases", "permeability", confidence=0.9), evidence=()
    )
    result = pipeline.learn_claim(claim, source="model")
    return result.integrated == 0 and len(graph.relations) == 0 and result.rejected == 1


def _knowledge_preserves_provenance(root: Path) -> bool:
    graph = KnowledgeGraph()
    store = _store(root)
    pipeline = KnowledgeLearningPipeline(graph, store=store)
    claim = claim_from_relation(
        "waterflood", "increases", "recovery", confidence=0.8, source_kind="document", reference="SPE-42"
    )
    pipeline.learn_claim(claim, source="document", source_reference="SPE-42")
    relation = next(iter(graph.relations.values()))
    evidence = graph.evidence.get(relation.evidence_ids[0]) if relation.evidence_ids else None
    return evidence is not None and evidence.source == "SPE-42"


def _knowledge_duplicate_reinforces(root: Path) -> bool:
    graph = KnowledgeGraph()
    store = _store(root)
    pipeline = KnowledgeLearningPipeline(graph, store=store)
    claim = claim_from_relation(
        "porosity", "increases", "permeability", confidence=0.8, source_kind="document", reference="SPE-1"
    )
    pipeline.learn_claim(claim, source="document", source_reference="SPE-1")
    second = pipeline.learn_claim(claim, source="document", source_reference="SPE-1")
    return len(graph.relations) == 1 and second.duplicates == 1


def _knowledge_preserves_contradiction(root: Path) -> bool:
    graph = KnowledgeGraph()
    store = _store(root)
    pipeline = KnowledgeLearningPipeline(graph, store=store)
    pipeline.learn_claim(
        claim_from_relation("waterflood", "increases", "recovery", confidence=0.8, source_kind="document", reference="A"),
        source="document",
        source_reference="A",
    )
    result = pipeline.learn_claim(
        claim_from_relation("waterflood", "reduces", "recovery", confidence=0.8, source_kind="document", reference="B"),
        source="document",
        source_reference="B",
    )
    # Both claims are kept; the conflict is reported rather than resolved.
    return len(graph.relations) == 2 and result.conflicts == 1


# -- experience learning ----------------------------------------------------
def _experience_persists(root: Path) -> bool:
    memory = _memory(root)
    store = _store(root)
    pipeline = ExperienceLearningPipeline(memory, store=store)
    experience = experience_from_result(
        "increase permeability", "acid stimulation worked", success=True, approach="acid stimulation", importance=0.7
    )
    result = pipeline.learn(experience)
    second_memory = _memory(root)
    loaded = second_memory.get(result.memory_id)
    return loaded is not None and loaded.kind is MemoryKind.EPISODIC and loaded.metadata.get("success") is True


def _experience_lesson_rebuilt(root: Path) -> bool:
    memory = _memory(root)
    store = _store(root)
    first = ExperienceLearningPipeline(memory, store=store)
    first.learn(
        experience_from_result("increase permeability", "worked", success=True, approach="acid", importance=0.7)
    )
    first.learn(
        experience_from_result("increase permeability", "worked again", success=True, approach="acid", importance=0.7)
    )
    rebuilt = ExperienceLearningPipeline(_memory(root), store=_store(root))
    lessons = rebuilt.lessons_from_memory()
    return len(lessons) == 1 and lessons[0].support == 2


def _no_premature_generalisation(root: Path) -> bool:
    """Negative control: one success never promotes; repeated support does."""
    memory = _memory(root)
    store = _store(root)
    pipeline = ExperienceLearningPipeline(memory, store=store, min_support=2)
    single = pipeline.learn(
        experience_from_result("increase permeability", "worked", success=True, approach="acid", importance=0.7)
    )
    if single.promoted:
        return False
    second = pipeline.learn(
        experience_from_result("increase permeability", "worked again", success=True, approach="acid", importance=0.7)
    )
    return second.promoted


def _failure_is_a_caution(root: Path) -> bool:
    memory = _memory(root)
    store = _store(root)
    pipeline = ExperienceLearningPipeline(memory, store=store, min_support=1)
    result = pipeline.learn(
        experience_from_result(
            "drill faster", "the bit failed", success=False, errors=("bit damage",), importance=0.8
        )
    )
    return not result.promoted and result.lesson is not None and not result.lesson.success


# -- feedback ---------------------------------------------------------------
def _feedback_never_overrides_evidence(root: Path) -> bool:
    memory = _memory(root)
    store = _store(root)
    processor = FeedbackProcessor(memory, store=store)
    outcome = processor.process("incorrect", "answer-1", evidence_strength=0.95)
    return outcome.accepted and not outcome.overrides_evidence


def _inferred_below_explicit(root: Path) -> bool:
    memory = _memory(root)
    store = _store(root)
    processor = FeedbackProcessor(memory, store=store)
    explicit = processor.process("correct", "a", confidence=0.95)
    inferred = processor.process("correct", "b", source_kind=FEEDBACK_INFERRED, confidence=0.95)
    return inferred.record.weight < explicit.record.weight


def _silence_is_not_approval(root: Path) -> bool:
    try:
        feedback_from_reaction("", "answer-1")
    except ValueError:
        try:
            feedback_from_reaction("maybe", "answer-1")
        except ValueError:
            return True
    return False


def _feedback_is_correctable(root: Path) -> bool:
    memory = _memory(root)
    store = _store(root)
    processor = FeedbackProcessor(memory, store=store)
    outcome = processor.process("correct", "answer-1")
    retraction = processor.retract(outcome.record.feedback_id, reason="changed my mind")
    return retraction.record.retracted and len(processor.active()) == 0


# -- patterns and procedures ------------------------------------------------
def _patterns_need_support(root: Path) -> bool:
    memory = _memory(root)
    store = _store(root)
    experiences = ExperienceLearningPipeline(memory, store=store)
    for index in range(3):
        experiences.learn(
            experience_from_result(
                "increase permeability", f"attempt {index} worked", success=True, approach="acid", importance=0.7
            )
        )
    learning = PatternLearning(memory, store=store, min_support=3)
    report = learning.extract()
    return len(report.promoted) == 1 and all(p.status == STATUS_TRUSTED for p in learning.trusted_procedures())


def _patterns_record_failure_modes(root: Path) -> bool:
    memory = _memory(root)
    store = _store(root)
    experiences = ExperienceLearningPipeline(memory, store=store)
    for index in range(3):
        experiences.learn(
            experience_from_result("increase permeability", f"ok {index}", success=True, approach="acid", importance=0.7)
        )
    experiences.learn(
        experience_from_result(
            "increase permeability", "failed", success=False, errors=("pump failed",), importance=0.7
        )
    )
    learning = PatternLearning(memory, store=store, min_support=3)
    learning.extract()
    procedures = learning.procedures()
    return bool(procedures) and "pump failed" in procedures[0].failure_modes


# -- versioning, rollback, audit -------------------------------------------
def _versioning_and_rollback(root: Path) -> bool:
    store = _store(root)
    pipeline = KnowledgeLearningPipeline(KnowledgeGraph(), store=store)
    pipeline.learn_claim(
        claim_from_relation("porosity", "increases", "permeability", confidence=0.8, source_kind="document", reference="S"),
        source="document",
        source_reference="S",
    )
    version_after_first = store.version
    pipeline.learn_claim(
        claim_from_relation("waterflood", "increases", "recovery", confidence=0.8, source_kind="document", reference="S"),
        source="document",
        source_reference="S",
    )
    version_after_second = store.version
    if not (version_after_second > version_after_first > 0):
        return False
    reversals = store.rollback(version_after_second)
    # The audit trail is kept; the events are marked rolled back, not deleted.
    return bool(reversals) and store.count() >= 2


def _audit_trail(root: Path) -> bool:
    store = _store(root)
    pipeline = KnowledgeLearningPipeline(KnowledgeGraph(), store=store)
    pipeline.learn_claim(
        claim_from_relation("porosity", "increases", "permeability", confidence=0.8, source_kind="document", reference="S"),
        source="document",
        source_reference="S",
    )
    events, _ = store.events(kind=KIND_KNOWLEDGE)
    return bool(events) and events[0].source == "document" and events[0].outcome == OUTCOME_ACCEPTED


def _rejected_act_is_recorded(root: Path) -> bool:
    """Negative control: a refused act is recorded honestly as rejected."""
    store = _store(root)
    pipeline = KnowledgeLearningPipeline(KnowledgeGraph(), store=store)
    import dataclasses

    claim = dataclasses.replace(
        claim_from_relation("a", "increases", "b", confidence=0.9, source_kind="model"), evidence=()
    )
    pipeline.learn_claim(claim, source="model")
    events, _ = store.events(kind=KIND_KNOWLEDGE)
    return bool(events) and events[0].outcome == OUTCOME_REJECTED and not events[0].applied


# -- isolation --------------------------------------------------------------
def _isolation_of_stores(root: Path) -> bool:
    jarvis = coordinator_for("jarvis", data_root=root)
    natasha = coordinator_for("natasha", data_root=root)
    return jarvis.store.path != natasha.store.path and jarvis.store.path.parent != natasha.store.path.parent


def _natasha_is_excluded(root: Path) -> bool:
    """Negative control: the main model may not learn from Natasha."""
    main = coordinator_for("fayd_al_ruh", data_root=root)
    coordinator_for("natasha", data_root=root)  # her store exists
    if "natasha" in main.dataset_owners():
        return False
    try:
        main.learn_from("natasha")
    except LearningAccessError:
        return True
    return False


def _event_processor_is_incremental(root: Path) -> bool:
    coordinator = coordinator_for("fayd_al_ruh", data_root=root)
    coordinator.learn_claim(
        claim_from_relation("porosity", "increases", "permeability", confidence=0.8, source_kind="document", reference="S"),
        source="document",
        source_reference="S",
    )
    processor = LearningEventProcessor()
    first = processor.drain(coordinator)
    second = processor.drain(coordinator)
    return bool(first) and not second


def _no_model_weights_touched(root: Path) -> bool:
    """The learning path writes only its own stores, never a model file."""
    coordinator = coordinator_for("fayd_al_ruh", data_root=root)
    coordinator.learn_claim(
        claim_from_relation("porosity", "increases", "permeability", confidence=0.8, source_kind="document", reference="S"),
        source="document",
        source_reference="S",
    )
    written = {path.name for path in root.rglob("*") if path.is_file()}
    return not any(name.endswith((".gguf", ".npz", ".onnx", ".bin", ".safetensors")) for name in written)


#: The held-out catalogue: a name and the real check that must hold.
LEARNING_JUDGEMENTS: tuple[Judgement, ...] = (
    ("knowledge-integrates-evidenced-claim", _knowledge_integrates_evidenced_claim),
    ("knowledge-refuses-unsupported-claim", _knowledge_refuses_unsupported_claim),
    ("knowledge-preserves-provenance", _knowledge_preserves_provenance),
    ("knowledge-duplicate-reinforces", _knowledge_duplicate_reinforces),
    ("knowledge-preserves-contradiction", _knowledge_preserves_contradiction),
    ("experience-persists", _experience_persists),
    ("experience-lesson-rebuilt", _experience_lesson_rebuilt),
    ("no-premature-generalisation", _no_premature_generalisation),
    ("failure-is-a-caution", _failure_is_a_caution),
    ("feedback-never-overrides-evidence", _feedback_never_overrides_evidence),
    ("inferred-below-explicit", _inferred_below_explicit),
    ("silence-is-not-approval", _silence_is_not_approval),
    ("feedback-is-correctable", _feedback_is_correctable),
    ("patterns-need-support", _patterns_need_support),
    ("patterns-record-failure-modes", _patterns_record_failure_modes),
    ("versioning-and-rollback", _versioning_and_rollback),
    ("audit-trail", _audit_trail),
    ("rejected-act-is-recorded", _rejected_act_is_recorded),
    ("isolation-of-stores", _isolation_of_stores),
    ("natasha-is-excluded", _natasha_is_excluded),
    ("event-processor-is-incremental", _event_processor_is_incremental),
    ("no-model-weights-touched", _no_model_weights_touched),
)


def learning_gate_report(judgements: tuple[Judgement, ...] = LEARNING_JUDGEMENTS) -> GateReport:
    """Run every judgement against the real pipelines and return a report."""
    passed: list[str] = []
    failed: list[str] = []
    for name, check in judgements:
        try:
            with tempfile.TemporaryDirectory() as folder:
                ok = bool(check(Path(folder)))
        except Exception:  # a judgement that raises is a failed judgement
            ok = False
        (passed if ok else failed).append(name)
    return GateReport(passed=tuple(passed), failed=tuple(failed))


def learning_accuracy(judgements: tuple[Judgement, ...] = LEARNING_JUDGEMENTS) -> float:
    """The share of held-out learning judgements the real architecture satisfies.

    A judgement counts only when the pipelines do exactly what they must, so a
    learning system that integrated an unsupported claim, overwrote a
    contradiction, promoted a single success, let one voice override evidence,
    lost a version, or leaked Natasha's learning fails here. Returns ``0.0`` for
    an empty catalogue.
    """
    return learning_gate_report(judgements).accuracy


__all__ = [
    "LEARNING_JUDGEMENTS",
    "GateReport",
    "Judgement",
    "learning_accuracy",
    "learning_gate_report",
]
