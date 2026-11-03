"""Phase 3: cognitive and experience-based learning.

These tests pin :mod:`core.learning_events`, :mod:`core.learning_store`,
:mod:`core.knowledge_learning`, :mod:`core.experience_learning`,
:mod:`core.feedback_processing`, :mod:`core.pattern_learning`,
:mod:`core.learning_coordinator`, :mod:`core.learning_evaluation`, and the
held-out gate :mod:`core.learning_gate`, plus the controller wiring.

They run against real SQLite stores in a temporary directory -- never the user's
own data -- and they check the honesty rules the task requires: an unsupported
claim is never integrated, a single success never promotes a procedure, one
feedback record never overrides evidence, silence is never approval, a
contradiction is preserved rather than resolved, versions roll back without
losing the audit trail, and Natasha's learning is excluded from the main model's
dataset.
"""

from __future__ import annotations

import sqlite3

import pytest
from core.cognitive_controller import CognitiveController
from core.experience_learning import (
    ExperienceLearningPipeline,
    ExperienceRecord,
    experience_from_result,
)
from core.feedback_processing import (
    DEFAULT_MAX_WEIGHT,
    FEEDBACK_INFERRED,
    INFERRED_MAX_WEIGHT,
    FeedbackProcessor,
    FeedbackRecord,
    feedback_from_reaction,
)
from core.knowledge_graph import KnowledgeGraph
from core.knowledge_learning import (
    KnowledgeLearningPipeline,
    claim_from_relation,
    source_reliability,
)
from core.learning_coordinator import (
    LearningEventProcessor,
    build_default_coordinator,
    coordinator_for,
)
from core.learning_evaluation import (
    compare,
    evaluate_learning,
    history_summary,
    learning_metrics,
    snapshot,
    version_history,
)
from core.learning_events import (
    KIND_EXPERIENCE,
    KIND_KNOWLEDGE,
    OUTCOME_REJECTED,
    OUTCOME_ROLLED_BACK,
    LearningEvent,
    LearningKind,
)
from core.learning_gate import (
    LEARNING_JUDGEMENTS,
    learning_accuracy,
    learning_gate_report,
)
from core.learning_isolation import LearningAccessError
from core.learning_store import LearningStore, LearningStoreError
from core.long_term_memory import LongTermMemory
from core.memory_management import MemoryManager
from core.memory_record import MemoryKind
from core.pattern_learning import (
    STATUS_PROPOSED,
    STATUS_TRUSTED,
    LearnedProcedure,
    PatternLearning,
)


@pytest.fixture
def data_root(tmp_path):
    return tmp_path / "data"


def _memory(data_root, owner="fayd_al_ruh") -> LongTermMemory:
    return LongTermMemory(owner, data_root / f"{owner}-memory.sqlite3")


def _store(data_root, owner="fayd_al_ruh") -> LearningStore:
    return LearningStore(owner, data_root / f"{owner}-learning.sqlite3")


# -- the event contract -----------------------------------------------------
class TestLearningEvent:
    def test_event_id_is_deterministic(self):
        first = LearningEvent(kind=KIND_KNOWLEDGE, target="a")
        second = LearningEvent(kind=KIND_KNOWLEDGE, target="a", created_at=first.created_at)
        assert first.event_id == second.event_id

    def test_unknown_kind_is_refused(self):
        with pytest.raises(ValueError):
            LearningEvent(kind="nonsense", target="a")

    def test_unknown_outcome_is_refused(self):
        with pytest.raises(ValueError):
            LearningEvent(kind=KIND_KNOWLEDGE, target="a", outcome="maybe")

    def test_unknown_source_is_refused(self):
        with pytest.raises(ValueError):
            LearningEvent(kind=KIND_KNOWLEDGE, target="a", source="telepathy")

    def test_empty_target_is_refused(self):
        with pytest.raises(ValueError):
            LearningEvent(kind=KIND_KNOWLEDGE, target="  ")

    def test_changed_requires_applied_and_a_difference(self):
        event = LearningEvent(kind=KIND_KNOWLEDGE, target="a", before="x", after="y", applied=True)
        assert event.changed
        not_applied = LearningEvent(kind=KIND_KNOWLEDGE, target="a", before="x", after="y")
        assert not not_applied.changed

    def test_round_trips_through_primitive(self):
        event = LearningEvent(
            kind=KIND_EXPERIENCE, target="goal", owner="jarvis", applied=True, confidence=0.7, version="3"
        )
        restored = LearningEvent.from_primitive(event.to_primitive())
        assert restored.kind == event.kind
        assert restored.owner == "jarvis"
        assert restored.version == "3"
        assert restored.applied is True

    def test_kind_enum_is_a_string_enum(self):
        assert LearningKind.KNOWLEDGE.value == "knowledge"
        assert str(LearningKind.KNOWLEDGE) == "LearningKind.KNOWLEDGE"


# -- the durable store ------------------------------------------------------
class TestLearningStore:
    def test_records_and_reads_back(self, data_root):
        store = _store(data_root)
        event = LearningEvent(kind=KIND_KNOWLEDGE, target="a", applied=True)
        store.record_event(event)
        assert store.count() == 1
        assert store.event(event.event_id) is not None

    def test_recording_is_idempotent(self, data_root):
        store = _store(data_root)
        event = LearningEvent(kind=KIND_KNOWLEDGE, target="a")
        store.record_event(event)
        store.record_event(event)
        assert store.count() == 1

    def test_survives_a_restart(self, data_root):
        store = _store(data_root)
        store.record_event(LearningEvent(kind=KIND_KNOWLEDGE, target="a", applied=True))
        store.close()
        reopened = _store(data_root)
        assert reopened.count() == 1

    def test_version_advances_monotonically(self, data_root):
        store = _store(data_root)
        first = store.next_version()
        second = store.next_version()
        assert (first, second) == (1, 2)

    def test_rollback_marks_without_deleting(self, data_root):
        store = _store(data_root)
        version = store.next_version()
        store.record_event(LearningEvent(kind=KIND_KNOWLEDGE, target="a", applied=True, version=str(version)))
        reversals = store.rollback(version)
        assert reversals
        assert store.count() >= 1  # the original is kept
        events, _ = store.events()
        assert any(event.outcome == OUTCOME_ROLLED_BACK for event in events)

    def test_backup_and_restore(self, data_root):
        store = _store(data_root)
        store.record_event(LearningEvent(kind=KIND_KNOWLEDGE, target="a", applied=True))
        backup = store.backup(data_root / "backup.sqlite3")
        assert backup.ok
        store.clear()
        assert store.count() == 0
        restore = store.restore(data_root / "backup.sqlite3")
        assert restore.ok
        assert store.count() == 1

    def test_bad_backup_is_refused(self, data_root):
        store = _store(data_root)
        store.record_event(LearningEvent(kind=KIND_KNOWLEDGE, target="a", applied=True))
        bad = data_root / "bad.sqlite3"
        bad.write_bytes(b"this is not a database")
        report = store.restore(bad)
        assert not report.ok
        assert store.count() == 1  # untouched

    def test_corrupt_row_is_skipped(self, data_root):
        store = _store(data_root)
        store.record_event(LearningEvent(kind=KIND_KNOWLEDGE, target="a", applied=True))
        connection = sqlite3.connect(str(store.path))
        connection.execute(
            "INSERT INTO learning_events VALUES('bad','knowledge','o','t','system','accepted',1,'','now','{not json')"
        )
        connection.commit()
        connection.close()
        events, report = store.events()
        assert report.skipped == 1
        assert len(events) == 1

    def test_counts_by_kind(self, data_root):
        store = _store(data_root)
        store.record_event(LearningEvent(kind=KIND_KNOWLEDGE, target="a"))
        store.record_event(LearningEvent(kind=KIND_EXPERIENCE, target="b"))
        counts = store.counts_by_kind()
        assert counts[KIND_KNOWLEDGE] == 1
        assert counts[KIND_EXPERIENCE] == 1

    def test_newer_schema_is_refused(self, data_root):
        store = _store(data_root)
        connection = sqlite3.connect(str(store.path))
        connection.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
        connection.commit()
        connection.close()
        with pytest.raises(LearningStoreError):
            _store(data_root)

    def test_saves_and_reads_items(self, data_root):
        store = _store(data_root)
        procedure = LearnedProcedure(procedure_id="P-1", name="acid", steps=("pump", "soak"))
        store.save_item(procedure)
        assert store.count_items(kind="procedure") == 1
        assert store.get_item("procedure:fayd_al_ruh:acid") is not None


# -- knowledge learning -----------------------------------------------------
class TestKnowledgeLearning:
    def test_integrates_an_evidenced_claim(self, data_root):
        graph = KnowledgeGraph()
        pipeline = KnowledgeLearningPipeline(graph, store=_store(data_root))
        claim = claim_from_relation("porosity", "increases", "permeability", confidence=0.8, source_kind="document")
        result = pipeline.learn_claim(claim, source="document")
        assert result.integrated == 1
        assert len(graph.relations) == 1

    def test_refuses_an_unsupported_claim(self, data_root):
        import dataclasses

        graph = KnowledgeGraph()
        pipeline = KnowledgeLearningPipeline(graph, store=_store(data_root))
        claim = dataclasses.replace(
            claim_from_relation("porosity", "increases", "permeability", confidence=0.9), evidence=()
        )
        result = pipeline.learn_claim(claim, source="model")
        assert result.integrated == 0
        assert result.rejected == 1
        assert len(graph.relations) == 0

    def test_preserves_provenance(self, data_root):
        graph = KnowledgeGraph()
        pipeline = KnowledgeLearningPipeline(graph, store=_store(data_root))
        claim = claim_from_relation("a", "increases", "b", confidence=0.8, source_kind="document", reference="SPE-7")
        pipeline.learn_claim(claim, source="document", source_reference="SPE-7")
        relation = next(iter(graph.relations.values()))
        evidence = graph.evidence[relation.evidence_ids[0]]
        assert evidence.source == "SPE-7"

    def test_duplicate_reinforces_without_a_second_edge(self, data_root):
        graph = KnowledgeGraph()
        pipeline = KnowledgeLearningPipeline(graph, store=_store(data_root))
        claim = claim_from_relation("a", "increases", "b", confidence=0.8, source_kind="document")
        pipeline.learn_claim(claim, source="document")
        second = pipeline.learn_claim(claim, source="document")
        assert len(graph.relations) == 1
        assert second.duplicates == 1

    def test_preserves_a_contradiction(self, data_root):
        graph = KnowledgeGraph()
        pipeline = KnowledgeLearningPipeline(graph, store=_store(data_root))
        pipeline.learn_claim(
            claim_from_relation("waterflood", "increases", "recovery", confidence=0.8, source_kind="document"),
            source="document",
        )
        result = pipeline.learn_claim(
            claim_from_relation("waterflood", "reduces", "recovery", confidence=0.8, source_kind="document"),
            source="document",
        )
        assert len(graph.relations) == 2
        assert result.conflicts == 1

    def test_extracts_claims_from_text(self, data_root):
        from core.cognitive_controller import CognitiveController

        controller = CognitiveController()
        pipeline = KnowledgeLearningPipeline(
            controller.graph, ontology=controller.ontology, learner=controller.ontology_learner
        )
        result = pipeline.learn(
            "porosity increases permeability", source="document", confidence=0.8, record=False
        )
        assert result.claims

    def test_source_reliability_orders_provenance(self):
        assert source_reliability("user") > source_reliability("model")
        assert source_reliability("unknown") == 0.3

    def test_rejected_act_is_recorded(self, data_root):
        import dataclasses

        store = _store(data_root)
        pipeline = KnowledgeLearningPipeline(KnowledgeGraph(), store=store)
        claim = dataclasses.replace(
            claim_from_relation("a", "increases", "b", confidence=0.9, source_kind="model"), evidence=()
        )
        pipeline.learn_claim(claim, source="model")
        events, _ = store.events(kind=KIND_KNOWLEDGE)
        assert events[0].outcome == OUTCOME_REJECTED
        assert not events[0].applied


# -- experience learning ----------------------------------------------------
class TestExperienceLearning:
    def test_captures_an_experience_as_episodic_memory(self, data_root):
        memory = _memory(data_root)
        pipeline = ExperienceLearningPipeline(memory, store=_store(data_root))
        result = pipeline.learn(
            experience_from_result("goal", "it worked", success=True, approach="a", importance=0.7)
        )
        assert memory.get(result.memory_id) is not None
        assert memory.get(result.memory_id).kind is MemoryKind.EPISODIC

    def test_experience_survives_a_restart(self, data_root):
        memory = _memory(data_root)
        pipeline = ExperienceLearningPipeline(memory, store=_store(data_root))
        result = pipeline.learn(
            experience_from_result("goal", "it worked", success=True, approach="a", importance=0.7)
        )
        reopened = _memory(data_root)
        assert reopened.get(result.memory_id) is not None

    def test_single_success_does_not_promote(self, data_root):
        memory = _memory(data_root)
        pipeline = ExperienceLearningPipeline(memory, store=_store(data_root), min_support=2)
        result = pipeline.learn(
            experience_from_result("goal", "worked", success=True, approach="a", importance=0.7)
        )
        assert not result.promoted

    def test_repeated_success_promotes(self, data_root):
        memory = _memory(data_root)
        pipeline = ExperienceLearningPipeline(memory, store=_store(data_root), min_support=2)
        pipeline.learn(experience_from_result("goal", "worked", success=True, approach="a", importance=0.7))
        result = pipeline.learn(experience_from_result("goal", "worked again", success=True, approach="a", importance=0.7))
        assert result.promoted

    def test_failure_is_a_caution_not_a_rule(self, data_root):
        memory = _memory(data_root)
        pipeline = ExperienceLearningPipeline(memory, store=_store(data_root), min_support=1)
        result = pipeline.learn(
            experience_from_result("goal", "it failed", success=False, errors=("bad",), importance=0.8)
        )
        assert not result.promoted
        assert not result.lesson.success

    def test_lesson_is_rebuilt_from_memory(self, data_root):
        memory = _memory(data_root)
        pipeline = ExperienceLearningPipeline(memory, store=_store(data_root))
        pipeline.learn(experience_from_result("goal", "worked", success=True, approach="a", importance=0.7))
        pipeline.learn(experience_from_result("goal", "worked again", success=True, approach="a", importance=0.7))
        rebuilt = ExperienceLearningPipeline(_memory(data_root), store=_store(data_root))
        lessons = rebuilt.lessons_from_memory()
        assert len(lessons) == 1
        assert lessons[0].support == 2

    def test_confidence_is_never_one_from_one_success(self, data_root):
        memory = _memory(data_root)
        pipeline = ExperienceLearningPipeline(memory, store=_store(data_root))
        result = pipeline.learn(experience_from_result("goal", "worked", success=True, importance=0.7))
        assert result.lesson.confidence < 1.0

    def test_low_importance_is_not_learned_from(self, data_root):
        memory = _memory(data_root)
        pipeline = ExperienceLearningPipeline(memory, store=_store(data_root))
        result = pipeline.learn(
            experience_from_result("goal", "worked", success=True, importance=0.1)
        )
        assert result.lesson is None

    def test_empty_goal_is_refused(self):
        with pytest.raises(ValueError):
            ExperienceRecord(goal="  ", outcome="x")

    def test_outcome_vocabulary_infers_failure(self):
        assert experience_from_result("g", "the pump failed").success is False
        assert experience_from_result("g", "it worked").success is True


# -- feedback processing ----------------------------------------------------
class TestFeedbackProcessing:
    def test_processes_explicit_feedback(self, data_root):
        processor = FeedbackProcessor(_memory(data_root), store=_store(data_root))
        outcome = processor.process("correct", "answer-1", content="porosity increases permeability")
        assert outcome.accepted
        assert outcome.record.explicit

    def test_never_overrides_evidence(self, data_root):
        processor = FeedbackProcessor(_memory(data_root), store=_store(data_root))
        outcome = processor.process("incorrect", "answer-1", evidence_strength=0.95)
        assert not outcome.overrides_evidence

    def test_inferred_is_bounded_below_explicit(self, data_root):
        processor = FeedbackProcessor(_memory(data_root), store=_store(data_root))
        explicit = processor.process("correct", "a", confidence=0.95)
        inferred = processor.process("correct", "b", source_kind=FEEDBACK_INFERRED, confidence=0.95)
        assert inferred.record.weight < explicit.record.weight
        assert inferred.record.weight <= INFERRED_MAX_WEIGHT

    def test_weight_is_bounded(self, data_root):
        processor = FeedbackProcessor(_memory(data_root), store=_store(data_root))
        outcome = processor.process("correct", "a", confidence=1.0)
        assert outcome.record.weight <= DEFAULT_MAX_WEIGHT

    def test_correction_requires_the_corrected_text(self):
        with pytest.raises(ValueError):
            FeedbackRecord(kind="correction", subject="a")

    def test_silence_is_not_approval(self):
        with pytest.raises(ValueError):
            feedback_from_reaction("", "a")
        with pytest.raises(ValueError):
            feedback_from_reaction("maybe", "a")

    def test_known_reaction_maps(self):
        assert feedback_from_reaction("wrong", "a").kind == "incorrect"

    def test_retraction_supersedes(self, data_root):
        processor = FeedbackProcessor(_memory(data_root), store=_store(data_root))
        outcome = processor.process("correct", "a")
        processor.retract(outcome.record.feedback_id, reason="changed my mind")
        assert len(processor.active()) == 0

    def test_counts_and_correction_rate(self, data_root):
        processor = FeedbackProcessor(_memory(data_root), store=_store(data_root))
        processor.process("correct", "a")
        processor.process("correction", "b", correction="the right answer")
        assert processor.counts()["correction"] == 1
        assert processor.correction_rate() == 0.5


# -- pattern and procedure learning -----------------------------------------
class TestPatternLearning:
    def _seed(self, memory, store, count=3, success=True):
        pipeline = ExperienceLearningPipeline(memory, store=store)
        for index in range(count):
            pipeline.learn(
                experience_from_result(
                    "increase permeability",
                    f"attempt {index}",
                    success=success,
                    approach="acid",
                    importance=0.7,
                )
            )

    def test_promotes_a_corroborated_pattern(self, data_root):
        memory = _memory(data_root)
        store = _store(data_root)
        self._seed(memory, store, 3)
        learning = PatternLearning(memory, store=store, min_support=3)
        report = learning.extract()
        assert report.promoted
        assert learning.trusted_procedures()

    def test_a_single_experience_is_not_a_pattern(self, data_root):
        memory = _memory(data_root)
        store = _store(data_root)
        self._seed(memory, store, 1)
        learning = PatternLearning(memory, store=store)
        assert learning.extract().found == ()

    def test_records_applicability_and_failure_modes(self, data_root):
        memory = _memory(data_root)
        store = _store(data_root)
        pipeline = ExperienceLearningPipeline(memory, store=store)
        for index in range(3):
            pipeline.learn(
                experience_from_result(
                    "increase permeability", f"ok {index}", success=True, approach="acid", importance=0.7, scope="reservoir"
                )
            )
        pipeline.learn(
            experience_from_result(
                "increase permeability", "failed", success=False, errors=("pump failed",), importance=0.7, scope="reservoir"
            )
        )
        learning = PatternLearning(memory, store=store, min_support=3)
        learning.extract()
        procedure = learning.procedures()[0]
        assert "reservoir" in procedure.applicability
        assert "pump failed" in procedure.failure_modes

    def test_weak_support_is_only_a_proposal(self, data_root):
        memory = _memory(data_root)
        store = _store(data_root)
        self._seed(memory, store, 2)
        learning = PatternLearning(memory, store=store, min_support=3)
        learning.extract()
        assert learning.procedures()[0].status != STATUS_TRUSTED

    def test_revision_keeps_history(self, data_root):
        memory = _memory(data_root)
        store = _store(data_root)
        learning = PatternLearning(memory, store=store, min_support=2)
        self._seed(memory, store, 2)
        learning.extract()
        first_version = learning.procedures()[0].version
        self._seed(memory, store, 1)
        learning.extract()
        assert learning.procedures()[0].version > first_version

    def test_procedure_round_trips(self):
        procedure = LearnedProcedure(procedure_id="P", name="acid", steps=("a",), status=STATUS_TRUSTED)
        restored = LearnedProcedure.from_primitive(procedure.to_primitive())
        assert restored.name == "acid"
        assert restored.status == STATUS_TRUSTED

    def test_new_procedure_starts_proposed(self):
        procedure = LearnedProcedure(procedure_id="P", name="acid")
        assert procedure.status == STATUS_PROPOSED
        assert not procedure.trusted


# -- the coordinator --------------------------------------------------------
class TestLearningCoordinator:
    def test_builds_one_owner_store(self, data_root):
        coordinator = coordinator_for("fayd_al_ruh", data_root=data_root)
        assert coordinator.owner == "fayd_al_ruh"
        assert coordinator.store.path.exists()

    def test_status_reflects_learning(self, data_root):
        coordinator = coordinator_for("fayd_al_ruh", data_root=data_root)
        coordinator.learn_claim(
            claim_from_relation("a", "increases", "b", confidence=0.8, source_kind="document"), source="document"
        )
        status = coordinator.status()
        assert status.events >= 1
        assert status.counts[KIND_KNOWLEDGE] == 1

    def test_owners_have_distinct_stores(self, data_root):
        jarvis = coordinator_for("jarvis", data_root=data_root)
        natasha = coordinator_for("natasha", data_root=data_root)
        assert jarvis.store.path != natasha.store.path

    def test_main_model_may_not_learn_from_natasha(self, data_root):
        main = coordinator_for("fayd_al_ruh", data_root=data_root)
        coordinator_for("natasha", data_root=data_root)
        with pytest.raises(LearningAccessError):
            main.learn_from("natasha")

    def test_character_may_not_learn_from_another(self, data_root):
        jarvis = coordinator_for("jarvis", data_root=data_root)
        with pytest.raises(LearningAccessError):
            jarvis.learn_from("ultron")

    def test_learning_from_a_shareable_character_copies_knowledge_only(self, data_root):
        jarvis = coordinator_for("jarvis", data_root=data_root)
        record = jarvis.memory.capture(
            "porosity measures void space",
            kind=MemoryKind.SEMANTIC,
            summary="porosity",
            confidence=0.8,
            importance=0.7,
        )
        MemoryManager(jarvis.memory).verify(record.record_id)
        main = coordinator_for("fayd_al_ruh", data_root=data_root)
        events = main.learn_from("jarvis")
        assert events
        copied = main.memory.page(kind=MemoryKind.SEMANTIC)[0][0]
        assert copied.metadata.get("shared_knowledge_only") is True

    def test_natasha_is_excluded_from_the_dataset(self, data_root):
        main = coordinator_for("fayd_al_ruh", data_root=data_root)
        assert "natasha" not in main.dataset_owners()
        assert "natasha" in main.dataset_summary()

    def test_rollback_returns_events(self, data_root):
        coordinator = coordinator_for("fayd_al_ruh", data_root=data_root)
        coordinator.learn_claim(
            claim_from_relation("a", "increases", "b", confidence=0.8, source_kind="document"), source="document"
        )
        reversals = coordinator.rollback(1)
        assert reversals

    def test_default_coordinator_is_the_main_model(self, data_root):
        coordinator = build_default_coordinator(data_root=data_root)
        assert coordinator.owner == "fayd_al_ruh"

    def test_unknown_owner_is_refused(self, data_root):
        with pytest.raises(LearningAccessError):
            coordinator_for("not-a-character", data_root=data_root)


class TestLearningEventProcessor:
    def test_drain_is_incremental(self, data_root):
        coordinator = coordinator_for("fayd_al_ruh", data_root=data_root)
        coordinator.learn_claim(
            claim_from_relation("a", "increases", "b", confidence=0.8, source_kind="document"), source="document"
        )
        processor = LearningEventProcessor()
        first = processor.drain(coordinator)
        assert first
        assert processor.drain(coordinator) == ()

    def test_reset_replays(self, data_root):
        coordinator = coordinator_for("fayd_al_ruh", data_root=data_root)
        coordinator.learn_claim(
            claim_from_relation("a", "increases", "b", confidence=0.8, source_kind="document"), source="document"
        )
        processor = LearningEventProcessor()
        processor.drain(coordinator)
        processor.reset(coordinator.owner)
        assert processor.drain(coordinator)


# -- learning evaluation ----------------------------------------------------
class TestLearningEvaluation:
    def test_measures_an_improvement(self, data_root):
        coordinator = coordinator_for("fayd_al_ruh", data_root=data_root)
        report = evaluate_learning(
            coordinator,
            lambda: coordinator.learn_claim(
                claim_from_relation("a", "increases", "b", confidence=0.8, source_kind="document"), source="document"
            ),
            name="learn a claim",
        )
        assert report.improved
        assert report.comparison.deltas["knowledge_events"] == 1

    def test_unchanged_is_reported_as_unchanged(self, data_root):
        coordinator = coordinator_for("fayd_al_ruh", data_root=data_root)
        report = evaluate_learning(coordinator, lambda: None, name="no-op")
        assert not report.comparison.changed
        assert not report.improved

    def test_snapshot_reads_the_real_stores(self, data_root):
        coordinator = coordinator_for("fayd_al_ruh", data_root=data_root)
        before = snapshot(coordinator)
        coordinator.learn_claim(
            claim_from_relation("a", "increases", "b", confidence=0.8, source_kind="document"), source="document"
        )
        after = snapshot(coordinator)
        assert after.get("knowledge_events") > before.get("knowledge_events")

    def test_compare_reports_regressions(self):
        before = snapshot.__wrapped__ if hasattr(snapshot, "__wrapped__") else None  # noqa: F841
        from core.learning_evaluation import LearningSnapshot

        first = LearningSnapshot(metrics={"a": 5.0}, version=2)
        second = LearningSnapshot(metrics={"a": 3.0}, version=3)
        comparison = compare(first, second)
        assert comparison.regressed == ("a",)

    def test_metrics_and_history_helpers(self, data_root):
        coordinator = coordinator_for("fayd_al_ruh", data_root=data_root)
        coordinator.learn_claim(
            claim_from_relation("a", "increases", "b", confidence=0.8, source_kind="document"), source="document"
        )
        assert learning_metrics(coordinator)["knowledge_events"] == 1.0
        assert "event(s)" in history_summary(coordinator)
        assert version_history(coordinator)


# -- the controller wiring --------------------------------------------------
class TestControllerWiring:
    def test_learning_is_lazy(self):
        controller = CognitiveController()
        assert controller.learning_status()["built"] is False

    def test_learns_through_the_controller(self, data_root):
        controller = CognitiveController()
        controller.learning(data_root=str(data_root))
        controller.learn_claim(
            claim_from_relation("porosity", "increases", "permeability", confidence=0.8, source_kind="document"),
            source="document",
        )
        assert controller.learning_status()["built"] is True
        assert controller.learning_status()["counts"][KIND_KNOWLEDGE] == 1

    def test_learning_uses_the_controllers_own_graph(self, data_root):
        controller = CognitiveController()
        controller.learning(data_root=str(data_root))
        controller.learn_claim(
            claim_from_relation("porosity", "increases", "permeability", confidence=0.8, source_kind="document"),
            source="document",
        )
        assert controller.learning().knowledge.graph is controller.graph

    def test_learning_publishes_an_event(self, data_root):
        controller = CognitiveController()
        controller.learning(data_root=str(data_root))
        seen = []
        controller.events.subscribe(lambda event: seen.append(event.name), prefix="learning")
        controller.learn_claim(
            claim_from_relation("a", "increases", "b", confidence=0.8, source_kind="document"), source="document"
        )
        assert "learning.recorded" in seen


# -- the held-out gate ------------------------------------------------------
class TestLearningGate:
    def test_gate_is_perfect_on_the_shipped_architecture(self):
        assert learning_accuracy() == 1.0

    def test_gate_report_names_every_judgement(self):
        report = learning_gate_report()
        assert report.total == len(LEARNING_JUDGEMENTS)
        assert not report.failed

    def test_empty_catalogue_scores_zero(self):
        assert learning_accuracy(()) == 0.0

    def test_gate_discriminates(self):
        """A broken capability lowers the metric -- the gate has teeth."""
        from core.learning_gate import LEARNING_JUDGEMENTS as judgements

        def _broken(_root) -> bool:
            return False

        sabotaged = tuple(
            (name, _broken if name == "no-premature-generalisation" else check)
            for name, check in judgements
        )
        assert learning_accuracy(sabotaged) < 1.0

    def test_gate_has_negative_controls(self):
        names = {name for name, _ in LEARNING_JUDGEMENTS}
        assert "knowledge-refuses-unsupported-claim" in names
        assert "natasha-is-excluded" in names
        assert "silence-is-not-approval" in names
        assert "feedback-never-overrides-evidence" in names
