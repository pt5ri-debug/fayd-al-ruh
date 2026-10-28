"""Learned graph knowledge survives a session boundary.

The project carried a whole persistence layer -- ``KnowledgeStore``,
``Snapshot``, ``Persistence``, ``JsonStore`` -- but it was never wired to the
answer path: every session rebuilt the graph from the seeds, so a relation the
architecture **learned** (from a confirmed statement, an ingested document) was
gone when the process ended. These tests pin :mod:`core.knowledge_carry` -- the
bridge that captures the graph's *learned* relations and replays them into a later
session's ontology -- and the held-out metric that proves it, with negative
controls for a curated seed and a record naming an unknown concept.
"""

from __future__ import annotations

import json

import pytest
from core.assistant import build_default_assistant
from core.knowledge_carry import (
    CARRY_JUDGEMENTS,
    CARRY_KEY,
    CARRY_NAMESPACE,
    CarryJudgement,
    LearnedKnowledgeStore,
    LearnedRelation,
    capture_learned,
    knowledge_carry_accuracy,
    restore_learned,
)
from core.ontology_learner import OntologyLearner
from storage.json_store import JsonStore


def _fresh_ontology():
    """A seeded ontology built the way the answer path builds one."""
    from core.curriculum_knowledge import seed_curriculum
    from core.domain_knowledge import seed_ontology
    from core.ontology import Ontology
    from core.petroleum_corpus import seed_petroleum_corpus

    ontology = Ontology()
    seed_ontology(ontology)
    seed_curriculum(ontology)
    seed_petroleum_corpus(ontology)
    return ontology


def _relation_present(ontology, learner, source, relation_type, target):
    source_concepts = ontology.find_concepts(source)
    target_concepts = ontology.find_concepts(target)
    if not source_concepts or not target_concepts:
        return False
    return (
        learner.relation_by_triple(
            source_concepts[0].id, relation_type, target_concepts[0].id
        )
        is not None
    )


# --------------------------------------------------------------------------- #
# LearnedRelation: the record
# --------------------------------------------------------------------------- #


def test_a_relation_record_rejects_a_bad_confidence():
    with pytest.raises(ValueError):
        LearnedRelation("a", "increases", "b", 1.5)


def test_a_relation_record_needs_two_labels_and_a_type():
    with pytest.raises(ValueError):
        LearnedRelation("", "increases", "b", 0.5)
    with pytest.raises(ValueError):
        LearnedRelation("a", "", "b", 0.5)


def test_a_relation_record_round_trips_through_its_primitive():
    record = LearnedRelation("waterflood", "increases", "water_saturation", 0.45)
    again = LearnedRelation.from_primitive(record.to_primitive())
    assert again == record


def test_the_record_key_is_case_folded():
    a = LearnedRelation("Waterflood", "Increases", "Water_Saturation", 0.4)
    b = LearnedRelation("waterflood", "increases", "water_saturation", 0.4)
    assert a.key == b.key


# --------------------------------------------------------------------------- #
# capture: learned only, never a seed
# --------------------------------------------------------------------------- #


def test_capture_includes_a_learned_relation():
    ontology = _fresh_ontology()
    learner = OntologyLearner(ontology)
    learner.observe_relation("waterflood", "water_saturation", "increases", 0.5)
    records = capture_learned(ontology, learner)
    assert any(
        record.source == "waterflood"
        and record.relation_type == "increases"
        and record.target == "water_saturation"
        for record in records
    )


def test_capture_excludes_a_curated_seed():
    ontology = _fresh_ontology()
    learner = OntologyLearner(ontology)
    # "porosity increases permeability" is a curated seed, never learned.
    records = capture_learned(ontology, learner)
    triples = {(record.source, record.relation_type, record.target) for record in records}
    assert ("porosity", "increases", "permeability") not in triples
    assert records == ()


def test_capture_is_empty_on_an_untouched_graph():
    ontology = _fresh_ontology()
    learner = OntologyLearner(ontology)
    assert capture_learned(ontology, learner) == ()


# --------------------------------------------------------------------------- #
# restore: replay through the learner, idempotent, never invents
# --------------------------------------------------------------------------- #


def test_a_learned_relation_is_replayed_into_a_fresh_session():
    first = _fresh_ontology()
    first_learner = OntologyLearner(first)
    first_learner.observe_relation("waterflood", "water_saturation", "increases", 0.5)
    records = capture_learned(first, first_learner)

    second = _fresh_ontology()
    second_learner = OntologyLearner(second)
    report = restore_learned(second, second_learner, records)
    assert report.changed
    assert _relation_present(second, second_learner, "waterflood", "increases", "water_saturation")


def test_restoring_twice_does_not_duplicate():
    first = _fresh_ontology()
    first_learner = OntologyLearner(first)
    first_learner.observe_relation("waterflood", "water_saturation", "increases", 0.5)
    records = capture_learned(first, first_learner)

    second = _fresh_ontology()
    second_learner = OntologyLearner(second)
    restore_learned(second, second_learner, records)
    count = len(second.relations)
    again = restore_learned(second, second_learner, records)
    assert not again.changed
    assert len(second.relations) == count


def test_a_record_naming_an_unknown_concept_is_skipped_not_invented():
    ontology = _fresh_ontology()
    learner = OntologyLearner(ontology)
    before = len(ontology.concepts)
    record = LearnedRelation("waterflood", "increases", "not_a_real_concept_xyz", 0.5)
    report = restore_learned(ontology, learner, (record,))
    assert not report.restored
    assert record in report.skipped
    assert len(ontology.concepts) == before


def test_restore_never_overrides_a_curated_seed():
    ontology = _fresh_ontology()
    learner = OntologyLearner(ontology)
    # A record that names a seeded pair: the relation already exists, so the
    # restore counts it as present and never rewrites it.
    source = ontology.find_concepts("porosity")[0]
    target = ontology.find_concepts("permeability")[0]
    seeded = learner.relation_by_triple(source.id, "increases", target.id)
    assert seeded is not None
    before = seeded.confidence
    record = LearnedRelation("porosity", "increases", "permeability", 0.9)
    report = restore_learned(ontology, learner, (record,))
    assert not report.restored
    assert report.already_present == 1
    after = learner.relation_by_triple(source.id, "increases", target.id)
    assert after is not None and after.confidence == before


# --------------------------------------------------------------------------- #
# LearnedKnowledgeStore: the durable wrapper
# --------------------------------------------------------------------------- #


def test_the_store_round_trips_through_json(tmp_path):
    store = LearnedKnowledgeStore(JsonStore(tmp_path))
    records = (LearnedRelation("waterflood", "increases", "water_saturation", 0.45),)
    store.save(records)
    assert store.load() == records


def test_the_store_writes_under_the_one_namespace(tmp_path):
    store = LearnedKnowledgeStore(JsonStore(tmp_path))
    store.save((LearnedRelation("waterflood", "increases", "water_saturation", 0.45),))
    payload = JsonStore(tmp_path).load(CARRY_NAMESPACE, CARRY_KEY)
    assert payload is not None
    assert payload["relations"][0]["source"] == "waterflood"


def test_an_empty_store_loads_nothing(tmp_path):
    store = LearnedKnowledgeStore(JsonStore(tmp_path))
    assert store.load() == ()


def test_the_store_drops_an_unreadable_record(tmp_path):
    json_store = JsonStore(tmp_path)
    json_store.save(
        CARRY_NAMESPACE,
        CARRY_KEY,
        {
            "version": 1,
            "relations": [
                {"source": "waterflood", "relation_type": "increases", "target": "x", "confidence": 0.5},
                {"source": "waterflood"},  # missing fields
                "not a dict",
            ],
        },
    )
    records = LearnedKnowledgeStore(json_store).load()
    assert len(records) == 1
    assert records[0].source == "waterflood"


def test_the_store_capture_then_restore_is_a_full_round_trip(tmp_path):
    store = LearnedKnowledgeStore(JsonStore(tmp_path))
    first = _fresh_ontology()
    first_learner = OntologyLearner(first)
    first_learner.observe_relation("waterflood", "water_saturation", "increases", 0.5)
    store.capture(first, first_learner)

    second = _fresh_ontology()
    second_learner = OntologyLearner(second)
    report = store.restore(second, second_learner)
    assert report.changed
    assert _relation_present(second, second_learner, "waterflood", "increases", "water_saturation")


# --------------------------------------------------------------------------- #
# The assistant wires the carry into the answer path
# --------------------------------------------------------------------------- #


def test_the_assistant_carries_a_learned_relation_to_the_next_session(tmp_path):
    store = LearnedKnowledgeStore(JsonStore(tmp_path))
    first = build_default_assistant(learned=store)
    first.remember_relation("waterflood", "increases", "water_saturation", 0.5)

    second = build_default_assistant(learned=store)
    ontology = second.retriever.ontology
    source = ontology.find_concepts("waterflood")
    target = ontology.find_concepts("water_saturation")
    assert source and target
    assert any(
        relation.source_id == source[0].id
        and relation.target_id == target[0].id
        and relation.relation_type == "increases"
        for relation in ontology.relations.values()
    )


def test_the_assistant_without_a_store_stays_read_only(tmp_path):
    # No carry store: learning must not touch disk and must not raise.
    assistant = build_default_assistant()
    assistant.remember_relation("waterflood", "increases", "water_saturation", 0.5)
    assert assistant.learned is None
    assert assistant.recall_learned().restored == ()


def test_learn_from_text_reads_a_typed_relation_and_carries_it(tmp_path):
    store = LearnedKnowledgeStore(JsonStore(tmp_path))
    assistant = build_default_assistant(learned=store)
    learning = assistant.learn_from_text("Waterflood increases water saturation.")
    assert learning.changed
    assert ("waterflood", "increases", "saturation") in learning.added
    # And it is now carried for the next session.
    assert store.load()


def test_learn_from_text_with_no_relation_learns_nothing(tmp_path):
    assistant = build_default_assistant()
    learning = assistant.learn_from_text("The weather is pleasant today.")
    assert not learning.changed


def test_the_learner_is_reused_so_repeated_teaching_accumulates(tmp_path):
    store = LearnedKnowledgeStore(JsonStore(tmp_path))
    assistant = build_default_assistant(learned=store)
    assistant.remember_relation("waterflood", "increases", "water_saturation", 0.5)
    first = len(store.load())
    assistant.remember_relation("waterflood", "increases", "water_saturation", 0.5)
    # Teaching the same relation again does not duplicate the record.
    assert len(store.load()) == first


# --------------------------------------------------------------------------- #
# The held-out metric, with negative controls
# --------------------------------------------------------------------------- #


def test_the_carry_metric_is_perfect_on_the_catalogue():
    assert knowledge_carry_accuracy() == 1.0


def test_the_carry_metric_rejects_a_bridge_that_captures_a_seed():
    # A "bridge" that captured every relation would report a seed as learned.
    judgements = (
        CarryJudgement(
            "a curated seed relation is never captured as learned",
            "seed-kept",
            source="porosity",
            relation_type="increases",
            target="permeability",
        ),
    )
    # The real bridge never captures it, so the judgement holds.
    assert knowledge_carry_accuracy(judgements) == 1.0


def test_the_carry_metric_is_zero_for_an_empty_catalogue():
    assert knowledge_carry_accuracy(()) == 0.0


def test_every_carry_judgement_kind_is_exercised():
    kinds = {judgement.kind for judgement in CARRY_JUDGEMENTS}
    assert kinds == {"round-trip", "idempotent", "seed-kept", "skip-unknown"}


# --------------------------------------------------------------------------- #
# The CLI carries the learned relation across two processes
# --------------------------------------------------------------------------- #


def test_the_carry_cli_teaches_then_shows_across_two_runs(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from cli import relations as relations_cli

    code = relations_cli.main(["carry", "Waterflood increases water saturation."])
    assert code == relations_cli.EXIT_OK
    out = capsys.readouterr().out
    assert "waterflood --increases--> saturation" in out
    assert "Carried 1 learned relation(s)" in out

    # A second, independent invocation must see what the first carried.
    code = relations_cli.main(["carry", "--show"])
    assert code == relations_cli.EXIT_OK
    assert "Carried 1 learned relation(s)" in capsys.readouterr().out


def test_the_carry_cli_json_reports_the_records(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from cli import relations as relations_cli

    relations_cli.main(["carry", "Waterflood increases water saturation."])
    capsys.readouterr()
    code = relations_cli.main(["carry", "--show", "--json"])
    assert code == relations_cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["carried"][0]["source"] == "waterflood"
    assert payload["carried"][0]["relation_type"] == "increases"


def test_the_carry_cli_with_no_input_is_a_usage_error(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from cli import relations as relations_cli

    assert relations_cli.main(["carry"]) == relations_cli.EXIT_NOTHING


# --------------------------------------------------------------------------- #
# Jarvis feedback teaches a typed relation (and carries it)
# --------------------------------------------------------------------------- #


def test_jarvis_feedback_teaches_and_carries_a_relation(tmp_path):
    from core.command_router import NaturalLanguageRouter
    from core.continuous_learning import ContinuousLearner
    from core.documents import build_document_store
    from core.jarvis import JarvisMode
    from core.knowledge_update import KnowledgeUpdater
    from core.tools import build_sandbox

    store = LearnedKnowledgeStore(JsonStore(tmp_path / "learned"))
    documents = build_document_store()
    assistant = build_default_assistant(documents=documents, learned=store)
    jarvis = JarvisMode(
        assistant,
        router=NaturalLanguageRouter(build_sandbox(tmp_path)),
        learner=ContinuousLearner(KnowledgeUpdater(documents)),
    )
    # The user corrects the last answer with a typed relation.
    jarvis.handle("what is porosity?", speak=False)
    jarvis.handle("that's wrong, it's waterflood increases water saturation", speak=False)
    # The typed relation was read and carried for the next session.
    assert store.load()
    assert any(record.source == "waterflood" for record in store.load())


def test_jarvis_without_a_learner_refuses_feedback(tmp_path):
    from core.command_router import NaturalLanguageRouter
    from core.jarvis import JarvisMode
    from core.tools import build_sandbox

    jarvis = JarvisMode(
        build_default_assistant(),
        router=NaturalLanguageRouter(build_sandbox(tmp_path)),
    )
    jarvis.handle("what is porosity?", speak=False)
    turn = jarvis.handle("that's right", speak=False)
    assert "not enabled" in turn.response.lower()

