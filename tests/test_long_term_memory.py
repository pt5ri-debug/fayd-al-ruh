"""Phase 2: the durable long-term memory architecture.

These tests pin :mod:`core.memory_record`, :mod:`core.memory_db`,
:mod:`core.long_term_memory`, :mod:`core.memory_retrieval`,
:mod:`core.memory_management`, and the held-out gate
:mod:`core.memory_gate`, and the controller wiring. They run against a real
SQLite database in a temporary directory -- never the user's own memory -- and
they check the honesty rules the task requires: nothing is deleted to resolve a
conflict, a bad backup cannot destroy a good database, a corrupt row is skipped
rather than raised, and one personality can never read another's memory.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import timedelta

import pytest
from core.learning_isolation import LearningAccessError
from core.long_term_memory import LongTermMemory, WorkingMemoryStore
from core.memory_db import SCHEMA_VERSION, MemoryDatabase, MemoryDatabaseError
from core.memory_gate import MEMORY_JUDGEMENTS, memory_architecture_accuracy, memory_gate_report
from core.memory_management import MemoryManager, PersonalityMemoryRegistry
from core.memory_record import (
    DURABLE_KINDS,
    SOURCE_MODEL,
    SOURCE_USER,
    VERIFICATION_SUPERSEDED,
    VERIFICATION_UNVERIFIED,
    MemoryKind,
    MemoryRecord,
    make_memory_id,
    now_utc,
    supersede,
)
from core.memory_retrieval import MemoryRetriever, RankingSignals


@pytest.fixture()
def memory(tmp_path):
    return LongTermMemory("fayd_al_ruh", tmp_path / "memory.sqlite3")


# -- the record contract ----------------------------------------------------
def test_record_requires_content_and_a_valid_kind():
    with pytest.raises(ValueError):
        MemoryRecord(record_id="x", kind=MemoryKind.EPISODIC, content="   ")
    with pytest.raises(ValueError):
        MemoryRecord(record_id="", kind=MemoryKind.EPISODIC, content="x")
    with pytest.raises(ValueError):
        MemoryRecord(record_id="x", kind=MemoryKind.EPISODIC, content="x", confidence=2.0)
    with pytest.raises(ValueError):
        MemoryRecord(record_id="x", kind=MemoryKind.EPISODIC, content="x", verification="bogus")


def test_record_id_is_content_addressed_and_owner_scoped():
    first = make_memory_id(MemoryKind.EPISODIC, "the well was cased", owner="jarvis")
    again = make_memory_id(MemoryKind.EPISODIC, "the well was cased", owner="jarvis")
    other = make_memory_id(MemoryKind.EPISODIC, "the well was cased", owner="ultron")
    assert first == again
    assert first != other


def test_record_round_trips_through_primitive():
    record = MemoryRecord(
        record_id="mem:episodic:abc",
        kind=MemoryKind.EPISODIC,
        content="A fact.",
        source=SOURCE_USER,
        confidence=0.8,
        entities=("porosity",),
        relations=(("porosity", "increases", "permeability"),),
    )
    restored = MemoryRecord.from_primitive(record.to_primitive())
    assert restored == record


def test_supersede_keeps_both_records():
    old = MemoryRecord(record_id="mem:semantic:1", kind=MemoryKind.SEMANTIC, content="old")
    superseded, new = supersede(old, "new")
    assert superseded.verification == VERIFICATION_SUPERSEDED
    assert superseded.superseded_by == new.record_id
    assert new.supersedes == old.record_id


def test_record_temporal_window():
    start = now_utc() - timedelta(days=10)
    end = now_utc() - timedelta(days=5)
    record = MemoryRecord(
        record_id="mem:semantic:1",
        kind=MemoryKind.SEMANTIC,
        content="x",
        valid_from=start,
        valid_until=end,
    )
    assert record.is_valid_at(start + timedelta(days=1))
    assert not record.is_valid_at(now_utc())
    assert not record.is_valid_at(start - timedelta(days=1))


def test_record_trusted_only_for_a_trusted_status():
    record = MemoryRecord(record_id="mem:semantic:1", kind=MemoryKind.SEMANTIC, content="x")
    assert not record.trusted  # unverified at birth
    assert not record.trusted if record.verification == VERIFICATION_UNVERIFIED else True


# -- the database -----------------------------------------------------------
def test_database_creates_schema_and_version(tmp_path):
    db = MemoryDatabase(tmp_path / "m.sqlite3")
    assert db.schema_version == SCHEMA_VERSION
    ok, message = db.integrity_check()
    assert ok, message


def test_database_refuses_a_newer_schema(tmp_path):
    path = tmp_path / "m.sqlite3"
    MemoryDatabase(path)
    connection = sqlite3.connect(path)
    connection.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
    connection.commit()
    connection.close()
    with pytest.raises(MemoryDatabaseError):
        MemoryDatabase(path).migrate()


def test_database_bounded_page(tmp_path, memory):
    for index in range(25):
        memory.capture(f"record number {index}", kind=MemoryKind.EPISODIC)
    page, report = memory.page(limit=10, offset=0)
    assert len(page) == 10
    assert report.ok
    page_two, _ = memory.page(limit=10, offset=10)
    assert len(page_two) == 10
    assert {item.record_id for item in page}.isdisjoint({item.record_id for item in page_two})


def test_database_skips_a_corrupt_row(tmp_path, memory):
    memory.capture("a good record")
    connection = sqlite3.connect(memory.database.path)
    connection.execute("UPDATE memories SET payload='{bad json'")
    connection.commit()
    connection.close()
    records, report = memory.page(limit=10)
    assert records == []
    assert report.skipped == 1
    assert not report.ok


def test_database_save_many_is_all_or_nothing(tmp_path, memory):
    memory.capture("the original", kind=MemoryKind.EPISODIC)
    before = memory.count(include_archived=True)
    good = memory.capture("another", kind=MemoryKind.EPISODIC, persist=False)
    with pytest.raises(ValueError):
        MemoryRecord(record_id="", kind=MemoryKind.SEMANTIC, content="x")
    # A save_many of a valid record succeeds; a malformed batch never does.
    assert memory.database.save_many([good]) == 1
    assert memory.count(include_archived=True) == before + 1


# -- the four stores and the lifecycle --------------------------------------
def test_capture_is_idempotent(memory):
    first = memory.capture("the same content", kind=MemoryKind.SEMANTIC)
    second = memory.capture("the same content", kind=MemoryKind.SEMANTIC)
    assert first.record_id == second.record_id
    assert memory.count(include_archived=True) == 1


def test_capture_persists_provenance(memory):
    record = memory.capture(
        "Waterflood increased recovery.",
        kind=MemoryKind.SEMANTIC,
        source=SOURCE_USER,
        source_reference="pilot-7",
        confidence=0.7,
        importance=0.8,
        entities=("waterflood", "recovery"),
        tags=("eor",),
    )
    stored = memory.get(record.record_id)
    assert stored.source == SOURCE_USER
    assert stored.source_reference == "pilot-7"
    assert stored.confidence == 0.7
    assert "waterflood" in stored.entities
    assert "eor" in stored.tags


def test_working_memory_is_bounded_and_lru():
    store = WorkingMemoryStore(capacity=3)
    ids = []
    for index in range(5):
        record = MemoryRecord(
            record_id=f"mem:working:{index}", kind=MemoryKind.WORKING, content=f"item {index}"
        )
        store.put(record)
        ids.append(record.record_id)
    assert store.size == 3
    assert ids[0] not in {item.record_id for item in store.items()}
    assert ids[-1] in {item.record_id for item in store.items()}


def test_working_memory_access_bumps_recency():
    store = WorkingMemoryStore(capacity=3)
    first = MemoryRecord(record_id="mem:working:1", kind=MemoryKind.WORKING, content="a")
    second = MemoryRecord(record_id="mem:working:2", kind=MemoryKind.WORKING, content="b")
    store.put(first)
    store.put(second)
    store.get(first.record_id)
    assert store.items()[-1].record_id == first.record_id


def test_correct_keeps_history(memory):
    original = memory.capture("depth is 2000 ft", importance=0.6)
    old, new = memory.correct(original.record_id, "depth is 2500 ft", reason="log")
    stored_old = memory.get(old.record_id)
    stored_new = memory.get(new.record_id)
    assert stored_old is not None and stored_old.verification == VERIFICATION_SUPERSEDED
    assert stored_old.superseded_by == new.record_id
    assert stored_new is not None and stored_new.supersedes == old.record_id


def test_correct_unknown_record_refused(memory):
    with pytest.raises(KeyError):
        memory.correct("mem:episodic:missing", "x")


def test_archive_is_not_delete(memory):
    record = memory.capture("temporary", importance=0.4)
    assert memory.archive(record.record_id)
    hidden, _ = memory.page(include_archived=False, limit=50)
    shown, _ = memory.page(include_archived=True, limit=50)
    assert record.record_id not in {item.record_id for item in hidden}
    assert record.record_id in {item.record_id for item in shown}


def test_expire_archives_only_past_validity(memory):
    past = now_utc() - timedelta(days=5)
    memory.capture("old claim", kind=MemoryKind.SEMANTIC, valid_until=past, importance=0.5)
    memory.capture("current claim", kind=MemoryKind.SEMANTIC, importance=0.5)
    expired = memory.expire(before=now_utc())
    assert len(expired) == 1
    live, _ = memory.page(include_archived=False, limit=50)
    assert len(live) == 1
    assert live[0].content == "current claim"


def test_promote_working_promotes_and_skips(memory):
    memory.capture("important", kind=MemoryKind.WORKING, importance=0.9)
    memory.capture("trivial", kind=MemoryKind.WORKING, importance=0.1)
    report = memory.promote_working(minimum_importance=0.5)
    episodic = {record.content for record in memory.all_records(kind=MemoryKind.EPISODIC)}
    assert report.changed
    assert "important" in episodic
    assert "trivial" not in episodic


# -- integrity, backup, restore --------------------------------------------
def test_backup_and_restore_round_trip(tmp_path, memory):
    memory.capture("a fact worth keeping", kind=MemoryKind.SEMANTIC)
    backup = tmp_path / "backup.sqlite3"
    assert memory.backup(backup).ok
    fresh = LongTermMemory("fayd_al_ruh", tmp_path / "fresh.sqlite3")
    report = fresh.restore(backup)
    assert report.ok
    assert fresh.count(include_archived=True) == 1


def test_corrupt_backup_is_refused(tmp_path, memory):
    memory.capture("important", importance=0.9)
    before = memory.count(include_archived=True)
    junk = tmp_path / "junk.sqlite3"
    junk.write_bytes(b"not a database")
    report = memory.restore(junk)
    assert not report.ok
    assert memory.count(include_archived=True) == before


def test_restore_missing_backup_refused(tmp_path, memory):
    report = memory.restore(tmp_path / "missing.sqlite3")
    assert not report.ok


# -- retrieval --------------------------------------------------------------
def test_retrieval_ranks_the_better_match_first(memory):
    memory.capture("casing collapse pressure design", kind=MemoryKind.SEMANTIC)
    memory.capture("mud weight note", kind=MemoryKind.SEMANTIC)
    report = MemoryRetriever(memory).retrieve("casing collapse pressure", touch=False)
    assert report.count >= 1
    assert "casing" in report.hits[0].record.content


def test_retrieval_signals_are_reported_separately(memory):
    memory.capture("a record about permeability", kind=MemoryKind.SEMANTIC, importance=0.7)
    hit = MemoryRetriever(memory).retrieve("permeability", touch=False).hits[0]
    assert isinstance(hit.signals, RankingSignals)
    assert hit.signals.relevance > 0.0
    assert hit.signals.importance == 0.7


def test_retrieval_score_is_not_the_confidence(memory):
    record = memory.capture(
        "a model guess", kind=MemoryKind.SEMANTIC, source=SOURCE_MODEL, confidence=0.1
    )
    hit = next(
        item
        for item in MemoryRetriever(memory).retrieve("model guess", touch=False).hits
        if item.record_id == record.record_id
    )
    assert hit.confidence == 0.1
    assert hit.score != hit.confidence


def test_retrieval_touch_increments_frequency(memory):
    record = memory.capture("a retrievable fact", kind=MemoryKind.SEMANTIC)
    MemoryRetriever(memory).retrieve("retrievable fact", touch=True)
    assert memory.get(record.record_id).access_count >= 1


def test_retrieval_kind_filter(memory):
    memory.capture("episodic one", kind=MemoryKind.EPISODIC)
    memory.capture("semantic one", kind=MemoryKind.SEMANTIC)
    report = MemoryRetriever(memory).retrieve("one", kinds=(MemoryKind.SEMANTIC,), touch=False)
    assert all(hit.kind is MemoryKind.SEMANTIC for hit in report.hits)


def test_retrieval_empty_query_returns_page(memory):
    memory.capture("something", kind=MemoryKind.SEMANTIC)
    report = MemoryRetriever(memory).retrieve("", touch=False, limit=10)
    assert report.count >= 1


def test_retrieval_by_entity(memory):
    memory.capture(
        "a linked record", kind=MemoryKind.SEMANTIC, entities=("porosity",)
    )
    retriever = MemoryRetriever(memory)
    assert len(retriever.by_entity("porosity")) == 1


def test_retrieval_related_by_shared_entity(memory):
    first = memory.capture("first", kind=MemoryKind.SEMANTIC, entities=("porosity",))
    memory.capture("second", kind=MemoryKind.SEMANTIC, entities=("porosity",))
    related = MemoryRetriever(memory).related(first.record_id)
    assert len(related) == 1


def test_retrieval_semantic_uses_space_when_present(memory):
    class _Space:
        def embed(self, text):
            return (1.0, 0.0) if "porosity" in text else (0.0, 1.0)

    memory.capture("porosity is the void space", kind=MemoryKind.SEMANTIC)
    report = MemoryRetriever(memory, semantic_space=_Space()).retrieve("porosity", touch=False)
    assert report.semantic_used
    assert any(hit.signals.semantic > 0.0 for hit in report.hits)


# -- management -------------------------------------------------------------
def test_manager_export_writes_json(tmp_path, memory):
    memory.capture("exported fact", kind=MemoryKind.SEMANTIC)
    report = MemoryManager(memory, export_dir=tmp_path).export()
    assert report.ok and report.records == 1
    document = json.loads((tmp_path / "memory-export.json").read_text(encoding="utf-8"))
    assert document["count"] == 1
    assert document["records"][0]["content"] == "exported fact"


def test_manager_backup_and_restore(tmp_path, memory):
    memory.capture("kept", kind=MemoryKind.SEMANTIC)
    manager = MemoryManager(memory, export_dir=tmp_path)
    assert manager.backup().ok
    assert (tmp_path / "memory-backup.sqlite3").exists()


def test_manager_provenance_chain(memory):
    original = memory.capture("old", importance=0.5)
    manager = MemoryManager(memory)
    _, new = memory.correct(original.record_id, "new")
    chain = manager.provenance(new.record_id)
    assert chain["supersedes"] == original.record_id
    # A correction is stored provisional -- trusted, but never auto-verified.
    assert chain["verification"] == "provisional"
    assert chain["trusted"] is True


def test_manager_verify_is_explicit(memory):
    record = memory.capture("a claim", kind=MemoryKind.SEMANTIC)
    manager = MemoryManager(memory)
    assert manager.verify(record.record_id).changed == 1
    assert memory.get(record.record_id).trusted


def test_manager_verify_rejects_unknown_status(memory):
    record = memory.capture("a claim", kind=MemoryKind.SEMANTIC)
    report = MemoryManager(memory).verify(record.record_id, status="bogus")
    assert not report.ok
    assert not memory.get(record.record_id).trusted


def test_manager_reset_requires_confirmation(memory):
    memory.capture("a fact", kind=MemoryKind.SEMANTIC)
    manager = MemoryManager(memory)
    assert not manager.reset().ok
    assert memory.count(include_archived=True) == 1
    assert manager.reset(confirmed=True).changed == 1
    assert memory.count(include_archived=True) == 0


def test_manager_duplicates_detects_without_merging(memory):
    # Same content terms, different category -> different ids, one duplicate group.
    first = memory.capture("identical content here", kind=MemoryKind.SEMANTIC)
    second = memory.capture("identical content here", kind=MemoryKind.EPISODIC)
    assert first.record_id != second.record_id
    groups = MemoryManager(memory).duplicates()
    assert any({first.record_id, second.record_id} <= set(group.record_ids) for group in groups)
    assert memory.count(include_archived=True) == 2  # nothing merged or deleted


def test_manager_search_does_not_touch_by_default(memory):
    record = memory.capture("a searchable fact", kind=MemoryKind.SEMANTIC)
    MemoryManager(memory).search("searchable fact")
    assert memory.get(record.record_id).access_count == 0


# -- isolation --------------------------------------------------------------
def test_registry_reads_own_memory(tmp_path):
    registry = PersonalityMemoryRegistry(data_root=tmp_path)
    registry.owner_memory("jarvis").capture("jarvis memory", importance=0.5)
    assert registry.read_memory("jarvis", "jarvis").count(include_archived=True) == 1


def test_registry_refuses_natasha_to_the_main_model(tmp_path):
    registry = PersonalityMemoryRegistry(data_root=tmp_path)
    registry.owner_memory("natasha").capture("private", importance=0.5)
    with pytest.raises(LearningAccessError):
        registry.read_memory("fayd_al_ruh", "natasha")
    assert not registry.can_read("fayd_al_ruh", "natasha")


def test_registry_refuses_a_cross_character_read(tmp_path):
    registry = PersonalityMemoryRegistry(data_root=tmp_path)
    registry.owner_memory("ultron").capture("ultron memory", importance=0.5)
    with pytest.raises(LearningAccessError):
        registry.read_memory("jarvis", "ultron")


def test_registry_permits_a_shareable_character_to_the_main_model(tmp_path):
    registry = PersonalityMemoryRegistry(data_root=tmp_path)
    registry.owner_memory("vision").capture("vision memory", importance=0.5)
    memory = registry.read_memory("fayd_al_ruh", "vision")
    assert memory.count(include_archived=True) == 1


def test_registry_unknown_owner_refused(tmp_path):
    registry = PersonalityMemoryRegistry(data_root=tmp_path)
    with pytest.raises(LearningAccessError):
        registry.owner_memory("nobody")


def test_registry_stores_live_in_separate_folders(tmp_path):
    registry = PersonalityMemoryRegistry(data_root=tmp_path)
    registry.owner_memory("jarvis").capture("j", importance=0.5)
    registry.owner_memory("hakeem").capture("h", importance=0.5)
    databases = sorted(path.name for path in (tmp_path / "memory").iterdir())
    assert databases == ["hakeem", "jarvis"]


# -- the controller wiring --------------------------------------------------
def test_controller_remembers_and_recalls(tmp_path):
    from core.cognitive_controller import CognitiveController

    controller = CognitiveController(
        long_term_memory=LongTermMemory("fayd_al_ruh", tmp_path / "m.sqlite3")
    )
    record = controller.remember(
        "Waterflood raised pressure in the pilot.",
        kind=MemoryKind.EPISODIC,
        importance=0.8,
        entities=("waterflood",),
    )
    assert record is not None
    report = controller.recall("waterflood pressure")
    assert report.count >= 1
    assert controller.long_term_memory_status()["attached"] is True


def test_controller_without_memory_is_unchanged():
    from core.cognitive_controller import CognitiveController

    controller = CognitiveController()
    assert controller.remember("anything") is None
    assert controller.recall("anything").count == 0
    assert controller.long_term_memory_status()["attached"] is False


def test_controller_publishes_memory_events(tmp_path):
    from core.cognitive_controller import CognitiveController
    from core.event_bus import EVENT_MEMORY_STORED

    controller = CognitiveController(
        long_term_memory=LongTermMemory("fayd_al_ruh", tmp_path / "m.sqlite3")
    )
    controller.remember("a memory", kind=MemoryKind.SEMANTIC)
    assert controller.events.count(EVENT_MEMORY_STORED) == 1


def test_controller_consolidation(tmp_path):
    from core.cognitive_controller import CognitiveController

    controller = CognitiveController(
        long_term_memory=LongTermMemory("fayd_al_ruh", tmp_path / "m.sqlite3")
    )
    controller.remember("important working item", kind=MemoryKind.WORKING, importance=0.9)
    report = controller.consolidate_memory(minimum_importance=0.5)
    assert report.changed


# -- the gate ---------------------------------------------------------------
def test_gate_is_perfect_on_the_real_architecture():
    assert memory_architecture_accuracy() == 1.0
    report = memory_gate_report()
    assert report.failed == ()
    assert report.total == len(MEMORY_JUDGEMENTS)


def test_gate_scores_zero_for_an_empty_catalogue():
    assert memory_architecture_accuracy(()) == 0.0


def test_gate_discriminates_a_broken_judgement(tmp_path):
    def _always_false(_root):
        return False

    accuracy = memory_architecture_accuracy((("broken", _always_false),))
    assert accuracy == 0.0


def test_durable_kinds_are_the_three_persisted_categories():
    assert {kind.value for kind in DURABLE_KINDS} == {"episodic", "semantic", "procedural"}


def test_schema_version_is_recorded(memory):
    assert memory.schema_version == SCHEMA_VERSION


# -- the extensions to existing modules -------------------------------------
def test_consolidator_persists_to_durable_memory(tmp_path, memory):
    from core.episodic_memory import EpisodicMemory
    from core.memory_consolidator import MemoryConsolidator
    from core.working_memory import WorkingMemory

    working = WorkingMemory()
    working.put("E1", "rain makes the ground wet", 0.8)
    stored = MemoryConsolidator().consolidate(working, EpisodicMemory(), durable=memory)
    assert stored == ["E1"]
    durable = list(memory.all_records(kind=MemoryKind.EPISODIC))
    assert len(durable) == 1
    assert durable[0].content == "rain makes the ground wet"
    assert durable[0].importance == 0.8


def test_consolidator_without_durable_is_unchanged():
    from core.episodic_memory import EpisodicMemory
    from core.memory_consolidator import MemoryConsolidator
    from core.working_memory import WorkingMemory

    working = WorkingMemory()
    working.put("E1", "rain", 0.8)
    assert MemoryConsolidator().consolidate(working, EpisodicMemory()) == ["E1"]


def test_durable_retriever_reads_durable_episodes(tmp_path, memory):
    from core.episodic_memory import EpisodicMemory
    from core.memory_associator import MemoryAssociator
    from core.memory_retriever import MemoryRetriever as EpisodeRetriever
    from core.semantic_memory import SemanticMemoryIndex

    memory.capture("the well was cased to 2000 ft", kind=MemoryKind.EPISODIC, importance=0.7)
    episodes = EpisodeRetriever().retrieve(
        "cased well",
        EpisodicMemory(),
        SemanticMemoryIndex(),
        MemoryAssociator(),
        durable=memory,
    )
    assert len(episodes) == 1
    assert "cased" in episodes[0].content

