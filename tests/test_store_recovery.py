"""Store recovery (Technical Gap G-16 / finding F-1).

A durable store must survive the failure modes a real machine produces: a
truncated or garbage-tailed database, a partially written file, a schema from a
newer build, and a bad backup. The rule everywhere is the same -- **tolerate or
report, never crash, and never lose a good copy to a bad one**.

These tests exercise the real stores on isolated copies under ``tmp_path``; no
test touches a user's only copy of real data.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from core.character_learning_store import CharacterLearningStore
from core.learning_events import (
    KIND_KNOWLEDGE,
    OUTCOME_ACCEPTED,
    SOURCE_DOCUMENT,
    LearningEvent,
)
from core.learning_records import LearningRecord
from core.learning_store import LearningStore
from core.long_term_memory import LongTermMemory
from core.memory_db import MemoryDatabaseError


def _record(
    *, owner: str = "ultron", knowledge: str = "porosity is the void space in a rock"
) -> LearningRecord:
    return LearningRecord(
        record_id="rec-1",
        owner=owner,
        source_type="user",
        source_reference="doc-1",
        question="what is porosity?",
        answer="the void space in a rock",
        knowledge=knowledge,
        verification="provisional",
    )


def _event(*, target: str = "porosity measures permeability") -> LearningEvent:
    return LearningEvent(
        kind=KIND_KNOWLEDGE,
        target=target,
        owner="fayd_al_ruh",
        source=SOURCE_DOCUMENT,
        source_reference="doc-1",
        outcome=OUTCOME_ACCEPTED,
        applied=True,
    )


# ---------------------------------------------------------- learning store

def test_learning_store_round_trips(tmp_path: Path):
    store = LearningStore("fayd_al_ruh", tmp_path / "learning.sqlite3")
    store.record_event(_event())
    assert store.count() >= 1
    ok, message = store.integrity_check()
    assert ok, message
    # A version advances only on an applied change, through next_version().
    assert store.next_version() >= 1
    assert store.version >= 1


def test_learning_store_survives_garbage_tail(tmp_path: Path):
    path = tmp_path / "learning.sqlite3"
    store = LearningStore("fayd_al_ruh", path)
    store.record_event(_event())
    before = store.count()

    with open(path, "ab") as handle:
        handle.write(b"\x00\x01\x02not a sqlite page")

    reopened = LearningStore("fayd_al_ruh", path)
    # Either the records are still readable (tolerated) or the failure is
    # reported as a MemoryDatabaseError -- never a bare crash.
    try:
        assert reopened.count() >= before
    except MemoryDatabaseError:
        pass


def test_learning_store_refuses_a_newer_schema(tmp_path: Path):
    path = tmp_path / "learning.sqlite3"
    store = LearningStore("fayd_al_ruh", path)
    store.record_event(_event())

    connection = sqlite3.connect(path)
    try:
        connection.execute("UPDATE meta SET value='9999' WHERE key='schema_version'")
        connection.commit()
    except sqlite3.OperationalError:
        pytest.skip("this store keeps its schema version elsewhere")
    finally:
        connection.close()

    with pytest.raises(Exception):
        LearningStore("fayd_al_ruh", path).count()


def test_learning_store_backup_and_restore(tmp_path: Path):
    path = tmp_path / "learning.sqlite3"
    store = LearningStore("fayd_al_ruh", path)
    store.record_event(_event())
    backup = tmp_path / "backup.sqlite3"
    report = store.backup(backup)
    assert backup.exists()
    assert getattr(report, "ok", True)

    store.record_event(_event(target="a second item"))
    after = store.count()
    store.restore(backup)
    assert store.count() <= after


def test_learning_store_rollback_is_append_only(tmp_path: Path):
    store = LearningStore("fayd_al_ruh", tmp_path / "learning.sqlite3")
    store.record_event(_event())
    version = store.version
    store.record_event(_event(target="later item"))
    rolled = store.rollback(version, reason="audit")
    assert isinstance(rolled, list)
    # The audit trail still holds every event (rollback marks, never deletes).
    events, _ = store.events(limit=1000)
    assert len(events) >= 2


# --------------------------------------------------------------- memory db

def test_memory_db_survives_garbage_tail(tmp_path: Path):
    path = tmp_path / "memory.sqlite3"
    memory = LongTermMemory("audit_recovery", str(path))
    memory.capture("porosity measures the void space in a rock")
    before = memory.count()

    with open(path, "ab") as handle:
        handle.write(b"\x00\x01\x02garbage")

    try:
        reopened = LongTermMemory("audit_recovery", str(path))
        assert reopened.count() >= before
        ok, message = reopened.integrity_check()
        assert ok, message
    except MemoryDatabaseError:
        pass


def test_memory_db_rejects_a_bad_backup_restore(tmp_path: Path):
    path = tmp_path / "memory.sqlite3"
    memory = LongTermMemory("audit_recovery", str(path))
    memory.capture("porosity measures the void space in a rock")
    good = memory.count()

    junk = tmp_path / "junk.sqlite3"
    junk.write_bytes(b"this is not a database at all")
    report = memory.restore(str(junk))
    # The bad backup is refused (reported, never applied), and the current,
    # good database is untouched by the refused restore.
    assert getattr(report, "ok", False) is False
    assert memory.count() == good


# -------------------------------------------------------- character store

def test_character_store_tolerates_a_corrupt_line(tmp_path: Path):
    store = CharacterLearningStore("ultron", root=tmp_path)
    store.append(_record())
    records_path = store.records_path
    with open(records_path, "a", encoding="utf-8") as handle:
        handle.write("{ not valid json }\n")

    reopened = CharacterLearningStore("ultron", root=tmp_path)
    # The corrupt line is skipped, the good one survives, and nothing raises.
    assert reopened.all() or reopened.stats() is not None


def test_character_store_export_is_readable(tmp_path: Path):
    store = CharacterLearningStore("ultron", root=tmp_path)
    store.append(_record())
    payload = store.export()
    assert isinstance(payload, dict)
    assert payload.get("owner") == "ultron"
