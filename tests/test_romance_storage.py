"""Phase 1: the Romance & Companion Studio foundation and database.

These tests pin the storage contract the whole Studio rests on: the record model,
the derived schema, the versioned/transactional database, the typed repository,
the audited domain service, and the retention/export/backup behaviour. They run
with no display, no model, and no network.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from core.romance import (
    RECORDS,
    RETENTION_ANONYMIZE,
    RETENTION_PURGE,
    SCHEMA_VERSION,
    TABLES,
    CompanionStorageService,
    RomanceDatabase,
    RomanceDatabaseError,
    RomanceDomainError,
)
from core.romance.model import CompanionMemory, CompanionMessage, utc_now


@pytest.fixture()
def storage(tmp_path: Path) -> CompanionStorageService:
    return CompanionStorageService(root=tmp_path)


# --------------------------------------------------------------- schema


def test_every_record_class_has_a_table():
    for name, record in RECORDS.items():
        assert name in TABLES, f"{name} has no TableSpec"
        assert TABLES[name].record is record


def test_columns_are_derived_from_the_dataclass_annotations():
    spec = TABLES["companion_characters"]
    types = {column.name: column.sql_type for column in spec.columns}
    assert types["record_id"] == "TEXT"
    assert types["fictional_age"] == "INTEGER"  # int | None
    assert types["active"] == "INTEGER"          # bool -> 0/1
    # A tuple field becomes a JSON column.
    languages = next(c for c in spec.columns if c.name == "languages")
    assert languages.json is True and languages.json_kind == "list"
    # A dict field becomes a JSON object column.
    metadata = next(c for c in spec.columns if c.name == "metadata")
    assert metadata.json is True and metadata.json_kind == "dict"


def test_foreign_keys_and_uniques_are_declared():
    assert ("character_id", "companion_characters", "record_id") in TABLES[
        "companion_memories"
    ].foreign_keys
    assert TABLES["companion_messages"].foreign_key_columns == frozenset({"conversation_id"})
    assert ("character_id", "user_id") in TABLES["relationship_states"].uniques
    assert ("conversation_id", "seq") in TABLES["companion_messages"].uniques


def test_nullable_columns_allow_an_unspecified_age():
    spec = TABLES["companion_characters"]
    age = next(c for c in spec.columns if c.name == "fictional_age")
    assert age.nullable is True


# ------------------------------------------------------------- database


def test_database_creates_and_reopens_idempotently(tmp_path: Path):
    path = tmp_path / "romance.db"
    first = RomanceDatabase(path)
    assert first.schema_version() == SCHEMA_VERSION
    # Re-opening an existing file must not lose data or re-run migrations badly.
    second = RomanceDatabase(path)
    assert second.schema_version() == SCHEMA_VERSION
    assert second.integrity_check() is True


def test_a_newer_schema_is_refused(tmp_path: Path):
    path = tmp_path / "romance.db"
    RomanceDatabase(path)
    connection = sqlite3.connect(path)
    connection.execute("UPDATE meta SET value=? WHERE key='schema_version'", (str(SCHEMA_VERSION + 5),))
    connection.commit()
    connection.close()
    with pytest.raises(RomanceDatabaseError, match="refusing to read it"):
        RomanceDatabase(path)


def test_a_failed_write_rolls_back(storage: CompanionStorageService):
    character = storage.domain.create("companion_characters", owner="natasha", name="Layla")
    conversation = storage.domain.create(
        "companion_conversations", character_id=character.record_id, title="First chat"
    )
    # A batch whose second row references a conversation that does not exist must
    # fail as a whole: the first (valid) row is rolled back with it.
    valid = CompanionMessage(
        record_id="MSG-1", conversation_id=conversation.record_id, seq=1, content="hi"
    )
    invalid = CompanionMessage(
        record_id="MSG-2", conversation_id="CNV-missing", seq=2, content="bye"
    )
    with pytest.raises(sqlite3.IntegrityError):
        storage.database.upsert_many("companion_messages", [valid, invalid])
    assert storage.domain.count("companion_messages") == 0


def test_foreign_key_is_enforced(storage: CompanionStorageService):
    # A memory that references a character which does not exist is refused.
    with pytest.raises(sqlite3.IntegrityError):
        storage.database.upsert(
            "companion_memories",
            CompanionMemory(record_id="MEM-x", character_id="CHR-missing"),
        )


def test_corrupt_row_is_skipped_and_counted(storage: CompanionStorageService):
    character = storage.domain.create("companion_characters", owner="natasha", name="Layla")
    # Corrupt the JSON payload of the languages column directly.
    connection = sqlite3.connect(storage.db_path)
    connection.execute(
        "UPDATE companion_characters SET languages='{not json' WHERE record_id=?",
        (character.record_id,),
    )
    connection.commit()
    connection.close()
    from core.romance.database import RomanceLoadReport

    load = RomanceLoadReport()
    rows = storage.database.list_records("companion_characters", report=load)
    assert load.skipped == 1
    assert rows and rows[0].languages == ()  # falls back to an empty tuple


# ------------------------------------------------------------ repository


def test_repository_round_trips_a_character_with_typed_fields(storage: CompanionStorageService):
    character = storage.domain.create(
        "companion_characters",
        owner="natasha",
        name="Layla",
        fictional_age=27,
        adult_eligible=True,
        languages=("ar", "en"),
        tags=("romantic",),
        metadata={"hair": "dark"},
    )
    loaded = storage.repository.get_record("companion_characters", character.record_id)
    assert loaded.name == "Layla"
    assert loaded.fictional_age == 27
    assert loaded.adult_eligible is True
    assert loaded.languages == ("ar", "en")
    assert loaded.tags == ("romantic",)
    assert loaded.metadata == {"hair": "dark"}
    assert loaded.active is True


def test_an_unspecified_age_round_trips_as_none(storage: CompanionStorageService):
    character = storage.domain.create("companion_characters", owner="natasha", name="NoAge")
    loaded = storage.repository.get_record("companion_characters", character.record_id)
    assert loaded.fictional_age is None


def test_deactivate_is_a_soft_delete_and_activate_restores(storage: CompanionStorageService):
    character = storage.domain.create("companion_characters", owner="natasha", name="Layla")
    storage.domain.deactivate("companion_characters", character.record_id)
    assert storage.domain.list_records("companion_characters") == []
    hidden = storage.domain.list_records("companion_characters", include_inactive=True)
    assert hidden and hidden[0].active is False
    storage.domain.activate("companion_characters", character.record_id)
    assert storage.domain.list_records("companion_characters")[0].active is True


def test_update_is_a_new_value_and_bumps_updated_at(storage: CompanionStorageService):
    character = storage.domain.create("companion_characters", owner="natasha", name="Layla")
    before = character.updated_at
    updated = storage.domain.update("companion_characters", character.record_id, name="Layla N.")
    assert updated.name == "Layla N."
    assert updated.updated_at >= before
    # The original value is untouched (frozen records are never mutated in place).
    assert character.name == "Layla"


def test_unknown_entity_and_field_are_clear_errors(storage: CompanionStorageService):
    with pytest.raises(RomanceDomainError, match="Unknown Romance entity"):
        storage.domain.create("no_such_table", name="x")
    character = storage.domain.create("companion_characters", owner="natasha", name="Layla")
    with pytest.raises(RomanceDomainError, match="has no field"):
        storage.domain.update("companion_characters", character.record_id, bogus=1)


def test_every_write_is_audited(storage: CompanionStorageService):
    character = storage.domain.create("companion_characters", owner="natasha", name="Layla")
    storage.domain.update("companion_characters", character.record_id, name="Layla N.")
    storage.domain.deactivate("companion_characters", character.record_id)
    trail = storage.domain.audit_for(character.record_id)
    actions = [row.action for row in trail]  # newest first
    assert actions == ["deactivate", "update", "create"]
    assert all(row.actor == "system" for row in trail)


# ------------------------------------------------------------- retention


def test_delete_with_keep_soft_deletes_and_leaves_data(storage: CompanionStorageService):
    character = storage.domain.create("companion_characters", owner="natasha", name="Layla")
    storage.domain.create(
        "companion_conversations", character_id=character.record_id, title="First chat"
    )
    storage.domain.create(
        "companion_memories", character_id=character.record_id, content="likes tea"
    )
    report = storage.delete_character(character.record_id, choice="keep")
    assert report.deactivated is True
    assert report.conversations_removed == 0
    # The conversation and memory survive (a restore brings the character back).
    assert storage.domain.count("companion_conversations") == 1
    assert storage.domain.count("companion_memories") == 1


def test_delete_with_purge_removes_conversations_and_memories(storage: CompanionStorageService):
    character = storage.domain.create("companion_characters", owner="natasha", name="Layla")
    conversation = storage.domain.create(
        "companion_conversations", character_id=character.record_id, title="First chat"
    )
    storage.domain.create(
        "companion_messages", conversation_id=conversation.record_id, seq=1, content="hi"
    )
    storage.domain.create("companion_memories", character_id=character.record_id, content="tea")
    report = storage.delete_character(character.record_id, choice=RETENTION_PURGE)
    assert report.conversations_removed == 1
    assert report.memories_removed == 1
    assert storage.domain.count("companion_conversations") == 0
    # The message went with its conversation (ON DELETE CASCADE).
    assert storage.domain.count("companion_messages") == 0
    assert storage.domain.count("companion_memories") == 0


def test_delete_with_anonymize_keeps_rows_without_the_name(storage: CompanionStorageService):
    character = storage.domain.create("companion_characters", owner="natasha", name="Layla")
    storage.domain.create(
        "companion_conversations", character_id=character.record_id, title="First chat"
    )
    storage.delete_character(character.record_id, choice=RETENTION_ANONYMIZE)
    stored = storage.repository.get_record("companion_characters", character.record_id)
    assert stored.name == "(deleted)"
    assert storage.domain.count("companion_conversations", include_inactive=True) == 1


def test_unknown_retention_choice_is_refused(storage: CompanionStorageService):
    character = storage.domain.create("companion_characters", owner="natasha", name="Layla")
    with pytest.raises(RomanceDomainError, match="Unknown retention choice"):
        storage.delete_character(character.record_id, choice="obliterate")


# ---------------------------------------------------------------- backup


def test_backup_and_restore_round_trip(storage: CompanionStorageService, tmp_path: Path):
    storage.domain.create("companion_characters", owner="natasha", name="Layla")
    report = storage.backup()
    assert report.ok is True
    assert Path(report.path).exists()
    # Add a second character, then restore: the second must be gone.
    storage.domain.create("companion_characters", owner="natasha", name="Mona")
    assert storage.domain.count("companion_characters") == 2
    restored = storage.restore(report.path)
    assert restored.ok is True
    assert storage.domain.count("companion_characters") == 1


def test_a_bad_backup_is_refused_and_the_good_database_survives(
    storage: CompanionStorageService, tmp_path: Path
):
    storage.domain.create("companion_characters", owner="natasha", name="Layla")
    junk = tmp_path / "junk.db"
    junk.write_bytes(b"this is not a database")
    report = storage.restore(junk)
    assert report.ok is False
    assert storage.domain.count("companion_characters") == 1


# ---------------------------------------------------------------- export


def test_export_writes_every_table(storage: CompanionStorageService, tmp_path: Path):
    character = storage.domain.create("companion_characters", owner="natasha", name="Layla")
    storage.domain.create(
        "companion_memories", character_id=character.record_id, content="likes tea"
    )
    bundle = storage.export(tmp_path / "export.json")
    assert Path(bundle.path).exists()
    assert bundle.tables["companion_characters"] == 1
    assert bundle.tables["companion_memories"] == 1
    import json

    payload = json.loads(Path(bundle.path).read_text(encoding="utf-8"))
    assert payload["companion_characters"][0]["name"] == "Layla"


def test_export_can_be_scoped_to_one_character(storage: CompanionStorageService, tmp_path: Path):
    a = storage.domain.create("companion_characters", owner="natasha", name="A")
    storage.domain.create("companion_characters", owner="natasha", name="B")
    storage.domain.create("companion_memories", character_id=a.record_id, content="x")
    bundle = storage.export(tmp_path / "one.json", character_id=a.record_id)
    assert bundle.tables["companion_characters"] == 1
    assert bundle.tables["companion_memories"] == 1


# ---------------------------------------------------------------- misc


def test_paths_resolve_under_the_data_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from core import paths

    assert paths.romance_dir() == tmp_path / "romance"
    assert paths.romance_db_path() == tmp_path / "romance" / "romance.db"
    assert paths.romance_backups_dir() == tmp_path / "romance" / "backups"


def test_ids_are_stable_and_kind_prefixed(storage: CompanionStorageService):
    character = storage.domain.create("companion_characters", owner="natasha", name="Layla")
    assert character.record_id.startswith("CHR-")
    memory = storage.domain.create(
        "companion_memories", character_id=character.record_id, content="x"
    )
    assert memory.record_id.startswith("MEM-")


def test_utc_timestamps_are_timezone_aware():
    stamp = utc_now()
    assert stamp.endswith("+00:00") or stamp.endswith("Z")


def test_paging_bounds_a_read(storage: CompanionStorageService):
    for index in range(5):
        storage.domain.create("companion_characters", owner="natasha", name=f"C{index}")
    page = storage.domain.list_records("companion_characters", limit=2)
    assert len(page) == 2


# ---------------------------------------------------------------- gate


def test_the_held_out_gate_is_satisfied():
    from core.romance.gate import STUDIO_JUDGEMENTS, companion_studio_accuracy

    assert companion_studio_accuracy() == 1.0
    assert len(STUDIO_JUDGEMENTS) >= 20


def test_the_gate_reports_zero_for_an_empty_catalogue():
    from core.romance.gate import companion_studio_accuracy

    assert companion_studio_accuracy(()) == 0.0
