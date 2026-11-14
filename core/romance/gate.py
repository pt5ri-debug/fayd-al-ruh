"""The held-out gate for the Romance & Companion Studio foundation.

This is the Phase-1 metric: the storage contract the whole Studio rests on, scored
as one number and wired into ``evaluation``, so a regression in the schema, the
database, the retention policy, or the isolation is a **gate failure** rather than
a quietly passing suite.

**What is measured, and how it is honest.** Every judgement runs against a *real*
:class:`~core.romance.CompanionStorageService` on a throwaway database under a
temporary directory -- the same code the window and the CLI drive -- so a schema
that stopped deriving a column, a migration that stopped refusing a newer file, or
a retention policy that stopped purging would fail here.

Several judgements are **negative controls**: a database from a newer build must be
*refused* rather than read, a dangling foreign key must be *refused*, a bad backup
must be *refused* while the good database survives, and a soft-deleted character
must *not* be listed by default. A gate that accepted any of these would be
measuring nothing.

The module is dependency-free, Tk-free, and loads no model, so it runs on a fresh
clone with no display and no weights.
"""

from __future__ import annotations

import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.romance import (
    RECORDS,
    RETENTION_ANONYMIZE,
    RETENTION_PURGE,
    SCHEMA_VERSION,
    TABLES,
    CompanionStorageService,
    RomanceDatabase,
    RomanceDatabaseError,
)
from core.romance.database import RomanceLoadReport
from core.romance.model import CompanionMemory, CompanionMessage, utc_now


@dataclass(frozen=True)
class StudioJudgement:
    """One held-out claim about the Studio foundation, and how to check it."""

    name: str
    kind: str
    note: str = ""


#: The held-out catalogue. The ``negative-*`` kinds are the controls described in
#: the module docstring.
STUDIO_JUDGEMENTS: tuple[StudioJudgement, ...] = (
    StudioJudgement("every record class has a table", "tables"),
    StudioJudgement("column types are derived from the annotations", "derived-types"),
    StudioJudgement("a nullable field is a nullable column", "nullable"),
    StudioJudgement("foreign keys and unique constraints are declared", "keys"),
    StudioJudgement("the database creates and reopens at the current version", "reopen"),
    StudioJudgement(
        "a database from a newer build is refused",
        "negative-newer-schema",
        note="negative control: a schema this build cannot read must not be touched",
    ),
    StudioJudgement("a failed batch write rolls back", "transaction"),
    StudioJudgement(
        "a dangling foreign key is refused",
        "negative-foreign-key",
        note="negative control: referential integrity must hold",
    ),
    StudioJudgement("a corrupt row is skipped and counted, not fatal", "corrupt-tolerance"),
    StudioJudgement("deactivate hides and activate restores", "soft-delete"),
    StudioJudgement(
        "a soft-deleted character is not listed by default",
        "negative-soft-delete-hidden",
        note="negative control: the default read must hide an inactive record",
    ),
    StudioJudgement("an update is a new value, not a mutation", "immutable-update"),
    StudioJudgement("every write is audited", "audit"),
    StudioJudgement("a backup restores the exact earlier state", "backup-restore"),
    StudioJudgement(
        "a bad backup is refused and the good database survives",
        "negative-bad-backup",
        note="negative control: a corrupt backup must not destroy a good database",
    ),
    StudioJudgement("retention 'keep' soft-deletes and leaves the data", "retention-keep"),
    StudioJudgement("retention 'purge' removes conversations and memories", "retention-purge"),
    StudioJudgement(
        "retention 'anonymize' keeps the rows without the name", "retention-anonymize"
    ),
    StudioJudgement("an export writes every table", "export"),
    StudioJudgement("ids are stable and kind-prefixed", "ids"),
    StudioJudgement("timestamps are UTC", "utc"),
    StudioJudgement(
        "a character's memory is scoped to that character",
        "isolation",
        note="a second character's memory is never returned for the first",
    ),
)


@contextmanager
def _storage() -> Iterator[CompanionStorageService]:
    with tempfile.TemporaryDirectory() as folder:
        yield CompanionStorageService(root=Path(folder))


def _new_character(storage: CompanionStorageService, name: str = "Layla") -> Any:
    return storage.domain.create("companion_characters", owner="natasha", name=name)


def _judgement_holds(judgement: StudioJudgement) -> bool:
    kind = judgement.kind

    if kind == "tables":
        return all(name in TABLES and TABLES[name].record is record for name, record in RECORDS.items())

    if kind == "derived-types":
        spec = TABLES["companion_characters"]
        by_name = {column.name: column for column in spec.columns}
        return (
            by_name["record_id"].sql_type == "TEXT"
            and by_name["fictional_age"].sql_type == "INTEGER"
            and by_name["active"].boolean is True
            and by_name["languages"].json is True
            and by_name["languages"].json_kind == "list"
            and by_name["metadata"].json is True
        )

    if kind == "nullable":
        spec = TABLES["companion_characters"]
        age = next(column for column in spec.columns if column.name == "fictional_age")
        return age.nullable is True

    if kind == "keys":
        memories = TABLES["companion_memories"]
        return (
            ("character_id", "companion_characters", "record_id") in memories.foreign_keys
            and ("character_id", "user_id") in TABLES["relationship_states"].uniques
            and ("conversation_id", "seq") in TABLES["companion_messages"].uniques
        )

    if kind == "reopen":
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "romance.db"
            RomanceDatabase(path)
            again = RomanceDatabase(path)
            return again.schema_version() == SCHEMA_VERSION and again.integrity_check()

    if kind == "negative-newer-schema":
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "romance.db"
            RomanceDatabase(path)
            connection = sqlite3.connect(path)
            connection.execute(
                "UPDATE meta SET value=? WHERE key='schema_version'",
                (str(SCHEMA_VERSION + 9),),
            )
            connection.commit()
            connection.close()
            try:
                RomanceDatabase(path)
            except RomanceDatabaseError:
                return True
            return False

    if kind == "transaction":
        with _storage() as storage:
            character = _new_character(storage)
            conversation = storage.domain.create(
                "companion_conversations", character_id=character.record_id, title="First chat"
            )
            valid = CompanionMessage(
                record_id="MSG-1", conversation_id=conversation.record_id, seq=1, content="hi"
            )
            invalid = CompanionMessage(
                record_id="MSG-2", conversation_id="CNV-missing", seq=2, content="bye"
            )
            try:
                storage.database.upsert_many("companion_messages", [valid, invalid])
            except sqlite3.IntegrityError:
                return storage.domain.count("companion_messages") == 0
            return False

    if kind == "negative-foreign-key":
        with _storage() as storage:
            try:
                storage.database.upsert(
                    "companion_memories",
                    CompanionMemory(record_id="MEM-x", character_id="CHR-missing"),
                )
            except sqlite3.IntegrityError:
                return True
            return False

    if kind == "corrupt-tolerance":
        with _storage() as storage:
            character = _new_character(storage)
            connection = sqlite3.connect(storage.db_path)
            connection.execute(
                "UPDATE companion_characters SET languages='{not json' WHERE record_id=?",
                (character.record_id,),
            )
            connection.commit()
            connection.close()
            load = RomanceLoadReport()
            rows = storage.database.list_records("companion_characters", report=load)
            return load.skipped == 1 and bool(rows) and rows[0].languages == ()

    if kind == "soft-delete":
        with _storage() as storage:
            character = _new_character(storage)
            storage.domain.deactivate("companion_characters", character.record_id)
            hidden = storage.domain.list_records("companion_characters") == []
            storage.domain.activate("companion_characters", character.record_id)
            restored = storage.domain.list_records("companion_characters")
            return hidden and bool(restored) and restored[0].active is True

    if kind == "negative-soft-delete-hidden":
        with _storage() as storage:
            character = _new_character(storage)
            storage.domain.deactivate("companion_characters", character.record_id)
            return storage.domain.list_records("companion_characters") == []

    if kind == "immutable-update":
        with _storage() as storage:
            character = _new_character(storage)
            updated = storage.domain.update(
                "companion_characters", character.record_id, name="Layla N."
            )
            return updated.name == "Layla N." and character.name == "Layla"

    if kind == "audit":
        with _storage() as storage:
            character = _new_character(storage)
            storage.domain.update("companion_characters", character.record_id, name="Layla N.")
            storage.domain.deactivate("companion_characters", character.record_id)
            actions = {row.action for row in storage.domain.audit_for(character.record_id)}
            return {"create", "update", "deactivate"} <= actions

    if kind == "backup-restore":
        with _storage() as storage:
            _new_character(storage, "Layla")
            backup = storage.backup()
            _new_character(storage, "Mona")
            if storage.domain.count("companion_characters") != 2:
                return False
            storage.restore(backup.path)
            return storage.domain.count("companion_characters") == 1

    if kind == "negative-bad-backup":
        with _storage() as storage:
            _new_character(storage)
            junk = Path(storage.root) / "junk.db"
            junk.write_bytes(b"not a database")
            report = storage.restore(junk)
            return report.ok is False and storage.domain.count("companion_characters") == 1

    if kind == "retention-keep":
        with _storage() as storage:
            character = _new_character(storage)
            storage.domain.create(
                "companion_conversations", character_id=character.record_id, title="First chat"
            )
            storage.domain.create(
                "companion_memories", character_id=character.record_id, content="tea"
            )
            deletion = storage.delete_character(character.record_id, choice="keep")
            return (
                deletion.deactivated
                and storage.domain.count("companion_conversations") == 1
                and storage.domain.count("companion_memories") == 1
            )

    if kind == "retention-purge":
        with _storage() as storage:
            character = _new_character(storage)
            conversation = storage.domain.create(
                "companion_conversations", character_id=character.record_id, title="First chat"
            )
            storage.domain.create(
                "companion_messages", conversation_id=conversation.record_id, seq=1, content="hi"
            )
            storage.domain.create(
                "companion_memories", character_id=character.record_id, content="tea"
            )
            storage.delete_character(character.record_id, choice=RETENTION_PURGE)
            return (
                storage.domain.count("companion_conversations") == 0
                and storage.domain.count("companion_messages") == 0
                and storage.domain.count("companion_memories") == 0
            )

    if kind == "retention-anonymize":
        with _storage() as storage:
            character = _new_character(storage)
            storage.domain.create(
                "companion_conversations", character_id=character.record_id, title="First chat"
            )
            storage.delete_character(character.record_id, choice=RETENTION_ANONYMIZE)
            stored = storage.repository.get_record("companion_characters", character.record_id)
            return (
                stored.name == "(deleted)"
                and storage.domain.count("companion_conversations", include_inactive=True) == 1
            )

    if kind == "export":
        with tempfile.TemporaryDirectory() as folder:
            with _storage() as storage:
                character = _new_character(storage)
                storage.domain.create(
                    "companion_memories", character_id=character.record_id, content="tea"
                )
                bundle = storage.export(Path(folder) / "export.json")
                return (
                    bundle.tables.get("companion_characters") == 1
                    and bundle.tables.get("companion_memories") == 1
                )

    if kind == "ids":
        with _storage() as storage:
            character = _new_character(storage)
            memory = storage.domain.create(
                "companion_memories", character_id=character.record_id, content="tea"
            )
            return character.record_id.startswith("CHR-") and memory.record_id.startswith("MEM-")

    if kind == "utc":
        stamp = utc_now()
        return stamp.endswith("+00:00") or stamp.endswith("Z")

    if kind == "isolation":
        with _storage() as storage:
            first = _new_character(storage, "Layla")
            second = _new_character(storage, "Mona")
            storage.domain.create("companion_memories", character_id=first.record_id, content="tea")
            storage.domain.create("companion_memories", character_id=second.record_id, content="coffee")
            mine = storage.domain.list_records(
                "companion_memories", where="character_id = ?", params=(first.record_id,)
            )
            return len(mine) == 1 and mine[0].content == "tea"

    return False


def companion_studio_accuracy(
    judgements: tuple[StudioJudgement, ...] = STUDIO_JUDGEMENTS,
) -> float:
    """The share of held-out foundation judgements the Studio satisfies.

    A judgement counts only when the real storage service does exactly what it
    must -- the schema is derived correctly, the database is versioned and
    transactional, a newer file and a bad backup are refused, retention honours
    the caller's choice, and memory is scoped to its character. So a regression in
    the foundation fails the relevant judgement. Returns ``0.0`` for an empty
    catalogue.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    correct = sum(1 for judgement in catalogue if _judgement_holds(judgement))
    return correct / len(catalogue)


__all__ = [
    "STUDIO_JUDGEMENTS",
    "StudioJudgement",
    "companion_studio_accuracy",
]
