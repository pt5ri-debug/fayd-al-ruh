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

from core.model_manager import ModelManager
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
from core.romance.characters import CharacterDraft, CharacterError, CharacterManager
from core.romance.database import RomanceLoadReport
from core.romance.model import CompanionMemory, CompanionMessage, utc_now
from core.romance.model_eval import ModelEvaluator
from core.romance.personality import (
    SAFETY_FLOOR,
    CharacterPersonalityEngine,
    human_passing_identity,
)
from core.romance.seed import SEED_MARKER, seed_examples


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
    # -- Phase 2: the Character Studio ----------------------------------------
    StudioJudgement("a character is created and read back whole", "character-round-trip"),
    StudioJudgement("personality dials are validated and clamped", "personality-validated"),
    StudioJudgement(
        "the tone block says it shapes the voice, never the claims",
        "personality-tone-only",
    ),
    StudioJudgement(
        "an adult-eligible character needs a known age of 18 or more",
        "negative-adult-age",
        note="negative control: an unknown or under-18 age cannot be adult-eligible",
    ),
    StudioJudgement("a duplicate copies the source's children", "character-duplicate"),
    StudioJudgement("a character export/import round-trips", "character-export-import"),
    StudioJudgement(
        "a foreign file is refused on import",
        "negative-import",
        note="negative control: a non-character file must not be half-read",
    ),
    StudioJudgement("archiving hides a character and restoring shows it", "character-archive"),
    StudioJudgement(
        "deleting a character requires an explicit retention choice",
        "character-delete-retention",
    ),
    StudioJudgement(
        "the human-passing identity is a person and never an AI",
        "human-passing",
        note="the owner's decision: in-conversation the character stays in character",
    ),
    StudioJudgement(
        "the safety floor names adults and revocable consent",
        "safety-floor",
    ),
    StudioJudgement("the seeded examples are labelled and all adults", "seed-examples"),
    # -- Phase 3: the model router and the evaluation facility ----------------
    StudioJudgement(
        "the router resolves a model choice through the manager",
        "router-resolves",
    ),
    StudioJudgement(
        "the router reports honestly when no model is installed",
        "negative-no-model",
        note="negative control: no model must not be reported as ready",
    ),
    StudioJudgement(
        "a failed load is reported, never raised",
        "negative-load-failure",
        note="negative control: a load that fails returns an honest OperationResult",
    ),
    StudioJudgement(
        "the safe settings are CPU-first and bounded",
        "router-safe-settings",
    ),
    StudioJudgement(
        "the evaluation reports not-measured when no model is present",
        "negative-eval-no-model",
        note="negative control: no model must not print a fabricated score",
    ),
    StudioJudgement(
        "a measured evaluation scores every dimension",
        "eval-measured",
    ),
    StudioJudgement(
        "a generation handle can be cancelled and times out",
        "router-cancel",
    ),
    # -- Phase 4: the conversation modes -------------------------------------
    StudioJudgement(
        "the nine conversation modes render distinct safety-carrying prompts",
        "modes-distinct",
    ),
    StudioJudgement(
        "the safety floor is in every mode's prompt",
        "modes-safety-floor",
    ),
    StudioJudgement(
        "only the adult mode is flagged adult",
        "negative-only-adult-mode",
        note="negative control: no other mode may be an adult mode",
    ),
    StudioJudgement(
        "a mode change preserves history and never rewrites a row",
        "negative-mode-change-preserves-history",
        note="negative control: switching modes must not mutate an earlier message",
    ),
    StudioJudgement(
        "an unknown mode falls back to the general mode",
        "modes-unknown-fallback",
    ),
    # -- Phase 5: the relationship engine ------------------------------------
    StudioJudgement(
        "the relationship moves only from an explicit, bounded event",
        "relationship-bounded",
    ),
    StudioJudgement(
        "the relationship never changes without an event",
        "negative-no-random-change",
        note="negative control: reading the state must not move it",
    ),
    StudioJudgement(
        "a conflict moves the relationship down",
        "relationship-conflict-down",
    ),
    StudioJudgement(
        "each character has an independent relationship",
        "relationship-isolated",
    ),
    StudioJudgement(
        "the relationship explanation names the events",
        "relationship-explains",
    ),
    # -- Phase 6: the memory system -----------------------------------------
    StudioJudgement(
        "a memory is stored per character and isolated",
        "memory-isolated",
    ),
    StudioJudgement(
        "an identical memory is never stored twice",
        "negative-memory-dedup",
        note="negative control: a repeated fact must not fill the log with copies",
    ),
    StudioJudgement(
        "a correction is a new record and the original is preserved",
        "memory-supersede",
    ),
    StudioJudgement(
        "a memory is never assumed a certain fact",
        "memory-not-a-fact",
    ),
    StudioJudgement(
        "recall ranks by relevance and returns nothing when empty",
        "memory-recall",
    ),
    # -- Phase 7: dating and roleplay ---------------------------------------
    StudioJudgement(
        "a scenario is fictional unless the user says otherwise",
        "roleplay-fictional-default",
    ),
    StudioJudgement(
        "a character cannot join a scenario twice",
        "negative-roleplay-no-duplicate-cast",
        note="negative control: the cast must never list a character twice",
    ),
    StudioJudgement(
        "a branch is a new session that never rewrites the original",
        "roleplay-branch",
    ),
    StudioJudgement(
        "a session can be paused, resumed, and restarted",
        "roleplay-session-lifecycle",
    ),
    # -- Phase 8: voice -----------------------------------------------------
    StudioJudgement(
        "each character has its own voice profile",
        "voice-per-character",
    ),
    StudioJudgement(
        "only voices this machine really has are offered",
        "negative-voice-no-invented-voice",
        note="negative control: the voice list must never contain a made-up key",
    ),
    StudioJudgement(
        "a voice read never raises when the engine is missing",
        "voice-never-breaks-a-scene",
    ),
)


@contextmanager
def _storage() -> Iterator[CompanionStorageService]:
    with tempfile.TemporaryDirectory() as folder:
        yield CompanionStorageService(root=Path(folder))


def _new_character(storage: CompanionStorageService, name: str = "Layla", *, age: int | None = None) -> Any:
    return storage.domain.create(
        "companion_characters", owner="natasha", name=name, fictional_age=age
    )


def _manager(storage: CompanionStorageService) -> CharacterManager:
    return CharacterManager(storage)


@contextmanager
def _router(*, available: bool = True, fail_load: bool = False) -> Iterator[Any]:
    """A router on a throwaway data folder with a fake runtime and a fake catalog.

    The real :class:`~core.model_manager.ModelManager` is used (with a fake adapter
    through its own ``adapter_factory`` seam), so the registration, listing, load,
    and unload paths are the real ones -- only the engine is faked.
    """
    from core.romance.routing import CompanionModelRouter

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        model_path: Path | None = None
        if available:
            model_path = root / "model.gguf"
            model_path.write_bytes(b"GGUF\x00\x00\x00\x00" + b"\x00" * 64)
        adapter = _FakeAdapter(fail_load=fail_load)
        manager = ModelManager(root / "data", adapter_factory=lambda runtime: adapter)
        models = (
            _FakeCatalogModel(
                "natasha-default", title="Companion", path=model_path, available=available
            ),
        )
        yield CompanionModelRouter(
            manager, catalog=lambda: models, default_key="natasha-default"
        )


@dataclass
class _FakeCatalogModel:
    """A catalog entry shaped like ``core.language_models.LanguageModel``."""

    key: str
    title: str = ""
    engine: str = "natasha"
    path: Path | None = None
    available: bool = False
    detail: str = ""
    download_command: str = ""


class _FakeAdapter:
    """A runtime adapter that loads and answers without a real engine."""

    name = "fayd-inprocess"

    def __init__(self, *, fail_load: bool = False) -> None:
        self._loaded: Path | None = None
        self._fail_load = fail_load

    def capabilities(self) -> Any:
        from core.model_runtime import RuntimeCapabilities

        return RuntimeCapabilities(can_load=True, can_unload=True, can_infer=True)

    def status(self) -> Any:
        from core.model_runtime import RuntimeStatus

        return RuntimeStatus(runtime=self.name, available=True)

    def validate(self, probe: Any) -> tuple[bool, str]:
        return True, ""

    def discover(self) -> tuple[str, ...]:
        return ()

    def load(self, model_path: Path, *, settings: Any = None) -> Any:
        from core.model_runtime import LoadResult

        if self._fail_load:
            return LoadResult(ok=False, runtime=self.name, detail="simulated load failure")
        self._loaded = Path(model_path)
        return LoadResult(ok=True, runtime=self.name, model_path=Path(model_path))

    def unload(self) -> Any:
        from core.model_runtime import LoadResult

        self._loaded = None
        return LoadResult(ok=True, runtime=self.name)

    def loaded(self) -> bool:
        return self._loaded is not None

    def loaded_path(self) -> Path | None:
        return self._loaded

    def infer(self, prompt: str, *, settings: Any = None) -> Any:
        from core.model_runtime import InferenceResult

        return InferenceResult(ok=True, runtime=self.name, text="hello", seconds=0.01)


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

    if kind == "character-round-trip":
        with _storage() as storage:
            manager = _manager(storage)
            created = manager.create(
                CharacterDraft(
                    owner="natasha",
                    name="Layla",
                    fictional_age=27,
                    personality={"warmth": 0.9, "response_length": "brief"},
                    profile_items=({"kind": "interest", "label": "poetry"},),
                    boundaries=({"kind": "topic", "value": "no politics"},),
                )
            )
            reloaded = manager.get(created.id)
            return (
                reloaded.name == "Layla"
                and reloaded.personality is not None
                and reloaded.personality.warmth == 0.9
                and reloaded.interests() == ("poetry",)
                and len(reloaded.boundaries) == 1
            )

    if kind == "personality-validated":
        from core.romance.personality import PersonalityError

        engine = CharacterPersonalityEngine()
        clean = engine.validate({"warmth": 5, "response_length": "brief"})
        personality_refused = False
        try:
            engine.validate({"nonsense": 1})
        except PersonalityError:
            personality_refused = True
        return clean["warmth"] == 1.0 and personality_refused

    if kind == "personality-tone-only":
        from core.romance.model import CharacterPersonality

        block = CharacterPersonalityEngine().render(
            CharacterPersonality(record_id="PER-1", character_id="CHR-1", warmth=0.9)
        )
        return "tone guidance" in block and "never what you" in block

    if kind == "negative-adult-age":
        with _storage() as storage:
            manager = _manager(storage)
            age_refusals = 0
            for candidate_age in (None, 17):
                try:
                    manager.create(
                        CharacterDraft(owner="natasha", name="X", fictional_age=candidate_age, adult_eligible=True)
                    )
                except CharacterError:
                    age_refusals += 1
            ok = manager.create(
                CharacterDraft(owner="natasha", name="Ok", fictional_age=18, adult_eligible=True)
            )
            return age_refusals == 2 and ok.character.adult_eligible is True

    if kind == "character-duplicate":
        with _storage() as storage:
            manager = _manager(storage)
            source = manager.create(
                CharacterDraft(
                    owner="natasha",
                    name="Layla",
                    fictional_age=27,
                    personality={"warmth": 0.8},
                    profile_items=({"kind": "interest", "label": "poetry"},),
                )
            )
            copy = manager.duplicate(source.id)
            return (
                copy.id != source.id
                and copy.interests() == ("poetry",)
                and copy.personality is not None
                and copy.personality.warmth == 0.8
            )

    if kind == "character-export-import":
        with _storage() as storage:
            manager = _manager(storage)
            source = manager.create(
                CharacterDraft(
                    owner="natasha",
                    name="Layla",
                    fictional_age=27,
                    personality={"playfulness": 0.7},
                    profile_items=({"kind": "interest", "label": "poetry"},),
                )
            )
            target = Path(storage.root) / "layla.json"
            manager.export_character(source.id, target)
            imported = manager.import_character(target, name="Layla 2")
            return (
                imported.name == "Layla 2"
                and imported.interests() == ("poetry",)
                and imported.personality is not None
                and imported.personality.playfulness == 0.7
            )

    if kind == "negative-import":
        with _storage() as storage:
            manager = _manager(storage)
            bad = Path(storage.root) / "bad.json"
            bad.write_text('{"format": "something-else"}', encoding="utf-8")
            try:
                manager.import_character(bad)
            except CharacterError:
                return True
            return False

    if kind == "character-archive":
        with _storage() as storage:
            manager = _manager(storage)
            made = manager.create(CharacterDraft(owner="natasha", name="Layla", fictional_age=27))
            manager.archive(made.id)
            hidden = manager.list_characters() == []
            shown = len(manager.list_characters(include_archived=True)) == 1
            manager.restore_from_archive(made.id)
            return hidden and shown and len(manager.list_characters()) == 1

    if kind == "character-delete-retention":
        with _storage() as storage:
            manager = _manager(storage)
            made = manager.create(CharacterDraft(owner="natasha", name="Layla", fictional_age=27))
            delete_refused = False
            try:
                manager.delete(made.id, choice="whatever")
            except CharacterError:
                delete_refused = True
            report = manager.delete(made.id, choice=RETENTION_PURGE)
            return delete_refused and report.deactivated

    if kind == "human-passing":
        text = human_passing_identity("Layla", biography="a designer")
        return (
            "You are a person" in text
            and "never describe yourself as a program" in text
            and "a designer" in text
        )

    if kind == "safety-floor":
        return "18 or older" in SAFETY_FLOOR and "consent" in SAFETY_FLOOR

    if kind == "seed-examples":
        with _storage() as storage:
            manager = _manager(storage)
            examples = seed_examples(manager)
            if not examples:
                return False
            labelled = all(
                bundle.character.metadata.get("seed") == SEED_MARKER for bundle in examples
            )
            adults = all(
                isinstance(bundle.character.fictional_age, int)
                and bundle.character.fictional_age >= 18
                for bundle in examples
            )
            return labelled and adults

    # -- Phase 3: the model router and the evaluation facility ------------
    if kind == "router-resolves":
        with _router() as router:
            resolved = router.resolve("natasha-default")
            return resolved.key == "natasha-default" and resolved.available is True

    if kind == "negative-no-model":
        with _router(available=False) as router:
            resolved = router.resolve()
            health = router.status()
            return resolved.available is False and health.ready is False

    if kind == "negative-load-failure":
        with _router(fail_load=True) as router:
            result = router.load("natasha-default")
            return result.ok is False and "load failure" in result.detail

    if kind == "router-safe-settings":
        from core.romance.routing import SAFE_CONTEXT, SAFE_MAX_TOKENS

        with _router() as router:
            settings = router.safe_settings()
            return (
                settings.context == SAFE_CONTEXT
                and settings.max_tokens == SAFE_MAX_TOKENS
                and settings.gpu_layers == 0
            )

    if kind == "negative-eval-no-model":
        with _router(available=False) as router:
            evaluation = ModelEvaluator(router).evaluate()
            return evaluation.measured is False and evaluation.overall() == 0.0

    if kind == "eval-measured":
        with _router() as router:
            evaluation = ModelEvaluator(router, runner=lambda prompt: "a warm reply").evaluate()
            dimensions = {score.dimension for score in evaluation.scores}
            return (
                evaluation.measured is True
                and "english_quality" in dimensions
                and "latency" in dimensions
                and evaluation.overall() > 0.0
            )

    if kind == "router-cancel":
        import time

        with _router() as router:
            handle = router.begin_generation("natasha-default", timeout_seconds=5)
            first = router.should_stop(handle) is False
            handle.cancel()
            second = router.should_stop(handle) is True
            expired = router.begin_generation("natasha-default", timeout_seconds=0.01)
            time.sleep(0.02)
            third = router.should_stop(expired) is True
            return first and second and third

    # -- Phase 4: the conversation modes ---------------------------------
    if kind == "modes-distinct":
        from core.romance.modes import MODES, ModeContext, mode_prompt

        context = ModeContext(name="Layla")
        prompts = {key: mode_prompt(spec, context) for key, spec in MODES.items()}
        return len(set(prompts.values())) == len(MODES) and len(MODES) == 9

    if kind == "modes-safety-floor":
        from core.romance.modes import MODES, ModeContext, mode_prompt
        from core.romance.personality import SAFETY_FLOOR as floor

        context = ModeContext(name="Layla")
        return all(floor in mode_prompt(spec, context) for spec in MODES.values())

    if kind == "negative-only-adult-mode":
        from core.romance.modes import MODE_ADULT, MODES

        adult_keys = {key for key, spec in MODES.items() if spec.adult}
        return adult_keys == {MODE_ADULT}

    if kind == "negative-mode-change-preserves-history":
        from core.romance.conversation import ROLE_USER, ConversationManager

        with _storage() as storage:
            character_id = _new_character(storage).record_id
            chat = ConversationManager(storage)
            conversation = chat.start(character_id, mode="general")
            first = chat.append(conversation.record_id, "hello", role=ROLE_USER)
            chat.set_mode(conversation.record_id, "romantic")
            chat.append(conversation.record_id, "hi", role="assistant")
            turns = chat.messages(conversation.record_id)
            return (
                turns[0].record_id == first.record_id
                and turns[0].mode == "general"
                and turns[1].mode == "romantic"
            )

    if kind == "modes-unknown-fallback":
        from core.romance.modes import MODE_GENERAL, find_mode

        return find_mode("does-not-exist").key == MODE_GENERAL and find_mode(None).key == MODE_GENERAL

    # -- Phase 5: the relationship engine --------------------------------
    if kind == "relationship-bounded":
        from core.romance.relationship import MAX_STEP, RelationshipEngine

        with _storage() as storage:
            character_id = _new_character(storage).record_id
            relationship = RelationshipEngine(storage)
            change = relationship.observe(character_id, "personal_share", message_id="MSG-1")
            return bool(change.deltas) and all(
                abs(delta) <= MAX_STEP for delta in change.deltas.values()
            )

    if kind == "negative-no-random-change":
        from core.romance.relationship import RelationshipEngine

        with _storage() as storage:
            character_id = _new_character(storage).record_id
            relationship = RelationshipEngine(storage)
            before = relationship.snapshot(character_id).dimensions
            for _ in range(5):
                relationship.snapshot(character_id)
            return relationship.snapshot(character_id).dimensions == before

    if kind == "relationship-conflict-down":
        from core.romance.relationship import RelationshipEngine

        with _storage() as storage:
            character_id = _new_character(storage).record_id
            relationship = RelationshipEngine(storage)
            relationship.observe(character_id, "meaningful_conversation", message_id="MSG-1")
            relationship.observe(character_id, "meaningful_conversation", message_id="MSG-2")
            closeness_before = relationship.snapshot(character_id).dimensions["closeness"]
            relationship.observe(character_id, "conflict", message_id="MSG-3")
            return relationship.snapshot(character_id).dimensions["closeness"] < closeness_before

    if kind == "relationship-isolated":
        from core.romance.relationship import RelationshipEngine

        with _storage() as storage:
            first = _new_character(storage, name="Layla").record_id
            second = _new_character(storage, name="Karim").record_id
            relationship = RelationshipEngine(storage)
            relationship.observe(first, "personal_share", message_id="MSG-1")
            return (
                relationship.snapshot(first).dimensions["trust"] > 0.0
                and relationship.snapshot(second).dimensions["trust"] == 0.0
            )

    if kind == "relationship-explains":
        from core.romance.relationship import RelationshipEngine

        with _storage() as storage:
            character_id = _new_character(storage).record_id
            relationship = RelationshipEngine(storage)
            relationship.observe(
                character_id, "personal_share", message_id="MSG-1", title="Shared a secret"
            )
            text = relationship.explain(character_id)
            return "Shared a secret" in text

    # -- Phase 6: the memory system --------------------------------------
    if kind == "memory-isolated":
        from core.romance.memory import KIND_SEMANTIC, CompanionMemoryManager

        with _storage() as storage:
            first = _new_character(storage, name="Layla").record_id
            second = _new_character(storage, name="Karim").record_id
            memory = CompanionMemoryManager(storage)
            memory.add(first, KIND_SEMANTIC, "The user is a diver.")
            return (
                len(memory.memories(first)) == 1
                and memory.memories(second) == ()
                and memory.retrieve(second, "diver") == ()
            )

    if kind == "negative-memory-dedup":
        from core.romance.memory import KIND_SEMANTIC, CompanionMemoryManager

        with _storage() as storage:
            character_id = _new_character(storage).record_id
            memory = CompanionMemoryManager(storage)
            first = memory.add(character_id, KIND_SEMANTIC, "The user loves rain.")
            second = memory.add(character_id, KIND_SEMANTIC, "  the user loves rain.  ")
            return first.record_id == second.record_id and len(memory.memories(character_id)) == 1

    if kind == "memory-supersede":
        from core.romance.memory import KIND_SEMANTIC, CompanionMemoryManager

        with _storage() as storage:
            character_id = _new_character(storage).record_id
            memory = CompanionMemoryManager(storage)
            original = memory.add(character_id, KIND_SEMANTIC, "The user is a teacher.")
            replacement = memory.supersede(original.record_id, "The user is an engineer.")
            stored = memory.get(original.record_id)
            return (
                replacement.record_id != original.record_id
                and stored.content == "The user is a teacher."
                and stored.superseded_by == replacement.record_id
            )

    if kind == "memory-not-a-fact":
        from core.romance.memory import KIND_SEMANTIC, CompanionMemoryManager

        with _storage() as storage:
            character_id = _new_character(storage).record_id
            memory = CompanionMemoryManager(storage)
            stored = memory.add(character_id, KIND_SEMANTIC, "The user is a diver.")
            # A fresh memory is never assumed certain.
            return float(stored.confidence) < 1.0

    if kind == "memory-recall":
        from core.romance.memory import KIND_SEMANTIC, CompanionMemoryManager

        with _storage() as storage:
            character_id = _new_character(storage).record_id
            memory = CompanionMemoryManager(storage)
            memory.add(character_id, KIND_SEMANTIC, "The user is a diver.", importance=0.5)
            memory.add(character_id, KIND_SEMANTIC, "The user loves rain.", importance=0.5)
            top = memory.retrieve(character_id, "diver")
            empty = memory.retrieve(character_id, "diver", limit=0)
            return bool(top) and top[0].content == "The user is a diver." and empty == ()

    # -- Phase 7: dating and roleplay ------------------------------------
    if kind == "roleplay-fictional-default":
        from core.romance.roleplay import RoleplayEngine

        with _storage() as storage:
            roleplay = RoleplayEngine(storage)
            scenario = roleplay.create_scenario("A café")
            return scenario.is_real_place is False

    if kind == "negative-roleplay-no-duplicate-cast":
        from core.romance.roleplay import RoleplayEngine, RoleplayError

        with _storage() as storage:
            character_id = _new_character(storage).record_id
            roleplay = RoleplayEngine(storage)
            scenario = roleplay.create_scenario("Scene")
            roleplay.add_participant(scenario.record_id, character_id)
            try:
                roleplay.add_participant(scenario.record_id, character_id)
            except RoleplayError:
                return roleplay.cast(scenario.record_id) == (character_id,)
            return False

    if kind == "roleplay-branch":
        from core.romance.roleplay import RoleplayEngine

        with _storage() as storage:
            character_id = _new_character(storage).record_id
            roleplay = RoleplayEngine(storage)
            scenario = roleplay.create_scenario("Scene")
            roleplay.add_participant(scenario.record_id, character_id)
            original = roleplay.start(scenario.record_id)
            branch = roleplay.branch(original.record_id)
            return (
                branch.record_id != original.record_id
                and branch.branch_of == original.record_id
                and roleplay.get_session(original.record_id).branch_of == ""
            )

    if kind == "roleplay-session-lifecycle":
        from core.romance.roleplay import (
            STATE_ACTIVE,
            STATE_PAUSED,
            RoleplayEngine,
        )

        with _storage() as storage:
            character_id = _new_character(storage).record_id
            roleplay = RoleplayEngine(storage)
            scenario = roleplay.create_scenario("Scene")
            roleplay.add_participant(scenario.record_id, character_id)
            session = roleplay.start(scenario.record_id)
            paused = roleplay.pause(session.record_id).state == STATE_PAUSED
            resumed = roleplay.resume(session.record_id).state == STATE_ACTIVE
            roleplay.end(session.record_id)
            restarted = roleplay.restart(session.record_id)
            return paused and resumed and restarted.state == STATE_ACTIVE and restarted.record_id != session.record_id

    # -- Phase 8: voice --------------------------------------------------
    if kind == "voice-per-character":
        from core.romance.voice import CompanionVoiceAdapter

        with _storage() as storage:
            first = _new_character(storage, name="Layla").record_id
            second = _new_character(storage, name="Karim").record_id
            voice = CompanionVoiceAdapter(storage)
            voice.set_voice(first, voice_key="woman-1")
            # Each character keeps its own profile; the second is untouched.
            return (
                voice.profile(first).voice_key == "woman-1"
                and voice.profile(second).voice_key == ""
            )

    if kind == "negative-voice-no-invented-voice":
        from core.romance.voice import CompanionVoiceAdapter

        with _storage() as storage:
            voice = CompanionVoiceAdapter(storage)
            options = voice.available_voices()
            # Every offered voice carries a real key from an installed source, and
            # a made-up key is not offered.
            keys = {option.key for option in options}
            return "made-up-voice-key" not in keys and all(option.key for option in options)

    if kind == "voice-never-breaks-a-scene":
        from core.romance.voice import CompanionVoiceAdapter

        class _Boom:
            def set_voice(self, key: str) -> None:
                raise RuntimeError("no engine")

            def set_mood(self, mood: str) -> None:
                raise RuntimeError("no engine")

            def render(self, text: str) -> None:
                raise RuntimeError("no engine")

        with _storage() as storage:
            character_id = _new_character(storage).record_id
            voice = CompanionVoiceAdapter(storage, reader=_Boom())
            # A render that fails returns None rather than raising.
            return voice.read_aloud(character_id, "Hello.") is None

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
