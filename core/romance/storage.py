"""The Studio's storage service: paths, retention, export, and backup.

This is the one place that answers "where does the Studio keep its data" and
"what happens to a character's conversations and memories when it is deleted".
It is Tk-free, model-free, and network-free.

The data lives under the git-ignored ``data/romance`` folder
(:func:`core.paths.romance_dir`), beside the project's other writable state, so
retraining, learning, and the user's private conversations never dirty the working
tree or a program update.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.paths import romance_backups_dir, romance_db_path, romance_dir
from core.romance.database import BackupReport, RomanceDatabase
from core.romance.domain import RomanceDomain
from core.romance.repository import RomanceRepository

#: The choices a caller may make about a deleted character's data.
RETENTION_KEEP = "keep"          # keep conversations and memories (soft delete only)
RETENTION_PURGE = "purge"        # delete the character's conversations and memories
RETENTION_ANONYMIZE = "anonymize"  # keep the records but detach them from the name
RETENTION_CHOICES: tuple[str, ...] = (RETENTION_KEEP, RETENTION_PURGE, RETENTION_ANONYMIZE)


@dataclass
class DeleteReport:
    """What deleting a character actually did to its data."""

    character_id: str
    choice: str
    conversations_removed: int = 0
    memories_removed: int = 0
    deactivated: bool = False

    def describe(self) -> str:
        if self.choice == RETENTION_PURGE:
            return (
                f"Deleted {self.character_id}: removed {self.conversations_removed} "
                f"conversation(s) and {self.memories_removed} memory record(s)."
            )
        if self.choice == RETENTION_ANONYMIZE:
            return (
                f"Deleted {self.character_id}: kept {self.conversations_removed} "
                f"conversation(s) and {self.memories_removed} memory record(s), detached "
                "from the character's name."
            )
        return (
            f"Deleted {self.character_id}: conversations and memories kept "
            "(soft delete)."
        )


@dataclass
class ExportBundle:
    """A portable export of the Studio's data (or of one character)."""

    path: str
    tables: dict[str, int] = field(default_factory=dict)

    def describe(self) -> str:
        total = sum(self.tables.values())
        return f"Exported {total} record(s) across {len(self.tables)} table(s) to {self.path}."


class CompanionStorageService:
    """Paths, retention, export, and backup for the Studio."""

    def __init__(self, *, root: str | Path | None = None, actor: str = "system") -> None:
        self._root = Path(root) if root is not None else None
        self.db_path = self._resolve_db_path()
        self.database = RomanceDatabase(self.db_path)
        self.repository = RomanceRepository(self.database)
        self.domain = RomanceDomain(self.repository, actor=actor)

    # -- paths ------------------------------------------------------------

    def _resolve_db_path(self) -> Path:
        if self._root is not None:
            return Path(self._root) / "romance.db"
        return romance_db_path()

    def _resolve_backups_dir(self) -> Path:
        if self._root is not None:
            return Path(self._root) / "backups"
        return romance_backups_dir()

    @property
    def root(self) -> Path:
        return self._root if self._root is not None else romance_dir()

    # -- backup -----------------------------------------------------------

    def backup(self, destination: str | Path | None = None) -> BackupReport:
        """Write a consistent copy of the database (a timestamped one by default)."""
        if destination is None:
            from core.romance.model import utc_now

            stamp = utc_now().replace(":", "").replace("-", "").split(".")[0]
            destination = self._resolve_backups_dir() / f"romance-{stamp}.db"
        return self.database.backup(destination)

    def restore(self, source: str | Path) -> BackupReport:
        """Replace the database from a backup, validating the copy first."""
        return self.database.restore(source)

    # -- retention --------------------------------------------------------

    def delete_character(self, character_id: str, *, choice: str = RETENTION_KEEP) -> DeleteReport:
        """Delete a character, honouring the caller's data-retention choice.

        ``keep`` soft-deletes the character and leaves its conversations and
        memories intact (a restore brings it back whole). ``purge`` removes the
        character's conversations (and, by cascade, its messages) and its memories,
        then soft-deletes the character. ``anonymize`` keeps the rows but blanks
        the character's identifying fields, so the exchange survives without the
        name.
        """
        if choice not in RETENTION_CHOICES:
            from core.romance.domain import RomanceDomainError

            raise RomanceDomainError(
                f"Unknown retention choice {choice!r}; choose one of {', '.join(RETENTION_CHOICES)}."
            )
        report = DeleteReport(character_id=character_id, choice=choice)
        conversations = self.domain.list_records(
            "companion_conversations", where="character_id = ?", params=(character_id,)
        )
        memories = self.domain.list_records(
            "companion_memories", where="character_id = ?", params=(character_id,)
        )
        if choice == RETENTION_PURGE:
            for conversation in conversations:
                self.domain.delete("companion_conversations", conversation.record_id)
            for memory in memories:
                self.domain.delete("companion_memories", memory.record_id)
            report.conversations_removed = len(conversations)
            report.memories_removed = len(memories)
        elif choice == RETENTION_ANONYMIZE:
            report.conversations_removed = len(conversations)
            report.memories_removed = len(memories)
            self.domain.update(
                "companion_characters",
                character_id,
                name="(deleted)",
                nickname="",
                biography="",
                appearance="",
                avatar_path="",
            )
        try:
            self.domain.deactivate("companion_characters", character_id)
            report.deactivated = True
        except Exception:
            report.deactivated = False
        return report

    # -- settings ---------------------------------------------------------

    def get_setting(self, key: str, default: str = "") -> str:
        """Read one Studio setting (``record_id`` **is** the setting key)."""
        record = self.domain.get("companion_settings", key)
        return default if record is None else str(record.value)

    def set_setting(self, key: str, value: str) -> Any:
        """Write one Studio setting, creating it on first use (audited)."""
        from core.romance.model import CompanionSetting

        existing = self.domain.get("companion_settings", key)
        if existing is None:
            record = CompanionSetting(record_id=key, value=value)
            self.repository.save_record("companion_settings", record)
            self.repository.audit(
                actor=self.domain.actor, action="create", entity="companion_settings",
                entity_id=key,
            )
            return record
        return self.domain.update("companion_settings", key, value=value)

    # -- export -----------------------------------------------------------

    def export(self, destination: str | Path, *, character_id: str = "") -> ExportBundle:
        """Export the Studio's data (or one character's) to a JSON file.

        Nothing private is *added*: the export contains exactly the rows the
        database holds, so a caller can move a library between machines without a
        remote sync. A conversation's messages are included with their
        conversation.
        """
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        from core.romance.model import RECORDS

        payload: dict[str, list[dict[str, Any]]] = {}
        for table in RECORDS:
            rows = self.domain.list_records(
                table, include_inactive=True, limit=None
            )
            if character_id:
                rows = [
                    row for row in rows if getattr(row, "character_id", "") == character_id
                    or getattr(row, "record_id", "") == character_id
                ]
            payload[table] = [dataclasses_asdict(row) for row in rows]
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return ExportBundle(path=str(target), tables={k: len(v) for k, v in payload.items()})


def dataclasses_asdict(record: Any) -> dict[str, Any]:
    """A record as a plain dict (tuples become lists for JSON)."""
    import dataclasses

    data = dataclasses.asdict(record)
    return data


__all__ = [
    "RETENTION_ANONYMIZE",
    "RETENTION_CHOICES",
    "RETENTION_KEEP",
    "RETENTION_PURGE",
    "CompanionStorageService",
    "DeleteReport",
    "ExportBundle",
]
