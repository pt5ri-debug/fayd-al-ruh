"""The Romance & Companion Studio: local-first companion, dating, and roleplay.

This package is the Studio's **domain layer** -- the record model, the versioned
SQLite store, the typed repository, the audited domain service, and the storage
service. Later phases add the character library, the personality engine, the
conversation modes, the relationship engine, the memory manager, the roleplay
engine, the model router, the voice adapter, the safety manager, and the
orchestrator, each as its own module beside these.

It is deliberately built like :mod:`core.workplace`: a domain module over the
project's *existing* machinery (the model manager, the speech layer, the memory
and isolation policies, the GUI component library), never a second copy of any of
them. Nothing here loads a model or reaches the network on import.

Public surface for Phase 1 (foundation and database)::

    from core.romance import CompanionStorageService

    storage = CompanionStorageService()
    character = storage.domain.create("companion_characters", owner="natasha", name="Layla")
"""

from __future__ import annotations

from core.romance.characters import (
    CHARACTER_EXPORT_FORMAT,
    CHARACTER_EXPORT_VERSION,
    DEFAULT_MODE,
    PROFILE_ITEM_KINDS,
    CharacterBundle,
    CharacterDraft,
    CharacterError,
    CharacterManager,
    CharacterSummary,
)
from core.romance.database import (
    BackupReport,
    RomanceDatabase,
    RomanceDatabaseError,
    RomanceLoadReport,
)
from core.romance.domain import ENTITY_NAMES, RomanceDomain, RomanceDomainError
from core.romance.model import (
    RECORDS,
    CharacterBoundary,
    CharacterPersonality,
    CharacterProfileItem,
    CharacterVoiceProfile,
    CompanionAuditEntry,
    CompanionCharacter,
    CompanionConversation,
    CompanionMemory,
    CompanionMessage,
    CompanionModelProfile,
    CompanionSafetyEvent,
    CompanionScenario,
    CompanionSetting,
    RecordBase,
    RelationshipEventRecord,
    RelationshipState,
    RomanceRecord,
    ScenarioParticipant,
    ScenarioSession,
    make_id,
    utc_now,
)
from core.romance.personality import (
    DIALS,
    RESPONSE_LENGTHS,
    SAFETY_FLOOR,
    CharacterPersonalityEngine,
    PersonalityError,
    human_passing_identity,
)
from core.romance.repository import PREFIXES, RomanceRepository
from core.romance.schema import MIGRATIONS, SCHEMA_VERSION, TABLES, Column, TableSpec
from core.romance.storage import (
    RETENTION_ANONYMIZE,
    RETENTION_CHOICES,
    RETENTION_KEEP,
    RETENTION_PURGE,
    CompanionStorageService,
    DeleteReport,
    ExportBundle,
)

__all__ = [
    "CHARACTER_EXPORT_FORMAT",
    "CHARACTER_EXPORT_VERSION",
    "DEFAULT_MODE",
    "DIALS",
    "ENTITY_NAMES",
    "MIGRATIONS",
    "PREFIXES",
    "PROFILE_ITEM_KINDS",
    "RECORDS",
    "RESPONSE_LENGTHS",
    "RETENTION_ANONYMIZE",
    "RETENTION_CHOICES",
    "RETENTION_KEEP",
    "RETENTION_PURGE",
    "SAFETY_FLOOR",
    "SCHEMA_VERSION",
    "TABLES",
    "BackupReport",
    "CharacterBoundary",
    "CharacterBundle",
    "CharacterDraft",
    "CharacterError",
    "CharacterManager",
    "CharacterPersonality",
    "CharacterPersonalityEngine",
    "CharacterProfileItem",
    "CharacterSummary",
    "CharacterVoiceProfile",
    "Column",
    "CompanionAuditEntry",
    "CompanionCharacter",
    "CompanionConversation",
    "CompanionMemory",
    "CompanionMessage",
    "CompanionModelProfile",
    "CompanionSafetyEvent",
    "CompanionScenario",
    "CompanionSetting",
    "CompanionStorageService",
    "DeleteReport",
    "ExportBundle",
    "PersonalityError",
    "RecordBase",
    "RelationshipEventRecord",
    "RelationshipState",
    "RomanceDatabase",
    "RomanceDatabaseError",
    "RomanceDomain",
    "RomanceDomainError",
    "RomanceLoadReport",
    "RomanceRecord",
    "RomanceRepository",
    "ScenarioParticipant",
    "ScenarioSession",
    "TableSpec",
    "human_passing_identity",
    "make_id",
    "utc_now",
]
