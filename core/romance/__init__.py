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
from core.romance.conversation import (
    ConversationError,
    ConversationManager,
    ConversationSummary,
    Turn,
)
from core.romance.database import (
    BackupReport,
    RomanceDatabase,
    RomanceDatabaseError,
    RomanceLoadReport,
)
from core.romance.domain import ENTITY_NAMES, RomanceDomain, RomanceDomainError
from core.romance.memory import (
    KIND_CHARACTER,
    KIND_EPISODIC,
    KIND_RELATIONSHIP,
    KIND_SEMANTIC,
    MEMORY_HEADER,
    MEMORY_KINDS,
    CompanionMemoryManager,
    MemoryError,
    MemoryExport,
    RecalledMemory,
)
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
from core.romance.model_eval import (
    MEASURED,
    NOT_MEASURED,
    ModelEvaluationReport,
    ModelEvaluator,
    Probe,
)
from core.romance.modes import (
    MODE_ORDER,
    MODES,
    ModeContext,
    ModeSpec,
    find_mode,
    mode_keys,
    mode_prompt,
    mode_summaries,
)
from core.romance.personality import (
    DIALS,
    RESPONSE_LENGTHS,
    SAFETY_FLOOR,
    CharacterPersonalityEngine,
    PersonalityError,
    human_passing_identity,
)
from core.romance.relationship import (
    DIMENSIONS,
    EVENT_EFFECTS,
    EVENT_KINDS,
    MAX_STEP,
    STAGES,
    RelationshipChange,
    RelationshipEngine,
    RelationshipError,
    RelationshipSnapshot,
)
from core.romance.repository import PREFIXES, RomanceRepository
from core.romance.routing import (
    SAFE_CONTEXT,
    SAFE_MAX_TOKENS,
    TASK_COMPANION,
    CompanionModelError,
    CompanionModelRouter,
    GenerationHandle,
    ModelHealth,
    ResolvedModel,
)
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
"CATEGORY_LABELS",
    "CHARACTER_EXPORT_FORMAT",
    "CHARACTER_EXPORT_VERSION",
"DEFAULT_MODE",
    "DIALS",
    "DIMENSIONS",
    "ENTITY_NAMES",
    "EVENT_EFFECTS",
    "EVENT_KINDS",
    "KIND_CHARACTER",
    "KIND_EPISODIC",
    "KIND_RELATIONSHIP",
    "KIND_SEMANTIC",
    "MAX_STEP",
    "MEASURED",
    "MEMORY_HEADER",
    "MEMORY_KINDS",
    "MIGRATIONS",
    "MODES",
    "MODE_ORDER",
    "NOT_MEASURED",
    "PREFIXES",
    "PROFILE_ITEM_KINDS",
    "RECORDS",
    "RESPONSE_LENGTHS",
    "RETENTION_ANONYMIZE",
    "RETENTION_CHOICES",
    "RETENTION_KEEP",
    "RETENTION_PURGE",
    "SAFETY_FLOOR",
    "SAFE_CONTEXT",
    "SAFE_MAX_TOKENS",
    "SCHEMA_VERSION",
"STAGES",
    "TABLES",
    "TASK_COMPANION",
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
    "CompanionMemoryManager",
    "CompanionMessage",
    "CompanionModelError",
    "CompanionModelProfile",
    "CompanionModelRouter",
    "CompanionSafetyEvent",
    "CompanionScenario",
    "CompanionSetting",
    "CompanionStorageService",
    "ConversationError",
    "ConversationManager",
    "ConversationSummary",
    "DeleteReport",
    "ExportBundle",
    "GenerationHandle",
    "MemoryError",
    "MemoryExport",
    "ModeContext",
    "ModeSpec",
    "ModelEvaluationReport",
    "ModelEvaluator",
    "ModelHealth",
    "PersonalityError",
    "Probe",
    "RecalledMemory",
    "RecordBase",
    "RelationshipChange",
    "RelationshipEngine",
    "RelationshipError",
    "RelationshipEventRecord",
    "RelationshipSnapshot",
    "RelationshipState",
    "ResolvedModel",
"RoleplayError",
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
    "Turn",
    "find_mode",
    "human_passing_identity",
    "make_id",
    "mode_keys",
    "mode_prompt",
    "mode_summaries",
    "utc_now",
]
