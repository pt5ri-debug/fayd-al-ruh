"""The Romance & Companion Studio record model.

Every Studio record is a small **frozen** dataclass with a stable string id, a
``created_at``/``updated_at`` audit pair, and an ``active`` **soft-delete** flag,
exactly like :mod:`core.workplace.model` -- the project's precedent for a
domain-module record store. The id is generated once (``make_id``) and never
reused, so a record that is deactivated keeps its identity and every foreign key
that points at it still resolves.

The records are deliberately plain data: no behaviour, no database, no Tk. The
SQL type of a column is derived from the field's own annotation
(:mod:`core.romance.schema`), so this module is the single place a table's shape
is written down, and the whole model is testable with no database file and no
display.

Nothing here loads a model or reaches the network.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

#: Prefixes keep an id readable in a log and let a bare id be recognised. An
#: unknown kind falls back to its own upper-cased name, so a later phase can mint
#: ids without editing this table.
_PREFIXES: dict[str, str] = {
    "character": "CHR",
    "personality": "PTY",
    "profile": "PRF",
    "boundary": "BND",
    "relationship_state": "RLS",
    "relationship_event": "RLE",
    "memory": "MEM",
    "conversation": "CNV",
    "message": "MSG",
    "scenario": "SCN",
    "participant": "SPT",
    "session": "SES",
    "model_profile": "MDP",
    "voice_profile": "VOP",
    "setting": "SET",
    "safety_event": "SAF",
    "audit": "AUD",
}


def utc_now() -> str:
    """The current UTC time as an ISO-8601 string (the one timestamp format)."""
    return datetime.now(UTC).isoformat()


def make_id(kind: str, *, prefix: str | None = None) -> str:
    """Return a stable, human-readable id for a record of ``kind``.

    The id is ``<PREFIX>-<hex>`` (e.g. ``CHR-3f9a1c2b``). It is generated once
    and never reused, so a record can be deactivated and still be referenced.
    """
    tag = prefix or _PREFIXES.get(kind, kind.upper()[:6] or "REC")
    return f"{tag}-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class RecordBase:
    """The columns every Studio record carries.

    ``record_id`` is the stable primary key; ``created_at``/``updated_at`` are the
    audit trail; ``active`` is the **soft** deletion flag -- a record is
    deactivated, never deleted, so history survives.
    """

    record_id: str
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    active: bool = True

    def touch(self, **changes: Any) -> Any:
        """Return a copy with ``changes`` applied and ``updated_at`` refreshed."""
        return replace(self, updated_at=utc_now(), **changes)

    def deactivate(self) -> Any:
        """Return a copy marked inactive (the soft delete)."""
        return self.touch(active=False)

    def activate(self) -> Any:
        """Return a copy marked active again."""
        return self.touch(active=True)


#: Anything with the base columns -- what the repository can store generically.
RomanceRecord = RecordBase


# --------------------------------------------------------------- characters


@dataclass(frozen=True)
class CompanionCharacter(RecordBase):
    """One companion character: an independent, persistent fictional identity.

    ``owner`` is the isolation key this character's memory and learning are
    scoped to (see :mod:`core.learning_isolation`). A Studio character references
    an existing persona *template* through this key rather than adding a new owner
    to the closed policy set, so isolation holds without touching a
    security-sensitive module.

    ``fictional_age`` is ``None`` when the character's age is unspecified, which
    the safety layer treats as age-ambiguous and refuses for adult content.
    ``adult_eligible`` is ``True`` only after an explicit, recorded confirmation.
    """

    owner: str = ""
    name: str = ""
    nickname: str = ""
    fictional_age: int | None = None
    adult_eligible: bool = False
    gender: str = ""
    biography: str = ""
    nationality: str = ""
    languages: tuple[str, ...] = ()
    appearance: str = ""
    avatar_path: str = ""
    relationship_style: str = ""
    default_mode: str = "general"
    model_key: str = ""
    tags: tuple[str, ...] = ()
    favorite: bool = False
    archived: bool = False
    profile_version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CharacterPersonality(RecordBase):
    """The character's editable personality dials (0..1) plus free instructions.

    Every dial is *guidance for how to speak*, never a widening of what the
    character may claim; the rendered block states that in its own text.
    """

    character_id: str = ""
    introversion: float = 0.5
    playfulness: float = 0.5
    warmth: float = 0.5
    expressiveness: float = 0.5
    confidence: float = 0.5
    humour: float = 0.5
    initiative: float = 0.5
    affection: float = 0.5
    romantic_intensity: float = 0.5
    formality: float = 0.5
    response_length: str = "balanced"
    dialect: str = ""
    traits: tuple[str, ...] = ()
    instructions: str = ""


@dataclass(frozen=True)
class CharacterProfileItem(RecordBase):
    """A free-form profile item: an interest, a hobby, a style, a preference."""

    character_id: str = ""
    kind: str = "interest"
    label: str = ""
    value: str = ""


@dataclass(frozen=True)
class CharacterBoundary(RecordBase):
    """An explicit, user-controlled conversation limit for a character."""

    character_id: str = ""
    kind: str = "topic"
    value: str = ""


# ------------------------------------------------------------- relationship


@dataclass(frozen=True)
class RelationshipState(RecordBase):
    """The scored relationship between one character and one user.

    The dimensions are bounded 0..1 and change **only** from an explicit event
    (see :class:`RelationshipEventRecord`), never at random.
    """

    character_id: str = ""
    user_id: str = "local"
    familiarity: float = 0.0
    trust: float = 0.0
    closeness: float = 0.0
    comfort: float = 0.0
    affection: float = 0.0
    shared_experiences: float = 0.0
    stability: float = 0.5
    stage: str = "stranger"


@dataclass(frozen=True)
class RelationshipEventRecord(RecordBase):
    """Why a relationship changed, with its evidence (append-only)."""

    character_id: str = ""
    kind: str = "milestone"
    title: str = ""
    detail: str = ""
    dimension: str = ""
    delta: float = 0.0
    message_id: str = ""
    conversation_id: str = ""
    occurred_at: str = field(default_factory=utc_now)


# ------------------------------------------------------------------- memory


@dataclass(frozen=True)
class CompanionMemory(RecordBase):
    """One durable companion memory.

    ``kind`` is one of ``episodic``/``semantic_personal``/``relationship``/
    ``character`` (short-term memory lives in the conversation window, not here).
    ``fictional`` marks a memory that belongs to a story rather than to reality,
    so the two can never be confused.
    """

    character_id: str = ""
    kind: str = "episodic"
    content: str = ""
    importance: float = 0.5
    confidence: float = 0.5
    source: str = "user"
    source_message_id: str = ""
    fictional: bool = False
    superseded_by: str = ""


# ------------------------------------------------------------- conversation


@dataclass(frozen=True)
class CompanionConversation(RecordBase):
    """One conversation; the mode and any scenario are conversation-scoped."""

    character_id: str = ""
    title: str = "Conversation"
    mode: str = "general"
    scenario_id: str = ""
    last_active: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class CompanionMessage(RecordBase):
    """One message. Append-only: a mode change never rewrites an old row.

    ``speaker`` names which character spoke in a multi-character scene, so a
    transcript can show who said what; ``mode`` records the mode in force when
    the message was written, so history stays honest after a mode change.
    """

    conversation_id: str = ""
    seq: int = 0
    role: str = "user"
    speaker: str = ""
    mode: str = ""
    content: str = ""


# ----------------------------------------------------------------- scenario


@dataclass(frozen=True)
class CompanionScenario(RecordBase):
    """A dating/roleplay scenario. Locations are fictional unless flagged real."""

    title: str = ""
    category: str = "first_meeting"
    description: str = ""
    location: str = ""
    setting_time: str = ""
    background: str = ""
    starting_situation: str = ""
    objectives: tuple[str, ...] = ()
    narrative_style: str = ""
    content_level: str = "general"
    boundaries: tuple[str, ...] = ()
    is_real_place: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioParticipant(RecordBase):
    """A character's role in a scenario."""

    scenario_id: str = ""
    character_id: str = ""
    role: str = ""
    order: int = 0


@dataclass(frozen=True)
class ScenarioSession(RecordBase):
    """A running roleplay: pause/resume/branch/save/restart."""

    scenario_id: str = ""
    conversation_id: str = ""
    state: str = "active"
    branch_of: str = ""


# -------------------------------------------------------------- profiles


@dataclass(frozen=True)
class CompanionModelProfile(RecordBase):
    """A model assignment for a character or a conversation mode."""

    scope: str = "character"
    scope_id: str = ""
    model_key: str = ""
    context_length: int = 4096
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95


@dataclass(frozen=True)
class CharacterVoiceProfile(RecordBase):
    """A character's voice settings, referencing the shared voice catalog by key."""

    character_id: str = ""
    engine: str = ""
    voice_key: str = ""
    language: str = ""
    dialect: str = ""
    speed: float = 1.0
    pitch: float = 0.0
    volume: float = 1.0
    mood: str = "neutral"
    enabled: bool = True


# ---------------------------------------------------------------- settings


@dataclass(frozen=True)
class CompanionSetting(RecordBase):
    """One Studio setting. ``record_id`` **is** the setting key."""

    value: str = ""


@dataclass(frozen=True)
class CompanionSafetyEvent(RecordBase):
    """An append-only safety event: an eligibility confirmation or a refusal."""

    character_id: str = ""
    conversation_id: str = ""
    kind: str = "refusal"
    reason: str = ""


@dataclass(frozen=True)
class CompanionAuditEntry(RecordBase):
    """One append-only audit row: who did what, to which record, and when."""

    actor: str = "system"
    action: str = ""
    entity: str = ""
    entity_id: str = ""
    detail: str = ""
    before: str = ""
    after: str = ""


#: Every record class this build knows, by table name. The schema builder and the
#: repository both read this, so a new table needs no change in either.
RECORDS: dict[str, type[RecordBase]] = {
    "companion_characters": CompanionCharacter,
    "character_personalities": CharacterPersonality,
    "character_profiles": CharacterProfileItem,
    "character_boundaries": CharacterBoundary,
    "relationship_states": RelationshipState,
    "relationship_events": RelationshipEventRecord,
    "companion_memories": CompanionMemory,
    "companion_conversations": CompanionConversation,
    "companion_messages": CompanionMessage,
    "companion_scenarios": CompanionScenario,
    "scenario_participants": ScenarioParticipant,
    "scenario_sessions": ScenarioSession,
    "companion_model_profiles": CompanionModelProfile,
    "character_voice_profiles": CharacterVoiceProfile,
    "companion_settings": CompanionSetting,
    "companion_safety_events": CompanionSafetyEvent,
    "companion_audit_entries": CompanionAuditEntry,
}


__all__ = [
    "RECORDS",
    "CharacterBoundary",
    "CharacterPersonality",
    "CharacterProfileItem",
    "CharacterVoiceProfile",
    "CompanionAuditEntry",
    "CompanionCharacter",
    "CompanionConversation",
    "CompanionMemory",
    "CompanionMessage",
    "CompanionModelProfile",
    "CompanionSafetyEvent",
    "CompanionScenario",
    "CompanionSetting",
    "RecordBase",
    "RelationshipEventRecord",
    "RelationshipState",
    "RomanceRecord",
    "ScenarioParticipant",
    "ScenarioSession",
    "make_id",
    "utc_now",
]
