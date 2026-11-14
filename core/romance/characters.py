"""The character library: create, edit, duplicate, import/export, archive, delete.

A **character** is one independent, persistent fictional identity
(:class:`~core.romance.model.CompanionCharacter`) with its own personality dials,
profile items, boundaries, voice settings, and model assignment. This module is
the one service that manages their life cycle, over the Phase-1 storage layer.

Three honesty rules the manager enforces:

* **Adults only, at the source.** A character may be marked ``adult_eligible``
  only when its fictional age is a known number of 18 or more. An unknown age is
  *age-ambiguous*, and the safety layer refuses adult content for it.
* **Deletion is a choice, not an accident.** :meth:`CharacterManager.delete`
  delegates to :meth:`~core.romance.storage.CompanionStorageService.delete_character`,
  so the caller must state what happens to the character's conversations and
  memories (keep / purge / anonymize).
* **Nothing is invented.** A duplicate copies the source's own fields; an import
  re-creates exactly what the exported file holds; a seed example is clearly
  labelled as an example.

The manager is dependency-free, imports no Tk and no model, and is the one engine
the Studio window and the CLI will both drive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.romance.model import (
    CharacterBoundary,
    CharacterPersonality,
    CharacterProfileItem,
    CharacterVoiceProfile,
    CompanionCharacter,
    CompanionModelProfile,
    utc_now,
)
from core.romance.personality import (
    DEFAULT_DIAL,
    DIALS,
    CharacterPersonalityEngine,
)

#: The file format a single-character export writes, and an import reads.
CHARACTER_EXPORT_FORMAT = "fayd-romance-character"
CHARACTER_EXPORT_VERSION = 1

#: The profile-item kinds a character may carry.
PROFILE_ITEM_KINDS: tuple[str, ...] = ("interest", "hobby", "preference", "style", "topic")

#: The default conversation mode a new character starts in.
DEFAULT_MODE = "general"

#: The character fields an import will restore from a file.
_CHARACTER_FIELDS: tuple[str, ...] = (
    "owner",
    "name",
    "nickname",
    "fictional_age",
    "adult_eligible",
    "gender",
    "biography",
    "nationality",
    "languages",
    "appearance",
    "avatar_path",
    "relationship_style",
    "default_mode",
    "model_key",
    "tags",
    "metadata",
)


class CharacterError(ValueError):
    """Raised when a character operation cannot be carried out as asked."""


@dataclass(frozen=True)
class CharacterDraft:
    """Everything needed to create (or import) a character in one value."""

    owner: str = "natasha"
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
    default_mode: str = DEFAULT_MODE
    model_key: str = ""
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    personality: dict[str, Any] = field(default_factory=dict)
    profile_items: tuple[dict[str, Any], ...] = ()
    boundaries: tuple[dict[str, Any], ...] = ()

    def character_fields(self) -> dict[str, Any]:
        """The fields that belong on the character record itself."""
        return {name: getattr(self, name) for name in _CHARACTER_FIELDS}


@dataclass(frozen=True)
class CharacterBundle:
    """A character and everything scoped to it, read together."""

    character: CompanionCharacter
    personality: CharacterPersonality | None = None
    profile_items: tuple[CharacterProfileItem, ...] = ()
    boundaries: tuple[CharacterBoundary, ...] = ()
    voice: CharacterVoiceProfile | None = None
    model_profile: CompanionModelProfile | None = None

    @property
    def id(self) -> str:
        return self.character.record_id

    @property
    def name(self) -> str:
        return self.character.name

    def interests(self) -> tuple[str, ...]:
        """The labels of the profile items, in creation order."""
        return tuple(item.label for item in self.profile_items if item.label)


@dataclass(frozen=True)
class CharacterSummary:
    """One row of the character library: the character plus its real counts."""

    character: CompanionCharacter
    conversation_count: int = 0
    memory_count: int = 0
    last_active: str = ""

    @property
    def id(self) -> str:
        return self.character.record_id

    @property
    def name(self) -> str:
        return self.character.name

    def matches(self, needle: str) -> bool:
        """Whether the summary carries ``needle`` in its name, nickname, or tags."""
        text = needle.strip().casefold()
        if not text:
            return True
        haystack = " ".join(
            (
                self.character.name,
                self.character.nickname,
                *self.character.tags,
            )
        ).casefold()
        return text in haystack


class CharacterManager:
    """Create, edit, duplicate, import/export, archive, and delete characters."""

    def __init__(self, storage: Any, *, actor: str = "system") -> None:
        self.storage = storage
        self.domain = storage.domain
        self.personality = CharacterPersonalityEngine()

    # -- create / read ----------------------------------------------------

    def create(self, draft: CharacterDraft) -> CharacterBundle:
        """Create a character (and its personality, items, and boundaries)."""
        name = draft.name.strip()
        if not name:
            raise CharacterError("a character needs a name")
        self._check_age(draft.fictional_age, draft.adult_eligible)
        personality_values = self.personality.validate(draft.personality)
        fields = draft.character_fields()
        fields["name"] = name
        character = self.domain.create("companion_characters", **fields)
        self._write_children(character.record_id, personality_values, draft)
        return self.get(character.record_id)

    def get(self, character_id: str) -> CharacterBundle:
        """Read a character and everything scoped to it."""
        character = self.domain.get("companion_characters", character_id)
        if character is None:
            raise CharacterError(f"no character with id {character_id!r}")
        return CharacterBundle(
            character=character,
            personality=self._personality_for(character_id),
            profile_items=tuple(self._children("character_profiles", character_id)),
            boundaries=tuple(self._children("character_boundaries", character_id)),
            voice=self._voice_for(character_id),
            model_profile=self._model_profile_for(character_id),
        )

    def exists(self, character_id: str) -> bool:
        return self.domain.get("companion_characters", character_id) is not None

    def list_characters(
        self,
        *,
        owner: str = "",
        include_archived: bool = False,
        include_inactive: bool = False,
        search: str = "",
        favorite: bool | None = None,
    ) -> list[CharacterSummary]:
        """The character library, newest first, with each row's real counts.

        The counts are gathered in **one pass** per supporting table (never a
        query per character), so a large library stays cheap.
        """
        characters = self.domain.list_records(
            "companion_characters", include_inactive=include_inactive
        )
        if owner:
            characters = [c for c in characters if c.owner == owner]
        if not include_archived:
            characters = [c for c in characters if not c.archived]
        if favorite is not None:
            characters = [c for c in characters if c.favorite is favorite]

        conversations = self._count_by("companion_conversations", "character_id", include_inactive)
        memories = self._count_by("companion_memories", "character_id", include_inactive)
        last_active = self._last_active_by_character()

        summaries = [
            CharacterSummary(
                character=character,
                conversation_count=conversations.get(character.record_id, 0),
                memory_count=memories.get(character.record_id, 0),
                last_active=last_active.get(character.record_id, ""),
            )
            for character in characters
        ]
        if search:
            summaries = [s for s in summaries if s.matches(search)]
        summaries.sort(key=lambda s: (s.character.created_at, s.character.record_id), reverse=True)
        return summaries

    # -- edit -------------------------------------------------------------

    def update(self, character_id: str, **changes: Any) -> CharacterBundle:
        """Edit a character's own fields (never its children)."""
        if "fictional_age" in changes or "adult_eligible" in changes:
            current = self.domain.get("companion_characters", character_id)
            if current is None:
                raise CharacterError(f"no character with id {character_id!r}")
            age = changes.get("fictional_age", current.fictional_age)
            eligible = changes.get("adult_eligible", current.adult_eligible)
            self._check_age(age, eligible)
        if "name" in changes and not str(changes["name"]).strip():
            raise CharacterError("a character needs a name")
        unknown = set(changes) - set(_CHARACTER_FIELDS)
        if unknown:
            raise CharacterError(f"cannot edit field(s): {', '.join(sorted(unknown))}")
        self.domain.update("companion_characters", character_id, **changes)
        return self.get(character_id)

    def set_personality(self, character_id: str, values: dict[str, Any]) -> CharacterBundle:
        """Set a character's personality dials (validated), creating the row if needed."""
        clean = self.personality.validate(values)
        row = self._personality_for(character_id)
        if row is None:
            self.domain.create("character_personalities", character_id=character_id, **clean)
        else:
            self.domain.update("character_personalities", row.record_id, **clean)
        return self.get(character_id)

    def add_profile_item(
        self, character_id: str, *, kind: str, label: str, value: str = ""
    ) -> CharacterBundle:
        """Add an interest/hobby/preference, refusing an unknown kind or blank label."""
        if kind not in PROFILE_ITEM_KINDS:
            raise CharacterError(
                f"unknown profile item kind {kind!r}; choose one of {', '.join(PROFILE_ITEM_KINDS)}"
            )
        if not label.strip():
            raise CharacterError("a profile item needs a label")
        self.domain.create(
            "character_profiles", character_id=character_id, kind=kind, label=label.strip(), value=value
        )
        return self.get(character_id)

    def remove_profile_item(self, character_id: str, item_id: str) -> CharacterBundle:
        self.domain.delete("character_profiles", item_id)
        return self.get(character_id)

    def add_boundary(self, character_id: str, *, value: str, kind: str = "topic") -> CharacterBundle:
        """Add an explicit conversation limit for a character."""
        if not value.strip():
            raise CharacterError("a boundary needs a value")
        self.domain.create("character_boundaries", character_id=character_id, kind=kind, value=value.strip())
        return self.get(character_id)

    def remove_boundary(self, character_id: str, boundary_id: str) -> CharacterBundle:
        self.domain.delete("character_boundaries", boundary_id)
        return self.get(character_id)

    def set_voice(self, character_id: str, **settings: Any) -> CharacterBundle:
        """Set a character's voice settings (engine, voice, language, prosody)."""
        row = self._voice_for(character_id)
        if row is None:
            self.domain.create("character_voice_profiles", character_id=character_id, **settings)
        else:
            self.domain.update("character_voice_profiles", row.record_id, **settings)
        return self.get(character_id)

    def set_model(self, character_id: str, *, model_key: str, **settings: Any) -> CharacterBundle:
        """Assign a language model (and optional generation settings) to a character."""
        rows = self.domain.list_records(
            "companion_model_profiles", where="scope = ? AND scope_id = ?", params=("character", character_id)
        )
        if rows:
            self.domain.update("companion_model_profiles", rows[0].record_id, model_key=model_key, **settings)
        else:
            self.domain.create(
                "companion_model_profiles", scope="character", scope_id=character_id, model_key=model_key, **settings
            )
        self.domain.update("companion_characters", character_id, model_key=model_key)
        return self.get(character_id)

    # -- duplicate / archive ----------------------------------------------

    def duplicate(self, character_id: str, *, name: str = "") -> CharacterBundle:
        """Copy a character, its personality, items, boundaries, voice, and model."""
        source = self.get(character_id)
        new_name = name.strip() or f"{source.character.name} (copy)"
        draft = CharacterDraft(
            owner=source.character.owner,
            name=new_name,
            nickname=source.character.nickname,
            fictional_age=source.character.fictional_age,
            adult_eligible=source.character.adult_eligible,
            gender=source.character.gender,
            biography=source.character.biography,
            nationality=source.character.nationality,
            languages=source.character.languages,
            appearance=source.character.appearance,
            avatar_path=source.character.avatar_path,
            relationship_style=source.character.relationship_style,
            default_mode=source.character.default_mode,
            model_key=source.character.model_key,
            tags=source.character.tags,
            metadata=dict(source.character.metadata),
            personality=self._personality_values(source.personality),
            profile_items=tuple(
                {"kind": item.kind, "label": item.label, "value": item.value}
                for item in source.profile_items
            ),
            boundaries=tuple(
                {"kind": b.kind, "value": b.value} for b in source.boundaries
            ),
        )
        bundle = self.create(draft)
        if source.voice is not None:
            self.domain.create(
                "character_voice_profiles",
                character_id=bundle.id,
                engine=source.voice.engine,
                voice_key=source.voice.voice_key,
                language=source.voice.language,
                dialect=source.voice.dialect,
                speed=source.voice.speed,
                pitch=source.voice.pitch,
                volume=source.voice.volume,
                mood=source.voice.mood,
                enabled=source.voice.enabled,
            )
        if source.model_profile is not None:
            self.set_model(
                bundle.id,
                model_key=source.model_profile.model_key,
                context_length=source.model_profile.context_length,
                max_tokens=source.model_profile.max_tokens,
                temperature=source.model_profile.temperature,
                top_p=source.model_profile.top_p,
            )
        return self.get(bundle.id)

    def archive(self, character_id: str) -> CharacterBundle:
        """Move a character to the archive (still present, hidden by default)."""
        self.domain.update("companion_characters", character_id, archived=True)
        return self.get(character_id)

    def restore_from_archive(self, character_id: str) -> CharacterBundle:
        self.domain.update("companion_characters", character_id, archived=False)
        return self.get(character_id)

    def set_favorite(self, character_id: str, favorite: bool = True) -> CharacterBundle:
        self.domain.update("companion_characters", character_id, favorite=bool(favorite))
        return self.get(character_id)

    # -- delete -----------------------------------------------------------

    def delete(self, character_id: str, *, choice: str) -> Any:
        """Delete a character with an explicit retention choice for its data."""
        if not self.exists(character_id):
            raise CharacterError(f"no character with id {character_id!r}")
        from core.romance.storage import RETENTION_CHOICES

        if choice not in RETENTION_CHOICES:
            raise CharacterError(
                f"unknown retention choice {choice!r}; choose one of {', '.join(RETENTION_CHOICES)}"
            )
        return self.storage.delete_character(character_id, choice=choice)

    # -- import / export --------------------------------------------------

    def export_character(self, character_id: str, destination: str | Path) -> Path:
        """Write one character (and its children) to a portable JSON file."""
        bundle = self.get(character_id)
        payload = {
            "format": CHARACTER_EXPORT_FORMAT,
            "version": CHARACTER_EXPORT_VERSION,
            "exported_at": utc_now(),
            "character": _character_to_dict(bundle.character),
            "personality": self._personality_values(bundle.personality),
            "profile_items": [
                {"kind": item.kind, "label": item.label, "value": item.value}
                for item in bundle.profile_items
            ],
            "boundaries": [{"kind": b.kind, "value": b.value} for b in bundle.boundaries],
            "voice": _voice_to_dict(bundle.voice) if bundle.voice else None,
            "model_profile": _model_profile_to_dict(bundle.model_profile)
            if bundle.model_profile
            else None,
        }
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def import_character(self, source: str | Path, *, name: str = "") -> CharacterBundle:
        """Create a character from a file :meth:`export_character` wrote.

        The imported character is **new** (a fresh id) and starts unarchived and
        un-favorited; the caller may rename it. A file that is not a character
        export is refused rather than half-read.
        """
        data = json.loads(Path(source).read_text(encoding="utf-8"))
        if data.get("format") != CHARACTER_EXPORT_FORMAT:
            raise CharacterError(f"{source} is not a Romance character export")
        raw = data.get("character") or {}
        draft = CharacterDraft(
            owner=str(raw.get("owner") or "natasha"),
            name=name.strip() or str(raw.get("name") or "Imported character"),
            nickname=str(raw.get("nickname") or ""),
            fictional_age=_as_optional_int(raw.get("fictional_age")),
            adult_eligible=bool(raw.get("adult_eligible")),
            gender=str(raw.get("gender") or ""),
            biography=str(raw.get("biography") or ""),
            nationality=str(raw.get("nationality") or ""),
            languages=tuple(raw.get("languages") or ()),
            appearance=str(raw.get("appearance") or ""),
            avatar_path=str(raw.get("avatar_path") or ""),
            relationship_style=str(raw.get("relationship_style") or ""),
            default_mode=str(raw.get("default_mode") or DEFAULT_MODE),
            model_key=str(raw.get("model_key") or ""),
            tags=tuple(raw.get("tags") or ()),
            metadata=dict(raw.get("metadata") or {}),
            personality=dict(data.get("personality") or {}),
            profile_items=tuple(data.get("profile_items") or ()),
            boundaries=tuple(data.get("boundaries") or ()),
        )
        bundle = self.create(draft)
        voice = data.get("voice")
        if isinstance(voice, dict):
            self.set_voice(
                bundle.id,
                engine=str(voice.get("engine") or "auto"),
                voice_key=str(voice.get("voice_key") or ""),
                language=str(voice.get("language") or ""),
                dialect=str(voice.get("dialect") or ""),
                speed=float(voice.get("speed") or 1.0),
                pitch=float(voice.get("pitch") or 1.0),
                volume=float(voice.get("volume") or 1.0),
                mood=str(voice.get("mood") or "neutral"),
                enabled=bool(voice.get("enabled", True)),
            )
        model = data.get("model_profile")
        if isinstance(model, dict) and model.get("model_key"):
            self.set_model(
                bundle.id,
                model_key=str(model["model_key"]),
                context_length=int(model.get("context_length") or 0),
                max_tokens=int(model.get("max_tokens") or 0),
                temperature=float(model.get("temperature") or 0.7),
                top_p=float(model.get("top_p") or 0.95),
            )
        return self.get(bundle.id)

    # -- internals --------------------------------------------------------

    def _check_age(self, age: Any, eligible: bool) -> None:
        """Refuse an adult-eligible character whose age is unknown or under 18."""
        if not eligible:
            return
        if not isinstance(age, int) or isinstance(age, bool):
            raise CharacterError(
                "an adult-eligible character needs a known age of 18 or more "
                "(an unknown age is age-ambiguous and cannot be adult-eligible)"
            )
        if age < 18:
            raise CharacterError("an adult-eligible character must be 18 or older")

    def _write_children(
        self,
        character_id: str,
        personality_values: dict[str, Any],
        draft: CharacterDraft,
    ) -> None:
        self.domain.create("character_personalities", character_id=character_id, **personality_values)
        for item in draft.profile_items:
            self.domain.create(
                "character_profiles",
                character_id=character_id,
                kind=str(item.get("kind") or "interest"),
                label=str(item.get("label") or ""),
                value=str(item.get("value") or ""),
            )
        for boundary in draft.boundaries:
            self.domain.create(
                "character_boundaries",
                character_id=character_id,
                kind=str(boundary.get("kind") or "topic"),
                value=str(boundary.get("value") or ""),
            )

    def _personality_for(self, character_id: str) -> CharacterPersonality | None:
        rows = self.domain.list_records(
            "character_personalities", where="character_id = ?", params=(character_id,)
        )
        return rows[0] if rows else None

    def _voice_for(self, character_id: str) -> CharacterVoiceProfile | None:
        rows = self.domain.list_records(
            "character_voice_profiles", where="character_id = ?", params=(character_id,)
        )
        return rows[0] if rows else None

    def _model_profile_for(self, character_id: str) -> CompanionModelProfile | None:
        rows = self.domain.list_records(
            "companion_model_profiles",
            where="scope = ? AND scope_id = ?",
            params=("character", character_id),
        )
        return rows[0] if rows else None

    def _children(self, entity: str, character_id: str) -> list[Any]:
        return self.domain.list_records(entity, where="character_id = ?", params=(character_id,))

    def _count_by(self, entity: str, column: str, include_inactive: bool) -> dict[str, int]:
        rows = self.domain.list_records(entity, include_inactive=include_inactive)
        counts: dict[str, int] = {}
        for row in rows:
            key = getattr(row, column, "")
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _last_active_by_character(self) -> dict[str, str]:
        rows = self.domain.list_records("companion_conversations")
        latest: dict[str, str] = {}
        for row in rows:
            stamp = row.last_active or row.updated_at
            current = latest.get(row.character_id, "")
            if stamp > current:
                latest[row.character_id] = stamp
        return latest

    def _personality_values(self, personality: CharacterPersonality | None) -> dict[str, Any]:
        if personality is None:
            return dict.fromkeys(DIALS, DEFAULT_DIAL)
        return {
            "introversion": personality.introversion,
            "playfulness": personality.playfulness,
            "warmth": personality.warmth,
            "expressiveness": personality.expressiveness,
            "confidence": personality.confidence,
            "humour": personality.humour,
            "initiative": personality.initiative,
            "affection": personality.affection,
            "romantic_intensity": personality.romantic_intensity,
            "formality": personality.formality,
            "response_length": personality.response_length,
            "dialect": personality.dialect,
            "traits": personality.traits,
            "instructions": personality.instructions,
        }


def _character_to_dict(character: CompanionCharacter) -> dict[str, Any]:
    return {
        "owner": character.owner,
        "name": character.name,
        "nickname": character.nickname,
        "fictional_age": character.fictional_age,
        "adult_eligible": character.adult_eligible,
        "gender": character.gender,
        "biography": character.biography,
        "nationality": character.nationality,
        "languages": list(character.languages),
        "appearance": character.appearance,
        "avatar_path": character.avatar_path,
        "relationship_style": character.relationship_style,
        "default_mode": character.default_mode,
        "model_key": character.model_key,
        "tags": list(character.tags),
        "metadata": character.metadata,
    }


def _voice_to_dict(voice: CharacterVoiceProfile) -> dict[str, Any]:
    return {
        "engine": voice.engine,
        "voice_key": voice.voice_key,
        "language": voice.language,
        "dialect": voice.dialect,
        "speed": voice.speed,
        "pitch": voice.pitch,
        "volume": voice.volume,
        "mood": voice.mood,
        "enabled": voice.enabled,
    }


def _model_profile_to_dict(profile: CompanionModelProfile) -> dict[str, Any]:
    return {
        "model_key": profile.model_key,
        "context_length": profile.context_length,
        "max_tokens": profile.max_tokens,
        "temperature": profile.temperature,
        "top_p": profile.top_p,
    }


def _as_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "CHARACTER_EXPORT_FORMAT",
    "CHARACTER_EXPORT_VERSION",
    "DEFAULT_MODE",
    "PROFILE_ITEM_KINDS",
    "CharacterBundle",
    "CharacterDraft",
    "CharacterError",
    "CharacterManager",
    "CharacterSummary",
]
