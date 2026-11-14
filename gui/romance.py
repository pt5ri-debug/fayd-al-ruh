"""The Romance & Companion Studio service: the Tk-free half of the Studio window.

This is where the Studio's work actually happens -- open the store, list the
characters, send a turn, read the relationship, manage memories, drive a
scenario -- with no display, so the whole workflow is testable on a fresh clone.
The window and the CLI both drive *this* service, so they can never disagree
about what the Studio holds.

It is a thin, honest layer over :mod:`core.romance`:

* the store is :class:`core.romance.storage.CompanionStorageService`;
* characters, conversations, memory, relationships, roleplay, and voice are the
  Phase-2..8 engines;
* the one turn is :class:`core.romance.orchestrator.CompanionOrchestrator`;
* the model is the project's own Model Manager, through
  :class:`core.romance.routing.CompanionModelRouter`.

Everything is local: nothing here opens a network connection. A missing model or
voice engine is reported with its real next step, never pretended.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.romance import CompanionStorageService
from core.romance.characters import CharacterDraft, CharacterManager
from core.romance.conversation import ConversationManager
from core.romance.memory import CompanionMemoryManager
from core.romance.modes import MODE_GENERAL, mode_summaries
from core.romance.orchestrator import CompanionOrchestrator, TurnResult
from core.romance.relationship import RelationshipEngine
from core.romance.roleplay import RoleplayEngine
from core.romance.routing import CompanionModelRouter
from core.romance.voice import CompanionVoiceAdapter

DEFAULT_CHARACTER_OWNER = "natasha"

#: The Studio settings keys and their human labels (the *Settings* window).
SETTING_DEFAULT_MODEL = "default_model_key"
SETTING_DEFAULT_MODE = "default_mode"
SETTING_DEFAULT_CONTENT_LEVEL = "default_content_level"
SETTING_CONTEXT_LENGTH = "context_length"
SETTING_MAX_TOKENS = "max_tokens"
SETTING_TEMPERATURE = "temperature"
SETTING_MEMORY_LIMIT = "memory_limit"
SETTING_VOICE_ENABLED = "voice_enabled"
SETTING_VOICE_ENGINE = "voice_engine"
SETTING_INTERFACE_LANGUAGE = "interface_language"
SETTING_ADULT_MODE = "adult_mode_enabled"

SETTING_LABELS: tuple[tuple[str, str, str], ...] = (
    (SETTING_DEFAULT_MODEL, "Default language model", "the model a new character starts on"),
    (SETTING_DEFAULT_MODE, "Default conversation mode", "the mode a new conversation opens in"),
    (SETTING_DEFAULT_CONTENT_LEVEL, "Default content level", "general, romantic, or adult"),
    (SETTING_CONTEXT_LENGTH, "Context length (tokens)", "how much of the conversation the model reads"),
    (SETTING_MAX_TOKENS, "Answer limit (tokens)", "the longest answer the model may write"),
    (SETTING_TEMPERATURE, "Temperature", "0 = focused, 1 = more varied"),
    (SETTING_MEMORY_LIMIT, "Memories recalled per turn", "how many stored memories the prompt may use"),
    (SETTING_VOICE_ENABLED, "Speak answers aloud", "on or off for new characters"),
    (SETTING_VOICE_ENGINE, "Default voice engine", "auto, espeak, piper, qwen3-tts, or audio.cpp"),
    (SETTING_INTERFACE_LANGUAGE, "Interface language", "en or ar"),
)

#: The Studio setting keys that hold a numeric value, with their safe bounds.
_NUMERIC_SETTINGS: dict[str, tuple[float, float, float]] = {
    SETTING_CONTEXT_LENGTH: (512.0, 32768.0, 4096.0),
    SETTING_MAX_TOKENS: (32.0, 4096.0, 512.0),
    SETTING_TEMPERATURE: (0.0, 2.0, 0.8),
    SETTING_MEMORY_LIMIT: (0.0, 20.0, 6.0),
}

#: The Studio setting keys that hold an on/off value.
_BOOL_SETTINGS: tuple[str, ...] = (SETTING_VOICE_ENABLED,)


def _clamp_setting(key: str, value: str) -> str:
    """Clamp a numeric setting to its documented range; pass anything else through."""
    spec = _NUMERIC_SETTINGS.get(key)
    if spec is None:
        return value
    low, high, _default = spec
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _format_number(spec[2])
    return _format_number(max(low, min(high, number)))


def _format_number(value: float) -> str:
    """A whole number reads as ``4096``, a fractional one keeps its digits."""
    return str(int(value)) if float(value).is_integer() else str(value)


def _default_setting(key: str) -> str:
    """The documented default of a Studio setting, so a fresh Studio is sensible."""
    if key in _NUMERIC_SETTINGS:
        return _format_number(_NUMERIC_SETTINGS[key][2])
    if key in _BOOL_SETTINGS:
        return "off"
    if key == SETTING_DEFAULT_MODE:
        return MODE_GENERAL
    if key == SETTING_DEFAULT_CONTENT_LEVEL:
        return "general"
    if key == SETTING_VOICE_ENGINE:
        return "auto"
    if key == SETTING_INTERFACE_LANGUAGE:
        return "en"
    return ""


@dataclass(frozen=True)
class ModelChoice:
    """One language model a character may be assigned, with its honest readiness."""

    key: str
    title: str
    engine: str = ""
    available: bool = False
    detail: str = ""

    def label(self) -> str:
        mark = "●" if self.available else "○"
        return f"{mark} {self.title or self.key}" + (f"  · {self.engine}" if self.engine else "")


@dataclass(frozen=True)
class StudioStatus:
    """An honest snapshot of the Studio: what exists and what can run."""

    character_count: int = 0
    conversation_count: int = 0
    memory_count: int = 0
    scenario_count: int = 0
    model_ready: bool = False
    model_key: str = ""
    model_note: str = ""
    voice_ready: bool = False
    voice_note: str = ""
    local_only: bool = True

    def describe(self) -> str:
        lines = [
            f"Characters: {self.character_count}",
            f"Conversations: {self.conversation_count}",
            f"Memories: {self.memory_count}",
            f"Scenarios: {self.scenario_count}",
            f"Language model: {self.model_note or ('ready' if self.model_ready else 'none')}",
            f"Voice: {self.voice_note or ('ready' if self.voice_ready else 'none')}",
            f"Local only: {'yes' if self.local_only else 'NO'}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class CharacterRow:
    """One row of the character library, for a list widget."""

    character_id: str
    name: str
    nickname: str = ""
    favorite: bool = False
    archived: bool = False
    conversation_count: int = 0
    memory_count: int = 0
    last_active: str = ""
    stage_title: str = ""

    def label(self) -> str:
        star = "★ " if self.favorite else ""
        return f"{star}{self.name}" + (f" ({self.nickname})" if self.nickname else "")


class RomanceStudioService:
    """The Studio's one engine, driven by the window and the CLI alike."""

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        router: CompanionModelRouter | None = None,
        safety: Any = None,
        generator: Any = None,
    ) -> None:
        self.storage = CompanionStorageService(root=root)
        self.characters = CharacterManager(self.storage)
        self.memory = CompanionMemoryManager(self.storage)
        self.conversations = ConversationManager(
            self.storage, characters=self.characters, memory_reader=self._memory_block
        )
        self.relationships = RelationshipEngine(self.storage)
        self.roleplay = RoleplayEngine(self.storage)
        self.voice = CompanionVoiceAdapter(self.storage)
        self.router = router
        self._router_built = router is not None
        if safety is None:
            from core.romance.safety import CompanionSafetyManager

            safety = CompanionSafetyManager(self.storage)
        self.safety = safety
        self.orchestrator = CompanionOrchestrator(
            self.storage,
            characters=self.characters,
            conversations=self.conversations,
            relationships=self.relationships,
            memory=self.memory,
            router=router,
            safety=self.safety,
            roleplay=self.roleplay,
            generator=generator,
        )

    # -- memory reader ----------------------------------------------------

    def model_router(self) -> CompanionModelRouter | None:
        """The Studio's model router, built lazily on the project's Model Manager.

        Built only when the caller supplied no router, so a headless test that
        injects a generator (or nothing) never touches the real model subsystem.
        A missing Model Manager returns ``None`` honestly rather than raising.
        """
        if self._router_built:
            return self.router
        try:
            self.router = CompanionModelRouter()
        except Exception:
            self.router = None
        self._router_built = True
        self.orchestrator.router = self.router
        return self.router

    def _memory_block(self, character_id: str, conversation_id: str) -> str:
        try:
            turns = self.conversations.messages(conversation_id, limit=3)
        except Exception:
            turns = ()
        query = " ".join(t.content for t in turns if getattr(t, "role", "") == "user")
        return self.memory.reference_block(character_id, query)

    # -- status -----------------------------------------------------------

    def status(self) -> StudioStatus:
        characters = self.characters.list_characters(include_archived=True)
        conversations = self.conversations.list_conversations()
        memories = self.storage.domain.list_records("companion_memories")
        scenarios = self.roleplay.scenarios()
        model_ready, model_key, model_note = self._model_status()
        voice_ready, voice_note = self._voice_status()
        local_only = True
        try:
            local_only = bool(self.safety.local_only())
        except Exception:
            local_only = True
        return StudioStatus(
            character_count=len(characters),
            conversation_count=len(conversations),
            memory_count=len(memories),
            scenario_count=len(scenarios),
            model_ready=model_ready,
            model_key=model_key,
            model_note=model_note,
            voice_ready=voice_ready,
            voice_note=voice_note,
            local_only=local_only,
        )

    def _model_status(self) -> tuple[bool, str, str]:
        router = self.model_router()
        if router is None:
            return False, "", "no model manager is available in this session"
        try:
            resolved = router.resolve()
        except Exception as error:
            return False, "", str(error)
        # A model is "ready" only when it is marked available **and** a real file
        # is on disk -- a catalog entry that needs training is not a runnable model.
        ready = bool(resolved.available and resolved.path is not None)
        return ready, resolved.key, resolved.detail or resolved.describe()

    def _voice_status(self) -> tuple[bool, str]:
        try:
            from core.speech import speech_available

            available = bool(speech_available())
        except Exception:
            available = False
        count = len(self.voice.available_voices())
        if available:
            return True, f"{count} voice(s) available"
        return False, "no speech engine is installed (install espeak-ng, Piper, or Qwen3-TTS)"

    # -- characters -------------------------------------------------------

    def character_rows(
        self,
        *,
        include_archived: bool = False,
        search: str = "",
        favorite: bool | None = None,
    ) -> list[CharacterRow]:
        rows: list[CharacterRow] = []
        for summary in self.characters.list_characters(
            include_archived=include_archived, search=search, favorite=favorite
        ):
            stage = ""
            try:
                stage = self.relationships.snapshot(summary.id).stage_title
            except Exception:
                stage = ""
            rows.append(
                CharacterRow(
                    character_id=summary.id,
                    name=summary.name,
                    nickname=getattr(summary.character, "nickname", ""),
                    favorite=bool(getattr(summary.character, "favorite", False)),
                    archived=bool(getattr(summary.character, "archived", False)),
                    conversation_count=summary.conversation_count,
                    memory_count=summary.memory_count,
                    last_active=summary.last_active,
                    stage_title=stage,
                )
            )
        return rows

    def character(self, character_id: str) -> Any:
        return self.characters.get(character_id)

    def discover_examples(self) -> tuple[Any, ...]:
        """The clearly-labelled example companions this build ships.

        A read of the shipped ``EXAMPLES`` set -- **nothing is created** here, so
        browsing the library never writes to the database. An example already
        present (by name) is marked so the window offers *Add* only for one the
        user does not yet have.
        """
        from core.romance.seed import EXAMPLES

        existing = {
            summary.name.casefold()
            for summary in self.characters.list_characters(include_archived=True)
        }
        rows: list[tuple[Any, bool]] = []
        for draft in EXAMPLES:
            rows.append((draft, draft.name.casefold() in existing))
        return tuple(rows)

    def add_example(self, name: str) -> Any:
        """Add one shipped example companion by name (the *Discover* action).

        Creates **only** the chosen draft through the same ``CharacterManager.create``
        the seeding path uses, so adding one example never adds the others. An
        example already present is returned rather than duplicated. Raises
        ``ValueError`` for a name the build does not ship.
        """
        from core.romance.seed import EXAMPLES

        wanted = name.casefold()
        draft = next((d for d in EXAMPLES if d.name.casefold() == wanted), None)
        if draft is None:
            raise ValueError(f"no example companion named {name!r}")
        for summary in self.characters.list_characters(include_archived=True):
            if summary.name.casefold() == wanted:
                return self.characters.get(summary.id)
        return self.characters.create(draft)

    def create_character(
        self,
        name: str,
        *,
        fictional_age: int | None = None,
        nickname: str = "",
        gender: str = "",
        biography: str = "",
        nationality: str = "",
        languages: tuple[str, ...] = (),
        appearance: str = "",
        relationship_style: str = "",
        default_mode: str = MODE_GENERAL,
        model_key: str = "",
        personality: dict[str, Any] | None = None,
        interests: tuple[str, ...] = (),
        boundaries: tuple[str, ...] = (),
        owner: str = DEFAULT_CHARACTER_OWNER,
    ) -> Any:
        draft = CharacterDraft(
            owner=owner,
            name=name,
            nickname=nickname,
            fictional_age=fictional_age,
            gender=gender,
            biography=biography,
            nationality=nationality,
            languages=tuple(languages),
            appearance=appearance,
            relationship_style=relationship_style,
            default_mode=default_mode,
            model_key=model_key,
            personality=dict(personality or {}),
            profile_items=tuple({"label": label, "kind": "interest"} for label in interests),
            boundaries=tuple({"value": value} for value in boundaries),
        )
        return self.characters.create(draft)

    def update_character(self, character_id: str, **changes: Any) -> Any:
        return self.characters.update(character_id, **changes)

    def duplicate_character(self, character_id: str, *, name: str = "") -> Any:
        return self.characters.duplicate(character_id, name=name)

    def archive_character(self, character_id: str) -> Any:
        return self.characters.archive(character_id)

    def restore_character(self, character_id: str) -> Any:
        return self.characters.restore_from_archive(character_id)

    def set_favorite(self, character_id: str, favorite: bool = True) -> Any:
        return self.characters.set_favorite(character_id, favorite)

    def delete_character(self, character_id: str, *, choice: str = "keep") -> Any:
        """Delete a character, following an explicit data-retention choice."""
        return self.characters.delete(character_id, choice=choice)

    def export_character(self, character_id: str, destination: str | Path) -> Path:
        return self.characters.export_character(character_id, destination)

    def import_character(self, source: str | Path, *, name: str = "") -> Any:
        return self.characters.import_character(source, name=name)

    # -- conversations ----------------------------------------------------

    def start_conversation(
        self, character_id: str, *, title: str = "Conversation", mode: str = MODE_GENERAL
    ) -> Any:
        return self.conversations.start(character_id, title=title, mode=mode)

    def conversations_for(self, character_id: str) -> tuple[Any, ...]:
        return self.conversations.list_conversations(character_id=character_id)

    def recent_conversations(self, *, limit: int = 20) -> tuple[Any, ...]:
        return self.conversations.list_conversations(limit=limit)

    def messages(self, conversation_id: str, *, limit: int | None = None) -> tuple[Any, ...]:
        return self.conversations.messages(conversation_id, limit=limit)

    def resume(self, conversation_id: str) -> tuple[Any, tuple[Any, ...]]:
        return self.conversations.resume(conversation_id)

    def set_mode(self, conversation_id: str, mode: str) -> Any:
        return self.conversations.set_mode(conversation_id, mode)

    def delete_conversation(self, conversation_id: str) -> None:
        """Deactivate a conversation and its messages (a soft delete)."""
        domain = self.storage.domain
        for message in domain.list_records(
            "companion_messages", where="conversation_id = ?", params=(conversation_id,)
        ):
            domain.deactivate("companion_messages", message.record_id)
        domain.deactivate("companion_conversations", conversation_id)

    def send(self, conversation_id: str, message: str, *, mode: str = "") -> TurnResult:
        return self.orchestrator.send(conversation_id, message, mode=mode)

    def send_multi(
        self,
        conversation_id: str,
        message: str,
        *,
        cast: tuple[str, ...] = (),
        scenario: str = "",
    ) -> TurnResult:
        return self.orchestrator.send_multi(
            conversation_id, message, cast=cast, scenario=scenario
        )

    def modes(self) -> tuple[tuple[str, str, str], ...]:
        return mode_summaries()

    # -- relationship -----------------------------------------------------

    def relationship(self, character_id: str) -> Any:
        return self.relationships.snapshot(character_id)

    def relationship_history(self, character_id: str) -> tuple[Any, ...]:
        return self.relationships.events(character_id)

    def reset_relationship(self, character_id: str) -> Any:
        return self.relationships.reset(character_id)

    # -- memory -----------------------------------------------------------

    def memories(self, character_id: str) -> tuple[Any, ...]:
        return self.memory.memories(character_id)

    def recall(self, character_id: str, query: str = "", *, limit: int = 6) -> tuple[Any, ...]:
        return self.memory.retrieve(character_id, query, limit=limit)

    def add_memory(
        self,
        character_id: str,
        kind: str,
        content: str,
        *,
        importance: float = 0.5,
        source: str = "user",
        source_message_id: str = "",
        fictional: bool = False,
    ) -> Any:
        return self.memory.add(
            character_id,
            kind,
            content,
            importance=importance,
            source=source,
            source_message_id=source_message_id,
            fictional=fictional,
        )

    def edit_memory(self, memory_id: str, **changes: Any) -> Any:
        return self.memory.edit(memory_id, **changes)

    def forget_memory(self, memory_id: str) -> Any:
        return self.memory.forget(memory_id)

    def export_memories(self, character_id: str) -> Any:
        return self.memory.export(character_id)

    def voice_engines(self) -> tuple[str, ...]:
        """The voice engines the Studio can name (``auto`` first)."""
        from core.romance.voice import ENGINE_NAMES

        return tuple(ENGINE_NAMES)

    # -- models -----------------------------------------------------------

    def model_choices(self) -> tuple[ModelChoice, ...]:
        """Every language model a character may be assigned, with its readiness.

        Read from the project's own catalog through the router, so the Studio can
        never disagree with the Model Manager about what exists. With no Model
        Manager available the result is an empty tuple (the window then says so).
        """
        router = self.model_router()
        if router is None:
            return ()
        try:
            return tuple(
                ModelChoice(
                    key=resolved.key,
                    title=resolved.title or resolved.key,
                    engine=resolved.engine,
                    available=bool(resolved.available and resolved.path is not None),
                    detail=resolved.detail,
                )
                for resolved in router.choices()
            )
        except Exception:
            return ()

    def model_key(self, character_id: str) -> str:
        """The model a character is assigned, or ``""`` for the Studio default."""
        try:
            return str(self.characters.get(character_id).character.model_key or "")
        except Exception:
            return ""

    def assign_model(self, character_id: str, model_key: str) -> Any:
        """Assign a language model to a character (persisted on its record)."""
        return self.characters.set_model(character_id, model_key=model_key)

    # -- studio settings --------------------------------------------------

    def settings(self) -> dict[str, str]:
        """Every Studio setting, with its documented default filled in."""
        values: dict[str, str] = {}
        for key, _label, _hint in SETTING_LABELS:
            values[key] = self.storage.get_setting(key, _default_setting(key))
        return values

    def setting(self, key: str, default: str = "") -> str:
        return self.storage.get_setting(key, default or _default_setting(key))

    def set_setting(self, key: str, value: str) -> str:
        """Write one Studio setting, clamped to its documented range."""
        if key not in {k for k, _l, _h in SETTING_LABELS}:
            raise ValueError(f"unknown Studio setting {key!r}")
        clean = _clamp_setting(key, str(value))
        self.storage.set_setting(key, clean)
        return clean

    def update_settings(self, **values: str) -> dict[str, str]:
        """Write several settings at once and return the whole map."""
        for key, value in values.items():
            self.set_setting(key, value)
        return self.settings()

    def reset_settings(self) -> dict[str, str]:
        """Return every Studio setting to its documented default."""
        for key, _label, _hint in SETTING_LABELS:
            self.storage.set_setting(key, _default_setting(key))
        return self.settings()

    def setting_choices(self, key: str) -> tuple[tuple[str, str], ...]:
        """The real choices for a dropdown setting, ``(value, label)`` pairs."""
        if key == SETTING_DEFAULT_MODEL:
            return tuple((choice.key, choice.label()) for choice in self.model_choices())
        if key == SETTING_DEFAULT_MODE:
            return tuple((k, title) for k, title, _s in self.modes())
        if key == SETTING_DEFAULT_CONTENT_LEVEL:
            from core.romance.roleplay import CONTENT_LEVELS

            return tuple((level, level) for level in CONTENT_LEVELS)
        if key == SETTING_VOICE_ENGINE:
            return tuple((engine, engine) for engine in self.voice_engines())
        if key == SETTING_INTERFACE_LANGUAGE:
            return (("en", "English"), ("ar", "العربية"))
        return ()

    # -- personality, profile items, boundaries ---------------------------

    def personality(self, character_id: str) -> dict[str, Any]:
        """A character's editable personality as a plain dict of dial values."""
        from core.romance.personality import DEFAULT_DIAL, DIALS

        bundle = self.characters.get(character_id)
        row = bundle.personality
        values: dict[str, Any] = dict.fromkeys(DIALS, DEFAULT_DIAL)
        if row is not None:
            for name in DIALS:
                value = getattr(row, name, None)
                if value is not None:
                    values[name] = float(value)
            values["response_length"] = getattr(row, "response_length", "balanced")
            values["dialect"] = getattr(row, "dialect", "")
            values["instructions"] = getattr(row, "instructions", "")
        else:
            values["response_length"] = "balanced"
            values["dialect"] = ""
            values["instructions"] = ""
        return values

    def set_personality(self, character_id: str, values: dict[str, Any]) -> Any:
        return self.characters.set_personality(character_id, values)

    def add_interest(self, character_id: str, label: str, *, kind: str = "interest") -> Any:
        return self.characters.add_profile_item(character_id, kind=kind, label=label)

    def remove_interest(self, character_id: str, item_id: str) -> Any:
        return self.characters.remove_profile_item(character_id, item_id)

    def add_boundary(self, character_id: str, value: str, *, kind: str = "topic") -> Any:
        return self.safety.add_boundary(character_id, value, kind=kind)

    def remove_boundary(self, boundary_id: str) -> Any:
        return self.safety.remove_boundary(boundary_id)

    def boundaries(self, character_id: str) -> tuple[Any, ...]:
        """A character's explicit conversation limits (the safety floor's list)."""
        return self.safety.boundaries(character_id)

    # -- roleplay / scenarios --------------------------------------------

    def scenarios(self, *, category: str | None = None) -> tuple[Any, ...]:
        return self.roleplay.summaries(category=category)

    def scenario(self, scenario_id: str) -> Any:
        return self.roleplay.get_scenario(scenario_id)

    def create_scenario(self, title: str, **fields: Any) -> Any:
        return self.roleplay.create_scenario(title, **fields)

    def scenario_categories(self) -> tuple[tuple[str, str], ...]:
        from core.romance.roleplay import CATEGORIES, CATEGORY_LABELS

        return tuple((key, CATEGORY_LABELS.get(key, key)) for key in CATEGORIES)

    def content_levels(self) -> tuple[str, ...]:
        from core.romance.roleplay import CONTENT_LEVELS

        return tuple(CONTENT_LEVELS)

    def add_participant(self, scenario_id: str, character_id: str, *, role: str = "") -> Any:
        return self.roleplay.add_participant(scenario_id, character_id, role=role)

    def remove_participant(self, participant_id: str) -> Any:
        return self.roleplay.remove_participant(participant_id)

    def scenario_participants(self, scenario_id: str) -> tuple[Any, ...]:
        return self.roleplay.participants(scenario_id)

    def start_scenario(self, scenario_id: str, *, character_id: str = "") -> Any:
        return self.roleplay.start(scenario_id, character_id=character_id)

    def scenario_sessions(self, scenario_id: str) -> tuple[Any, ...]:
        return self.roleplay.sessions(scenario_id)

    def pause_scenario(self, session_id: str) -> Any:
        return self.roleplay.pause(session_id)

    def resume_scenario(self, session_id: str) -> Any:
        return self.roleplay.resume(session_id)

    def end_scenario(self, session_id: str) -> Any:
        return self.roleplay.end(session_id)

    def branch_scenario(self, session_id: str) -> Any:
        return self.roleplay.branch(session_id)

    def restart_scenario(self, session_id: str) -> Any:
        return self.roleplay.restart(session_id)

    # -- relationship editing --------------------------------------------

    def set_relationship(self, character_id: str, **dimensions: float) -> Any:
        """Set one or more relationship dimensions directly (an explicit edit)."""
        return self.relationships.edit(character_id, **dimensions)  # type: ignore[arg-type]

    def relationship_dimensions(self) -> tuple[str, ...]:
        from core.romance.relationship import DIMENSIONS

        return tuple(DIMENSIONS)

    # -- voices -----------------------------------------------------------

    def voices(self, *, engine: str | None = None, language: str | None = None) -> tuple[Any, ...]:
        return self.voice.available_voices(engine=engine, language=language)

    def set_voice(self, character_id: str, **settings: Any) -> Any:
        return self.voice.set_voice(character_id, **settings)

    def voice_profile(self, character_id: str) -> Any:
        return self.voice.view(character_id)

    def speak(self, character_id: str, text: str) -> Any:
        return self.voice.speak(character_id, text)

    # -- safety -----------------------------------------------------------

    def safety_status(self) -> dict[str, Any]:
        try:
            return {
                "adult_mode": self.safety.adult_mode_enabled(),
                "local_only": self.safety.local_only(),
                "disclosure": self.safety.privacy_disclosure(),
            }
        except Exception as error:
            return {"adult_mode": False, "local_only": True, "disclosure": str(error)}

    def set_adult_mode(self, enabled: bool, *, confirmed: bool = False) -> bool:
        return bool(self.safety.set_adult_mode(enabled, confirmed=confirmed))

    def confirm_eligibility(self, character_id: str) -> Any:
        """Record an explicit adult-eligibility confirmation for a character."""
        return self.safety.confirm_eligibility(character_id)

    def character_is_eligible(self, character_id: str) -> bool:
        return bool(self.safety.character_is_eligible(character_id))

    def character_is_adult(self, character_id: str) -> bool:
        return bool(self.safety.character_is_adult(character_id))


__all__ = ["CharacterRow", "ModelChoice", "RomanceStudioService", "StudioStatus"]
