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

    # -- roleplay / scenarios --------------------------------------------

    def scenarios(self, *, category: str | None = None) -> tuple[Any, ...]:
        return self.roleplay.summaries(category=category)

    def create_scenario(self, title: str, **fields: Any) -> Any:
        return self.roleplay.create_scenario(title, **fields)

    def scenario_categories(self) -> tuple[tuple[str, str], ...]:
        from core.romance.roleplay import CATEGORIES, CATEGORY_LABELS

        return tuple((key, CATEGORY_LABELS.get(key, key)) for key in CATEGORIES)

    def start_scenario(self, scenario_id: str, *, character_id: str = "") -> Any:
        return self.roleplay.start(scenario_id, character_id=character_id)

    # -- voice ------------------------------------------------------------

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

    def add_boundary(self, character_id: str, value: str, *, kind: str = "topic") -> Any:
        return self.safety.add_boundary(character_id, value, kind=kind)

    def boundaries(self, character_id: str) -> tuple[Any, ...]:
        return self.safety.boundaries(character_id)


__all__ = ["CharacterRow", "RomanceStudioService", "StudioStatus"]
