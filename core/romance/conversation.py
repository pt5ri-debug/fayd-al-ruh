"""The Studio's conversation manager: conversations, messages, and mode changes.

A thin, honest layer over the Phase-1 storage (``companion_conversations`` /
``companion_messages``). It owns no model: it stores the conversation, appends
messages, records the **mode in force when each message was written**, and renders
the request a mode gives the model through :func:`core.romance.modes.mode_prompt`.

The rule this module exists to keep: **a mode change never rewrites history.** The
mode is a property of the conversation *and* of each message already written, so
changing the mode selects a different template for the **next** turn and leaves
every earlier row exactly as it was. :meth:`ConversationManager.set_mode` records
the change; :meth:`messages` reads back each message with its own mode intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core.romance.characters import CharacterManager
from core.romance.modes import (
    ModeContext,
    ModeSpec,
    find_mode,
    mode_prompt,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle avoided at runtime
    from core.romance.storage import CompanionStorageService

#: The roles a message may carry.
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_SYSTEM = "system"
ROLES: tuple[str, ...] = (ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM)

#: The most characters of recalled memory a prompt carries (a bound, not a target).
DEFAULT_MEMORY_CHARS = 4000


class ConversationError(RuntimeError):
    """Raised when a conversation operation cannot be carried out as asked."""


@dataclass(frozen=True)
class ConversationSummary:
    """A conversation as a listing shows it."""

    conversation_id: str
    title: str
    mode: str
    character_id: str
    last_active: str
    message_count: int = 0

    def describe(self) -> str:
        return (
            f"{self.title} [{find_mode(self.mode).title}] — "
            f"{self.message_count} message(s), last active {self.last_active}"
        )


@dataclass(frozen=True)
class Turn:
    """One stored message, read back with the mode it was written under."""

    record_id: str
    seq: int
    role: str
    speaker: str
    mode: str
    content: str
    created_at: str

    def as_pair(self) -> tuple[str, str]:
        """The ``(speaker, text)`` a prompt's history block uses."""
        return (self.speaker or self.role, self.content)


class ConversationManager:
    """Create conversations, append messages, change modes, render prompts."""

    def __init__(
        self,
        storage: CompanionStorageService,
        *,
        characters: CharacterManager | None = None,
        memory_reader: Any = None,
    ) -> None:
        self.storage = storage
        self.characters = characters if characters is not None else CharacterManager(storage)
        # A callable ``(character_id, conversation_id) -> str`` returning the memory
        # block; the memory manager supplies a real one in Phase 6. Default: none.
        self._memory_reader = memory_reader

    # -- conversations ----------------------------------------------------

    def start(
        self,
        character_id: str,
        *,
        title: str = "Conversation",
        mode: str = "general",
        scenario_id: str = "",
    ) -> Any:
        """Open a new conversation for a character."""
        if not character_id:
            raise ConversationError("a conversation needs a character")
        return self.storage.domain.create(
            "companion_conversations",
            character_id=character_id,
            title=title or "Conversation",
            mode=find_mode(mode).key,
            scenario_id=scenario_id,
        )

    def get(self, conversation_id: str) -> Any:
        record = self.storage.domain.get("companion_conversations", conversation_id)
        if record is None:
            raise ConversationError(f"no conversation with id {conversation_id!r}")
        return record

    def list_conversations(
        self, *, character_id: str = "", limit: int | None = None
    ) -> tuple[ConversationSummary, ...]:
        """The conversations, newest first, each with its real message count.

        The message counts are gathered in **one pass** over the messages table
        (never a query per conversation).
        """
        where = "character_id = ?" if character_id else ""
        params: tuple[Any, ...] = (character_id,) if character_id else ()
        conversations = self.storage.domain.list_records(
            "companion_conversations", where=where, params=params, order_by="last_active DESC, record_id"
        )
        counts = self._message_counts()
        summaries = [
            ConversationSummary(
                conversation_id=conversation.record_id,
                title=conversation.title,
                mode=conversation.mode,
                character_id=conversation.character_id,
                last_active=conversation.last_active,
                message_count=counts.get(conversation.record_id, 0),
            )
            for conversation in conversations
        ]
        return tuple(summaries[:limit] if limit else summaries)

    def _message_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for message in self.storage.domain.list_records("companion_messages"):
            counts[message.conversation_id] = counts.get(message.conversation_id, 0) + 1
        return counts

    def set_mode(self, conversation_id: str, mode: str) -> Any:
        """Change the conversation's mode for **future** turns.

        Earlier messages keep their own recorded mode; this only updates the
        conversation row, so history is never rewritten.
        """
        spec = find_mode(mode)
        return self.storage.domain.update(
            "companion_conversations", conversation_id, mode=spec.key
        )

    def mode_of(self, conversation_id: str) -> ModeSpec:
        return find_mode(self.get(conversation_id).mode)

    # -- messages ---------------------------------------------------------

    def append(
        self,
        conversation_id: str,
        content: str,
        *,
        role: str = ROLE_USER,
        speaker: str = "",
        mode: str = "",
    ) -> Any:
        """Append one message, recording the mode in force.

        The message is **append-only**: nothing here edits an earlier row. ``seq``
        is the next number for the conversation, so the order is explicit and
        stable.
        """
        if role not in ROLES:
            raise ConversationError(f"unknown role: {role!r}")
        conversation = self.get(conversation_id)
        effective_mode = find_mode(mode or conversation.mode).key
        seq = self._next_seq(conversation_id)
        message = self.storage.domain.create(
            "companion_messages",
            conversation_id=conversation_id,
            seq=seq,
            role=role,
            speaker=speaker,
            mode=effective_mode,
            content=content,
        )
        # Touch the conversation so a listing orders by real activity.
        self.storage.domain.update(
            "companion_conversations", conversation_id, last_active=message.created_at
        )
        return message

    def _next_seq(self, conversation_id: str) -> int:
        existing = self.storage.domain.list_records(
            "companion_messages",
            where="conversation_id = ?",
            params=(conversation_id,),
        )
        return (max((m.seq for m in existing), default=0)) + 1

    def messages(self, conversation_id: str, *, limit: int | None = None) -> tuple[Turn, ...]:
        """Every message in the conversation, oldest first, with its own mode."""
        rows = self.storage.domain.list_records(
            "companion_messages",
            where="conversation_id = ?",
            params=(conversation_id,),
            order_by="seq, record_id",
        )
        turns = tuple(
            Turn(
                record_id=row.record_id,
                seq=row.seq,
                role=row.role,
                speaker=row.speaker,
                mode=row.mode,
                content=row.content,
                created_at=row.created_at,
            )
            for row in rows
        )
        return turns[-limit:] if limit else turns

    def recent_history(
        self, conversation_id: str, *, window: int | None = None
    ) -> tuple[tuple[str, str], ...]:
        """The bounded recent turns a mode's prompt carries, oldest first."""
        spec = self.mode_of(conversation_id)
        bound = window if window is not None else spec.history_window
        turns = self.messages(conversation_id, limit=bound)
        return tuple(turn.as_pair() for turn in turns)

    # -- prompt rendering -------------------------------------------------

    def build_prompt(
        self,
        conversation_id: str,
        *,
        mode: str = "",
        memory: str = "",
        scenario: str = "",
        cast: tuple[str, ...] = (),
    ) -> str:
        """Render the request the conversation's mode gives the model.

        The memory block is fetched through the injected reader (Phase 6) unless
        one is supplied; the history is the bounded recent window, so a long
        conversation never pushes the prompt out of the window.
        """
        conversation = self.get(conversation_id)
        spec = find_mode(mode or conversation.mode)
        character = self._character_context(conversation.character_id)
        if not memory and self._memory_reader is not None:
            memory = self._read_memory(conversation.character_id, conversation_id)
        context = ModeContext(
            name=character.get("name", "your character"),
            identity=character.get("identity", ""),
            memory=_clip(memory, DEFAULT_MEMORY_CHARS),
            history=self.recent_history(conversation_id, window=spec.history_window),
            cast=cast,
            scenario=scenario,
        )
        return mode_prompt(spec, context)

    def _read_memory(self, character_id: str, conversation_id: str) -> str:
        try:
            return str(self._memory_reader(character_id, conversation_id) or "")
        except Exception:
            # A memory read must never break a turn; the prompt simply has no block.
            return ""

    def _character_context(self, character_id: str) -> dict[str, str]:
        if not character_id:
            return {}
        try:
            bundle = self.characters.get(character_id)
        except Exception:
            return {}
        name = getattr(bundle.character, "name", "") or "your character"
        identity = getattr(bundle.character, "biography", "") or ""
        return {"name": name, "identity": identity}

    # -- restore ----------------------------------------------------------

    def resume(self, conversation_id: str) -> tuple[Any, tuple[Turn, ...]]:
        """Reopen a conversation: the record and its full message history."""
        return self.get(conversation_id), self.messages(conversation_id)


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


__all__ = [
    "DEFAULT_MEMORY_CHARS",
    "ROLES",
    "ROLE_ASSISTANT",
    "ROLE_SYSTEM",
    "ROLE_USER",
    "ConversationError",
    "ConversationManager",
    "ConversationSummary",
    "Turn",
]
