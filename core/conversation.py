"""Multi-turn conversation memory: sessions, roles, and bounded history.

The cognitive architecture already stores observations, beliefs, and episodes,
but it has no notion of a *conversation*: an ordered exchange with roles and a
bounded context window. This module adds exactly that, and nothing more.

Design choices:

* A :class:`Conversation` is an ordered list of :class:`Message` turns. History
  is bounded by ``max_turns`` so the prompt sent to a model cannot grow without
  limit; the oldest turns are dropped, the newest kept.
* Messages carry the citations that grounded them, so a later turn can still say
  where an earlier answer came from.
* Persistence uses the project's :class:`storage.json_store.JsonStore`, so a
  session survives restarts without a database.

Nothing here calls a model or touches the knowledge graph; it is pure state, and
the assistant layer composes it with retrieval and generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable
from uuid import uuid4

from storage.json_store import JsonStore

_NAMESPACE = "conversations"


class Role(str, Enum):
    """Who produced a turn."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    """One turn in a conversation."""

    role: Role
    text: str
    citations: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: str = field(default_factory=lambda: f"MSG-{uuid4().hex[:8]}")

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Message text cannot be empty.")

    def to_primitive(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role.value,
            "text": self.text,
            "citations": list(self.citations),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_primitive(cls, data: dict[str, Any]) -> "Message":
        return cls(
            role=Role(data["role"]),
            text=str(data["text"]),
            citations=tuple(data.get("citations", ())),
            created_at=datetime.fromisoformat(data["created_at"]),
            id=str(data["id"]),
        )


@dataclass
class Conversation:
    """An ordered, bounded exchange with a stable id and title."""

    title: str = "Conversation"
    max_turns: int = 20
    id: str = field(default_factory=lambda: f"CNV-{uuid4().hex[:8]}")
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _messages: list[Message] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("Conversation title cannot be empty.")
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1.")

    def add(
        self,
        role: Role,
        text: str,
        citations: Iterable[str] = (),
    ) -> Message:
        """Append a turn, dropping the oldest once the window is full."""
        message = Message(role=role, text=text, citations=tuple(citations))
        self._messages.append(message)
        overflow = len(self._messages) - self.max_turns
        if overflow > 0:
            del self._messages[:overflow]
        return message

    def add_user(self, text: str) -> Message:
        return self.add(Role.USER, text)

    def add_assistant(self, text: str, citations: Iterable[str] = ()) -> Message:
        return self.add(Role.ASSISTANT, text, citations)

    def add_system(self, text: str) -> Message:
        return self.add(Role.SYSTEM, text)

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(self._messages)

    def last_user_message(self) -> Message | None:
        for message in reversed(self._messages):
            if message.role is Role.USER:
                return message
        return None

    def last_assistant_message(self) -> Message | None:
        for message in reversed(self._messages):
            if message.role is Role.ASSISTANT:
                return message
        return None

    def last_exchange(self) -> tuple[str, str] | None:
        """The last (question, answer) pair, when the transcript ends on a reply.

        This is the window an explicit "that's right" / "that's wrong" refers to:
        a user turn the assistant just answered, with no turn in between.
        """
        messages = self._messages
        if len(messages) >= 2 and messages[-1].role is Role.ASSISTANT and messages[-2].role is Role.USER:
            return messages[-2].text, messages[-1].text
        return None

    def render(self, *, include_system: bool = True) -> str:
        """Render the window as a plain transcript for a model prompt."""
        lines: list[str] = []
        for message in self._messages:
            if not include_system and message.role is Role.SYSTEM:
                continue
            prefix = {
                Role.SYSTEM: "Instructions",
                Role.USER: "User",
                Role.ASSISTANT: "Assistant",
            }[message.role]
            lines.append(f"{prefix}: {message.text}")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "max_turns": self.max_turns,
            "created_at": self.created_at.isoformat(),
            "messages": [message.to_primitive() for message in self._messages],
        }

    @classmethod
    def from_primitive(cls, data: dict[str, Any]) -> "Conversation":
        conversation = cls(
            title=str(data["title"]),
            max_turns=int(data.get("max_turns", 20)),
            id=str(data["id"]),
            created_at=datetime.fromisoformat(data["created_at"]),
        )
        conversation._messages = [Message.from_primitive(item) for item in data.get("messages", [])]
        return conversation


class ConversationMemory:
    """Creates, tracks, and persists conversations by id."""

    def __init__(self, store: JsonStore | None = None) -> None:
        self._conversations: dict[str, Conversation] = {}
        self._store = store

    def start(self, title: str = "Conversation", *, max_turns: int = 20) -> Conversation:
        conversation = Conversation(title=title, max_turns=max_turns)
        self._conversations[conversation.id] = conversation
        return conversation

    def get(self, conversation_id: str) -> Conversation | None:
        return self._conversations.get(conversation_id)

    def conversations(self) -> tuple[Conversation, ...]:
        return tuple(self._conversations.values())

    def save(self, conversation: Conversation) -> None:
        """Persist one conversation (no-op when no store is configured)."""
        if self._store is None:
            return
        self._conversations.setdefault(conversation.id, conversation)
        self._store.save(_NAMESPACE, conversation.id, conversation.to_primitive())

    def load(self, conversation_id: str) -> Conversation | None:
        if conversation_id in self._conversations:
            return self._conversations[conversation_id]
        if self._store is None:
            return None
        record = self._store.load(_NAMESPACE, conversation_id)
        if record is None:
            return None
        conversation = Conversation.from_primitive(record)
        self._conversations[conversation.id] = conversation
        return conversation

    def list_saved(self) -> list[str]:
        return self._store.list_keys(_NAMESPACE) if self._store is not None else []

    @property
    def persistent(self) -> bool:
        """True when a store is configured, so sessions survive a restart."""
        return self._store is not None


__all__ = [
    "Conversation",
    "ConversationMemory",
    "Message",
    "Role",
]
