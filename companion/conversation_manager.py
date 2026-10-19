"""Independent conversation sessions for Natasha, with ordered, persistent turns.

Each session is a separate conversation: its own messages, its own sequence, its
own summaries. Nothing leaks between sessions -- every read and write is scoped by
``conversation_id``. Messages keep their insertion order through an explicit,
monotonic ``seq`` (not just a timestamp), so a fast exchange still replays in the
order it happened.

Persistence is SQLite through :mod:`companion.database`, so a session survives the
program closing and reopening. The manager holds no state of its own beyond the
database, so recreating it (or a new process) sees exactly the same sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from companion.database import CompanionDatabase, _utc_now

#: The roles a stored turn can have. ``user`` and ``assistant`` are the exchange;
#: ``system`` is reserved for a stored note the companion itself wrote.
ROLES = ("user", "assistant", "system")


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:10]}"


@dataclass(frozen=True)
class Message:
    """One stored turn."""

    id: str
    conversation_id: str
    seq: int
    role: str
    content: str
    created_at: str

    def to_primitive(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "seq": self.seq,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ConversationSummary:
    """A cumulative summary of a span of a session's messages."""

    id: str
    conversation_id: str
    summary: str
    from_seq: int
    to_seq: int
    created_at: str


@dataclass(frozen=True)
class SessionInfo:
    """A session's header: id, title, and when it was created and last active."""

    id: str
    title: str
    created_at: str
    last_active: str
    message_count: int = 0


class ConversationManager:
    """Creates, reads, and writes Natasha's conversation sessions."""

    def __init__(self, database: CompanionDatabase) -> None:
        self._db = database

    # -- sessions ---------------------------------------------------------
    def create_session(self, title: str = "Conversation", *, session_id: str | None = None) -> SessionInfo:
        """Start a new, empty session and return its header."""
        identifier = session_id or _new_id("CONV")
        now = _utc_now()
        clean_title = (title or "Conversation").strip() or "Conversation"
        self._db.execute(
            "INSERT INTO conversations(id, title, created_at, last_active) VALUES(?, ?, ?, ?)",
            (identifier, clean_title[:120], now, now),
        )
        return SessionInfo(id=identifier, title=clean_title[:120], created_at=now, last_active=now)

    def get_session(self, conversation_id: str) -> SessionInfo | None:
        row = self._db.query_one(
            "SELECT id, title, created_at, last_active FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        if row is None:
            return None
        return SessionInfo(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            last_active=row["last_active"],
            message_count=self.message_count(conversation_id),
        )

    def list_sessions(self) -> list[SessionInfo]:
        """Every session, most recently active first."""
        rows = self._db.query(
            "SELECT id, title, created_at, last_active FROM conversations ORDER BY last_active DESC"
        )
        return [
            SessionInfo(
                id=row["id"],
                title=row["title"],
                created_at=row["created_at"],
                last_active=row["last_active"],
                message_count=self.message_count(row["id"]),
            )
            for row in rows
        ]

    def delete_session(self, conversation_id: str) -> bool:
        """Delete a session and everything scoped to it (messages, summaries)."""
        with self._db.transaction() as connection:
            cursor = connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            connection.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            connection.execute(
                "DELETE FROM conversation_summaries WHERE conversation_id = ?", (conversation_id,)
            )
            connection.execute("DELETE FROM emotion_states WHERE conversation_id = ?", (conversation_id,))
        return cursor.rowcount > 0

    def rename_session(self, conversation_id: str, title: str) -> bool:
        clean = (title or "").strip()
        if not clean:
            raise ValueError("A session title cannot be empty.")
        cursor = self._db.execute(
            "UPDATE conversations SET title = ? WHERE id = ?", (clean[:120], conversation_id)
        )
        return cursor.rowcount > 0

    # -- messages ---------------------------------------------------------
    def _next_seq(self, conversation_id: str) -> int:
        row = self._db.query_one(
            "SELECT COALESCE(MAX(seq), 0) AS last FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        return int(row["last"]) + 1 if row is not None else 1

    def append(self, conversation_id: str, role: str, content: str) -> Message:
        """Append one turn to a session, in order, and touch the session.

        The sequence number and the session's ``last_active`` are written in the
        same transaction as the message, so a crash cannot record a message
        without advancing the session, or vice versa.
        """
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}.")
        text = (content or "").strip()
        if not text:
            raise ValueError("Message content cannot be empty.")
        if self.get_session(conversation_id) is None:
            raise KeyError(f"Unknown conversation: {conversation_id}")
        now = _utc_now()
        message_id = _new_id("MSG")
        with self._db.transaction() as connection:
            seq = int(
                connection.execute(
                    "SELECT COALESCE(MAX(seq), 0) FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()[0]
            ) + 1
            connection.execute(
                "INSERT INTO messages(id, conversation_id, seq, role, content, created_at) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                (message_id, conversation_id, seq, role, text, now),
            )
            connection.execute(
                "UPDATE conversations SET last_active = ? WHERE id = ?", (now, conversation_id)
            )
        return Message(
            id=message_id,
            conversation_id=conversation_id,
            seq=seq,
            role=role,
            content=text,
            created_at=now,
        )

    def messages(self, conversation_id: str, *, limit: int | None = None) -> list[Message]:
        """A session's turns in order; ``limit`` keeps only the newest N (still ordered)."""
        if limit is None:
            rows = self._db.query(
                "SELECT id, conversation_id, seq, role, content, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY seq ASC",
                (conversation_id,),
            )
        else:
            rows = self._db.query(
                "SELECT id, conversation_id, seq, role, content, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY seq DESC LIMIT ?",
                (conversation_id, max(0, limit)),
            )
            rows = list(reversed(rows))
        return [
            Message(
                id=row["id"],
                conversation_id=row["conversation_id"],
                seq=row["seq"],
                role=row["role"],
                content=row["content"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def recent(self, conversation_id: str, window: int) -> list[Message]:
        """The newest ``window`` turns, in order -- the direct context for a prompt."""
        return self.messages(conversation_id, limit=window)

    def message_count(self, conversation_id: str) -> int:
        row = self._db.query_one(
            "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?", (conversation_id,)
        )
        return int(row["n"]) if row is not None else 0

    def last_message(self, conversation_id: str) -> Message | None:
        """The most recent turn, used to avoid saving a reply twice on a retry."""
        rows = self.messages(conversation_id, limit=1)
        return rows[-1] if rows else None

    # -- summaries --------------------------------------------------------
    def add_summary(self, conversation_id: str, summary: str, from_seq: int, to_seq: int) -> ConversationSummary:
        """Store a cumulative summary covering the message range ``[from_seq, to_seq]``."""
        text = (summary or "").strip()
        if not text:
            raise ValueError("A summary cannot be empty.")
        if self.get_session(conversation_id) is None:
            raise KeyError(f"Unknown conversation: {conversation_id}")
        identifier = _new_id("SUM")
        now = _utc_now()
        self._db.execute(
            "INSERT INTO conversation_summaries(id, conversation_id, summary, from_seq, to_seq, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (identifier, conversation_id, text, int(from_seq), int(to_seq), now),
        )
        return ConversationSummary(
            id=identifier,
            conversation_id=conversation_id,
            summary=text,
            from_seq=int(from_seq),
            to_seq=int(to_seq),
            created_at=now,
        )

    def summaries(self, conversation_id: str, *, limit: int | None = None) -> list[ConversationSummary]:
        """A session's summaries, oldest first (optionally only the newest ``limit``)."""
        rows = self._db.query(
            "SELECT id, conversation_id, summary, from_seq, to_seq, created_at FROM conversation_summaries "
            "WHERE conversation_id = ? ORDER BY to_seq ASC",
            (conversation_id,),
        )
        items = [
            ConversationSummary(
                id=row["id"],
                conversation_id=row["conversation_id"],
                summary=row["summary"],
                from_seq=row["from_seq"],
                to_seq=row["to_seq"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
        return items[-limit:] if limit else items

    def delete_summary(self, summary_id: str) -> bool:
        cursor = self._db.execute("DELETE FROM conversation_summaries WHERE id = ?", (summary_id,))
        return cursor.rowcount > 0


__all__ = [
    "ConversationManager",
    "ConversationSummary",
    "Message",
    "ROLES",
    "SessionInfo",
]
