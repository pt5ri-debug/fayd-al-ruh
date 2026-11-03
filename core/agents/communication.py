"""The agent communication interface: structured messages between personalities.

Beyond a delegation (a request for *work*), personalities sometimes need to pass a
structured **message** -- a notification, a shared observation, a question. The
task lists an "Agent Communication Interface" as a component, and the collaboration
section requires "structured requests and responses", "shared task identifiers",
and "provenance tracking".

A :class:`AgentMessage` is that structure: a shared id, a sender, a recipient (or a
broadcast), a kind, a body, and the provenance of what it is about. A
:class:`MessageBus` delivers it, keeping a bounded, inspectable log. It is
deliberately **not** an event bus for the whole system (that is
:mod:`core.event_bus`): it is the *agent-to-agent* channel, and it carries only
what a caller puts in it -- never a private memory, which the memory access layer
would refuse.

Dependency-free and Tk-free: it holds no model and touches no network.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from core.learning_isolation import CHARACTER_OWNERS, MAIN_MODEL_OWNER, owner_for

#: How many messages the bus keeps. Bounded on purpose: the log is for inspection,
#: not an archive (the durable record is the learning system's).
DEFAULT_MESSAGE_LOG = 500


class MessageRefused(Exception):
    """Raised when a message crosses a boundary the policy does not permit."""


@dataclass(frozen=True)
class AgentMessage:
    """One structured message from one personality to another.

    ``recipient`` may be a specific owner or ``BROADCAST``. ``about`` names what
    the message concerns (a task id, a question, a topic) and ``shared_id`` groups
    a conversation of messages, so a reply can be matched to its request.
    """

    sender: str
    recipient: str
    kind: str
    body: str
    shared_id: str = ""
    about: str = ""
    message_id: str = field(default_factory=lambda: f"msg-{uuid.uuid4().hex[:12]}")
    at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        object.__setattr__(self, "sender", owner_for(self.sender))
        if self.recipient != BROADCAST:
            object.__setattr__(self, "recipient", owner_for(self.recipient))
        if not self.body.strip():
            raise ValueError("A message needs a body.")

    def describe(self) -> str:
        return f"[{self.kind}] {self.sender} -> {self.recipient}: {self.body[:120]}"

    def to_primitive(self) -> dict[str, object]:
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "recipient": self.recipient,
            "kind": self.kind,
            "body": self.body,
            "shared_id": self.shared_id,
            "about": self.about,
            "at": self.at,
        }


#: The recipient value meaning "every personality".
BROADCAST = "*"


@dataclass
class MessageBus:
    """The bounded, inspectable agent-to-agent channel.

    Delivery is a no-op beyond logging (there is no background loop by default), so
    the bus is deterministic and testable. A ``deliver`` callback, when supplied,
    observes each message -- the natural place to hand it to a live agent.
    """

    log_limit: int = DEFAULT_MESSAGE_LOG
    on_deliver: object | None = None
    messages: list[AgentMessage] = field(default_factory=list)

    def send(
        self,
        sender: str,
        recipient: str,
        kind: str,
        body: str,
        *,
        shared_id: str = "",
        about: str = "",
    ) -> AgentMessage:
        """Send one message, recording it (bounded) and observing it."""
        resolved_recipient = recipient
        if recipient != BROADCAST:
            resolved_recipient = owner_for(recipient)
            if resolved_recipient not in (*CHARACTER_OWNERS, MAIN_MODEL_OWNER):
                raise MessageRefused(f"{recipient!r} is not a known personality.")
        message = AgentMessage(
            sender=sender,
            recipient=resolved_recipient,
            kind=kind,
            body=body,
            shared_id=shared_id,
            about=about,
        )
        self.messages.append(message)
        if len(self.messages) > self.log_limit:
            del self.messages[: len(self.messages) - self.log_limit]
        if callable(self.on_deliver):
            from core.best_effort import best_effort

            with best_effort("deliver an agent message"):
                self.on_deliver(message)  # type: ignore[misc]
        return message

    def inbox(self, owner: str, *, limit: int = 20) -> tuple[AgentMessage, ...]:
        """The messages addressed to ``owner`` (or broadcast), newest last."""
        resolved = owner_for(owner)
        chosen = [
            message
            for message in self.messages
            if message.recipient in (resolved, BROADCAST)
        ]
        return tuple(chosen[-limit:])

    def thread(self, shared_id: str) -> tuple[AgentMessage, ...]:
        """Every message sharing ``shared_id``, in order."""
        return tuple(message for message in self.messages if message.shared_id == shared_id)

    def describe(self) -> str:
        return f"message bus: {len(self.messages)} message(s) retained"


__all__ = [
    "BROADCAST",
    "DEFAULT_MESSAGE_LOG",
    "AgentMessage",
    "MessageBus",
    "MessageRefused",
]
