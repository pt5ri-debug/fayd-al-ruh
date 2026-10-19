"""Natasha's entry point: the companion that ties every layer together.

:class:`CompanionAgent` is the one object a front end needs. It owns the database
and the managers (conversations, memory, relationship, emotion), builds each
prompt through :class:`~companion.prompt_builder.PromptBuilder`, and talks to the
local model through :mod:`companion.model` -- Natasha's own model layer, which
names her catalog, her client seam, and her honest status exactly as
:mod:`core.ultron`, :mod:`core.vision_model`, and :mod:`core.banner` do for the
other mouths. That layer speaks through the project's **existing** Ollama
transport (``core.coder_model.OllamaCoderClient``, standard library only, no new
dependency); no second Ollama client is created, the project's own is reused.

It is deliberately free of any user interface: the CLI (:mod:`ai.natasha`), the
chat mouth, and the GUI all call this same object, so they can never disagree
about a session, a memory, or the tone.

Honesty and safety built into the flow:

* the model is asked to *reply*, never to remember or to decide facts -- storage
  and recall are this object's job;
* a user turn is saved **before** the model is called, and the reply is saved
  once; a retry that fails does not double-save the user's message;
* the reply is never stored if the model call failed, so a session is never left
  with a phantom answer;
* everything recalled is passed to the model as *information*, and the system
  prompt states plainly that recalled text is never an instruction;
* nothing leaves the machine: the only network call is to the local Ollama server.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from companion.config import CompanionConfig
from companion.conversation_manager import ConversationManager, SessionInfo
from companion.database import CompanionDatabase
from companion.emotion_state import DialogueEmotion, DialogueEmotionTracker
from companion.memory_manager import MemoryManager, MemoryCandidate
from companion.model import (
    NatashaModelClient,
    NatashaModelStatus,
    OllamaNatashaClient,
    natasha_model_status,
)
from companion.personality import NAME, PersonalitySettings, PersonalityStore, NatashaPersonality
from companion.prompt_builder import PromptBuilder
from companion.relationship_manager import EventCandidate, RelationshipEvent, RelationshipManager
from core.coder_model import CoderMessage, CoderUnavailable


class CompanionError(RuntimeError):
    """Raised when Natasha cannot produce a reply (server down, model missing, timeout)."""


#: An honest report of whether Natasha's model can answer right now. It lives in
#: :mod:`companion.model` (the model layer, beside the client and the catalog, as
#: in every other mouth) and is re-exported here for the companion's public shape.
ModelStatus = NatashaModelStatus


@dataclass
class CompanionReply:
    """The result of one exchange, with the state that produced it."""

    text: str
    conversation_id: str
    message_id: str
    used_memories: tuple[str, ...] = ()
    emotion: DialogueEmotion | None = None
    prompt_notes: tuple[str, ...] = ()
    history_used: int = 0

    def to_primitive(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "used_memories": list(self.used_memories),
            "emotion": self.emotion.to_primitive() if self.emotion else None,
            "prompt_notes": list(self.prompt_notes),
            "history_used": self.history_used,
        }


#: A model caller takes the chat messages and returns the reply text. The default
#: uses the project's Ollama client; a test injects its own, so the whole agent is
#: exercised with no server and no model.
ModelCaller = Callable[[Sequence[dict[str, str]], float, int], str]


class CompanionAgent:
    """Runs Natasha: sessions, memory, tone, prompts, and the local model."""

    def __init__(
        self,
        config: CompanionConfig | None = None,
        *,
        client: NatashaModelClient | None = None,
        caller: ModelCaller | None = None,
        database: CompanionDatabase | None = None,
    ) -> None:
        self.config = config or CompanionConfig.from_env()
        self._database = database or CompanionDatabase(
            self.config.db_path, persistent=self.config.persistent
        )
        self.conversations = ConversationManager(self._database)
        self.memories = MemoryManager(self._database)
        self.relationships = RelationshipManager(self._database)
        self.personality_store = PersonalityStore(self._database)
        self._emotions = DialogueEmotionTracker(
            self._database, enabled=self.config.emotion_enabled
        )
        self._prompt_builder = PromptBuilder(
            budget=self.config.context_chars, history_window=self.config.history_window
        )
        self._caller = caller
        self._client = client if caller is None else None

    # -- model connection -------------------------------------------------
    def _active_client(self) -> NatashaModelClient:
        if self._client is None:
            self._client = OllamaNatashaClient(
                model=self.config.model,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_tokens=self.config.max_tokens,
                think=self.config.think,
            )
        client = self._client
        assert client is not None
        return client

    def _call_model(self, messages: Sequence[dict[str, str]]) -> str:
        if self._caller is not None:
            return self._caller(messages, self.config.temperature, self.config.max_tokens)
        client = self._active_client()
        coder_messages = [
            CoderMessage(role=m["role"], content=m["content"])
            for m in messages
            if m["role"] in {"system", "user", "assistant"}
        ]
        return client.chat(coder_messages, temperature=self.config.temperature)

    def model_status(self) -> ModelStatus:
        """Report honestly whether the model can answer, without pretending.

        The check lives in :func:`companion.model.natasha_model_status`, the same
        shape every other mouth uses; an injected ``caller`` (a test's) answers
        without touching a server.
        """
        if self._caller is not None:
            return ModelStatus(True, True, self.config.model, True, detail="injected caller")
        return natasha_model_status(client=self._active_client())

    def is_available(self) -> bool:
        """True when a reply could be produced now."""
        return self.model_status().available

    # -- personality ------------------------------------------------------
    def personality(self) -> NatashaPersonality:
        """Natasha's current character (traits + the user's chosen style)."""
        return self.personality_store.personality()

    def settings(self) -> PersonalitySettings:
        return self.personality_store.load()

    def update_settings(self, **changes: Any) -> PersonalitySettings:
        """Change the adjustable style and persist it."""
        updated = self.settings().with_overrides(**changes)
        self.personality_store.save(updated)
        return updated

    # -- sessions ---------------------------------------------------------
    def new_session(self, title: str = "Conversation", *, session_id: str | None = None) -> SessionInfo:
        """Start a fresh session and make it the active one."""
        info = self.conversations.create_session(title, session_id=session_id)
        self._emotions.load(info.id)
        return info

    def resume_session(self, conversation_id: str) -> SessionInfo:
        """Reopen an existing session, restoring its tone."""
        info = self.conversations.get_session(conversation_id)
        if info is None:
            raise KeyError(f"Unknown session: {conversation_id}")
        self._emotions.load(conversation_id)
        return info

    def list_sessions(self) -> list[SessionInfo]:
        return self.conversations.list_sessions()

    def delete_session(self, conversation_id: str) -> bool:
        return self.conversations.delete_session(conversation_id)

    # -- the exchange -----------------------------------------------------
    def ask(
        self,
        conversation_id: str,
        message: str,
        *,
        save_user: bool = True,
    ) -> CompanionReply:
        """Send one user message and return Natasha's reply.

        The user's turn is stored first; then the prompt is built from the session
        summary, the relevant memories (factual and story, kept separate), the
        current tone, and a bounded window of recent turns; then the model is
        called. On failure the reply is **not** stored and
        :class:`CompanionError` is raised, so a retry cannot leave a half-written
        exchange.
        """
        question = (message or "").strip()
        if not question:
            raise ValueError("A message is required.")
        if self.conversations.get_session(conversation_id) is None:
            raise KeyError(f"Unknown session: {conversation_id}")

        # Move the simulated tone in response to the user's words (gradual, bounded).
        emotion = self._emotions.observe(question, conversation_id=conversation_id)

        if save_user:
            self.conversations.append(conversation_id, "user", question)
        else:
            latest = self.conversations.last_message(conversation_id)
            if latest is None or latest.role != "user":
                raise CompanionError("No user message to answer.")

        history = self.conversations.recent(conversation_id, self.config.history_window)
        relevant = self.memories.retrieve(
            question, limit=self.config.memory_limit, fictional=False
        )
        story = self.memories.retrieve(
            question, limit=self.config.memory_limit, fictional=True
        )
        summary = self._latest_summary(conversation_id)

        built = self._prompt_builder.build(
            personality=self.personality(),
            question=question,
            history=history,
            memories=relevant,
            fictional_memories=story,
            summary=summary,
            emotion=emotion,
            emotion_enabled=self.config.emotion_enabled,
        )

        try:
            reply_text = self._call_model(built.messages)
        except (CoderUnavailable, TimeoutError) as error:
            raise CompanionError(str(error)) from error
        reply_text = (reply_text or "").strip()
        if not reply_text:
            raise CompanionError("The model returned an empty reply.")

        assistant = self.conversations.append(conversation_id, "assistant", reply_text)
        self._maybe_summarise(conversation_id)
        return CompanionReply(
            text=reply_text,
            conversation_id=conversation_id,
            message_id=assistant.id,
            used_memories=tuple(memory.citation for memory in relevant),
            emotion=emotion if self.config.emotion_enabled else None,
            prompt_notes=built.notes,
            history_used=built.history_used,
        )

    def _latest_summary(self, conversation_id: str) -> str | None:
        summaries = self.conversations.summaries(conversation_id, limit=self.config.summary_limit)
        if not summaries:
            return None
        return " ".join(summary.summary for summary in summaries)

    def _maybe_summarise(self, conversation_id: str) -> None:
        """Store a cumulative summary once a session grows past the threshold.

        The original messages are never deleted: a summary is added *beside* them,
        covering the range it describes. Re-summarising replaces the newest
        summary rather than piling up, so the count stays bounded.
        """
        count = self.conversations.message_count(conversation_id)
        if count <= self.config.summary_threshold:
            return
        messages = self.conversations.messages(conversation_id)
        covered_upto = count - self.config.history_window
        if covered_upto <= 0:
            return
        existing = self.conversations.summaries(conversation_id)
        if existing and existing[-1].to_seq >= covered_upto:
            return
        lines: list[str] = []
        for msg in messages:
            if msg.seq > covered_upto:
                break
            who = "User" if msg.role == "user" else NAME if msg.role == "assistant" else "Note"
            lines.append(f"{who}: {msg.content}")
        summary_text = " ".join(lines)[:2000]
        if not summary_text.strip():
            return
        from_seq = messages[0].seq if messages else 1
        if existing:
            # Replace the newest summary so the range keeps growing without
            # unbounded duplication; the messages themselves stay.
            self.conversations.delete_summary(existing[-1].id)
        self.conversations.add_summary(conversation_id, summary_text, from_seq, covered_upto)

    # -- memory -----------------------------------------------------------
    def remember(
        self,
        content: str,
        *,
        kind: str = "fact",
        source: str = "user",
        fictional: bool = False,
        allow_sensitive: bool = False,
    ):
        """Store a memory (the explicit path; never automatic)."""
        return self.memories.add(
            content, kind=kind, source=source, fictional=fictional, allow_sensitive=allow_sensitive
        )

    def recall(self, query: str, *, limit: int | None = None, fictional: bool = False):
        """The memories most relevant to ``query`` (factual by default)."""
        return self.memories.retrieve(query, limit=limit or self.config.memory_limit, fictional=fictional)

    def list_memories(self, *, include_inactive: bool = False):
        return self.memories.all(include_inactive=include_inactive)

    def forget(self, memory_id: str) -> bool:
        return self.memories.delete(memory_id)

    def propose_memories(self, text: str) -> list[MemoryCandidate]:
        """Candidate memories from a message -- suggested, never auto-stored."""
        return self.memories.propose(text)

    # -- relationship -----------------------------------------------------
    def relationship_summary(self):
        return self.relationships.summary()

    def relationship_events(self, *, include_fiction: bool = True) -> list[RelationshipEvent]:
        return self.relationships.events(include_fiction=include_fiction)

    def add_relationship_event(self, kind: str, title: str, detail: str = "", *, approved: bool):
        return self.relationships.add_event(kind, title, detail, approved=approved)

    def propose_relationship_events(self, text: str) -> list[EventCandidate]:
        return self.relationships.propose_event(text)

    # -- emotion ----------------------------------------------------------
    @property
    def emotion(self) -> DialogueEmotion:
        return self._emotions.state

    def reset_emotion(self, conversation_id: str | None = None) -> DialogueEmotion:
        return self._emotions.reset(conversation_id)

    # -- maintenance ------------------------------------------------------
    def backup(self, keep: int | None = None) -> Path | None:
        """Write a bounded, timestamped backup of the database."""
        return self._database.backup(self.config.backup_keep if keep is None else keep)

    def export_data(self) -> dict[str, Any]:
        """Everything Natasha holds, as plain data, for an explicit export."""
        return {
            "memories": self.memories.export(),
            "relationship_events": self.relationships.export(),
            "sessions": [info.__dict__ for info in self.conversations.list_sessions()],
            "settings": self.settings().to_primitive(),
        }

    def close(self) -> None:
        self._database.close()


__all__ = [
    "CompanionAgent",
    "CompanionError",
    "CompanionReply",
    "ModelCaller",
    "ModelStatus",
]
