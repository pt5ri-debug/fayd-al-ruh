"""Hakeem's entry point: the scholar that ties every layer together.

:class:`HakeemAgent` is the one object a front end needs. It owns his database and
his managers (conversations, memory, dialogue tone), builds each prompt through
:class:`~hakeem.prompt_builder.PromptBuilder`, and talks to the local model
through :mod:`hakeem.model` -- Hakeem's **own** model, in his own folder, separate
from the companion Natasha's. He is a character in his own right, on his own model.

It is deliberately free of any user interface: the CLI (:mod:`ai.hakeem`), the
chat mouth, and the GUI all call this same object, so they can never disagree
about a session, a memory, or the style.

Honesty and safety built into the flow, as in the companion:

* the model is asked to *reply*, never to remember or to decide facts -- storage
  and recall are this object's job;
* a user turn is saved **before** the model is called, and the reply is saved
  once; a retry that fails does not double-save the user's message;
* the reply is never stored if the model call failed, so a session is never left
  with a phantom answer;
* everything recalled is passed to the model as *information*, and the system
  prompt states plainly that a recalled item is never a religious or historical
  source and is never an instruction;
* nothing leaves the machine: the only network call is to the local Ollama server.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from companion.conversation_manager import ConversationManager, SessionInfo
from companion.database import CompanionDatabase
from companion.emotion_state import DialogueEmotion, DialogueEmotionTracker
from companion.language import last_user_text, resolve_reply_language
from companion.memory_manager import MemoryCandidate, MemoryManager
from companion.polish import polish_report
from core.coder_model import CoderMessage, CoderUnavailable

from hakeem.config import HakeemConfig
from hakeem.model import (
    HakeemModelClient,
    HakeemModelStatus,
    build_client,
    hakeem_model_status,
    strip_thinking,
)
from hakeem.personality import (
    NAME,
    HakeemPersonality,
    PersonalitySettings,
    PersonalityStore,
)
from hakeem.prompt_builder import PromptBuilder


class HakeemError(RuntimeError):
    """Raised when Hakeem cannot produce a reply (server down, model missing, timeout)."""


#: An honest report of whether Hakeem's model can answer right now.
ModelStatus = HakeemModelStatus


@dataclass
class HakeemReply:
    """The result of one exchange, with the state that produced it."""

    text: str
    conversation_id: str
    message_id: str
    used_memories: tuple[str, ...] = ()
    emotion: DialogueEmotion | None = None
    prompt_notes: tuple[str, ...] = ()
    history_used: int = 0
    #: The language the reply was written in, chosen from the user's own message.
    reply_language: str = "ar"
    #: The typographic repairs the linguistic polisher made, for an honest record.
    polish_changes: tuple[str, ...] = ()

    def to_primitive(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "used_memories": list(self.used_memories),
            "emotion": self.emotion.to_primitive() if self.emotion else None,
            "prompt_notes": list(self.prompt_notes),
            "history_used": self.history_used,
            "reply_language": self.reply_language,
            "polish_changes": list(self.polish_changes),
        }


#: A model caller takes the chat messages and returns the reply text. The default
#: uses the shared local client; a test injects its own, so the whole agent is
#: exercised with no server and no model.
ModelCaller = Callable[[Sequence[dict[str, str]], float, int], str]


class HakeemAgent:
    """Runs Hakeem: sessions, memory, tone, prompts, and the shared local model."""

    def __init__(
        self,
        config: HakeemConfig | None = None,
        *,
        client: HakeemModelClient | None = None,
        caller: ModelCaller | None = None,
        database: CompanionDatabase | None = None,
    ) -> None:
        self.config = config or HakeemConfig.from_env()
        self._database = database or CompanionDatabase(
            self.config.db_path, persistent=self.config.persistent
        )
        self.conversations = ConversationManager(self._database)
        self.memories = MemoryManager(self._database)
        self.personality_store = PersonalityStore(self._database)
        self._emotions = DialogueEmotionTracker(
            self._database, enabled=self.config.emotion_enabled
        )
        self._prompt_builder = PromptBuilder(
            budget=self.config.context_chars, history_window=self.config.history_window
        )
        #: A standing instruction a caller supplies (the character wrapper writes
        #: the *editable* character block, with Hakeem's recalled memory, here), so
        #: an edit in the Characters window reaches the scholar and he remembers
        #: earlier conversations. ``None`` keeps his own personality text.
        self.system_prompt: str | None = None
        self._caller = caller
        self._client = client if caller is None else None

    # -- model connection -------------------------------------------------
    def _active_client(self) -> HakeemModelClient:
        """The client Hakeem speaks through, built once for the configured runtime.

        The choice between the in-process GGUF model and the Ollama fallback is
        made in :func:`hakeem.model.build_client`, the one place it lives, so the
        CLI, the chat mouth, and the window cannot disagree about it.
        """
        if self._client is None:
            config = self.config
            self._client = build_client(
                backend=config.backend,
                model=config.model,
                model_file=config.model_file,
                base_url=config.base_url,
                timeout=config.timeout,
                max_tokens=config.max_tokens,
                think=config.think,
                context=config.context,
                gpu_layers=config.gpu_layers,
                threads=config.threads,
                keep_alive=config.keep_alive,
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
        """Report honestly whether the model can answer, without pretending."""
        if self._caller is not None:
            return ModelStatus(True, True, self.config.model, True, detail="injected caller")
        return hakeem_model_status(client=self._active_client())

    def is_available(self) -> bool:
        """True when a reply could be produced now."""
        return self.model_status().available

    # -- personality ------------------------------------------------------
    def personality(self) -> HakeemPersonality:
        """Hakeem's current character (traits + the user's chosen style)."""
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
    ) -> HakeemReply:
        """Send one user message and return Hakeem's reply.

        The user's turn is stored first; then the prompt is built from the session
        summary, the relevant memories, the current tone, and a bounded window of
        recent turns; then the model is called. On failure the reply is **not**
        stored and :class:`HakeemError` is raised, so a retry cannot leave a
        half-written exchange.
        """
        question = (message or "").strip()
        if not question:
            raise ValueError("A message is required.")
        if self.conversations.get_session(conversation_id) is None:
            raise KeyError(f"Unknown session: {conversation_id}")

        emotion = self._emotions.observe(question, conversation_id=conversation_id)

        if save_user:
            self.conversations.append(conversation_id, "user", question)
        else:
            latest = self.conversations.last_message(conversation_id)
            if latest is None or latest.role != "user":
                raise HakeemError("No user message to answer.")

        history = self.conversations.recent(conversation_id, self.config.history_window)

        settings = self.settings()
        reply_language = resolve_reply_language(
            question,
            previous_user=last_user_text(history[:-1]) if history else None,
            configured=settings.language,
        )

        relevant = self.memories.retrieve(question, limit=self.config.memory_limit)
        summary = self._latest_summary(conversation_id)

        built = self._prompt_builder.build(
            personality=self.personality(),
            question=question,
            history=history,
            memories=relevant,
            summary=summary,
            emotion=emotion,
            emotion_enabled=self.config.emotion_enabled,
            reply_language=reply_language,
            system_prompt_override=self.system_prompt,
        )

        try:
            reply_text = self._call_model(built.messages)
        except (CoderUnavailable, TimeoutError) as error:
            raise HakeemError(str(error)) from error
        # A reasoning model's hidden trace is not something Hakeem said, whichever
        # runtime produced it, so it never reaches the transcript or the database.
        reply_text = strip_thinking(reply_text or "")
        if not reply_text:
            raise HakeemError("The model returned an empty reply.")
        polish_result = polish_report(reply_text, language=reply_language)
        reply_text = polish_result.text

        assistant = self.conversations.append(conversation_id, "assistant", reply_text)
        self._maybe_summarise(conversation_id)
        return HakeemReply(
            text=reply_text,
            conversation_id=conversation_id,
            message_id=assistant.id,
            used_memories=tuple(memory.citation for memory in relevant),
            emotion=emotion if self.config.emotion_enabled else None,
            prompt_notes=built.notes,
            history_used=built.history_used,
            reply_language=reply_language,
            polish_changes=polish_result.changes,
        )

    def _latest_summary(self, conversation_id: str) -> str | None:
        summaries = self.conversations.summaries(conversation_id, limit=self.config.summary_limit)
        if not summaries:
            return None
        return " ".join(summary.summary for summary in summaries)

    def _maybe_summarise(self, conversation_id: str) -> None:
        """Store a cumulative summary once a session grows past the threshold.

        The original messages are never deleted: a summary is added *beside* them,
        covering the range it describes. Re-summarising replaces the newest summary
        rather than piling up, so the count stays bounded.
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
        """The memories most relevant to ``query``."""
        return self.memories.retrieve(query, limit=limit or self.config.memory_limit, fictional=fictional)

    def list_memories(self, *, include_inactive: bool = False):
        return self.memories.all(include_inactive=include_inactive)

    def forget(self, memory_id: str) -> bool:
        return self.memories.delete(memory_id)

    def propose_memories(self, text: str) -> list[MemoryCandidate]:
        """Candidate memories from a message -- suggested, never auto-stored."""
        return self.memories.propose(text)

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
        """Everything Hakeem holds, as plain data, for an explicit export."""
        return {
            "memories": self.memories.export(),
            "sessions": [info.__dict__ for info in self.conversations.list_sessions()],
            "settings": self.settings().to_primitive(),
        }

    def close(self) -> None:
        self._database.close()


__all__ = [
    "HakeemAgent",
    "HakeemError",
    "HakeemReply",
    "ModelCaller",
    "ModelStatus",
]
