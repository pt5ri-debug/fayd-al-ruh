"""The companion orchestrator: one turn, from a message to a reply.

This is the single entry point the window, the CLI, and the tests drive. It
composes the Studio's own layers -- characters (Phase 2), the model router
(Phase 3), conversation modes (Phase 4), the relationship engine (Phase 5), the
memory manager (Phase 6), the roleplay engine (Phase 7), voice (Phase 8), and the
safety floor (Phase 9) -- into the one thing a user actually does: *say something
and hear the character answer*.

Four rules it keeps:

* **One turn at a time.** A multi-character scene generates **sequentially**
  through the one local model, so a low-memory machine never runs two generations
  at once.
* **A character is a person, and never impersonates another.** Every reply is
  generated from *that* character's own prompt (its identity, its memory, its
  relationship), and its speaker is that character's own name. A multi-character
  reply never speaks for a character that was not asked.
* **The safety floor is checked first.** An adult-mode turn is refused by the
  Phase-9 manager before any generation happens; a refusal is a real answer, not a
  crash.
* **Nothing is invented.** When no model is installed the turn returns honestly
  (``ok=False`` with the router's own reason and next step); a failed generation
  is never written to the transcript as a reply.

It loads no model on construction and reaches no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.romance.characters import CharacterManager
from core.romance.conversation import ConversationManager
from core.romance.memory import CompanionMemoryManager
from core.romance.modes import MODE_ADULT, MODE_MULTI_CHARACTER_ROLEPLAY, find_mode
from core.romance.relationship import RelationshipEngine
from core.romance.roleplay import RoleplayEngine
from core.romance.routing import CompanionModelRouter, GenerationHandle
from core.romance.storage import CompanionStorageService


class OrchestratorError(RuntimeError):
    """Raised when a turn cannot be started as asked."""


@dataclass(frozen=True)
class TurnResult:
    """What one turn produced: the reply, the messages, and what moved."""

    ok: bool = True
    refused: bool = False
    error: str = ""
    reason: str = ""
    conversation_id: str = ""
    character_id: str = ""
    mode: str = ""
    user_message: Any = None
    replies: tuple[Any, ...] = ()
    speakers: tuple[str, ...] = ()
    relationship: Any = None
    model_key: str = ""

    @property
    def reply(self) -> str:
        """The first reply's text (a single-character turn's answer)."""
        return getattr(self.replies[0], "content", "") if self.replies else ""

    def describe(self) -> str:
        if self.refused:
            return f"refused: {self.reason}"
        if not self.ok:
            return f"no answer: {self.error}"
        return self.reply


class CompanionOrchestrator:
    """Compose the Studio's layers into a single conversation turn."""

    def __init__(
        self,
        storage: CompanionStorageService,
        *,
        characters: CharacterManager | None = None,
        conversations: ConversationManager | None = None,
        relationships: RelationshipEngine | None = None,
        memory: CompanionMemoryManager | None = None,
        router: CompanionModelRouter | None = None,
        safety: Any = None,
        roleplay: RoleplayEngine | None = None,
        generator: Any = None,
    ) -> None:
        self.storage = storage
        self.characters = characters if characters is not None else CharacterManager(storage)
        self.memory = memory if memory is not None else CompanionMemoryManager(storage)
        self.conversations = (
            conversations
            if conversations is not None
            else ConversationManager(
                storage, characters=self.characters, memory_reader=self._memory_block
            )
        )
        self.relationships = (
            relationships if relationships is not None else RelationshipEngine(storage)
        )
        self.router = router
        self.safety = safety
        self.roleplay = roleplay if roleplay is not None else RoleplayEngine(storage)
        # A generation seam: ``(prompt, key, handle) -> str``. Default: the router.
        self._generator = generator

    # -- injection seams --------------------------------------------------

    def _generate(self, prompt: str, key: str | None, handle: GenerationHandle | None) -> Any:
        """Run one generation, through the injected seam or the real router.

        Returns a ``(text, ok, error)`` triple. The default path is the router,
        which resolves the chosen model and runs the project's own runtime; a test
        injects its own seam, so the whole orchestrator is exercised with no model.
        """
        if self._generator is not None:
            return self._generator(prompt, key, handle)
        if self.router is None:
            return "", False, "no language model is wired into this session"
        result = self.router.generate(prompt, key=key, handle=handle)
        if not getattr(result, "ok", False):
            return "", False, str(getattr(result, "detail", "") or "generation failed")
        return str(getattr(result, "text", "") or ""), True, ""

    def _memory_block(self, character_id: str, conversation_id: str) -> str:
        """The memory the prompt may use: relevant recall for this conversation."""
        try:
            turns = self.conversations.messages(conversation_id, limit=3)
        except Exception:
            turns = ()
        query = " ".join(turn.content for turn in turns if getattr(turn, "role", "") == "user")
        return self.memory.reference_block(character_id, query)

    # -- one turn ---------------------------------------------------------

    def send(
        self,
        conversation_id: str,
        message: str,
        *,
        mode: str = "",
        handle: GenerationHandle | None = None,
    ) -> TurnResult:
        """Append the user's message, generate the character's reply, record both.

        The order is deliberate: the safety floor is checked **before** anything is
        written, the user's message is appended before the model runs (so a
        generation failure still keeps what the user said), and a reply is appended
        only when one was really produced.
        """
        conversation = self.conversations.get(conversation_id)
        character_id = str(conversation.character_id)
        text = (message or "").strip()
        if not text:
            raise OrchestratorError("a turn needs a message")

        effective_mode = find_mode(mode or conversation.mode).key
        refusal = self._safety_refusal(character_id, effective_mode, text)
        if refusal is not None:
            return TurnResult(
                ok=False,
                refused=True,
                reason=refusal,
                conversation_id=conversation_id,
                character_id=character_id,
                mode=effective_mode,
            )

        user_message = self.conversations.append(
            conversation_id, text, role="user", mode=effective_mode
        )
        self._observe_relationship(character_id, conversation_id, user_message)

        prompt = self.conversations.build_prompt(conversation_id, mode=effective_mode)
        answer, ok, error = self._generate(prompt, None, handle)
        if not ok:
            return TurnResult(
                ok=False,
                error=error,
                conversation_id=conversation_id,
                character_id=character_id,
                mode=effective_mode,
                user_message=user_message,
            )

        speaker = self._speaker_name(character_id)
        reply = self.conversations.append(
            conversation_id, answer, role="assistant", speaker=speaker, mode=effective_mode
        )
        return TurnResult(
            ok=True,
            conversation_id=conversation_id,
            character_id=character_id,
            mode=effective_mode,
            user_message=user_message,
            replies=(reply,),
            speakers=(speaker,),
            relationship=self.relationships.snapshot(character_id),
            model_key=self._model_key(),
        )

    def send_multi(
        self,
        conversation_id: str,
        message: str,
        *,
        cast: tuple[str, ...] = (),
        scenario: str = "",
        handle: GenerationHandle | None = None,
    ) -> TurnResult:
        """A multi-character turn: the user speaks once, each character answers.

        Generation is **sequential** -- one character, then the next -- so a
        low-memory machine never runs two models at once. Each reply is generated
        from its own character's prompt and carries that character's own name as
        its speaker, so a character never impersonates another.
        """
        conversation = self.conversations.get(conversation_id)
        text = (message or "").strip()
        if not text:
            raise OrchestratorError("a turn needs a message")
        members = cast or (str(conversation.character_id),)
        members = tuple(dict.fromkeys(member for member in members if member))

        effective_mode = MODE_MULTI_CHARACTER_ROLEPLAY
        for member in members:
            refusal = self._safety_refusal(member, effective_mode, text)
            if refusal is not None:
                return TurnResult(
                    ok=False,
                    refused=True,
                    reason=refusal,
                    conversation_id=conversation_id,
                    character_id=member,
                    mode=effective_mode,
                )

        user_message = self.conversations.append(
            conversation_id, text, role="user", mode=effective_mode
        )
        for member in members:
            self._observe_relationship(member, conversation_id, user_message)

        names = tuple(self._speaker_name(member) for member in members)
        replies: list[Any] = []
        speakers: list[str] = []
        for _member, name in zip(members, names, strict=True):
            prompt = self.conversations.build_prompt(
                conversation_id, mode=effective_mode, cast=names, scenario=scenario
            )
            answer, ok, error = self._generate(prompt, None, handle)
            if not ok:
                return TurnResult(
                    ok=False,
                    error=error,
                    conversation_id=conversation_id,
                    character_id=members[0],
                    mode=effective_mode,
                    user_message=user_message,
                    replies=tuple(replies),
                    speakers=tuple(speakers),
                )
            replies.append(
                self.conversations.append(
                    conversation_id,
                    answer,
                    role="assistant",
                    speaker=name,
                    mode=effective_mode,
                )
            )
            speakers.append(name)
        return TurnResult(
            ok=True,
            conversation_id=conversation_id,
            character_id=members[0],
            mode=effective_mode,
            user_message=user_message,
            replies=tuple(replies),
            speakers=tuple(speakers),
            model_key=self._model_key(),
        )

    # -- helpers ----------------------------------------------------------

    def _safety_refusal(self, character_id: str, mode: str, text: str) -> str | None:
        """The safety floor's verdict on a turn, or ``None`` when it may proceed."""
        if self.safety is None:
            return None
        try:
            if mode == MODE_ADULT:
                decision = self.safety.check_content(character_id, "adult", text=text)
                if not decision.allowed:
                    return decision.reason or "adult content is not permitted for this character"
                return None
            decision = self.safety.check_content(character_id, "general", text=text)
            return None if decision.allowed else (decision.reason or "refused by the safety floor")
        except Exception as error:  # a safety check must never crash a turn
            return f"the safety floor could not be evaluated: {error}"

    def _observe_relationship(self, character_id: str, conversation_id: str, message: Any) -> None:
        """Record the turn as a relationship event, with the message as evidence."""
        try:
            kind = "first_conversation" if int(getattr(message, "seq", 0) or 0) <= 1 else "meaningful_conversation"
            self.relationships.observe(
                character_id,
                kind,
                message_id=getattr(message, "record_id", ""),
                conversation_id=conversation_id,
            )
        except Exception:
            # A relationship note must never break a turn.
            return

    def _speaker_name(self, character_id: str) -> str:
        try:
            return self.characters.get(character_id).name or "Character"
        except Exception:
            return "Character"

    def _model_key(self) -> str:
        if self.router is None:
            return ""
        try:
            return self.router.resolve().key
        except Exception:
            return ""


__all__ = ["CompanionOrchestrator", "OrchestratorError", "TurnResult"]
