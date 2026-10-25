"""Answer from **Natasha**, the project's companion, inside the chat.

The chat already offers several mouths (the grounded sources, the trained model,
Ultron, Vision, Banner, and Cloud). This adds the *companion* one: Natasha, who
keeps a stable personality, a durable memory, independent persistent sessions, a
simulated dialogue tone, and an evolving relationship story (see
:mod:`companion.companion_agent`).

It exists so Natasha is a *mouth inside the project's own chat* -- the same
session, the same transcript -- rather than a separate window. The chat's own
:class:`~core.conversation.Conversation` carries the visible transcript; Natasha's
own database carries her memory and tone, so a companion answer can be warmer and
more personal than a cited extract while the honesty rules still hold.

Two things stay honest:

* a Natasha answer is the local model's own prose, shaped by her personality, so
  it is labelled with :data:`companion.personality.NAME` and carries no citations;
* she runs entirely on this machine -- the model loads in-process from a GGUF file
  by default, or through the project's existing Ollama transport as a fallback;
  when the model cannot answer,
  :class:`~companion.companion_agent.CompanionError` carries the real next step
  instead of silently answering from a different mouth.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from companion.companion_agent import CompanionAgent, CompanionError
from companion.personality import NAME as NATASHA_NAME
from core.assistant import AssistantAnswer
from core.conversation import Conversation

#: The honesty label a companion answer carries. Dropped from a previous turn
#: before it is fed back, so the model is not prompted with a label it did not
#: produce.
HEADER = f"— {NATASHA_NAME} —"

#: Shown in place of a body when the model produced no text at all.
NATASHA_EMPTY = f"{NATASHA_NAME} produced no answer for this message."

#: One line explaining the companion switch, appended by ``chat`` to the help text.
NATASHA_HELP = (
    "Natasha mode: type 'from natasha' to talk with the companion "
    "(a stable personality, a lasting memory, and a changing tone), 'from the "
    "sources' to answer from retrieved passages with citations again."
)

#: The controls ``chat`` recognises to switch to the Natasha mouth. Kept here,
#: beside the mouth they select, and never added to the shared Jarvis controls.
SWITCH_TO_NATASHA = (
    "from natasha", "use natasha", "answer from natasha", "natasha on",
    "من ناتاشا", "ناتاشا", "استخدم ناتاشا",
)

_LABEL_LINES = frozenset({HEADER})


def _strip_label(text: str) -> str:
    """Drop a leading honesty label from a stored assistant turn."""
    lines = text.splitlines()
    while lines and lines[0].strip() in _LABEL_LINES:
        lines.pop(0)
    return "\n".join(lines).strip()


class NatashaAssistant:
    """Answers from Natasha, the companion, without retrieval or citations.

    Duck-typed to the grounded assistant's ``ask`` signature, so the Jarvis loop
    and the chat's switcher call it unchanged. It holds one
    :class:`~companion.companion_agent.CompanionAgent` and one companion session
    for the life of the chat, so follow-up questions keep both the visible context
    and the companion's own memory.
    """

    def __init__(self, agent: CompanionAgent, *, session_id: str | None = None) -> None:
        self.agent = agent
        # Kept for the grounded assistant's shape; the companion retrieves from
        # her own memory inside the agent, not from a Document store here.
        self.memory = None
        if session_id is None:
            existing = agent.list_sessions()
            session_id = existing[0].id if existing else agent.new_session("Natasha chat").id
        else:
            try:
                agent.resume_session(session_id)
            except KeyError:
                agent.new_session("Natasha chat", session_id=session_id)
        self.session_id = session_id

    def is_available(self) -> bool:
        """True when a reply could be produced now."""
        return self.agent.is_available()

    def ensure_available(self) -> None:
        """Raise :class:`CompanionError` with the real next step, if needed."""
        status = self.agent.model_status()
        if not status.available:
            raise CompanionError(status.summary())

    def describe(self) -> str:
        """A one-line, honest description of the companion and her model."""
        return f"{NATASHA_NAME} ({self.agent.config.model}) — {self.agent.settings().describe()}"

    def answer(self, question: str, *, history: Sequence[tuple[str, str]] = ()) -> str:
        """Natasha's reply to ``question``.

        The companion keeps her own conversation history, so ``history`` (the
        chat's own transcript) is not replayed into her prompt; it is accepted so
        the signature matches the other mouths.
        """
        if not question.strip():
            raise ValueError("Message cannot be empty.")
        self.ensure_available()
        reply = self.agent.ask(self.session_id, question)
        return reply.text.strip()

    def ask(
        self,
        question: str,
        *,
        conversation: Conversation | None = None,
        limit: int = 4,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> AssistantAnswer:
        """Match ``GroundedAssistant.ask``: record the turn and answer as Natasha.

        The chat's visible conversation keeps the exchange (so the transcript and
        the window show it), while the companion's own database keeps her memory
        and tone. The answer carries no citations on purpose: it is the local
        model's own words, shaped by her personality.
        """
        if not question.strip():
            raise ValueError("Message cannot be empty.")
        if conversation is None:
            conversation = Conversation(title=question.strip()[:40])
        conversation.add_user(question)
        try:
            body = self.answer(question) or NATASHA_EMPTY
        except CompanionError:
            # Record nothing but the user's turn on failure, so a retry cannot
            # leave a phantom answer; re-raise so the caller reports the cause.
            raise
        text = f"{HEADER}\n\n{body}"
        conversation.add_assistant(text, ())
        return AssistantAnswer(
            text=text,
            citations=(),
            passages=(),
            confidence=0.0,
            used_model=True,
            conversation_id=conversation.id,
        )


def load_natasha_assistant(
    *,
    session_id: str | None = None,
    model: str | None = None,
    model_file: str | Path | None = None,
    base_url: str | None = None,
    agent: CompanionAgent | None = None,
) -> NatashaAssistant:
    """Build the Natasha mouth and check she can answer.

    Raises :class:`CompanionError` when the local model cannot answer, so the
    caller reports the real next step *before* switching the chat to a mouth that
    could not answer. ``agent`` is the seam a test injects, so the whole mouth is
    exercised with no server and no model. ``model_file`` names an explicit GGUF
    file to load in-process, so a caller that let the user choose one of several
    downloaded models gets *that* file rather than the folder's largest.
    """
    from companion.config import CompanionConfig

    if agent is None:
        overrides: dict[str, object] = {}
        if model:
            overrides["model"] = model
        if model_file:
            overrides["model_file"] = Path(model_file)
        if base_url:
            overrides["base_url"] = base_url
        agent = CompanionAgent(CompanionConfig.from_env(**overrides))
    assistant = NatashaAssistant(agent, session_id=session_id)
    assistant.ensure_available()
    return assistant


__all__ = [
    "HEADER",
    "NATASHA_EMPTY",
    "NATASHA_HELP",
    "SWITCH_TO_NATASHA",
    "NatashaAssistant",
    "load_natasha_assistant",
]
