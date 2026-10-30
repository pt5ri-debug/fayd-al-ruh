"""Answer from **Hakeem** -- the scholar of religion and history -- inside the chat.

The chat already offers several mouths (the grounded sources, the trained model,
Ultron, Vision, Banner, Cloud, and the companion Natasha). This adds the *scholar*
one: Hakeem, who discusses the Abrahamic faiths -- Islam, Christianity, and
Judaism -- compares them fairly, and is at home in history and the ancient
civilisations (see :mod:`hakeem.hakeem_agent`). He has his **own** local model, in
his own folder; only his character differs.

It exists so Hakeem is a *mouth inside the project's own chat* -- the same session,
the same transcript -- rather than a separate window. The chat's own
:class:`~core.conversation.Conversation` carries the visible transcript; Hakeem's
own database carries his memory and tone, so a scholar's answer can be fuller and
more contextual while the honesty rules still hold.

Two things stay honest:

* a Hakeem answer is the local model's own prose, shaped by his character, so it
  is labelled with :data:`hakeem.personality.NAME` and carries no citations;
* he runs entirely on this machine -- the model loads in-process from a GGUF file
  by default, or through the project's existing Ollama transport as a fallback;
  when the model cannot answer, :class:`~hakeem.hakeem_agent.HakeemError` carries
  the real next step instead of silently answering from a different mouth.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from core.assistant import AssistantAnswer
from core.conversation import Conversation
from hakeem.hakeem_agent import HakeemAgent, HakeemError
from hakeem.personality import ARABIC_NAME
from hakeem.personality import NAME as HAKEEM_NAME

#: The honesty label a Hakeem answer carries. Dropped from a previous turn before
#: it is fed back, so the model is not prompted with a label it did not produce.
HEADER = f"— {HAKEEM_NAME} ({ARABIC_NAME}) —"

#: Shown in place of a body when the model produced no text at all.
HAKEEM_EMPTY = f"{HAKEEM_NAME} produced no answer for this message."

#: One line explaining the scholar switch, appended by ``chat`` to the help text.
HAKEEM_HELP = (
    "Hakeem mode: type 'from hakeem' to talk with the scholar of religion and "
    "history (compares the Abrahamic faiths, discusses the ancient civilisations), "
    "'from the sources' to answer from retrieved passages with citations again."
)

#: The controls ``chat`` recognises to switch to the Hakeem mouth. Kept here,
#: beside the mouth they select, and never added to the shared Jarvis controls.
SWITCH_TO_HAKEEM = (
    "from hakeem", "use hakeem", "answer from hakeem", "hakeem on",
    "من حكيم", "حكيم", "استخدم حكيم",
)

_LABEL_LINES = frozenset({HEADER})


def _strip_label(text: str) -> str:
    """Drop a leading honesty label from a stored assistant turn."""
    lines = text.splitlines()
    while lines and lines[0].strip() in _LABEL_LINES:
        lines.pop(0)
    return "\n".join(lines).strip()


class HakeemAssistant:
    """Answers from Hakeem, the scholar, without retrieval or citations.

    Duck-typed to the grounded assistant's ``ask`` signature, so the Jarvis loop
    and the chat's switcher call it unchanged. It holds one
    :class:`~hakeem.hakeem_agent.HakeemAgent` and one session for the life of the
    chat, so follow-up questions keep both the visible context and his own memory.
    """

    def __init__(self, agent: HakeemAgent, *, session_id: str | None = None) -> None:
        self.agent = agent
        # Kept for the grounded assistant's shape; Hakeem retrieves from his own
        # memory inside the agent, not from a Document store here.
        self.memory = None
        if session_id is None:
            existing = agent.list_sessions()
            session_id = existing[0].id if existing else agent.new_session("Hakeem chat").id
        else:
            try:
                agent.resume_session(session_id)
            except KeyError:
                agent.new_session("Hakeem chat", session_id=session_id)
        self.session_id = session_id

    def is_available(self) -> bool:
        """True when a reply could be produced now."""
        return self.agent.is_available()

    def ensure_available(self) -> None:
        """Raise :class:`HakeemError` with the real next step, if needed."""
        status = self.agent.model_status()
        if not status.available:
            raise HakeemError(status.summary())

    def describe(self) -> str:
        """A one-line, honest description of the scholar and his model."""
        return f"{HAKEEM_NAME} ({self.agent.config.model}) — {self.agent.settings().describe()}"

    def answer(self, question: str, *, history: Sequence[tuple[str, str]] = ()) -> str:
        """Hakeem's reply to ``question``.

        He keeps his own conversation history, so ``history`` (the chat's own
        transcript) is not replayed into his prompt; it is accepted so the
        signature matches the other mouths.
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
        temperature: float = 0.6,
        seed: int | None = None,
    ) -> AssistantAnswer:
        """Match ``GroundedAssistant.ask``: record the turn and answer as Hakeem.

        The chat's visible conversation keeps the exchange (so the transcript and
        the window show it), while Hakeem's own database keeps his memory and tone.
        The answer carries no citations on purpose: it is the local model's own
        words, shaped by his character.
        """
        if not question.strip():
            raise ValueError("Message cannot be empty.")
        if conversation is None:
            conversation = Conversation(title=question.strip()[:40])
        conversation.add_user(question)
        try:
            body = self.answer(question) or HAKEEM_EMPTY
        except HakeemError:
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


def load_hakeem_assistant(
    *,
    session_id: str | None = None,
    model: str | None = None,
    model_file: str | Path | None = None,
    base_url: str | None = None,
    agent: HakeemAgent | None = None,
) -> HakeemAssistant:
    """Build the Hakeem mouth and check he can answer.

    Raises :class:`HakeemError` when the local model cannot answer, so the caller
    reports the real next step *before* switching the chat to a mouth that could
    not answer. ``agent`` is the seam a test injects, so the whole mouth is
    exercised with no server and no model. ``model_file`` names an explicit GGUF
    file to load in-process, so a caller that let the user choose one of several
    downloaded models gets *that* file rather than the folder's largest.
    """
    from hakeem.config import HakeemConfig

    if agent is None:
        overrides: dict[str, object] = {}
        if model:
            overrides["model"] = model
        if model_file:
            overrides["model_file"] = Path(model_file)
        if base_url:
            overrides["base_url"] = base_url
        agent = HakeemAgent(HakeemConfig.from_env(**overrides))
    assistant = HakeemAssistant(agent, session_id=session_id)
    assistant.ensure_available()
    return assistant


__all__ = [
    "HAKEEM_EMPTY",
    "HAKEEM_HELP",
    "HEADER",
    "SWITCH_TO_HAKEEM",
    "HakeemAssistant",
    "load_hakeem_assistant",
]
