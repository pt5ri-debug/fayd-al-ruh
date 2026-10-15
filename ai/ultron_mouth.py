"""Answer from Ultron -- the general, advanced local model -- inside the chat.

The chat has two mouths already: the grounded assistant answers from the
petroleum corpus with citations, and :mod:`ai.mind` speaks the small in-project
checkpoint. This module adds the third one the chat's model selector offers:
**Ultron**, the real, general model that runs locally through
``llama-cpp-python`` and talks about every field of life (see
:mod:`core.ultron`).

It exists so Ultron is a *mouth inside the project's own chat* -- the same
session, the same transcript, the same conversation controls -- rather than a
separate window or a separate loop. :class:`UltronAssistant` is duck-typed to
the grounded assistant's ``ask`` signature, so :class:`core.jarvis.JarvisMode`
and :class:`gui.service.ChatService` call it unchanged when the selector swaps
it in; :class:`ai.chat.ModelSwitcher` owns the swap, exactly as it does for the
trained-model mouth.

Two things stay honest, as everywhere else in the project:

* an Ultron answer is the model's own prose, not a cited extract, so it is
  labelled with :data:`core.ultron.HEADER` and carries no citations;
* when the runtime or the model file is missing, :class:`UltronUnavailable`
  carries an actionable message (install the extra, or download a model) instead
  of silently answering from a different mouth.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from core.assistant import AssistantAnswer
from core.conversation import Conversation, Role
from core.ultron import (
    HEADER as ULTRON_HEADER,
    NAME as ULTRON_NAME,
    UltronEngine,
    UltronUnavailable,
)

#: Shown in place of a body when Ultron produced no text at all, so the answer is
#: never a header with nothing under it (which the window would draw as an empty
#: bubble carrying only its Copy button).
ULTRON_EMPTY = f"{ULTRON_NAME} produced no answer for this question."

#: One line explaining the Ultron switch, appended by ``chat`` to the help text.
ULTRON_HELP = (
    "Ultron mode: type 'from ultron' to answer from the general local model "
    "(any subject, no citations), 'from the sources' to answer from retrieved "
    "passages with citations again."
)

#: The controls ``chat`` recognises to switch to the Ultron mouth. Kept here,
#: beside the mouth they select, and never added to the shared Jarvis controls.
SWITCH_TO_ULTRON = (
    "from ultron", "use ultron", "answer from ultron", "ultron on",
    "من ألترون", "ألترون", "استخدم ألترون",
)

#: The honesty header a mouth writes above its answer. Dropped from a previous
#: assistant turn before it is fed back to the model, so Ultron is not prompted
#: with a label it did not produce.
_LABEL_LINES = frozenset({ULTRON_HEADER})


def unavailable_message(engine: UltronEngine | None = None) -> str:
    """An honest, actionable line for why Ultron cannot answer right now.

    Reuses the CLI's own status wording (``ai.ultron.status_line``), imported
    lazily so this module never pulls the CLI in just to be imported.
    """
    from ai.ultron import status_line

    return status_line(engine)


def _strip_label(text: str) -> str:
    """Drop a leading honesty header from a stored assistant turn."""
    lines = text.splitlines()
    while lines and lines[0].strip() in _LABEL_LINES:
        lines.pop(0)
    return "\n".join(lines).strip()


def _history(conversation: Conversation) -> list[tuple[str, str]]:
    """The prior turns as ``(role, text)`` pairs, newest last, for the model.

    Only user and assistant turns are kept -- a system turn is not something the
    chat model produced -- and the honesty header a previous mouth wrote is
    dropped, so the model is not fed a label it did not produce. Every other word
    is preserved.
    """
    pairs: list[tuple[str, str]] = []
    for message in conversation.messages:
        text = _strip_label(message.text)
        if not text:
            continue
        if message.role is Role.USER:
            pairs.append(("user", text))
        elif message.role is Role.ASSISTANT:
            pairs.append(("assistant", text))
    return pairs


class UltronAssistant:
    """Answers from Ultron, the general local model, without retrieval or citations.

    Duck-typed to the grounded assistant's ``ask`` signature, so the Jarvis loop
    calls it unchanged when the chat's model selector selects it. It holds one
    :class:`~core.ultron.UltronEngine` for the life of the session, so the model
    is loaded once and follow-up questions keep their context.
    """

    def __init__(
        self,
        engine: UltronEngine,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self.engine = engine
        self.max_tokens = max_tokens
        self.temperature = temperature
        # Kept for the grounded assistant's shape; Ultron never retrieves, so
        # there is nothing to persist here.
        self.memory = None

    def is_available(self) -> bool:
        """True when a reply could be produced now (runtime and model present)."""
        return self.engine.is_available()

    def ensure_available(self) -> None:
        """Raise :class:`UltronUnavailable` with the real next step, if needed."""
        if not self.engine.is_available():
            raise UltronUnavailable(unavailable_message(self.engine))

    def describe(self) -> str:
        """A one-line, honest description of the model in use."""
        return self.engine.describe()

    def answer(self, question: str, *, history: Sequence[tuple[str, str]] = ()) -> str:
        """Ultron's own reply to ``question``, in the context of ``history``."""
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        self.ensure_available()
        return self.engine.reply(
            question,
            history=history,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        ).strip()

    def ask(
        self,
        question: str,
        *,
        conversation: Conversation | None = None,
        limit: int = 4,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> AssistantAnswer:
        """Match ``GroundedAssistant.ask``: record the turn and answer from Ultron.

        The history is taken *before* the question is appended, so the model is
        asked to continue the conversation rather than to complete its own prompt.
        The answer carries no citations on purpose -- it is the model's own words.
        """
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        if conversation is None:
            conversation = Conversation(title=question.strip()[:40])
        history = _history(conversation)
        conversation.add_user(question)
        body = self.answer(question, history=history) or ULTRON_EMPTY
        text = f"{ULTRON_HEADER}\n\n{body}"
        conversation.add_assistant(text, ())
        return AssistantAnswer(
            text=text,
            citations=(),
            passages=(),
            confidence=0.0,
            used_model=True,
            conversation_id=conversation.id,
        )


def load_ultron_assistant(
    model_path: str | Path | None = None,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> UltronAssistant:
    """Build the Ultron mouth and check it can answer.

    Raises :class:`UltronUnavailable` when the optional runtime or a model file
    is missing, so the caller can report the real next step *before* switching
    the chat to a mouth that could not answer. The heavy work is deferred: the
    engine loads the GGUF lazily on the first reply.
    """
    assistant = UltronAssistant(
        UltronEngine(model_path=model_path),
        max_tokens=max_tokens,
        temperature=temperature,
    )
    assistant.ensure_available()
    return assistant


__all__ = [
    "SWITCH_TO_ULTRON",
    "ULTRON_EMPTY",
    "ULTRON_HELP",
    "UltronAssistant",
    "load_ultron_assistant",
    "unavailable_message",
]
