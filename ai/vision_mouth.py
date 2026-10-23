"""Answer from Vision -- the petroleum-engineering local model -- inside the chat.

The chat has three mouths already: the grounded assistant answers from the
petroleum corpus with citations, :mod:`ai.mind` speaks the small in-project
checkpoint, and :mod:`ai.ultron_mouth` runs the *general* local model. This
module adds the fourth one the chat's model selector offers: **Vision**, the
petroleum-engineering specialist that runs locally through ``llama-cpp-python``
and is built on the OGAI Reasoner weights (see :mod:`core.vision_model`).

It exists so Vision is a *mouth inside the project's own chat* -- the same
session, the same transcript, the same conversation controls -- rather than a
separate window or a separate loop. :class:`VisionAssistant` is duck-typed to
the grounded assistant's ``ask`` signature, so :class:`core.jarvis.JarvisMode`
and :class:`gui.service.ChatService` call it unchanged when the selector swaps
it in; :class:`ai.chat.ModelSwitcher` owns the swap, exactly as it does for the
trained-model and Ultron mouths.

Two things stay honest, as everywhere else in the project:

* a Vision answer is the model's own prose, not a cited extract, so it is
  labelled with :data:`core.vision_model.HEADER` and carries no citations;
* when the runtime or the model file is missing, :class:`VisionUnavailable`
  carries an actionable message (install the extra, or download a model) instead
  of silently answering from a different mouth.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from core.assistant import AssistantAnswer
from core.conversation import Conversation, Role
from core.vision_model import (
    HEADER as VISION_HEADER,
)
from core.vision_model import (
    NAME as VISION_NAME,
)
from core.vision_model import (
    VisionEngine,
    VisionUnavailable,
)

#: Shown in place of a body when Vision produced no text at all, so the answer is
#: never a header with nothing under it (which the window would draw as an empty
#: bubble carrying only its Copy button).
VISION_EMPTY = f"{VISION_NAME} produced no answer for this question."

#: One line explaining the Vision switch, appended by ``chat`` to the help text.
VISION_HELP = (
    "Vision mode: type 'from vision' to answer from the petroleum local model "
    "(reservoir, drilling, production, PVT; no citations), 'from the sources' to "
    "answer from retrieved passages with citations again."
)

#: The controls ``chat`` recognises to switch to the Vision mouth. Kept here,
#: beside the mouth they select, and never added to the shared Jarvis controls.
SWITCH_TO_VISION = (
    "from vision", "use vision", "answer from vision", "vision on",
    "من فيجن", "فيجن", "استخدم فيجن",
)

#: Opens the reference block when the conversation carries an attached file, so
#: the model is told the text below is the user's own material rather than its
#: own earlier words.
REFERENCE_LABEL = "Reference material the user attached:"

#: How much of an attached file's text is handed to the model in one prompt. The
#: conversation keeps the whole file; the model's context window is finite
#: (``core.vision_model.DEFAULT_CONTEXT``), so a longer file is cut here with a
#: visible marker instead of overflowing the window and failing the reply.
DEFAULT_REFERENCE_CHARS = 8000

_REFERENCE_CUT = "\n… [the rest of the attached file is not shown here]"

#: The honesty header a mouth writes above its answer. Dropped from a previous
#: assistant turn before it is fed back to the model, so Vision is not prompted
#: with a label it did not produce.
_LABEL_LINES = frozenset({VISION_HEADER})


def unavailable_message(engine: VisionEngine | None = None) -> str:
    """An honest, actionable line for why Vision cannot answer right now.

    Reuses the CLI's own status wording (``ai.vision_model.status_line``), imported
    lazily so this module never pulls the CLI in just to be imported.
    """
    from ai.vision_model import status_line

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


def _context_prompt(conversation: Conversation, question: str, *, limit: int) -> str:
    """Attach the conversation's reference material to ``question``, when present.

    The project's conversation renders reference material in its own block, so it
    is passed the same way here: the model is told, in the prompt, that the text
    below is material the user attached. A conversation with no attachment returns
    ``question`` unchanged. ``limit`` bounds how much text is sent (``<= 0`` sends
    all of it); a cut is marked so the model does not mistake it for the file's end.
    """
    reference = conversation.render_context()
    if not reference:
        return question
    if 0 < limit < len(reference):
        reference = reference[:limit].rstrip() + _REFERENCE_CUT
    return f"{REFERENCE_LABEL}\n{reference}\n\nQuestion: {question}"


class VisionAssistant:
    """Answers from Vision, the petroleum local model, without retrieval or citations.

    Duck-typed to the grounded assistant's ``ask`` signature, so the Jarvis loop
    calls it unchanged when the chat's model selector selects it. It holds one
    :class:`~core.vision_model.VisionEngine` for the life of the session, so the
    model is loaded once and follow-up questions keep their context.
    """

    def __init__(
        self,
        engine: VisionEngine,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reference_chars: int = DEFAULT_REFERENCE_CHARS,
    ) -> None:
        self.engine = engine
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reference_chars = reference_chars
        # Kept for the grounded assistant's shape; Vision never retrieves, so
        # there is nothing to persist here.
        self.memory = None

    def is_available(self) -> bool:
        """True when a reply could be produced now (runtime and model present)."""
        return self.engine.is_available()

    def ensure_available(self) -> None:
        """Raise :class:`VisionUnavailable` with the real next step, if needed."""
        if not self.engine.is_available():
            raise VisionUnavailable(unavailable_message(self.engine))

    def describe(self) -> str:
        """A one-line, honest description of the model in use."""
        return self.engine.describe()

    def answer(self, question: str, *, history: Sequence[tuple[str, str]] = ()) -> str:
        """Vision's own reply to ``question``, in the context of ``history``."""
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
        temperature: float = 0.6,
        seed: int | None = None,
    ) -> AssistantAnswer:
        """Match ``GroundedAssistant.ask``: record the turn and answer from Vision.

        The history is taken *before* the question is appended, so the model is
        asked to continue the conversation rather than to complete its own prompt.
        Any reference material the conversation carries -- the text of a file the
        user attached -- is put in front of the question, so Vision can read it
        like the grounded mouth does. The answer carries no citations on purpose:
        it is the model's own words.
        """
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        if conversation is None:
            conversation = Conversation(title=question.strip()[:40])
        history = _history(conversation)
        prompt = _context_prompt(conversation, question, limit=self.reference_chars)
        conversation.add_user(question)
        body = self.answer(prompt, history=history) or VISION_EMPTY
        text = f"{VISION_HEADER}\n\n{body}"
        conversation.add_assistant(text, ())
        return AssistantAnswer(
            text=text,
            citations=(),
            passages=(),
            confidence=0.0,
            used_model=True,
            conversation_id=conversation.id,
        )


def load_vision_assistant(
    model_path: str | Path | None = None,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    reference_chars: int = DEFAULT_REFERENCE_CHARS,
) -> VisionAssistant:
    """Build the Vision mouth and check it can answer.

    Raises :class:`VisionUnavailable` when the optional runtime or a model file
    is missing, so the caller can report the real next step *before* switching
    the chat to a mouth that could not answer. The heavy work is deferred: the
    engine loads the GGUF lazily on the first reply.
    """
    assistant = VisionAssistant(
        VisionEngine(model_path=model_path),
        max_tokens=max_tokens,
        temperature=temperature,
        reference_chars=reference_chars,
    )
    assistant.ensure_available()
    return assistant


__all__ = [
    "DEFAULT_REFERENCE_CHARS",
    "REFERENCE_LABEL",
    "SWITCH_TO_VISION",
    "VISION_EMPTY",
    "VISION_HELP",
    "VisionAssistant",
    "load_vision_assistant",
    "unavailable_message",
]
