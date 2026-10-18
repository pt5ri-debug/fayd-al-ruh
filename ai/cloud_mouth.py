"""Answer from a remote cloud model -- ChatGPT, Claude, Gemini, and others.

The chat has five mouths already: the grounded assistant answers from the
petroleum corpus with citations, :mod:`ai.mind` speaks the small in-project
checkpoint, :mod:`ai.ultron_mouth` runs the *general* local model,
:mod:`ai.vision_mouth` the *petroleum* specialist, and :mod:`ai.banner_mouth` the
Presentation Studio's own model. This module adds the sixth one the chat's model
selector offers: **Cloud**, the one mouth that does *not* run on this machine
(see :mod:`core.cloud_model`). It talks to a remote, OpenAI-compatible endpoint
-- OpenAI's own API (the ChatGPT models), Anthropic, Gemini, Groq, OpenRouter,
Mistral, DeepSeek, or an OpenAI-compatible server on this machine.

It exists so the cloud model is a *mouth inside the project's own chat* -- the
same session, the same transcript, the same conversation controls -- rather than
a separate window or a separate loop. :class:`CloudAssistant` is duck-typed to the
grounded assistant's ``ask`` signature, so :class:`core.jarvis.JarvisMode` and
:class:`gui.service.ChatService` call it unchanged when the selector swaps it in;
:class:`ai.chat.ModelSwitcher` owns the swap, exactly as it does for the other
five mouths.

Three things stay honest, as everywhere else in the project:

* a cloud answer is the remote model's own prose, not a cited extract, so it is
  labelled with :data:`core.cloud_model.HEADER` and carries no citations;
* the cloud mouth is **off by default**: it is a governed external action, so a
  plain engine reaches nothing until the user selects this mouth (which is their
  explicit approval) -- and the API key is read from the environment, never
  written to the project;
* when the runtime, the key, or the endpoint is missing,
  :class:`CloudUnavailable` carries an actionable message (set an environment
  variable, check the endpoint) instead of silently answering from a different
  mouth.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.assistant import AssistantAnswer
from core.cloud_model import (
    HEADER as CLOUD_HEADER,
    NAME as CLOUD_NAME,
    CloudEngine,
    CloudUnavailable,
    selected_model,
    selected_provider,
)
from core.conversation import Conversation, Role

#: Shown in place of a body when the cloud model produced no text at all, so the
#: answer is never a header with nothing under it (which the window would draw as
#: an empty bubble carrying only its Copy button).
CLOUD_EMPTY = f"{CLOUD_NAME} produced no answer for this question."

#: One line explaining the cloud switch, appended by ``chat`` to the help text.
CLOUD_HELP = (
    "Cloud mode: type 'from cloud' to answer from a remote cloud model "
    "(ChatGPT/Claude/Gemini/... over the network, no citations), 'from the "
    "sources' to answer from retrieved passages with citations again."
)

#: The controls ``chat`` recognises to switch to the Cloud mouth. Kept here,
#: beside the mouth they select, and never added to the shared Jarvis controls.
SWITCH_TO_CLOUD = (
    "from cloud", "use cloud", "answer from cloud", "cloud on",
    "من السحابة", "السحابة", "استخدم السحابة",
)

#: Opens the reference block when the conversation carries an attached file, so
#: the model is told the text below is the user's own material rather than its
#: own earlier words.
REFERENCE_LABEL = "Reference material the user attached:"

#: How much of an attached file's text is handed to the model in one prompt. The
#: conversation keeps the whole file; a remote model's context window is finite,
#: so a longer file is cut here with a visible marker instead of being sent whole
#: (which also bounds how much of the user's file leaves the machine).
DEFAULT_REFERENCE_CHARS = 8000

_REFERENCE_CUT = "\n… [the rest of the attached file is not shown here]"

#: The honesty header a mouth writes above its answer. Dropped from a previous
#: assistant turn before it is fed back to the model, so the cloud model is not
#: prompted with a label it did not produce.
_LABEL_LINES = frozenset({CLOUD_HEADER})


def unavailable_message(engine: CloudEngine | None = None) -> str:
    """An honest, actionable line for why the cloud model cannot answer right now.

    Reuses the CLI's own status wording (``ai.cloud.status_line``), imported
    lazily so this module never pulls the CLI in just to be imported.
    """
    from ai.cloud import status_line

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


class CloudAssistant:
    """Answers from a remote cloud model, without retrieval or citations.

    Duck-typed to the grounded assistant's ``ask`` signature, so the Jarvis loop
    calls it unchanged when the chat's model selector selects it. It holds one
    :class:`~core.cloud_model.CloudEngine` for the life of the session, so the
    endpoint is configured once and follow-up questions keep their context.
    """

    def __init__(
        self,
        engine: CloudEngine,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reference_chars: int = DEFAULT_REFERENCE_CHARS,
    ) -> None:
        self.engine = engine
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reference_chars = reference_chars
        # Kept for the grounded assistant's shape; the cloud mouth never
        # retrieves, so there is nothing to persist here.
        self.memory = None

    def is_available(self) -> bool:
        """True when a reply could be attempted now (approval and key present)."""
        return self.engine.is_available()

    def ensure_available(self) -> None:
        """Raise :class:`CloudUnavailable` with the real next step, if needed."""
        if not self.engine.is_available():
            raise CloudUnavailable(unavailable_message(self.engine))

    def describe(self) -> str:
        """A one-line, honest description of the endpoint in use."""
        return self.engine.describe()

    def answer(self, question: str, *, history: Sequence[tuple[str, str]] = ()) -> str:
        """The cloud model's own reply to ``question``, in the context of ``history``."""
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
        """Match ``GroundedAssistant.ask``: record the turn and answer from the cloud.

        The history is taken *before* the question is appended, so the model is
        asked to continue the conversation rather than to complete its own prompt.
        Any reference material the conversation carries -- the text of a file the
        user attached -- is put in front of the question, so the cloud model can
        read it like the grounded mouth does. The answer carries no citations on
        purpose: it is the remote model's own words.
        """
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        if conversation is None:
            conversation = Conversation(title=question.strip()[:40])
        history = _history(conversation)
        prompt = _context_prompt(conversation, question, limit=self.reference_chars)
        conversation.add_user(question)
        body = self.answer(prompt, history=history) or CLOUD_EMPTY
        text = f"{CLOUD_HEADER}\n\n{body}"
        conversation.add_assistant(text, ())
        return AssistantAnswer(
            text=text,
            citations=(),
            passages=(),
            confidence=0.0,
            used_model=True,
            conversation_id=conversation.id,
        )


def load_cloud_assistant(
    model: str | None = None,
    *,
    provider: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    reference_chars: int = DEFAULT_REFERENCE_CHARS,
    client=None,
) -> CloudAssistant:
    """Build the Cloud mouth and check it can answer.

    Raises :class:`CloudUnavailable` when no key is set (or the endpoint cannot
    answer), so the caller can report the real next step *before* switching the
    chat to a mouth that could not answer. The engine is built with a *live*
    approval: selecting this mouth is the user's explicit approval to send their
    words to the remote model. ``client`` is the seam a test injects, so the whole
    mouth is exercised with no network and no account.

    When ``provider`` is ``None`` the provider the user last chose (through
    ``python -m cli cloud use <key>`` or the window's provider picker) is used, so
    the mouth answers from the same provider the rest of the program does; only an
    explicit name overrides the saved choice. The *model* follows the same rule: an
    explicit ``model`` wins, else the model saved for that provider, else the
    provider's own default -- so a provider whose default was retired can be
    pointed at a working model once and every later answer uses it.
    """
    if provider is None and client is None:
        from core.paths import data_dir

        provider = selected_provider(data_dir()).key
    if model is None and client is None:
        from core.paths import data_dir

        model = selected_model(data_dir(), provider)
    engine = CloudEngine(
        client,
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        api_key_env=api_key_env,
        max_tokens=max_tokens,
        temperature=temperature,
        approved=True,
    )
    assistant = CloudAssistant(
        engine,
        max_tokens=max_tokens,
        temperature=temperature,
        reference_chars=reference_chars,
    )
    assistant.ensure_available()
    return assistant


__all__ = [
    "CLOUD_EMPTY",
    "CLOUD_HELP",
    "DEFAULT_REFERENCE_CHARS",
    "REFERENCE_LABEL",
    "SWITCH_TO_CLOUD",
    "CloudAssistant",
    "load_cloud_assistant",
    "unavailable_message",
]
