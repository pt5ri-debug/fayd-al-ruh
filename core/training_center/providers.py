"""The learning providers: one small protocol over every source of answers.

The Training Center can gather answers from Gemini (or any configured
OpenAI-compatible cloud provider), from a local model, from one of the project's
characters, or from the project's own knowledge base. Each of those already has
its own engine in the project -- :mod:`core.cloud_model`, :mod:`ai.mouths`,
:mod:`core.rag` -- so this module is a *thin adapter*: it exposes them behind one
:class:`LearningProvider` protocol, so the review queue and the window do not need
to know which engine answered.

Two hard rules run through every provider:

* **nothing is invented.** A provider that cannot answer reports the real next
  step (no key, no model, no corpus) rather than a made-up answer;
* **nothing reaches the network unasked.** The cloud provider runs through the
  cloud engine's own permission + human-approval gate, so a bare provider with no
  approval reaches nothing.

The engines are imported lazily, so importing this module loads no model and no
Tk. A test injects its own provider, so the whole workflow is exercised with no
network, no account, and no weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ProviderAnswer:
    """One answer, with where it came from -- never presented as verified."""

    question: str
    text: str
    provider: str = ""
    model: str = ""
    citations: tuple[str, ...] = ()
    confidence: float = 0.0
    note: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())


@dataclass(frozen=True)
class ProviderStatus:
    """Whether a provider can answer now, and the real next step when it cannot."""

    key: str
    title: str
    available: bool
    detail: str = ""
    hint: str = ""
    local: bool = True
    requires_network: bool = False

    def describe(self) -> str:
        state = "ready" if self.available else "unavailable"
        line = f"{self.title}: {state}"
        if self.detail:
            line += f" — {self.detail}"
        if not self.available and self.hint:
            line += f" (next step: {self.hint})"
        return line


@runtime_checkable
class LearningProvider(Protocol):
    """What the Training Center needs from a source of answers."""

    #: Read-only: a provider's key and title are its own (a plain attribute or a
    #: property both satisfy this), never something a caller sets.
    @property
    def key(self) -> str: ...

    @property
    def title(self) -> str: ...

    def status(self) -> ProviderStatus: ...

    def ask(self, question: str, *, system: str = "") -> ProviderAnswer: ...

    def generate(self, topic: str, *, count: int = 5) -> tuple[str, ...]: ...


#: The fixed instruction used to ask a provider for questions on a topic. It is
#: deliberately simple and bilingual-friendly: the model is asked for one question
#: per line, nothing else, so the answer is trivially parsed and nothing is
#: invented by the parser.
GENERATE_INSTRUCTION = (
    "Write {count} short, distinct, factual questions about: {topic}. "
    "One question per line. No numbering, no answers, no extra text. "
    "If the topic is Arabic, write the questions in Arabic."
)


@dataclass
class ManualProvider:
    """The user's own answers: a provider that never calls anything.

    Used for a hand-written question/answer pair, so the review queue has one
    provider that is always available and always honest about where the answer
    came from.
    """

    key: str = "manual"
    title: str = "A hand-written answer"
    answers: dict[str, str] = field(default_factory=dict)

    def status(self) -> ProviderStatus:
        return ProviderStatus(
            key=self.key, title=self.title, available=True,
            detail="you type the answer yourself", local=True,
        )

    def ask(self, question: str, *, system: str = "") -> ProviderAnswer:
        return ProviderAnswer(
            question=question,
            text=self.answers.get(question, ""),
            provider=self.key,
            model="",
            note="no answer was supplied for this question",
        )

    def generate(self, topic: str, *, count: int = 5) -> tuple[str, ...]:
        return ()


class CloudLearningProvider:
    """Gemini (or any OpenAI-compatible provider) through the cloud engine.

    The engine is reached lazily and only through its own permission and
    human-approval gate: ``approved=True`` is the caller's explicit consent, the
    same click the chat's Cloud mouth is. Without it, :meth:`ask` raises the
    engine's own :class:`core.cloud_model.CloudUnavailable`.
    """

    requires_network = True

    def __init__(
        self,
        provider: str = "gemini",
        *,
        model: str | None = None,
        approved: bool = False,
        engine: Any = None,
        system_prompt: str = "",
    ) -> None:
        self.provider_key = provider
        self.model_name = model
        self.approved = approved
        self._engine = engine
        self._system = system_prompt

    @property
    def key(self) -> str:
        return self.provider_key

    @property
    def title(self) -> str:
        from core.cloud_model import resolve_provider

        return f"{resolve_provider(self.provider_key).title} (cloud)"

    def _build_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        from core.cloud_model import CloudEngine

        kwargs: dict[str, Any] = {"provider": self.provider_key, "approved": self.approved}
        if self.model_name:
            kwargs["model"] = self.model_name
        if self._system:
            kwargs["system_prompt"] = self._system
        self._engine = CloudEngine(**kwargs)
        return self._engine

    def status(self) -> ProviderStatus:
        from core.cloud_model import cloud_status, resolve_provider

        spec = resolve_provider(self.provider_key)
        state = cloud_status(provider=self.provider_key, model=self.model_name)
        return ProviderStatus(
            key=self.key,
            title=self.title,
            available=bool(state.available) and self.approved,
            detail=state.detail or state.summary(),
            hint=state.hint or ("set the provider's API key, then approve the request" if not self.approved else ""),
            local=bool(getattr(spec, "local", False)),
            requires_network=not bool(getattr(spec, "local", False)),
        )

    def ask(self, question: str, *, system: str = "") -> ProviderAnswer:
        engine = self._build_engine()
        text = engine.reply(question)
        return ProviderAnswer(
            question=question,
            text=text,
            provider=self.key,
            model=engine.model,
            note="a cloud model's own prose, not grounded in cited sources",
        )

    def generate(self, topic: str, *, count: int = 5) -> tuple[str, ...]:
        instruction = GENERATE_INSTRUCTION.format(count=count, topic=topic)
        answer = self.ask(instruction)
        return _parse_questions(answer.text, count)


class LocalModelLearningProvider:
    """An installed local model, built through :func:`ai.mouths.build_mouth`.

    The mouth is built lazily (loading the weights only when a question is really
    asked) and through the same loader the per-mouth chat windows use, so the
    choice of model is honoured in exactly one way. A mouth that cannot load
    reports its own availability error with the real next step.
    """

    def __init__(self, mouth: str = "ultron", *, model_key: str | None = None) -> None:
        self.mouth = mouth
        self.model_key = model_key
        self._assistant: Any = None

    @property
    def key(self) -> str:
        return f"local:{self.mouth}"

    @property
    def title(self) -> str:
        return f"the local model {self.mouth}"

    def _build(self) -> Any:
        if self._assistant is not None:
            return self._assistant
        from ai.mouths import build_mouth

        self._assistant = build_mouth(self.mouth, model_key=self.model_key)
        return self._assistant

    def status(self) -> ProviderStatus:
        try:
            from core import language_models as catalog

            model = catalog.ready_language_model(self.mouth)
        except Exception as error:  # pragma: no cover - defensive
            return ProviderStatus(
                key=self.key, title=self.title, available=False, detail=str(error),
            )
        if model is None:
            return ProviderStatus(
                key=self.key, title=self.title, available=False,
                detail=f"no model is installed for {self.mouth}",
                hint="download a model for this mouth first",
            )
        return ProviderStatus(
            key=self.key, title=self.title, available=True,
            detail=f"{model.title} ({'ready' if model.available else 'not downloaded'})",
            local=True,
        )

    def ask(self, question: str, *, system: str = "") -> ProviderAnswer:
        assistant = self._build()
        result = assistant.ask(question)
        text = _answer_text(result)
        return ProviderAnswer(
            question=question, text=text, provider=self.key, model=self.mouth,
            note="a local model's own prose, not grounded in cited sources",
        )

    def generate(self, topic: str, *, count: int = 5) -> tuple[str, ...]:
        instruction = GENERATE_INSTRUCTION.format(count=count, topic=topic)
        answer = self.ask(instruction)
        return _parse_questions(answer.text, count)


class CharacterLearningProvider(LocalModelLearningProvider):
    """One of the project's characters as a learning participant.

    The character's own mouth is built (its personality, memory, and knowledge
    scope are its own), but nothing here *changes* a character's personality: it
    only asks, and attributes the answer to the character.
    """

    def __init__(self, character: str) -> None:
        super().__init__(mouth=character)
        self.character = character

    @property
    def key(self) -> str:
        return f"character:{self.character}"

    @property
    def title(self) -> str:
        return f"the character {self.character}"


class DocumentLearningProvider:
    """The project's own knowledge base, read through its retriever.

    This provider never calls a model: it answers with the passages the project's
    own retrieval returns, each with its citation, so the answer is traceable.
    """

    def __init__(self, *, retriever: Any = None) -> None:
        self._retriever = retriever

    key = "document"
    title = "the project's own knowledge base"

    def _build(self) -> Any:
        if self._retriever is not None:
            return self._retriever
        from core.documents import build_document_store
        from core.rag import RagRetriever

        self._retriever = RagRetriever(build_document_store())
        return self._retriever

    def status(self) -> ProviderStatus:
        try:
            retriever = self._build()
            count = len(getattr(retriever, "store", []) or [])
        except Exception as error:  # pragma: no cover - defensive
            return ProviderStatus(
                key=self.key, title=self.title, available=False, detail=str(error),
            )
        available = count > 0
        return ProviderStatus(
            key=self.key, title=self.title, available=available,
            detail=f"{count} document(s) indexed" if available else "the store is empty",
            hint="" if available else "import documents into the knowledge base first",
            local=True,
        )

    def ask(self, question: str, *, system: str = "") -> ProviderAnswer:
        retriever = self._build()
        passages = retriever.retrieve(question, limit=4)
        text = "\n".join(p.text for p in passages)
        citations = tuple(p.citation for p in passages)
        return ProviderAnswer(
            question=question, text=text, provider=self.key, model="retrieval",
            citations=citations,
            note="extracted from the project's own sources, with citations",
        )

    def generate(self, topic: str, *, count: int = 5) -> tuple[str, ...]:
        return ()


def _answer_text(result: Any) -> str:
    """Read the answer text out of whatever a mouth's ``ask`` returned."""
    for attr in ("text", "answer", "content", "reply"):
        value = getattr(result, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    if isinstance(result, str):
        return result
    return str(result)


def _parse_questions(text: str, count: int) -> tuple[str, ...]:
    """One question per line, de-duplicated and bounded -- nothing invented.

    A line that is empty, a bare number, or a bullet with nothing after it is
    dropped rather than turned into a question.
    """
    questions: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*0123456789. )").strip()
        if not line or len(line) < 4:
            continue
        if not line.endswith(("?", "؟", ".", "!")):
            line = line.rstrip(".") + "?"
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        questions.append(line)
        if len(questions) >= max(1, count):
            break
    return tuple(questions)


def provider_names() -> tuple[str, ...]:
    """Every provider kind the Training Center can build, for a menu."""
    return ("manual", "document", "gemini", "local", "character")


def build_provider(kind: str, **kwargs: Any) -> LearningProvider:
    """Build one provider by kind, so a surface never imports an engine itself."""
    if kind == "manual":
        return ManualProvider(**kwargs)
    if kind == "document":
        return DocumentLearningProvider(**kwargs)
    if kind in ("gemini", "cloud"):
        return CloudLearningProvider(**kwargs)
    if kind == "local":
        return LocalModelLearningProvider(**kwargs)
    if kind == "character":
        return CharacterLearningProvider(**kwargs)
    raise ValueError(f"Unknown learning provider {kind!r}; expected one of {provider_names()}.")


def provider_statuses(*, cloud_provider: str = "gemini") -> tuple[ProviderStatus, ...]:
    """The status of every provider the Training Center offers, for the window."""
    return (
        DocumentLearningProvider().status(),
        CloudLearningProvider(cloud_provider).status(),
    )


__all__ = [
    "GENERATE_INSTRUCTION",
    "CharacterLearningProvider",
    "CloudLearningProvider",
    "DocumentLearningProvider",
    "LearningProvider",
    "LocalModelLearningProvider",
    "ManualProvider",
    "ProviderAnswer",
    "ProviderStatus",
    "build_provider",
    "provider_names",
    "provider_statuses",
]
