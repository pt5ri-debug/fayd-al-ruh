"""A grounded, evidence-aware domain assistant.

This is the user-facing layer of the project: it answers a question in English,
retrieves the passages that support the answer, and returns an answer that
*cites* those passages. It composes the pieces that already exist rather than
duplicating them:

* :class:`core.conversation.ConversationMemory` keeps the multi-turn context;
* :class:`core.rag.RagRetriever` finds grounded passages;
* an optional :class:`core.model_interface.CognitiveModel` (the in-project
  language model, or any injected model) writes the prose.

Two guarantees matter and are deliberate:

1. The model only ever receives a rendered prompt. It cannot touch the knowledge
   graph, mutate beliefs, or run tools.
2. When no model is available, the assistant still answers -- extractively, from
   the retrieved passages -- so the project never needs an external service to
   be useful. Every answer carries the citations it was built from.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.conversation import Conversation, ConversationMemory
from core.documents import DocumentStore
from core.model_interface import CognitiveModel, ModelRequest
from core.rag import RagRetriever, RetrievedPassage

SYSTEM_INSTRUCTION = (
    "You are a petroleum-engineering expert assistant. Answer using only the "
    "reference passages provided, cite your sources, and say plainly when the "
    "information is insufficient."
)


@dataclass(frozen=True)
class AssistantAnswer:
    """An answer plus the evidence it was built from."""

    text: str
    citations: tuple[str, ...]
    passages: tuple[RetrievedPassage, ...]
    confidence: float
    used_model: bool
    conversation_id: str

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Assistant answer cannot be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Answer confidence must be between 0 and 1.")


class GroundedAssistant:
    """Answers questions from retrieved evidence, with citations and memory."""

    def __init__(
        self,
        retriever: RagRetriever,
        *,
        model: CognitiveModel | None = None,
        memory: ConversationMemory | None = None,
        system_instruction: str = SYSTEM_INSTRUCTION,
    ) -> None:
        self.retriever = retriever
        self.model = model
        self.memory = memory or ConversationMemory()
        self.system_instruction = system_instruction

    def ask(
        self,
        question: str,
        *,
        conversation: Conversation | None = None,
        limit: int = 4,
        temperature: float = 0.7,
        seed: int | None = None,
    ) -> AssistantAnswer:
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        if conversation is None:
            conversation = self.memory.start(title=question.strip()[:40])

        conversation.add_user(question)
        passages = self.retriever.retrieve(question, limit=limit)
        confidence = _confidence(passages)

        if self.model is not None and passages:
            text, used_model = self._generate(question, conversation, passages, temperature, seed)
        else:
            text = _extractive_answer(question, passages)
            used_model = False

        citations = tuple(passage.citation for passage in passages)
        conversation.add_assistant(text, citations)
        self.memory.save(conversation)
        return AssistantAnswer(
            text=text,
            citations=citations,
            passages=tuple(passages),
            confidence=confidence,
            used_model=used_model,
            conversation_id=conversation.id,
        )

    def _generate(
        self,
        question: str,
        conversation: Conversation,
        passages: list[RetrievedPassage],
        temperature: float,
        seed: int | None,
    ) -> tuple[str, bool]:
        context = tuple(
            f"{passage.citation}\n{passage.text}" for passage in passages
        )
        request = ModelRequest(
            inputs=(),
            system_instruction=self.system_instruction,
            context=context + (conversation.render(),),
        )
        try:
            response = self.model.generate(request)  # type: ignore[union-attr]
        except (ValueError, KeyError):
            return _extractive_answer(question, passages), False
        text = response.text.strip()
        if not text or not _is_grounded(text, passages):
            return _extractive_answer(question, passages), False
        return text, True


def _is_grounded(text: str, passages: list[RetrievedPassage]) -> bool:
    """A weak groundedness gate: the answer must share a content term with its evidence.

    The in-project model is tiny, so it can produce fluent-looking text that has
    nothing to do with the retrieved passages. Refusing such output and falling
    back to the extractive answer keeps every published answer traceable to a
    source.
    """
    from core.rag import _query_terms

    answer_terms = _query_terms(text)
    evidence_terms: set[str] = set()
    for passage in passages:
        evidence_terms |= _query_terms(f"{passage.title} {passage.text}")
    return bool(answer_terms & evidence_terms)


def _confidence(passages: list[RetrievedPassage]) -> float:
    """Evidence-aware confidence: coverage and strength of the retrieved passages."""
    if not passages:
        return 0.0
    top = passages[0].score
    coverage = min(len(passages), 4) / 4
    return round(min(1.0, 0.7 * top + 0.3 * coverage), 6)


def _extractive_answer(question: str, passages: list[RetrievedPassage]) -> str:
    """A citation-only answer used when no model is available."""
    if not passages:
        return f"Not enough information to answer: {question}"
    lines = [f'According to the available sources for "{question}":']
    for passage in passages:
        first_lines = [line for line in passage.text.splitlines() if line.strip()][:3]
        lines.append(f"- {passage.title}: " + " ".join(first_lines))
    lines.append("Sources: " + "; ".join(passage.citation for passage in passages))
    return "\n".join(lines)


def build_default_assistant(
    *,
    model: CognitiveModel | None = None,
    memory: ConversationMemory | None = None,
    documents: DocumentStore | None = None,
) -> GroundedAssistant:
    """Assemble the assistant over the project's own knowledge corpus.

    Pass ``documents`` to supply a (possibly persistent) store; the default is a
    fresh in-memory store seeded from the bundled corpus.
    """
    from core.curriculum_knowledge import seed_curriculum
    from core.documents import build_document_store
    from core.domain_knowledge import seed_ontology
    from core.ontology import Ontology
    from core.petroleum_corpus import seed_petroleum_corpus

    ontology = Ontology()
    seed_ontology(ontology)
    seed_curriculum(ontology)
    seed_petroleum_corpus(ontology)
    retriever = RagRetriever(documents or build_document_store(), ontology=ontology)
    return GroundedAssistant(retriever, model=model, memory=memory)


__all__ = [
    "AssistantAnswer",
    "GroundedAssistant",
    "SYSTEM_INSTRUCTION",
    "build_default_assistant",
]
