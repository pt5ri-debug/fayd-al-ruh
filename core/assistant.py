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
from core.documents import Document, DocumentStore
from core.model_interface import CognitiveModel, ModelRequest
from core.rag import RagRetriever, RetrievedPassage

SYSTEM_INSTRUCTION = (
    "You are a petroleum-engineering expert assistant. Answer using only the "
    "reference passages provided, cite your sources, and say plainly when the "
    "information is insufficient."
)

# Below this top-passage score the internal corpus is too thin to answer well,
# so the assistant consults the web. The value is transparent and deliberately
# inspectable; a strong internal hit (0.7+) never triggers a search.
_WEAK_EVIDENCE = 0.35


@dataclass(frozen=True)
class WebAugmentation:
    """What one web lookup contributed to an answer."""

    answer: str
    citations: tuple[str, ...]
    trust: float
    learned: bool
    learned_doc_id: str | None
    learned_document: Document | None = None
    reason: str = ""


@dataclass(frozen=True)
class AssistantAnswer:
    """An answer plus the evidence it was built from."""

    text: str
    citations: tuple[str, ...]
    passages: tuple[RetrievedPassage, ...]
    confidence: float
    used_model: bool
    conversation_id: str
    used_web: bool = False
    learned: bool = False
    learned_doc_id: str | None = None
    web_trust: float = 0.0

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
        web=None,
        web_enabled: bool = False,
        weak_evidence: float = _WEAK_EVIDENCE,
        on_learn=None,
    ) -> None:
        if not 0.0 <= weak_evidence <= 1.0:
            raise ValueError("weak_evidence must be between 0 and 1.")
        self.retriever = retriever
        self.model = model
        self.memory = memory or ConversationMemory()
        self.system_instruction = system_instruction
        self.web = web
        self.web_enabled = web_enabled and web is not None
        self.weak_evidence = weak_evidence
        self._on_learn = on_learn

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

        augmentation: WebAugmentation | None = None
        if self.web_enabled and _is_weak(passages, self.weak_evidence):
            augmentation = self._augment(question, passages)
            if augmentation is not None and augmentation.learned_document is not None:
                self._index_learned(augmentation)
                passages = self.retriever.retrieve(question, limit=limit)

        confidence = _confidence(passages)

        if self.model is not None and passages:
            text, used_model = self._generate(question, conversation, passages, temperature, seed)
        else:
            text = _extractive_answer(question, passages)
            used_model = False

        if augmentation is not None:
            text = _blend(text, augmentation)
            passages = list(passages) + list(_learned_passages(self.retriever, augmentation, passages))

        citations = tuple(dict.fromkeys(passage.citation for passage in passages))
        if augmentation is not None:
            citations = tuple(dict.fromkeys(citations + augmentation.citations))
        conversation.add_assistant(text, citations)
        self.memory.save(conversation)
        return AssistantAnswer(
            text=text,
            citations=citations,
            passages=tuple(passages),
            confidence=confidence,
            used_model=used_model,
            conversation_id=conversation.id,
            used_web=bool(augmentation and augmentation.answer.strip()),
            learned=bool(augmentation and augmentation.learned),
            learned_doc_id=augmentation.learned_doc_id if augmentation else None,
            web_trust=augmentation.trust if augmentation else 0.0,
        )

    def _augment(self, question: str, internal: list[RetrievedPassage]) -> WebAugmentation | None:
        """Search the web for a weak question, compare, and learn if trusted."""
        try:
            assessment = self.web.verify(question)
        except (ValueError, RuntimeError, OSError) as error:
            return WebAugmentation("", (), 0.0, False, None, reason=f"Web lookup failed: {error}")
        learned_document = None
        if assessment.learned and assessment.learned_doc_id:
            learned_document = self._learned_document(assessment.learned_doc_id)
        if learned_document is not None and callable(self._on_learn):
            self._on_learn(learned_document)
        return WebAugmentation(
            answer=assessment.answer,
            citations=assessment.citations,
            trust=assessment.trust,
            learned=assessment.learned,
            learned_doc_id=assessment.learned_doc_id,
            learned_document=learned_document,
            reason=assessment.reason,
        )

    def _learned_document(self, doc_id: str):
        documents = getattr(self.retriever, "documents", None)
        getter = getattr(documents, "get", None)
        return getter(doc_id) if callable(getter) else None

    def _index_learned(self, augmentation: WebAugmentation) -> None:
        """Make the freshly learned document retrievable for this very answer."""
        index_document = getattr(self.retriever, "index_document", None)
        if callable(index_document) and augmentation.learned_document is not None:
            try:
                index_document(augmentation.learned_document)
            except ValueError:
                pass

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


def _is_weak(passages: list[RetrievedPassage], threshold: float) -> bool:
    """True when the internal corpus answers too weakly to stand on its own."""
    if not passages:
        return True
    return passages[0].score < threshold


def _blend(extractive: str, augmentation: WebAugmentation) -> str:
    """Lead with the web answer, keeping the internal extracts after it.

    The search fires only when internal evidence is weak, so the web answer is
    the informative one; the internal extracts stay for provenance and are not
    thrown away, but they no longer bury the answer.
    """
    if not augmentation.answer.strip():
        return extractive
    if not extractive.strip():
        return augmentation.answer
    return f"{augmentation.answer}\n\nInternal knowledge on file:\n{extractive}"


def _learned_passages(retriever, augmentation: WebAugmentation, existing) -> tuple[RetrievedPassage, ...]:
    """Return the learned document as a passage for citation, if it is new."""
    document = augmentation.learned_document
    if document is None:
        return ()
    if any(getattr(passage, "doc_id", None) == document.doc_id for passage in existing):
        return ()
    return (
        RetrievedPassage(
            doc_id=document.doc_id,
            title=document.title,
            text=document.text,
            source=document.source,
            score=augmentation.trust,
            signal="learned",
        ),
    )


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
    web_enabled: bool = False,
    web=None,
    web_threshold: float = _WEAK_EVIDENCE,
    on_learn=None,
) -> GroundedAssistant:
    """Assemble the assistant over the project's own knowledge corpus.

    Pass ``documents`` to supply a (possibly persistent) store; the default is a
    fresh in-memory store seeded from the bundled corpus. With ``web_enabled``
    the assistant consults the internet when its own evidence is weak, compares
    the result with internal knowledge, and learns trusted answers into the same
    store (so they are retrievable immediately).
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
    store = documents or build_document_store()
    retriever = RagRetriever(store, ontology=ontology)

    if web_enabled and web is None:
        from core.web_verifier import (
            approved_gate,
            build_web_verifier,
            granted_permissions,
            internet_provider,
        )

        web = build_web_verifier(
            provider=internet_provider(),
            permissions=granted_permissions(),
            approvals=approved_gate(),
            retriever=retriever,
        )

    return GroundedAssistant(
        retriever,
        model=model,
        memory=memory,
        web=web,
        web_enabled=web_enabled,
        weak_evidence=web_threshold,
        on_learn=on_learn,
    )


__all__ = [
    "AssistantAnswer",
    "GroundedAssistant",
    "SYSTEM_INSTRUCTION",
    "WebAugmentation",
    "build_default_assistant",
]
