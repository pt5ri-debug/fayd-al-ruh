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
from typing import TYPE_CHECKING

from core.answer_confidence import AnswerConfidence, answer_confidence
from core.answer_decision import AnswerDecision, answer_decision
from core.answer_plan import AnswerPlan, answer_plan
from core.causal_paths import CausalExplanation, causal_explanations
from core.conversation import Conversation, ConversationMemory
from core.documents import Document, DocumentStore
from core.grounding import check_grounding
from core.model_interface import CognitiveModel, ModelRequest
from core.rag import RagRetriever, RetrievedPassage
from core.reasoning_report import ReasoningReport, reasoning_report
from core.relation_connections import ConceptConnection, concept_connections
from core.relation_consistency import RelationConflict, relation_conflicts
from core.relation_facts import RelationFact, relation_facts
from core.relation_inference import InferredRelation, inferred_relations

if TYPE_CHECKING:
    from core.rag import BulkCorpusRetriever

SYSTEM_INSTRUCTION = (
    "You are a petroleum-engineering expert assistant. Answer using only the "
    "reference passages provided, cite your sources, and say plainly when the "
    "information is insufficient."
)

# Below this top-passage score the internal corpus is too thin to answer well,
# so the assistant consults the web. The value is transparent and deliberately
# inspectable; a strong internal hit (0.7+) never triggers a search. Set at 0.5
# so unfamiliar petroleum terms -- which the ontology still associates above a
# low bar -- actually reach the web instead of being answered from a weak match.
_WEAK_EVIDENCE = 0.5


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
    relation_facts: tuple[RelationFact, ...] = ()
    """The typed relations the question's own concepts participate in.

    Returned as *data*, like the citations: the answer's prose stays the model's
    or the sources' words, and the graph's own edges are rendered once at the end
    by the caller that shows the answer. Empty when the graph holds no relation on
    the question's concepts, so nothing is ever invented.
    """
    relation_conflicts: tuple[RelationConflict, ...] = ()
    """Contradictory relations the graph holds on the question's own concepts.

    Returned as *data* for the same reason as the facts: an answer that states one
    side of a disagreement without naming the other is quietly misleading. Empty
    when the graph is consistent on the question's concepts.
    """
    inferred: tuple[InferredRelation, ...] = ()
    """Relations the graph *entails* on the question's own concepts.

    The graph's own reasoning (``core.ontology_reasoner``) is surfaced here for the
    first time: these are conclusions the program derived from stored edges through
    transparent rules, never stored themselves. Returned as *data* and rendered as
    a block that is labelled as an inference, so a reader can always tell a claim
    someone wrote from one the program derived. Empty when the rules derive nothing
    on the question's concepts, so nothing is ever invented.
    """
    connections: tuple[ConceptConnection, ...] = ()
    """Chains of typed relations connecting the concepts the question itself names.

    When a question names two concepts with no single edge between them, the graph
    can still say *how* they are connected (``core.relation_paths`` walks the
    explicit relations breadth-first). Returned as *data* and rendered under a label
    that says the chain was walked, not stored, so a reader can tell a multi-hop
    conclusion from a single written edge. Empty when the question names fewer than
    two concepts or no chain within the hop bound connects them.
    """
    causal_paths: tuple[CausalExplanation, ...] = ()
    """Directed causal chains the graph states between the concepts the question names.

    The project's whole causal layer (`CausalGraph`/`CausalReasoner`/
    `CausalExplainer`) was reachable only from the controller and the evaluation
    scenarios; this is the first time a causal mechanism reaches an answer. Only
    the ontology's **directed causal** edges are read, and the walk never reverses
    one, so an association is never reported as a cause. Returned as *data* and
    rendered under a label that says the path is directed (cause → effect). Empty
    when the question names fewer than two concepts or no causal chain connects
    them.
    """
    plan: AnswerPlan | None = None
    """The ordered procedure the corpus states, for a question that asks *how*.

    The planning layer (`Planner`/`Action`/`Goal`/`PlanStep`) was reachable only
    from the controller and the evaluation scenarios, so a procedural question
    ("how do I measure porosity on a core sample?") got a definition back, never
    the ordered procedure the corpus already states. This is the first time a plan
    reaches an answer: the corpus's own procedure for the entry the question is
    about, ordered by the project's real ``Planner.order_steps``. ``None`` when the
    question asks no procedure (a definitional "what is porosity?") or no entry
    matches -- never a guessed plan.
    """
    decision: AnswerDecision | None = None
    """The ranked recommendation the graph states, for a question that asks what to do.

    The decision layer (`DecisionEngine`/`UtilityModel`/`RiskModel`/
    `DecisionExplainer`) was reachable only from the controller and the evaluation
    scenarios, so a decisional question ("what should I do to increase
    permeability?") got a definition back, never the ranked recommendation the
    knowledge graph already states. This is the first time a recommendation reaches
    an answer: the graph's own typed edges into the concept the question is about,
    ranked by the project's real ``DecisionEngine``. ``None`` when the question
    asks nothing to do (a definitional "what is permeability?") or the graph states
    no option in the direction asked for -- never a guessed recommendation.
    """
    confidence_report: AnswerConfidence | None = None
    """How the confidence was reached: the evidence term, the graph support it
    gained, and the contradiction penalty it lost.

    The confidence number was always computed from the retrieved passages alone;
    this is the first time the graph's *reasoning* -- entailed relations, walked
    connections, stated causal paths -- and its *contradictions* weigh into it.
    Carried as *data* so a caller can show which term moved the number, and
    ``None`` only for an answer built without the report (a hand-made one).
    """
    reasoning: ReasoningReport | None = None
    """The whole trace of *why* this answer: what was read, what was derived, and
    how the confidence was reached, as one ordered, labelled account.

    The reasoning layer existed but no surface showed it; this is the first time
    an answer carries its own auditable trace (``core.reasoning_report``). Carried
    as *data* and rendered under a label, so a reader can see the steps rather
    than taking the answer on trust. ``None`` only for a hand-made answer.
    """

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
        facts = relation_facts(getattr(self.retriever, "ontology", None), question)
        graph_conflicts = relation_conflicts(getattr(self.retriever, "ontology", None), question)
        graph_inferences = inferred_relations(getattr(self.retriever, "ontology", None), question)
        graph_connections = concept_connections(getattr(self.retriever, "ontology", None), question)
        graph_causal = causal_explanations(getattr(self.retriever, "ontology", None), question)
        graph_plan = answer_plan(question)
        graph_decision = answer_decision(
            question, ontology=getattr(self.retriever, "ontology", None)
        )

        # The answer's confidence weighs the retrieved evidence *and* what the graph
        # reasoned: entailed relations, walked connections, stated causal paths, and
        # the contradictions it holds. Kept in one place so a reader can see which
        # term moved the number (``answer_confidence`` returns the breakdown).
        confidence_report = answer_confidence(
            passages,
            inferences=graph_inferences,
            connections=graph_connections,
            causal_paths=graph_causal,
            conflicts=graph_conflicts,
        )
        confidence = confidence_report.final
        # The whole trace of *why* this answer, assembled from the data above in one
        # place (``core.reasoning_report``), so a caller can show the steps a reader
        # would otherwise have to take on trust.
        trace = reasoning_report(
            question,
            passages=passages,
            relation_facts=facts,
            plan=graph_plan,
            decision=graph_decision,
            inferred=graph_inferences,
            connections=graph_connections,
            causal_paths=graph_causal,
            conflicts=graph_conflicts,
            confidence=confidence_report,
        )
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
            relation_facts=facts,
            relation_conflicts=graph_conflicts,
            inferred=graph_inferences,
            connections=graph_connections,
            causal_paths=graph_causal,
            plan=graph_plan,
            decision=graph_decision,
            confidence_report=confidence_report,
            reasoning=trace,
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
            context=(*context, conversation.render()),
        )
        try:
            response = self.model.generate(request)  # type: ignore[union-attr]
        except (ValueError, KeyError):
            return _extractive_answer(question, passages), False
        text = response.text.strip()
        if not text:
            return _extractive_answer(question, passages), False
        check = check_grounding(text, passages, ontology=getattr(self.retriever, "ontology", None))
        if not check.grounded:
            return _extractive_answer(question, passages), False
        return text, True


def _is_grounded(text: str, passages: list[RetrievedPassage]) -> bool:
    """Backward-compatible thin wrapper over :func:`core.grounding.check_grounding`.

    Kept because it is the historical name of the gate; the real work -- term
    coverage, per-sentence support, and causal-direction checking -- lives in
    :mod:`core.grounding`.
    """
    return check_grounding(text, passages).grounded


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


def _extractive_answer(question: str, passages: list[RetrievedPassage]) -> str:
    """A citation-only answer used when no model is available.

    Composed by :mod:`core.composition` so the fallback reads as connected prose
    rather than a raw dump, while keeping every claim's citation.
    """
    from core.composition import Source, grounded_answer

    sources = [
        Source(title=passage.title, text=passage.text, label=passage.citation)
        for passage in passages
    ]
    return grounded_answer(question, sources)


def _default_bulk_retriever() -> BulkCorpusRetriever | None:
    """A streaming retriever over the built knowledge base, or ``None`` if absent."""
    from core.knowledge_base import default_knowledge_base_dir, iter_local_documents
    from core.knowledge_index import KnowledgeTextIndex
    from core.rag import BulkCorpusRetriever

    store_dir = default_knowledge_base_dir()
    if not (store_dir / "manifest.json").exists() and not list(store_dir.glob("shard-*.jsonl")):
        return None
    index = KnowledgeTextIndex(store_dir)
    return BulkCorpusRetriever(
        lambda: iter_local_documents(store_dir),
        index=index if index.exists() else None,
    )


def bulk_knowledge_available() -> bool:
    """True when a built *and indexed* bulk knowledge base is present.

    The index is required because an unindexed bulk scan takes tens of seconds
    per query, which would stall an interactive voice/Jarvis turn.
    """
    from core.knowledge_base import default_knowledge_base_dir
    from core.knowledge_index import KnowledgeTextIndex

    return KnowledgeTextIndex(default_knowledge_base_dir()).exists()


def build_default_assistant(
    *,
    model: CognitiveModel | None = None,
    memory: ConversationMemory | None = None,
    documents: DocumentStore | None = None,
    web_enabled: bool = False,
    web=None,
    web_threshold: float = _WEAK_EVIDENCE,
    on_learn=None,
    bulk: bool = False,
) -> GroundedAssistant:
    """Assemble the assistant over the project's own knowledge corpus.

    Pass ``documents`` to supply a (possibly persistent) store; the default is a
    fresh in-memory store seeded from the bundled corpus. With ``web_enabled``
    the assistant consults the internet when its own evidence is weak, compares
    the result with internal knowledge, and learns trusted answers into the same
    store (so they are retrievable immediately). With ``bulk`` the assistant
    also consults a built bulk knowledge base (``python -m cli knowledge build``)
    as an additional, citation-carrying evidence source. When the fast full-text
    index has been built (``python -m cli knowledge build-index``) queries are
    served from it in milliseconds; otherwise they fall back to a streaming scan,
    and the store is never loaded whole either way.
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
    bulk_retriever = _default_bulk_retriever() if bulk else None
    retriever = RagRetriever(store, ontology=ontology, bulk=bulk_retriever)

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
    "SYSTEM_INSTRUCTION",
    "AssistantAnswer",
    "GroundedAssistant",
    "WebAugmentation",
    "build_default_assistant",
    "bulk_knowledge_available",
]
