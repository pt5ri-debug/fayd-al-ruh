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

from core.answer_analogy import AnswerAnalogy, answer_analogy, is_analogical
from core.answer_belief import AnswerBelief, answer_belief
from core.answer_confidence import AnswerConfidence, answer_confidence
from core.answer_decision import AnswerDecision, answer_decision
from core.answer_lead import AnswerLead, answer_lead
from core.answer_plan import AnswerPlan, answer_plan
from core.answer_temporal import AnswerTemporal, answer_temporal
from core.causal_paths import CausalExplanation, causal_explanations
from core.conversation import Conversation, ConversationMemory
from core.documents import Document, DocumentStore
from core.grounding import check_grounding
from core.knowledge_carry import CarryReport, LearnedRelation
from core.model_interface import CognitiveModel, ModelRequest
from core.rag import RagRetriever, RetrievedPassage
from core.reasoning_report import ReasoningReport, reasoning_report
from core.relation_connections import ConceptConnection, concept_connections
from core.relation_consistency import RelationConflict, relation_conflicts
from core.relation_facts import RelationFact, relation_facts
from core.relation_inference import InferredRelation, inferred_relations

if TYPE_CHECKING:
    from core.knowledge_carry import LearnedKnowledgeStore
    from core.ontology_learner import OntologyLearner, OntologyLearning
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
    belief: AnswerBelief | None = None
    """The evidence-weighed belief the graph supports, for a question that asks
    *whether a claim holds*.

    The belief layer (`Belief`/`BeliefRevisionEngine`/`ConfidenceModel`/
    `ContradictionDetector`) was reachable only from the controller and the
    evaluation scenarios, so an evaluative question ("is acid stimulation
    increasing permeability?") got a definition back, never the evidence-weighed
    verdict the knowledge graph already supports. This is the first time a belief
    reaches an answer: the graph's own typed edges between the concepts the
    question names, weighed by the project's real ``BeliefRevisionEngine`` (its own
    evidence-weighted posterior and contradiction detection). ``None`` when the
    question asks no verdict (a definitional "what is permeability?") or the graph
    holds no edge between the concepts it names -- never a guessed belief.
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
    lead: AnswerLead | None = None
    """The reasoning the question warranted, as the **answer body's own opening**.

    Phases 41-43 bridged planning, decision, and belief to the answer, but the
    bridge stopped at the trailing *trace*: the answer body was still the
    retrieval template, so the reasoning sat in a footer a user may never open.
    This is the reasoning *in the answer itself* -- the lead the body opens with,
    chosen strongest-first (procedure, then recommendation, then verdict, then
    timeline) by :mod:`core.answer_lead`. ``None`` when the question warranted no
    reasoning, so a definitional question keeps the retrieval answer it had and
    nothing is invented to fill a lead.
    """
    temporal: AnswerTemporal | None = None
    """The timeline the conversation holds, for a question that asks *what changed*.

    The temporal layer (`TemporalReasoner`/`Timeline`/`TemporalEvent`/
    `TemporalPattern`/`TimePrediction`) was reachable only from tests, so a
    question like *"what changed over the session?"* got a definition back, never
    the ordered account of what actually happened. This is the first time a
    timeline reaches an answer: the conversation's own timestamped turns, ordered
    by the project's real ``Timeline`` and related by its real ``TemporalReasoner``
    (with its own pattern detection and next-time prediction). ``None`` when the
    question asks nothing temporal or the conversation holds fewer than two turns
    -- never a guessed timeline.
    """
    analogy: AnswerAnalogy | None = None
    """The structural comparison the graph holds, for a question that asks how two
    things are alike.

    The analogical layer (`AnalogyEngine`/`StructureMappingEngine`) was reachable
    only from the controller, the ``analogy`` CLI, and the evaluation scenarios, so
    a question like *"is porosity like permeability?"* got a definition back -- or,
    worse, the belief bridge read its ``is ... ?`` opening as a verdict about a
    *different* claim. This is the first time a structural comparison reaches an
    answer: the two concepts' relational neighbourhoods aligned by the real
    structure-mapping engine, with its own matches, its own systematicity, and its
    own candidate inferences (never asserted). ``None`` when the question asks no
    comparison, names fewer than two concepts, or the mapping has no match to show.
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
        learned: LearnedKnowledgeStore | None = None,
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
        # The learned-knowledge carry (Gap 8): a store the answer path writes the
        # graph's *learned* relations to and reads them back from, so knowledge
        # gained in one session reaches the next. Optional and inert by default,
        # so a read-only session is untouched.
        self.learned = learned
        self._learner: OntologyLearner | None = None

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

        # The reasoning layers are read *before* the answer is written, because the
        # lead they produce is the answer's own opening (``core.answer_lead``): a
        # procedural question is answered with the procedure, a decisional one with
        # the recommendation, and an evaluative one with the verdict -- not with a
        # footnote a user may never open.
        ontology = getattr(self.retriever, "ontology", None)
        graph_plan = answer_plan(question)
        graph_decision = answer_decision(question, ontology=ontology)
        graph_analogy = answer_analogy(question, ontology=ontology)
        # A question that asks *how two things are alike* is a comparison, not a
        # yes/no claim about a relation between them: without this, "is porosity
        # like permeability?" would also earn a belief verdict about a *different*
        # statement ("does porosity increase permeability?") and the trace would
        # show reasoning the reader never asked for.
        graph_belief = None if is_analogical(question) else answer_belief(question, ontology=ontology)
        graph_temporal = answer_temporal(question, conversation=conversation)
        lead = answer_lead(
            question,
            plan=graph_plan,
            decision=graph_decision,
            analogy=graph_analogy,
            belief=graph_belief,
            temporal=graph_temporal,
        )
        lead_text = lead.describe() if lead is not None else ""

        if self.model is not None and passages:
            text, used_model = self._generate(
                question, conversation, passages, temperature, seed, lead=lead_text
            )
        else:
            text = _extractive_answer(question, passages, lead=lead_text)
            used_model = False

        if augmentation is not None:
            text = _blend(text, augmentation)
            passages = list(passages) + list(_learned_passages(self.retriever, augmentation, passages))

        citations = tuple(dict.fromkeys(passage.citation for passage in passages))
        if augmentation is not None:
            citations = tuple(dict.fromkeys(citations + augmentation.citations))
        facts = relation_facts(ontology, question)
        graph_conflicts = relation_conflicts(ontology, question)
        graph_inferences = inferred_relations(ontology, question)
        graph_connections = concept_connections(ontology, question)
        graph_causal = causal_explanations(ontology, question)

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
            analogy=graph_analogy,
            belief=graph_belief,
            inferred=graph_inferences,
            connections=graph_connections,
            causal_paths=graph_causal,
            temporal=graph_temporal,
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
            belief=graph_belief,
            confidence_report=confidence_report,
            reasoning=trace,
            lead=lead,
            temporal=graph_temporal,
            analogy=graph_analogy,
        )

    def remember_relation(
        self,
        source: str,
        relation_type: str,
        target: str,
        confidence: float = 0.5,
    ) -> LearnedRelation:
        """Teach the graph one typed relation and persist it for the next session.

        This is the write half of the carry (Gap 8): the relation joins the
        ontology through the project's own ``OntologyLearner`` (so it is bounded by
        the learner's ceiling and can never override a seed), the retrieval cache is
        dropped so the very next answer can use it, and -- when a carry store is
        configured -- the learned set is captured so it survives the process.

        The learner is created lazily and cached, so repeated calls accumulate into
        one learner and the capture reflects every relation learned this session.
        """
        from core.ontology_learner import OntologyLearner

        ontology = getattr(self.retriever, "ontology", None)
        if ontology is None:
            raise ValueError("This assistant has no ontology to teach.")
        if self._learner is None:
            self._learner = OntologyLearner(ontology)
        learning = self._learner.observe_relation(source, target, relation_type, confidence)
        if learning.changed:
            invalidate = getattr(self.retriever, "invalidate", None)
            if callable(invalidate):
                invalidate()
        record = LearnedRelation(
            source=source,
            relation_type=relation_type,
            target=target,
            confidence=confidence,
        )
        if self.learned is not None:
            self.learned.capture(ontology, self._learner)
        return record

    def recall_learned(self) -> CarryReport:
        """Replay every carried learned relation into this session's ontology.

        The read half of the carry: called once at startup (see
        ``build_default_assistant``) so a relation learned in an earlier session is
        in the graph before the first question. Idempotent, so calling it again
        adds nothing, and it never invents a concept. Returns ``None``-safe report;
        with no store configured it is an empty report.
        """
        if self.learned is None:
            return CarryReport(restored=(), already_present=0, skipped=())
        from core.ontology_learner import OntologyLearner

        ontology = getattr(self.retriever, "ontology", None)
        if ontology is None:
            return CarryReport(restored=(), already_present=0, skipped=())
        if self._learner is None:
            self._learner = OntologyLearner(ontology)
        report = self.learned.restore(ontology, self._learner)
        if report.changed:
            invalidate = getattr(self.retriever, "invalidate", None)
            if callable(invalidate):
                invalidate()
        return report

    def learn_from_text(self, text: str) -> OntologyLearning:
        """Read typed relations out of a sentence and carry them across sessions.

        The feedback path's bridge (Gap 8): a confirmed or corrected statement is
        read by the project's own ``RelationExtractor``, which maps the sentence's
        verb phrase onto one of the ontology's real relation types, and every
        candidate joins the graph through the learner -- bounded by its ceiling and
        restricted to concepts the ontology already knows. The learned set is then
        captured to the carry store, so a relation taught in this session is in the
        graph the next time.

        Returns the ``OntologyLearning`` (what was added or reinforced); an empty
        learning means the sentence carried no typed relation the ontology could
        read, which is an honest "nothing learned" rather than an error.
        """
        from core.ontology_learner import OntologyLearner, OntologyLearning
        from core.relation_extraction import RelationExtractor

        ontology = getattr(self.retriever, "ontology", None)
        if ontology is None or not text.strip():
            return OntologyLearning(concepts=(), added=(), skipped=0)
        if self._learner is None:
            self._learner = OntologyLearner(ontology)
        extractor = RelationExtractor(ontology, self._learner)
        learning = extractor.apply(text)
        if learning.changed:
            invalidate = getattr(self.retriever, "invalidate", None)
            if callable(invalidate):
                invalidate()
            if self.learned is not None:
                self.learned.capture(ontology, self._learner)
        return learning

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
        *,
        lead: str = "",
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
            return _extractive_answer(question, passages, lead=lead), False
        text = response.text.strip()
        if not text:
            return _extractive_answer(question, passages, lead=lead), False
        check = check_grounding(text, passages, ontology=getattr(self.retriever, "ontology", None))
        if not check.grounded:
            return _extractive_answer(question, passages, lead=lead), False
        # The reasoning leads even when a model wrote the body, so the answer a
        # reader sees always carries the project's own reasoning *and* the model's
        # prose -- never one silently in place of the other.
        return _with_lead(question, lead, text), True


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


def _extractive_answer(
    question: str, passages: list[RetrievedPassage], *, lead: str = ""
) -> str:
    """A citation-only answer used when no model is available.

    Composed by :mod:`core.composition` so the fallback reads as connected prose
    rather than a raw dump, while keeping every claim's citation. ``lead`` is the
    reasoning the question warranted (``core.answer_lead``); it opens the answer
    and the retrieved evidence follows under a label, so a procedural question is
    answered with the procedure rather than with a definition and a footnote.
    """
    from core.composition import Source, grounded_answer

    sources = [
        Source(title=passage.title, text=passage.text, label=passage.citation)
        for passage in passages
    ]
    return grounded_answer(question, sources, lead=lead)


def _with_lead(question: str, lead: str, body: str) -> str:
    """Open an answer with its reasoning lead, then the model's own prose."""
    from core.composition import with_lead

    return with_lead(question, lead, body)


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
    learned: LearnedKnowledgeStore | None = None,
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

    Pass ``learned`` (a :class:`core.knowledge_carry.LearnedKnowledgeStore`) to
    carry the graph's *learned* relations across sessions: the store is replayed
    into the fresh ontology before the first question, so knowledge gained in an
    earlier session is already in the graph (Gap 8). With no store the assistant
    is exactly as before -- a read-only session never touches disk.
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

    assistant = GroundedAssistant(
        retriever,
        model=model,
        memory=memory,
        web=web,
        web_enabled=web_enabled,
        weak_evidence=web_threshold,
        on_learn=on_learn,
        learned=learned,
    )
    if learned is not None:
        # Replay the previous session's learned relations before the first
        # question, so the graph is whole from the start.
        assistant.recall_learned()
    return assistant


__all__ = [
    "SYSTEM_INSTRUCTION",
    "AssistantAnswer",
    "GroundedAssistant",
    "WebAugmentation",
    "build_default_assistant",
    "bulk_knowledge_available",
]
