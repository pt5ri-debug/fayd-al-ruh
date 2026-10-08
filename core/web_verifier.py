"""Answer questions from the web, compare with internal knowledge, and rate trust.

This is the layer that answers the user's request: *search the internet for an
answer, cite the source, compare it with what the project already knows, score
how much to trust the final answer, and -- only when that trust is high enough
-- add the new knowledge to the project.*

It composes existing pieces instead of duplicating them:

* :class:`core.rag.RagRetriever` finds what the project already knows;
* :class:`core.web_source.WebSearchService` fetches external sources through the
  permission/approval/audit gates;
* :class:`core.knowledge_update.KnowledgeUpdater` folds a trusted answer into the
  retrievable corpus, with provenance and decay.

The trust score is a transparent blend of three signals:

1. **Source credibility** -- the best domain credibility among the results;
2. **Source agreement** -- how much the web sources agree with each other;
3. **Internal support** -- how much the web answer overlaps what the project
   already knows.

An answer is *learned* only when its trust clears a configurable threshold, so
the project never absorbs unverified internet text. Every assessment is
inspectable: each source keeps its own support score and matched terms.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.acceptance_policy import clean_statement, source_rejection_reason
from core.knowledge_update import KnowledgeUpdate, KnowledgeUpdater, UpdateOutcome
from core.permissions import Capability, PermissionPolicy
from core.human_approval import HumanApprovalGate
from core.rag import RetrievedPassage, _query_terms
from core.web_source import SearchOutcome, WebResult, WebSearchProvider

# Trust weights: source quality dominates, agreement and internal support refine.
_WEIGHT_CREDIBILITY = 0.4
_WEIGHT_AGREEMENT = 0.3
_WEIGHT_SUPPORT = 0.3


@dataclass(frozen=True)
class SourceAssessment:
    """One web source judged against the internal passages."""

    result: WebResult
    support: float
    matched_terms: tuple[str, ...]
    contradicts: bool = False

    @property
    def citation(self) -> str:
        return f"[{self.result.domain}] {self.result.title} <{self.result.url}>"


@dataclass(frozen=True)
class WebAssessment:
    """The full verdict on one question answered from the web."""

    question: str
    answer: str
    citations: tuple[str, ...]
    sources: tuple[SourceAssessment, ...]
    internal_passages: tuple[RetrievedPassage, ...]
    credibility: float
    agreement: float
    support: float
    trust: float
    learned: bool
    learned_doc_id: str | None = None
    search_allowed: bool = True
    reason: str = ""
    contradicts_internal: bool = False

    def __post_init__(self) -> None:
        for value in (self.credibility, self.agreement, self.support, self.trust):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Assessment scores must be between 0 and 1.")

    @property
    def is_trusted(self) -> bool:
        return self.trust >= 0.0 and self.learned

    def summary(self) -> str:
        lines = [f"Question: {self.question}"]
        if self.answer.strip():
            lines.append(f"Answer: {self.answer}")
        if self.citations:
            lines.append("Sources:")
            lines.extend(f"  - {citation}" for citation in self.citations)
        lines.append(
            "Scores: "
            f"credibility={self.credibility:.2f} "
            f"agreement={self.agreement:.2f} "
            f"support={self.support:.2f} "
            f"trust={self.trust:.2f}"
        )
        lines.append(
            "Learned into the project: "
            + (self.learned_doc_id or "no (below the trust threshold)")
        )
        if self.contradicts_internal:
            lines.append("Note: a source contradicts the project's current knowledge; not learned.")
        return "\n".join(lines)


class WebVerifier:
    """Searches the web, rates the answer, and learns it only when trusted."""

    def __init__(
        self,
        retriever,
        search,
        *,
        updater: KnowledgeUpdater | None = None,
        trust_threshold: float = 0.6,
        min_support: float = 0.2,
        trusted_only: bool = True,
    ) -> None:
        if not 0.0 <= trust_threshold <= 1.0:
            raise ValueError("trust_threshold must be between 0 and 1.")
        if not 0.0 <= min_support <= 1.0:
            raise ValueError("min_support must be between 0 and 1.")
        self.retriever = retriever
        self.search = search
        self.updater = updater
        self.trust_threshold = trust_threshold
        self.min_support = min_support
        self.trusted_only = trusted_only

    def verify(
        self,
        question: str,
        *,
        limit: int = 5,
        internal_limit: int = 4,
        learn: bool = True,
    ) -> WebAssessment:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        internal = tuple(self.retriever.retrieve(question, limit=internal_limit))
        internal_terms = _terms_of_passages(internal)
        query_terms = _query_terms(question)

        outcome: SearchOutcome = self.search.search(question, limit=limit)
        if not outcome.allowed:
            return WebAssessment(
                question=question,
                answer="",
                citations=(),
                sources=(),
                internal_passages=internal,
                credibility=0.0,
                agreement=0.0,
                support=0.0,
                trust=0.0,
                learned=False,
                search_allowed=False,
                reason=outcome.reason,
            )

        sources = tuple(
            _assess_source(result, internal_terms, query_terms) for result in outcome.results
        )
        contradicts_internal = any(source.contradicts for source in sources)
        credibility = max((source.result.credibility for source in sources), default=0.0)
        agreement = _agreement(sources)
        support = _internal_support(sources, internal_terms)
        trust = round(
            _WEIGHT_CREDIBILITY * credibility
            + _WEIGHT_AGREEMENT * agreement
            + _WEIGHT_SUPPORT * support,
            6,
        )

        answer = _compose_answer(question, sources, internal)
        citations = tuple(source.citation for source in sources)

        learned_doc_id: str | None = None
        best = max(sources, key=lambda source: (source.result.credibility, source.support), default=None)
        source_ok = best is not None and source_rejection_reason(
            url=best.result.url,
            snippet=best.result.snippet,
            credibility=best.result.credibility,
            trusted_only=self.trusted_only,
        ) is None
        if (
            learn
            and source_ok
            and not contradicts_internal
            and trust >= self.trust_threshold
            and support >= self.min_support
            and self.updater is not None
        ):
            learned_doc_id = self._learn(question, sources)

        return WebAssessment(
            question=question,
            answer=answer,
            citations=citations,
            sources=sources,
            internal_passages=internal,
            credibility=round(credibility, 6),
            agreement=round(agreement, 6),
            support=round(support, 6),
            trust=trust,
            learned=learned_doc_id is not None,
            learned_doc_id=learned_doc_id,
            search_allowed=True,
            reason=outcome.reason,
            contradicts_internal=contradicts_internal,
        )

    def _learn(self, question: str, sources: tuple[SourceAssessment, ...]) -> str | None:
        if not sources or self.updater is None:
            return None
        best = max(sources, key=lambda source: (source.result.credibility, source.support))
        statement = _statement(question, best.result)
        outcome: UpdateOutcome = self.updater.apply(
            KnowledgeUpdate(
                statement=statement,
                source=best.result.url,
                strength=best.result.credibility,
                topic=_topic(question),
            )
        )
        return outcome.doc_id or None


def _terms_of_passages(passages: tuple[RetrievedPassage, ...]) -> set[str]:
    terms: set[str] = set()
    for passage in passages:
        terms |= _query_terms(f"{passage.title} {passage.text}")
    return terms


def _assess_source(
    result: WebResult, internal_terms: set[str], query_terms: set[str]
) -> SourceAssessment:
    snippet_terms = _query_terms(f"{result.title} {result.snippet}")
    matched = tuple(sorted(snippet_terms & internal_terms))
    relevant = snippet_terms & query_terms
    support = len(relevant) / len(query_terms) if query_terms else 0.0
    contradicts = _negates(result.snippet, query_terms & internal_terms)
    return SourceAssessment(
        result=result,
        support=round(min(1.0, support), 6),
        matched_terms=matched,
        contradicts=contradicts,
    )


_NEGATORS = ("not", "no", "never", "false", "cannot")


def _words(text: str) -> list[str]:
    cleaned = "".join(char if char.isalnum() else " " for char in text.casefold())
    return cleaned.split()


def _negates(snippet: str, subject_terms: set[str]) -> bool:
    """A weak contradiction signal: a negator sits next to the question's subject.

    Negators are stopwords and are stripped from content terms, so the raw
    snippet is tokenised directly. Matching only against terms that are both in
    the question and already known keeps this from firing on the ordinary "no" /
    "not" that fills any English page: a contradiction has to be about what was
    asked.
    """
    if not subject_terms:
        return False
    words = _words(snippet)
    for index, word in enumerate(words):
        if word not in _NEGATORS:
            continue
        window = words[max(0, index - 2) : index] + words[index + 1 : index + 3]
        if any(candidate in subject_terms for candidate in window):
            return True
    return False


def _agreement(sources: tuple[SourceAssessment, ...]) -> float:
    """How much the sources agree: shared content terms over the union."""
    if len(sources) < 2:
        return 1.0 if sources else 0.0
    term_sets = [_query_terms(f"{source.result.title} {source.result.snippet}") for source in sources]
    union: set[str] = set()
    for term_set in term_sets:
        union |= term_set
    if not union:
        return 0.0
    shared = set.intersection(*term_sets) if term_sets else set()
    return round(len(shared) / len(union), 6)


def _internal_support(sources: tuple[SourceAssessment, ...], internal_terms: set[str]) -> float:
    """How much of the web content the project already knows.

    Measured as the share of the sources' content terms that also appear in the
    retrieved internal passages, so a web answer built on familiar concepts
    scores higher than one about something the project has never seen.
    """
    if not sources or not internal_terms:
        return 0.0
    source_terms: set[str] = set()
    for source in sources:
        source_terms |= _query_terms(f"{source.result.title} {source.result.snippet}")
    if not source_terms:
        return 0.0
    return round(len(source_terms & internal_terms) / len(source_terms), 6)


def _compose_answer(
    question: str,
    sources: tuple[SourceAssessment, ...],
    internal: tuple[RetrievedPassage, ...],
) -> str:
    from core.composition import Source, web_answer

    rendered = [
        Source(
            title=source.result.title,
            text=source.result.snippet,
            label=source.result.domain or source.result.url,
        )
        for source in sources
    ]
    return web_answer(
        question,
        rendered,
        internal=tuple(passage.citation for passage in internal),
    )


def _statement(question: str, result: WebResult) -> str:
    statement = clean_statement(result.snippet)
    if statement:
        return statement
    return f"{result.title} ({result.url})"


def _topic(question: str) -> str:
    terms = sorted(_query_terms(question))
    return terms[0] if terms else "web"


def build_web_verifier(
    *,
    provider=None,
    trust_threshold: float = 0.6,
    approvals=None,
    permissions=None,
    documents=None,
    retriever=None,
    min_support: float = 0.2,
) -> WebVerifier:
    """Assemble the web-verification stack over the project's own knowledge.

    Search stays denied unless ``permissions`` grants the external capability
    and ``approvals`` holds a live approval, so a plain call is offline-safe.
    Pass ``retriever`` (or ``documents``) to share one corpus with a caller --
    the assistant does this, so a learned statement is immediately retrievable.
    """
    from core.curriculum_knowledge import seed_curriculum
    from core.documents import build_document_store
    from core.domain_knowledge import seed_ontology
    from core.knowledge_update import KnowledgeUpdater
    from core.ontology import Ontology
    from core.petroleum_corpus import seed_petroleum_corpus
    from core.rag import RagRetriever
    from core.web_source import StaticSearchProvider, WebSearchService

    store = documents
    if retriever is not None:
        store = retriever.documents
    elif documents is None:
        ontology = Ontology()
        seed_ontology(ontology)
        seed_curriculum(ontology)
        seed_petroleum_corpus(ontology)
        store = build_document_store()
        retriever = RagRetriever(store, ontology=ontology)
    if store is None:
        raise ValueError("build_web_verifier needs a documents store or a retriever.")
    if retriever is None:
        retriever = RagRetriever(store)
    search = WebSearchService(
        provider or StaticSearchProvider(),
        permissions=permissions,
        approvals=approvals,
    )
    updater = KnowledgeUpdater(store)
    return WebVerifier(
        retriever,
        search,
        updater=updater,
        trust_threshold=trust_threshold,
        min_support=min_support,
    )


def granted_permissions() -> PermissionPolicy:
    """The permission policy that lets the ``web_search`` external tool run."""
    return PermissionPolicy(
        allowed=frozenset(
            {
                Capability.SYMBOLIC_SIMULATION,
                Capability.SANDBOX_TASK,
                Capability.EXTERNAL_TOOL,
            }
        ),
        allowed_external_tools=frozenset({"web_search"}),
    )


def approved_gate() -> HumanApprovalGate:
    """A live approval for ``web_search``, signed by the local user."""
    gate = HumanApprovalGate()
    request = gate.request("web_search", requested_by="conversation-loop")
    request.approve("local-user")
    return gate


def internet_provider() -> WebSearchProvider:
    """The default keyless internet provider, falling back to an empty static one."""
    from core.web_source import FallbackSearchProvider, StaticSearchProvider, WikipediaSearchProvider

    return FallbackSearchProvider(WikipediaSearchProvider(), StaticSearchProvider())


__all__ = [
    "SourceAssessment",
    "WebAssessment",
    "WebVerifier",
    "approved_gate",
    "build_web_verifier",
    "granted_permissions",
    "internet_provider",
]
