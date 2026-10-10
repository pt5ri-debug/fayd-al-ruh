"""Deterministic, dependency-free prose composition.

The project's in-project language model is deliberately small, so its free
generation is neither fluent nor always on-topic. This module is the other half
of the answer: a rule-based *verbaliser* that turns evidence the project already
holds -- retrieved passages, web sources, causal and reasoning traces -- into
clean English sentences. Because every word is placed by a rule and every claim
is attributed to a citation, the output is professional in form and fully
traceable in substance, with no hallucination and no external service.

Composition is intentionally shallow: it does not invent facts, it *renders*
facts. Callers hand it structured evidence (see :class:`Source`) and it returns
readable prose. The helpers at the bottom let the reasoning explainers share the
same voice.

Attribution is kept *out* of the prose. The answer body states the facts, and the
citations travel alongside it as data (``GroundedAssistant.answer.citations``);
the command-line layers render them once, after the answer, so a citation never
interrupts a sentence. :func:`sources_block` builds that trailing block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

_WHITESPACE_RE = re.compile(r"\s+")
# Split on sentence-final punctuation followed by a new sentence. A decimal
# point ("0.052") has no space after it, so it is never split.
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\u201c])')
_LEAD_RE = re.compile(r"^\s*(?:[-*\u2022>]+\s*|\d+[.)]\s*)")
_TERMINAL = ".!?"

_CONNECTORS = ("First, ", "In addition, ", "Finally, ")
# A rendered statement with fewer words than this is a fragment (a bare heading
# such as "Wellbore."), not a quotable point, so it is skipped when better
# evidence exists.
_MIN_STATEMENT_WORDS = 3

# Question words and generic terms that name no topic. Dropped from the topic
# phrase so "what is the meaning of drilling oil wells" reads "drilling oil
# wells" rather than "drilling meaning oil wells".
_TOPIC_STOPWORDS = frozenset(
    {
        "what", "how", "why", "when", "where", "which", "who", "whom",
        "is", "are", "was", "were", "be", "being", "been",
        "the", "a", "an", "of", "to", "for", "in", "on", "at", "by", "with",
        "and", "or", "about", "into", "from", "that", "this", "these", "those",
        "mean", "means", "meaning", "definition", "define", "explain", "explanation",
        "please", "tell", "give", "list", "describe",
        "do", "does", "did", "can", "could", "should", "would", "will",
        # Arabic question and generic words
        "ما", "ماذا", "كيف", "هل", "من", "في", "على", "إلى", "عن", "هو", "هي",
        "معنى", "تعريف", "اشرح", "شرح",
    }
)
_TOPIC_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


@dataclass(frozen=True)
class Source:
    """One piece of evidence to render: a short text plus how to attribute it."""

    title: str
    text: str
    label: str = ""

    def __post_init__(self) -> None:
        if not self.title.strip() and not self.text.strip():
            raise ValueError("A source needs a title or a text.")


def clean(text: str) -> str:
    """Collapse whitespace and strip leading list markers and stray spacing."""
    value = _WHITESPACE_RE.sub(" ", text.replace("\n", " ")).strip()
    value = _LEAD_RE.sub("", value)
    return value.strip()


def sentences(text: str) -> list[str]:
    """Split text into clean sentences (order preserved, blanks dropped)."""
    value = clean(text)
    if not value:
        return []
    return [part.strip() for part in _SENTENCE_RE.split(value) if part.strip()]


def first_sentences(text: str, limit: int = 1) -> str:
    """The first ``limit`` sentences of ``text`` joined into one string."""
    return " ".join(sentences(text)[:limit])


def complete_sentences(text: str, limit: int | None = None) -> str:
    """The leading run of *whole* sentences in ``text``, dropping fragments.

    Learned text often begins mid-sentence ("drawbacks—mud filtrate soaks...")
    or ends mid-word ("...into the earth's"). Rendering such text produces broken
    prose, so only sentences that both start at a boundary and end with terminal
    punctuation survive. Returns ``""`` when no whole sentence is present.
    """
    value = clean(text)
    if not value:
        return ""
    if value[-1] not in _TERMINAL:
        boundary = max((value.rfind(mark) for mark in _TERMINAL), default=-1)
        value = value[: boundary + 1] if boundary >= 0 else ""
    parts = sentences(value)
    while parts and not (parts[0][0].isupper() or parts[0][0].isdigit()):
        parts.pop(0)
    if limit is not None:
        parts = parts[:limit]
    return " ".join(parts)


def ensure_terminal(text: str) -> str:
    """Capitalise the first letter and end with terminal punctuation."""
    value = clean(text)
    if not value:
        return value
    value = value[0].upper() + value[1:]
    if value[-1] not in _TERMINAL:
        value += "."
    return value


def topic_phrase(question: str) -> str:
    """A compact noun phrase naming what the question is about.

    Word order is preserved (``"drilling oil wells"``, not the alphabetised
    ``"drilling meaning oil wells"``) and question words are dropped, so the
    phrase reads as the subject the user named.
    """
    terms = [
        token
        for token in _TOPIC_TOKEN_RE.findall(question)
        if token.casefold() not in _TOPIC_STOPWORDS
    ]
    if terms:
        return " ".join(terms)
    value = clean(question).rstrip("?.!")
    return value or "this question"


def sources_block(citations: Sequence[str]) -> str:
    """The trailing ``Sources:`` block for an answer, de-duplicated in order."""
    unique = list(dict.fromkeys(clean(citation) for citation in citations if clean(citation)))
    if not unique:
        return ""
    return "Sources:\n" + "\n".join(f"  - {citation}" for citation in unique)


def _compose_statements(statements: Sequence[str]) -> str:
    """Join whole sentences into one flowing paragraph.

    Connectors are drawn from the shared vocabulary so every composed answer
    -- grounded, web, or model -- reads with the same voice.
    """
    cleaned = [statement.strip() for statement in statements if statement.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    parts: list[str] = []
    for index, statement in enumerate(cleaned):
        connector = _CONNECTORS[index] if index < len(_CONNECTORS) else ""
        parts.append(f"{connector}{statement}")
    return " ".join(parts)


def _attributed(sources: Sequence[Source], limit: int = 3) -> str:
    """A flowing paragraph that states each source's key sentence.

    Attribution is *not* embedded here -- citations are returned as data and
    rendered once after the answer. Statements are read as whole sentences
    (one to two), and a source whose text is only a fragment is skipped when a
    longer statement is available, so the paragraph does not degenerate into
    bare headings.
    """
    chosen = [source for source in sources if clean(source.text)]
    if not chosen:
        return ""
    usable = [source for source in chosen if len(clean(source.text).split()) >= _MIN_STATEMENT_WORDS]
    if usable:
        chosen = usable
    statements: list[str] = []
    for source in chosen[:limit]:
        statement = complete_sentences(source.text, 2)
        if not statement:
            continue
        statements.append(ensure_terminal(statement))
    return _compose_statements(statements)


def grounded_answer(question: str, sources: Sequence[Source]) -> str:
    """Render an answer body from retrieved passages, with no inline citations.

    This is what the assistant falls back to when no model is available (or its
    output is not grounded). The body reads as connected prose; the citations
    that support it are exposed separately (``GroundedAssistant.answer.citations``)
    and printed once at the end by the CLI, never mid-sentence.
    """
    topic = topic_phrase(question)
    present = [source for source in sources if clean(source.text)]
    if not present:
        return f"There is not enough evidence on file to answer this question about {topic}."
    lead = f"Here is what the available sources say about {topic}."
    body = _attributed(present)
    return "\n\n".join(block for block in (lead, body) if block)


def web_answer(question: str, sources: Sequence[Source], internal: Sequence[str] = ()) -> str:
    """Render a web-sourced answer from assessed search results.

    ``sources`` carry a title, the snippet, and a domain/URL label; ``internal``
    is the list of internal citations that the verifier compared the answer
    against. The body states the facts only; the web source is attributed through
    the citation list, which the caller lists after the answer, so the reader can
    still see it came from the web and not from the project's own knowledge.
    """
    topic = topic_phrase(question)
    present = [source for source in sources if clean(source.text)] or list(sources)
    if not present:
        return f"No web source answered the question about {topic}."
    lead = f"Searching externally for {topic}, the best-supported answer is:"
    primary = present[0]
    statement = ensure_terminal(complete_sentences(primary.text, 2) or primary.title)
    # No inline attribution: the domain/title travel as citations and are listed
    # once after the answer (see ``WebAssessment.summary`` and the CLI layers).
    blocks = [lead, statement]
    if len(present) > 1:
        titles = "; ".join(clean(source.title) for source in present[1:4])
        blocks.append(f"{len(present) - 1} further source(s) corroborate this: {titles}.")
    if internal:
        blocks.append("The internal corpus also touches on this: " + "; ".join(internal[:2]) + ".")
    return "\n\n".join(blocks)


# A word carrying an underscore or a dotted name is a corpus identifier
# (``differential_equations``, ``petroleum.system``), not prose; a sentence that
# contains one reads as a data dump, so it is dropped from a composed answer.
_IDENTIFIER_RE = re.compile(r"\w+_\w+|\w+\.\w+")

# Model continuations often join a new sentence onto the previous one without
# capitalising it ("...known as porosity. matrix is a concept..."). ``sentences``
# only splits before a capital, so the model answer splits on any terminal
# punctuation and lets ``complete_sentences`` drop the lowercase fragment.
_MODEL_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# A sentence built around an equation ("PI = 0", "mu/V") is a formula the model
# half-copied from the corpus, not prose, so it is dropped from an answer.
_FORMULA_RE = re.compile(r"[=≈≠≤≥∑√∫]|[\w)]\s*/\s*[\w(]")

#: How many sentences a model answer may run to. The first complete sentence is
#: the anchored, in-distribution one the model was steered into; later sentences
#: tend to drift onto a neighbouring topic, so one is the default.
_MODEL_ANSWER_SENTENCES = 1


def model_answer(question: str, text: str, limit: int = _MODEL_ANSWER_SENTENCES) -> str:
    """Render the trained model's own answer as clean prose.

    The in-project model is small, so its raw continuation is often broken: it
    begins mid-clause (``", and faults are fracture properties"``), trails off
    mid-word, or leaks the corpus's internal identifiers. This is the same fix
    the grounded answers already use -- a deterministic verbaliser -- applied to
    the model mouth: the continuation is reduced to the *whole* sentences it
    actually contains, joined into a paragraph with the project's shared voice.
    A sentence carrying a corpus identifier (``differential_equations``) reads as
    a data dump and is dropped, and the answer is capped at ``limit`` sentences
    so it states a point instead of rambling.

    Only the model's own words survive (``text`` is never supplemented, unlike
    :func:`grounded_answer`), so the output stays an honest sample of what the
    model produced. When the continuation holds no complete sentence, the
    function returns ``""`` and the caller can say so plainly.
    """
    present = clean(text)
    if not present or limit < 1:
        return ""
    statements: list[str] = []
    for sentence in _MODEL_SENTENCE_RE.split(present):
        whole = complete_sentences(sentence)
        if not whole or _IDENTIFIER_RE.search(whole) or _FORMULA_RE.search(whole):
            continue
        statements.append(ensure_terminal(whole))
        if len(statements) >= limit:
            break
    return _compose_statements(statements)


def is_coherent(text: str) -> bool:
    """Whether ``text`` is safe to show as prose, not raw model noise.

    A usable answer has at least one word and its first character is one a
    sentence can start with -- never a stray punctuation mark such as the comma
    a truncated continuation begins with. This is a cheap gate, not a quality
    score; it only keeps obviously broken output out of the answer.
    """
    value = clean(text)
    return bool(value) and value[0].isalnum()


# --- shared voice for the reasoning explainers ----------------------------


def causal_chain_sentence(
    cause: str,
    effect: str,
    nodes: Sequence[str],
    confidence: float,
    mechanisms: Sequence[str] = (),
) -> str:
    """One sentence describing a cause-to-effect path and its confidence."""
    path = " \u2192 ".join([cause, *nodes, effect]) if nodes else f"{cause} \u2192 {effect}"
    text = f"{cause.capitalize()} can influence {effect} along the path {path}, with an overall confidence of {confidence:.2f}."
    if mechanisms:
        text += f" The apparent mechanism is {'; '.join(mechanisms)}."
    return text


def causal_effect_sentence(cause: str, effect: str, effect_size: float) -> str:
    """One sentence describing an interventional effect size."""
    return f"Changing {cause} moved {effect} by {effect_size:+.2f} relative to the control."


def counterfactual_sentence(
    cause: str, effect: str, factual: float, counterfactual: float
) -> str:
    """One sentence contrasting an observed and a counterfactual confidence."""
    return (
        f"With {cause} present, {effect} held at a confidence of {factual:.2f}; "
        f"in its absence the simulated confidence falls to {counterfactual:.2f}."
    )


def reasoning_step_sentence(
    method: str, premises: Sequence[str], conclusion: str, confidence: float
) -> str:
    """One sentence describing an inference step, keeping the premise arrow."""
    chain = f"{', '.join(premises)} \u2192 {conclusion}" if premises else conclusion
    return f"By {method} reasoning, {chain} (confidence {confidence:.2f})."


def decision_sentences(
    selected: str | None,
    reason: str,
    candidates: Sequence[tuple[str, float, float, float]],
) -> tuple[str, ...]:
    """The chosen action plus each candidate's scores, as readable sentences."""
    head = (f"Selected: {selected}. {ensure_terminal(reason)}",) if selected is not None else (ensure_terminal(reason) or "No action was selected.",)
    ranked = tuple(
        f"{name}: utility {utility:.2f}, risk {risk:.2f}, overall score {score:.2f}."
        for name, utility, risk, score in candidates
    )
    return head + ranked


def evidence_summary_sentence(conclusion: str, supporting: int, contradicting: int) -> str:
    """The one-line summary shared by the explanation engine."""
    return (
        f"{ensure_terminal(conclusion)} This rests on {supporting} supporting and "
        f"{contradicting} contradicting evidence items."
    )


__all__ = [
    "Source",
    "causal_chain_sentence",
    "causal_effect_sentence",
    "clean",
    "complete_sentences",
    "counterfactual_sentence",
    "decision_sentences",
    "ensure_terminal",
    "evidence_summary_sentence",
    "first_sentences",
    "grounded_answer",
    "is_coherent",
    "model_answer",
    "reasoning_step_sentence",
    "sentences",
    "sources_block",
    "topic_phrase",
    "web_answer",
]
