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
# A leading list marker ("- ", "1) ", "2. "). The marker must be followed by
# whitespace: a bare "1." is the integer half of a decimal ("1.5 times the
# gradient"), which the byte model emits, and stripping it would corrupt the
# number. So "1. Do X" loses its marker but "1.5 times" keeps its digit.
_LEAD_RE = re.compile(r"^\s*(?:[-*\u2022>]+\s*|\d+[.)](?=\s|$))")
_TERMINAL = ".!?"

# Raw model output is looser than authored prose: a continuation often runs on
# past a sentence end starting the next one in lower case, so the model-specific
# split keys on terminal punctuation plus whitespace alone (a decimal point has
# no space after it, so "0.052 psi/ft" is still never split).
_MODEL_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
# Runs the byte-level model repeats, such as the ")))))" or "..." at a stopping
# point; a repeated mark is collapsed to one. When that mark is not itself a
# sentence end it becomes a period, so the run reads as a sentence boundary.
# A hyphen run *between* word characters is left alone: "10--20" and "5CT--spec"
# are a range and a compound name, and collapsing them to a period would corrupt
# the model's own number or identifier.
_REPEAT_MARK_RE = re.compile(r"([^\w\s])\1+")
# A word or short phrase (up to four words) the model loops on: "well well",
# "the same the same", "the production forecast the production forecast". Only a
# phrase repeated back-to-back with nothing between it counts, so a word reused
# in a real sentence is never touched.
_REPEATED_PHRASE_RE = re.compile(
    r"(?<!\w)(\w+(?:\s+\w+){0,3})(?:\s+\1\b)+", flags=re.IGNORECASE
)
# Leading/trailing decoration the model leaves around a sentence.
_EDGE_CHARS = " \t\"'\u201c\u201d\u2018\u2019()[]{}*_-"

# A sentence the byte model leaves ending on one of these is an unfinished clause
# ("... and", "... the rock and"), not a statement, so the composer drops it when
# a clean sentence exists -- the same honesty rule the grounded composer applies
# to bare headings.
_DANGLING_WORDS = frozenset(
    {
        "and", "or", "but", "nor", "so", "the", "a", "an", "of", "to", "with",
        "for", "at", "by", "from", "in", "on", "into", "as", "than", "then",
        "that", "which", "while", "when", "where", "is", "are", "was", "were",
        "be", "been", "its", "their", "this", "these", "those",
    }
)
# Words that open a section label the model repeats ("Step 2 of ...", "The
# procedure for ..."). Such a sentence names a part of a document rather than
# answering, so it is ranked last -- kept only when nothing better survives.
_META_LEADS = ("step ", "the procedure for ", "the formula for ", "table ")
# A sentence with fewer real words than this is a fragment, not a statement.
_MIN_MODEL_WORDS = 2
# At most this many model sentences are kept, so the mouth reads as an answer
# rather than a paragraph of loosely joined continuations.
_MAX_MODEL_SENTENCES = 3
# One token, its letters only; used to compare a sentence with the question.
_MODEL_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)

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


def _tidy_marks(text: str) -> str:
    """Collapse punctuation runs and repeated words/phrases the byte model repeats.

    A run of the *same* mark is collapsed; when the mark is not itself a
    sentence-end it becomes a period, so a stopping run such as ``")))))"``
    turns into a boundary instead of an unreadable cluster. A hyphen run between
    word characters is a range or compound name, not decoration, so it is left
    intact. A repeated word *or short phrase* ("the same the same", "the
    production forecast the production forecast") collapses to one, which is the
    model looping rather than a statement said twice.
    """
    def _collapse(match: re.Match[str]) -> str:
        mark = match.group(1)
        # A hyphen run *between* word characters ("10--20", "5CT--spec") is a
        # range or compound name, not a repeated mark, so it is kept as written.
        if mark == "-":
            before = text[match.start() - 1] if match.start() else ""
            after = text[match.end()] if match.end() < len(text) else ""
            if before.isalnum() and after.isalnum():
                return match.group(0)
        return mark if mark in _TERMINAL else "."

    value = _REPEAT_MARK_RE.sub(_collapse, text)
    value = _REPEATED_PHRASE_RE.sub(r"\1", value)
    return clean(value)


def _question_vocabulary(question: str) -> frozenset[str]:
    """The content words the question names, lower-cased (empty when none).

    Used only to *centre* the model's answer on what was asked: a sentence that
    mentions a question word is preferred as the opening, and a sentence that
    repeats an earlier sentence is dropped. No word is ever added to the answer.
    """
    seen: list[str] = []
    for token in _MODEL_TOKEN_RE.findall(question):
        folded = token.casefold()
        if folded not in _TOPIC_STOPWORDS and len(folded) > 1:
            seen.append(folded)
    return frozenset(seen)


def _sentence_quality(sentence: str) -> tuple[int, int]:
    """Rank a whole model sentence: content first, section labels last.

    Returns ``(is_dangling, is_meta)`` so the caller can keep every clean
    sentence while preferring real statements over a repeated ``"Step 2 of ..."``
    label. Nothing is dropped by position alone.
    """
    lowered = sentence.casefold()
    words = _MODEL_TOKEN_RE.findall(lowered)
    dangling = 0
    if words and words[-1] in _DANGLING_WORDS:
        dangling = 1
    meta = 1 if lowered.lstrip().startswith(_META_LEADS) else 0
    return dangling, meta


#: A conjunction a continuation may start a repeated sentence with ("The pump.
#: And the pump."). Stripped before comparing sentences so the same statement is
#: recognised however it is joined.
_DEDUP_LEADS = frozenset({"and", "but", "plus", "also", "then"})


def _repeat_key(sentence: str) -> str:
    """A content key for a sentence, ignoring decoration and a leading conjunction.

    Two sentences that only differ by edge punctuation or an opening "and"/"then"
    are the same statement, so they collapse to one; the model's words are not
    changed, the repeat is simply not shown twice.
    """
    words = _MODEL_TOKEN_RE.findall(sentence.casefold())
    while words and words[0] in _DEDUP_LEADS:
        words.pop(0)
    return " ".join(words)


def _model_sentences(raw: str) -> list[str]:
    """Split tidied model output into whole, clean sentences.

    A sentence survives when it starts at a boundary (upper case or a digit) and
    ends with terminal punctuation; the leading fragment and the trailing
    mid-word fragment the byte model leaves are dropped here.
    """
    cleaned = _tidy_marks(raw)
    if not cleaned:
        return []
    segments = [segment.strip(_EDGE_CHARS) for segment in _MODEL_SENTENCE_RE.split(cleaned)]
    segments = [segment for segment in segments if segment]
    if segments and segments[-1][-1] not in _TERMINAL:
        segments.pop()
    # A leading fragment is dropped only when a real sentence follows it, so a
    # lone lower-case sentence is still returned as the model's whole answer.
    if len(segments) > 1 and not (segments[0][0].isupper() or segments[0][0].isdigit()):
        segments.pop(0)
    return segments


def compose_model_answer(raw: str, *, question: str = "") -> str:
    """Tidy raw model output into readable prose without rewriting its words.

    The in-project model is a small byte-level continuation model, so its raw
    output is messy in ways the authored prose (:func:`complete_sentences`) never
    is: it begins mid-sentence, runs on past a lower-case sentence boundary, ends
    mid-word, repeats punctuation such as ``")))))"``, and joins clauses with no
    verb ("... the well and reservoi").

    This is the model-output half of the composer. It *does not paraphrase and
    does not add facts* -- it only tidies the container so the model's own words
    read as sentences, the way :func:`grounded_answer` already tidies retrieved
    passages:

    * keep the whole sentences the model actually finished, in order;
    * drop a trailing sentence that ends on a dangling conjunction/article
      ("... the rock and") when a clean sentence exists, so the answer never
      stops mid-clause;
    * drop a sentence that only repeats one already kept;
    * prefer, as the opening, a sentence that names a word from the question, so
      the answer is centred on what was asked, and demote a bare section label
      ("Step 2 of ...") below the real statements;
    * collapse repeated punctuation/doubled words, capitalise, and punctuate.

    When ``question`` is given the model's answer is centred on it; ``raw`` alone
    still composes (the leading fragment and unfinished tail are dropped as
    before). When no whole sentence survives it falls back to the tidied raw text
    so the mouth never goes silent -- unless that tidied text carries no word at
    all (an empty continuation, or a stopping run such as ``")))))"``), in which
    case it returns ``""`` so the caller can say the model produced nothing rather
    than render a wordless message. The result stays the model's own prose, which
    is why the caller still labels it as ungrounded.
    """
    cleaned = _tidy_marks(raw)
    if not cleaned:
        return ""
    candidates = [sentence for sentence in _model_sentences(raw) if sentence]
    if candidates:
        vocabulary = _question_vocabulary(question) if question.strip() else frozenset()

        kept: list[str] = []
        seen: set[str] = set()
        for sentence in candidates:
            key = _repeat_key(sentence)
            if key in seen:
                continue
            seen.add(key)
            kept.append(sentence)
        kept = [s for s in kept if len(_MODEL_TOKEN_RE.findall(s)) >= _MIN_MODEL_WORDS]
        if not kept:
            kept = candidates
        # Prefer real statements over unfinished clauses and bare section labels,
        # dropping them only when something cleaner survives -- so the answer
        # never ends mid-clause when the model did finish a sentence.
        clean = [s for s in kept if not _sentence_quality(s)[0]]
        if clean:
            kept = clean
        non_meta = [s for s in kept if not _sentence_quality(s)[1]]
        if non_meta:
            kept = non_meta
        # Centre the answer on the question: open with the sentence that names the
        # most of what was asked, when it is not already first.
        if vocabulary and len(kept) > 1:
            overlap = [
                len(vocabulary.intersection(_MODEL_TOKEN_RE.findall(s.casefold())))
                for s in kept
            ]
            if max(overlap) > overlap[0]:
                best = max(range(len(kept)), key=lambda i: overlap[i])
                kept.insert(0, kept.pop(best))
        kept = kept[:_MAX_MODEL_SENTENCES]
        body = " ".join(kept)
    else:
        body = cleaned
    body = ensure_terminal(body)
    # A stopping run such as ")))))" tidies down to bare punctuation; that is not
    # an answer the model gave, so return nothing rather than a lone period (which
    # would render, in the GUI, as a message with no text and only its Copy
    # button). A real answer always carries at least one word.
    if not _MODEL_TOKEN_RE.search(body):
        return ""
    return body


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
    if not statements:
        return ""
    if len(statements) == 1:
        return statements[0]
    parts: list[str] = []
    for index, statement in enumerate(statements):
        connector = _CONNECTORS[index] if index < len(_CONNECTORS) else ""
        parts.append(f"{connector}{statement}")
    return " ".join(parts)


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
    "compose_model_answer",
    "counterfactual_sentence",
    "decision_sentences",
    "ensure_terminal",
    "evidence_summary_sentence",
    "first_sentences",
    "grounded_answer",
    "reasoning_step_sentence",
    "sentences",
    "sources_block",
    "topic_phrase",
    "web_answer",
]
