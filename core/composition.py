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
    """A compact noun phrase naming what the question is about."""
    from core.rag import _query_terms

    terms = sorted(_query_terms(question))
    if not terms:
        value = clean(question).rstrip("?.!")
        return value or "this question"
    return " ".join(terms)


def _attributed(sources: Sequence[Source], limit: int = 3) -> str:
    """A flowing paragraph that states each source's key sentence and cites it."""
    chosen = [source for source in sources if clean(source.text)][:limit]
    if not chosen:
        return ""
    parts: list[str] = []
    for index, source in enumerate(chosen):
        statement = ensure_terminal(first_sentences(source.text, 1) or source.title)
        label = clean(source.label)
        connector = _CONNECTORS[index] if len(chosen) > 1 and index < len(_CONNECTORS) else ""
        citation = f" (source: {label})" if label else ""
        parts.append(f"{connector}{statement}{citation}")
    return " ".join(parts)


def grounded_answer(question: str, sources: Sequence[Source]) -> str:
    """Render a citation-only answer from retrieved passages.

    This is what the assistant falls back to when no model is available (or its
    output is not grounded). The answer keeps the provenance that makes the
    project evidence-aware while reading as connected prose rather than a dump.
    """
    topic = topic_phrase(question)
    present = [source for source in sources if clean(source.text)]
    if not present:
        return f"There is not enough evidence on file to answer this question about {topic}."
    lead = f"Here is what the available sources say about {topic}."
    body = _attributed(present)
    citation_line = "Sources: " + "; ".join(clean(source.label) for source in present if clean(source.label))
    blocks = [lead, body]
    if citation_line != "Sources: ":
        blocks.append(citation_line)
    return "\n\n".join(block for block in blocks if block)


def web_answer(question: str, sources: Sequence[Source], internal: Sequence[str] = ()) -> str:
    """Render a web-sourced answer from assessed search results.

    ``sources`` carry a title, the snippet, and a domain/URL label; ``internal``
    is the list of internal citations that the verifier compared the answer
    against. Attribution stays explicit so the reader can see it came from the
    web, not from the project's own knowledge.
    """
    topic = topic_phrase(question)
    present = [source for source in sources if clean(source.text)] or list(sources)
    if not present:
        return f"No web source answered the question about {topic}."
    lead = f"Searching externally for {topic}, the best-supported answer is:"
    primary = present[0]
    statement = ensure_terminal(first_sentences(primary.text, 2) or primary.title)
    label = clean(primary.label)
    attribution = f" (source: {primary.title}{', ' + label if label else ''})"
    blocks = [lead, f"{statement}{attribution}"]
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
    "counterfactual_sentence",
    "decision_sentences",
    "ensure_terminal",
    "evidence_summary_sentence",
    "first_sentences",
    "grounded_answer",
    "reasoning_step_sentence",
    "sentences",
    "topic_phrase",
    "web_answer",
]
