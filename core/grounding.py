"""A stronger groundedness gate for model output.

The assistant never publishes model text it cannot trace to a source, but the gate
that decided this -- ``_is_grounded`` in :mod:`core.assistant` -- was deliberately
weak: it accepted an answer as soon as it shared **one** content term with the
evidence. So a fluent paragraph could ride one incidental word into a published
answer, a sentence unsupported by any passage was never noticed, and a model that
**reversed** a mechanism the graph states (saying *"production drives waterflood"*
when the graph stores ``waterflood → production``) was published as-is. This is Gap
5 of ``docs/REASONING_ASSESSMENT.md``.

This module is the one place the stronger gate lives. It makes three deterministic
checks and **fails closed** -- when in doubt the caller falls back to the
extractive answer, which is always traceable:

1. **Term coverage** -- enough of the answer's content terms must actually appear
   in the evidence (not just one).
2. **Sentence support** -- each content-bearing sentence of the answer must share a
   term with the evidence; a majority of unsupported sentences fails the gate, so
   an answer cannot smuggle in a paragraph the sources do not carry.
3. **Causal direction** -- if the answer states a causal link between two concepts
   in the direction the graph **contradicts** (the graph stores the reverse edge
   and not the forward one), the gate fails. This is the honesty rule the causal
   layer already applies to its own walks, now applied to the model's words.

Nothing here rewrites the model's text or asserts anything into the graph; it only
decides whether that text may be shown.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from core.ontology import Ontology
from core.rag import RetrievedPassage
from core.tokenize import content_terms, normalize_phrase

#: The share of an answer's content terms that must appear in the evidence.
#: Deliberately modest -- a grounded answer may use a paraphrase or a connective,
#: and a short answer legitimately shares few terms -- but it scales: a long answer
#: that shares only a word or two fails, which the original "one shared term" rule
#: could never catch.
DEFAULT_MIN_COVERAGE = 0.3

#: The share of the answer's content-bearing sentences that must share a term with
#: the evidence. Below this the gate fails, so a majority of unsupported sentences
#: (a fluent paragraph riding one incidental word) cannot be published.
DEFAULT_MIN_SENTENCE_SUPPORT = 0.5

#: Words that read a causal link *forward*: "A <marker> B" means A causes B.
_CAUSAL_MARKERS: tuple[str, ...] = (
    "causes", "cause", "drives", "drive", "controls", "control", "creates",
    "determines", "governs", "increases", "increasing", "reduces", "reducing",
    "prevents", "maintains", "produces", "enables", "affects", "improves",
    "protects", "leads to", "results in", "gives rise to", "brings about",
)

#: Words that read a causal link *backward*: "B <marker> A" means A causes B.
_REVERSE_MARKERS: tuple[str, ...] = (
    "caused by", "driven by", "controlled by", "created by", "determined by",
    "governed by", "produced by", "enabled by", "affected by", "improved by",
    "reduced by", "prevented by", "maintained by", "due to", "results from",
    "comes from", "because of", "arises from", "stems from",
)

_SENTENCE_RE = re.compile(r"[^.!?؟\n]+[.!?؟]?", flags=re.UNICODE)


@dataclass(frozen=True)
class GroundingCheck:
    """Whether model text may be published, with every reason it may not."""

    grounded: bool
    """True only when every check passed."""

    coverage: float
    """The share of the answer's content terms that appear in the evidence."""

    unsupported: tuple[str, ...] = field(default_factory=tuple)
    """The answer's sentences that share no term with the evidence."""

    reversals: tuple[str, ...] = field(default_factory=tuple)
    """Causal links the answer states in the direction the graph contradicts."""

    reasons: tuple[str, ...] = field(default_factory=tuple)
    """One readable line per failed check, for the record."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("Coverage must be between 0 and 1.")

    def describe(self) -> str:
        """Why the text passed or failed, in one line."""
        if self.grounded:
            return f"grounded (coverage {self.coverage:.2f})"
        if not self.reasons:
            return f"ungrounded (coverage {self.coverage:.2f})"
        return f"ungrounded (coverage {self.coverage:.2f}): " + "; ".join(self.reasons)


def _sentences(text: str) -> list[str]:
    """Split text into sentences deterministically, dropping empty pieces."""
    return [piece.strip() for piece in _SENTENCE_RE.findall(text) if piece.strip()]


def _label_key(label: str) -> str:
    """A concept label in the same space as a sentence's tokens.

    An ontology identifier is written ``reservoir_pressure`` while a sentence says
    "reservoir pressure", so the underscore is turned into a space before
    normalising -- otherwise a label could never be found in natural text.
    """
    return normalize_phrase(label.replace("_", " "))


def _causal_edges(ontology: Ontology | None) -> tuple[set[tuple[str, str]], set[str]]:
    """The graph's directed causal edges as (source, target) label pairs.

    Returns the edges and the set of labels that take part in one, so the reversal
    check only ever reasons about concepts the graph actually connects causally.
    """
    if ontology is None:
        return set(), set()
    from core.causal_paths import CAUSAL_RELATION_TYPES

    labels = {cid: concept.label for cid, concept in ontology.concepts.items()}
    edges: set[tuple[str, str]] = set()
    endpoints: set[str] = set()
    for relation in ontology.relations.values():
        if relation.relation_type not in CAUSAL_RELATION_TYPES:
            continue
        source = labels.get(relation.source_id, relation.source_id)
        target = labels.get(relation.target_id, relation.target_id)
        if source == target:
            continue
        edges.add((_label_key(source), _label_key(target)))
        endpoints.add(_label_key(source))
        endpoints.add(_label_key(target))
    return edges, endpoints


def _find_marker(sentence_norm: str, markers: Iterable[str]) -> tuple[int, int] | None:
    """The span of the earliest causal marker in a normalised sentence, if any.

    Markers are matched on word boundaries, so ``controls`` never fires inside
    another word, and the earliest match wins so "A drives B because of C" reads
    forward on ``drives`` rather than reverse on ``because of``.
    """
    best: tuple[int, int] | None = None
    for marker in markers:
        match = re.search(rf"\b{re.escape(marker)}\b", sentence_norm)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), match.end())
    return best


def _endpoint_spans(sentence_norm: str, endpoints: Iterable[str]) -> list[tuple[int, int, str]]:
    """Whole-phrase occurrences of the graph's causal concepts in a sentence.

    Matched on word boundaries, and a match nested inside a longer one is dropped
    -- so ``pressure`` inside "reservoir pressure" is not counted as a second,
    separate concept. The result is ordered by position.
    """
    spans: list[tuple[int, int, str]] = []
    for label in endpoints:
        if not label:
            continue
        for match in re.finditer(rf"\b{re.escape(label)}\b", sentence_norm):
            spans.append((match.start(), match.end(), label))
    spans.sort()
    kept: list[tuple[int, int, str]] = []
    for start, end, label in spans:
        if any(start >= s and end <= e and (s, e) != (start, end) for s, e, _ in spans):
            continue
        kept.append((start, end, label))
    return kept


def _causal_reversals(text: str, ontology: Ontology | None) -> tuple[str, ...]:
    """Causal links the text states in the direction the graph contradicts.

    For each sentence with a causal marker, the concepts the graph causally
    connects are located before and after the marker. A forward marker reads
    "before causes after"; a reverse marker ("caused by") reads "after causes
    before". When the graph stores only the **opposite** edge, the text has
    reversed a mechanism the graph states, and that is reported. The check is
    conservative: it only fires when both endpoints are known causal concepts and
    the graph holds the reverse edge but not the forward one.
    """
    edges, endpoints = _causal_edges(ontology)
    if not edges:
        return ()
    found: list[str] = []
    for sentence in _sentences(text):
        sentence_norm = normalize_phrase(sentence)
        forward = _find_marker(sentence_norm, _CAUSAL_MARKERS)
        reverse = _find_marker(sentence_norm, _REVERSE_MARKERS)
        if forward is None and reverse is None:
            continue
        if reverse is not None and (forward is None or reverse[0] < forward[0]):
            span, flipped = reverse, True
        else:
            span, flipped = forward, False  # type: ignore[assignment]
        assert span is not None
        spans = _endpoint_spans(sentence_norm, endpoints)
        before = [label for start, end, label in spans if end <= span[0]]
        after = [label for start, end, label in spans if start >= span[1]]
        for left in before:
            for right in after:
                cause, effect = (right, left) if flipped else (left, right)
                if cause == effect:
                    continue
                if (effect, cause) in edges and (cause, effect) not in edges:
                    found.append(
                        f"the graph stores {effect} → {cause}, but the answer says "
                        f"{cause} → {effect}"
                    )
    # De-duplicate while keeping order, so one reversal is reported once.
    return tuple(dict.fromkeys(found))


def check_grounding(
    text: str,
    passages: Iterable[RetrievedPassage],
    *,
    ontology: Ontology | None = None,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    min_sentence_support: float = DEFAULT_MIN_SENTENCE_SUPPORT,
) -> GroundingCheck:
    """Decide whether model ``text`` may be published against its evidence.

    Fails closed: no evidence, too little term overlap, a majority of unsupported
    sentences, or a reversed causal mechanism each make ``grounded`` false, so the
    caller falls back to the extractive answer. The returned
    :class:`GroundingCheck` names every reason.
    """
    items = list(passages)
    answer_terms = content_terms(text)
    evidence_terms: set[str] = set()
    for passage in items:
        evidence_terms |= content_terms(f"{passage.title} {passage.text}")

    reasons: list[str] = []

    if not items:
        return GroundingCheck(False, 0.0, reasons=("no evidence was retrieved",))
    if not answer_terms:
        return GroundingCheck(
            False, 0.0, reasons=("the answer shares no content term with the evidence",)
        )

    shared = answer_terms & evidence_terms
    coverage = len(shared) / len(answer_terms)
    if coverage < min_coverage:
        reasons.append(
            f"only {coverage:.2f} of the answer's terms appear in the evidence"
            f" (needs {min_coverage:.2f})"
        )

    # Per-sentence support: a sentence with content terms but no shared term is
    # unsupported, and a majority of them fails the gate.
    content_sentences = [s for s in _sentences(text) if content_terms(s)]
    unsupported: list[str] = []
    for sentence in content_sentences:
        if not (content_terms(sentence) & evidence_terms):
            unsupported.append(sentence)
    if content_sentences:
        support = 1.0 - len(unsupported) / len(content_sentences)
        if support < min_sentence_support:
            reasons.append(
                f"{len(unsupported)} of {len(content_sentences)} sentences share no "
                "term with the evidence"
            )

    reversals = _causal_reversals(text, ontology)
    if reversals:
        reasons.append("a causal mechanism is reversed: " + "; ".join(reversals))

    return GroundingCheck(
        grounded=not reasons,
        coverage=round(coverage, 6),
        unsupported=tuple(unsupported),
        reversals=reversals,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class GroundingJudgement:
    """One held-out expectation about whether model text may be published.

    ``evidence`` is the passage text the answer must be grounded in. When
    ``ontology`` is true the project's seeded ontology is used, so a judgement can
    also check the causal-direction rule. ``expected`` is whether the gate must
    accept the text, so a **negative control** (a fluent answer with no support, a
    paragraph riding one incidental word, a reversed mechanism) must be rejected.
    """

    name: str
    text: str
    evidence: str
    expected: bool
    ontology: bool = False
    note: str = ""


def _judgement_passages(judgement: GroundingJudgement) -> list[RetrievedPassage]:
    return [
        RetrievedPassage(
            doc_id=f"gate:{judgement.name}",
            title=judgement.name,
            text=judgement.evidence,
            source="gate",
            score=0.9,
            signal="lexical",
        )
    ]


#: The held-out grounding catalogue -- hand-written, with **negative controls**, so
#: a gate that publishes unsupported or reversed text fails as surely as one that
#: refuses a genuinely grounded answer.
GROUNDING_JUDGEMENTS: tuple[GroundingJudgement, ...] = (
    GroundingJudgement(
        "short grounded answer",
        "Permeability is a property of the formation.",
        "Permeability is a property of the formation that measures flow.",
        True,
        note="a short answer that shares the evidence's own terms",
    ),
    GroundingJudgement(
        "long grounded answer",
        "Reservoir pressure drives production, and the well produces oil and gas.",
        "Reservoir pressure drives production of oil and gas from the well.",
        True,
        note="a two-clause answer, every clause supported",
    ),
    GroundingJudgement(
        "one incidental word",
        "The drilling rig is expensive, the market is volatile, and the weather "
        "delays every operation, yet management approves the budget anyway.",
        "The drilling rig drills the hole.",
        False,
        note="a fluent paragraph riding one shared word (negative control)",
    ),
    GroundingJudgement(
        "unsupported sentences",
        "The drilling rig drills the hole. Porosity is a measure of void space. "
        "The market fell sharply last quarter.",
        "The drilling rig drills the hole into the formation.",
        False,
        note="a supported sentence outnumbered by unsupported ones (negative control)",
    ),
    GroundingJudgement(
        "no shared term",
        "Prices fell sharply last quarter.",
        "The drilling rig drills the hole.",
        False,
        note="an answer with nothing in common with the evidence (negative control)",
    ),
    GroundingJudgement(
        "no evidence",
        "Permeability is a property of the formation.",
        "",
        False,
        note="there is nothing to be grounded in (negative control)",
    ),
    GroundingJudgement(
        "forward mechanism",
        "Reservoir pressure drives production.",
        "Reservoir pressure drives production in the field.",
        True,
        ontology=True,
        note="the answer states the graph's own direction",
    ),
    GroundingJudgement(
        "reversed mechanism",
        "Production drives reservoir pressure.",
        "Reservoir pressure drives production in the field.",
        False,
        ontology=True,
        note="the answer reverses a mechanism the graph states (negative control)",
    ),
)


def grounding_accuracy(
    ontology: Ontology | None = None,
    judgements: tuple[GroundingJudgement, ...] = GROUNDING_JUDGEMENTS,
) -> float:
    """The share of held-out judgements the gate gets right.

    A judgement counts only when the gate's verdict matches ``expected`` exactly --
    so publishing unsupported or reversed text fails as surely as refusing a
    grounded answer. Returns ``0.0`` for an empty catalogue. ``ontology`` is
    injectable for a test; with none it builds the project's seeded ontology, which
    the causal-direction judgements need.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    if ontology is None:
        from core.cognitive_controller import CognitiveController

        ontology = CognitiveController().ontology
    correct = 0
    for judgement in catalogue:
        check = check_grounding(
            judgement.text,
            _judgement_passages(judgement),
            ontology=ontology if judgement.ontology else None,
        )
        if check.grounded == judgement.expected:
            correct += 1
    return correct / len(catalogue)


__all__ = [
    "DEFAULT_MIN_COVERAGE",
    "DEFAULT_MIN_SENTENCE_SUPPORT",
    "GROUNDING_JUDGEMENTS",
    "GroundingCheck",
    "GroundingJudgement",
    "check_grounding",
    "grounding_accuracy",
]
