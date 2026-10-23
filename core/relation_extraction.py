"""Extract candidate typed relations from a sentence, deterministically.

The project already learns *weak, untyped* associations from text: when two
known concepts appear together, :class:`core.ontology_learner.OntologyLearner`
records an ``associated_with`` link. That is useful but blunt -- "porosity
increases permeability" and "porosity and permeability are measured by logs"
both collapse to the same association, so the direction and the *kind* of the
relation are thrown away.

This module reads the sentence's own grammar instead. A small, transparent set
of **patterns** maps a verb phrase between two known concepts onto one of the
ontology's real relation types (``increases``, ``reduces``, ``measures``,
``contains``, ``is_a``, ``causes``, ``depends_on``, ``affects``), in either
direction -- so "A increases B" and "B is increased by A" yield the same
relation with the endpoints in the right order. It is a *pattern matcher, not a
parser*: it claims nothing it did not match, and every extraction carries the
pattern that produced it and the sentence it came from, so a reviewer can check
it.

Honesty is the whole point:

* An extraction is a **candidate**, never an assertion. It is handed to
  :class:`core.ontology_learner.OntologyLearner` (through
  :class:`RelationExtractor.apply`) as a *typed* learned relation whose
  confidence is bounded by the learner's own ceiling, so an extracted relation
  can never override the curated seeds.
* Both endpoints must be concepts the ontology already knows, and each side must
  resolve to *exactly one* concept -- an unknown noun, or a side naming two
  concepts, yields nothing rather than a guess.
* The object's head is the noun the verb actually governs, and it must end its
  phrase: a concept inside a prepositional phrase ("a blowout in the reservoir")
  or used as a modifier ("well" in "well control") is not the object.
* The relation type must be one the ontology actually uses. An unmapped verb
  yields nothing.
* A sentence is read clause by clause, so a compound sentence yields one relation
  per clause and a relative clause is never glued onto the object before it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

from core.concept import Concept
from core.ontology import Ontology
from core.ontology_learner import OntologyLearner, OntologyLearning
from core.tokenize import content_terms, normalize_token

# --- the pattern table ----------------------------------------------------
#
# Each entry is (compiled regex, relation type, passive). The regex has two
# capture groups -- the subject phrase and the object phrase -- with the verb
# phrase between them matched literally; ``passive`` is true for a pattern whose
# voice swaps the endpoints ("X is measured by Y" means Y measures X). Patterns
# are tried in order and the *first* match wins, so the most specific verb
# ("is a") is listed before the general ones ("affects"). The regexes are
# anchored to a whole clause, not the whole sentence, so a clause inside a longer
# sentence is still read.
#
# Every relation type below is one the seeded ontology actually uses (see
# ``core.domain_knowledge`` / ``core.petroleum_corpus``), so an extracted
# relation joins the same vocabulary as the curated ones.

_PATTERNS: tuple[tuple[re.Pattern[str], str, bool], ...] = tuple(
    (re.compile(pattern, flags=re.IGNORECASE), relation, passive)
    for pattern, relation, passive in (
        # Definition / taxonomy -- checked first, it is the most explicit.
        (r"^(.+?)\s+(?:is|are)\s+(?:a|an|the)?\s*(?:kind|type|form|class)\s+of\s+(.+)$", "is_a", False),
        (r"^(.+?)\s+(?:is|are)\s+(?:a|an)\s+(.+)$", "is_a", False),
        # Measurement: "X measures Y"; passive "X is measured by Y" means Y
        # measures X, so the endpoints are swapped for a passive pattern.
        (r"^(.+?)\s+(?:measures|estimates|quantifies|determines|indicates|reveals|characteri[sz]es)\s+(.+)$", "measures", False),
        (r"^(.+?)\s+(?:is|are)\s+(?:measured|estimated|quantified|determined|indicated|revealed|characteri[sz]ed)\s+by\s+(.+)$", "measures", True),
        # Causation.
        (r"^(.+?)\s+(?:causes|cause|triggers|induces|produces|generates|creates|drives|initiates)\s+(.+)$", "causes", False),
        (r"^(.+?)\s+(?:is|are)\s+(?:caused|triggered|induced|produced|generated|created|driven|initiated)\s+by\s+(.+)$", "causes", True),
        # Increase / decrease, in both directions.
        (r"^(.+?)\s+(?:increases|improves|raises|boosts|enhances|maximizes|maximises)\s+(.+)$", "increases", False),
        (r"^(.+?)\s+(?:is|are)\s+(?:increased|improved|raised|boosted|enhanced|maximi[sz]ed)\s+by\s+(.+)$", "increases", True),
        (r"^(.+?)\s+(?:reduces|decreases|lowers|limits|prevents|mitigates|suppresses|inhibits)\s+(.+)$", "reduces", False),
        (r"^(.+?)\s+(?:is|are)\s+(?:reduced|decreased|lowered|limited|prevented|mitigated|suppressed|inhibited)\s+by\s+(.+)$", "reduces", True),
        # Composition / containment.
        (r"^(.+?)\s+(?:contains|includes|has|holds|hosts|stores|comprises)\s+(.+)$", "contains", False),
        (r"^(.+?)\s+(?:is|are)\s+(?:contained|included|held|hosted|stored)\s+in\s+(.+)$", "contains", True),
        # Dependency / requirement.
        (r"^(.+?)\s+(?:depends on|requires|needs|relies on)\s+(.+)$", "depends_on", False),
        # Influence -- the most general, so it is checked late.
        (r"^(.+?)\s+(?:affects|influences|controls|governs|determines)\s+(.+)$", "affects", False),
    )
)

#: The relation types this extractor can produce, for a caller that wants to
#: report its coverage rather than guess.
EXTRACTABLE_RELATIONS: tuple[str, ...] = tuple(
    dict.fromkeys(relation for _pattern, relation, _passive in _PATTERNS)
)

#: Prepositions that end an object's head noun phrase. A known concept after one
#: of these is *not* the verb's object -- it is a location, an instrument, or a
#: condition -- so it must not become the target of the relation.
_PREPOSITIONS = frozenset(
    {
        "in", "on", "at", "by", "with", "from", "into", "onto", "within",
        "during", "after", "before", "under", "over", "through", "across",
        "for", "of", "to", "as", "about", "against", "between", "among",
    }
)

#: How sure the extractor is of a pattern match before the learner's own
#: ceiling is applied. A definition ("X is a Y") is more reliable than a bare
#: influence claim, so the confidence is per-relation rather than one number.
BASE_CONFIDENCE: dict[str, float] = {
    "is_a": 0.55,
    "measures": 0.5,
    "contains": 0.5,
    "causes": 0.45,
    "increases": 0.45,
    "reduces": 0.45,
    "depends_on": 0.4,
    "affects": 0.35,
}
DEFAULT_CONFIDENCE = 0.35


class ChunkSource(Protocol):
    """The one method :func:`extract_from_store` needs from a knowledge store.

    A structural type, not an import: the extractor reads any object that yields
    chunks, so it does not depend on :mod:`core.knowledge_base` and a test can
    pass a plain list-backed double.
    """

    def iter_chunks(self) -> Iterator[object]: ...


@dataclass(frozen=True)
class ExtractedRelation:
    """One candidate relation read from a sentence, with its provenance."""

    source: str
    target: str
    relation_type: str
    confidence: float
    pattern: str
    """The verb phrase that produced the extraction, so a reviewer can check it."""
    sentence: str

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.target.strip():
            raise ValueError("An extracted relation needs two non-empty endpoints.")
        if not self.relation_type.strip():
            raise ValueError("An extracted relation needs a relation type.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("An extracted relation confidence must be in [0, 1].")

    def describe(self) -> str:
        return f"{self.source} --{self.relation_type}--> {self.target}"


@dataclass(frozen=True)
class RelationExtraction:
    """What one text yielded: the candidates, and why the rest were skipped."""

    relations: tuple[ExtractedRelation, ...]
    sentences: int
    unmatched: int
    """Sentences with two or more known concepts and no pattern that fired."""
    single_concept: int
    """Sentences with fewer than two known concepts -- nothing to relate."""

    @property
    def extracted(self) -> int:
        return len(self.relations)

    def summary(self) -> str:
        return (
            f"{self.extracted} relation(s) from {self.sentences} sentence(s) "
            f"({self.single_concept} with <2 known concepts, "
            f"{self.unmatched} with 2+ but no matching pattern)"
        )


def _sentences(text: str) -> list[str]:
    """Split text into sentences on terminal punctuation, keeping it simple.

    A period inside a decimal or an abbreviation is not a sentence end, so the
    split only breaks on ``. ! ?`` followed by whitespace or end of text. This is
    deliberately not a full sentence tokenizer: the extractor reads clauses, and
    an over-split sentence simply yields fewer candidates, never a wrong one.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


#: Clause separators. A sentence is read clause by clause so a compound sentence
#: ("A increases B and B controls C") yields both relations rather than one
#: mangled one. A coordinating conjunction, a relative pronoun, or punctuation
#: opens a new clause -- a relative clause ("permeability, which controls flow")
#: must not be glued onto the object of the clause before it, or its own subject
#: would be mistaken for the first clause's (the bug that read "porosity
#: increases permeability, which controls flow" as "porosity increases flow"). A
#: clause with no pattern simply contributes nothing.
_CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:;|,\s*(?:and|but|while|whereas|which|who|whose|where|not)\b"
    r"|\b(?:and|but|while|whereas|which|who|whose)\b)\s*",
    flags=re.IGNORECASE,
)


def _clauses(sentence: str) -> list[str]:
    """The clauses of a sentence, each still a candidate for one relation."""
    parts = [part.strip(" .;,") for part in _CLAUSE_SPLIT_RE.split(sentence)]
    return [part for part in parts if part]


class RelationExtractor:
    """Read typed candidate relations out of text, using the ontology's own words.

    The extractor needs the ontology only to know *which* noun phrases are
    concepts and to hand a matched relation back to the learner. It never adds a
    concept, so an unknown word is invisible to it by construction.
    """

    def __init__(self, ontology: Ontology, learner: OntologyLearner | None = None) -> None:
        self.ontology = ontology
        # The learner is optional so the extractor can be used purely to *read*
        # a text (a report, a preview) without changing the graph.
        self.learner = learner

    # -- reading ----------------------------------------------------------

    def concepts_in(self, sentence: str) -> list[tuple[int, int, Concept]]:
        """Known concepts in ``sentence``, as ``(start, end, concept)`` spans.

        Matching is on whole words (a concept term must appear as a run of whole
        tokens), so ``porosity`` does not match inside ``microporosity`` and
        invent a relation. Where a compound term and a bare one overlap
        (``water saturation`` vs ``saturation``), the longer, leftmost span wins.
        """
        found: list[tuple[int, int, Concept]] = []
        for concept in self.ontology.concepts.values():
            for term in concept.terms:
                term_tokens = [normalize_token(part) for part in re.findall(r"\w+", term)]
                if not term_tokens:
                    continue
                pattern = re.compile(
                    r"\b" + r"\W+".join(re.escape(part) for part in term_tokens) + r"\b",
                    flags=re.IGNORECASE,
                )
                for match in pattern.finditer(sentence):
                    found.append((match.start(), match.end(), concept))
        # Longest span first, then leftmost; drop any span overlapping one kept,
        # and keep only one span per concept.
        found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        kept: list[tuple[int, int, Concept]] = []
        for start, end, concept in found:
            if any(start < other_end and end > other_start for other_start, other_end, _ in kept):
                continue
            if any(kept_concept.id == concept.id for _s, _e, kept_concept in kept):
                continue
            kept.append((start, end, concept))
        kept.sort(key=lambda item: item[0])
        return kept

    def extract(self, text: str) -> RelationExtraction:
        """Every candidate relation in ``text``, deterministically."""
        if not text.strip():
            return RelationExtraction((), 0, 0, 0)
        relations: list[ExtractedRelation] = []
        sentences = _sentences(text)
        unmatched = 0
        single = 0
        for sentence in sentences:
            fired: list[ExtractedRelation] = []
            had_pair = False
            for clause in _clauses(sentence):
                spans = self.concepts_in(clause)
                if len(spans) < 2:
                    continue
                had_pair = True
                relation = self._match_clause(clause, spans)
                if relation is not None:
                    fired.append(relation)
            if fired:
                relations.extend(fired)
            elif had_pair:
                # Two concepts were named but no pattern linked them: a real miss.
                unmatched += 1
            else:
                # No clause carried two concepts, so there was nothing to link.
                single += 1
        return RelationExtraction(tuple(relations), len(sentences), unmatched, single)

    def _match_clause(
        self, clause: str, spans: list[tuple[int, int, Concept]]
    ) -> ExtractedRelation | None:
        """Match one clause against the pattern table, in concept order.

        The clause is tried against each pattern *whole* -- the subject phrase is
        the text up to a concept, the object phrase the text after the next --
        rather than only the words between two concepts. That is what makes "The
        density log measures bulk density" read as ``density_log measures
        bulk_density``: the leading article is part of the subject phrase, not an
        obstacle. Each side must then resolve to exactly one known concept, so
        the endpoints are never guessed.
        """
        for pattern, relation_type, passive in _PATTERNS:
            match = pattern.match(clause.strip())
            if match is None:
                continue
            left = self._sole_concept(clause.strip(), spans, match.start(1), match.end(1))
            right = self._head_concept(clause.strip(), spans, match.start(2), match.end(2))
            if left is None or right is None or left.id == right.id:
                continue
            # A passive pattern ("X is measured by Y") states the relation in the
            # other direction: Y is the actor and X the thing acted on.
            source, target = (right, left) if passive else (left, right)
            return ExtractedRelation(
                source=source.label,
                target=target.label,
                relation_type=relation_type,
                confidence=BASE_CONFIDENCE.get(relation_type, DEFAULT_CONFIDENCE),
                pattern=match.group(0).strip(),
                sentence=clause.strip(),
            )
        return None

    def _sole_concept(
        self, text: str, spans: list[tuple[int, int, Concept]], start: int, end: int
    ) -> Concept | None:
        """The single concept span inside ``[start, end)``, or ``None``.

        The spans are the clause's already-resolved concepts (longest whole-word
        match wins), so this reads the *same* resolution the extractor uses
        elsewhere instead of re-matching a substring, which would let ``density
        log`` also look like ``density`` and ``well_log``. ``None`` for zero
        concepts (an unknown noun) and for two or more (an ambiguous side), so a
        relation is only emitted when each side resolves to exactly one concept.
        """
        inside = [
            (span_start, span_end, concept)
            for span_start, span_end, concept in spans
            if span_start >= start and span_end <= end
        ]
        if not inside:
            return None
        # The head of a noun phrase is its *last* noun ("well control" is a kind
        # of control, not of well), so the concept that names the side is the one
        # whose span ends where the region ends. A concept that merely sits
        # earlier in the phrase ("well" in "well control", "rock" in "rock
        # property") is a modifier, not the head, so it is not the endpoint --
        # which is what stops a known modifier from being read as the object.
        region_end = max(span_end for _s, span_end, _c in inside)
        heads = {concept.id: concept for _s, span_end, concept in inside if span_end == region_end}
        if len(heads) != 1:
            return None
        head = next(iter(heads.values()))
        # The head must actually *end* its phrase: a content word after the
        # concept's span means the concept is only a modifier ("well" in "well
        # control", "rock" in "rock property"), not the noun the verb governs.
        if content_terms(text[region_end:end]):
            return None
        return head

    def _head_concept(
        self, clause: str, spans: list[tuple[int, int, Concept]], start: int, end: int
    ) -> Concept | None:
        """The concept that is the *head* of a verb's object phrase, or ``None``.

        The object's head noun is the one the verb governs, which is the noun
        *before* any preposition: in "causes a blowout in the reservoir" the verb
        governs ``blowout``, and ``reservoir`` is only a location. So the object's
        character range is cut at its first preposition and the concept inside
        that prefix is the target -- which is exactly what stops a known concept
        sitting inside a prepositional phrase from being the verb's object.
        """
        prefix_end = end
        for word in re.finditer(r"\w+", clause[start:end]):
            if normalize_token(word.group()) in _PREPOSITIONS:
                prefix_end = start + word.start()
                break
        return self._sole_concept(clause, spans, start, prefix_end)

    # -- applying ---------------------------------------------------------

    def apply(self, text: str) -> OntologyLearning:
        """Extract relations from ``text`` and fold them into the learner.

        Each extraction is handed to the learner as a *typed* learned relation,
        so it is bounded by the learner's ceiling and never overrides a seeded
        relation. The learner is what actually writes to the ontology, which is
        why extraction and learning stay separate concerns.
        """
        if self.learner is None:
            raise ValueError("This extractor has no learner; construct it with one to apply.")
        extraction = self.extract(text)
        added: list[tuple[str, str, str]] = []
        reinforced: list[tuple[str, str, float]] = []
        for relation in extraction.relations:
            learning = self.learner.observe_relation(
                relation.source, relation.target, relation.relation_type, relation.confidence
            )
            added.extend(learning.added)
            reinforced.extend(learning.reinforced)
        return OntologyLearning(
            concepts=tuple(
                dict.fromkeys(
                    name
                    for relation in extraction.relations
                    for name in (relation.source, relation.target)
                )
            ),
            added=tuple(added),
            reinforced=tuple(reinforced),
            skipped=extraction.unmatched,
        )


def merge_extractions(extractions: Iterable[RelationExtraction]) -> RelationExtraction:
    """Merge many extractions into one, de-duplicating identical triples.

    Two chunks of the same document often repeat a relation; the merged result
    keeps the first occurrence of each ``(source, type, target)`` so a report is
    not padded with copies, while the counters still reflect every sentence read.
    """
    relations: list[ExtractedRelation] = []
    seen: set[tuple[str, str, str]] = set()
    sentences = 0
    unmatched = 0
    single = 0
    for extraction in extractions:
        sentences += extraction.sentences
        unmatched += extraction.unmatched
        single += extraction.single_concept
        for relation in extraction.relations:
            key = (relation.source, relation.relation_type, relation.target)
            if key in seen:
                continue
            seen.add(key)
            relations.append(relation)
    return RelationExtraction(tuple(relations), sentences, unmatched, single)


def extract_from_store(
    store: ChunkSource,
    extractor: RelationExtractor,
    *,
    limit: int | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> RelationExtraction:
    """Read every chunk of a :class:`core.knowledge_base.KnowledgeBaseStore`.

    Streams the store rather than materialising it, so a large knowledge base is
    read one chunk at a time; ``limit`` bounds how many chunks are read. The
    store is duck-typed (it only needs ``iter_chunks``), so this works with a
    real store or a test double without importing the store here.
    """
    if limit is not None and limit < 0:
        raise ValueError("Chunk limit cannot be negative.")
    extractions: list[RelationExtraction] = []
    for index, chunk in enumerate(store.iter_chunks(), start=1):
        if limit is not None and index > limit:
            break
        extractions.append(extractor.extract(getattr(chunk, "text", "")))
        if on_progress is not None:
            on_progress(index)
    return merge_extractions(extractions)


class ExtractionJudgement:
    """One held-out sentence and the relation it should (or should not) yield.

    ``expected`` is the ``(source, relation_type, target)`` triple the sentence
    states, or ``None`` when the sentence must yield *no* relation at all -- a
    negative control, so a reader that over-fires is caught, not only one that
    misses. ``note`` records why the judgement is what it is.
    """

    __slots__ = ("expected", "note", "sentence")

    def __init__(
        self,
        sentence: str,
        expected: tuple[str, str, str] | None,
        note: str = "",
    ) -> None:
        if not sentence.strip():
            raise ValueError("An extraction judgement needs a sentence.")
        self.sentence = sentence
        self.expected = expected
        self.note = note


#: The held-out extraction catalogue. It is external to the extractor (written by
#: hand, with negative controls), so a change that makes the reader over-fire or
#: miss is a measured regression rather than a silent one.
EXTRACTION_JUDGEMENTS: tuple[ExtractionJudgement, ...] = (
    ExtractionJudgement(
        "Porosity increases permeability.", ("porosity", "increases", "permeability"),
        "active voice",
    ),
    ExtractionJudgement(
        "The density log measures porosity.", ("density_log", "measures", "porosity"),
        "a leading article must not hide the subject",
    ),
    ExtractionJudgement(
        "Porosity is measured by the density log.", ("density_log", "measures", "porosity"),
        "passive voice swaps the endpoints",
    ),
    ExtractionJudgement(
        "Permeability is reduced by water saturation.",
        ("saturation", "reduces", "permeability"),
        "a passive reduction",
    ),
    ExtractionJudgement(
        "The reservoir contains hydrocarbon.", ("reservoir", "contains", "hydrocarbon"),
        "containment",
    ),
    ExtractionJudgement(
        "Wettability affects relative permeability.",
        ("wettability", "affects", "relative_permeability"),
        "influence",
    ),
    ExtractionJudgement(
        "Porosity is a rock property.", None,
        "the object is a noun the ontology does not know",
    ),
    ExtractionJudgement(
        "A gas kick causes a blowout in the reservoir.", None,
        "the object head is unknown; the known concept is inside a preposition",
    ),
    ExtractionJudgement(
        "Porosity juggles permeability.", None,
        "the verb is not in the pattern table",
    ),
    ExtractionJudgement(
        "Porosity increases permeability, which controls flow.",
        ("porosity", "increases", "permeability"),
        "a relative clause must not be glued onto the object",
    ),
    ExtractionJudgement(
        "Drilling requires drilling fluid.", ("drilling", "depends_on", "drilling_fluid"),
        "a two-word verb phrase and a compound concept",
    ),
    ExtractionJudgement(
        "Porosity increases permeability, not flow.",
        ("porosity", "increases", "permeability"),
        "a trailing negation must not become the object",
    ),
    ExtractionJudgement(
        "Porosity does not increase permeability.", None,
        "a negated clause asserts nothing",
    ),
)


def extraction_accuracy(
    judgements: Iterable[ExtractionJudgement] = EXTRACTION_JUDGEMENTS,
    *,
    extractor: RelationExtractor | None = None,
) -> float:
    """The share of held-out judgements the extractor gets right.

    A judgement counts when the set of triples the extractor reads from the
    sentence equals what the judgement expects -- so an extra invented relation
    costs as much as a missed one, and a negative control is a real test rather
    than a formality. Returns ``0.0`` for an empty catalogue, so a caller can tell
    "no check" from "a perfect check". ``extractor`` is injectable for a test;
    with none it builds the project's seeded ontology (the controller is imported
    lazily, because the controller imports this module).
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    if extractor is None:
        from core.cognitive_controller import CognitiveController

        extractor = RelationExtractor(CognitiveController().ontology)
    correct = 0
    for judgement in catalogue:
        found = {
            (relation.source, relation.relation_type, relation.target)
            for relation in extractor.extract(judgement.sentence).relations
        }
        expected = {judgement.expected} if judgement.expected is not None else set()
        if found == expected:
            correct += 1
    return correct / len(catalogue)


__all__ = [
    "BASE_CONFIDENCE",
    "EXTRACTABLE_RELATIONS",
    "EXTRACTION_JUDGEMENTS",
    "ChunkSource",
    "ExtractedRelation",
    "ExtractionJudgement",
    "RelationExtraction",
    "RelationExtractor",
    "extract_from_store",
    "extraction_accuracy",
    "merge_extractions",
]
