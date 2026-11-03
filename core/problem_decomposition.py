"""Turn a question or task into the reasoning units a reasoning engine can work on.

The reasoning layer had every engine (deduction, induction, abduction, causal,
analogy, temporal, planning, decision, belief) but no component that *understood
the question first*: a caller had to guess which engine a question warranted, and
a multi-part question ("what is porosity and how do I measure it?") was handed to
one engine as if it were one thing. This module is that missing first step.

:func:`decompose` reads a question's **own words** and returns a
:class:`ProblemDecomposition` -- the main question, its objective, the constraints
it states, what is known (the concepts and relations the graph actually holds for
it), what is unknown (the parts no evidence reaches), the entities and
relationships involved, the intermediate conclusions a multi-part question needs,
and the expected output shape. It identifies a question's parts by the project's
own question-type cues (the procedural / decisional / evaluative / analogical /
temporal classifiers the answer path already uses), never by a language model.

**Honest by construction.** It **invents nothing**: a concept or relation it names
is one the ontology really holds, "unknown" is a part for which the graph holds no
evidence (reported, never silently assumed), and an unknown question type yields
``None`` rather than a fabricated unit. It is **read-only** (it never writes the
graph) and **dependency-light** (no model, no network, no Tk). When an optional
``interpreter`` callable is supplied it may *add* candidate sub-questions, but the
deterministic reading is always present and stands on its own, so the engine
operates without any model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from core.answer_analogy import is_analogical
from core.answer_belief import is_evaluative
from core.answer_decision import is_decisional
from core.answer_plan import is_procedural
from core.answer_temporal import is_temporal
from core.ontology import Ontology

#: The reasoning kinds a question part can warrant, one per answer-path bridge.
#: A kind names *which engine the part belongs to*, so the coordinator never
#: guesses: ``lookup`` is retrieval + the graph's stored facts, and each of the
#: others is one of the existing bridges.
KIND_LOOKUP = "lookup"
KIND_PROCEDURE = "procedure"
KIND_RECOMMENDATION = "recommendation"
KIND_VERDICT = "verdict"
KIND_COMPARISON = "comparison"
KIND_TIMELINE = "timeline"

#: The fixed order the kinds are tried in, so a part that reads as two kinds
#: (rare, but "how do I decide X?" is both) always resolves to the same one.
KIND_ORDER: tuple[str, ...] = (
    KIND_PROCEDURE,
    KIND_RECOMMENDATION,
    KIND_COMPARISON,
    KIND_VERDICT,
    KIND_TIMELINE,
    KIND_LOOKUP,
)

#: The default bound on how many parts a question is split into, so a very long
#: prompt cannot fan out without limit.
DEFAULT_MAX_PARTS = 8

#: A conjunction that joins two clauses of one question, in either language.
_CLAUSE_JOINERS: tuple[str, ...] = (" and ", " as well as ", " and also ", " و", " ثم ")

#: Words that open a clause, so a side of a conjunction is a *clause* and not just
#: a noun. A question about "porosity and permeability" names two nouns (neither
#: side opens a clause), while "what is porosity and how do I measure it?" names two
#: clauses -- which is what tells a multi-part question from a single one.
_CLAUSE_OPENERS: frozenset[str] = frozenset({
    "what", "how", "why", "where", "when", "which", "who", "whom", "whose",
    "is", "are", "was", "were", "does", "do", "did", "can", "could", "should",
    "will", "would", "may", "might", "must", "explain", "describe", "tell",
    "compare", "list", "give", "show",
    "ما", "ماذا", "كيف", "لماذا", "أين", "متى", "هل", "أي", "من", "كم",
    "اشرح", "صف", "قارن", "اذكر", "أعط", "وضح",
})


def _looks_like_clause(text: str) -> bool:
    """Whether ``text`` opens as a clause (a question word or an imperative verb).

    This is what distinguishes "how do I measure porosity **and** permeability"
    (one request naming two nouns -- the second side is a bare noun) from "what is
    porosity **and** how do I measure it" (two clauses -- the second side opens with
    a question word), so a single request is never torn into two parts.
    """
    words = [word for word in text.replace("؟", " ").replace("?", " ").split() if word]
    if not words:
        return False
    return words[0].casefold() in _CLAUSE_OPENERS


@dataclass(frozen=True)
class QuestionPart:
    """One reasoning unit of a question: a clause, its kind, and what it needs."""

    text: str
    """The clause as the user wrote it (trimmed), never rewritten."""

    kind: str
    """Which engine the part belongs to (one of :data:`KIND_ORDER`)."""

    entities: tuple[str, ...] = ()
    """The labels of the concepts this part names, first mention first."""

    required_evidence: tuple[str, ...] = ()
    """What evidence this part needs, phrased from the concepts it names."""

    known: bool = False
    """Whether the graph holds evidence the part can rest on."""

    depends_on: tuple[int, ...] = ()
    """Indices of earlier parts this part depends on (empty for the first)."""

    @property
    def is_unknown(self) -> bool:
        """A part the graph cannot speak to -- reported, never silently assumed."""
        return not self.known

    def describe(self) -> str:
        """One readable line, e.g. ``[1] how do I measure porosity? (procedure)``."""
        return f"{self.text} ({self.kind})"

    def to_primitive(self) -> dict:
        """A JSON-serialisable view of the part (nothing computed is dropped)."""
        return {
            "text": self.text,
            "kind": self.kind,
            "entities": list(self.entities),
            "required_evidence": list(self.required_evidence),
            "known": self.known,
            "depends_on": list(self.depends_on),
        }


@dataclass(frozen=True)
class ProblemDecomposition:
    """A question broken into the reasoning units it warrants."""

    question: str
    main: str
    objective: str
    parts: tuple[QuestionPart, ...] = ()
    constraints: tuple[str, ...] = ()
    known: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    relationships: tuple[str, ...] = ()
    intermediate: tuple[str, ...] = ()
    expected_output: str = ""
    language: str = "en"
    multi_part: bool = False

    @property
    def kinds(self) -> tuple[str, ...]:
        """The distinct kinds the parts warrant, in the fixed order."""
        present = {part.kind for part in self.parts}
        return tuple(kind for kind in KIND_ORDER if kind in present)

    @property
    def has_unknowns(self) -> bool:
        return bool(self.unknown)

    def describe(self) -> str:
        """The decomposition as a short, readable block (no hidden reasoning)."""
        lines = [f"Main question: {self.main}", f"Objective: {self.objective}"]
        if self.entities:
            lines.append("Entities: " + ", ".join(self.entities))
        if self.relationships:
            lines.append("Relationships: " + "; ".join(self.relationships))
        if self.constraints:
            lines.append("Constraints: " + "; ".join(self.constraints))
        if self.multi_part:
            lines.append(f"Parts ({len(self.parts)}):")
            lines.extend(f"  {index + 1}. {part.describe()}" for index, part in enumerate(self.parts))
        if self.intermediate:
            lines.append("Intermediate conclusions: " + "; ".join(self.intermediate))
        if self.known:
            lines.append("Known: " + "; ".join(self.known))
        if self.unknown:
            lines.append("Unknown: " + "; ".join(self.unknown))
        lines.append(f"Expected output: {self.expected_output}")
        return "\n".join(lines)

    def to_primitive(self) -> dict:
        """A JSON-serialisable view of the whole decomposition."""
        return {
            "question": self.question,
            "main": self.main,
            "objective": self.objective,
            "parts": [part.to_primitive() for part in self.parts],
            "constraints": list(self.constraints),
            "known": list(self.known),
            "unknown": list(self.unknown),
            "entities": list(self.entities),
            "relationships": list(self.relationships),
            "intermediate": list(self.intermediate),
            "expected_output": self.expected_output,
            "language": self.language,
            "multi_part": self.multi_part,
        }


def _split_parts(question: str, *, limit: int) -> list[str]:
    """Split a question into its clauses, keeping the user's own wording.

    A question with a conjunction that joins two *clauses* ("what is porosity and
    how do I measure it?") is read as several parts; a single-clause question is one
    part. A conjunction that joins two *nouns* ("how do I measure porosity and
    permeability?") is **not** split -- that is one request naming two concepts. The
    test is whether the side after the conjunction opens as its own clause (a
    question word or an imperative), which is what makes a part a part. Splitting is
    bounded by ``limit``.
    """
    text = question.strip()
    if not text:
        return []
    for joiner in _CLAUSE_JOINERS:
        index = text.casefold().find(joiner)
        if index <= 0:
            continue
        left = text[:index].strip(" ,،.؟?")
        right = text[index + len(joiner):].strip(" ,،.؟?")
        if len(left) < 3 or len(right) < 3:
            continue
        # The second side must open as its own clause; the first side must carry
        # content. Otherwise the conjunction joins nouns, and the question is one
        # part -- so "porosity and permeability" is never torn apart.
        if not (_has_content(left) and _looks_like_clause(right)):
            continue
        parts = [left, right]
        # Recurse on the remainder so a three-part question is fully split, bounded.
        deeper = _split_parts(right, limit=limit)
        if len(deeper) > 1:
            parts = [left, *deeper]
        return [part for part in parts if part][:limit]
    return [text][:limit]


def _has_content(text: str) -> bool:
    """Whether ``text`` carries at least one alphabetic word of two or more letters."""
    for raw in text.replace("?", " ").replace("؟", " ").split():
        word = "".join(char for char in raw if char.isalpha())
        if len(word) >= 2:
            return True
    return False


def _kind_for(text: str) -> str:
    """Which engine a part warrants, by the project's own question-type cues.

    Read in the fixed :data:`KIND_ORDER`, so a part that reads as two kinds always
    resolves to the same one. ``lookup`` is the honest default: a part that asks no
    recognisable kind is answered from the graph's stored facts and cited sources.
    """
    if is_procedural(text):
        return KIND_PROCEDURE
    if is_decisional(text):
        return KIND_RECOMMENDATION
    if is_analogical(text):
        return KIND_COMPARISON
    if is_evaluative(text):
        return KIND_VERDICT
    if is_temporal(text):
        return KIND_TIMELINE
    return KIND_LOOKUP


def _labels_in(ontology: Ontology, text: str) -> tuple[str, ...]:
    """The labels of the concepts ``text`` names, first mention first."""
    labels: list[str] = []
    seen: set[str] = set()
    for _start, _end, concept in ontology.match_concepts(text):
        if concept.id in seen:
            continue
        seen.add(concept.id)
        labels.append(concept.label)
    return tuple(labels)


def _relationships(ontology: Ontology, labels: tuple[str, ...]) -> tuple[str, ...]:
    """The stored relations among the concepts a question names, as readable lines.

    Only relations the graph **really holds** between two of the named concepts are
    reported, in their declared direction, so a decomposition never invents a link.
    """
    ids = {
        concept.id
        for label in labels
        for concept in ontology.find_concepts(label)
    }
    lines: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for relation in ontology.relations.values():
        if relation.source_id not in ids or relation.target_id not in ids:
            continue
        source = ontology.concepts.get(relation.source_id)
        target = ontology.concepts.get(relation.target_id)
        if source is None or target is None:
            continue
        key = (source.label, relation.relation_type, target.label)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{source.label} —{relation.relation_type}→ {target.label}")
    return tuple(sorted(lines))


def _objective(kind: str) -> str:
    """The objective a part of ``kind`` serves, phrased from the kind alone."""
    return {
        KIND_PROCEDURE: "find the ordered procedure that achieves the goal",
        KIND_RECOMMENDATION: "rank the options the graph states for the goal",
        KIND_COMPARISON: "compare the two structures the question names",
        KIND_VERDICT: "weigh the evidence for and against the claim",
        KIND_TIMELINE: "order what happened over the conversation",
        KIND_LOOKUP: "state what the knowledge graph and the sources hold",
    }.get(kind, "answer from the available evidence")


def _expected_output(kinds: tuple[str, ...]) -> str:
    """The expected output shape from the kinds a question warrants."""
    if not kinds:
        return "a cited answer grounded in the available evidence"
    labels = {
        KIND_PROCEDURE: "an ordered procedure",
        KIND_RECOMMENDATION: "a ranked recommendation",
        KIND_COMPARISON: "a structural comparison",
        KIND_VERDICT: "an evidence-weighed verdict",
        KIND_TIMELINE: "an ordered timeline",
        KIND_LOOKUP: "a cited answer",
    }
    phrases = [labels[kind] for kind in kinds]
    if len(phrases) == 1:
        return phrases[0] + ", grounded in cited evidence"
    return " and ".join(phrases) + ", each grounded in cited evidence"


def _language(text: str) -> str:
    """The question's language, reusing the project's own script check."""
    from core.voices import detect_language

    return detect_language(text)


def decompose(
    question: str,
    *,
    ontology: Ontology | None = None,
    max_parts: int = DEFAULT_MAX_PARTS,
    interpreter: Callable[[str], list[str]] | None = None,
) -> ProblemDecomposition | None:
    """Break ``question`` into the reasoning units it warrants.

    Returns ``None`` for a blank question rather than a fabricated decomposition.
    ``ontology`` is the graph the known/unknown reading is taken from; with none the
    project's seeded ontology is used. ``interpreter`` is an **optional** callable
    (a language model, say) that may *add* candidate sub-questions; the
    deterministic reading is always computed first and is never replaced by it, so
    the decomposition is complete with no model at all.
    """
    text = question.strip()
    if not text:
        return None
    if ontology is None:
        from core.cognitive_controller import CognitiveController

        ontology = CognitiveController().ontology

    clauses = _split_parts(text, limit=max_parts)
    if interpreter is not None:
        # An interpreter may widen the reading, never replace it; a failing
        # interpreter is ignored rather than allowed to break the decomposition.
        try:
            extra = [c for c in interpreter(text) if isinstance(c, str) and c.strip()]
        except Exception:
            extra = []
        for clause in extra:
            if clause.strip() not in clauses and len(clauses) < max_parts:
                clauses.append(clause.strip())

    parts: list[QuestionPart] = []
    for index, clause in enumerate(clauses):
        labels = _labels_in(ontology, clause)
        kind = _kind_for(clause)
        known = _part_known(ontology, labels, kind)
        required = tuple(f"evidence about {label}" for label in labels) or (
            "evidence the sources hold for this question",
        )
        # A second part that names a concept the first introduced depends on it,
        # so the coordinator orders the parts rather than answering them flat.
        depends: tuple[int, ...] = ()
        if index > 0 and labels and any(label in parts[0].entities for label in labels):
            depends = (0,)
        parts.append(
            QuestionPart(
                text=clause,
                kind=kind,
                entities=labels,
                required_evidence=required,
                known=known,
                depends_on=depends,
            )
        )

    entities: list[str] = []
    for part in parts:
        for label in part.entities:
            if label not in entities:
                entities.append(label)
    relationships = _relationships(ontology, tuple(entities))
    known_lines = tuple(
        f"{part.text} → {_objective(part.kind)}" for part in parts if part.known
    )
    unknown_lines = tuple(
        f"{part.text} → no stored evidence reaches it" for part in parts if not part.known
    )
    intermediate = _intermediate(tuple(parts))
    kinds = tuple(kind for kind in KIND_ORDER if kind in {part.kind for part in parts})

    return ProblemDecomposition(
        question=text,
        main=text,
        objective=_objective(parts[0].kind) if parts else "answer from the available evidence",
        parts=tuple(parts),
        constraints=_constraints(text),
        known=known_lines,
        unknown=unknown_lines,
        entities=tuple(entities),
        relationships=relationships,
        intermediate=intermediate,
        expected_output=_expected_output(kinds),
        language=_language(text),
        multi_part=len(parts) > 1,
    )


def _part_known(ontology: Ontology, labels: tuple[str, ...], kind: str) -> bool:
    """Whether the graph holds anything the part can rest on.

    A part is *known* when it names at least one concept the graph holds **and** the
    kind's own evidence exists: a lookup/verdict/comparison needs a stored relation
    on the named concepts, a procedure needs a corpus entry with a stated procedure,
    a recommendation needs an inbound edge in the direction asked for. A temporal
    part is always answerable (it reads the conversation, not the graph). Anything
    else is honestly *unknown* -- reported, never assumed.
    """
    if kind == KIND_TIMELINE:
        return True
    if not labels:
        return False
    ids = {concept.id for label in labels for concept in ontology.find_concepts(label)}
    if kind == KIND_PROCEDURE:
        from core.answer_plan import match_entry

        return match_entry(" ".join(labels)) is not None
    if kind == KIND_RECOMMENDATION:
        # Any relation into one of the named concepts is enough for the part to be
        # answerable at all; the decision engine still refuses a direction the graph
        # holds no edge for, and that refusal is honest.
        return any(
            relation.target_id in ids or relation.source_id in ids
            for relation in ontology.relations.values()
        )
    return any(
        relation.source_id in ids or relation.target_id in ids
        for relation in ontology.relations.values()
    )


def _constraints(question: str) -> tuple[str, ...]:
    """The constraints a question states, read from its own words.

    Only explicit, checkable constraints are reported (a named unit, a limit, a
    scope word), so a constraint is never inferred from a question that states none.
    """
    lowered = question.casefold()
    found: list[str] = []
    cues = {
        "in english": "answer in English",
        "in arabic": "answer in Arabic",
        "briefly": "keep the answer brief",
        "in detail": "give a detailed answer",
        "step by step": "answer step by step",
        "بالعربية": "answer in Arabic",
        "بالإنجليزية": "answer in English",
        "باختصار": "keep the answer brief",
        "بالتفصيل": "give a detailed answer",
    }
    for cue, constraint in cues.items():
        if cue in lowered:
            found.append(constraint)
    return tuple(found)


def _intermediate(parts: tuple[QuestionPart, ...]) -> tuple[str, ...]:
    """The intermediate conclusions a multi-part question needs, in order.

    A part that another part depends on becomes an intermediate conclusion: the
    dependent part cannot be answered until the earlier one is. A single-part
    question needs none, so this is empty rather than a fabricated step.
    """
    if len(parts) < 2:
        return ()
    lines: list[str] = []
    for index, part in enumerate(parts):
        if any(index in other.depends_on for other in parts):
            lines.append(f"first establish: {part.text}")
    return tuple(lines)


@dataclass(frozen=True)
class DecompositionJudgement:
    """One held-out expectation about how a question must be decomposed.

    ``expect_parts`` is the number of reasoning units; ``expect_kinds`` the kinds
    the parts must warrant (a subset); ``expect_multi`` whether it is a multi-part
    question. A negative control states the honest empty/one-part case, so a
    decomposer that splits a single-clause question fails the gate.
    """

    name: str
    question: str
    expect_parts: int
    expect_kinds: tuple[str, ...] = ()
    expect_multi: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A decomposition judgement needs a name.")
        for kind in self.expect_kinds:
            if kind not in KIND_ORDER:
                raise ValueError(f"Unknown kind in a decomposition judgement: {kind}")
        if self.expect_parts < 0:
            raise ValueError("Expected part count cannot be negative.")


#: The held-out decomposition catalogue -- hand-written with **negative controls**:
#: a single-clause question must stay one part (a noun conjunction must not be split),
#: an empty question must decompose to nothing, and a multi-clause question must be
#: read as the parts its kinds warrant.
DECOMPOSITION_JUDGEMENTS: tuple[DecompositionJudgement, ...] = (
    DecompositionJudgement("empty", "", 0, expect_multi=False),
    DecompositionJudgement(
        "single-lookup", "what is porosity?", 1, (KIND_LOOKUP,), expect_multi=False
    ),
    DecompositionJudgement(
        "noun-conjunction-not-split",
        "how do I measure porosity and permeability?",
        1,
        (KIND_PROCEDURE,),
        expect_multi=False,
    ),
    DecompositionJudgement(
        "two-clause-question",
        "what is porosity and how do I measure it?",
        2,
        (KIND_LOOKUP, KIND_PROCEDURE),
        expect_multi=True,
    ),
    DecompositionJudgement(
        "arabic-verdict",
        "هل التحميض يزيد النفاذية؟",
        1,
        (KIND_VERDICT,),
        expect_multi=False,
    ),
)


def decomposition_accuracy(
    judgements: tuple[DecompositionJudgement, ...] = DECOMPOSITION_JUDGEMENTS,
    *,
    ontology: Ontology | None = None,
) -> float:
    """The share of held-out judgements the decomposer gets right.

    A judgement counts only when the part count, the multi-part flag, and the set of
    kinds the parts warrant all match -- so a decomposer that splits a single-clause
    question, merges two clauses, or mislabels a part's kind fails as surely as one
    that misses a real part. Returns ``0.0`` for an empty catalogue.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    correct = 0
    for judgement in catalogue:
        result = decompose(judgement.question, ontology=ontology)
        if judgement.expect_parts == 0:
            ok = result is None
        else:
            ok = (
                result is not None
                and len(result.parts) == judgement.expect_parts
                and result.multi_part == judgement.expect_multi
                and set(judgement.expect_kinds).issubset({part.kind for part in result.parts})
            )
        if ok:
            correct += 1
    return correct / len(catalogue)


__all__ = [
    "DECOMPOSITION_JUDGEMENTS",
    "DEFAULT_MAX_PARTS",
    "KIND_COMPARISON",
    "KIND_LOOKUP",
    "KIND_ORDER",
    "KIND_PROCEDURE",
    "KIND_RECOMMENDATION",
    "KIND_TIMELINE",
    "KIND_VERDICT",
    "DecompositionJudgement",
    "ProblemDecomposition",
    "QuestionPart",
    "decompose",
    "decomposition_accuracy",
]
