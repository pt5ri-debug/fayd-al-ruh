"""Surface the *structural analogy* for a question that asks how two things are alike.

The project carries a whole analogical layer -- ``AnalogyEngine``,
``StructureMappingEngine``, ``Structure``, ``StructureMapping`` -- but it was
reachable only from the ``CognitiveController``, ``cli/analogy.py``, and the
evaluation scenarios, never from an answer. So a question like *"is porosity like
permeability?"* got a definition back -- or, worse, the belief bridge read its
``is ... ?`` opening as a yes/no verdict about a *different* claim ("does porosity
increase permeability?") -- never the structural comparison the graph already
supports. That is the analogical half of Gap 7 of ``docs/REASONING_ASSESSMENT.md``.

This module reads the concepts the question names, builds each one's relational
neighbourhood, and aligns them with the project's real ``AnalogyEngine`` -- so the
comparison that reaches the reader is the one the structure-mapping engine
actually produced, with its own relation matches, its own systematicity, and its
own candidate inferences (each a *candidate*, discounted by how much of the
structure really aligned). It is **read-only** and **honest by construction**: it
only builds a comparison when the question actually *asks how two things are
alike* (a definitional "what is porosity?" gets none), it never invents a relation
the graph did not state, it reports what it could **not** map as well as what it
could, and it returns ``None`` rather than a guess when the question names fewer
than two concepts.

The comparison is a *reported* artifact: nothing here is asserted into the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.analogy import Analogy, AnalogyEngine
from core.voices import detect_language

if TYPE_CHECKING:
    from core.ontology import Ontology

#: The phrases that make a question *analogical* -- a request to compare two
#: things by their structure, in either language. A comparison is only produced for
#: such a question, so a definitional question never drags one into its answer. The
#: English cue is the word *like* between spaces, so a question must be normalised
#: (punctuation to spaces) before it is tested -- otherwise "what is porosity
#: like?" would slip through while "is porosity like permeability?" fired.
ANALOGY_CUES: tuple[str, ...] = (
    " like ",
    "similar to",
    "similarity",
    "analogous",
    "analogy",
    "compare",
    "comparison",
    "compared to",
    "same as",
    "resembles",
    "in common",
    "يشبه",
    "تشبه",
    "شبيه",
    "شبيهة",
    "مماثل",
    "مماثلة",
    "قارن",
    "مقارنة",
    "على غرار",
)

#: How many candidate inferences one comparison reports. A pair whose neighbourhoods
#: align broadly would otherwise flood the answer.
DEFAULT_TRANSFER_LIMIT = 6


@dataclass(frozen=True)
class AnswerAnalogy:
    """The structural comparison the graph supports for a question that asks for one."""

    base: str
    """The first concept the question named (the base of the mapping)."""

    target: str
    """The second concept the question named (the target of the mapping)."""

    meaningful: bool
    """Whether the alignment is a real shared *system* of relations rather than one
    incidental fact -- the engine's own ``is_meaningful`` verdict, reported honestly
    even when it is ``False``."""

    score: float
    """The mapping's score (``core.structure_mapping.mapping_score``)."""

    matched: tuple[str, ...]
    """The relations that mapped, as readable ``base == target`` lines."""

    transferred: tuple[tuple[str, str, str, float], ...]
    """The candidate inferences the analogy suggests for the target, as
    ``(source, relation, target, confidence)`` rows -- never asserted, and each
    discounted by how much of the structure aligned."""

    unmapped_base: tuple[str, ...]
    """The base's concepts the mapping could not place, reported rather than hidden."""

    unmapped_target: tuple[str, ...]
    """The target's concepts the mapping could not place."""

    language: str
    """``"ar"`` or ``"en"`` -- the language the comparison was written in."""

    def describe(self) -> str:
        """The comparison as a heading followed by its matches and its gaps."""
        if self.language == "ar":
            if self.meaningful:
                head = (
                    f"المقارنة البنيوية بين {self.base} و{self.target}: "
                    f"تشابه في البنية (درجة {self.score:.2f})."
                )
            else:
                head = (
                    f"المقارنة البنيوية بين {self.base} و{self.target}: "
                    f"لا يوجد نظام علاقات مشترك (درجة {self.score:.2f})."
                )
            lines = [head]
            lines.extend(f"  - تطابق: {line}" for line in self.matched)
            if self.transferred:
                lines.append("  - استنتاجات مرشَّحة (غير مؤكَّدة):")
                lines.extend(
                    f"    {source} —{relation}→ {target} (ثقة {confidence:.2f})"
                    for source, relation, target, confidence in self.transferred
                )
            if self.unmapped_base:
                lines.append(f"  - لم يُطابَق في {self.base}: {'، '.join(self.unmapped_base)}")
            if self.unmapped_target:
                lines.append(f"  - لم يُطابَق في {self.target}: {'، '.join(self.unmapped_target)}")
            return "\n".join(lines)
        if self.meaningful:
            head = (
                f"The structural analogy between {self.base} and {self.target}: "
                f"a shared relational system (score {self.score:.2f})."
            )
        else:
            head = (
                f"The structural analogy between {self.base} and {self.target}: "
                f"no shared relational system (score {self.score:.2f})."
            )
        lines = [head]
        lines.extend(f"  - matched: {line}" for line in self.matched)
        if self.transferred:
            lines.append("  - candidate inferences (not asserted):")
            lines.extend(
                f"    {source} --{relation}--> {target} (confidence {confidence:.2f})"
                for source, relation, target, confidence in self.transferred
            )
        if self.unmapped_base:
            lines.append(f"  - unmapped in {self.base}: {', '.join(self.unmapped_base)}")
        if self.unmapped_target:
            lines.append(f"  - unmapped in {self.target}: {', '.join(self.unmapped_target)}")
        return "\n".join(lines)


def _normalise(question: str) -> str:
    """The question with punctuation turned to spaces, so a cue can be whole-word.

    ``str.split`` then collapses runs of whitespace, and the leading/trailing space
    makes a cue like ``" like "`` match the *word* ``like`` even at the question's
    edge -- without it, "what is porosity like?" would not fire while "is porosity
    like permeability?" would.
    """
    flattened = question.casefold()
    for mark in "?!.,;:()[]{}\"'؟،؛«»":
        flattened = flattened.replace(mark, " ")
    return " " + " ".join(flattened.split()) + " "


def is_analogical(question: str) -> bool:
    """Whether a question asks how two things are alike, so a comparison is warranted.

    Whole-phrase and case-insensitive over the *normalised* question, so a
    definitional question ("what is porosity?") is never read as analogical.
    """
    normalised = _normalise(question)
    return any(cue in normalised for cue in ANALOGY_CUES)


def _named_concepts(ontology: Ontology, question: str) -> list[str]:
    """The labels of the concepts the question names, first mention first."""
    labels: list[str] = []
    seen: set[str] = set()
    for _start, _end, concept in ontology.match_concepts(question):
        if concept.id in seen:
            continue
        seen.add(concept.id)
        labels.append(concept.label)
    return labels


def _matched_lines(analogy: Analogy) -> tuple[str, ...]:
    """The mapped relations as readable lines, literal matches first."""
    rows: list[tuple[int, str]] = []
    for match in analogy.mapping.matched_edges:
        base, counterpart = match.base, match.target
        marker = "" if match.exact else " (by role)"
        rows.append(
            (
                0 if match.exact else 1,
                f"{base.source} --{base.relation_type}--> {base.target}"
                f"  ==  {counterpart.source} --{counterpart.relation_type}-->"
                f" {counterpart.target}{marker}",
            )
        )
    rows.sort(key=lambda row: (row[0], row[1]))
    return tuple(line for _rank, line in rows)


def _build(
    ontology: Ontology,
    base: str,
    target: str,
    *,
    hops: int,
    limit: int,
    language: str,
) -> AnswerAnalogy:
    """Align two named concepts and package the engine's own verdict."""
    analogy = AnalogyEngine(ontology).between_concepts(base, target, hops=hops)
    transferred = analogy.mapping.transferred()[:limit]
    return AnswerAnalogy(
        base=base,
        target=target,
        meaningful=analogy.is_meaningful,
        score=analogy.score,
        matched=_matched_lines(analogy),
        transferred=transferred,
        unmapped_base=analogy.mapping.unmapped_base_nodes,
        unmapped_target=analogy.mapping.unmapped_target_nodes,
        language=language,
    )


def answer_analogy(
    question: str,
    *,
    ontology: Ontology | None = None,
    hops: int = 1,
    limit: int = DEFAULT_TRANSFER_LIMIT,
    language: str | None = None,
) -> AnswerAnalogy | None:
    """The structural comparison for a question, or ``None`` when none is warranted.

    A comparison is produced only when the question is *analogical*
    (``is_analogical``) **and** it names at least two concepts the ontology knows.
    Otherwise ``None`` -- a definitional or unrelated question gets no comparison
    rather than a guess. The verdict itself (``meaningful``) is the engine's, so a
    question about two concepts that share no relational system is reported as
    exactly that, not dressed up as an analogy.
    """
    if hops < 0:
        raise ValueError("Hops must not be negative.")
    if limit < 1:
        raise ValueError("Transfer limit must be at least 1.")
    if not is_analogical(question):
        return None
    if ontology is None:
        from core.cognitive_controller import CognitiveController

        ontology = CognitiveController().ontology
    labels = _named_concepts(ontology, question)
    if len(labels) < 2:
        return None
    lang = language or detect_language(question)
    return _build(ontology, labels[0], labels[1], hops=hops, limit=limit, language=lang)


@dataclass(frozen=True)
class AnalogyBridgeJudgement:
    """One held-out expectation about the analogy bridge.

    ``kind`` selects what is checked: ``"meaningful"`` that an analogical question
    reaches the engine's ``expected`` verdict, ``"transfers"`` that the candidate
    inferences are the ones the mapping actually suggests, and ``"no-analogy"``
    that a question warrants no comparison at all (a negative control: a
    definitional question, an unrelated one, or an analogical question naming fewer
    than two known concepts).
    """

    name: str
    kind: str
    question: str = ""
    expected: bool = False
    expected_inferences: tuple[tuple[str, str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("An analogy judgement needs a name.")
        if self.kind not in ("meaningful", "transfers", "no-analogy"):
            raise ValueError(f"Unknown analogy judgement kind: {self.kind}")


#: The held-out analogy-bridge catalogue -- hand-written with **negative controls**:
#: a definitional question, an unrelated one, and an analogical question naming one
#: concept must all yield no comparison, and a pair with no shared relational system
#: must be reported as *not* an analogy rather than dressed up as one.
ANALOGY_ANSWER_JUDGEMENTS: tuple[AnalogyBridgeJudgement, ...] = (
    AnalogyBridgeJudgement(
        "a definitional question gets no comparison",
        "no-analogy",
        question="what is porosity?",
    ),
    AnalogyBridgeJudgement(
        "an unrelated question gets no comparison",
        "no-analogy",
        question="what is the capital of France?",
    ),
    AnalogyBridgeJudgement(
        "an analogical question naming one concept gets no comparison",
        "no-analogy",
        question="what is porosity like?",
    ),
    AnalogyBridgeJudgement(
        "two concepts sharing a relational system are an analogy",
        "meaningful",
        question="is porosity like permeability?",
        expected=True,
    ),
    AnalogyBridgeJudgement(
        "two instruments of the same shape are an analogy",
        "meaningful",
        question="is the density log similar to the sonic log?",
        expected=True,
    ),
    AnalogyBridgeJudgement(
        "a property and an operation are not an analogy",
        "meaningful",
        question="is porosity like waterflood?",
        expected=False,
    ),
    AnalogyBridgeJudgement(
        "the comparison transfers the base's own relations",
        "transfers",
        question="is porosity like permeability?",
        expected_inferences=(("core_sample", "measures", "porosity"),),
    ),
)


def analogy_answer_accuracy(
    judgements: tuple[AnalogyBridgeJudgement, ...] = ANALOGY_ANSWER_JUDGEMENTS,
) -> float:
    """The share of held-out analogy-bridge judgements the bridge gets right.

    A judgement counts only when the bridge does exactly what it must: reach the
    engine's verdict, transfer the relations the mapping actually suggests, or
    produce **no** comparison for a question that warrants none. So a bridge that
    invents an analogy, hides a non-analogy, or fabricates a transfer fails the
    relevant judgement. Returns ``0.0`` for an empty catalogue.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    correct = 0
    for judgement in catalogue:
        comparison = answer_analogy(judgement.question)
        if judgement.kind == "no-analogy":
            ok = comparison is None
        elif comparison is None:
            ok = False
        elif judgement.kind == "meaningful":
            ok = comparison.meaningful == judgement.expected
        else:  # transfers
            rows = {(source, relation, target) for source, relation, target, _c in comparison.transferred}
            ok = all(row in rows for row in judgement.expected_inferences)
        if ok:
            correct += 1
    return correct / len(catalogue)


__all__ = [
    "ANALOGY_ANSWER_JUDGEMENTS",
    "ANALOGY_CUES",
    "DEFAULT_TRANSFER_LIMIT",
    "AnalogyBridgeJudgement",
    "AnswerAnalogy",
    "analogy_answer_accuracy",
    "answer_analogy",
    "is_analogical",
]
