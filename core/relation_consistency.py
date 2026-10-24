"""Flag contradictory relations in the graph instead of silently choosing a side.

:mod:`core.relation_polarity` can tell that ``increases`` opposes ``reduces``, and
``find_polarity_conflicts`` finds the pairs that assert both. But a conflict the
answer path never mentions is a conflict the user never learns about: the graph
holds "A increases B" and "A reduces B" and the reader is shown whichever one
retrieval happened to reach.

This module turns that scan into an honest, bounded **report**: which pairs of
concepts the graph holds contradictory claims about, how strongly, and which two
relation types disagree. It is **read-only** -- it never resolves a conflict,
never picks a winner, and never edits the graph -- because choosing between two
evidenced claims is the user's call, not the program's. When a question is given,
only conflicts on the concepts that question names are reported, so the answer
carries the caveat that is actually relevant to it.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.ontology import Ontology
from core.relation_polarity import find_polarity_conflicts


@dataclass(frozen=True)
class RelationConflict:
    """One contradictory pair, with its endpoints written as readable labels."""

    source: str
    relation_type: str
    target: str
    opposing_type: str
    severity: float

    def describe(self) -> str:
        """The disagreement as a line, e.g. ``a —increases→ b vs a —reduces→ b``."""
        return (
            f"{self.source} —{self.relation_type}→ {self.target}"
            f" vs {self.source} —{self.opposing_type}→ {self.target}"
            f" (severity {self.severity:.2f})"
        )


def relation_conflicts(
    ontology: Ontology | None,
    question: str | None = None,
) -> tuple[RelationConflict, ...]:
    """Contradictory relations the graph holds, optionally scoped to a question.

    With ``question=None`` every conflict in the graph is returned. With a
    question, only the conflicts whose **either** endpoint is a concept the
    question names are kept, so an answer carries the caveat that matters to it
    rather than the graph's whole history of disagreement. Ordered by severity
    (strongest first), then by the labels, so the same question always yields the
    same report. Returns ``()` when the graph is consistent, has no ontology, or
    the question names no conflicting concept -- never a placeholder.
    """
    if ontology is None:
        return ()
    seeds: set[str] | None = None
    if question is not None:
        seeds = set(ontology.concepts_in(question))
        if not seeds:
            return ()

    labels = {concept_id: concept.label for concept_id, concept in ontology.concepts.items()}
    conflicts = find_polarity_conflicts(ontology.relations.values())
    result: list[RelationConflict] = []
    for conflict in conflicts:
        if seeds is not None and conflict.source_id not in seeds and conflict.target_id not in seeds:
            continue
        source = labels.get(conflict.source_id)
        target = labels.get(conflict.target_id)
        if source is None or target is None:
            continue
        result.append(
            RelationConflict(
                source,
                conflict.first_relation,
                target,
                conflict.second_relation,
                conflict.severity,
            )
        )
    result.sort(key=lambda item: (-item.severity, item.source, item.target, item.relation_type))
    return tuple(result)


def describe_relation_conflicts(conflicts: tuple[RelationConflict, ...]) -> str:
    """Render the conflicts as a short honest warning block, or ``""``.

    The wording is deliberately neutral -- "the graph holds both ... and ..." --
    because the program does not know which side is right and must not imply it.
    """
    if not conflicts:
        return ""
    lines = [
        "Note: the knowledge graph holds contradictory claims on these concepts, "
        "so treat them as uncertain:"
    ]
    lines.extend(f"  - {conflict.describe()}" for conflict in conflicts)
    return "\n".join(lines)


class ConsistencyScenario:
    """A hand-written graph and the number of contradictions it should report.

    ``triples`` are ``(source, relation_type, target, confidence)`` edges; the
    scenario is *external* to the detector, so a change that starts hiding a real
    contradiction or inventing a false one is a measured regression.
    """

    __slots__ = ("expected", "note", "triples")

    def __init__(
        self,
        triples: tuple[tuple[str, str, str, float], ...],
        expected: int,
        note: str = "",
    ) -> None:
        if expected < 0:
            raise ValueError("Expected conflict count cannot be negative.")
        self.triples = triples
        self.expected = expected
        self.note = note


#: The held-out consistency catalogue: a planted conflict must be reported, a
#: consistent graph and an opposite-*direction* claim (a different statement, not
#: a contradiction) must not.
CONSISTENCY_SCENARIOS: tuple[ConsistencyScenario, ...] = (
    ConsistencyScenario(
        (
            ("a", "increases", "b", 0.9),
            ("a", "reduces", "b", 0.4),
        ),
        1,
        "two opposing relations on the same ordered pair",
    ),
    ConsistencyScenario(
        (
            ("a", "increases", "b", 0.9),
            ("b", "reduces", "a", 0.9),
        ),
        0,
        "opposite directions are different claims, not a contradiction",
    ),
    ConsistencyScenario(
        (("a", "increases", "b", 0.9),),
        0,
        "a single relation is consistent",
    ),
)


def consistency_accuracy(
    scenarios: tuple[ConsistencyScenario, ...] = CONSISTENCY_SCENARIOS,
) -> float:
    """The share of held-out scenarios whose contradiction count is exactly right.

    A scenario counts only when the number of conflicts the detector reports
    equals the expected number -- so a hidden contradiction and a false positive
    both fail. Returns ``0.0`` for an empty catalogue.
    """
    catalogue = tuple(scenarios)
    if not catalogue:
        return 0.0
    from core.semantic_relation import SemanticRelation

    correct = 0
    for scenario in catalogue:
        ontology = Ontology()
        ids: dict[str, str] = {}
        for source, relation_type, target, confidence in scenario.triples:
            for name in (source, target):
                if name not in ids:
                    ids[name] = ontology.add_concept_once(name).id
            ontology.add_relation(
                SemanticRelation(ids[source], ids[target], relation_type, confidence)
            )
        if len(relation_conflicts(ontology)) == scenario.expected:
            correct += 1
    return correct / len(catalogue)


__all__ = [
    "CONSISTENCY_SCENARIOS",
    "ConsistencyScenario",
    "RelationConflict",
    "consistency_accuracy",
    "describe_relation_conflicts",
    "relation_conflicts",
]
