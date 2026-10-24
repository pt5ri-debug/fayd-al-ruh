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
from core.relation_polarity import PolarityConflict, find_polarity_conflicts


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


__all__ = [
    "RelationConflict",
    "describe_relation_conflicts",
    "relation_conflicts",
]
