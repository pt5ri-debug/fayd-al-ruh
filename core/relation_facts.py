"""Surface the knowledge graph's typed relations as evidence for an answer.

Retrieval returns *passages*: whole sentences a document states. But the project
also holds a **concept graph** whose edges are typed relations (``porosity
increases permeability``, ``density_log measures porosity``) -- including links a
text taught through :mod:`core.relation_extraction`. Those links are facts in
their own right, and an answer that never states them is quietly less complete
than the knowledge on file.

This module reads the relations a question's own concepts participate in and
renders them as a short, deterministic, attributed block. It is **read-only**:
it never edits the graph, never invents a relation, and states only the edges the
ontology actually carries, in the declared direction. An unknown concept yields
nothing, so a question the graph knows nothing about produces no block rather
than a guess.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from core.ontology import Ontology

#: How many relations one answer states. A question that names a well-connected
#: concept can reach dozens of edges; the block is evidence, not a dump, so it is
#: bounded and the strongest edges are kept.
DEFAULT_FACT_LIMIT = 6


@dataclass(frozen=True)
class RelationFact:
    """One typed relation, with its endpoints written as readable labels."""

    source: str
    relation_type: str
    target: str
    confidence: float

    @property
    def statement(self) -> str:
        """The relation as a phrase, e.g. ``porosity increases permeability``.

        The relation type's underscores are read as spaces, so a stored
        ``depends_on`` reads as the words a sentence would use. The type is never
        otherwise rewritten -- the graph's own word is what the reader sees.
        """
        return f"{self.source} {self.relation_type.replace('_', ' ')} {self.target}"

    def describe(self) -> str:
        """The relation with its type marked, e.g. ``porosity —increases→ permeability``."""
        return f"{self.source} —{self.relation_type}→ {self.target}"


def relation_facts(
    ontology: Ontology | None,
    question: str,
    *,
    limit: int = DEFAULT_FACT_LIMIT,
) -> tuple[RelationFact, ...]:
    """The typed relations the question's own concepts participate in.

    A relation is included when **either** endpoint is a concept the question
    names, so both an outgoing claim ("porosity increases permeability") and an
    incoming one ("the density log measures porosity", asked about porosity) are
    reachable. Relations are de-duplicated by their triple and ordered by
    confidence, then by their labels, so the same question always yields the same
    block. Returns ``()`` for no ontology, an empty question, or a question whose
    concepts carry no relation -- never a placeholder.
    """
    if ontology is None or not question.strip():
        return ()
    if limit < 1:
        raise ValueError("Relation fact limit must be at least 1.")

    seeds = ontology.concepts_in(question)
    if not seeds:
        return ()

    labels = {concept_id: concept.label for concept_id, concept in ontology.concepts.items()}
    facts: dict[tuple[str, str, str], RelationFact] = {}
    for relation in ontology.relations.values():
        if relation.source_id not in seeds and relation.target_id not in seeds:
            continue
        source = labels.get(relation.source_id)
        target = labels.get(relation.target_id)
        if source is None or target is None:
            continue
        key = (source, relation.relation_type, target)
        if key in facts:
            continue
        facts[key] = RelationFact(source, relation.relation_type, target, relation.confidence)

    ordered = sorted(facts.values(), key=lambda fact: (-fact.confidence, fact.source, fact.relation_type, fact.target))
    return tuple(ordered[:limit])


def describe_relation_facts(facts: Iterable[RelationFact]) -> str:
    """Render the facts as a short attributed block, or ``""`` when there are none.

    The block names itself as the *knowledge graph* on purpose: a reader must be
    able to tell a typed link the graph stores from a sentence a document states,
    because the two have different provenance.
    """
    present = list(facts)
    if not present:
        return ""
    lines = ["The knowledge graph records these relations on file:"]
    lines.extend(f"  - {fact.describe()}" for fact in present)
    return "\n".join(lines)


__all__ = [
    "DEFAULT_FACT_LIMIT",
    "RelationFact",
    "describe_relation_facts",
    "relation_facts",
]
