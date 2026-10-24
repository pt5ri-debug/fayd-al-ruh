"""Explain *how* two concepts are connected, one typed link at a time.

The graph can already state a direct relation, and :mod:`core.ontology_reasoner`
can derive relations that follow from the stored ones -- but neither answers the
question a reader actually asks: **"how is porosity connected to recovery
factor?"** when there is no single edge between them. A multi-hop answer needs a
*chain*, and a chain needs to be bounded, deterministic, and honest about the
directions it walked.

This module finds the shortest path between two concepts over the ontology's
explicit relations and returns it as an ordered list of typed steps. It is
**read-only**: it never asserts the path back into the graph, never invents an
edge, and returns ``None`` when no path exists within the hop bound -- so "I
cannot connect these" is a real answer rather than a fabricated one. Each step
records whether it was walked **forward** (source → target, the stored direction)
or **backward** (target → source), so a reader sees exactly which claims the
explanation rests on.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.ontology import Ontology

#: The default hop bound. Petroleum concepts sit close together, so three hops
#: reach most real connections while keeping the search bounded and the chain
#: short enough to read.
DEFAULT_MAX_HOPS = 3


@dataclass(frozen=True)
class RelationStep:
    """One link in a connection path, with the direction it was walked."""

    source: str
    relation_type: str
    target: str
    confidence: float
    forward: bool
    """``True`` when walked source → target (the stored direction); ``False``
    when walked target → source, so the reader knows the claim's direction."""

    def describe(self) -> str:
        """The step as a directional arrow, e.g. ``porosity —increases→ permeability``."""
        if self.forward:
            return f"{self.source} —{self.relation_type}→ {self.target}"
        return f"{self.source} ←{self.relation_type}— {self.target}"


@dataclass(frozen=True)
class ConnectionPath:
    """A bounded chain of typed relations connecting two concepts."""

    source: str
    target: str
    steps: tuple[RelationStep, ...]

    @property
    def hops(self) -> int:
        return len(self.steps)

    @property
    def confidence(self) -> float:
        """The chain's confidence: the product of its steps', like a derivation.

        A longer or weaker chain is honestly less certain than a direct edge,
        which is why the product (not the minimum) is reported.
        """
        result = 1.0
        for step in self.steps:
            result *= step.confidence
        return round(result, 6)

    def describe(self) -> str:
        """The path as a single line: endpoints, hops, confidence, then each step."""
        if not self.steps:
            return f"{self.source} and {self.target} are the same concept."
        header = f"{self.source} and {self.target} are connected in {self.hops} step(s) (confidence {self.confidence:.2f}):"
        return "\n".join([header, *(f"  {index}. {step.describe()}" for index, step in enumerate(self.steps, start=1))])


def _label(ontology: Ontology, concept_id: str) -> str:
    concept = ontology.concepts.get(concept_id)
    return concept.label if concept is not None else concept_id


def find_connection(
    ontology: Ontology | None,
    source: str,
    target: str,
    *,
    max_hops: int = DEFAULT_MAX_HOPS,
) -> ConnectionPath | None:
    """The shortest typed chain between two concepts, or ``None`` if none exists.

    ``source`` and ``target`` are concept *labels* (resolved the same Arabic-aware
    way the rest of the project resolves concepts). The search is breadth-first
    over the explicit relations, walking each edge in **both** directions, so a
    chain that reaches one endpoint through an incoming relation is found too; a
    deterministic tie-break (by label then relation type) means the same question
    always yields the same path. Returns ``None`` for an unknown concept, a
    non-positive hop bound, or two concepts no chain within the bound connects.
    """
    if ontology is None or max_hops < 1:
        return None
    sources = ontology.find_concepts(source)
    targets = ontology.find_concepts(target)
    if not sources or not targets:
        return None
    start = sources[0].id
    goal = targets[0].id
    if start == goal:
        return ConnectionPath(_label(ontology, start), _label(ontology, goal), ())

    # Undirected adjacency: (neighbour_id, step-when-walked-this-way).
    adjacency: dict[str, list[tuple[str, RelationStep]]] = {}
    for relation in ontology.relations.values():
        forward = RelationStep(
            _label(ontology, relation.source_id),
            relation.relation_type,
            _label(ontology, relation.target_id),
            relation.confidence,
            True,
        )
        backward = RelationStep(
            _label(ontology, relation.target_id),
            relation.relation_type,
            _label(ontology, relation.source_id),
            relation.confidence,
            False,
        )
        adjacency.setdefault(relation.source_id, []).append((relation.target_id, forward))
        adjacency.setdefault(relation.target_id, []).append((relation.source_id, backward))
    for neighbours in adjacency.values():
        neighbours.sort(key=lambda item: (item[1].source, item[1].relation_type, item[1].target))

    # Breadth-first, so the first time the goal is reached the chain is shortest.
    parents: dict[str, tuple[str, RelationStep]] = {}
    visited = {start}
    frontier = [start]
    for _ in range(max_hops):
        next_frontier: list[str] = []
        for node in frontier:
            for neighbour, step in adjacency.get(node, ()):
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                parents[neighbour] = (node, step)
                if neighbour == goal:
                    return _reconstruct(ontology, parents, start, goal)
                next_frontier.append(neighbour)
        if not next_frontier:
            break
        frontier = next_frontier
    return None


def _reconstruct(
    ontology: Ontology,
    parents: dict[str, tuple[str, RelationStep]],
    start: str,
    goal: str,
) -> ConnectionPath:
    steps: list[RelationStep] = []
    node = goal
    while node != start:
        parent, step = parents[node]
        steps.append(step)
        node = parent
    steps.reverse()
    return ConnectionPath(_label(ontology, start), _label(ontology, goal), tuple(steps))


def describe_connection(path: ConnectionPath | None, source: str, target: str) -> str:
    """Render a connection (or its honest absence) as a readable paragraph."""
    if path is None:
        return f"No chain of relations within the hop bound connects {source} and {target}."
    return path.describe()


__all__ = [
    "DEFAULT_MAX_HOPS",
    "ConnectionPath",
    "RelationStep",
    "describe_connection",
    "find_connection",
]
