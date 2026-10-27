"""Surface the graph's *multi-hop connections* between the concepts a question names.

:mod:`core.relation_facts` states the edges the graph **stores** on a question's
concepts, and :mod:`core.relation_inference` states the edges it **entails** by
composition. Neither answers the question a reader actually asks when two of the
question's concepts have no single edge between them: **"how are porosity and
recovery factor connected?"** The project already knows how to find that chain --
:mod:`core.relation_paths` walks the explicit relations breadth-first and returns
the shortest typed path -- but that reasoning was reachable only from the
``relations connect`` CLI, never from an answer.

This module finds the shortest chain between **each pair of concepts the question
itself names** and renders them as one short, deterministic block. It is
**read-only** and **honest by construction**: it never asserts a path back into
the graph, never invents an edge, and reports each step with the direction it was
walked (a step walked backwards is shown as such), so a reader sees exactly which
claims the connection rests on. A pair no chain within the hop bound connects
produces nothing -- "they are not connected here" is a real answer, and the block
says nothing at all when the question names fewer than two concepts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from core.ontology import Ontology
from core.relation_paths import DEFAULT_MAX_HOPS, ConnectionPath, find_connection

#: How many connections one answer states. A question that names several
#: concepts has many pairs; the block is evidence, not a closure dump, so it is
#: bounded and the strongest chains are kept.
DEFAULT_CONNECTION_LIMIT = 3


@dataclass(frozen=True)
class ConceptConnection:
    """A chain of typed relations connecting two concepts the question named."""

    source: str
    target: str
    hops: int
    confidence: float
    steps: tuple[str, ...]
    """Each step rendered as a directional arrow, so a backward walk is visible."""

    def describe(self) -> str:
        """The chain on one line: endpoints, hops, then each step in order."""
        if not self.steps:
            return f"{self.source} and {self.target} are the same concept."
        header = f"{self.source} → {self.target} in {self.hops} step(s) (confidence {self.confidence:.2f}):"
        return " ".join([header, " then ".join(self.steps)])


def _concepts_in_order(ontology: Ontology, question: str) -> list[str]:
    """The concept ids the question names, strongest match first, de-duplicated.

    ``Ontology.match_concepts`` returns matches in text order with their span; a
    concept is kept once, by its first (leftmost, then longest) mention, so a
    concept named twice yields one pair rather than a duplicate.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for _start, _end, concept in ontology.match_concepts(question):
        if concept.id in seen:
            continue
        seen.add(concept.id)
        ordered.append(concept.id)
    return ordered


def concept_connections(
    ontology: Ontology | None,
    question: str,
    *,
    limit: int = DEFAULT_CONNECTION_LIMIT,
    max_hops: int = DEFAULT_MAX_HOPS,
) -> tuple[ConceptConnection, ...]:
    """The shortest typed chains between the concepts the question itself names.

    Every unordered pair of named concepts is searched with
    :func:`core.relation_paths.find_connection`, so a question that names two
    unrelated concepts is answered honestly ("no chain") by the absence of a
    connection. Results are ordered by confidence (strongest first), then by
    fewer hops, then by their labels, so the same question always yields the same
    block. Returns ``()`` for no ontology, an empty question, a question that
    names fewer than two concepts, or a set of concepts no chain within the hop
    bound connects -- never a placeholder.
    """
    if ontology is None or not question.strip():
        return ()
    if limit < 1:
        raise ValueError("Connection limit must be at least 1.")
    if max_hops < 1:
        raise ValueError("Maximum hops must be at least 1.")

    seeds = _concepts_in_order(ontology, question)
    if len(seeds) < 2:
        return ()

    labels = {concept_id: concept.label for concept_id, concept in ontology.concepts.items()}
    connections: list[ConceptConnection] = []
    for index, first in enumerate(seeds):
        for second in seeds[index + 1 :]:
            path: ConnectionPath | None = find_connection(
                ontology, labels[first], labels[second], max_hops=max_hops
            )
            if path is None or not path.steps:
                # ``find_connection`` returns an empty path for a concept paired
                # with itself; the two seeds are distinct, so a chain is required.
                continue
            connections.append(
                ConceptConnection(
                    source=path.source,
                    target=path.target,
                    hops=path.hops,
                    confidence=path.confidence,
                    steps=tuple(step.describe() for step in path.steps),
                )
            )

    ordered = sorted(
        connections,
        key=lambda item: (-item.confidence, item.hops, item.source, item.target),
    )
    return tuple(ordered[:limit])


def describe_concept_connections(connections: Iterable[ConceptConnection]) -> str:
    """Render the connections as a short block, or ``""`` when there are none.

    The block names itself as a *path through the graph* on purpose: a multi-hop
    chain is a conclusion the program walked, not a sentence a document states and
    not a single stored edge, and the reader must be able to tell the three apart.
    """
    present = list(connections)
    if not present:
        return ""
    lines = ["How the graph connects the concepts in the question (walked, not stored):"]
    lines.extend(f"  - {connection.describe()}" for connection in present)
    return "\n".join(lines)


@dataclass(frozen=True)
class ConnectionJudgement:
    """One held-out expectation about whether two concepts are connected.

    ``expected_hops`` is the number of steps the shortest chain should take, or
    ``None`` when **no** chain within the bound may be found -- a negative
    control, so an over-eager search that fabricates a link is caught, not only
    one that misses a real connection.
    """

    source: str
    target: str
    expected_hops: int | None
    note: str = ""

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.target.strip():
            raise ValueError("A connection judgement needs two concepts.")
        if self.expected_hops is not None and self.expected_hops < 1:
            raise ValueError("Expected hops must be at least 1 when given.")


#: The held-out connection catalogue -- external to the search (written by hand,
#: with negative controls), so a change that makes the walk over-reach or miss a
#: real chain is a measured regression rather than a silent one.
CONNECTION_JUDGEMENTS: tuple[ConnectionJudgement, ...] = (
    ConnectionJudgement("porosity", "permeability", 1, "a single stored edge relates them"),
    ConnectionJudgement("casing", "perforating", 2, "a two-hop chain through the well"),
    ConnectionJudgement("porosity", "recovery factor", None, "no short chain within the bound"),
    ConnectionJudgement("porosity", "zzz-not-a-concept", None, "an unknown endpoint has no chain"),
)


def connection_reasoning_accuracy(
    ontology: Ontology | None = None,
    judgements: tuple[ConnectionJudgement, ...] = CONNECTION_JUDGEMENTS,
) -> float:
    """The share of held-out judgements the connection search gets right.

    A judgement counts when a chain is found exactly when one is expected **and**
    its hop count matches -- so a fabricated link and a wrong-length chain each
    fail as surely as a miss. Returns ``0.0`` for an empty catalogue. ``ontology``
    is injectable for a test; with none it builds the project's seeded ontology.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    if ontology is None:
        from core.cognitive_controller import CognitiveController

        ontology = CognitiveController().ontology
    correct = 0
    for judgement in catalogue:
        path = find_connection(ontology, judgement.source, judgement.target)
        hops = path.hops if path is not None and path.steps else None
        if hops == judgement.expected_hops:
            correct += 1
    return correct / len(catalogue)


__all__ = [
    "CONNECTION_JUDGEMENTS",
    "DEFAULT_CONNECTION_LIMIT",
    "ConceptConnection",
    "ConnectionJudgement",
    "concept_connections",
    "connection_reasoning_accuracy",
    "describe_concept_connections",
]
