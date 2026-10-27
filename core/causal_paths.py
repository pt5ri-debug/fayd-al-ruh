"""Surface the graph's *causal chains* between the concepts a question names.

The project carries a whole causal layer -- `CausalGraph`, `CausalLink`,
`CausalChain`, `CausalReasoner`, `CausalExplainer`, `core.counterfactual` -- but
it was reachable only from the `CognitiveController` and the evaluation
scenarios, never from an answer: `CausalGraph` is empty by default and nothing in
the answer path ever consulted it. A reader asking *"why does waterflood increase
production?"* therefore got prose, never the mechanism the graph can actually
walk.

The seeded ontology, however, already states **directed causal edges**
(``causes``, ``drives``, ``controls``, ``reduces``, ``increases``, ``prevents``,
``influences``, ``affects``, ``determines``, ``governs``, ``creates``,
``produces``, ``enables``, ``improves``, ``maintains``, ``protects``) -- 117 of
them on the real graph. This module reads those edges as a causal graph and walks
the **strongest directed chain** from one concept the question names to another,
rendering it with the existing `CausalExplainer` wording, so the mechanism the
graph holds reaches the reader for the first time.

It is **directed** on purpose: a causal claim is not symmetric, so unlike the
general relation walk (`core.relation_connections`) it never traverses an edge
backwards -- "A causes B" does not license "B causes A". It is **read-only** and
**honest by construction**: it never adds a link to the `CausalGraph`, never
invents a causal edge (an association such as ``measures`` is *not* causal and is
excluded), and returns ``()`` when the question names fewer than two concepts or
no causal chain connects them -- so "the graph states no causal path here" is a
real answer.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from core.causal_chain import CausalChain
from core.causal_explainer import CausalExplainer
from core.causal_graph import CausalGraph
from core.causal_link import CausalLink
from core.ontology import Ontology

#: The ontology relation types that state a **directed causal** claim. Kept as a
#: closed set so an associative relation (`measures`, `contains`, `is_a`) can
#: never be read as causation -- the single most important honesty rule here.
CAUSAL_RELATION_TYPES: frozenset[str] = frozenset({
    "affects",
    "causes",
    "controls",
    "creates",
    "determines",
    "drives",
    "enables",
    "governs",
    "improves",
    "increases",
    "influences",
    "leads_to",
    "maintains",
    "prevents",
    "produces",
    "protects",
    "reduces",
})

#: How many causal chains one answer states. A question that names several
#: concepts has many pairs; the block is evidence, not a closure dump.
DEFAULT_CAUSAL_LIMIT = 3

#: How deep a causal chain may run. Deeper than the general relation bound because
#: a mechanism is often genuinely several steps (weight on bit → rate of
#: penetration → cost → net present value) and a chain is read as one mechanism.
DEFAULT_MAX_DEPTH = 3


@dataclass(frozen=True)
class CausalExplanation:
    """A directed causal chain connecting two concepts the question named."""

    cause: str
    effect: str
    hops: int
    confidence: float
    nodes: tuple[str, ...]
    """Every concept on the chain, from the cause to the effect inclusive."""
    mechanisms: tuple[str, ...]
    """The mechanism each link names, when it names one."""
    sentence: str
    """The rendered mechanism, produced by the project's own `CausalExplainer`."""

    def describe(self) -> str:
        """The chain as one readable line: the path, then the explainer sentence."""
        path = " → ".join(self.nodes)
        return f"{path} (confidence {self.confidence:.2f}) -- {self.sentence}"


def causal_graph_from_ontology(ontology: Ontology | None) -> CausalGraph:
    """A :class:`CausalGraph` of the ontology's **directed causal** edges.

    Only edges whose relation type is in :data:`CAUSAL_RELATION_TYPES` become
    links, so an associative relation can never be read as a cause. The link's
    confidence is the ontology relation's own confidence, and its mechanism is the
    relation type (the graph's own word for *how* the two concepts are related),
    so the rendered explanation says which claim it rests on. Returns an empty
    graph for no ontology -- never a fabricated link.
    """
    graph = CausalGraph()
    if ontology is None:
        return graph
    labels = {concept_id: concept.label for concept_id, concept in ontology.concepts.items()}
    for relation in ontology.relations.values():
        if relation.relation_type not in CAUSAL_RELATION_TYPES:
            continue
        source = labels.get(relation.source_id, relation.source_id)
        target = labels.get(relation.target_id, relation.target_id)
        if source == target:
            # A concept cannot cause itself, and `CausalLink` rejects it anyway.
            continue
        link = CausalLink(
            cause=source,
            effect=target,
            confidence=relation.confidence,
            mechanism=relation.relation_type,
        )
        if link.id in graph.links:
            continue
        graph.add_link(link)
    return graph


def _concepts_in_order(ontology: Ontology, question: str) -> list[str]:
    """The concept labels the question names, first mention first, de-duplicated."""
    labels: list[str] = []
    seen: set[str] = set()
    for _start, _end, concept in ontology.match_concepts(question):
        if concept.id in seen:
            continue
        seen.add(concept.id)
        labels.append(concept.label)
    return labels


def causal_explanations(
    ontology: Ontology | None,
    question: str,
    *,
    limit: int = DEFAULT_CAUSAL_LIMIT,
    max_depth: int = DEFAULT_MAX_DEPTH,
    reasoner=None,
) -> tuple[CausalExplanation, ...]:
    """The strongest directed causal chains between the concepts the question names.

    Each **ordered** pair of named concepts is searched in the direction the
    question's word order implies (first-mentioned → later-mentioned) *and* the
    reverse, because "how does A affect B?" and "what does B come from?" are the
    same graph question asked two ways; a chain is kept only when it runs
    forward through the causal edges. Results are ordered by confidence (strongest
    first), then fewer hops, then cause/effect, so the same question always yields
    the same block. Returns ``()`` for no ontology, an empty question, a question
    that names fewer than two concepts, or a pair no causal chain connects.
    """
    if ontology is None or not question.strip():
        return ()
    if limit < 1:
        raise ValueError("Causal limit must be at least 1.")
    if max_depth < 1:
        raise ValueError("Maximum depth must be at least 1.")

    seeds = _concepts_in_order(ontology, question)
    if len(seeds) < 2:
        return ()

    graph = causal_graph_from_ontology(ontology)
    if reasoner is None:
        from core.causal_reasoner import CausalReasoner

        reasoner = CausalReasoner()
    explainer = CausalExplainer()

    explanations: list[CausalExplanation] = []
    seen: set[tuple[str, str]] = set()
    for index, first in enumerate(seeds):
        for second in seeds[index + 1 :]:
            for cause, effect in ((first, second), (second, first)):
                chain: CausalChain | None = reasoner.explain(
                    graph, cause, effect, max_depth=max_depth
                )
                if chain is None or (chain.cause, chain.effect) in seen:
                    continue
                seen.add((chain.cause, chain.effect))
                explanations.append(
                    CausalExplanation(
                        cause=chain.cause,
                        effect=chain.effect,
                        hops=len(chain.links),
                        confidence=chain.confidence,
                        nodes=chain.nodes,
                        mechanisms=tuple(
                            link.mechanism for link in chain.links if link.mechanism
                        ),
                        sentence=explainer.explain_chain(chain),
                    )
                )

    ordered = sorted(
        explanations,
        key=lambda item: (-item.confidence, item.hops, item.cause, item.effect),
    )
    return tuple(ordered[:limit])


def describe_causal_explanations(explanations: Iterable[CausalExplanation]) -> str:
    """Render the causal chains as a short block, or ``""`` when there are none.

    The block names itself as a **directed** walk on purpose: an association is not
    a cause, and a reader must be able to tell a causal path the graph states from
    a general relation chain (`core.relation_connections`) or a stored single edge.
    """
    present = list(explanations)
    if not present:
        return ""
    lines = ["Causal paths the graph states (directed, cause → effect):"]
    lines.extend(f"  - {explanation.describe()}" for explanation in present)
    return "\n".join(lines)


@dataclass(frozen=True)
class CausalJudgement:
    """One held-out expectation about a causal chain between two concepts.

    ``expected_hops`` is the number of links the strongest chain should take, or
    ``None`` when **no** directed causal chain may be found -- a negative control,
    so a walk that reads an association as causation is caught, not only one that
    misses a real mechanism.
    """

    cause: str
    effect: str
    expected_hops: int | None
    note: str = ""

    def __post_init__(self) -> None:
        if not self.cause.strip() or not self.effect.strip():
            raise ValueError("A causal judgement needs two concepts.")
        if self.expected_hops is not None and self.expected_hops < 1:
            raise ValueError("Expected hops must be at least 1 when given.")


#: The held-out causal catalogue -- hand-written with **negative controls** (an
#: associative-only pair such as porosity/measurements, which must yield no causal
#: chain), so a change that reads an association as causation fails the gate.
CAUSAL_JUDGEMENTS: tuple[CausalJudgement, ...] = (
    CausalJudgement("porosity", "permeability", 1, "a single directed causal edge"),
    CausalJudgement("waterflood", "production", 2, "through reservoir pressure"),
    CausalJudgement("porosity", "permeability measures", None, "association only, no cause"),
)


def causal_reasoning_accuracy(
    ontology: Ontology | None = None,
    judgements: tuple[CausalJudgement, ...] = CAUSAL_JUDGEMENTS,
) -> float:
    """The share of held-out judgements the causal walk gets right.

    A judgement counts when a chain is found exactly when one is expected **and**
    its hop count matches -- so a fabricated cause and a wrong-length chain each
    fail as surely as a miss. Returns ``0.0`` for an empty catalogue. ``ontology``
    is injectable for a test; with none it builds the project's seeded ontology.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    if ontology is None:
        from core.cognitive_controller import CognitiveController

        ontology = CognitiveController().ontology
    graph = causal_graph_from_ontology(ontology)
    from core.causal_reasoner import CausalReasoner

    reasoner = CausalReasoner()
    correct = 0
    for judgement in catalogue:
        chain = reasoner.explain(graph, judgement.cause, judgement.effect)
        hops = len(chain.links) if chain is not None else None
        if hops == judgement.expected_hops:
            correct += 1
    return correct / len(catalogue)


__all__ = [
    "CAUSAL_JUDGEMENTS",
    "CAUSAL_RELATION_TYPES",
    "DEFAULT_CAUSAL_LIMIT",
    "DEFAULT_MAX_DEPTH",
    "CausalExplanation",
    "CausalJudgement",
    "causal_explanations",
    "causal_graph_from_ontology",
    "causal_reasoning_accuracy",
    "describe_causal_explanations",
]
