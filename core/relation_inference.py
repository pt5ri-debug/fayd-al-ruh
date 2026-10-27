"""Surface the graph's *inferred* relations as evidence for an answer.

:mod:`core.relation_facts` states the edges the graph **stores**. But the graph
also *entails* edges it never stored: :mod:`core.ontology_reasoner` composes the
explicit relations through a small set of transparent rules, so ``well
penetrates reservoir`` and ``reservoir contained_in formation`` entail ``well
penetrates formation`` even though no one wrote that edge down.

Until now that reasoning never reached a reader: ``reasoning_engine`` and
``OntologyReasoner`` were reachable only from the controller, never from the
answer path, so the derived conclusions stayed invisible. This module reads the
derivations the graph entails on a question's own concepts and renders them as a
short, deterministic block that is **clearly separated** from the stored facts --
a reader must be able to tell a claim someone wrote from one the program derived.

It is **read-only** and **honest by construction**: it never asserts a derived
relation back into the graph, never invents one, and states each derivation with
the steps it rests on and its propagated confidence. The block is *always*
labelled as an inference, and a derivation the graph does not actually entail is
never shown -- so an answer that carries no inference says nothing at all.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from core.ontology import Ontology
from core.ontology_reasoner import OntologyInference, OntologyReasoner

#: How many derivations one answer states. A well-connected concept entails a
#: handful of edges; the block is evidence, not a closure dump, so it is bounded
#: and the strongest derivations are kept.
DEFAULT_INFERENCE_LIMIT = 4

#: Derivations below this confidence are not worth stating: a two-hop chain of
#: modest edges lands near 0.4, and anything weaker reads as a guess rather than
#: an inference. The threshold is transparent and deliberately inspectable.
DEFAULT_MIN_CONFIDENCE = 0.35


@dataclass(frozen=True)
class InferredRelation:
    """A relation the graph entails, with the explicit steps it rests on."""

    source: str
    relation_type: str
    target: str
    confidence: float
    steps: tuple[str, ...]

    def describe(self) -> str:
        """The derivation with its type marked, e.g. ``well —penetrates→ formation``."""
        return f"{self.source} —{self.relation_type}→ {self.target}"

    def explain(self) -> str:
        """The derivation plus the chain of stored relations that entails it."""
        if not self.steps:
            return self.describe()
        return f"{self.describe()} (from {' then '.join(self.steps)})"


def _step_phrase(ontology: Ontology, inference: OntologyInference) -> str:
    """One explicit step as ``source —type→ target`` with readable labels."""
    return " ".join(
        f"{_label(ontology, step.source_id)} —{step.relation_type}→ {_label(ontology, step.target_id)}"
        for step in inference.steps
    )


def _label(ontology: Ontology, concept_id: str) -> str:
    concept = ontology.concepts.get(concept_id)
    return concept.label if concept is not None else concept_id


def inferred_relations(
    ontology: Ontology | None,
    question: str,
    *,
    limit: int = DEFAULT_INFERENCE_LIMIT,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    reasoner: OntologyReasoner | None = None,
) -> tuple[InferredRelation, ...]:
    """The relations the graph *entails* on the question's own concepts.

    A derivation is included when **either** endpoint is a concept the question
    names, so both an inferred outgoing claim and an inferred incoming one are
    reachable. Only derivations the reasoner actually produces are returned --
    an explicit, stored relation is never re-labelled as an inference. Results are
    ordered by confidence (strongest first), then by their labels, so the same
    question always yields the same block. Returns ``()`` for no ontology, an
    empty question, a question that names no concept, or a graph whose rules
    derive nothing on those concepts -- never a placeholder.
    """
    if ontology is None or not question.strip():
        return ()
    if limit < 1:
        raise ValueError("Inference limit must be at least 1.")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("Minimum confidence must be between 0 and 1.")

    seeds = ontology.concepts_in(question)
    if not seeds:
        return ()

    engine = reasoner or OntologyReasoner()
    derived = engine.derive(ontology)
    if not derived:
        return ()

    labels = {concept_id: concept.label for concept_id, concept in ontology.concepts.items()}
    results: dict[tuple[str, str, str], InferredRelation] = {}
    for inference in derived:
        if inference.source_id not in seeds and inference.target_id not in seeds:
            continue
        if inference.confidence < min_confidence:
            continue
        source = labels.get(inference.source_id)
        target = labels.get(inference.target_id)
        if source is None or target is None:
            continue
        key = (source, inference.relation_type, target)
        candidate = InferredRelation(
            source,
            inference.relation_type,
            target,
            inference.confidence,
            (_step_phrase(ontology, inference),),
        )
        existing = results.get(key)
        if existing is None or candidate.confidence > existing.confidence:
            results[key] = candidate

    ordered = sorted(
        results.values(),
        key=lambda item: (-item.confidence, item.source, item.relation_type, item.target),
    )
    return tuple(ordered[:limit])


def describe_inferred_relations(facts: Iterable[InferredRelation]) -> str:
    """Render the derivations as a short block, or ``""`` when there are none.

    The block names itself as an *inference* on purpose: a derived relation is a
    conclusion the program drew, not a sentence a document states and not an edge
    the graph stores, and the reader must be able to tell the three apart.
    """
    present = list(facts)
    if not present:
        return ""
    lines = ["Inferred from the knowledge graph (derived, not stored):"]
    lines.extend(f"  - {fact.explain()}" for fact in present)
    return "\n".join(lines)


@dataclass(frozen=True)
class InferenceJudgement:
    """One held-out expectation about what the graph entails between two concepts."""

    source: str
    target: str
    relation_type: str
    expected: bool
    note: str = ""

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.target.strip():
            raise ValueError("An inference judgement needs two concepts.")
        if not self.relation_type.strip():
            raise ValueError("An inference judgement needs a relation type.")


#: The held-out inference catalogue -- external to the reasoner (written by hand,
#: with negative controls), so a change that makes the rules over-reach or miss a
#: derivation is a measured regression rather than a silent one.
INFERENCE_JUDGEMENTS: tuple[InferenceJudgement, ...] = (
    InferenceJudgement("well", "formation", "penetrates", True, "a well penetrates the formation holding its reservoir"),
    InferenceJudgement("hydrocarbon", "formation", "exists_in", True, "a fluid exists in the formation holding its reservoir"),
    InferenceJudgement("reservoir", "porosity", "has_property", True, "a reservoir inherits the formation's property"),
    InferenceJudgement("reservoir", "permeability", "has_property", True, "a reservoir inherits the formation's property"),
    InferenceJudgement("porosity", "permeability", "has_property", False, "no rule entails porosity having permeability"),
    InferenceJudgement("porosity", "zzz-not-a-concept", "increases", False, "an unknown endpoint entails nothing"),
)


def inference_accuracy(
    ontology: Ontology | None = None,
    judgements: tuple[InferenceJudgement, ...] = INFERENCE_JUDGEMENTS,
    *,
    reasoner: OntologyReasoner | None = None,
) -> float:
    """The share of held-out judgements the derivation gets right.

    A judgement counts when a derivation is present exactly when one is expected --
    so a fabricated inference fails as surely as a missed one. Returns ``0.0`` for
    an empty catalogue. ``ontology`` is injectable for a test; with none it builds
    the project's seeded ontology.
    """
    catalogue = tuple(judgements)
    if not catalogue:
        return 0.0
    if ontology is None:
        from core.cognitive_controller import CognitiveController

        ontology = CognitiveController().ontology
    engine = reasoner or OntologyReasoner()
    derived = {
        (inference.source_id, inference.target_id, inference.relation_type)
        for inference in engine.derive(ontology)
    }
    labels = {concept_id: concept.label.casefold() for concept_id, concept in ontology.concepts.items()}
    correct = 0
    for judgement in catalogue:
        present = any(
            labels.get(source_id) == judgement.source.casefold()
            and labels.get(target_id) == judgement.target.casefold()
            and relation_type == judgement.relation_type
            for source_id, target_id, relation_type in derived
        )
        if present == judgement.expected:
            correct += 1
    return correct / len(catalogue)


__all__ = [
    "DEFAULT_INFERENCE_LIMIT",
    "DEFAULT_MIN_CONFIDENCE",
    "INFERENCE_JUDGEMENTS",
    "InferenceJudgement",
    "InferredRelation",
    "describe_inferred_relations",
    "inference_accuracy",
    "inferred_relations",
]
