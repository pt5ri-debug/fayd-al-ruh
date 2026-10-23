"""Reasoning by analogy over the project's own concept graph.

:mod:`core.structure_mapping` is the engine; this module is the domain half. It
reads the seeded ontology (disciplines, curriculum, petroleum corpus) and builds
a :class:`~core.structure_mapping.Structure` for any group of concepts -- a
discipline, a course, or an arbitrary set of labels -- so two of them can be
aligned by their relations rather than by their vocabulary.

The payoff is transfer: when the internal relation structure of one discipline
matches another, the base's relations that the target does not already have
become **candidate inferences** for the target. A reservoir's pressure driving
production and a pipeline's pressure driving flow share a shape, so what is known
about one is a hypothesis about the other.

Two rules keep this honest:

* the analogy is reported with what it could *not* map, never only its matches;
* every transfer is a **candidate** -- it is never asserted into the ontology, and
  its confidence is discounted by how much of the structure actually aligned, so
  a shape shared by two facts does not read like a proven law.

This is Gentner-style structure mapping over the project's knowledge, not a
similarity score over words.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.domain_knowledge import disciplines
from core.ontology import Ontology
from core.structure_mapping import (
    MatchedEdge,
    Structure,
    StructureEdge,
    StructureMapping,
    StructureMappingEngine,
    mapping_score,
    role_for,
)


@dataclass(frozen=True)
class Analogy:
    """A structure mapping between two named groups of the project's concepts."""

    mapping: StructureMapping
    base_nodes: tuple[str, ...]
    target_nodes: tuple[str, ...]

    @property
    def score(self) -> float:
        return mapping_score(self.mapping)

    @property
    def substantive_matches(self) -> tuple[MatchedEdge, ...]:
        """Literal matches that carry analogical signal, i.e. not pure taxonomy.

        ``is_a`` (and its close synonyms) says two things are both *kinds* of
        something, which is true of almost any pair of concepts, so a shared
        taxonomy is not evidence of a shared shape. Only a literal match on a
        non-taxonomic relation is treated as substantive.
        """
        return tuple(
            match
            for match in self.mapping.exact_matches
            if role_for(match.base.relation_type) != "is_a"
        )

    @property
    def is_meaningful(self) -> bool:
        """A single shared fact is not an analogy; a shared *system* is.

        The system must also rest on at least one **substantive literal** relation
        match -- a literal match on a non-taxonomic relation. An alignment built
        only from role abstractions is evidence that the role table is broad, not
        that the two structures share a shape, and a shared ``is_a`` is true of
        almost any pair of concepts, so neither is called an analogy.
        """
        return (
            len(self.mapping.matched_edges) >= 2
            and self.mapping.systematicity > 0.0
            and bool(self.substantive_matches)
        )

    def describe(self) -> str:
        return self.mapping.describe()


def structure_from_labels(
    ontology: Ontology,
    name: str,
    labels: tuple[str, ...],
) -> Structure:
    """Build a structure from concept labels, using their relations in ``ontology``.

    Only relations whose *both* endpoints are in the group are kept, so the
    structure is the group's internal relational shape. Unknown labels are
    ignored rather than invented.
    """
    resolved: dict[str, str] = {}
    for label in labels:
        matches = ontology.find_concepts(label)
        if matches:
            resolved[label] = matches[0].id
    ids = set(resolved.values())
    label_of = {concept_id: label for label, concept_id in resolved.items()}

    edges: list[StructureEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for relation in ontology.relations.values():
        if relation.source_id not in ids or relation.target_id not in ids:
            continue
        source = label_of[relation.source_id]
        target = label_of[relation.target_id]
        key = (source, target, relation.relation_type)
        if key in seen:
            continue
        seen.add(key)
        edges.append(StructureEdge(source, target, relation.relation_type, relation.confidence))

    return Structure(name=name, nodes=tuple(sorted(resolved)), edges=tuple(edges))


def discipline_structure(ontology: Ontology, key: str) -> Structure:
    """The internal relational structure of one seeded discipline."""
    for discipline in disciplines():
        if discipline.key == key:
            labels = tuple(label for label, _ in discipline.concepts)
            return structure_from_labels(ontology, discipline.label, labels)
    raise KeyError(f"Unknown discipline: {key}")


def discipline_names() -> tuple[str, ...]:
    return tuple(discipline.key for discipline in disciplines())


def neighbourhood_structure(ontology: Ontology, label: str, hops: int = 1) -> Structure:
    """The relational neighbourhood of one concept, up to ``hops`` relations away.

    A single concept is too small to be a structure, and a whole discipline's seed
    is often too sparse. A concept's *neighbourhood* is the natural middle scale:
    it is the local relational system a person reasons over when they say one
    thing is "like" another, and it is where the project's ontology actually
    carries enough relations for a structural comparison to mean something.
    """
    if hops < 0:
        raise ValueError("Hops must not be negative.")
    matches = ontology.find_concepts(label)
    if not matches:
        raise KeyError(f"Unknown concept: {label}")
    id_to_label = {concept.id: concept.label for concept in ontology.concepts.values()}

    frontier = {concept.id for concept in matches}
    reached = set(frontier)
    for _ in range(hops):
        neighbours: set[str] = set()
        for relation in ontology.relations.values():
            if relation.source_id in frontier:
                neighbours.add(relation.target_id)
            if relation.target_id in frontier:
                neighbours.add(relation.source_id)
        frontier = neighbours - reached
        reached |= neighbours

    labels = tuple(sorted(id_to_label[concept_id] for concept_id in reached if concept_id in id_to_label))
    return structure_from_labels(ontology, label, labels)


class AnalogyEngine:
    """Finds and ranks analogies between groups of the project's concepts."""

    def __init__(
        self,
        ontology: Ontology,
        engine: StructureMappingEngine | None = None,
    ) -> None:
        self.ontology = ontology
        self.engine = engine or StructureMappingEngine()

    def between(self, base: Structure, target: Structure) -> Analogy:
        """Align two structures and return the analogy with its honest gaps."""
        mapping = self.engine.map(base, target)
        return Analogy(mapping, base.nodes, target.nodes)

    def between_disciplines(self, base_key: str, target_key: str) -> Analogy:
        base = discipline_structure(self.ontology, base_key)
        target = discipline_structure(self.ontology, target_key)
        return self.between(base, target)

    def between_concepts(self, base_label: str, target_label: str, hops: int = 1) -> Analogy:
        """Align two concepts by their relational neighbourhoods.

        This is the analogy that works on the project's real knowledge: the
        neighbourhoods carry enough relations for a structural comparison, so
        "porosity is like permeability" is answered by how the two are embedded in
        the graph, not by how their names look.
        """
        base = neighbourhood_structure(self.ontology, base_label, hops)
        target = neighbourhood_structure(self.ontology, target_label, hops)
        return self.between(base, target)

    def neighbours_of(self, label: str, limit: int = 5) -> tuple[Analogy, ...]:
        """The concepts whose neighbourhood is most analogous to ``label``'s.

        Candidates are drawn from the concepts that share a relation with
        ``label``, so the search is over genuinely related ideas rather than every
        concept in the ontology. A candidate is kept only when its mapping is
        meaningful, so the list is never padded with noise.
        """
        if limit < 1:
            raise ValueError("Limit must be at least 1.")
        target = neighbourhood_structure(self.ontology, label)
        id_to_label = {concept.id: concept.label for concept in self.ontology.concepts.values()}
        origin = {concept.id for concept in self.ontology.find_concepts(label)}
        adjacent: set[str] = set()
        for relation in self.ontology.relations.values():
            if relation.source_id in origin:
                adjacent.add(relation.target_id)
            if relation.target_id in origin:
                adjacent.add(relation.source_id)
        adjacent -= origin

        ranked: list[Analogy] = []
        for concept_id in adjacent:
            other = id_to_label.get(concept_id)
            if not other:
                continue
            analogy = self.between(neighbourhood_structure(self.ontology, other), target)
            if analogy.is_meaningful:
                ranked.append(analogy)
        ranked.sort(key=lambda item: (-item.score, item.mapping.base))
        return tuple(ranked[:limit])

    def rank_for(self, target_key: str, limit: int = 5) -> tuple[Analogy, ...]:
        """Every discipline ranked as an analogy for ``target_key``.

        A discipline is only a candidate if its mapping is meaningful -- a shared
        *system* of relations, not one incidental fact -- so the ranking cannot be
        padded with noise.
        """
        if limit < 1:
            raise ValueError("Limit must be at least 1.")
        target = discipline_structure(self.ontology, target_key)
        ranked: list[Analogy] = []
        for discipline in disciplines():
            if discipline.key == target_key:
                continue
            analogy = self.between(discipline_structure(self.ontology, discipline.key), target)
            if analogy.is_meaningful:
                ranked.append(analogy)
        ranked.sort(key=lambda item: (-item.score, item.mapping.base))
        return tuple(ranked[:limit])

    def transfer_to_discipline(self, base_key: str, target_key: str) -> Analogy:
        """The analogy whose candidate inferences are new statements for the target."""
        return self.between_disciplines(base_key, target_key)


@dataclass(frozen=True)
class AnalogyJudgement:
    """One held-out check: whether ``base ~ target`` should read as an analogy.

    ``expected`` is the verdict an honest engine must reach. The catalogue mixes
    positive cases (concepts the project's graph embeds the same way) with
    **negative controls** (concepts that share a neighbourhood but are not
    structurally analogous), so a scorer cannot pass by answering "yes" to
    everything.
    """

    base: str
    target: str
    expected: bool
    reason: str


#: Held-out analogy judgements. These are external to the engine -- they are not
#: derived from a mapping -- and the negative controls make a permissive engine
#: fail. They are the analogy counterpart of the acceptance gate.
ANALOGY_JUDGEMENTS: tuple[AnalogyJudgement, ...] = (
    AnalogyJudgement(
        "porosity", "permeability", True,
        "both are rock properties measured by logs and core and affecting flow",
    ),
    AnalogyJudgement(
        "density_log", "sonic_log", True,
        "both are well logs of the same shape: a tool measuring a rock property",
    ),
    AnalogyJudgement(
        "well_log", "core_sample", True,
        "both are measuring instruments of rock properties",
    ),
    AnalogyJudgement(
        "porosity", "waterflood", False,
        "a rock property and an operation share no relational system",
    ),
    AnalogyJudgement(
        "gas_kick", "recovery_factor", False,
        "a hazard and a ratio are not structurally analogous",
    ),
    AnalogyJudgement(
        "cement", "permeability", False,
        "a material and a rock property have no shared relation shape",
    ),
)


def analogy_accuracy(hops: int = 1, ontology: Ontology | None = None) -> float:
    """The share of the held-out analogy judgements the engine gets right.

    A judgement counts when the engine's verdict (``is_meaningful``) matches the
    expected one, so both a missed analogy and an invented one cost the score. An
    unknown concept counts against the score too, rather than being skipped, so a
    catalogue entry that stops resolving is reported instead of hidden. Returns
    0.0 for an empty catalogue so a caller can tell "no check" from "a perfect
    check". This is a real held-out measure: the catalogue is external to the
    engine and includes negative controls.

    ``ontology`` is injectable for a test; with none it builds the project's
    seeded ontology, importing the controller lazily because the controller
    imports this module.
    """
    if not ANALOGY_JUDGEMENTS:
        return 0.0
    if ontology is None:
        from core.cognitive_controller import CognitiveController

        ontology = CognitiveController().ontology
    engine = AnalogyEngine(ontology)
    correct = 0
    for judgement in ANALOGY_JUDGEMENTS:
        try:
            analogy = engine.between_concepts(judgement.base, judgement.target, hops)
        except KeyError:
            continue
        if analogy.is_meaningful == judgement.expected:
            correct += 1
    return correct / len(ANALOGY_JUDGEMENTS)


__all__ = [
    "ANALOGY_JUDGEMENTS",
    "Analogy",
    "AnalogyEngine",
    "AnalogyJudgement",
    "analogy_accuracy",
    "discipline_names",
    "discipline_structure",
    "neighbourhood_structure",
    "structure_from_labels",
]
