"""Structure mapping and analogy: relation shape, never node names."""

import pytest
from core.analogy import (
    AnalogyEngine,
    discipline_names,
    discipline_structure,
    structure_from_labels,
)
from core.cognitive_controller import CognitiveController
from core.ontology import Ontology
from core.structure_mapping import (
    Structure,
    StructureEdge,
    StructureMappingEngine,
)


def _chain_structure(name: str, labels: tuple[str, ...], relation: str = "drives") -> Structure:
    """A straight chain A -relation-> B -relation-> C ... under custom names."""
    edges = tuple(
        StructureEdge(labels[i], labels[i + 1], relation, 0.7)
        for i in range(len(labels) - 1)
    )
    return Structure(name=name, nodes=labels, edges=edges)


# ------------------------------------------------------------------ the engine


def test_identical_shape_under_different_names_maps_completely():
    """The whole point: names differ, relations match, mapping is total."""
    base = _chain_structure("base", ("alpha", "beta", "gamma"))
    target = _chain_structure("target", ("one", "two", "three"))

    mapping = StructureMappingEngine().map(base, target)

    assert mapping.mapped_nodes == {"alpha": "one", "beta": "two", "gamma": "three"}
    assert len(mapping.matched_edges) == 2
    assert mapping.relation_coverage == 1.0
    assert mapping.node_coverage == 1.0
    assert not mapping.unmapped_base_nodes
    assert not mapping.unmapped_target_nodes


def test_mapping_is_one_to_one_never_merging_two_base_nodes():
    base = _chain_structure("base", ("a", "b"))
    target = Structure(
        name="target",
        nodes=("x",),
        edges=(),
    )
    mapping = StructureMappingEngine().map(base, target)
    # Only one base node can claim the single target node.
    assert len(mapping.mapped_nodes) == 1


def test_different_relation_types_do_not_match():
    """Parallel connectivity: a relation may only map onto the same type."""
    base = Structure("base", ("a", "b"), (StructureEdge("a", "b", "drives"),))
    target = Structure("target", ("x", "y"), (StructureEdge("x", "y", "contains"),))

    mapping = StructureMappingEngine().map(base, target)

    assert mapping.is_empty
    assert mapping.relation_coverage == 0.0


def test_candidate_inference_is_a_base_relation_missing_in_the_target():
    """A shared system plus one unmatched base relation becomes a proposal."""
    base = Structure(
        "base",
        ("a", "b", "c"),
        (
            StructureEdge("a", "b", "drives"),
            StructureEdge("b", "c", "measures"),
            StructureEdge("a", "c", "predicts"),  # absent from the target
        ),
    )
    target = Structure(
        "target",
        ("x", "y", "z"),
        (
            StructureEdge("x", "y", "drives"),
            StructureEdge("y", "z", "measures"),
        ),
    )

    mapping = StructureMappingEngine().map(base, target)

    assert len(mapping.matched_edges) == 2
    inferences = {(e.source, e.relation_type, e.target) for e in mapping.candidate_inferences}
    assert inferences == {("a", "predicts", "c")}
    # The inference names the base's own concepts; the caller maps them onward.
    rows = mapping.transferred()
    assert rows and rows[0][1] == "predicts"


def test_systematicity_prefers_a_connected_system_over_isolated_facts():
    """Two matched facts that share a node beat two matched facts that do not."""
    connected_base = Structure(
        "connected",
        ("a", "b", "c"),
        (StructureEdge("a", "b", "r1"), StructureEdge("b", "c", "r2")),
    )
    connected_target = Structure(
        "connected_t",
        ("x", "y", "z"),
        (StructureEdge("x", "y", "r1"), StructureEdge("y", "z", "r2")),
    )
    scattered_base = Structure(
        "scattered",
        ("a", "b", "c", "d"),
        (StructureEdge("a", "b", "r1"), StructureEdge("c", "d", "r2")),
    )
    scattered_target = Structure(
        "scattered_t",
        ("w", "x", "y", "z"),
        (StructureEdge("w", "x", "r1"), StructureEdge("y", "z", "r2")),
    )

    engine = StructureMappingEngine()
    connected = engine.map(connected_base, connected_target)
    scattered = engine.map(scattered_base, scattered_target)

    assert connected.systematicity == 1.0
    assert scattered.systematicity == 0.0


def test_engine_terminates_on_a_large_structure_within_its_budget():
    """The state budget bounds the search, so a big structure cannot hang it."""
    nodes = tuple(f"n{i}" for i in range(40))
    edges = tuple(StructureEdge(nodes[i], nodes[i + 1], "r") for i in range(len(nodes) - 1))
    base = Structure("base", nodes, edges)
    target = Structure("target", tuple(f"m{i}" for i in range(40)), tuple(
        StructureEdge(f"m{i}", f"m{i + 1}", "r") for i in range(39)
    ))

    engine = StructureMappingEngine(max_states=5000)
    mapping = engine.map(base, target)

    assert mapping.states_visited <= 5001
    assert mapping.mapped_nodes  # it still returns its best found mapping


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        StructureEdge("", "b", "r")
    with pytest.raises(ValueError):
        StructureEdge("a", "b", "r", confidence=2.0)
    with pytest.raises(ValueError):
        Structure("s", ("a", "a"))
    with pytest.raises(ValueError):
        Structure("s", ("a",), (StructureEdge("a", "missing", "r"),))
    with pytest.raises(ValueError):
        StructureMappingEngine(max_states=0)
    with pytest.raises(ValueError):
        StructureMappingEngine(max_candidates=0)


# ------------------------------------------------------------- the domain layer


def test_structure_from_labels_keeps_only_internal_relations():
    ontology = Ontology()
    a = ontology.add_concept_once("alpha")
    b = ontology.add_concept_once("beta")
    c = ontology.add_concept_once("gamma")
    ontology.add_relation_once(a.id, b.id, "drives", 0.6)
    ontology.add_relation_once(b.id, c.id, "drives", 0.6)
    ontology.add_relation_once(a.id, c.id, "drives", 0.6)  # c is not in the group

    structure = structure_from_labels(ontology, "group", ("alpha", "beta"))

    assert structure.nodes == ("alpha", "beta")
    assert len(structure.edges) == 1
    assert structure.edges[0].key == ("alpha", "beta", "drives")


def test_discipline_structure_reads_the_seeded_relations():
    controller = CognitiveController()
    structure = discipline_structure(controller.ontology, "reservoir")
    assert structure.edges
    assert all(edge.source in structure.node_set for edge in structure.edges)


def test_unknown_discipline_is_reported_not_invented():
    controller = CognitiveController()
    with pytest.raises(KeyError):
        discipline_structure(controller.ontology, "not-a-discipline")


def test_discipline_structures_share_almost_no_relation_types():
    """An honest limit of the seeds: disciplines are relationally sparse.

    This is why concept-level analogy (below) is the usable entry point; if a
    future seed gives a discipline a richer internal structure, this test is the
    one to revisit.
    """
    controller = CognitiveController()
    shared = 0
    names = discipline_names()
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            types_a = {e.relation_type for e in discipline_structure(controller.ontology, first).edges}
            types_b = {e.relation_type for e in discipline_structure(controller.ontology, second).edges}
            shared += len(types_a & types_b)
    assert shared <= 4  # deliberately tight: the seeds really are this sparse


def test_discipline_analogy_reports_no_mapping_rather_than_inventing_one():
    """Two relationally unrelated disciplines must produce an empty, honest result."""
    controller = CognitiveController()
    analogy = controller.analogize("drilling", "reservoir")
    assert not analogy.is_meaningful
    assert analogy.mapping.relation_coverage == 0.0
    assert "unmapped" in analogy.describe() or "candidate" in analogy.describe()


def test_concept_neighbourhood_analogy_finds_a_real_structural_match():
    """porosity and permeability are embedded almost identically in the graph."""
    controller = CognitiveController()
    analogy = controller.analogize_concepts("porosity", "permeability")

    assert analogy.is_meaningful
    assert len(analogy.mapping.matched_edges) >= 5
    assert analogy.mapping.systematicity > 0.5
    # Most of the matches are literal relations, not loose role matches.
    assert analogy.mapping.exact_share > 0.5


def test_abstract_role_matching_maps_different_relation_words():
    """'drives' and 'causes' play one role, so they may correspond."""
    from core.structure_mapping import role_for

    assert role_for("drives") == role_for("causes") == "causes"
    assert role_for("measures") == role_for("estimates")
    # An unclassified relation keeps its own name, so it only matches itself.
    assert role_for("some_unknown_relation") == "some_unknown_relation"


def test_analogous_concepts_are_ranked_and_meaningful():
    controller = CognitiveController()
    ranked = controller.analogous_concepts("porosity", limit=5)
    assert ranked
    for analogy in ranked:
        assert analogy.is_meaningful
        assert analogy.mapping.base != "porosity"


def test_analogy_is_never_asserted_into_the_ontology():
    """An analogy proposes; it must not add a single relation to the graph."""
    controller = CognitiveController()
    before = len(controller.ontology.relations)
    analogy = controller.analogize_concepts("porosity", "permeability")
    analogy.mapping.transferred()
    assert len(controller.ontology.relations) == before


def test_describe_reports_the_gaps_not_only_the_matches():
    controller = CognitiveController()
    text = controller.analogize_concepts("porosity", "permeability").describe()
    assert "relation coverage" in text
    assert "candidate inferences" in text or "unmapped" in text


def test_neighbourhood_unknown_concept_is_reported():
    controller = CognitiveController()
    with pytest.raises(KeyError):
        controller.analogize_concepts("porosity", "not-a-concept")


def test_engine_rejects_an_unknown_ranking_limit():
    controller = CognitiveController()
    with pytest.raises(ValueError):
        AnalogyEngine(controller.ontology).rank_for("reservoir", limit=0)
    with pytest.raises(ValueError):
        AnalogyEngine(controller.ontology).neighbours_of("porosity", limit=0)
