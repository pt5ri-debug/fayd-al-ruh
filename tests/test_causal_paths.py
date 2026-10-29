from core.causal_chain import CausalChain
from core.causal_explainer import CausalExplainer
from core.causal_link import CausalLink
from core.causal_paths import (
    CAUSAL_JUDGEMENTS,
    CAUSAL_RELATION_TYPES,
    DEFAULT_CAUSAL_LIMIT,
    CausalExplanation,
    CausalJudgement,
    causal_explanations,
    causal_graph_from_ontology,
    causal_reasoning_accuracy,
    describe_causal_explanations,
)
from core.cognitive_controller import CognitiveController
from core.ontology import Ontology


def _ontology() -> Ontology:
    return CognitiveController().ontology


def test_causal_relation_types_exclude_associations():
    assert "causes" in CAUSAL_RELATION_TYPES
    assert "drives" in CAUSAL_RELATION_TYPES
    # An association must never be read as a cause.
    assert "measures" not in CAUSAL_RELATION_TYPES
    assert "contains" not in CAUSAL_RELATION_TYPES
    assert "is_a" not in CAUSAL_RELATION_TYPES


def test_causal_graph_from_ontology_holds_only_causal_edges():
    graph = causal_graph_from_ontology(_ontology())
    assert graph.link_count > 0
    for link in graph.links.values():
        assert link.mechanism in CAUSAL_RELATION_TYPES


def test_causal_graph_from_ontology_none_is_empty():
    assert causal_graph_from_ontology(None).link_count == 0


def test_single_directed_causal_edge_is_found():
    explanations = causal_explanations(_ontology(), "does porosity increase permeability?")
    assert explanations
    top = explanations[0]
    assert top.cause == "porosity"
    assert top.effect == "permeability"
    assert top.hops == 1
    assert top.nodes == ("porosity", "permeability")


def test_multi_hop_causal_chain_is_walked():
    explanations = causal_explanations(_ontology(), "why does waterflood increase production?")
    chains = {(item.cause, item.effect): item for item in explanations}
    assert ("waterflood", "production") in chains
    chain = chains[("waterflood", "production")]
    assert chain.hops == 2
    assert chain.nodes == ("waterflood", "reservoir_pressure", "production")
    assert set(chain.mechanisms) == {"maintains", "drives"}


def test_causal_walk_never_reverses_an_edge():
    # `production` drives nothing into `waterflood`; a walk that reversed the
    # causal edges would fabricate this cause.
    explanations = causal_explanations(_ontology(), "production waterflood")
    assert all(item.cause != "production" or item.effect != "waterflood" for item in explanations)


def test_question_naming_one_concept_has_no_causal_block():
    assert causal_explanations(_ontology(), "what is porosity?") == ()


def test_unknown_question_has_no_causal_block():
    assert causal_explanations(_ontology(), "xyzzy plugh") == ()


def test_no_causal_chain_between_unrelated_concepts():
    # Both are real concepts, but no directed causal chain connects them.
    assert causal_explanations(_ontology(), "is_a isotope") == ()


def test_no_ontology_returns_nothing():
    assert causal_explanations(None, "porosity permeability") == ()


def test_empty_question_returns_nothing():
    assert causal_explanations(_ontology(), "   ") == ()


def test_limit_is_enforced_and_validated():
    limited = causal_explanations(_ontology(), "porosity permeability waterflood", limit=1)
    assert len(limited) <= 1
    import pytest

    with pytest.raises(ValueError):
        causal_explanations(_ontology(), "porosity permeability", limit=0)


def test_max_depth_is_validated():
    import pytest

    with pytest.raises(ValueError):
        causal_explanations(_ontology(), "porosity permeability", max_depth=0)


def test_explanations_are_ordered_strongest_first():
    explanations = causal_explanations(_ontology(), "porosity permeability waterflood")
    confidences = [item.confidence for item in explanations]
    assert confidences == sorted(confidences, reverse=True)


def test_explanations_are_deterministic():
    first = causal_explanations(_ontology(), "why does waterflood increase production?")
    second = causal_explanations(_ontology(), "why does waterflood increase production?")
    assert [item.describe() for item in first] == [item.describe() for item in second]


def test_explanation_confidence_is_the_product_of_its_links():
    explanations = causal_explanations(_ontology(), "why does waterflood increase production?")
    chain = next(item for item in explanations if item.effect == "production")
    graph = causal_graph_from_ontology(_ontology())
    expected = 1.0
    for link in graph.links.values():
        if link.cause == "waterflood" and link.effect == "reservoir_pressure":
            expected *= link.confidence
        if link.cause == "reservoir_pressure" and link.effect == "production":
            expected *= link.confidence
    assert abs(chain.confidence - round(expected, 6)) < 1e-9


def test_describe_renders_path_and_sentence():
    explanations = causal_explanations(_ontology(), "does porosity increase permeability?")
    line = explanations[0].describe()
    assert "porosity → permeability" in line
    assert "confidence" in line


def test_describe_block_is_labelled_directed():
    explanations = causal_explanations(_ontology(), "why does waterflood increase production?")
    block = describe_causal_explanations(explanations)
    assert "directed" in block.lower()
    assert "cause" in block.lower()


def test_describe_empty_block_is_blank():
    assert describe_causal_explanations(()) == ""


def test_explainer_does_not_duplicate_the_endpoints():
    chain = CausalChain((CausalLink("a", "b", 0.5), CausalLink("b", "c", 0.5)))
    sentence = CausalExplainer().explain_chain(chain)
    assert "a → b → c" in sentence
    assert "a → a" not in sentence
    assert "c → c" not in sentence


def test_explainer_single_link_reads_cleanly():
    chain = CausalChain((CausalLink("a", "b", 0.5),))
    sentence = CausalExplainer().explain_chain(chain)
    assert "a → b" in sentence
    assert "a → a" not in sentence


def test_judgement_validates_endpoints_and_hops():
    import pytest

    with pytest.raises(ValueError):
        CausalJudgement("", "b", 1)
    with pytest.raises(ValueError):
        CausalJudgement("a", "b", 0)


def test_judgement_allows_a_negative_control():
    control = CausalJudgement("a", "b", None)
    assert control.expected_hops is None


def test_causal_reasoning_accuracy_is_perfect_on_the_catalogue():
    assert causal_reasoning_accuracy(_ontology()) == 1.0


def test_causal_reasoning_accuracy_empty_catalogue_is_zero():
    assert causal_reasoning_accuracy(_ontology(), judgements=()) == 0.0


def test_causal_reasoning_accuracy_rejects_a_fabricated_chain():
    # `porosity` and `permeability measures` are associated, not causal: a walk
    # that reports a cause here must fail the gate.
    from core.causal_reasoner import CausalReasoner

    graph = causal_graph_from_ontology(_ontology())
    reasoner = CausalReasoner()
    assert reasoner.explain(graph, "porosity", "permeability measures") is None


def test_negative_control_in_catalogue():
    controls = [item for item in CAUSAL_JUDGEMENTS if item.expected_hops is None]
    assert controls, "the catalogue must carry at least one negative control"


def test_controller_loads_the_seeded_causal_edges():
    # Phase 52: the controller's own causal graph must be populated from the seed,
    # not left empty while the answer-path walk rebuilds one on every call.
    controller = CognitiveController()
    assert controller.causal_graph.link_count > 0


def test_controller_causal_accuracy_is_perfect_on_the_catalogue():
    from core.causal_paths import controller_causal_accuracy

    assert controller_causal_accuracy() == 1.0


def test_controller_causal_accuracy_empty_catalogue_is_zero():
    from core.causal_paths import controller_causal_accuracy

    assert controller_causal_accuracy(judgements=()) == 0.0


def test_controller_causal_graph_walks_a_seeded_multi_hop_chain():
    from core.causal_reasoner import CausalReasoner

    controller = CognitiveController()
    chain = CausalReasoner().explain(
        controller.causal_graph, "waterflood", "production"
    )
    assert chain is not None
    assert chain.cause == "waterflood" and chain.effect == "production"
    assert len(chain.links) == 2


def test_controller_causal_graph_holds_only_causal_edges():
    # An association (`measures`) must never appear as a link, so the loaded graph
    # cannot read a correlation as a cause.
    controller = CognitiveController()
    mechanisms = {link.mechanism for link in controller.causal_graph.links.values()}
    assert mechanisms <= CAUSAL_RELATION_TYPES
    assert "measures" not in mechanisms


def test_default_limit_is_positive():
    assert DEFAULT_CAUSAL_LIMIT >= 1


def test_assistant_surfaces_causal_paths():
    from core.assistant import build_default_assistant

    assistant = build_default_assistant()
    answer = assistant.ask("why does waterflood increase production?")
    assert answer.causal_paths
    assert any(item.effect == "production" for item in answer.causal_paths)


def test_assistant_causal_paths_empty_for_single_concept():
    from core.assistant import build_default_assistant

    assistant = build_default_assistant()
    answer = assistant.ask("what is porosity?")
    assert answer.causal_paths == ()


def test_jarvis_turn_carries_causal_lines(tmp_path, monkeypatch):
    monkeypatch.setenv("FAYD_DATA_DIR", str(tmp_path))
    from ai.jarvis import build_session

    mode = build_session(text_only=True)
    turn = mode.handle("why does waterflood increase production?", speak=False)
    assert turn.routed == "answer"
    assert turn.causal_paths
    assert any("production" in line for line in turn.causal_paths)


def test_format_turn_renders_causal_block():
    from ai.chat import format_turn
    from core.jarvis import JarvisTurn

    turn = JarvisTurn(
        heard="q",
        response="an answer",
        woke=False,
        routed="answer",
        causal_paths=("waterflood → reservoir_pressure → production (confidence 0.49)",),
    )
    rendered = format_turn(turn)
    assert "Causal paths the graph states (directed, cause → effect):" in rendered
    assert "waterflood → reservoir_pressure → production" in rendered


def test_turn_payload_exposes_causal_paths():
    from ai.chat import turn_payload
    from core.jarvis import JarvisTurn

    turn = JarvisTurn(heard="q", response="a", woke=False, routed="answer", causal_paths=("p",))
    assert turn_payload(turn)["causal_paths"] == ["p"]


def test_causal_explanation_is_frozen():
    import pytest

    explanation = CausalExplanation(
        cause="a", effect="b", hops=1, confidence=0.5, nodes=("a", "b"), mechanisms=(), sentence="s"
    )
    with pytest.raises(Exception):
        explanation.cause = "c"  # type: ignore[misc]
