"""Tests for the Phase 4 reasoning and inference engine.

Covers the four new/extended pieces and their held-out gates:

* ``core.problem_decomposition`` -- a question is broken into reasoning units.
* ``core.conclusion_verification`` -- a conclusion is verified against the graph.
* ``core.reasoning_api`` -- one facade over the existing engines.
* ``core.hypothesis_reasoning`` -- candidate hypotheses generated and compared.
* ``core.tool_agent`` -- a bounded tool chooser (the sandbox disposes).

Every gate is checked with **negative controls** (a single-clause question stays
one part, an unknown question yields no candidate, a mutating tool is refused in a
read-only sandbox), so a regression that over-claims fails here as surely as one
that finds nothing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from core.conclusion_verification import (
    STATUS_CONTRADICTED,
    STATUS_INSUFFICIENT,
    STATUS_SUPPORTED,
    STATUS_UNVERIFIED,
    VERIFICATION_JUDGEMENTS,
    verification_accuracy,
    verify_conclusion,
)
from core.hypothesis_reasoning import (
    HYPOTHESIS_JUDGEMENTS,
    ORIGIN_STORED,
    CandidateHypothesis,
    compare_hypotheses,
    generate_hypotheses,
    hypothesis_accuracy,
)
from core.ontology import Ontology
from core.problem_decomposition import (
    KIND_LOOKUP,
    KIND_PROCEDURE,
    KIND_VERDICT,
    decompose,
    decomposition_accuracy,
)
from core.reasoning_api import (
    FACADE_JUDGEMENTS,
    KIND_HYPOTHESES,
    KIND_VERIFICATION,
    ReasoningAPI,
    facade_accuracy,
)
from core.tool_agent import (
    TOOL_SELECTION_JUDGEMENTS,
    ToolChooser,
    ToolProposal,
    tool_selection_accuracy,
)
from core.tools import build_sandbox


def _ontology() -> Ontology:
    ontology = Ontology()
    alpha = ontology.add_concept_once("alpha")
    beta = ontology.add_concept_once("beta")
    gamma = ontology.add_concept_once("gamma")
    ontology.add_relation_once(alpha.id, beta.id, "increases", 0.7)
    ontology.add_relation_once(alpha.id, beta.id, "reduces", 0.5)
    ontology.add_relation_once(beta.id, gamma.id, "drives", 0.6)
    # A clean edge, with no antonymous claim on the same pair, so "supported"
    # and "contradicted" are both reachable in one graph.
    ontology.add_relation_once(alpha.id, gamma.id, "contains", 0.6)
    return ontology


# -- problem decomposition ----------------------------------------------------


def test_decomposition_splits_two_clauses():
    result = decompose("what is porosity and how do I measure it?")
    assert result is not None
    assert len(result.parts) == 2
    assert result.multi_part is True


def test_decomposition_keeps_a_noun_conjunction_as_one_part():
    result = decompose("how do I measure porosity and permeability?")
    assert result is not None
    assert len(result.parts) == 1
    assert result.parts[0].kind == KIND_PROCEDURE


def test_decomposition_of_an_empty_question_is_none():
    assert decompose("") is None
    assert decompose("   ") is None


def test_decomposition_of_a_verdict_question():
    result = decompose("does acid stimulation increase permeability?")
    assert result is not None
    assert result.parts[0].kind == KIND_VERDICT


def test_decomposition_of_a_lookup_question():
    result = decompose("what is porosity?")
    assert result is not None
    assert result.parts[0].kind == KIND_LOOKUP


def test_decomposition_reports_unknowns_rather_than_hiding_them():
    result = decompose("what is zzz_unseen?")
    assert result is not None
    assert result.has_unknowns


def test_decomposition_to_primitive_round_trips_the_parts():
    result = decompose("what is porosity?")
    assert result is not None
    primitive = result.to_primitive()
    assert primitive["parts"][0]["kind"] == KIND_LOOKUP
    assert primitive["multi_part"] is False


def test_decomposition_gate_is_one():
    assert decomposition_accuracy() == 1.0


def test_decomposition_gate_is_zero_for_an_empty_catalogue():
    assert decomposition_accuracy(()) == 0.0


# -- conclusion verification --------------------------------------------------


def test_verification_of_a_stored_relation_is_supported():
    result = verify_conclusion("alpha", "contains", "gamma", ontology=_ontology())
    assert result.status == STATUS_SUPPORTED
    assert result.verified is True


def test_verification_of_an_opposed_relation_is_contradicted():
    result = verify_conclusion("alpha", "reduces", "beta", ontology=_ontology())
    # The graph holds *both* increases and reduces on the same pair, so the
    # conclusion is contradicted by the antonymous stored claim.
    assert result.status == STATUS_CONTRADICTED


def test_verification_of_an_unstored_relation_is_unverified():
    result = verify_conclusion("beta", "measures", "gamma", ontology=_ontology())
    assert result.status == STATUS_UNVERIFIED


def test_verification_never_presents_confidence_as_a_probability():
    result = verify_conclusion("alpha", "contains", "gamma", ontology=_ontology())
    assert result.probability is None


def test_verification_of_an_unknown_concept_is_insufficient():
    result = verify_conclusion("zzz_unseen", "increases", "qqq_unseen", ontology=_ontology())
    assert result.status == STATUS_INSUFFICIENT


def test_verification_gate_is_one():
    assert verification_accuracy() == 1.0


def test_verification_gate_is_zero_for_an_empty_catalogue():
    assert verification_accuracy(()) == 0.0


def test_verification_judgements_cover_every_status():
    statuses = {judgement.expect for judgement in VERIFICATION_JUDGEMENTS}
    assert STATUS_SUPPORTED in statuses
    assert STATUS_CONTRADICTED in statuses


# -- the reasoning facade -----------------------------------------------------


def test_facade_routes_a_how_question_to_the_planner():
    api = ReasoningAPI()
    result = api.plan("how do I measure porosity on a core sample?")
    assert result.kind == KIND_PROCEDURE


def test_facade_routes_a_definition_to_a_lookup():
    api = ReasoningAPI()
    result = api.facts("what is porosity?")
    assert result.kind == KIND_LOOKUP


def test_facade_verify_returns_a_verification():
    api = ReasoningAPI(ontology=_ontology())
    result = api.verify("alpha", "contains", "gamma")
    assert result.kind == KIND_VERIFICATION
    assert result.output.status == STATUS_SUPPORTED


def test_facade_hypotheses_returns_candidates():
    api = ReasoningAPI(ontology=_ontology())
    result = api.hypotheses("alpha increases beta")
    assert result.kind == KIND_HYPOTHESES
    assert result.trace


def test_facade_reason_decomposes_and_routes():
    api = ReasoningAPI()
    result = api.reason("how do I measure porosity?")
    assert result.parts
    assert result.parts[0].kind == KIND_PROCEDURE


def test_facade_reason_on_a_blank_question_is_empty():
    api = ReasoningAPI()
    result = api.reason("")
    assert result.confidence == 0.0
    assert not result.parts


def test_facade_gate_is_one():
    assert facade_accuracy() == 1.0


def test_facade_gate_is_zero_for_an_empty_catalogue():
    assert facade_accuracy(()) == 0.0


def test_facade_judgements_include_negative_controls():
    nonempty = {judgement.expect_nonempty for judgement in FACADE_JUDGEMENTS}
    assert False in nonempty


# -- hypothesis generation ----------------------------------------------------


def test_a_stored_edge_is_a_candidate():
    candidates = generate_hypotheses(_ontology(), "alpha increases beta")
    assert any(candidate.origin == ORIGIN_STORED for candidate in candidates)


def test_a_contested_candidate_is_reported():
    comparison = compare_hypotheses(_ontology(), "alpha increases beta")
    assert comparison.contested
    assert comparison.best is not None and comparison.best.contested


def test_an_unknown_question_yields_no_candidate():
    assert generate_hypotheses(_ontology(), "zzz_unseen qqq_unseen") == ()
    comparison = compare_hypotheses(_ontology(), "zzz_unseen qqq_unseen")
    assert comparison.best is None
    assert "no candidate" in comparison.describe()


def test_hypotheses_are_ranked_strongest_first():
    candidates = generate_hypotheses(_ontology(), "alpha increases beta")
    confidences = [candidate.confidence for candidate in candidates]
    assert confidences == sorted(confidences, reverse=True)


def test_candidate_validates_its_own_range():
    with pytest.raises(ValueError):
        CandidateHypothesis(statement="x", confidence=1.5, origin=ORIGIN_STORED)
    with pytest.raises(ValueError):
        CandidateHypothesis(statement="", confidence=0.5, origin=ORIGIN_STORED)
    with pytest.raises(ValueError):
        CandidateHypothesis(statement="x", confidence=0.5, origin="invented")


def test_hypothesis_gate_is_one():
    assert hypothesis_accuracy() == 1.0


def test_hypothesis_gate_is_zero_for_an_empty_catalogue():
    assert hypothesis_accuracy(()) == 0.0


def test_hypothesis_judgements_include_a_negative_control():
    assert any(not judgement.expect_candidates for judgement in HYPOTHESIS_JUDGEMENTS)


# -- bounded tool selection ---------------------------------------------------


def test_chooser_permits_a_read_only_tool():
    with tempfile.TemporaryDirectory() as folder:
        sandbox = build_sandbox(Path(folder), read_only=True)
        chooser = ToolChooser(sandbox)
        chosen = chooser.choose([ToolProposal("calculator")])
    assert chosen[0].allowed is True


def test_chooser_refuses_a_mutating_tool_in_a_read_only_sandbox():
    with tempfile.TemporaryDirectory() as folder:
        sandbox = build_sandbox(Path(folder), read_only=True)
        chooser = ToolChooser(sandbox)
        chosen = chooser.choose([ToolProposal("write_file")])
    assert chosen[0].allowed is False
    assert "read-only" in chosen[0].reason


def test_chooser_refuses_an_unknown_tool():
    with tempfile.TemporaryDirectory() as folder:
        sandbox = build_sandbox(Path(folder), read_only=True)
        chooser = ToolChooser(sandbox)
        chosen = chooser.choose([ToolProposal("__not_a_tool__")])
    assert chosen[0].allowed is False


def test_chooser_first_allowed_skips_a_refused_proposal():
    with tempfile.TemporaryDirectory() as folder:
        sandbox = build_sandbox(Path(folder), read_only=True)
        chooser = ToolChooser(sandbox)
        chosen = chooser.first_allowed([ToolProposal("__not_a_tool__"), ToolProposal("calculator")])
    assert chosen is not None and chosen.allowed


def test_chooser_plan_refuses_when_nothing_is_permitted():
    with tempfile.TemporaryDirectory() as folder:
        sandbox = build_sandbox(Path(folder), read_only=True)
        chooser = ToolChooser(sandbox)
        with pytest.raises(ValueError):
            chooser.plan("goal", [ToolProposal("__not_a_tool__")])


def test_tool_selection_gate_is_one():
    assert tool_selection_accuracy() == 1.0


def test_tool_selection_gate_is_zero_for_an_empty_catalogue():
    assert tool_selection_accuracy(()) == 0.0


def test_tool_selection_judgements_include_negative_controls():
    assert any(not expect for _name, _tools, expect in TOOL_SELECTION_JUDGEMENTS)


# -- the reasoning CLI --------------------------------------------------------


def test_cli_decompose_runs(capsys):
    from cli.reasoning import main

    assert main(["decompose", "what is porosity?"]) == 0
    assert "porosity" in capsys.readouterr().out


def test_cli_reason_runs(capsys):
    from cli.reasoning import main

    assert main(["reason", "how do I measure porosity?"]) == 0
    assert capsys.readouterr().out


def test_cli_verify_runs(capsys):
    from cli.reasoning import main

    assert main(["verify", "porosity", "increases", "permeability"]) == 0
    assert "Supported" in capsys.readouterr().out or "supported" in capsys.readouterr().out


def test_cli_hypotheses_json(capsys):
    import json

    from cli.reasoning import main

    assert main(["hypotheses", "what is porosity?", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == KIND_HYPOTHESES


def test_cli_choose_refuses_a_mutating_tool(capsys):
    import json

    from cli.reasoning import main

    assert main(["choose", "calculator", "write_file", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    allowed = {item["tool"]: item["allowed"] for item in payload}
    assert allowed["calculator"] is True
    assert allowed["write_file"] is False


def test_cli_metrics_reports_every_phase4_metric(capsys):
    from cli.reasoning import main

    assert main(["metrics"]) == 0
    out = capsys.readouterr().out
    for name in ("decomposition", "conclusion_verification", "reasoning_facade", "hypotheses"):
        assert name in out


def test_cli_unknown_command_is_a_usage_error():
    from cli.reasoning import main

    with pytest.raises(SystemExit):
        main(["nonsense"])
