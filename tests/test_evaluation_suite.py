import pytest

from evaluation.benchmark import (
    METRIC_SPECS,
    MetricSpec,
    compare_metric_dicts,
    compare_reports,
    metrics_to_dict,
    report_to_dict,
)
from evaluation.metrics import metric_names
from evaluation.runner import EvaluationMetrics, EvaluationReport, run_scenario, run_suite
from evaluation.scenarios import SCENARIOS, Domain, scenario_ids


def test_scenarios_cover_every_domain():
    domains = {scenario.domain for scenario in SCENARIOS}
    assert domains == set(Domain)
    assert len(scenario_ids()) == len(set(scenario_ids()))


def test_petroleum_scenarios_exist_and_pass():
    petroleum = [scenario for scenario in SCENARIOS if scenario.id.startswith("petroleum-")]
    assert petroleum, "expected at least one petroleum scenario"
    report = run_suite()
    results = {result.scenario_id: result for result in report.scenarios}
    for scenario in petroleum:
        assert scenario.id in results
        assert results[scenario.id].passed, scenario.id


def test_petroleum_scenarios_cover_causal_retrieval_and_ontology():
    petroleum = {scenario.id: scenario for scenario in SCENARIOS if scenario.id.startswith("petroleum-")}
    assert any(scenario.causal for scenario in petroleum.values())
    assert any(scenario.retrieval for scenario in petroleum.values())
    assert any(scenario.ontology_relations for scenario in petroleum.values())
    assert any(scenario.ontology_inferences for scenario in petroleum.values())
    assert any(scenario.ontology_retrieval for scenario in petroleum.values())
    assert any(scenario.ontology_learning for scenario in petroleum.values())
    assert any(scenario.ontology_reinforcement for scenario in petroleum.values())
    assert any(scenario.ontology_feedback for scenario in petroleum.values())
    assert any(scenario.uncertainties for scenario in petroleum.values())
    assert any(scenario.plans for scenario in petroleum.values())


def test_petroleum_ontology_inferences_are_derived_not_stored():
    from core.cognitive_controller import CognitiveController

    controller = CognitiveController()
    stored = {
        (relation.source_id, relation.target_id, relation.relation_type)
        for relation in controller.ontology.relations.values()
    }
    inferences = [
        scenario
        for scenario in SCENARIOS
        if scenario.id.startswith("petroleum-") and scenario.ontology_inferences
    ]
    assert inferences, "expected a petroleum ontology-inference scenario"
    checked = 0
    for scenario in inferences:
        for source_label, relation_type, target_label in scenario.ontology_inferences:
            sources = controller.ontology.find_concepts(source_label)
            targets = controller.ontology.find_concepts(target_label)
            assert sources and targets, (source_label, target_label)
            key = (sources[0].id, targets[0].id, relation_type)
            assert key not in stored, f"{source_label} {relation_type} {target_label} should not be stored"
            inference = controller.infer_ontology_relation(source_label, target_label, relation_type)
            assert inference is not None
            assert inference.derived is True
            assert inference.steps
            checked += 1
    assert checked >= 4


def test_controller_exposes_derived_ontology_relations():
    from core.cognitive_controller import CognitiveController

    controller = CognitiveController()
    derived = controller.derive_ontology_relations()
    assert derived
    assert all(inference.derived for inference in derived)


def test_petroleum_multihop_expectations_are_lexically_disjoint():
    from core.memory import CognitiveMemory

    memory = CognitiveMemory()
    scenarios = [s for s in SCENARIOS if s.ontology_retrieval]
    assert scenarios, "expected a petroleum multi-hop retrieval scenario"
    for scenario in scenarios:
        for expectation in scenario.ontology_retrieval:
            assert expectation.requires_ontology
            shared = memory._tokens(expectation.query) & memory._tokens(expectation.expected_top)
            assert not shared, (expectation.query, expectation.expected_top, shared)


def test_petroleum_multihop_retrieval_metric_is_reported():
    report = run_suite()
    assert report.metrics.petroleum_multihop_retrieval_accuracy == 1.0


def test_petroleum_ontology_learning_scenario_exists_and_passes():
    scenarios = [s for s in SCENARIOS if s.ontology_learning]
    assert scenarios, "expected a petroleum ontology-learning scenario"
    for scenario in scenarios:
        assert scenario.domain == Domain.PETROLEUM
        result = run_scenario(scenario)
        assert result.passed
        assert result.checks["ontology_learning_accuracy"] == 1.0
        assert result.checks["ontology_learning_forbidden"] == 0.0


def test_petroleum_ontology_learning_metric_is_reported():
    report = run_suite()
    assert report.metrics.petroleum_ontology_learning_accuracy == 1.0


def test_petroleum_ontology_reinforcement_scenario_exists_and_passes():
    scenarios = [s for s in SCENARIOS if s.ontology_reinforcement]
    assert scenarios, "expected a petroleum ontology-reinforcement scenario"
    for scenario in scenarios:
        assert scenario.domain == Domain.PETROLEUM
        result = run_scenario(scenario)
        assert result.passed
        assert result.checks["ontology_reinforcement_accuracy"] == 1.0
        assert result.checks["ontology_reinforcement_removed"] > 0


def test_petroleum_ontology_reinforcement_metric_is_reported():
    report = run_suite()
    assert report.metrics.petroleum_ontology_reinforcement_accuracy == 1.0


def test_petroleum_ontology_feedback_scenario_exists_and_passes():
    scenarios = [s for s in SCENARIOS if s.ontology_feedback]
    assert scenarios, "expected a petroleum ontology-feedback scenario"
    for scenario in scenarios:
        assert scenario.domain == Domain.PETROLEUM
        result = run_scenario(scenario)
        assert result.passed
        assert result.checks["ontology_feedback_accuracy"] == 1.0
        assert result.checks["ontology_feedback_reinforced"] > 0
        assert result.checks["ontology_feedback_removed"] > 0


def test_petroleum_ontology_feedback_metric_is_reported():
    report = run_suite()
    assert report.metrics.petroleum_ontology_feedback_accuracy == 1.0


def test_useful_recall_reinforces_the_link_that_produced_it():
    from core.cognitive_controller import CognitiveController
    from core.observation import Observation

    controller = CognitiveController()
    controller.learn_ontology_from_observation("ارتفاع الضغط يرافقه تحسن في النفاذية")
    controller.memory.store(
        "obs-permeability", "observation", Observation("permeability controls flow.", "study", 0.9), importance=0.9
    )
    recall = next(r for r in controller.recall_with_ontology("pressure") if r.linked_relation)
    before = controller.ontology_learner.relation_by_triple(*recall.linked_relation).confidence

    feedback = controller.feedback_ontology_recall(recall, useful=True, amount=0.1)

    assert feedback.outcome == "reinforced"
    assert feedback.confidence > before
    assert feedback.confidence <= controller.ontology_learner.ceiling


def test_unhelpful_recall_decays_the_link_until_the_path_is_gone():
    from core.cognitive_controller import CognitiveController
    from core.observation import Observation

    controller = CognitiveController()
    controller.learn_ontology_from_observation("ارتفاع الضغط يرافقه تحسن في النفاذية")
    controller.memory.store(
        "obs-permeability", "observation", Observation("permeability controls flow.", "study", 0.9), importance=0.9
    )
    recall = next(r for r in controller.recall_with_ontology("pressure") if r.linked_relation)

    feedback = controller.feedback_ontology_recall(recall, useful=False, amount=0.5)

    assert feedback.removed
    assert controller.recall_with_ontology("pressure") == []


def test_repetition_raises_learned_confidence_below_the_ceiling():
    from core.cognitive_controller import CognitiveController

    controller = CognitiveController()
    controller.learn_ontology_from_observation("ارتفاع الضغط في الطبقة يرافقه تحسن في النفاذية")
    initial = controller.ontology_learner.confidence

    for _ in range(5):
        controller.learn_ontology_from_observation("ارتفاع الضغط في الطبقة يرافقه تحسن في النفاذية")

    learned = [
        relation.confidence
        for relation in controller.ontology.relations.values()
        if relation.relation_type == controller.ontology_learner.relation_type
    ]
    assert learned
    assert max(learned) > initial
    assert max(learned) <= controller.ontology_learner.ceiling


def test_controller_learns_ontology_relations_from_observations():
    from core.cognitive_controller import CognitiveController

    controller = CognitiveController()
    stored_before = len(controller.ontology.relations)
    learning = controller.learn_ontology_from_observation("الضغط يرافقه النفاذية")

    assert learning.changed
    assert len(controller.ontology.relations) > stored_before
    assert controller.ontology_learner.confidence < 1.0


def test_petroleum_ontology_relations_are_seeded():
    from core.cognitive_controller import CognitiveController

    controller = CognitiveController()
    for scenario in SCENARIOS:
        if not scenario.id.startswith("petroleum-"):
            continue
        for source_label, relation_type, target_label in scenario.ontology_relations:
            sources = controller.ontology.find_concepts(source_label)
            targets = controller.ontology.find_concepts(target_label)
            assert sources and targets, (source_label, target_label)
            target_ids = {concept.id for concept in targets}
            assert any(
                relation.relation_type == relation_type and relation.target_id in target_ids
                for relation in controller.ontology.outgoing(sources[0].id)
            ), (source_label, relation_type, target_label)


def test_petroleum_domain_score_is_reported():
    report = run_suite()
    assert Domain.PETROLEUM.value in report.domain_scores
    assert 0.0 <= report.domain_scores[Domain.PETROLEUM.value] <= 1.0


def test_metric_specs_cover_every_metric():
    assert set(spec.name for spec in METRIC_SPECS) == set(metric_names(EvaluationMetrics))


def test_suite_is_deterministic():
    first = run_suite()
    second = run_suite()
    assert metrics_to_dict(first.metrics) == metrics_to_dict(second.metrics)


def test_suite_scenarios_all_pass_on_current_version():
    report = run_suite()
    failing = [result.scenario_id for result in report.scenarios if not result.passed]
    assert failing == []


def test_report_metrics_are_bounded():
    metrics = run_suite().metrics
    for name in metric_names(EvaluationMetrics):
        value = getattr(metrics, name)
        assert 0.0 <= value <= 1.0, name


def test_report_to_dict_is_json_ready():
    import json

    payload = json.dumps(report_to_dict(run_suite()))
    assert "metrics" in payload and "scenarios" in payload


def test_compare_reports_flags_no_regression_for_identical_runs():
    report = run_suite()
    regression = compare_reports(report, report)
    assert regression.passed
    assert regression.regressions == ()


def test_compare_reports_detects_lower_is_better_regression():
    baseline = {"brier_score": 0.1}
    candidate = {"brier_score": 0.2}
    regression = compare_metric_dicts(
        baseline, candidate, specs=(MetricSpec("brier_score", higher_is_better=False),)
    )
    assert not regression.passed


def test_compare_reports_tolerance_absorbs_noise():
    baseline = {"overall": 0.90}
    candidate = {"overall": 0.895}
    tolerant = compare_metric_dicts(
        baseline, candidate, specs=(MetricSpec("overall", higher_is_better=True, tolerance=0.01),)
    )
    assert tolerant.passed


def test_regression_summary_mentions_failures():
    regression = compare_metric_dicts(
        {"prediction_accuracy": 0.9},
        {"prediction_accuracy": 0.5},
        specs=(MetricSpec("prediction_accuracy", higher_is_better=True),),
    )
    assert "FAIL" in regression.summary()
    assert regression.regressions[0].name == "prediction_accuracy"


def test_metrics_are_frozen_and_comparable():
    report = run_suite()
    assert isinstance(report, EvaluationReport)
    assert isinstance(report.metrics, EvaluationMetrics)
    with pytest.raises(Exception):
        report.metrics.overall = 0.0  # type: ignore[misc]
