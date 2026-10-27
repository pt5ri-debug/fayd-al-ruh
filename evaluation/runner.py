from __future__ import annotations

from dataclasses import dataclass, field

from core.acceptance_policy import rejection_reason
from core.action import Action
from core.analogy import analogy_accuracy
from core.belief import Belief
from core.belief_revision import BeliefRevisionEngine
from core.causal_link import CausalLink
from core.cognitive_controller import CognitiveController
from core.curriculum_knowledge import curriculum_coverage_score
from core.embeddings import evaluate_held_out
from core.evidence import Evidence
from core.goal import Goal
from core.observation import Observation
from core.relation_consistency import consistency_accuracy
from core.relation_extraction import extraction_accuracy
from core.relation_inference import inference_accuracy
from core.relation_paths import connection_accuracy
from core.relation_polarity import polarity_accuracy
from core.state import State
from ml.petroleum import petroleum_generalisation_accuracy, petroleum_learning_accuracy

from evaluation.metrics import (
    CalibrationBin,
    brier_summary,
    calibration_bins,
    clamp,
    expected_calibration_error,
    mean,
)
from evaluation.scenarios import (
    SCENARIOS,
    DecisionActionSpec,
    Domain,
    Scenario,
)


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    domain: str
    passed: bool
    checks: dict[str, float] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationMetrics:
    # Prediction quality
    prediction_accuracy: float
    brier_score: float
    calibration_error: float
    reliability: float
    resolution: float
    # Consistency
    contradiction_rate: float
    # Retrieval
    retrieval_accuracy: float
    retrieval_mrr: float
    # Planning
    planning_success_rate: float
    # Explanation
    decision_explanation_quality: float
    # Causal
    causal_chain_accuracy: float
    causal_explanation_quality: float
    # Petroleum domain
    petroleum_causal_accuracy: float
    petroleum_retrieval_accuracy: float
    petroleum_ontology_accuracy: float
    petroleum_inference_accuracy: float
    petroleum_multihop_retrieval_accuracy: float
    petroleum_ontology_learning_accuracy: float
    petroleum_ontology_reinforcement_accuracy: float
    petroleum_ontology_feedback_accuracy: float
    # Petroleum learning across disciplines (trained tasks, held-out success)
    petroleum_learning_accuracy: float
    # Held-out disciplines never trained on (a genuine generalisation check)
    petroleum_generalisation_accuracy: float
    # Acceptance gate: held-out judgements the policy must get right
    acceptance_gate_accuracy: float
    # Curriculum: fraction of the seeded degree the ontology carries
    curriculum_coverage: float
    # Analogy: held-out judgements of which concept pairs are structurally alike
    analogy_accuracy: float
    # Relation extraction: held-out sentences, including negative controls
    relation_extraction_accuracy: float
    polarity_accuracy: float
    connection_accuracy: float
    consistency_accuracy: float
    inference_accuracy: float
    semantic_separation: float
    # Aggregate
    overall: float


@dataclass(frozen=True)
class EvaluationReport:
    metrics: EvaluationMetrics
    scenarios: tuple[ScenarioResult, ...]
    domain_scores: dict[str, float]
    calibration: tuple[CalibrationBin, ...] = ()


def _actions(specs: tuple[DecisionActionSpec, ...]) -> list[Action]:
    return [
        Action(spec.name, spec.preconditions, spec.add_effects, spec.remove_effects, spec.cost)
        for spec in specs
    ]


def _retrieval_scores(controller: CognitiveController, scenario: Scenario) -> tuple[float, float, int]:
    hits = 0
    reciprocal_ranks: list[float] = []
    for expectation in scenario.retrieval:
        recalls = controller.memory.recall(
            expectation.query, item_types=("observation",), limit=5
        )
        texts = [str(getattr(recall.item.content, "content", recall.item.content)) for recall in recalls]
        rank = texts.index(expectation.expected_top) + 1 if expectation.expected_top in texts else 0
        if rank == 1:
            hits += 1
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    total = len(scenario.retrieval)
    return (hits / total if total else 0.0), mean(reciprocal_ranks), hits


def _ontology_accuracy(controller: CognitiveController, scenario: Scenario) -> float:
    """Fraction of expected seeded relations present in the controller ontology."""
    if not scenario.ontology_relations:
        return 0.0
    matched = 0
    for source_label, relation_type, target_label in scenario.ontology_relations:
        sources = controller.ontology.find_concepts(source_label)
        targets = controller.ontology.find_concepts(target_label)
        if not sources or not targets:
            continue
        target_ids = {concept.id for concept in targets}
        if any(
            relation.relation_type == relation_type and relation.target_id in target_ids
            for relation in controller.ontology.outgoing(sources[0].id)
        ):
            matched += 1
    return matched / len(scenario.ontology_relations)


def _ontology_inference_accuracy(controller: CognitiveController, scenario: Scenario) -> float:
    """Fraction of expected relations the ontology derives without storing them.

    A relation only counts when it is genuinely inferred: it must not already be
    stored explicitly, and it must carry at least one composing step.
    """
    if not scenario.ontology_inferences:
        return 0.0
    stored = {
        (relation.source_id, relation.target_id, relation.relation_type)
        for relation in controller.ontology.relations.values()
    }
    matched = 0
    for source_label, relation_type, target_label in scenario.ontology_inferences:
        sources = controller.ontology.find_concepts(source_label)
        targets = controller.ontology.find_concepts(target_label)
        if not sources or not targets:
            continue
        key = (sources[0].id, targets[0].id, relation_type)
        if key in stored:
            continue
        inference = controller.infer_ontology_relation(source_label, target_label, relation_type)
        if inference is not None and inference.derived and inference.steps:
            matched += 1
    return matched / len(scenario.ontology_inferences)


def _ontology_retrieval_accuracy(controller: CognitiveController, scenario: Scenario) -> float:
    """Fraction of expectations reached only by following ontology relations.

    Each marked expectation must be lexically disjoint from its query and then
    hit at rank 1 by ontology-guided recall, proving the relation walk adds
    reach rather than echoing the lexical result.
    """
    if not scenario.ontology_retrieval:
        return 0.0
    hits = 0
    for expectation in scenario.ontology_retrieval:
        query_tokens = controller.memory._tokens(expectation.query)
        expected_tokens = controller.memory._tokens(expectation.expected_top)
        if query_tokens & expected_tokens:
            continue
        guided = controller.recall_with_ontology(expectation.query, limit=5)
        guided_texts = [str(getattr(recall.item.content, "content", recall.item.content)) for recall in guided]
        if guided_texts and guided_texts[0] == expectation.expected_top:
            hits += 1
    return hits / len(scenario.ontology_retrieval)


def _run_memory(scenario: Scenario) -> ScenarioResult:
    controller = CognitiveController()
    for index, spec in enumerate(scenario.observations):
        controller.memory.store(
            f"{scenario.id}-obs-{index}",
            "observation",
            Observation(spec.content, spec.source, spec.confidence),
            importance=spec.confidence,
        )

    accuracy, mrr, hits = _retrieval_scores(controller, scenario)
    return ScenarioResult(
        scenario.id,
        scenario.domain.value,
        passed=accuracy == 1.0,
        checks={"retrieval_accuracy": round(accuracy, 6), "retrieval_mrr": round(mrr, 6)},
        details={"expected": str(len(scenario.retrieval)), "top1_hits": str(hits)},
    )


def _run_consistency(scenario: Scenario) -> ScenarioResult:
    controller = CognitiveController()
    for spec in scenario.beliefs:
        controller.state.add_belief(Belief(spec.statement, spec.confidence))
    report = controller.check_consistency()
    detected = len(report.conflicts)
    matched = 1.0 if detected == scenario.expected_conflicts else 0.0
    return ScenarioResult(
        scenario.id,
        scenario.domain.value,
        passed=matched == 1.0,
        checks={"consistency_match": matched},
        details={"detected": str(detected), "expected": str(scenario.expected_conflicts)},
    )


def _run_causal(scenario: Scenario) -> ScenarioResult:
    controller = CognitiveController()
    chain_hits = 0
    explanation_scores: list[float] = []

    for spec in scenario.causal:
        controller.add_causal_link(CausalLink(spec.cause, spec.effect, spec.confidence))

    for spec in scenario.causal:
        chain = controller.explain_cause(spec.cause, spec.effect)
        if chain is not None and chain.effect == spec.effect and chain.cause == spec.cause:
            chain_hits += 1
            explanation_scores.append(1.0 if chain.confidence > 0 else 0.0)
        else:
            explanation_scores.append(0.0)

    total = len(scenario.causal)
    accuracy = chain_hits / total if total else 0.0
    quality = mean(explanation_scores)
    return ScenarioResult(
        scenario.id,
        scenario.domain.value,
        passed=accuracy == 1.0,
        checks={
            "causal_chain_accuracy": round(accuracy, 6),
            "causal_explanation_quality": round(quality, 6),
        },
        details={"chains_found": str(chain_hits), "expected": str(total)},
    )


def _run_planning(scenario: Scenario) -> ScenarioResult:
    controller = CognitiveController()
    successes = 0
    for spec in scenario.plans:
        goal = Goal(spec.goal, spec.desired_facts)
        plan = controller.planner.create_plan(
            State(spec.initial_facts), goal, _actions(spec.actions), max_depth=spec.max_depth
        )
        if plan is not None and plan.predicted_final_state.satisfies(spec.desired_facts):
            successes += 1
    total = len(scenario.plans)
    rate = successes / total if total else 0.0
    return ScenarioResult(
        scenario.id,
        scenario.domain.value,
        passed=rate == 1.0,
        checks={"planning_success_rate": round(rate, 6)},
        details={"plans_solved": str(successes), "expected": str(total)},
    )


def _run_decisions(scenario: Scenario) -> ScenarioResult:
    controller = CognitiveController()
    scores: list[float] = []
    for spec in scenario.decisions:
        state = State(spec.initial_facts)
        goal = Goal(spec.goal, spec.desired_facts)
        trace = controller.decide(state, goal, _actions(spec.actions))
        selected_correct = trace.selected is not None and trace.selected.action.name == spec.expected_selected
        explanation = controller.explain(spec.goal, decision_trace=trace)
        explained = bool(explanation.decision) and spec.expected_selected in " ".join(explanation.decision)
        scores.append(0.5 * float(selected_correct) + 0.5 * float(explained))
    quality = mean(scores)
    return ScenarioResult(
        scenario.id,
        scenario.domain.value,
        passed=quality == 1.0,
        checks={"decision_explanation_quality": round(quality, 6)},
        details={"decisions": str(len(scenario.decisions))},
    )


def _run_uncertainty(scenario: Scenario) -> ScenarioResult:
    engine = BeliefRevisionEngine()
    errors: list[float] = []
    for spec in scenario.uncertainties:
        belief = Belief(spec.statement, spec.prior_confidence)
        evidence_items = [
            Evidence(item.content, item.source, item.supports, item.confidence)
            for item in spec.evidence
        ]
        revision = engine.revise(belief, evidence_items)
        errors.append(abs(revision.revised.confidence - spec.true_probability))
    calibration_error = mean(errors)
    return ScenarioResult(
        scenario.id,
        scenario.domain.value,
        passed=calibration_error <= 0.35,
        checks={"uncertainty_error": round(calibration_error, 6)},
        details={"cases": str(len(scenario.uncertainties))},
    )


def _run_reasoning(scenario: Scenario) -> ScenarioResult:
    forecasts = [
        (spec.confidence, 1.0 if spec.outcome >= 0.5 else 0.0)
        for spec in scenario.predictions
    ]
    summary = brier_summary(forecasts)
    calibration_error = expected_calibration_error(forecasts)
    hits = sum(
        (spec.confidence >= 0.5) == (spec.outcome >= 0.5)
        for spec in scenario.predictions
    )
    accuracy = hits / len(scenario.predictions) if scenario.predictions else 0.0
    return ScenarioResult(
        scenario.id,
        scenario.domain.value,
        passed=calibration_error <= 0.25,
        checks={
            "prediction_accuracy": round(accuracy, 6),
            "brier_score": summary.brier,
            "calibration_error": calibration_error,
        },
        details={"forecasts": str(summary.count)},
    )


def _run_petroleum_retrieval(scenario: Scenario) -> ScenarioResult:
    """Petroleum scenario mixing log retrieval and seeded-ontology grounding."""
    controller = CognitiveController()
    for index, spec in enumerate(scenario.observations):
        controller.memory.store(
            f"{scenario.id}-obs-{index}",
            "observation",
            Observation(spec.content, spec.source, spec.confidence),
            importance=spec.confidence,
        )

    retrieval_accuracy, mrr, hits = _retrieval_scores(controller, scenario)
    ontology_accuracy = _ontology_accuracy(controller, scenario)
    passed = retrieval_accuracy == 1.0 and ontology_accuracy == 1.0
    return ScenarioResult(
        scenario.id,
        scenario.domain.value,
        passed=passed,
        checks={
            "retrieval_accuracy": round(retrieval_accuracy, 6),
            "retrieval_mrr": round(mrr, 6),
            "ontology_accuracy": round(ontology_accuracy, 6),
        },
        details={"top1_hits": str(hits), "ontology_relations": str(len(scenario.ontology_relations))},
    )


def _run_petroleum_ontology(scenario: Scenario) -> ScenarioResult:
    """Petroleum scenario asserting the seeded ontology carries the identity frame."""
    controller = CognitiveController()
    ontology_accuracy = _ontology_accuracy(controller, scenario)
    inference_accuracy = _ontology_inference_accuracy(controller, scenario)
    passed = ontology_accuracy == 1.0
    checks = {"ontology_accuracy": round(ontology_accuracy, 6)}
    if scenario.ontology_inferences:
        checks["ontology_inference_accuracy"] = round(inference_accuracy, 6)
        passed = passed and inference_accuracy == 1.0
    return ScenarioResult(
        scenario.id,
        scenario.domain.value,
        passed=passed,
        checks=checks,
        details={"expected_relations": str(len(scenario.ontology_relations))},
    )


def _run_petroleum_inference(scenario: Scenario) -> ScenarioResult:
    """Petroleum scenario asserting relations are derived, not merely stored."""
    controller = CognitiveController()
    inference_accuracy = _ontology_inference_accuracy(controller, scenario)
    return ScenarioResult(
        scenario.id,
        scenario.domain.value,
        passed=inference_accuracy == 1.0,
        checks={"ontology_inference_accuracy": round(inference_accuracy, 6)},
        details={"expected_inferences": str(len(scenario.ontology_inferences))},
    )


_RUNNERS = {
    Domain.MEMORY: _run_memory,
    Domain.REASONING: _run_reasoning,
    Domain.CAUSAL: _run_causal,
    Domain.PLANNING: _run_planning,
    Domain.UNCERTAINTY: _run_uncertainty,
    Domain.PETROLEUM: _run_causal,
}


def _run_petroleum_multihop_retrieval(scenario: Scenario) -> ScenarioResult:
    """Petroleum scenario: retrieval follows derived relations, not shared words."""
    controller = CognitiveController()
    for index, spec in enumerate(scenario.observations):
        controller.memory.store(
            f"{scenario.id}-obs-{index}",
            "observation",
            Observation(spec.content, spec.source, spec.confidence),
            importance=spec.confidence,
        )

    lexical_accuracy, _, _ = _retrieval_scores(controller, scenario)
    ontology_accuracy = _ontology_retrieval_accuracy(controller, scenario)
    passed = lexical_accuracy == 1.0 and ontology_accuracy == 1.0
    return ScenarioResult(
        scenario.id,
        scenario.domain.value,
        passed=passed,
        checks={
            "retrieval_accuracy": round(lexical_accuracy, 6),
            "ontology_retrieval_accuracy": round(ontology_accuracy, 6),
        },
        details={"ontology_expectations": str(len(scenario.ontology_retrieval))},
    )


def _run_petroleum_ontology_learning(scenario: Scenario) -> ScenarioResult:
    """Petroleum scenario: observations teach weak relations without overwriting the seed."""
    controller = CognitiveController()
    expected_total = 0
    expected_hits = 0
    forbidden_violations = 0
    learned_relations = 0
    for spec in scenario.ontology_learning:
        learning = controller.learn_ontology_from_observation(spec.text)
        learned_relations += len(learning.added)
        for source_label, target_label in spec.expected_links:
            expected_total += 1
            if _has_relation(controller, source_label, target_label, "associated_with"):
                expected_hits += 1
        for source_label, target_label in spec.forbidden_links:
            if _has_relation(controller, source_label, target_label, "associated_with"):
                forbidden_violations += 1

    learning_accuracy = expected_hits / expected_total if expected_total else 0.0
    passed = learning_accuracy == 1.0 and forbidden_violations == 0 and learned_relations > 0
    return ScenarioResult(
        scenario.id,
        scenario.domain.value,
        passed=passed,
        checks={
            "ontology_learning_accuracy": round(learning_accuracy, 6),
            "ontology_learning_forbidden": float(forbidden_violations),
        },
        details={"learned_relations": str(learned_relations)},
    )


def _run_petroleum_ontology_reinforcement(scenario: Scenario) -> ScenarioResult:
    """Petroleum scenario: repetition strengthens learned links, disuse decays them."""
    checks_ok = 0
    total_checks = 0
    max_confidence = 0.0
    removed_total = 0

    for spec in scenario.ontology_reinforcement:
        controller = CognitiveController()
        total_checks += 1
        for _ in range(spec.repeats):
            controller.learn_ontology_from_observation(spec.text)

        learned = _learned_relations(controller)
        strongest = max((relation.confidence for relation in learned), default=0.0)
        max_confidence = max(max_confidence, strongest)
        ceiling_ok = all(relation.confidence <= spec.ceiling for relation in learned)
        reached = strongest >= spec.min_confidence
        if reached and ceiling_ok:
            checks_ok += 1

        if spec.decay_rounds:
            total_checks += 1
            for _ in range(spec.decay_rounds):
                decay = controller.decay_ontology_learning()
            removed_total += len(decay.removed)
            no_learned_left = not _learned_relations(controller)
            if spec.expect_removed and no_learned_left:
                checks_ok += 1

    accuracy = checks_ok / total_checks if total_checks else 0.0
    passed = accuracy == 1.0
    return ScenarioResult(
        scenario.id,
        scenario.domain.value,
        passed=passed,
        checks={
            "ontology_reinforcement_accuracy": round(accuracy, 6),
            "ontology_reinforcement_max_confidence": round(max_confidence, 6),
            "ontology_reinforcement_removed": float(removed_total),
        },
        details={"checks_ok": str(checks_ok), "checks_total": str(total_checks)},
    )


def _learned_relations(controller: CognitiveController) -> list:
    relation_type = controller.ontology_learner.relation_type
    return [
        relation
        for relation in controller.ontology.relations.values()
        if relation.relation_type == relation_type
    ]


def _run_petroleum_ontology_feedback(scenario: Scenario) -> ScenarioResult:
    """Petroleum scenario: a recall's outcome credits or blames the learned link behind it."""
    checks_ok = 0
    total_checks = 0
    reinforced_total = 0
    removed_total = 0

    for spec in scenario.ontology_feedback:
        controller = CognitiveController()
        controller.learn_ontology_from_observation(spec.text)
        controller.memory.store(
            "obs-feedback",
            "observation",
            Observation(spec.item_text, "study", 0.9),
            importance=0.9,
        )

        recalls = controller.recall_with_ontology(spec.probe, limit=5)
        linked = [recall for recall in recalls if recall.linked_relation is not None]
        total_checks += 2
        if not linked or linked[0].linked_relation is None:
            continue
        source_id, relation_type, target_id = linked[0].linked_relation
        relation = controller.ontology_learner.relation_by_triple(source_id, relation_type, target_id)
        if relation is None:
            continue
        before = relation.confidence

        feedback = controller.feedback_ontology_recall(linked[0], useful=spec.useful, amount=spec.amount)
        if feedback is None:
            continue

        outcome_ok = feedback.outcome == spec.expected_outcome
        if spec.useful:
            outcome_ok = outcome_ok and feedback.confidence > before
            reinforced_total += 1
        else:
            outcome_ok = outcome_ok and feedback.confidence < before
        if outcome_ok:
            checks_ok += 1

        if feedback.removed:
            removed_total += 1
        reachable = bool(controller.recall_with_ontology(spec.probe, limit=5))
        if reachable == spec.expect_recall_after:
            checks_ok += 1

    accuracy = checks_ok / total_checks if total_checks else 0.0
    passed = accuracy == 1.0
    return ScenarioResult(
        scenario.id,
        scenario.domain.value,
        passed=passed,
        checks={
            "ontology_feedback_accuracy": round(accuracy, 6),
            "ontology_feedback_reinforced": float(reinforced_total),
            "ontology_feedback_removed": float(removed_total),
        },
        details={"checks_ok": str(checks_ok), "checks_total": str(total_checks)},
    )


def _has_relation(controller: CognitiveController, source_label: str, target_label: str, relation_type: str) -> bool:
    sources = controller.ontology.find_concepts(source_label)
    targets = controller.ontology.find_concepts(target_label)
    if not sources or not targets:
        return False
    target_ids = {concept.id for concept in targets}
    return any(
        relation.relation_type == relation_type and relation.target_id in target_ids
        for relation in controller.ontology.outgoing(sources[0].id)
    )


def _run_gate_discrimination(scenario: Scenario) -> ScenarioResult:
    """Score the acceptance policy against a held-out verdict for each source.

    This is the one check in the suite whose answer does not come from the
    project's own seeds: it states whether each source *should* be accepted, so a
    gate that drifts (accepting an untrusted host, a water reservoir, or broken
    prose) scores below 1.0 and regresses the metric.
    """
    hits = 0
    for case in scenario.gate_cases:
        accepted = (
            rejection_reason(
                url=case.url,
                title=case.title,
                topic=case.topic,
                snippet=case.snippet,
                credibility=case.credibility,
            )
            is None
        )
        hits += int(accepted == case.expect_accept)
    total = len(scenario.gate_cases)
    accuracy = hits / total if total else 0.0
    return ScenarioResult(
        scenario.id,
        scenario.domain.value,
        passed=accuracy == 1.0,
        checks={"acceptance_gate_accuracy": round(accuracy, 6)},
        details={"correct": str(hits), "expected": str(total)},
    )


def run_scenario(scenario: Scenario) -> ScenarioResult:
    if scenario.gate_cases:
        return _run_gate_discrimination(scenario)
    if scenario.decisions:
        return _run_decisions(scenario)
    if scenario.expected_conflicts or (scenario.beliefs and not scenario.observations and not scenario.predictions):
        return _run_consistency(scenario)
    if scenario.domain == Domain.PETROLEUM:
        if scenario.ontology_feedback:
            return _run_petroleum_ontology_feedback(scenario)
        if scenario.ontology_reinforcement:
            return _run_petroleum_ontology_reinforcement(scenario)
        if scenario.ontology_learning:
            return _run_petroleum_ontology_learning(scenario)
        if scenario.ontology_retrieval:
            return _run_petroleum_multihop_retrieval(scenario)
        if scenario.retrieval:
            return _run_petroleum_retrieval(scenario)
        if scenario.ontology_inferences:
            return _run_petroleum_inference(scenario)
        if scenario.ontology_relations:
            return _run_petroleum_ontology(scenario)
    return _RUNNERS[scenario.domain](scenario)


def run_suite(scenarios: tuple[Scenario, ...] = SCENARIOS) -> EvaluationReport:
    results = tuple(run_scenario(scenario) for scenario in scenarios)

    # Prediction forecasts: explicit reasoning predictions plus uncertainty revisions.
    reasoning_forecasts: list[tuple[float, float]] = []
    uncertainty_forecasts: list[tuple[float, float]] = []
    for scenario in scenarios:
        for prediction in scenario.predictions:
            reasoning_forecasts.append((prediction.confidence, 1.0 if prediction.outcome >= 0.5 else 0.0))
        if scenario.uncertainties:
            engine = BeliefRevisionEngine()
            for uncertainty in scenario.uncertainties:
                belief = Belief(uncertainty.statement, uncertainty.prior_confidence)
                evidence_items = [
                    Evidence(item.content, item.source, item.supports, item.confidence)
                    for item in uncertainty.evidence
                ]
                revision = engine.revise(belief, evidence_items)
                uncertainty_forecasts.append((revision.revised.confidence, uncertainty.true_probability))

    all_forecasts = reasoning_forecasts + uncertainty_forecasts
    summary = brier_summary(all_forecasts)
    calibration_error = expected_calibration_error(all_forecasts)

    accuracy = (
        sum((confidence >= 0.5) == (outcome >= 0.5) for confidence, outcome in all_forecasts) / len(all_forecasts)
        if all_forecasts
        else 0.0
    )

    def domain_metric(domain: Domain, key: str) -> float:
        values = [
            result.checks[key]
            for result in results
            if result.domain == domain.value and key in result.checks
        ]
        return mean(values)

    retrieval_accuracy = domain_metric(Domain.MEMORY, "retrieval_accuracy")
    retrieval_mrr = domain_metric(Domain.MEMORY, "retrieval_mrr")
    planning_success = domain_metric(Domain.PLANNING, "planning_success_rate")
    decision_quality = domain_metric(Domain.PLANNING, "decision_explanation_quality")
    causal_accuracy = domain_metric(Domain.CAUSAL, "causal_chain_accuracy")
    causal_quality = domain_metric(Domain.CAUSAL, "causal_explanation_quality")
    consistency_values = [
        result.checks["consistency_match"]
        for result in results
        if "consistency_match" in result.checks
    ]
    contradiction_rate = mean(consistency_values)

    petroleum_causal = domain_metric(Domain.PETROLEUM, "causal_chain_accuracy")
    petroleum_retrieval = domain_metric(Domain.PETROLEUM, "retrieval_accuracy")
    petroleum_ontology = domain_metric(Domain.PETROLEUM, "ontology_accuracy")
    petroleum_inference = domain_metric(Domain.PETROLEUM, "ontology_inference_accuracy")
    petroleum_multihop = domain_metric(Domain.PETROLEUM, "ontology_retrieval_accuracy")
    petroleum_ontology_learning = domain_metric(Domain.PETROLEUM, "ontology_learning_accuracy")
    petroleum_reinforcement = domain_metric(Domain.PETROLEUM, "ontology_reinforcement_accuracy")
    petroleum_feedback = domain_metric(Domain.PETROLEUM, "ontology_feedback_accuracy")

    # Petroleum learning: train the policy + brain on every discipline's task and
    # take the mean held-out success. This is a real learning metric, not a
    # retrieval score, so it regresses if any discipline stops being learnable.
    petroleum_task_learning = petroleum_learning_accuracy()

    # Generalisation: the disciplines the learner never trained on. A closed
    # self-test cannot see this, so it is the honest half of "did it learn?".
    petroleum_generalisation = petroleum_generalisation_accuracy()

    # Acceptance gate: judgements the project must get right against an external
    # standard (a held-out verdict per source), not against its own seeds.
    acceptance_gate = domain_metric(Domain.SAFETY, "acceptance_gate_accuracy")

    # Curriculum: how much of the seeded degree (every course's concepts and
    # relations) the live ontology actually carries.
    curriculum_coverage_value = curriculum_coverage_score(CognitiveController().ontology)

    # Analogy: held-out judgements of which concept pairs are structurally alike,
    # including negative controls, so a permissive engine cannot pass.
    analogy_value = analogy_accuracy(ontology=CognitiveController().ontology)

    # Relation extraction: held-out sentences the reader must read *exactly* --
    # both directions of voice, plus negative controls (an unknown endpoint, a
    # concept inside a preposition, an unmapped verb), so an over-firing extractor
    # fails the gate as surely as one that misses.
    extraction_value = extraction_accuracy()

    # Relation polarity: held-out pairs of relation types the model must call a
    # conflict exactly when they are antonyms on the same ordered pair -- with
    # negative controls (an unrelated type, a symmetric relation), so an
    # over-firing polarity model fails the gate as surely as one that misses.
    polarity_value = polarity_accuracy()

    # Connection explanation: held-out concept pairs the search must connect
    # exactly when a chain exists and with the expected number of hops -- with
    # negative controls (an unknown endpoint), so a search that fabricates a
    # link fails the gate as surely as one that misses a real connection.
    connection_value = connection_accuracy()

    # Consistency: held-out graphs whose contradiction count must be exact, so
    # a hidden contradiction and a false positive both fail the gate.
    consistency_value = consistency_accuracy()

    # Graph inference: held-out entailments the reasoner must derive exactly when
    # one exists -- with negative controls (an unrelated pair, an unknown
    # endpoint), so a rule that over-reaches fails the gate as surely as one that
    # misses a real derivation. This is the metric for the reasoning layer
    # actually reaching the answer path.
    inference_value = inference_accuracy()

    # Semantic space: does the *learned* distributional space place related terms
    # closer than unrelated ones, on relations removed from its training text? A
    # memorising space scores near zero here, so the gate cannot be satisfied by
    # the hash-based signal it replaced.
    semantic_report = evaluate_held_out()[1]
    semantic_separation = semantic_report.separation

    petroleum_score = mean([
        petroleum_causal,
        petroleum_retrieval,
        petroleum_ontology,
        petroleum_inference,
        petroleum_multihop,
        petroleum_ontology_learning,
        petroleum_reinforcement,
        petroleum_feedback,
        petroleum_task_learning,
    ])

    calibrated = clamp(1.0 - calibration_error)
    overall = mean([
        accuracy,
        calibrated,
        contradiction_rate,
        retrieval_accuracy,
        planning_success,
        decision_quality,
        causal_accuracy,
        petroleum_causal,
        petroleum_retrieval,
        petroleum_ontology,
        petroleum_inference,
        petroleum_multihop,
        petroleum_ontology_learning,
        petroleum_reinforcement,
        petroleum_feedback,
        petroleum_task_learning,
        petroleum_generalisation,
        acceptance_gate,
        curriculum_coverage_value,
        analogy_value,
        clamp(semantic_separation),
        extraction_value,
        polarity_value,
        connection_value,
        consistency_value,
        inference_value,
    ])

    metrics = EvaluationMetrics(
        prediction_accuracy=round(accuracy, 6),
        brier_score=summary.brier,
        calibration_error=calibration_error,
        reliability=summary.reliability,
        resolution=summary.resolution,
        contradiction_rate=round(contradiction_rate, 6),
        retrieval_accuracy=round(retrieval_accuracy, 6),
        retrieval_mrr=round(retrieval_mrr, 6),
        planning_success_rate=round(planning_success, 6),
        decision_explanation_quality=round(decision_quality, 6),
        causal_chain_accuracy=round(causal_accuracy, 6),
        causal_explanation_quality=round(causal_quality, 6),
        petroleum_causal_accuracy=round(petroleum_causal, 6),
        petroleum_retrieval_accuracy=round(petroleum_retrieval, 6),
        petroleum_ontology_accuracy=round(petroleum_ontology, 6),
        petroleum_inference_accuracy=round(petroleum_inference, 6),
        petroleum_multihop_retrieval_accuracy=round(petroleum_multihop, 6),
        petroleum_ontology_learning_accuracy=round(petroleum_ontology_learning, 6),
        petroleum_ontology_reinforcement_accuracy=round(petroleum_reinforcement, 6),
        petroleum_ontology_feedback_accuracy=round(petroleum_feedback, 6),
        petroleum_learning_accuracy=round(petroleum_task_learning, 6),
        petroleum_generalisation_accuracy=round(petroleum_generalisation, 6),
        acceptance_gate_accuracy=round(acceptance_gate, 6),
        curriculum_coverage=round(curriculum_coverage_value, 6),
        analogy_accuracy=round(analogy_value, 6),
        relation_extraction_accuracy=round(extraction_value, 6),
        polarity_accuracy=round(polarity_value, 6),
        connection_accuracy=round(connection_value, 6),
        consistency_accuracy=round(consistency_value, 6),
        inference_accuracy=round(inference_value, 6),
        semantic_separation=round(semantic_separation, 6),
        overall=round(overall, 6),
    )

    domain_scores = {
        Domain.MEMORY.value: round(retrieval_accuracy, 6),
        Domain.REASONING.value: round(accuracy, 6),
        Domain.CAUSAL.value: round(causal_accuracy, 6),
        Domain.PLANNING.value: round(planning_success, 6),
        Domain.UNCERTAINTY.value: round(calibrated, 6),
        Domain.PETROLEUM.value: round(petroleum_score, 6),
        Domain.SAFETY.value: round(acceptance_gate, 6),
    }

    return EvaluationReport(
        metrics=metrics,
        scenarios=results,
        domain_scores=domain_scores,
        calibration=tuple(calibration_bins(all_forecasts)),
    )
