from __future__ import annotations

from dataclasses import dataclass, field

from cognition.asking import asking_loop_accuracy
from cognition.development import development_accuracy
from cognition.knowledge_depth import knowledge_depth_accuracy
from core.acceptance_policy import rejection_reason
from core.action import Action
from core.agents.gate import personality_agents_accuracy
from core.analogy import analogy_accuracy
from core.answer_analogy import analogy_answer_accuracy
from core.answer_audit import (
    ARABIC_READABLE_EXPECTATIONS,
    AUDIT_EXPECTATIONS_AR,
    answer_path_accuracy,
)
from core.answer_belief import belief_accuracy
from core.answer_confidence import answer_confidence_accuracy
from core.answer_decision import decision_recommendation_accuracy
from core.answer_lead import lead_accuracy
from core.answer_plan import plan_accuracy
from core.answer_temporal import temporal_accuracy
from core.belief import Belief
from core.belief_revision import BeliefRevisionEngine
from core.causal_link import CausalLink
from core.causal_paths import causal_reasoning_accuracy, controller_causal_accuracy
from core.character_identity_gate import character_identity_accuracy
from core.character_isolation_gate import character_isolation_accuracy
from core.character_knowledge_gate import character_knowledge_accuracy
from core.character_memory_gate import memory_seeding_accuracy
from core.cognition_catalog import catalog_accuracy
from core.cognitive_controller import CognitiveController
from core.cognitive_core_gate import knowledge_validation_accuracy
from core.conclusion_verification import verification_accuracy
from core.curriculum_knowledge import curriculum_coverage_score
from core.embeddings import evaluate_held_out
from core.evidence import Evidence
from core.goal import Goal
from core.grounding import grounding_accuracy
from core.hypothesis_reasoning import hypothesis_accuracy
from core.improvement.gate import self_improvement_loop_accuracy
from core.knowledge_carry import knowledge_carry_accuracy
from core.learning_gate import learning_accuracy
from core.memory_gate import memory_architecture_accuracy
from core.observation import Observation
from core.perception_gate import perception_bridge_accuracy
from core.planning_gate import task_execution_accuracy
from core.polish_gate import polish_accuracy
from core.problem_decomposition import decomposition_accuracy
from core.reasoning_api import facade_accuracy
from core.reasoning_report import reasoning_trace_accuracy
from core.relation_connections import connection_reasoning_accuracy
from core.relation_consistency import consistency_accuracy
from core.relation_extraction import extraction_accuracy
from core.relation_inference import inference_accuracy
from core.relation_paths import connection_accuracy
from core.relation_polarity import polarity_accuracy
from core.state import State
from core.tool_agent import tool_selection_accuracy
from core.training_center.gate import training_center_accuracy
from core.workplace.phase10_gate import workplace_phase10_accuracy
from ml.petroleum import petroleum_generalisation_accuracy, petroleum_learning_accuracy
from ml.resume_gate import training_resume_accuracy
from ml.training.gate import training_accuracy

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
    connection_reasoning_accuracy: float
    causal_reasoning_accuracy: float
    controller_causal_accuracy: float
    answer_confidence_accuracy: float
    grounding_accuracy: float
    reasoning_trace_accuracy: float
    plan_accuracy: float
    decision_recommendation_accuracy: float
    belief_accuracy: float
    lead_accuracy: float
    temporal_accuracy: float
    analogy_answer_accuracy: float
    knowledge_carry_accuracy: float
    # Answer path: the *connected* path, not the isolated layer (Gap 9)
    answer_path_accuracy: float
    # ... and the same path in Arabic (Phase 49)
    arabic_answer_path_accuracy: float
    """The Arabic twin of ``answer_path_accuracy``: the connected path in Arabic."""

    arabic_readable_body_accuracy: float
    """An Arabic answer that names concepts must name them readably (Phase 51):
    the share of held-out Arabic questions whose body carries no snake_case
    identifier (``shut_in``, ``injection_pattern``)."""

    character_isolation_accuracy: float
    """The share of held-out isolation and knowledge-sharing judgements the
    backend satisfies: a cross-character read refused, the main model kept out of
    Natasha's store, provenance carried onto its own copy, no duplicate on a
    second share (Phase: independent characters)."""

    polish_accuracy: float
    """The share of held-out linguistic-polish judgements satisfied: an Arabic
    reply is vocalized only when asked (and English never is), a Latin identifier,
    a decimal, and a range survive untouched, the typography is repaired in both
    languages, a second pass changes nothing, no word is added or removed, the
    colloquial mode drops the case endings, and every answer is polished on the
    real Jarvis path (Phase: the universal linguistic polish)."""

    character_identity_accuracy: float
    character_knowledge_accuracy: float
    """The share of held-out character-knowledge judgements satisfied: a
    character reads only its own corpus, the reference block is labelled and
    bounded, an absent corpus is honest and changes nothing, the mouth injects
    what it retrieved, and reading a corpus opens no socket (Phase: the
    connectors + the per-character knowledge loader)."""

    memory_seeding_accuracy: float
    """The share of held-out identity judgements the wiring satisfies: a
    character's personality reaches the *answer text* it produces (not only its
    configuration), the identity does not bleed across mouths, an unknown key
    falls back to the project's own model, and the two windows that show one
    character share its memory and its display name (Phase C: one identity)."""

    development_accuracy: float
    """The share of held-out cognitive-development judgements the mind's own
    faculties satisfy: the self-model favours the strategy that pays off and stays
    neutral on an untried one, the stagnation detector fires on a genuinely flat
    history but not when novelty rises or something was promoted, a rule needs real
    support before it is stated, the spaced-review box advances and resets, and an
    escalation stays inside its ceilings (Phase: maximal development)."""

    cognition_catalog_accuracy: float
    """The share of held-out catalogue judgements satisfied: every control the
    window draws has a catalogue entry, every entry has a window control, a
    governed action names its confirmation in the CLI, every CLI command resolves
    to a real cognition subcommand, and every section is described in both
    languages (Phase: clearer GUI -- one source of truth for every control)."""

    knowledge_depth_accuracy: float
    """The share of held-out knowledge-depth judgements satisfied: the graph's
    reach is measured exactly, connectivity counts only concepts that participate,
    a harvest previews without writing and writes only through the bounded
    learner, a bridge is a real walked path (and an unconnected pair yields
    nothing), and an Arabic label with no alias is reported rather than counted
    (Phase: widest knowledge reach)."""

    asking_loop_accuracy: float
    """The share of held-out asking-loop judgements satisfied: the learner asks and
    the **teacher** always answers (the learner never answers its own question), no
    question repeats within a run, asking the loop to stop ends it early, only an
    answer the graph corroborates becomes a qualified example, an empty answer is
    reported rather than hidden, a spent question source ends the run, and a
    qualified answer reaches the learner's own fact store (the learner-asks-teacher
    loop)."""

    memory_architecture_accuracy: float
    """The share of held-out long-term-memory judgements the Phase-2 store
    satisfies: a record survives a restart with its provenance, a correction keeps
    the old record beside the new one, archive is not delete, a backup restores
    while a corrupt one is refused, a corrupt row is skipped not raised, a
    failed batch rolls back, a validity window is honoured, ranking reports its
    signals separately, consolidation promotes and skips, Natasha's memory is
    unreadable by anyone else, and retrieval similarity is never the record's own
    confidence."""

    learning_accuracy: float
    """The share of held-out learning judgements the Phase-3 pipelines satisfy: an
    evidenced claim is integrated while an unsupported one is not, provenance is
    preserved, a duplicate reinforces without a second edge, a contradiction is
    preserved rather than resolved, an experience survives a restart and its lesson
    is rebuilt, a single success never promotes a procedure (nor does a failure),
    one feedback record never overrides evidence, an inferred signal is bounded
    below an explicit one, silence is refused, feedback is correctable, versions
    advance and roll back while the audit trail survives, and Natasha's learning is
    excluded from the main model's dataset."""

    problem_decomposition_accuracy: float
    """The share of held-out decomposition judgements satisfied: a question is broken
    into the reasoning units it warrants, with negative controls (a single-clause
    question stays one part, a noun conjunction is not split, an empty question
    decomposes to nothing, and a multi-clause question is read as the parts its
    kinds warrant) (Phase 4: reasoning and inference)."""

    conclusion_verification_accuracy: float
    """The share of held-out conclusion-verification judgements satisfied: a
    conclusion is verified against the graph as supported / contradicted / partially
    supported / insufficient-information / unverified, with negative controls, so a
    verifier that over-claims or under-claims fails (Phase 4)."""

    reasoning_facade_accuracy: float
    """The share of held-out facade judgements satisfied: the one reasoning API
    routes a question to the engine its kind warrants and returns the engine's own
    object, with negative controls (a definitional or unrelated question must not
    produce a plan/decision/verdict/comparison/hypothesis) (Phase 4)."""

    hypothesis_accuracy: float
    """The share of held-out hypothesis judgements satisfied: candidate hypotheses
    are generated only from what the graph licenses (a stored relation, an
    entailment, a walked chain, a causal path, or a decision option), ranked, with a
    contested candidate reported and an unknown question yielding none (Phase 4)."""

    tool_selection_accuracy: float
    """The share of held-out tool-selection judgements satisfied: a bounded chooser
    picks only tools the sandbox allow-list permits and refuses a mutating tool in a
    read-only sandbox -- the model proposes, the sandbox disposes (Phase 4)."""

    personality_agents_accuracy: float
    """The share of held-out personality-agent judgements satisfied: the seven
    personalities are independent agents -- each with its own identity, capabilities
    (bounded by a hard ceiling a configuration cannot widen), working context,
    memory and learning scope -- with Natasha's memory and learning isolated, the
    router honest about when no personality is suitable, delegation bounded by
    depth and budget, and conflicts preserved rather than voted away (Phase 5)."""

    training_resume_accuracy: float
    """The share of held-out resumable-training judgements satisfied: the progress
    ledger round-trips and tolerates corruption; a changed configuration
    invalidates a record while a raised episode budget tops it up; a finished unit
    is loaded rather than retrained; a partial run continues to the same policy an
    uninterrupted run reaches; and the language model's own resume still raises the
    cumulative step count (Phase: resumable training)."""

    training_accuracy: float
    """The share of held-out Phase-6 training judgements satisfied: dataset
    validation, provenance, versioning, and splitting; the Natasha exclusion for the
    project model's aggregate; model compatibility (a GGUF is not a training base,
    QLoRA needs a CUDA GPU); config validation; the training job life cycle; the
    base model provably unchanged by a run; and controlled promotion plus rollback
    -- each on the real implementation, with negative controls (Phase 6)."""

    training_center_accuracy: float
    """The share of held-out Training Center judgements satisfied: a review status
    governs export (an unapproved example is refused, a rejected one is never
    approved), a governed pipeline stage is refused without approval and an
    unimplemented stage is reported unavailable, the catalogue is complete and its
    CLI commands resolve, a provider with no key/approval is honest, an
    unreportable resource reads as unknown rather than zero, and a pipeline
    round-trips through its store -- each on the real implementation, with negative
    controls."""

    perception_bridge_accuracy: float
    """The share of held-out Phase-7 perception judgements satisfied: a document,
    PDF, image, or audio source is read into the one normalized result shape with
    its provenance; a perceived artifact becomes a citable Document and evidence on
    the answer path; a model-generated description is labelled as a reading and its
    evidence is weaker than a stated fact's; an unsupported claim is never stored;
    and an unreadable, unsupported, or cancelled source is refused rather than
    guessed at -- each on the real implementation, with negative controls."""

    task_execution_accuracy: float
    """The share of held-out Phase-8 planning/execution judgements satisfied: a
    goal is decomposed into tasks each bound to a permitted tool; the plan is
    validated before it runs; it executes under a permission and approval gate;
    the outcome is verified against the goal's own criteria; and a failure is
    recovered from safely -- with negative controls (a suggested goal does not
    authorize, an action outside the allow-list is refused, a high-risk action
    needs approval, an irreversible failure is not retried, a tool's success is
    not the goal's, an invalid plan is never executed, a failure is never
    hidden), each on the real implementation."""

    self_improvement_loop_accuracy: float
    """The share of held-out Phase-9 self-improvement judgements satisfied: the
    monitor records structured, provenance-carrying, bounded metrics; a repeated
    failure raises a weakness while a single one does not; a weakness carries its
    evidence, component, frequency, and confidence without claiming a cause; a
    proposal carries every field the phase lists and names the finding it answers;
    a priority is explained by named factors and the user's override is honoured;
    a candidate runs in an isolated git worktree with a throwaway data root and a
    protected path is refused; an improvement is confirmed by the project's own
    acceptance rule while a regression is rejected and a metric-less run is
    inconclusive; a change needs a human approval the loop cannot grant itself and
    the approval is bound to one proposal; a deployment is versioned, backed up,
    refuses a weight/dataset/security change, and never applies a source change
    automatically; a rollback restores and verifies without deleting history; and
    the audit log is append-only with no forbidden self-modification operation
    exposed -- each on the real implementation, with negative controls."""

    workplace_phase10_accuracy: float
    """The share of held-out Workplace Phase-10 judgements satisfied: one search
    query reaches every record kind that carries it, a blank query returns nothing,
    a kind filter narrows to that kind, the row carrying the whole query ranks
    first, a query naming nothing yields no hit, the search adds no parallel index,
    a classified record is withheld from an uncleared reader, and the project
    listing's per-project counts are correct and gathered with a bounded number of
    reads (the N+1 fix) -- each on the real search engine and repository, with
    negative controls."""

    knowledge_validation_accuracy: float
    """The share of held-out cognitive-core judgements satisfied: the event bus
    delivers in order, isolates a raising subscriber, and drops the oldest from a
    bounded queue; the governor defers a request over its budget and one whose need
    cannot be judged; the validator never calls an unsupported claim knowledge and
    reports an opposing relation as a contradiction while an opposite-*direction*
    claim is not one; the integrator writes, refuses, preserves a conflict, and
    rolls back exactly; and the query engine finds the shortest chain with its
    product confidence (Phase 1: cognitive core enhancement)."""
    """The share of held-out asking-loop judgements satisfied: the learner asks and
    the **teacher** always answers (the learner never answers its own question), no
    question repeats within a run, asking the loop to stop ends it early, only an
    answer the graph corroborates becomes a qualified example, an empty answer is
    reported rather than hidden, a spent question source ends the run, and a
    qualified answer reaches the learner's own fact store (the learner-asks-teacher
    loop)."""

    semantic_separation: float
    # Aggregate
    overall: float


@dataclass(frozen=True)
class EvaluationReport:
    metrics: EvaluationMetrics
    scenarios: tuple[ScenarioResult, ...]
    domain_scores: dict[str, float]
    calibration: tuple[CalibrationBin, ...] = ()


def _lead_answer_for():
    """The assistant's own ``ask``, or ``None`` when it cannot be built here.

    The lead metric's body check must exercise the *real* answer path, so it is
    handed the project's assistant. Building it can need the corpus and the
    retriever; a failure means the check is skipped rather than faked, and the
    metric is then computed over the pure-bridge judgements only.
    """
    try:
        from core.assistant import build_default_assistant

        assistant = build_default_assistant()
    except Exception:
        return None
    return assistant.ask


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

    # Connection reasoning: held-out pairs the *answer-path* walk must connect
    # exactly when a chain exists and with the expected number of hops -- with
    # negative controls (an unknown endpoint), so a walk that fabricates a chain
    # fails the gate as surely as one that misses a real connection. This is the
    # metric for the multi-hop reasoning layer reaching the answer path.
    connection_reasoning_value = connection_reasoning_accuracy()

    # Causal reasoning: held-out causal pairs the *answer-path* walk must connect
    # exactly when a directed causal chain exists and with the expected number of
    # links -- with negative controls (an associative-only pair that must yield no
    # cause), so a walk that reads an association as causation fails the gate as
    # surely as one that misses a real mechanism. This is the metric for the causal
    # layer reaching the answer path.
    causal_reasoning_value = causal_reasoning_accuracy()

    # The controller's *own* causal graph (Phase 52): the walk above rebuilt a graph
    # from the ontology on every call, so it stayed green while the controller's
    # ``CausalGraph`` was empty (0 links against the 111 the seed states). This walks
    # the graph the controller actually builds, so an unwired causal layer fails here.
    controller_causal_value = controller_causal_accuracy()

    # Answer confidence: held-out expectations about how graph support and stored
    # contradictions must move an answer's confidence -- with negative controls (a
    # graph signal with no evidence must stay at zero, a contradiction must lower
    # the number). This is the metric for the confidence layer weighing the graph's
    # own reasoning.
    confidence_value = answer_confidence_accuracy()

    # Grounding gate: held-out expectations about whether model text may be
    # published -- with negative controls (a paragraph riding one incidental word,
    # a majority of unsupported sentences, a reversed causal mechanism must each be
    # refused). This is the metric for the gate that keeps an untraceable answer
    # from being shown.
    grounding_value = grounding_accuracy()

    # Reasoning trace: held-out expectations about the *shape* of the user-visible
    # trace -- with negative controls (a trace with nothing must render empty; an
    # evidence-only trace must keep every other step present-but-empty, never
    # dropped; a derivation must be labelled derived, not stored). This is the
    # metric for the layer that shows *how* an answer was reached.
    trace_value = reasoning_trace_accuracy()

    # Plan: held-out expectations about the plan bridge -- with negative controls
    # (a definitional or unrelated question must yield *no* plan; a circular or
    # dangling dependency must be refused, not reordered). This is the metric for
    # the layer that finally lets a procedural question reach the planner.
    plan_value = plan_accuracy()

    # Decision: held-out expectations about the decision bridge -- with negative
    # controls (a definitional or unrelated question must yield *no*
    # recommendation; a direction the graph holds no edge for must not be answered
    # with an edge that runs the other way). This is the metric for the layer that
    # finally lets a "what should I do?" question reach the decision engine.
    decision_value = decision_recommendation_accuracy()

    # Belief: held-out expectations about the belief bridge -- with negative
    # controls (a definitional question, a plain statement, or an evaluative
    # question the graph cannot speak to must yield *no* belief). This is the
    # metric for the layer that finally lets a "is ... true?" question reach the
    # belief revision engine.
    belief_value = belief_accuracy()

    # Lead: held-out expectations about the *answer body* -- with negative controls
    # (a definitional or unrelated question must earn no lead). This is the metric
    # for the last step of the reasoning bridge: the reasoning is in the body the
    # reader sees, not only in the trailing trace. It exercises the real answer
    # path through the assistant's own ``ask``.
    lead_value = lead_accuracy(answer_for=_lead_answer_for())

    # Temporal: held-out expectations about the *timeline* -- with negative controls
    # (a definitional or unrelated question must earn no timeline, and a conversation
    # with fewer than two turns must earn none either). This is the metric for the
    # layer that finally lets a "what changed?" question reach the temporal
    # reasoner's own ordering, pattern detection, and next-time prediction.
    temporal_value = temporal_accuracy()

    # Analogy: held-out expectations about the *comparison* the answer carries --
    # with negative controls (a definitional question, an unrelated one, or an
    # analogical question naming one concept must earn none, and a pair with no
    # shared relational system must be reported as *not* an analogy). This is the
    # metric for the last isolated engine of Gap 7: the analogical layer finally
    # reaches a "is X like Y?" question instead of a definition -- or a belief
    # verdict about a different claim.
    analogy_answer_value = analogy_answer_accuracy()

    # Knowledge carry: held-out expectations about *what survives a session* --
    # with negative controls (a curated seed is never captured as learned, a
    # record naming an unknown concept is skipped rather than invented, and a
    # second restore is a no-op). This is the metric for Gap 8: the graph's
    # *learned* relations now survive a session boundary instead of being rebuilt
    # from the seeds every time.
    carry_value = knowledge_carry_accuracy()

    # Answer path: held-out expectations about the *connected* answer path, not
    # the isolated layer -- with negative controls (a definitional, unrelated, or
    # vague question must carry no reasoning layer at all). Each expectation runs
    # the real assistant's own ``ask`` and checks what the answer *object* carries
    # (the plan, recommendation, verdict, comparison, timeline, lead, trace, and
    # the graph's facts/inferences/connections/causal paths), and, where a layer
    # is meant to reach the body, that it is in the text the reader sees. This is
    # the metric for Gap 9: the evaluation measures the connected path, so a layer
    # that stops reaching the answer fails even while its isolated bridge passes.
    path_value = answer_path_accuracy(answer_for=_lead_answer_for())

    # Arabic answer path: the same connected-path audit, in Arabic (Phase 49).
    # The project is Arabic-first, so the metric must measure the Arabic path too;
    # until Phase 49 only 39 of 495 concepts carried an Arabic alias, so an Arabic
    # question reached almost none of the graph. Each English expectation has an
    # Arabic twin, so a regression that breaks only Arabic fails here while the
    # English catalogue stays green.
    arabic_path_value = answer_path_accuracy(
        AUDIT_EXPECTATIONS_AR, answer_for=_lead_answer_for()
    )

    # Arabic readable body (Phase 51): an Arabic answer that names concepts must
    # name them *readably* -- the graph's snake_case identifiers (``shut_in``,
    # ``injection_pattern``) reached the body as code tokens before this phase. The
    # metric runs the same connected path and fails if any identifier survives, so
    # a regression that stops rendering them is caught here.
    arabic_readable_value = answer_path_accuracy(
        ARABIC_READABLE_EXPECTATIONS, answer_for=_lead_answer_for()
    )

    # Character isolation and sharing: held-out judgements about the seven
    # characters' independent learning stores and the main model's controlled
    # knowledge sharing -- with negative controls (a cross-character read must be
    # refused, the main model must never reach Natasha's records, and a hand-edited
    # configuration must not be able to enable sharing for her). This is the gate
    # for the character upgrade: a regression that opened one boundary fails here
    # even though the unit tests that pinned the good case still pass.
    character_isolation_value = character_isolation_accuracy()

    # Character identity: a character's personality must reach the *answer text*
    # it produces, not only its configuration -- and the two windows that show one
    # character (the main chat and its dedicated window) must share its memory and
    # its display name. Held-out judgements with negative controls (no bleed across
    # mouths, an unknown key falls back to the project's own model, a non-character
    # mouth is left unwrapped), so a regression in the identity wiring fails here.
    character_identity_value = character_identity_accuracy()

    # The universal linguistic polish: every character's answer passes through one
    # shared polisher that can fully vocalize an Arabic reply (tashkeel). Held-out
    # judgements with negative controls (English is never vocalized, an
    # already-vocalized reply is not double-marked, a Latin identifier/decimal/range
    # survives untouched, no word is added or removed), and a wiring judgement that
    # runs a *real* JarvisMode so a regression that dropped the call fails here.
    polish_value = polish_accuracy()

    # The per-character knowledge corpus: a character reads only its own built
    # corpus, the retrieved block is labelled and bounded, an absent corpus is
    # honest and changes nothing, the mouth injects what it retrieved, and reading
    # opens no socket. Held-out judgements with negative controls (a
    # cross-character query returns nothing, an absent corpus adds no block), so a
    # regression that leaked a corpus or reached the network fails here.
    character_knowledge_value = character_knowledge_accuracy()

    # Memory revival: a fresh character's memory is real but empty. Held-out
    # judgements that a character is seeded from the project's *own authored*
    # knowledge -- with negative controls (an unknown key resolves rather than
    # raising, Hakeem is never seeded with an invented religious text, the seed is
    # bounded and idempotent). So a seeding regression fails the gate.
    memory_seeding_value = memory_seeding_accuracy()

    # Cognitive development: held-out judgements about the mind's ability to
    # improve itself -- the self-model favouring what pays off, the stagnation
    # detector not crying wolf, a rule needing real support, the spaced-review box
    # advancing and resetting, and an escalation staying inside its ceilings. With
    # negative controls throughout, so an over-eager faculty fails as surely as a
    # broken one. This is the gate for the maximal-development phase.
    development_value = development_accuracy()

    # Cognition catalogue: every control the window draws must be explained (title,
    # summary, outcome, and an Arabic twin), every entry must have a window control,
    # and a governed action must name its confirmation in the CLI -- so the window
    # and the catalogue can never drift and a reader always knows what a control
    # does before pressing it. Held-out, with the CLI subcommands read from the real
    # parser, so a control that stopped being explained fails here.
    catalog_value = catalog_accuracy()

    # Knowledge depth: held-out judgements about how far the graph reaches and how
    # it is widened -- the coverage count exact, connectivity counting only
    # participants, a harvest previewing without writing, a bridge a real walked
    # path (and an unconnected pair yielding nothing), and an unaliased Arabic
    # label reported rather than counted. With negative controls throughout.
    depth_value = knowledge_depth_accuracy()

    # The learner-asks-teacher loop: held-out judgements about the endless question
    # loop -- the teacher always answers (the learner never answers its own
    # question), no question repeats, a stop takes effect, only a verified answer
    # qualifies, an empty answer is honest, a spent source ends the run, and a
    # qualified answer reaches the learner's own fact store. With negative
    # controls throughout.
    asking_value = asking_loop_accuracy()

    # The Phase-1 cognitive core: held-out judgements about the event bus, the
    # resource governor, the knowledge validator, the transactional integrator,
    # and the query engine, each run against the real service with negative
    # controls, so a core regression fails here even when its unit tests pass.
    core_value = knowledge_validation_accuracy()

    # Long-term memory (Phase 2): the durable four-category memory over one
    # SQLite store. Held-out judgements run against the real store with negative
    # controls (a corrupt backup refused, a corrupt row tolerated, Natasha's
    # memory unreadable, consolidation not promoting a trivial item), so a memory
    # regression fails here even when its unit tests pass.
    memory_value = memory_architecture_accuracy()

    # Cognitive and experience-based learning (Phase 3): the knowledge pipeline,
    # the experience pipeline, the feedback processor, the pattern learner, and the
    # per-personality isolation. Held-out judgements run against the real pipelines
    # with negative controls (an unsupported claim not integrated, a single success
    # not promoting a procedure, one voice not overriding evidence, Natasha's
    # learning excluded), so a learning regression fails here even when its unit
    # tests pass.
    learning_value = learning_accuracy()

    # Reasoning and inference (Phase 4): held-out judgements for the one reasoning
    # facade, the problem decomposer, the conclusion verifier, the bounded tool
    # chooser, and hypothesis generation -- each with negative controls (a
    # definitional/unrelated question must earn no plan/decision/verdict/comparison/
    # hypothesis, a single-clause question must not be split, an unknown question
    # must yield no candidate, and a mutating tool must be refused in a read-only
    # sandbox). Each runs the *real* engine, so a regression that stopped reaching
    # the engine fails here even while the unit tests that pinned the good case pass.
    decomposition_value = decomposition_accuracy()
    verification_value = verification_accuracy()
    facade_value = facade_accuracy()
    hypothesis_value = hypothesis_accuracy()
    tool_selection_value = tool_selection_accuracy()
    personality_agents_value = personality_agents_accuracy()
    training_value = training_accuracy()
    training_center_value = training_center_accuracy()
    # Resumable training: a second run continues from what the first left -- the
    # ledger round-trips and tolerates corruption, a changed config invalidates a
    # record while a raised budget tops it up, a finished unit is loaded not
    # retrained, a partial run continues to the same bytes an uninterrupted one
    # reaches, and the language model's own resume still holds (Phase: resumable
    # training). Held-out judgements with negative controls, run on the real trainers.
    resume_value = training_resume_accuracy()

    # Perception (Phase 7): a perceived artifact must reach the answer path as
    # *evidence* with its provenance, never as a verified fact. Held-out judgements
    # over the real detector, pipelines, schema, validator, and bridge -- with
    # negative controls (a model description is weaker than a stated fact and never
    # native text, an unsupported claim is never stored, an unreadable/unsupported/
    # cancelled source is refused), so a regression in perception fails here.
    perception_value = perception_bridge_accuracy()

    # Planning and execution (Phase 8): a goal becomes a validated plan whose tasks
    # are bound to permitted tools, run under the permission/approval gate, verified
    # against the goal's own criteria, and recovered from safely. Held-out
    # judgements over the real planner, engine, permission manager, verifier, and
    # recovery manager -- with negative controls (a suggested goal does not
    # authorize, an action outside the allow-list is refused, a high-risk action
    # needs approval, an irreversible failure is not retried, a tool's success is
    # not the goal's, an invalid plan is never executed, a failure is never hidden),
    # so a regression in autonomy fails here.
    planning_value = task_execution_accuracy()

    # Controlled self-improvement (Phase 9): the human-gated loop -- monitor,
    # detect, propose, isolate, measure, accept/reject, deploy, roll back. Held-out
    # judgements over the real monitor, detector, proposal generator, prioritizer,
    # sandbox, experiment manager, evaluator, approval manager, deployment
    # coordinator, rollback manager, and history -- with negative controls (a
    # single failure raises nothing, a candidate that regresses is rejected, a
    # protected path is refused, an unapproved deployment is refused, a sensitive
    # change is refused, a source change is never automatic, the audit log is
    # append-only, no forbidden self-modification operation is exposed, and
    # Natasha stays out of the aggregated training data), so a regression in
    # self-improvement fails here.
    self_improvement_value = self_improvement_loop_accuracy()

    # The Workplace (EPRI/TSTC's digital office): the enterprise module's Phase 10
    # measures its *advanced search* -- one query reaching every record kind, a
    # blank query returning nothing, a kind filter narrowing, the strongest match
    # ranking first, no parallel index -- and that the project listing's counts are
    # correct and gathered with a bounded number of reads (the N+1 fix). Held-out
    # judgements over the real search engine and repository, with negative controls,
    # so a fabricated hit or a re-introduced N+1 fails here.
    workplace_value = workplace_phase10_accuracy()

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
        connection_reasoning_value,
        causal_reasoning_value,
        confidence_value,
        grounding_value,
        trace_value,
        plan_value,
        decision_value,
        belief_value,
        lead_value,
        temporal_value,
        analogy_answer_value,
        carry_value,
        path_value,
        arabic_path_value,
        arabic_readable_value,
        character_isolation_value,
        character_identity_value,
        polish_value,
        character_knowledge_value,
        memory_seeding_value,
        development_value,
        catalog_value,
        depth_value,
        asking_value,
        core_value,
        memory_value,
        learning_value,
        decomposition_value,
        verification_value,
        facade_value,
        hypothesis_value,
        tool_selection_value,
        personality_agents_value,
        training_value,
        training_center_value,
        resume_value,
        perception_value,
        planning_value,
        self_improvement_value,
        workplace_value,
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
        connection_reasoning_accuracy=round(connection_reasoning_value, 6),
        causal_reasoning_accuracy=round(causal_reasoning_value, 6),
        controller_causal_accuracy=round(controller_causal_value, 6),
        answer_confidence_accuracy=round(confidence_value, 6),
        grounding_accuracy=round(grounding_value, 6),
        reasoning_trace_accuracy=round(trace_value, 6),
        plan_accuracy=round(plan_value, 6),
        decision_recommendation_accuracy=round(decision_value, 6),
        belief_accuracy=round(belief_value, 6),
        lead_accuracy=round(lead_value, 6),
        temporal_accuracy=round(temporal_value, 6),
        analogy_answer_accuracy=round(analogy_answer_value, 6),
        knowledge_carry_accuracy=round(carry_value, 6),
        answer_path_accuracy=round(path_value, 6),
        arabic_answer_path_accuracy=round(arabic_path_value, 6),
        arabic_readable_body_accuracy=round(arabic_readable_value, 6),
        character_isolation_accuracy=round(character_isolation_value, 6),
        character_identity_accuracy=round(character_identity_value, 6),
        polish_accuracy=round(polish_value, 6),
        character_knowledge_accuracy=round(character_knowledge_value, 6),
        memory_seeding_accuracy=round(memory_seeding_value, 6),
        development_accuracy=round(development_value, 6),
        cognition_catalog_accuracy=round(catalog_value, 6),
        knowledge_depth_accuracy=round(depth_value, 6),
        asking_loop_accuracy=round(asking_value, 6),
        knowledge_validation_accuracy=round(core_value, 6),
        memory_architecture_accuracy=round(memory_value, 6),
        learning_accuracy=round(learning_value, 6),
        problem_decomposition_accuracy=round(decomposition_value, 6),
        conclusion_verification_accuracy=round(verification_value, 6),
        reasoning_facade_accuracy=round(facade_value, 6),
        hypothesis_accuracy=round(hypothesis_value, 6),
        tool_selection_accuracy=round(tool_selection_value, 6),
        personality_agents_accuracy=round(personality_agents_value, 6),
        training_accuracy=round(training_value, 6),
        training_center_accuracy=round(training_center_value, 6),
        training_resume_accuracy=round(resume_value, 6),
        perception_bridge_accuracy=round(perception_value, 6),
        task_execution_accuracy=round(planning_value, 6),
        self_improvement_loop_accuracy=round(self_improvement_value, 6),
        workplace_phase10_accuracy=round(workplace_value, 6),
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
