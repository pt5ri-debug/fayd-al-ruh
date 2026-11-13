from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evaluation.runner import EvaluationMetrics, EvaluationReport


@dataclass(frozen=True)
class MetricSpec:
    """Declares how a metric should be compared across versions.

    ``higher_is_better`` fixes the direction; ``tolerance`` absorbs harmless
    floating-point or sampling noise. When ``tolerance`` is zero the metric
    must not move at all in the wrong direction.
    """

    name: str
    higher_is_better: bool
    tolerance: float = 0.0
    label: str = ""


METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec("prediction_accuracy", True, label="Prediction accuracy"),
    MetricSpec("brier_score", False, label="Brier score (lower is better)"),
    MetricSpec("calibration_error", False, label="Calibration error (ECE)"),
    MetricSpec("reliability", False, label="Reliability (lower is better)"),
    MetricSpec("resolution", True, label="Resolution"),
    MetricSpec("contradiction_rate", True, label="Contradiction detection"),
    MetricSpec("retrieval_accuracy", True, label="Retrieval accuracy"),
    MetricSpec("retrieval_mrr", True, label="Retrieval MRR"),
    MetricSpec("planning_success_rate", True, label="Planning success"),
    MetricSpec("decision_explanation_quality", True, label="Decision explanation quality"),
    MetricSpec("causal_chain_accuracy", True, label="Causal chain accuracy"),
    MetricSpec("causal_explanation_quality", True, label="Causal explanation quality"),
    MetricSpec("petroleum_causal_accuracy", True, label="Petroleum causal accuracy"),
    MetricSpec("petroleum_retrieval_accuracy", True, label="Petroleum retrieval accuracy"),
    MetricSpec("petroleum_ontology_accuracy", True, label="Petroleum ontology accuracy"),
    MetricSpec("petroleum_inference_accuracy", True, label="Petroleum ontology inference accuracy"),
    MetricSpec("petroleum_multihop_retrieval_accuracy", True, label="Petroleum ontology-guided retrieval accuracy"),
    MetricSpec("petroleum_ontology_learning_accuracy", True, label="Petroleum ontology learning accuracy"),
    MetricSpec("petroleum_ontology_reinforcement_accuracy", True, label="Petroleum ontology reinforcement accuracy"),
    MetricSpec("petroleum_ontology_feedback_accuracy", True, label="Petroleum ontology feedback accuracy"),
    MetricSpec("petroleum_learning_accuracy", True, label="Petroleum discipline learning accuracy"),
    MetricSpec("petroleum_generalisation_accuracy", True, label="Petroleum held-out generalisation accuracy"),
    MetricSpec("acceptance_gate_accuracy", True, label="Acceptance gate accuracy (held-out verdicts)"),
    MetricSpec("curriculum_coverage", True, label="Curriculum coverage"),
    MetricSpec("analogy_accuracy", True, label="Analogy accuracy (held-out judgements)"),
    MetricSpec("relation_extraction_accuracy", True, label="Relation extraction accuracy (held-out sentences)"),
    MetricSpec("polarity_accuracy", True, label="Relation polarity accuracy (held-out opposing pairs)"),
    MetricSpec("connection_accuracy", True, label="Connection explanation accuracy (held-out concept pairs)"),
    MetricSpec("consistency_accuracy", True, label="Contradiction detection accuracy (held-out graphs)"),
    MetricSpec("inference_accuracy", True, label="Graph inference accuracy (held-out entailments)"),
    MetricSpec("connection_reasoning_accuracy", True, label="Connection reasoning accuracy (held-out answer-path pairs)"),
    MetricSpec("causal_reasoning_accuracy", True, label="Causal reasoning accuracy (held-out answer-path causal pairs)"),
    MetricSpec("controller_causal_accuracy", True, label="Controller causal accuracy (held-out pairs walked on the controller's own causal graph)"),
    MetricSpec("answer_confidence_accuracy", True, label="Answer confidence accuracy (held-out term expectations)"),
    MetricSpec("grounding_accuracy", True, label="Grounding gate accuracy (held-out publish decisions)"),
    MetricSpec("reasoning_trace_accuracy", True, label="Reasoning trace accuracy (held-out trace shapes)"),
    MetricSpec("plan_accuracy", True, label="Plan accuracy (held-out plan shapes)"),
    MetricSpec("decision_recommendation_accuracy", True, label="Decision recommendation accuracy (held-out decision shapes)"),
    MetricSpec("belief_accuracy", True, label="Belief accuracy (held-out belief shapes)"),
    MetricSpec("lead_accuracy", True, label="Lead accuracy (held-out answer-body lead shapes)"),
    MetricSpec("temporal_accuracy", True, label="Temporal accuracy (held-out timeline shapes)"),
    MetricSpec("analogy_answer_accuracy", True, label="Analogy answer accuracy (held-out comparison shapes)"),
    MetricSpec("knowledge_carry_accuracy", True, label="Knowledge carry accuracy (held-out cross-session shapes)"),
    MetricSpec("answer_path_accuracy", True, label="Answer path accuracy (held-out connected-path shapes)"),
    MetricSpec("arabic_answer_path_accuracy", True, label="Arabic answer path accuracy (held-out connected-path shapes)"),
    MetricSpec("arabic_readable_body_accuracy", True, label="Arabic readable body accuracy (held-out: no identifier leaks into an Arabic body)"),
    MetricSpec("character_isolation_accuracy", True, label="Character isolation and sharing accuracy (held-out isolation/sharing judgements)"),
    MetricSpec("character_identity_accuracy", True, label="Character identity accuracy (held-out: a character's personality reaches its answer text, and the two windows share one memory and one name)"),
    MetricSpec("polish_accuracy", True, label="Universal linguistic polish accuracy (held-out: Arabic vocalized only when asked, English never, identifiers/formulas untouched, typography repaired, idempotent, no word changed, and every answer polished on the real Jarvis path)"),
    MetricSpec("character_knowledge_accuracy", True, label="Character knowledge accuracy (held-out: a character reads only its own corpus, the block is labelled and bounded, an absent corpus is honest, the mouth injects what it retrieved, and reading opens no socket)"),
    MetricSpec("memory_seeding_accuracy", True, label="Memory seeding accuracy (held-out: a character is seeded from the project's own authored knowledge, bounded and idempotent, with no invented fact)"),
    MetricSpec("development_accuracy", True, label="Cognitive development accuracy (held-out: the mind's self-model, stagnation detector, consolidation, spaced review, and bounded escalation behave as claimed)"),
    MetricSpec("cognition_catalog_accuracy", True, label="Cognition catalogue accuracy (held-out: every control is explained, a governed action names its confirmation, every CLI command resolves, and the window and catalogue cannot drift)"),
    MetricSpec("knowledge_depth_accuracy", True, label="Knowledge depth accuracy (held-out: the graph's reach is measured honestly, the harvest writes only through the bounded learner, and a bridge is a real walked path)"),
    MetricSpec("asking_loop_accuracy", True, label="Asking loop accuracy (held-out: the learner asks, the teacher always answers, no question repeats, a stop takes effect, and only a verified answer qualifies)"),
    MetricSpec("knowledge_validation_accuracy", True, label="Knowledge validation accuracy (held-out: the event bus, resource governor, validator, transactional integrator, and query engine behave as claimed)"),
    MetricSpec("memory_architecture_accuracy", True, label="Memory architecture accuracy (held-out: durable persistence, correction keeps history, archive not delete, backup/restore with a refused bad backup, corrupt-row tolerance, transaction rollback, temporal validity, explainable ranking, consolidation, and personality isolation)"),
    MetricSpec("learning_accuracy", True, label="Learning accuracy (held-out: evidenced claim integrated but unsupported refused, provenance preserved, duplicate reinforces, contradiction preserved, experience persists and its lesson rebuilt, a single success never promotes, feedback never overrides evidence, inferred bounded below explicit, silence refused, correction supported, versioning and rollback, and Natasha's learning excluded)"),
    MetricSpec("problem_decomposition_accuracy", True, label="Problem decomposition accuracy (held-out: a question is broken into the reasoning units it warrants, with negative controls for a single-clause and an empty question)"),
    MetricSpec("conclusion_verification_accuracy", True, label="Conclusion verification accuracy (held-out: a conclusion is verified as supported/contradicted/partially supported/insufficient/unverified, with negative controls)"),
    MetricSpec("reasoning_facade_accuracy", True, label="Reasoning facade accuracy (held-out: the one reasoning API routes a question to the engine its kind warrants and returns the engine's own object, with negative controls)"),
    MetricSpec("hypothesis_accuracy", True, label="Hypothesis accuracy (held-out: candidate hypotheses are generated only from what the graph licenses, ranked, with a contested candidate reported and an unknown question yielding none)"),
    MetricSpec("tool_selection_accuracy", True, label="Tool selection accuracy (held-out: a bounded chooser picks only tools the sandbox allow-list permits, and refuses a mutating tool in a read-only sandbox)"),
    MetricSpec("personality_agents_accuracy", True, label="Personality agents accuracy (held-out: seven independent personalities as agents -- capability ceilings enforced, a configuration cannot widen them, independent working contexts, memory isolation with Natasha refused, the router honest about no-suitable-personality, bounded delegation, conflicts preserved not voted away, and Natasha excluded from the aggregated training dataset)"),
    MetricSpec("training_accuracy", True, label="Training accuracy (held-out Phase 6: dataset validation/provenance/versioning/splitting, the Natasha exclusion for the project model's aggregate, model compatibility -- a GGUF is not a training base and QLoRA needs a CUDA GPU, config validation, the training job life cycle, the base model provably unchanged, and controlled promotion plus rollback)"),
    MetricSpec("training_center_accuracy", True, label="Training Center accuracy (held-out: a review status governs export, a governed pipeline stage is refused without approval, an unimplemented stage is reported unavailable, the catalogue is complete and its CLI commands resolve, a provider with no key/approval is honest, an unreportable resource reads as unknown, and a pipeline round-trips through its store)"),
    MetricSpec("training_resume_accuracy", True, label="Resumable training accuracy (a second training run continues from what the first left: the progress ledger round-trips and tolerates a corrupt file, a changed configuration invalidates a record while a raised episode budget tops it up, a finished task is loaded rather than retrained, a partial run continues to the same policy an uninterrupted run reaches, and the language model's own resume still raises the cumulative step count)"),
    MetricSpec("checkpoint_import_accuracy", True, label="Checkpoint import accuracy (held-out: a ready-made .npz weight file is recognised by its own array names -- a checkpoint with its tokenizer and optimiser-state sidecars -- while a random archive is reported unknown, the real trainer continues the imported weights, an existing checkpoint is backed up rather than clobbered, a policy goes to its task's own file and an unnamed one is refused, a fetch is gated, and an unreadable input is refused)"),
    MetricSpec("weights_pack_accuracy", True, label="Weights pack accuracy (held-out: the committed pack's manifest is read into entries and a missing/newer pack is reported, an entry verifies against its recorded digest/kind/architecture while a tampered file, a mismatched architecture, and a missing sidecar are each refused, the real trainer resumes from an installed entry, an unknown key is refused, an existing file is backed up rather than clobbered, an entry records and surfaces its provenance without inventing a field, and the whole-pack integrity verdict is complete only when every entry is verified and installed)"),
    MetricSpec("learned_weights_accuracy", True, label="Learned weights accuracy (held-out: each learned capacity tier names a real preset and the larger tier really is larger, measured; building a tier writes the checkpoint, its tokenizer sidecar, and honest provenance; a second build does not retrain over an existing checkpoint without --force; listing reads the checkpoints from their own structure and reports a foreign file as unusable; the pack packs a tier under its own key and installs it beside the base checkpoint; and an unknown tier is refused)"),
    MetricSpec("perception_bridge_accuracy", True, label="Perception bridge accuracy (held-out Phase 7: a document/PDF/image/audio source reads into the one normalized shape with its provenance, a perceived artifact becomes a citable Document and evidence, a model description is labelled a reading and weaker than a stated fact, an unsupported claim is never stored, and an unreadable/unsupported/cancelled source is refused)"),
    MetricSpec("task_execution_accuracy", True, label="Planning and execution accuracy (held-out Phase 8: a goal is decomposed into tasks each bound to a permitted tool, validated before it runs, executed under a permission and approval gate, verified against the goal's own criteria, and recovered from safely -- with negative controls: a suggested goal does not authorize, an action outside the allow-list is refused, a high-risk action needs approval, an irreversible failure is not retried, a tool's success is not the goal's, an invalid plan is never executed, and a failure is never hidden)"),
    MetricSpec("self_improvement_loop_accuracy", True, label="Controlled self-improvement accuracy (held-out Phase 9: the human-gated monitor -> detect -> propose -> isolate -> measure -> accept/reject -> deploy -> roll back loop -- with negative controls: a single failure raises nothing, a candidate that regresses is rejected, a protected path is refused, an unapproved deployment is refused, a sensitive change is refused, a source change is never automatic, the audit log is append-only, no forbidden self-modification operation is exposed, and Natasha stays out of the aggregated training data)"),
    MetricSpec("workplace_phase10_accuracy", True, label="Workplace Phase-10 accuracy (held-out: one search query reaches every record kind that carries it, a blank query returns nothing, a kind filter narrows to that kind, the row carrying the whole query ranks first, a query naming nothing yields no hit, the search adds no parallel index, a classified record is withheld from an uncleared reader, and the project listing's per-project counts are correct and gathered with a bounded number of reads -- the N+1 fix)"),
    MetricSpec("workplace_phase11_accuracy", True, label="Workplace Phase-11 accuracy (held-out: the institutional knowledge base is cumulative -- a completed project, an approved report, and a closed NCR each become a knowledge record, an in-progress project, a draft report, and an open NCR yield nothing, a second capture of the same source is skipped, the record preserves its source and inherits the source's confidentiality, an unknown source id is refused, and the capture writes to the existing knowledge table -- no second store)"),
    MetricSpec("workplace_security_accuracy", True, label="Workplace security accuracy (held-out: a role cannot perform a permission it does not hold, a per-user grant only narrows a role, an unknown role falls back to least privilege, a department- or project-scoped user cannot reach a record outside their scope, a record above the user's clearance is withheld and counted, an administrator is still bound by confidentiality, an approved record is read-only until reopened, the AI can never be an approver, and a refusal leaves an audit row -- run on the real SecurityManager over a real database, with negative controls)"),
    MetricSpec("semantic_separation", True, label="Semantic separation (held-out relations)"),
    MetricSpec("overall", True, label="Overall"),
)


@dataclass(frozen=True)
class MetricDelta:
    name: str
    baseline: float
    candidate: float
    delta: float
    regressed: bool


@dataclass(frozen=True)
class RegressionReport:
    deltas: tuple[MetricDelta, ...]

    @property
    def regressions(self) -> tuple[MetricDelta, ...]:
        return tuple(delta for delta in self.deltas if delta.regressed)

    @property
    def passed(self) -> bool:
        return not self.regressions

    def summary(self) -> str:
        lines = [
            f"{'REGRESSED' if delta.regressed else 'ok':>9}  {delta.name:<30} "
            f"{delta.baseline:.4f} -> {delta.candidate:.4f} ({delta.delta:+.4f})"
            for delta in self.deltas
        ]
        verdict = "PASS: no performance regression" if self.passed else "FAIL: performance regressed"
        return "\n".join([verdict, *lines])


def metrics_to_dict(metrics: EvaluationMetrics) -> dict[str, float]:
    return {
        spec.name: float(getattr(metrics, spec.name))
        for spec in METRIC_SPECS
    }


def compare_reports(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
    specs: tuple[MetricSpec, ...] = METRIC_SPECS,
) -> RegressionReport:
    """Compare two reports metric-by-metric and flag any regression."""
    deltas: list[MetricDelta] = []
    for spec in specs:
        baseline_value = float(getattr(baseline.metrics, spec.name))
        candidate_value = float(getattr(candidate.metrics, spec.name))
        raw = candidate_value - baseline_value
        if spec.higher_is_better:
            regressed = candidate_value < baseline_value - spec.tolerance
        else:
            regressed = candidate_value > baseline_value + spec.tolerance
        deltas.append(
            MetricDelta(
                name=spec.name,
                baseline=baseline_value,
                candidate=candidate_value,
                delta=round(raw, 6),
                regressed=regressed,
            )
        )
    return RegressionReport(tuple(deltas))


def compare_metric_dicts(
    baseline: dict[str, float],
    candidate: dict[str, float],
    specs: tuple[MetricSpec, ...] = METRIC_SPECS,
) -> RegressionReport:
    deltas: list[MetricDelta] = []
    for spec in specs:
        if spec.name not in baseline or spec.name not in candidate:
            continue
        baseline_value = float(baseline[spec.name])
        candidate_value = float(candidate[spec.name])
        if spec.higher_is_better:
            regressed = candidate_value < baseline_value - spec.tolerance
        else:
            regressed = candidate_value > baseline_value + spec.tolerance
        deltas.append(
            MetricDelta(
                name=spec.name,
                baseline=baseline_value,
                candidate=candidate_value,
                delta=round(candidate_value - baseline_value, 6),
                regressed=regressed,
            )
        )
    return RegressionReport(tuple(deltas))


def report_to_dict(report: EvaluationReport) -> dict[str, Any]:
    return {
        "version": 1,
        "metrics": metrics_to_dict(report.metrics),
        "domain_scores": dict(report.domain_scores),
        "scenarios": [
            {
                "id": result.scenario_id,
                "domain": result.domain,
                "passed": result.passed,
                "checks": dict(result.checks),
            }
            for result in report.scenarios
        ],
        "calibration": [
            {
                "lower": bucket.lower,
                "upper": bucket.upper,
                "count": bucket.count,
                "mean_confidence": bucket.mean_confidence,
                "empirical_accuracy": bucket.empirical_accuracy,
            }
            for bucket in report.calibration
        ],
    }
