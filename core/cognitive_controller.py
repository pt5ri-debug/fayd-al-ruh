from __future__ import annotations

from dataclasses import dataclass

from core.belief import Belief
from core.cognitive_evaluator import (
    CognitiveEvaluation,
    CognitiveEvaluator,
)
from core.cognitive_loop import (
    CognitiveLoop,
    CognitiveLoopResult,
)
from core.cognitive_report import CognitiveReport
from core.cognitive_state import CognitiveState
from core.evidence_processor import EvidencePolarity
from core.knowledge_graph import KnowledgeGraph
from core.learning_metrics import LearningMetrics
from core.memory import CognitiveMemory, MemoryRecall
from core.observation import Observation
from core.state_history import StateHistory
from core.cognitive_trace import CognitiveTrace
from core.causal_graph import CausalGraph
from core.causal_link import CausalLink
from core.causal_reasoner import CausalReasoner
from core.causal_validator import CausalValidator
from core.causal_discovery import CausalDiscovery
from core.causal_intervention import CausalIntervention, InterventionResult
from core.counterfactual_reasoner import CounterfactualReasoner
from core.counterfactual import Counterfactual
from core.reasoning_engine import ReasoningEngine
from core.curiosity import Curiosity
from core.question_generator import QuestionGenerator
from core.learner import Learner, LearningCycle
from core.prediction_error import PredictionError
from core.belief_revision import BeliefRevision, BeliefRevisionEngine
from core.consistency_checker import ConsistencyChecker, ConsistencyReport
from core.conflict_resolver import ConflictResolver, ConflictResolution
from core.evaluation_suite import EvaluationSuite, QualityAssessment
from core.goal import Goal, PlanStep
from core.planner import Planner
from core.plan_evaluator import PlanEvaluator, PlanEvaluation
from core.action import Action
from core.state import State
from core.plan import Plan
from core.decision_engine import DecisionEngine
from core.decision_trace import DecisionTrace
from core.experiment import Experiment
from core.experiment_result import ExperimentResult
from core.experiment_runner import ExperimentRunner
from core.ontology import Ontology
from core.ontology_reasoner import OntologyInference, OntologyReasoner
from core.ontology_retrieval import OntologyGuidedRetriever
from core.domain_knowledge import domain_summary, seed_ontology
from core.semantic_parser import SemanticParser
from core.meaning_model import MeaningModel
from core.language_reasoner import LanguageReasoner
from core.concept import Concept
from core.metacognition import Metacognition, MetacognitiveReport
from core.introspection import Introspection, BeliefExplanation
from core.self_learning import SelfLearning, SelfLearningReport
from core.curriculum import CurriculumItem
from core.knowledge_store import KnowledgeStore
from core.persistence import Persistence
from core.snapshot import Snapshot
from core.explanation_engine import ExplanationEngine
from core.explanation import Explanation
from core.reasoning_trace import ReasoningTrace
from core.model_interface import CognitiveModel, ModelRequest, ModelResponse
from core.structured_output import MODEL_OUTPUT_SCHEMA
from core.llm_reasoner import LLMReasoner, ModelAnalysis
from core.safe_learning import LearningSandbox
from core.cognitive_context import ContextRegistry
from core.provenance import ProvenanceLedger, ProvenanceRecord, ProvenanceStage
from core.retention_policy import RetentionPolicy


@dataclass(frozen=True)
class CognitiveCycleResult:
    loop_result: CognitiveLoopResult
    metrics: LearningMetrics
    evaluation: CognitiveEvaluation
    report: CognitiveReport
    context_id: str | None = None

    @property
    def working_memory(self):
        """Relevant prior memories selected for this cognitive cycle."""
        return self.loop_result.recalled_memories


class CognitiveController:
    """
    Coordinates the complete cognitive process.

    The controller orchestrates the components but does not
    implement their internal reasoning logic.
    """

    def __init__(
        self,
        graph: KnowledgeGraph | None = None,
        memory: CognitiveMemory | None = None,
        state: CognitiveState | None = None,
        history: StateHistory | None = None,
    ) -> None:

        self.graph = (
            graph
            if graph is not None
            else KnowledgeGraph()
        )

        self.memory = (
            memory
            if memory is not None
            else CognitiveMemory()
        )

        self.state = (
            state
            if state is not None
            else CognitiveState()
        )

        self.history = (
            history
            if history is not None
            else StateHistory()
        )

        self.loop = CognitiveLoop(
            graph=self.graph,
            memory=self.memory,
            state=self.state,
        )

        self.evaluator = CognitiveEvaluator()
        self.causal_graph = CausalGraph()
        self.causal_reasoner = CausalReasoner()
        self.causal_validator = CausalValidator()
        self.causal_discovery = CausalDiscovery()
        self.causal_intervention = CausalIntervention()
        self.counterfactual_reasoner = CounterfactualReasoner()
        self.reasoning_engine = ReasoningEngine()
        self.curiosity = Curiosity()
        self.question_generator = QuestionGenerator()
        self.learner = Learner()
        self.belief_revision = BeliefRevisionEngine()
        self.consistency_checker = ConsistencyChecker()
        self.conflict_resolver = ConflictResolver()
        self.quality_suite = EvaluationSuite()
        self.planner = Planner()
        self.plan_evaluator = PlanEvaluator()
        self.decision_engine = DecisionEngine()
        self.experiment_runner = ExperimentRunner()
        self.ontology = Ontology()
        seed_ontology(self.ontology)
        self.ontology_reasoner = OntologyReasoner()
        self.ontology_retriever = OntologyGuidedRetriever(self.ontology, self.ontology_reasoner)
        self.semantic_parser = SemanticParser()
        self.meaning_model = MeaningModel()
        self.language_reasoner = LanguageReasoner()
        self.metacognition = Metacognition()
        self.introspection_service = Introspection()
        self.self_learning = SelfLearning()
        self.explanation_engine = ExplanationEngine()
        self.learning_sandbox = LearningSandbox()
        self.contexts = ContextRegistry()
        self.provenance = ProvenanceLedger()

    def reason(
        self,
        model: CognitiveModel,
        request: ModelRequest,
    ) -> ModelResponse:
        """Delegate language/multimodal reasoning to an injected model.

        The controller adds current episodic context but never grants the
        model autonomous tools or system access.
        """
        return model.generate(self._enrich_request(request))

    def _enrich_request(self, request: ModelRequest) -> ModelRequest:
        context = tuple(
            item.item.content
            for item in self.memory.recall(
                " ".join(part.value for part in request.inputs if part.modality == "text") or "context",
                item_types=("observation",),
                minimum_relevance=0.01,
            )
            if isinstance(item.item.content, str)
        )
        return ModelRequest(
            inputs=request.inputs,
            system_instruction=request.system_instruction,
            context=tuple(request.context) + context + (domain_summary(self.metacognition.self_model),),
            response_schema=request.response_schema,
        )

    def analyze_with_model(
        self,
        model: CognitiveModel,
        request: ModelRequest,
        belief: Belief | None = None,
    ) -> ModelAnalysis:
        """Force structured model output and register it through validation layers.

        The model is prompted for a strict JSON schema (concepts, evidence,
        hypotheses, questions, plans). Results are validated and registered by
        :class:`LLMReasoner`; the model cannot mutate belief confidence/status
        or execute tools.
        """
        structured_request = ModelRequest(
            inputs=request.inputs,
            system_instruction=request.system_instruction,
            context=request.context,
            response_schema=MODEL_OUTPUT_SCHEMA,
        )
        return LLMReasoner(self).analyze(model, structured_request, belief)

    def assess_quality(self) -> QualityAssessment:
        return self.quality_suite.assess(self.graph, self.memory, self.state)

    def explain_cause(self, cause: str, effect: str):
        """Return the strongest directed explanation currently supported."""
        return self.causal_reasoner.explain(
            self.causal_graph,
            cause,
            effect,
        )

    def discover_causal_candidates(self, ordered_events: list[str]):
        """Return candidates only; validate them before adding causal links."""
        return self.causal_discovery.discover(ordered_events)

    def intervene_without(self, cause: str, effect: str) -> InterventionResult:
        return self.causal_intervention.predict_without(self.causal_graph, cause, effect)

    def counterfactual(self, cause: str, effect: str) -> Counterfactual:
        return self.counterfactual_reasoner.what_if_not(self.causal_graph, cause, effect)

    def discover_questions(self, limit: int = 5):
        """Find high-priority uncertainty in current beliefs and phrase questions."""
        gaps = self.curiosity.discover(self.state.beliefs.values())
        return self.question_generator.generate(gaps, limit=limit)

    def open_inquiries(self, limit: int = 5):
        """Turn current information gaps into trackable, answerable inquiries."""
        gaps = self.curiosity.discover(self.state.beliefs.values())
        return self.question_generator.open_inquiries(gaps, limit=limit)

    def recall_episodes(self, query: str, limit: int = 5):
        """Retrieve past experiences through semantic relevance and association."""
        return self.memory.recall_episodes(query, limit=limit)

    def apply_memory_decay(self, rate: float = .01):
        """Apply the forgetting curve to stored experiences."""
        return self.memory.apply_decay(rate)

    def prune_memory(self, policy: RetentionPolicy | None = None):
        """Apply a long-term retention policy and evict items that fall outside it."""
        return self.memory.apply_retention(policy or RetentionPolicy())

    def learn_from_prediction_error(
        self,
        belief: Belief,
        prediction_error: PredictionError,
        action: str | None = None,
    ) -> LearningCycle:
        """Apply the new online learning system from one verified outcome."""
        self.state.add_belief(belief)
        cycle = self.learner.learn_from_prediction(belief, prediction_error, action)
        self.contexts.add_learning_target(prediction_error.id, belief.id)

        context = self.contexts.for_prediction(prediction_error.prediction_id)
        context_id = context.id if context is not None else None
        self._record_provenance(
            ProvenanceRecord.from_artifact(
                prediction_error,
                ProvenanceStage.PREDICTION_ERROR,
                source="outcome",
                context_id=context_id,
                parent_ids=[prediction_error.prediction_id],
            )
        )
        self._record_provenance(
            ProvenanceRecord.from_artifact(
                belief,
                ProvenanceStage.LEARNING,
                context_id=context_id,
                parent_ids=[prediction_error.id],
                metadata={"action": action} if action is not None else None,
            )
        )
        return cycle

    def add_causal_link(self, link: CausalLink) -> None:
        """Add an evidenced causal claim and retain its originating contexts."""
        self.causal_graph.add_link(link)
        contexts = self.contexts.add_causal_link(link)
        context_id = contexts[0].id if contexts else None
        self._record_provenance(
            ProvenanceRecord.from_artifact(
                link,
                ProvenanceStage.CAUSAL_LINK,
                context_id=context_id,
                evidence_ids=list(link.evidence_ids),
            )
        )

    def _record_provenance(self, record: ProvenanceRecord) -> None:
        """Attach a central provenance record; never overwrite existing data."""
        self.provenance.record(record)

    def provenance_for(self, entity_id: str) -> ProvenanceRecord | None:
        """Return the latest provenance record for an entity, if any."""
        return self.provenance.get(entity_id)

    def provenance_history(self, entity_id: str) -> list[ProvenanceRecord]:
        """Return the full provenance chain for an entity."""
        return self.provenance.history(entity_id)

    def validate_contexts(self) -> list[str]:
        """Verify that context references still resolve after cross-layer transitions."""
        return self.contexts.validate(self.graph, self.state)

    def revise_belief(self, belief_id: str) -> BeliefRevision:
        """Recompute confidence and uncertainty from all attached evidence."""
        belief = self.graph.get_belief(belief_id)
        if belief is None:
            raise KeyError(f"Unknown belief: {belief_id}")
        revision = self.belief_revision.revise(
            belief,
            self.graph.get_belief_evidence(belief_id),
        )
        self.state.add_belief(belief)
        return revision

    def check_consistency(self) -> ConsistencyReport:
        return self.consistency_checker.check(self.state.beliefs.values())

    def resolve_conflict(self, conflict_index: int = 0) -> ConflictResolution:
        report = self.check_consistency()
        if conflict_index < 0 or conflict_index >= len(report.conflicts):
            raise IndexError("Unknown belief conflict.")
        return self.conflict_resolver.resolve(report.conflicts[conflict_index])

    def create_plan(
        self,
        initial_state: State,
        goal: Goal,
        actions: list[Action],
    ) -> Plan | None:
        return self.planner.create_plan(initial_state, goal, actions)

    def order_plan_steps(self, goal: Goal, steps: list[PlanStep]) -> list[PlanStep]:
        """Topologically order an explicit task plan for a goal."""
        return self.planner.order_steps(goal, steps)

    def evaluate_plan(
        self,
        plan: Plan,
        goal: Goal,
        observed_state: State,
    ) -> PlanEvaluation:
        return self.plan_evaluator.evaluate(plan, goal, observed_state)

    def decide(self, state: State, goal: Goal, actions: list[Action]) -> DecisionTrace:
        """Compare declared action alternatives; selection has no side effects."""
        return self.decision_engine.decide(state, goal, actions)

    def run_experiment(self, experiment: Experiment, source, belief: Belief | None = None, action: str | None = None):
        """Complete experiment -> prediction-error -> optional belief-learning loop."""
        result: ExperimentResult = self.experiment_runner.run(experiment, source)
        learning = self.learner.learn_from_prediction(belief, result.prediction_error, action) if belief is not None else None
        return result, learning

    def bind_entity_to_concept(self, entity_id: str, concept: Concept) -> None:
        """Ground an instance in the knowledge graph to a reusable meaning."""
        entity = self.graph.get_entity(entity_id)
        if entity is None:
            raise KeyError(f"Unknown entity: {entity_id}")
        if concept.id not in self.ontology.concepts:
            self.ontology.add_concept(concept)
        entity.metadata["concept_id"] = concept.id
        entity.touch()

    def introspect(self) -> MetacognitiveReport:
        return self.metacognition.assess(self.state)

    @property
    def identity(self) -> str:
        """The declared domain identity this system reasons from."""
        return domain_summary(self.metacognition.self_model)

    def infer_ontology_relation(self, source_label: str, target_label: str, relation_type: str) -> OntologyInference | None:
        """Return an explicit or derived relation between two named concepts.

        The relation need not be stored: ``OntologyReasoner`` composes the seeded
        relations to derive it, and the result carries its composing steps and a
        propagated confidence. Derived relations are never asserted into the graph.
        """
        sources = self.ontology.find_concepts(source_label)
        targets = self.ontology.find_concepts(target_label)
        if not sources or not targets:
            return None
        return self.ontology_reasoner.infer(self.ontology, sources[0].id, targets[0].id, relation_type)

    def derive_ontology_relations(self) -> tuple[OntologyInference, ...]:
        """Return every relation the seeded ontology entails but does not store."""
        return self.ontology_reasoner.derive(self.ontology)

    def recall_with_ontology(self, query: str, *, item_types: tuple[str, ...] = ("observation",), limit: int = 5) -> list[MemoryRecall]:
        """Recall memory items that connect to the query through ontology relations.

        Combines ordinary lexical/semantic recall with a bounded walk over the
        seeded (and derived) relations, so an item grounding a merely related
        concept is reachable even without a shared word. The returned recalls
        carry the relational score in ``semantic_similarity``.
        """
        allowed = set(item_types)
        items = {
            item.item_id: self.memory._text_for(item.content)
            for item in self.memory._items.values()
            if item.item_type in allowed
        }
        base = {
            recall.item.item_id: recall.relevance
            for recall in self.memory.recall(query, item_types=item_types, limit=max(limit * 4, self.memory.size))
        }
        matches = self.ontology_retriever.rank(items, query, base_scores=base, limit=limit)
        recalls: list[MemoryRecall] = []
        for match in matches:
            item = self.memory.get(match.item_id)
            if item is None:
                continue
            recalls.append(
                MemoryRecall(
                    item=item,
                    relevance=round(max(base.get(match.item_id, 0.0), match.score), 6),
                    activation=round(match.score, 6),
                    semantic_similarity=round(match.score, 6),
                )
            )
        return recalls

    def explain_belief(self, belief_id: str) -> BeliefExplanation:
        belief = self.graph.get_belief(belief_id)
        if belief is None:
            raise KeyError(f"Unknown belief: {belief_id}")
        return self.introspection_service.explain_belief(
            belief,
            self.graph.get_belief_evidence(belief_id),
        )

    def assess_self_learning(self, curriculum: list[CurriculumItem], mastered: set[str]) -> SelfLearningReport:
        """Produce a next learning topic and reviewable improvement proposals."""
        quality = self.assess_quality()
        metrics = LearningMetrics.calculate(self.state, self.history)
        return self.self_learning.assess(curriculum, mastered, metrics.average_prediction_error, quality.memory_retrieval)

    def create_snapshot(self, name: str, persistence: Persistence) -> Snapshot:
        """Persist a verifiable graph-and-memory snapshot through a chosen adapter."""
        return KnowledgeStore(persistence).snapshot(
            name,
            self.graph,
            self.memory,
            self.state,
            self.contexts,
            self.causal_graph,
            self.provenance,
        )

    def restore_snapshot(self, name: str, persistence: Persistence) -> bool:
        """Restore a verified snapshot while preserving graph, memory, state, and contexts."""
        restored = KnowledgeStore(persistence).restore_snapshot(name)
        if restored is None:
            return False
        self.graph = restored.graph
        self.memory = restored.memory
        self.state = restored.state
        self.contexts = restored.contexts
        self.causal_graph = restored.causal_graph
        self.provenance = restored.provenance
        self.loop = CognitiveLoop(self.graph, self.memory, self.state)
        return True

    def explain(self, conclusion: str, evidence_ids: list[str] | None = None, reasoning_trace: ReasoningTrace | None = None, decision_trace: DecisionTrace | None = None, causal: tuple[str, ...] = ()) -> Explanation:
        evidence = [self.graph.evidence[item_id] for item_id in (evidence_ids or []) if item_id in self.graph.evidence]
        return self.explanation_engine.explain(conclusion, evidence, reasoning_trace, decision_trace, causal)

    def process(
        self,
        observation: Observation,
        belief: Belief,
        polarity: EvidencePolarity,
    ) -> CognitiveCycleResult:

        loop_result = self.loop.observe(
            observation=observation,
            belief=belief,
            polarity=polarity,
        )

        if loop_result.evidence_id is None:
            raise RuntimeError("Cognitive loop did not produce evidence context.")
        evidence = self.graph.get_evidence(loop_result.evidence_id)
        if evidence is None:
            raise RuntimeError("Cognitive loop produced unregistered evidence.")
        context = self.contexts.start(observation, evidence, belief.id)

        self._record_provenance(
            ProvenanceRecord.from_artifact(
                observation,
                ProvenanceStage.OBSERVATION,
                context_id=context.id,
                evidence_ids=[evidence.id],
            )
        )
        self._record_provenance(
            ProvenanceRecord.from_artifact(
                evidence,
                ProvenanceStage.EVIDENCE,
                context_id=context.id,
                evidence_ids=[evidence.id],
                parent_ids=[observation.id],
            )
        )
        self._record_provenance(
            ProvenanceRecord.from_artifact(
                belief,
                ProvenanceStage.BELIEF,
                context_id=context.id,
                evidence_ids=[evidence.id],
                parent_ids=[evidence.id],
            )
        )

        trace = CognitiveTrace(
            observation_id=observation.id,
        )

        trace.set_belief(
            belief.id
        )

        for pattern in loop_result.patterns:
            trace.add_pattern(
                pattern.key
            )

        for hypothesis in loop_result.hypotheses:
            trace.add_hypothesis(
                hypothesis.id
            )

        for prediction in loop_result.predictions:
            trace.add_prediction(
                prediction.id
            )
            self.contexts.add_prediction(observation.id, prediction.id)
            self._record_provenance(
                ProvenanceRecord.from_artifact(
                    prediction,
                    ProvenanceStage.PREDICTION,
                    source=observation.source,
                    context_id=context.id,
                    evidence_ids=[evidence.id],
                    parent_ids=[belief.id],
                )
            )

        self.state.add_trace(
            trace
        )

        self.history.record(
            self.state
        )

        metrics = LearningMetrics.calculate(
            state=self.state,
            history=self.history,
        )

        evaluation = self.evaluator.evaluate(
            metrics
        )

        report = CognitiveReport.generate(
            state=self.state,
            metrics=metrics,
            evaluation=evaluation,
        )

        return CognitiveCycleResult(
            loop_result=loop_result,
            metrics=metrics,
            evaluation=evaluation,
            report=report,
            context_id=context.id,
        )

    def evaluate_prediction(
        self,
        prediction_id: str,
        belief: Belief,
        actual_result: float,
    ) -> CognitiveCycleResult:

        prediction = self.state.predictions.get(
            prediction_id
        )

        if prediction is None:
            raise KeyError(
                f"Unknown prediction: {prediction_id}"
            )

        prediction_error, _ = self.loop.evaluate_prediction(
            prediction=prediction,
            belief=belief,
            actual_result=actual_result,
        )
        context = self.contexts.add_prediction_error(prediction_error)

        self.history.record(
            self.state
        )

        metrics = LearningMetrics.calculate(
            state=self.state,
            history=self.history,
        )

        evaluation = self.evaluator.evaluate(
            metrics
        )

        report = CognitiveReport.generate(
            state=self.state,
            metrics=metrics,
            evaluation=evaluation,
        )

        return CognitiveCycleResult(
            loop_result=CognitiveLoopResult(
                observation=None,
                patterns=[],
                hypotheses=[],
                predictions=[prediction],
                prediction_errors=[
                    prediction_error
                ],
                learning_results=[],
                recalled_memories=[],
            ),
            metrics=metrics,
            evaluation=evaluation,
            report=report,
            context_id=context.id if context is not None else None,
        )
