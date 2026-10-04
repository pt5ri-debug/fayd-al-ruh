from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from core.concept import Concept
from core.evidence import Evidence
from core.goal import Goal, PlanStep
from core.hypothesis import Hypothesis
from core.model_interface import CognitiveModel, ModelRequest, ModelResponse
from core.provenance import ProvenanceRecord, ProvenanceStage
from core.question import Question
from core.structured_output import (
    ConceptProposal,
    PlanProposal,
    StructuredProposal,
    parse_structured_output,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from core.cognitive_controller import CognitiveController


@dataclass(frozen=True)
class OrderedPlan:
    """A goal with a topologically ordered, non-executed task plan."""

    goal: Goal
    steps: tuple[PlanStep, ...]


@dataclass
class ModelAnalysis:
    """Result of validating and registering a model's structured proposal."""

    response: ModelResponse
    proposal: StructuredProposal
    concepts: list[Concept] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    plans: list[OrderedPlan] = field(default_factory=list)
    context_id: str | None = None

    @property
    def registered_ids(self) -> list[str]:
        return [
            artifact.id  # type: ignore[attr-defined]
            for artifact in (*self.concepts, *self.evidence, *self.hypotheses)
        ]


class LLMReasoner:
    """Validates model proposals and registers them through existing layers.

    The model is treated as an untrusted proposer. It can suggest concepts,
    evidence, hypotheses, questions, and plans, but this bridge:

    * parses and type-checks the JSON response (rejecting forbidden keys),
    * registers evidence, hypotheses, concepts, and plans only via the
      knowledge graph, cognitive state, ontology, and planner,
    * never changes a belief's confidence or status (that stays with
      ``BeliefRevisionEngine`` and the controller), and
    * never executes a tool or plan (plans are ordered, never run).
    """

    def __init__(self, controller: CognitiveController) -> None:
        self._controller = controller

    def analyze(
        self,
        model: CognitiveModel,
        request: ModelRequest,
        belief=None,
    ) -> ModelAnalysis:
        response = self._controller.reason(model, request)
        proposal = parse_structured_output(response.text)
        analysis = ModelAnalysis(response=response, proposal=proposal)

        for concept_item in proposal.concepts:
            concept = self._register_concept(concept_item)
            analysis.concepts.append(concept)
            self._record(concept, "model:concept")

        for evidence_item in proposal.evidence:
            evidence = Evidence(
                evidence_item.description,
                evidence_item.source,
                evidence_item.supports,
                evidence_item.strength,
            )
            self._controller.graph.add_evidence(evidence)
            if belief is not None:
                # Evidence attachment is allowed; confidence/status are not touched.
                belief.add_evidence(evidence)
            analysis.evidence.append(evidence)
            self._record(evidence, "model:evidence")

        for hypothesis_item in proposal.hypotheses:
            hypothesis = Hypothesis(hypothesis_item.statement, hypothesis_item.confidence, "model")
            self._controller.state.add_hypothesis(hypothesis)
            analysis.hypotheses.append(hypothesis)
            self._record(hypothesis, "model:hypothesis")

        for question_item in proposal.questions:
            text_value = question_item.text if question_item.text.endswith("?") else f"{question_item.text}?"
            analysis.questions.append(
                Question(text_value, question_item.topic or "model", question_item.priority)
            )

        for plan_item in proposal.plans:
            analysis.plans.append(self._register_plan(plan_item))

        return analysis

    def _register_concept(self, item: ConceptProposal) -> Concept:
        existing = self._controller.ontology.find_concepts(item.label)
        if existing:
            return existing[0]
        concept = Concept(item.label, frozenset(item.aliases))
        self._controller.ontology.add_concept(concept)
        return concept

    def _register_plan(self, item: PlanProposal) -> OrderedPlan:
        goal = Goal(item.goal, priority=item.priority)
        steps = [
            PlanStep(step.description, step.depends_on)
            for step in item.steps
        ]
        ordered = self._controller.planner.order_steps(goal, steps)
        return OrderedPlan(goal, tuple(ordered))

    def _record(self, artifact: Any, source: str) -> None:
        self._controller._record_provenance(
            ProvenanceRecord.from_artifact(
                artifact,
                ProvenanceStage.MODEL_PROPOSAL,
                source=source,
            )
        )
