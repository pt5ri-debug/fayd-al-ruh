from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from core.causal_link import CausalLink
from core.causal_graph import CausalGraph
from core.cognitive_state import CognitiveState
from core.evidence import Evidence
from core.knowledge_graph import KnowledgeGraph
from core.observation import Observation
from core.prediction_error import PredictionError


@dataclass
class CognitiveContext:
    """Provenance preserved as data moves through one cognitive episode."""

    observation_id: str
    source: str
    observed_at: datetime
    evidence_ids: list[str]
    belief_id: str
    id: str = field(default_factory=lambda: f"CTX-{uuid4().hex[:8]}")
    prediction_ids: list[str] = field(default_factory=list)
    prediction_error_ids: list[str] = field(default_factory=list)
    learning_target_ids: list[str] = field(default_factory=list)
    causal_link_ids: list[str] = field(default_factory=list)

    def add_prediction(self, prediction_id: str) -> None:
        if prediction_id not in self.prediction_ids:
            self.prediction_ids.append(prediction_id)

    def add_prediction_error(self, error_id: str) -> None:
        if error_id not in self.prediction_error_ids:
            self.prediction_error_ids.append(error_id)


class ContextRegistry:
    """Indexes cognitive contexts without changing the underlying cognitive conclusions."""

    def __init__(self) -> None:
        self.contexts: dict[str, CognitiveContext] = {}
        self._by_observation: dict[str, str] = {}
        self._by_prediction: dict[str, str] = {}
        self._by_error: dict[str, str] = {}

    def start(self, observation: Observation, evidence: Evidence, belief_id: str) -> CognitiveContext:
        context = CognitiveContext(
            observation_id=observation.id,
            source=observation.source,
            observed_at=observation.created_at,
            evidence_ids=[evidence.id],
            belief_id=belief_id,
        )
        self.contexts[context.id] = context
        self._by_observation[observation.id] = context.id
        return context

    def add_prediction(self, observation_id: str, prediction_id: str) -> None:
        context_id = self._by_observation.get(observation_id)
        if context_id is None:
            raise KeyError(f"Unknown observation context: {observation_id}")
        self.contexts[context_id].add_prediction(prediction_id)
        self._by_prediction[prediction_id] = context_id

    def add_prediction_error(self, error: PredictionError) -> CognitiveContext | None:
        context_id = self._by_prediction.get(error.prediction_id)
        if context_id is None:
            return None
        context = self.contexts[context_id]
        context.add_prediction_error(error.id)
        self._by_error[error.id] = context_id
        return context

    def add_learning_target(self, error_id: str, target_id: str) -> None:
        context_id = self._by_error.get(error_id)
        if context_id is not None and target_id not in self.contexts[context_id].learning_target_ids:
            self.contexts[context_id].learning_target_ids.append(target_id)

    def add_causal_link(self, link: CausalLink) -> list[CognitiveContext]:
        linked = [context for context in self.contexts.values() if set(context.evidence_ids) & set(link.evidence_ids)]
        for context in linked:
            if link.id not in context.causal_link_ids:
                context.causal_link_ids.append(link.id)
        return linked

    def for_prediction(self, prediction_id: str) -> CognitiveContext | None:
        context_id = self._by_prediction.get(prediction_id)
        return self.contexts.get(context_id) if context_id is not None else None

    def dump(self) -> dict:
        """Return JSON-safe context data; indexes are rebuilt on restore."""
        return {
            "contexts": {
                context_id: {
                    "id": context.id,
                    "observation_id": context.observation_id,
                    "source": context.source,
                    "observed_at": context.observed_at.isoformat(),
                    "evidence_ids": list(context.evidence_ids),
                    "belief_id": context.belief_id,
                    "prediction_ids": list(context.prediction_ids),
                    "prediction_error_ids": list(context.prediction_error_ids),
                    "learning_target_ids": list(context.learning_target_ids),
                    "causal_link_ids": list(context.causal_link_ids),
                }
                for context_id, context in self.contexts.items()
            }
        }

    @classmethod
    def restore(cls, data: dict) -> "ContextRegistry":
        registry = cls()
        for item in data.get("contexts", {}).values():
            context = CognitiveContext(
                id=item["id"],
                observation_id=item["observation_id"],
                source=item["source"],
                observed_at=datetime.fromisoformat(item["observed_at"]),
                evidence_ids=list(item["evidence_ids"]),
                belief_id=item["belief_id"],
                prediction_ids=list(item.get("prediction_ids", [])),
                prediction_error_ids=list(item.get("prediction_error_ids", [])),
                learning_target_ids=list(item.get("learning_target_ids", [])),
                causal_link_ids=list(item.get("causal_link_ids", [])),
            )
            registry.contexts[context.id] = context
            registry._by_observation[context.observation_id] = context.id
            for prediction_id in context.prediction_ids:
                registry._by_prediction[prediction_id] = context.id
            for error_id in context.prediction_error_ids:
                registry._by_error[error_id] = context.id
        return registry

    def validate(
        self,
        graph: KnowledgeGraph,
        state: CognitiveState,
        causal_graph: CausalGraph | None = None,
    ) -> list[str]:
        errors: list[str] = []
        known_errors = {error.id for error in state.prediction_errors}
        for context in self.contexts.values():
            for evidence_id in context.evidence_ids:
                if evidence_id not in graph.evidence:
                    errors.append(f"Context {context.id}: missing evidence {evidence_id}.")
            if context.belief_id not in graph.beliefs:
                errors.append(f"Context {context.id}: missing belief {context.belief_id}.")
            for prediction_id in context.prediction_ids:
                if prediction_id not in state.predictions:
                    errors.append(f"Context {context.id}: missing prediction {prediction_id}.")
            for error_id in context.prediction_error_ids:
                if error_id not in known_errors:
                    errors.append(f"Context {context.id}: missing prediction error {error_id}.")
            if causal_graph is not None:
                for link_id in context.causal_link_ids:
                    if link_id not in causal_graph.links:
                        errors.append(f"Context {context.id}: missing causal link {link_id}.")
        return errors
