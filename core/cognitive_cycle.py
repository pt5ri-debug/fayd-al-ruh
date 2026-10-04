from __future__ import annotations

from dataclasses import dataclass

from core.belief import Belief
from core.evidence import Evidence
from core.evidence_processor import (
    EvidencePolarity,
    EvidenceProcessor,
)
from core.knowledge_graph import KnowledgeGraph
from core.learning_engine import (
    LearningEngine,
    LearningResult,
)
from core.memory import CognitiveMemory
from core.observation import Observation


@dataclass
class CognitiveCycleResult:
    observation_id: str
    evidence_id: str
    belief_id: str
    learning_result: LearningResult


class CognitiveCycle:
    """
    Executes one complete cognitive cycle.

    Observe
        ↓
    Process
        ↓
    Remember
        ↓
    Learn
        ↓
    Update Knowledge Graph
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        memory: CognitiveMemory,
        processor: EvidenceProcessor | None = None,
        learning_engine: LearningEngine | None = None,
    ) -> None:

        self.graph = graph
        self.memory = memory

        self.processor = (
            processor
            if processor is not None
            else EvidenceProcessor()
        )

        self.learning_engine = (
            learning_engine
            if learning_engine is not None
            else LearningEngine()
        )

    def process(
        self,
        observation: Observation,
        belief: Belief,
        polarity: EvidencePolarity,
    ) -> CognitiveCycleResult:

        # -----------------------------------------------------
        # 1. Remember the observation
        # -----------------------------------------------------

        self.memory.store(
            item_id=observation.id,
            item_type="observation",
            content=observation,
            importance=observation.confidence,
        )

        # -----------------------------------------------------
        # 2. Convert observation into evidence
        # -----------------------------------------------------

        evidence: Evidence = self.processor.process(
            observation,
            polarity,
        )

        # -----------------------------------------------------
        # 3. Store evidence
        # -----------------------------------------------------

        self.graph.add_evidence(evidence)

        self.memory.store(
            item_id=evidence.id,
            item_type="evidence",
            content=evidence,
            importance=evidence.strength,
        )

        # -----------------------------------------------------
        # 4. Ensure belief exists in the graph
        # -----------------------------------------------------

        if self.graph.get_belief(belief.id) is None:

            belief.add_evidence(evidence)

            self.graph.add_belief(belief)

        else:

            belief.add_evidence(evidence)

        # -----------------------------------------------------
        # 5. Learn from accumulated evidence
        # -----------------------------------------------------

        evidence_items = (
            self.graph.get_belief_evidence(
                belief.id
            )
        )

        learning_result = (
            self.learning_engine.learn(
                belief,
                evidence_items,
            )
        )

        # -----------------------------------------------------
        # 6. Remember the updated belief
        # -----------------------------------------------------

        self.memory.store(
            # Evidence IDs are unique per observation.  Using one here avoids
            # collisions when two cycles occur within the same clock tick.
            item_id=f"{belief.id}:state:{evidence.id}",
            item_type="belief_state",
            content={
                "belief_id": belief.id,
                "confidence": belief.confidence,
                "status": belief.status.value,
            },
            importance=belief.confidence,
        )

        return CognitiveCycleResult(
            observation_id=observation.id,
            evidence_id=evidence.id,
            belief_id=belief.id,
            learning_result=learning_result,
        )
