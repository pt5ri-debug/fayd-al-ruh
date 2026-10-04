from core.observation import Observation
from core.memory import CognitiveMemory
from core.pattern_detector import PatternDetector
from core.hypothesis import Hypothesis
from core.prediction import Prediction
from core.prediction_error import PredictionErrorCalculator
from core.belief import Belief, BeliefStatus
from core.knowledge_graph import KnowledgeGraph
from core.belief_updater import BeliefUpdater


def test_end_to_end_causal_path():

    # =========================================================
    # 1. Observations
    # =========================================================

    observation_1 = Observation(
        content="The ball moved.",
        source="sensor_a",
        confidence=0.9,
    )

    observation_2 = Observation(
        content="The ball moved.",
        source="sensor_b",
        confidence=0.8,
    )

    # =========================================================
    # 2. Cognitive Memory
    # =========================================================

    memory = CognitiveMemory()

    memory.store(
        item_id=observation_1.id,
        item_type="observation",
        content=observation_1,
        importance=observation_1.confidence,
    )

    memory.store(
        item_id=observation_2.id,
        item_type="observation",
        content=observation_2,
        importance=observation_2.confidence,
    )

    assert memory.size == 2

    # =========================================================
    # 3. Pattern Detection
    # =========================================================

    detector = PatternDetector()

    patterns = detector.detect(
        memory,
        item_type="observation",
        minimum_occurrences=2,
    )

    assert len(patterns) == 1

    pattern = patterns[0]

    assert pattern.key == "the ball moved."
    assert pattern.occurrences == 2
    assert pattern.confidence == 0.85

    assert set(pattern.memory_ids) == {
        observation_1.id,
        observation_2.id,
    }

    # =========================================================
    # 4. Hypothesis
    # =========================================================

    hypothesis = Hypothesis(
        statement="The ball is moving repeatedly.",
        confidence=pattern.confidence,
        source_pattern=pattern.key,
    )

    assert hypothesis.source_pattern == pattern.key
    assert hypothesis.confidence == 0.85

    # =========================================================
    # 5. Prediction
    # =========================================================

    prediction = Prediction(
        statement="The ball will move again.",
        confidence=hypothesis.confidence,
        hypothesis_id=hypothesis.id,
    )

    assert prediction.hypothesis_id == hypothesis.id
    assert prediction.confidence == hypothesis.confidence
    assert prediction.fulfilled is None

    # =========================================================
    # 6. Prediction Resolution
    # =========================================================

    prediction.resolve(True)

    assert prediction.fulfilled is True

    # =========================================================
    # 7. Prediction Error
    # =========================================================

    error_calculator = PredictionErrorCalculator()

    prediction_error = error_calculator.calculate(
        prediction,
        actual_result=1.0,
    )

    assert prediction_error.prediction_id == prediction.id
    assert prediction_error.expected_confidence == 0.85
    assert prediction_error.actual_result == 1.0
    assert prediction_error.error == 0.15

    # =========================================================
    # 8. Convert observation into evidence
    # =========================================================

    evidence = observation_1.to_evidence()

    assert evidence.description == observation_1.content
    assert evidence.source == observation_1.source
    assert evidence.supports is True
    assert evidence.strength == observation_1.confidence

    # =========================================================
    # 9. Belief
    # =========================================================

    belief = Belief(
        statement=hypothesis.statement,
        confidence=hypothesis.confidence,
        status=BeliefStatus.PROVISIONAL,
    )

    belief.add_evidence(evidence)

    assert belief.statement == hypothesis.statement
    assert belief.confidence == 0.85
    assert belief.status == BeliefStatus.PROVISIONAL
    assert evidence.id in belief.supporting_evidence

    # =========================================================
    # 10. Knowledge Graph
    # =========================================================

    graph = KnowledgeGraph()

    graph.add_evidence(evidence)
    graph.add_belief(belief)

    assert graph.evidence_count == 1
    assert graph.belief_count == 1

    stored_belief = graph.get_belief(
        belief.id
    )

    assert stored_belief is belief

    belief_evidence = graph.get_belief_evidence(
        belief.id
    )

    assert evidence in belief_evidence

    # =========================================================
    # 11. Final graph validation
    # =========================================================

    errors = graph.validate()

    assert errors == []


def test_end_to_end_causal_path_with_failed_prediction():

    # =========================================================
    # 1. Observations
    # =========================================================

    observation_1 = Observation(
        content="The ball moved.",
        source="sensor_a",
        confidence=0.9,
    )

    observation_2 = Observation(
        content="The ball moved.",
        source="sensor_b",
        confidence=0.8,
    )

    # =========================================================
    # 2. Cognitive Memory
    # =========================================================

    memory = CognitiveMemory()

    memory.store(
        item_id=observation_1.id,
        item_type="observation",
        content=observation_1,
        importance=observation_1.confidence,
    )

    memory.store(
        item_id=observation_2.id,
        item_type="observation",
        content=observation_2,
        importance=observation_2.confidence,
    )

    # =========================================================
    # 3. Pattern
    # =========================================================

    detector = PatternDetector()

    patterns = detector.detect(
        memory,
        item_type="observation",
        minimum_occurrences=2,
    )

    assert len(patterns) == 1

    pattern = patterns[0]

    assert pattern.key == "the ball moved."
    assert pattern.occurrences == 2
    assert pattern.confidence == 0.85

    # =========================================================
    # 4. Hypothesis
    # =========================================================

    hypothesis = Hypothesis(
        statement="The ball is moving repeatedly.",
        confidence=pattern.confidence,
        source_pattern=pattern.key,
    )

    assert hypothesis.source_pattern == pattern.key

    # =========================================================
    # 5. Prediction
    # =========================================================

    prediction = Prediction(
        statement="The ball will move again.",
        confidence=hypothesis.confidence,
        hypothesis_id=hypothesis.id,
    )

    assert prediction.hypothesis_id == hypothesis.id
    assert prediction.fulfilled is None

    # =========================================================
    # 6. Prediction fails
    # =========================================================

    prediction.resolve(False)

    assert prediction.fulfilled is False

    # =========================================================
    # 7. Prediction Error
    # =========================================================

    calculator = PredictionErrorCalculator()

    prediction_error = calculator.calculate(
        prediction,
        actual_result=0.0,
    )

    assert prediction_error.prediction_id == prediction.id
    assert prediction_error.expected_confidence == 0.85
    assert prediction_error.actual_result == 0.0
    assert prediction_error.error == 0.85

    # =========================================================
    # 8. Convert observation to evidence
    # =========================================================

    evidence = observation_1.to_evidence()

    assert evidence.description == observation_1.content
    assert evidence.source == observation_1.source
    assert evidence.strength == observation_1.confidence

    # =========================================================
    # 9. Create provisional belief
    # =========================================================

    belief = Belief(
        statement=hypothesis.statement,
        confidence=hypothesis.confidence,
        status=BeliefStatus.PROVISIONAL,
    )

    belief.add_evidence(evidence)

    assert belief.confidence == 0.85
    assert belief.status == BeliefStatus.PROVISIONAL
    assert evidence.id in belief.supporting_evidence

    # =========================================================
    # 10. Store in Knowledge Graph
    # =========================================================

    graph = KnowledgeGraph()

    graph.add_evidence(evidence)
    graph.add_belief(belief)

    assert graph.evidence_count == 1
    assert graph.belief_count == 1

    # =========================================================
    # 11. Verify final graph
    # =========================================================

    stored_belief = graph.get_belief(
        belief.id
    )

    assert stored_belief is belief

    stored_evidence = graph.get_belief_evidence(
        belief.id
    )

    assert evidence in stored_evidence

    # =========================================================
    # 12. Graph integrity
    # =========================================================

    assert graph.validate() == []


def test_end_to_end_prediction_error_updates_belief():

    # =========================================================
    # 1. Observations
    # =========================================================

    observation_1 = Observation(
        content="The ball moved.",
        source="sensor_a",
        confidence=0.9,
    )

    observation_2 = Observation(
        content="The ball moved.",
        source="sensor_b",
        confidence=0.8,
    )

    # =========================================================
    # 2. Memory
    # =========================================================

    memory = CognitiveMemory()

    memory.store(
        item_id=observation_1.id,
        item_type="observation",
        content=observation_1,
        importance=observation_1.confidence,
    )

    memory.store(
        item_id=observation_2.id,
        item_type="observation",
        content=observation_2,
        importance=observation_2.confidence,
    )

    # =========================================================
    # 3. Pattern
    # =========================================================

    detector = PatternDetector()

    patterns = detector.detect(
        memory,
        item_type="observation",
        minimum_occurrences=2,
    )

    assert len(patterns) == 1

    pattern = patterns[0]

    assert pattern.key == "the ball moved."
    assert pattern.occurrences == 2
    assert pattern.confidence == 0.85

    # =========================================================
    # 4. Hypothesis
    # =========================================================

    hypothesis = Hypothesis(
        statement="The ball is moving repeatedly.",
        confidence=pattern.confidence,
        source_pattern=pattern.key,
    )

    # =========================================================
    # 5. Prediction
    # =========================================================

    prediction = Prediction(
        statement="The ball will move again.",
        confidence=hypothesis.confidence,
        hypothesis_id=hypothesis.id,
    )

    prediction.resolve(False)

    assert prediction.fulfilled is False

    # =========================================================
    # 6. Prediction Error
    # =========================================================

    calculator = PredictionErrorCalculator()

    prediction_error = calculator.calculate(
        prediction,
        actual_result=0.0,
    )

    assert prediction_error.prediction_id == prediction.id
    assert prediction_error.error == 0.85

    # =========================================================
    # 7. Belief
    # =========================================================

    evidence = observation_1.to_evidence()

    belief = Belief(
        statement=hypothesis.statement,
        confidence=hypothesis.confidence,
        status=BeliefStatus.PROVISIONAL,
    )

    belief.add_evidence(evidence)

    assert belief.confidence == 0.85

    # =========================================================
    # 8. Update belief from prediction error
    # =========================================================

    updater = BeliefUpdater()

    updated_belief = updater.update(
        belief,
        prediction_error,
    )

    assert updated_belief is belief
    assert updated_belief.confidence == 0.0

    # =========================================================
    # 9. Store updated belief in Knowledge Graph
    # =========================================================

    graph = KnowledgeGraph()

    graph.add_evidence(evidence)
    graph.add_belief(updated_belief)

    stored_belief = graph.get_belief(
        belief.id
    )

    assert stored_belief is belief
    assert stored_belief.confidence == 0.0

    # =========================================================
    # 10. Verify evidence is preserved
    # =========================================================

    belief_evidence = graph.get_belief_evidence(
        belief.id
    )

    assert evidence in belief_evidence

    # =========================================================
    # 11. Final integrity check
    # =========================================================

    assert graph.validate() == []


def test_hypothesis_to_belief_causal_link():

    hypothesis = Hypothesis(
        statement="The ball is moving repeatedly.",
        confidence=0.85,
        source_pattern="the ball moved.",
    )

    belief = Belief(
        statement=hypothesis.statement,
        confidence=hypothesis.confidence,
        status=BeliefStatus.PROVISIONAL,
        source_hypothesis_id=hypothesis.id,
    )

    assert belief.source_hypothesis_id == hypothesis.id
    assert belief.statement == hypothesis.statement
    assert belief.confidence == hypothesis.confidence