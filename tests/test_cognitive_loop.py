from core.belief import Belief
from core.cognitive_loop import CognitiveLoop
from core.cognitive_state import CognitiveState
from core.evidence_processor import EvidencePolarity
from core.knowledge_graph import KnowledgeGraph
from core.memory import CognitiveMemory
from core.observation import Observation


def test_cognitive_loop_detects_pattern():

    graph = KnowledgeGraph()
    memory = CognitiveMemory()

    loop = CognitiveLoop(
        graph=graph,
        memory=memory,
    )

    belief = Belief(
        statement="The ball moves.",
        confidence=0.5,
    )

    first = Observation(
        content="The ball moved.",
        source="sensor",
        confidence=0.9,
    )

    second = Observation(
        content="The ball moved.",
        source="camera",
        confidence=0.8,
    )

    loop.observe(
        first,
        belief,
        EvidencePolarity.SUPPORTS,
    )

    result = loop.observe(
        second,
        belief,
        EvidencePolarity.SUPPORTS,
    )

    assert len(result.patterns) == 1

    assert (
        result.patterns[0].key
        == "the ball moved."
    )


def test_cognitive_loop_creates_hypothesis():

    graph = KnowledgeGraph()
    memory = CognitiveMemory()

    loop = CognitiveLoop(
        graph=graph,
        memory=memory,
    )

    belief = Belief(
        statement="The ball moves.",
        confidence=0.5,
    )

    for source in (
        "sensor_a",
        "sensor_b",
    ):

        observation = Observation(
            content="The ball moved.",
            source=source,
            confidence=0.9,
        )

        result = loop.observe(
            observation,
            belief,
            EvidencePolarity.SUPPORTS,
        )

    assert len(result.hypotheses) == 1

    assert (
        "the ball moved."
        in result.hypotheses[0].statement
    )


def test_cognitive_loop_creates_prediction():

    graph = KnowledgeGraph()
    memory = CognitiveMemory()

    loop = CognitiveLoop(
        graph=graph,
        memory=memory,
    )

    belief = Belief(
        statement="The ball moves.",
        confidence=0.5,
    )

    for source in (
        "sensor_a",
        "sensor_b",
    ):

        observation = Observation(
            content="The ball moved.",
            source=source,
            confidence=0.9,
        )

        result = loop.observe(
            observation,
            belief,
            EvidencePolarity.SUPPORTS,
        )

    assert len(result.predictions) == 1

    prediction = result.predictions[0]

    assert prediction.hypothesis_id == \
        result.hypotheses[0].id


def test_prediction_error_changes_belief():

    graph = KnowledgeGraph()
    memory = CognitiveMemory()

    loop = CognitiveLoop(
        graph=graph,
        memory=memory,
    )

    belief = Belief(
        statement="The ball moves.",
        confidence=0.9,
    )

    observation = Observation(
        content="The ball moved.",
        source="sensor",
        confidence=0.9,
    )

    loop.observe(
        observation,
        belief,
        EvidencePolarity.SUPPORTS,
    )

    prediction = loop.prediction_engine.create(
        loop.hypothesis_engine.create(
            loop.pattern_detector.detect(
                memory,
                minimum_occurrences=1,
            )[0]
        )
    )

    error, learning = (
        loop.evaluate_prediction(
            prediction=prediction,
            belief=belief,
            actual_result=0.2,
        )
    )

    assert error.error > 0.0

    assert learning.new_confidence < 0.9

def test_cognitive_loop_updates_cognitive_state():

    graph = KnowledgeGraph()
    memory = CognitiveMemory()
    state = CognitiveState()

    loop = CognitiveLoop(
        graph=graph,
        memory=memory,
        state=state,
    )

    belief = Belief(
        statement="The ball moves.",
        confidence=0.5,
    )

    first = Observation(
        content="The ball moved.",
        source="sensor_a",
        confidence=0.9,
    )

    second = Observation(
        content="The ball moved.",
        source="sensor_b",
        confidence=0.8,
    )

    loop.observe(
        first,
        belief,
        EvidencePolarity.SUPPORTS,
    )

    result = loop.observe(
        second,
        belief,
        EvidencePolarity.SUPPORTS,
    )

    assert state.cycle_count == 2
    assert state.belief_count == 1
    assert state.pattern_count == 1
    assert state.hypothesis_count == 1
    assert state.prediction_count == 1

    assert result.patterns[0].key == \
        "the ball moved."


def test_prediction_error_is_saved_in_state():

    graph = KnowledgeGraph()
    memory = CognitiveMemory()
    state = CognitiveState()

    loop = CognitiveLoop(
        graph=graph,
        memory=memory,
        state=state,
    )

    belief = Belief(
        statement="The ball moves.",
        confidence=0.9,
    )

    first = Observation(
        content="The ball moved.",
        source="sensor",
        confidence=0.9,
    )

    loop.observe(
        first,
        belief,
        EvidencePolarity.SUPPORTS,
    )

    pattern = loop.pattern_detector.detect(
        memory,
        minimum_occurrences=1,
    )[0]

    hypothesis = (
        loop.hypothesis_engine.create(
            pattern
        )
    )

    prediction = (
        loop.prediction_engine.create(
            hypothesis
        )
    )

    loop.evaluate_prediction(
        prediction=prediction,
        belief=belief,
        actual_result=0.2,
    )

    assert state.error_count == 1

    assert (
        state.prediction_errors[0]
        .prediction_id
        == prediction.id
    )
