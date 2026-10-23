from core.memory import CognitiveMemory
from core.observation import Observation
from core.pattern_detector import PatternDetector


def test_detect_repeated_observation():

    memory = CognitiveMemory()

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

    detector = PatternDetector()

    patterns = detector.detect(
        memory,
        minimum_occurrences=2,
    )

    assert len(patterns) == 1

    pattern = patterns[0]

    assert pattern.key == "the ball moved."
    assert pattern.occurrences == 2
    assert pattern.confidence == 0.85


def test_single_observation_is_not_a_pattern():

    memory = CognitiveMemory()

    observation = Observation(
        content="The ball moved.",
        source="sensor",
        confidence=0.9,
    )

    memory.store(
        item_id=observation.id,
        item_type="observation",
        content=observation,
        importance=observation.confidence,
    )

    detector = PatternDetector()

    patterns = detector.detect(
        memory,
        minimum_occurrences=2,
    )

    assert patterns == []


def test_different_observations_create_different_patterns():

    memory = CognitiveMemory()

    first = Observation(
        content="The ball moved.",
        source="sensor",
        confidence=0.9,
    )

    second = Observation(
        content="The ball stopped.",
        source="sensor",
        confidence=0.9,
    )

    memory.store(
        item_id=first.id,
        item_type="observation",
        content=first,
        importance=first.confidence,
    )

    memory.store(
        item_id=second.id,
        item_type="observation",
        content=second,
        importance=second.confidence,
    )

    detector = PatternDetector()

    patterns = detector.detect(
        memory,
        minimum_occurrences=1,
    )

    assert len(patterns) == 2


def test_pattern_contains_source_memory_ids():

    memory = CognitiveMemory()

    first = Observation(
        content="The light is on.",
        source="sensor",
        confidence=0.7,
    )

    second = Observation(
        content="The light is on.",
        source="camera",
        confidence=0.9,
    )

    memory.store(
        item_id=first.id,
        item_type="observation",
        content=first,
        importance=first.confidence,
    )

    memory.store(
        item_id=second.id,
        item_type="observation",
        content=second,
        importance=second.confidence,
    )

    detector = PatternDetector()

    patterns = detector.detect(
        memory,
        minimum_occurrences=2,
    )

    assert set(patterns[0].memory_ids) == {
        first.id,
        second.id,
    }
