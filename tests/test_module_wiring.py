from datetime import datetime, timedelta, timezone

from core.belief import Belief
from core.causal_experiment import CausalExperiment
from core.causal_intervention import CausalIntervention
from core.cognitive_controller import CognitiveController
from core.hypothesis import Hypothesis
from core.inference import Inference
from core.inference_rule import InferenceRule
from core.learner import Learner
from core.learning_signal import LearningSignal
from core.pattern_detector import Pattern
from core.reasoning_engine import ReasoningEngine
from core.temporal_event import TemporalEvent
from core.temporal_reasoner import TemporalReasoner


def test_memory_retrieval_reaches_semantically_related_episode():
    controller = CognitiveController()
    controller.memory.store("EP-1", "observation", "heavy rain over the garden", 0.9)

    recalled = controller.recall_episodes("rain garden")

    assert [episode.id for episode in recalled] == ["EP-1"]


def test_memory_decay_preserves_frequently_recalled_episode():
    controller = CognitiveController()
    controller.memory.store("EP-1", "observation", "rain", 1.0)
    episode = controller.memory.episodic_memory.episodes["EP-1"]
    episode.access_count = 5
    episode.created_at = datetime.now(timezone.utc) - timedelta(hours=10)

    [decayed] = controller.apply_memory_decay(rate=0.1)

    assert 0.0 < decayed < 1.0


def test_reasoning_engine_summarizes_trace_and_propagates_uncertainty():
    engine = ReasoningEngine()
    trace = engine.deduce(Inference(("rain",)), [InferenceRule(("rain",), "wet")])

    summary = engine.summarize(trace)
    assert summary.conclusions == ("wet",)
    assert engine.propagate_uncertainty([0.9, 0.8]).confidence == 0.72


def test_learner_updates_hypothesis_and_pattern_confidences():
    learner = Learner()
    signal = LearningSignal("P", 0.5, 0.5, 1)
    hypothesis = Hypothesis("Rain recurs.", 0.4, "rain")
    pattern = Pattern("rain", 3, 0.4, ["a", "b", "c"])

    assert learner.learn_hypothesis(hypothesis, signal).estimated_confidence > 0.4
    assert learner.learn_pattern(pattern, signal).estimated_confidence > 0.4


def test_question_generator_opens_trackable_inquiries():
    controller = CognitiveController()
    controller.state.add_belief(Belief("Uncertain claim.", 0.3))

    inquiries = controller.open_inquiries()

    assert inquiries and inquiries[0].status.value == "open"


def test_causal_intervention_estimates_effect_from_experiment():
    experiment = CausalExperiment("rain", "wet", (0.9, 0.8), (0.2, 0.3))

    effect = CausalIntervention().estimate_effect(experiment)

    assert effect.effect == "wet"
    assert effect.effect_size > 0.0


def test_temporal_reasoner_uses_timeline_for_ordering():
    now = datetime.now(timezone.utc)
    reasoner = TemporalReasoner()
    later = TemporalEvent("sun", now + timedelta(hours=1))
    reasoner.timeline.add(TemporalEvent("rain", now))
    reasoner.timeline.add(later)

    assert [event.description for event in reasoner.timeline.ordered()] == ["rain", "sun"]
