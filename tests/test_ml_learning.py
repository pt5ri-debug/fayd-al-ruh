from ml.cognitive_brain import CognitiveBrain
from ml.dataset import FeatureHasher
from ml.learner import QLearningAgent, evaluate_policy, train_policy
from core.sandbox_tasks import task_by_name


def _wet_success(state) -> bool:
    return "wet_soil" in state.facts


def test_q_learning_solves_wet_garden():
    task = task_by_name("wet-garden")
    agent, report = train_policy(task, _wet_success, episodes=200, hasher=FeatureHasher(64))

    assert report.success_rate_window == 1.0
    assert evaluate_policy(task, _wet_success, agent, episodes=10, hasher=FeatureHasher(64)) == 1.0


def test_q_learning_solves_stack_blocks():
    task = task_by_name("stack-blocks")
    success = lambda state: "tower_built" in state.facts  # noqa: E731
    agent, report = train_policy(task, success, episodes=200, hasher=FeatureHasher(64))

    assert report.success_rate_window == 1.0
    assert evaluate_policy(task, success, agent, episodes=10, hasher=FeatureHasher(64)) == 1.0


def test_q_learning_improves_over_random():
    task = task_by_name("stack-blocks")
    success = lambda state: "tower_built" in state.facts  # noqa: E731
    _agent, report = train_policy(task, success, episodes=200, hasher=FeatureHasher(64))

    # The tail of the learning curve (after exploration decays) must beat the
    # first exploratory episodes, and epsilon must have decayed.
    tail = report.learning_curve[-50:]
    assert max(tail) >= 0.9
    assert report.episodes[-1].epsilon <= report.episodes[0].epsilon
    assert report.success_rate_window == 1.0


def test_q_learning_agent_returns_valid_actions():
    agent = QLearningAgent(4, 3, seed=0)
    features = [0.1, 0.2, 0.3, 0.4]
    assert 0 <= agent.act(features) < 3
    assert 0 <= agent.act(features, greedy=True) < 3


def test_cognitive_brain_reduces_prediction_error():
    task = task_by_name("wet-garden")
    brain = CognitiveBrain(task, _wet_success, seed=0)
    report = brain.learn(episodes=150)

    assert report.final_error <= report.initial_error
    assert report.policy_success_rate == 1.0
    assert report.memory_size > 0


def test_cognitive_brain_self_tunes_learning_rate():
    task = task_by_name("wet-garden")
    brain = CognitiveBrain(task, _wet_success, learning_rate=0.05, seed=0)
    initial_rate = brain.learning_rate
    brain.learn(episodes=120)

    assert brain.learning_rate != initial_rate
    assert 0.005 <= brain.learning_rate <= 0.5


def test_cognitive_brain_metacognition_tracks_errors():
    task = task_by_name("wet-garden")
    brain = CognitiveBrain(task, _wet_success, seed=0)
    brain.run_episode()
    state = brain.reflect()

    assert state.average_error >= 0.0
    assert 0.0 <= state.confidence <= 1.0
