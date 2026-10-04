import numpy as np

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
    assert brain.min_learning_rate <= brain.learning_rate <= brain.max_learning_rate


def test_cognitive_brain_metacognition_tracks_errors():
    task = task_by_name("wet-garden")
    brain = CognitiveBrain(task, _wet_success, seed=0)
    brain.run_episode()
    state = brain.reflect()

    assert state.average_error >= 0.0
    assert 0.0 <= state.confidence <= 1.0


def test_cognitive_brain_stays_numerically_stable_over_long_training():
    """Regression: the world model must never diverge, however long it trains."""
    task = task_by_name("light-candle")
    success = lambda state: "lit_candle" in state.facts  # noqa: E731
    brain = CognitiveBrain(task, success, seed=0)

    report = brain.learn(episodes=1500)

    assert all(error < 1e6 for error in report.error_curve)
    assert report.final_error <= report.initial_error
    assert brain.min_learning_rate <= brain.learning_rate <= brain.max_learning_rate


def test_replay_buffer_actually_shapes_the_world_model():
    """Replay must change the weights, not just log experience."""
    task = task_by_name("wet-garden")
    brain = CognitiveBrain(task, _wet_success, seed=0, replay_batch_size=16)
    brain.run_episode()
    before = brain.world_model.W.copy()

    assert len(brain.memory) > 0
    error = brain.replay()

    assert error >= 0.0
    assert not np.allclose(before, brain.world_model.W)


def test_replay_is_a_noop_when_disabled():
    task = task_by_name("wet-garden")
    brain = CognitiveBrain(task, _wet_success, seed=0, replay_batch_size=0)
    brain.run_episode()
    before = brain.world_model.W.copy()

    assert brain.replay() == 0.0
    assert np.allclose(before, brain.world_model.W)


def test_brain_persists_and_continues_learning(tmp_path):
    task = task_by_name("wet-garden")
    brain = CognitiveBrain(task, _wet_success, seed=0, replay_batch_size=8)
    brain.learn(episodes=30)

    brain.save(tmp_path)
    restored = CognitiveBrain.load(tmp_path, task, _wet_success)

    assert restored.episodes_trained == brain.episodes_trained
    assert len(restored.memory) == len(brain.memory)
    assert np.allclose(restored.world_model.W, brain.world_model.W)

    restored.learn(episodes=20)
    assert restored.episodes_trained == brain.episodes_trained + 20


def test_replay_speeds_up_world_model_convergence():
    """Consolidating memory should reach a low error in fewer episodes."""
    task = task_by_name("wet-garden")

    def episodes_to(threshold: float, replay: int, seeds=(0, 1, 2)) -> float:
        reached = []
        for seed in seeds:
            brain = CognitiveBrain(task, _wet_success, seed=seed, replay_batch_size=replay)
            episode = 120
            for index in range(1, 121):
                brain.run_episode()
                brain.episodes_trained += 1
                if brain.reflect().recent_error < threshold:
                    episode = index
                    break
            reached.append(episode)
        return sum(reached) / len(reached)

    threshold = 0.001
    assert episodes_to(threshold, replay=32) < episodes_to(threshold, replay=0)
