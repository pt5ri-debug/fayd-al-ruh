"""Tests for wiring a learned policy into the core agent loop."""

import pytest

from core.action import Action
from core.agent import Agent
from core.belief import Belief
from core.cognitive_controller import CognitiveController
from core.environment import Environment
from core.evidence_processor import EvidencePolarity
from core.goal import Goal
from core.sandbox_tasks import task_by_name
from core.state import State
from ml.dataset import FeatureHasher
from ml.learner import train_policy
from ml.policy_adapter import LearnedActionPolicy, rank_actions


def _wet_success(state: State) -> bool:
    return "wet_soil" in state.facts


def _trained_wet_policy() -> LearnedActionPolicy:
    task = task_by_name("wet-garden")
    hasher = FeatureHasher(64)
    agent, _report = train_policy(task, _wet_success, episodes=150, hasher=hasher)
    return LearnedActionPolicy.from_task(task, agent, hasher)


def test_learned_policy_selects_the_effective_action():
    policy = _trained_wet_policy()
    chosen = policy.select(frozenset({"dry_soil"}), ("water", "wait"))

    assert chosen == "water"
    assert policy.select(frozenset({"dry_soil"}), ("wait",)) == "wait"
    assert policy.select(frozenset({"dry_soil"}), ()) is None


def test_rank_actions_exposes_learned_values():
    policy = _trained_wet_policy()
    ranked = rank_actions(policy.agent, policy.hasher, frozenset({"dry_soil"}), policy.action_names)

    assert ranked[0][0] == "water"
    assert ranked[0][1] > ranked[1][1]


def test_policy_adapter_validates_shape():
    policy = _trained_wet_policy()
    with pytest.raises(ValueError):
        LearnedActionPolicy(policy.agent, ("only-one",))
    with pytest.raises(ValueError):
        LearnedActionPolicy(policy.agent, policy.action_names, FeatureHasher(8))


def test_agent_loop_uses_the_learned_policy():
    policy = _trained_wet_policy()
    environment = Environment(State(frozenset({"dry_soil"})))
    agent = Agent(CognitiveController(), environment, policy=policy)
    belief = Belief("Watering makes the soil wet.")

    actions = [
        Action("wait", frozenset({"dry_soil"}), frozenset()),
        Action("water", frozenset({"dry_soil"}), frozenset({"wet_soil"})),
    ]
    cycle = agent.step(belief, EvidencePolarity.SUPPORTS, Goal("Wet the soil.", frozenset({"wet_soil"})), actions)

    assert cycle.decision.selected is not None
    assert cycle.decision.selected.action.name == "water"
    assert cycle.decision.reason == "Selected by the learned policy."
    assert "wet_soil" in environment.state.facts


def test_agent_without_policy_is_unchanged():
    environment = Environment(State(frozenset({"dry_soil"})))
    agent = Agent(CognitiveController(), environment)
    belief = Belief("Watering makes the soil wet.")
    actions = [Action("water", frozenset({"dry_soil"}), frozenset({"wet_soil"}))]

    cycle = agent.step(belief, EvidencePolarity.SUPPORTS, Goal("Wet the soil.", frozenset({"wet_soil"})), actions)

    assert cycle.decision.selected is not None
    assert cycle.decision.selected.action.name == "water"
