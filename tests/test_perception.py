from core.environment import Environment
from core.perception import Perception


def test_perception_creates_environment_observation():
    assert Perception().perceive(Environment()).source == "environment"
