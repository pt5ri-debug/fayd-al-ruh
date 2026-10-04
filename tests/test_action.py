from core.action import Action


def test_action_declares_preconditions_and_effects():
    assert "dry" in Action("dry", frozenset({"wet"}), frozenset({"dry"})).add_effects
