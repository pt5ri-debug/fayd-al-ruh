from core.state import State


def test_state_satisfies_required_facts():
    assert State(frozenset({"wet", "rain"})).satisfies(frozenset({"wet"}))
