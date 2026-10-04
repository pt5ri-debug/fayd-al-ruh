from core.utility import Utility


def test_utility_contains_benefit_and_cost():
    assert Utility(.7, .8, .4).value == .7
