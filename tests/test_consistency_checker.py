from core.belief import Belief
from core.consistency_checker import ConsistencyChecker


def test_checker_finds_explicit_negation_conflict():
    report = ConsistencyChecker().check([Belief("rain occurred", .8), Belief("not rain occurred", .6)])
    assert report.consistent is False
    assert report.conflicts[0].severity == .6
