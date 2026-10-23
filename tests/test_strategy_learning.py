from core.strategy_learning import StrategyLearning


def test_strategy_learning_selects_best_recorded_strategy():
    learning = StrategyLearning()
    learning.record("deduction", .9)
    learning.record("guess", .2)
    assert learning.select(["deduction", "guess"]).name == "deduction"
