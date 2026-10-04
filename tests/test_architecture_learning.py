from core.architecture_learning import ArchitectureLearning


def test_architecture_learning_returns_reviewable_recommendation():
    assert ArchitectureLearning().recommend(.5, .9)[0].area == "prediction"
