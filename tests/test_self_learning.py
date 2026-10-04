from core.curriculum import CurriculumItem
from core.self_learning import SelfLearning


def test_self_learning_combines_curriculum_and_recommendations():
    report = SelfLearning().assess([CurriculumItem("basic", .1)], set(), .4, .5)
    assert report.next_topic.topic == "basic" and report.improvement_plan.recommendations
