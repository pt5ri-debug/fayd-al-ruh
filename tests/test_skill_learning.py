from core.skill_learning import SkillLearning


def test_skill_practice_updates_proficiency():
    assert SkillLearning().practice("planning", 1.).proficiency == .2
