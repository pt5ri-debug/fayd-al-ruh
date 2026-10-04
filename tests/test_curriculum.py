from core.curriculum import Curriculum, CurriculumItem


def test_curriculum_selects_available_lowest_difficulty_topic():
    items = [CurriculumItem("advanced", .8, frozenset({"basic"})), CurriculumItem("basic", .2)]
    assert Curriculum().next_item(items, set()).topic == "basic"
