from core.information_gap import InformationGap


def test_gap_priority_represents_missing_important_knowledge():
    gap = InformationGap("rain causes wet ground", 0.2, 0.8, 0.9)
    assert gap.priority == 0.54
    assert gap.resolved is False
