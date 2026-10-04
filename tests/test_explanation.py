from core.explanation import Explanation


def test_explanation_preserves_conclusion_and_sources():
    assert Explanation("wet", "because rain", ("EV-1",)).evidence == ("EV-1",)
