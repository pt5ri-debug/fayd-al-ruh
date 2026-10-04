from core.inductive_reasoner import InductiveReasoner


def test_induction_marks_repeated_observation_as_tentative_pattern():
    result = InductiveReasoner().infer(["rain", "rain", "sun"])[0]
    assert result.method == "inductive"
    assert result.confidence == 0.666667
