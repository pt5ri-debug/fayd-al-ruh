from core.information_gap import InformationGap
from core.question_generator import QuestionGenerator


def test_generator_prioritises_unresolved_gaps():
    questions = QuestionGenerator().generate([InformationGap("low", .6, .8, .2), InformationGap("high", .0, .8, 1.0)])
    assert questions[0].topic == "high"
    assert questions[0].text.endswith("?")
