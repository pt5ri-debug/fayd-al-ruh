import pytest
from core.question import Question


def test_question_requires_interrogative_text():
    assert Question("What caused rain?", "rain", .5).topic == "rain"
    with pytest.raises(ValueError):
        Question("rain", "rain", .5)
