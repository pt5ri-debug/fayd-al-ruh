"""Tests for grounding a language model's proposal into an allowed action."""

from core.model_advisor import ModelActionAdvisor
from core.model_interface import CallableModel


def _advisor(reply: str) -> ModelActionAdvisor:
    model = CallableModel("fake", lambda request: reply)
    return ModelActionAdvisor(model)


def test_advisor_selects_an_allowed_action():
    advisor = _advisor("water")
    assert advisor.select(frozenset({"dry_soil"}), ("water", "wait")) == "water"
    assert advisor.last_reply == "water"


def test_advisor_tolerates_chatty_replies():
    advisor = _advisor("I would choose: light\n")
    assert advisor.select(frozenset({"unlit_candle", "match_lit"}), ("strike", "light")) == "light"


def test_advisor_refuses_an_action_outside_the_allow_list():
    advisor = _advisor("explode")
    assert advisor.select(frozenset({"dry_soil"}), ("water", "wait")) is None


def test_advisor_returns_none_when_no_actions_are_offered():
    advisor = _advisor("water")
    assert advisor.select(frozenset({"dry_soil"}), ()) is None


def test_advisor_survives_a_failing_model():
    def boom(request):
        raise RuntimeError("model offline")

    advisor = ModelActionAdvisor(CallableModel("broken", boom))
    assert advisor.select(frozenset({"dry_soil"}), ("water",)) is None
