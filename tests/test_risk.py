from core.risk import Risk


def test_risk_contains_reason():
    assert Risk(.2, ("cost",)).reasons == ("cost",)
