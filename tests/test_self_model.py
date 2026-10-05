from core.self_model import SelfModel


def test_self_model_declares_limits():
    assert "confidence is not truth" in SelfModel().limitations


def test_self_model_declares_petroleum_identity():
    model = SelfModel()
    assert model.domain == "petroleum engineering"
    assert model.identity == "petroleum engineering mind"
    assert "petroleum engineering reasoning" in model.capabilities
    assert "reservoir" in model.expertise


def test_identity_statement_mentions_domain_and_evidence():
    statement = SelfModel().identity_statement
    assert "petroleum engineering" in statement
    assert "evidence" in statement
