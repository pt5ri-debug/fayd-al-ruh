from core.self_model import SelfModel


def test_self_model_declares_limits():
    assert "confidence is not truth" in SelfModel().limitations
