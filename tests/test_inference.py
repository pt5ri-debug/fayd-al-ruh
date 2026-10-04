import pytest
from core.inference import Inference


def test_inference_holds_facts_and_optional_query():
    assert Inference(("rain",), "wet_ground").facts == ("rain",)
    with pytest.raises(ValueError):
        Inference(("",))
