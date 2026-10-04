from core.serializer import Serializer


def test_serializer_handles_sets_and_round_trips_json():
    assert Serializer().loads(Serializer().dumps({"facts": {"rain"}})) == {"facts": ["rain"]}
