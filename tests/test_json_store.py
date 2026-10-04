from storage.json_store import JsonStore


def test_json_store_saves_and_loads(tmp_path):
    store = JsonStore(tmp_path); store.save("state", "one", {"value": 1})
    assert store.load("state", "one") == {"value": 1}
