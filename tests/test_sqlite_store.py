from storage.sqlite_store import SqliteStore


def test_sqlite_store_saves_and_loads(tmp_path):
    store = SqliteStore(tmp_path / "state.db"); store.save("state", "one", {"value": 1})
    assert store.load("state", "one") == {"value": 1}
