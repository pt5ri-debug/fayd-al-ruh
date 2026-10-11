from storage.sqlite_store import SqliteStore


def test_sqlite_store_saves_and_loads(tmp_path):
    store = SqliteStore(tmp_path / "state.db"); store.save("state", "one", {"value": 1})
    assert store.load("state", "one") == {"value": 1}


def test_sqlite_store_deletes_a_key(tmp_path):
    store = SqliteStore(tmp_path / "state.db")
    store.save("state", "one", {"value": 1})
    assert store.delete("state", "one") is True
    assert store.load("state", "one") is None
    assert store.delete("state", "one") is False


def test_sqlite_store_delete_leaves_other_namespaces(tmp_path):
    store = SqliteStore(tmp_path / "state.db")
    store.save("a", "key", 1)
    store.save("b", "key", 2)
    assert store.delete("a", "key") is True
    assert store.load("b", "key") == 2
