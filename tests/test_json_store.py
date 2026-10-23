from storage.json_store import JsonStore


def test_json_store_saves_and_loads(tmp_path):
    store = JsonStore(tmp_path)
    store.save("state", "one", {"value": 1})
    assert store.load("state", "one") == {"value": 1}


def test_json_store_key_with_colon_round_trips(tmp_path):
    # Learned document ids look like "learned:5ab188c4"; ':' is illegal in a
    # Windows filename, so the store must encode keys rather than use them raw.
    store = JsonStore(tmp_path)
    store.save("documents", "learned:5ab188c4", {"value": 2})
    assert store.load("documents", "learned:5ab188c4") == {"value": 2}
    assert store.list_keys("documents") == ["learned:5ab188c4"]


def test_json_store_encodes_every_windows_illegal_character(tmp_path):
    store = JsonStore(tmp_path)
    for key in ("a:b", "a/b", "a\\b", "a?b", "a*b", 'a"b', "a<b", "a>b", "a|b"):
        store.save("ns", key, {"key": key})
        assert store.load("ns", key) == {"key": key}
    assert set(store.list_keys("ns")) == {
        "a:b", "a/b", "a\\b", "a?b", "a*b", 'a"b', "a<b", "a>b", "a|b",
    }


def test_json_store_reads_legacy_raw_key_files(tmp_path):
    # A file written before the fix used the raw key as the filename. Such a
    # file (possible on Linux/macOS) must still load and be listed.
    (tmp_path / "documents").mkdir()
    (tmp_path / "documents" / "learned:abc123.json").write_text(
        '{"doc_id": "learned:abc123"}', encoding="utf-8"
    )
    store = JsonStore(tmp_path)
    assert store.load("documents", "learned:abc123") == {"doc_id": "learned:abc123"}
    assert "learned:abc123" in store.list_keys("documents")
