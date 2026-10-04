from core.snapshot import Snapshot


def test_snapshot_has_verifiable_checksum():
    assert Snapshot.create("state", {"beliefs": 1}).verify()
