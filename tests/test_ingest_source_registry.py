"""Tests for the persisted source registry (registration, provenance, state)."""

from __future__ import annotations

import pytest

from ingest.source_registry import (
    DONE,
    LOCAL,
    PENDING,
    REMOTE,
    SourceRegistration,
    SourceRegistry,
    source_registry_path,
    source_slug,
)


def test_source_registration_rejects_bad_fields():
    with pytest.raises(ValueError):
        SourceRegistration(source_id="", uri="u")
    with pytest.raises(ValueError):
        SourceRegistration(source_id="s", uri="")
    with pytest.raises(ValueError):
        SourceRegistration(source_id="s", uri="u", origin="mars")
    with pytest.raises(ValueError):
        SourceRegistration(source_id="s", uri="u", status="maybe")


def test_source_slug_is_stable_and_safe():
    assert source_slug("/data/Oil Wells.csv") == "data-oil-wells-csv"
    assert source_slug("   ") == "source"
    assert source_slug("https://example.com/a?b=c") == "https-example-com-a-b-c"


def test_registry_register_get_list_remove(tmp_path):
    registry = SourceRegistry(source_registry_path(tmp_path))
    registration = registry.add("/data/oil.csv", connector="files", tags=("oil",))
    assert registration.source_id == "data-oil-csv"
    assert registry.get(registration.source_id) is registration or registry.get(
        registration.source_id
    ).uri == "/data/oil.csv"
    assert [item.uri for item in registry.list()] == ["/data/oil.csv"]
    assert registry.remove(registration.source_id) is True
    assert registry.remove(registration.source_id) is False
    assert registry.get(registration.source_id) is None


def test_registry_external_source_is_not_trusted_by_default(tmp_path):
    registry = SourceRegistry(source_registry_path(tmp_path))
    registration = registry.add("https://spe.org/x", origin=REMOTE, provider="SPE")
    assert registration.trusted is False
    assert registration.status == PENDING
    assert registration.origin == REMOTE


def test_registry_mark_run_keeps_identity_and_records_outcome(tmp_path):
    registry = SourceRegistry(source_registry_path(tmp_path))
    registration = registry.add("/data/a.txt", connector="files", title="A")
    updated = registration.mark_run(DONE, chunks=3, bytes_written=120, documents=1)
    registry.register(updated)
    stored = registry.get(registration.source_id)
    assert stored.status == DONE
    assert stored.chunks == 3
    assert stored.documents == 1
    assert stored.last_run  # timestamp set
    assert stored.uri == "/data/a.txt"  # registration preserved
    assert stored.connector == "files"


def test_registry_round_trips_options_and_tags(tmp_path):
    registry = SourceRegistry(source_registry_path(tmp_path))
    registry.add(
        "sqlite:///db.sqlite",
        origin=LOCAL,
        connector="sql",
        tags=("wells", "legacy"),
        options={"table": "wells", "chunk": 500},
    )
    stored = registry.list()[0]
    assert stored.options["table"] == "wells"
    assert stored.tags == ("wells", "legacy")


def test_registry_list_skips_unreadable_records(tmp_path):
    from storage.sqlite_store import SqliteStore

    path = source_registry_path(tmp_path)
    registry = SourceRegistry(path)
    registry.add("/keep.txt")
    SqliteStore(path).save("sources", "broken", {"uri": "no-id"})  # missing source_id
    listed = registry.list()
    assert [item.uri for item in listed] == ["/keep.txt"]
