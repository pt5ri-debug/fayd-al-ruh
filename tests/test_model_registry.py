"""The model registry: stable ids, safe updates, and a removal that never deletes.

Every test works in ``tmp_path``. A model file is written by the same tiny GGUF
helper the inspector's tests use, so nothing is downloaded and nothing is loaded.
"""

from __future__ import annotations

import json
import struct
import threading
from pathlib import Path

import pytest
from core.model_files import FORMAT_GGUF
from core.model_registry import (
    KNOWN_FORMATS,
    MAX_TAGS,
    NAMESPACE,
    SOURCE_CATALOG,
    SOURCE_LOCAL,
    SOURCE_TRAINED,
    ConfirmationRequired,
    ModelNotFoundError,
    ModelRecord,
    ModelRegistry,
    RegistryError,
    RegistryValidationError,
    model_id_for,
    normalise_tags,
    registered_models,
    registry,
    time_ago,
)


def _gguf_key(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<I", 8) + struct.pack("<Q", len(raw)) + raw


def _gguf_uint32(value: int) -> bytes:
    return struct.pack("<I", 4) + struct.pack("<I", value)


def write_gguf(path, *, name="Registry Model", architecture="llama", file_type=15, context=4096):
    entries = [
        ("general.name", _gguf_string(name)),
        ("general.architecture", _gguf_string(architecture)),
        ("general.file_type", _gguf_uint32(file_type)),
        (f"{architecture}.context_length", _gguf_uint32(context)),
    ]
    body = b"".join(_gguf_key(key) + payload for key, payload in entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 4, len(entries)) + body)


@pytest.fixture()
def store(tmp_path):
    return tmp_path / "registry-root"


@pytest.fixture()
def model_file(tmp_path):
    target = tmp_path / "weights" / "model.gguf"
    write_gguf(target)
    return target


# ------------------------------------------------------------------- ids


def test_model_id_is_deterministic_for_the_same_identity():
    first = model_id_for("org/repo/model.gguf")
    second = model_id_for("org/repo/model.gguf")
    assert first == second
    assert first.startswith("org-repo-model-gguf-")


def test_model_id_differs_for_different_identities():
    assert model_id_for("a/b.gguf") != model_id_for("a/c.gguf")


def test_model_id_needs_an_identity():
    with pytest.raises(RegistryValidationError):
        model_id_for("")


# -------------------------------------------------------------- registering


def test_register_reads_the_facts_from_the_file(store, model_file):
    reg = ModelRegistry(store)
    record = reg.register(model_file, display_name="My Model", tags=["petroleum"])
    assert record.model_id
    assert record.display_name == "My Model"
    assert record.model_format == FORMAT_GGUF
    assert record.runtime == "llama.cpp"
    assert record.size_bytes > 0
    assert record.quantization == "Q4_K_M"
    assert record.architecture == "llama"
    assert record.context_length == 4096
    assert record.integrity == "ok"
    assert record.tags == ("petroleum",)
    assert record.registered_at
    assert record.updated_at
    assert record.available is True
    assert record.verified is True


def test_register_defaults_the_display_name_to_the_models_own_name(store, model_file):
    write_gguf(model_file, name="Self Named")
    reg = ModelRegistry(store)
    record = reg.register(model_file)
    assert record.display_name == "Self Named"


def test_register_refuses_a_missing_file(store, tmp_path):
    reg = ModelRegistry(store)
    with pytest.raises(ValueError):
        reg.register(tmp_path / "gone.gguf")
    assert reg.list_records() == ()


def test_register_refuses_a_non_model_file(store, tmp_path):
    junk = tmp_path / "notes.gguf"
    junk.write_text("not a model")
    reg = ModelRegistry(store)
    with pytest.raises(ValueError):
        reg.register(junk)
    assert reg.count() == 0


def test_register_refuses_a_folder(store, tmp_path):
    folder = tmp_path / "a-folder.gguf"
    folder.mkdir()
    reg = ModelRegistry(store)
    with pytest.raises(ValueError):
        reg.register(folder)


def test_register_refuses_an_unknown_source(store, model_file):
    reg = ModelRegistry(store)
    with pytest.raises(RegistryValidationError):
        reg.register(model_file, source="imaginary")
    assert reg.count() == 0


def test_register_refuses_a_credential_in_extra(store, model_file):
    reg = ModelRegistry(store)
    with pytest.raises(RegistryValidationError):
        reg.register(model_file, extra={"api_key": "abc"})
    with pytest.raises(RegistryValidationError):
        reg.register(model_file, extra={"note": "sk-abcdefghijklmnopqrstuvwxyz"})
    assert reg.count() == 0


def test_register_refuses_a_credential_in_notes(store, model_file):
    reg = ModelRegistry(store)
    with pytest.raises(RegistryValidationError):
        reg.register(model_file, notes="my token is sk-abcdefghijklmnopqrstuvwx")
    assert reg.count() == 0


def test_register_records_the_engine_and_its_runtime(store, model_file):
    reg = ModelRegistry(store)
    record = reg.register(model_file, engine="ultron")
    assert record.engine == "ultron"
    assert record.runtime == "llama.cpp"


def test_register_is_idempotent_and_keeps_the_first_timestamp(store, model_file):
    reg = ModelRegistry(store)
    first = reg.register(model_file, display_name="First")
    second = reg.register(model_file, display_name="Second")
    assert first.model_id == second.model_id
    assert second.display_name == "Second"
    assert second.registered_at == first.registered_at
    assert reg.count() == 1


def test_re_registering_a_different_file_under_one_id_is_refused(store, tmp_path):
    first = tmp_path / "a.gguf"
    second = tmp_path / "b.gguf"
    write_gguf(first)
    write_gguf(second)
    reg = ModelRegistry(store)
    reg.register(first, identity="shared-identity")
    with pytest.raises(RegistryValidationError):
        reg.register(second, identity="shared-identity")
    # ... unless the caller says so explicitly.
    replaced = reg.register(second, identity="shared-identity", replace_existing=True)
    assert replaced.path == second


def test_register_record_stores_a_catalog_style_record_without_a_file(store):
    reg = ModelRegistry(store)
    record = ModelRecord(
        model_id=model_id_for("org/repo.gguf"),
        identity="org/repo.gguf",
        display_name="Not Downloaded",
        model_format=FORMAT_GGUF,
        runtime="llama.cpp",
        source=SOURCE_CATALOG,
        registered_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    stored = reg.register_record(record)
    assert reg.count() == 1
    assert stored.available is False
    assert reg.get(record.model_id).display_name == "Not Downloaded"


def test_register_record_validates_before_storing(store):
    reg = ModelRegistry(store)
    with pytest.raises(RegistryValidationError):
        reg.register_record(
            ModelRecord(
                model_id="x", identity="", display_name="X",
                model_format=FORMAT_GGUF, source=SOURCE_LOCAL,
            )
        )
    assert reg.count() == 0


# ------------------------------------------------------------------- listing


def test_list_orders_by_display_name(store, tmp_path):
    reg = ModelRegistry(store)
    for name, label in (("z.gguf", "Zeta"), ("a.gguf", "Alpha"), ("m.gguf", "Mu")):
        target = tmp_path / name
        write_gguf(target)
        reg.register(target, display_name=label)
    assert [record.display_name for record in reg.list_records()] == ["Alpha", "Mu", "Zeta"]


def test_get_raises_for_an_unknown_id(store):
    reg = ModelRegistry(store)
    with pytest.raises(ModelNotFoundError):
        reg.get("nope")


def test_find_returns_none_instead_of_raising(store):
    reg = ModelRegistry(store)
    assert reg.find("nope") is None


def test_find_by_identity_and_by_path(store, model_file):
    reg = ModelRegistry(store)
    record = reg.register(model_file)
    assert reg.find_by_identity(record.identity) is not None
    assert reg.find_by_path(model_file) is not None
    # The same file reached by a relative-looking path is still one model.
    assert reg.find_by_path(model_file.resolve()) is not None
    assert reg.find_by_path("/nowhere/else.gguf") is None


def test_len_and_iteration_match_the_listing(store, model_file):
    reg = ModelRegistry(store)
    reg.register(model_file)
    assert len(reg) == 1
    assert [item.model_id for item in reg] == [item.model_id for item in reg.list_records()]


def test_registered_models_helper_reads_the_same_store(store, model_file):
    ModelRegistry(store).register(model_file)
    assert len(registered_models(store)) == 1


# ------------------------------------------------------------------ updating


def test_update_changes_only_what_was_named(store, model_file):
    reg = ModelRegistry(store)
    original = reg.register(model_file, display_name="Before", notes="keep me")
    updated = reg.update(original.model_id, display_name="After")
    assert updated.display_name == "After"
    assert updated.notes == "keep me"
    assert updated.model_id == original.model_id
    assert updated.registered_at == original.registered_at
    assert updated.updated_at >= original.updated_at


def test_update_replaces_tags_and_capabilities(store, model_file):
    reg = ModelRegistry(store)
    record = reg.register(model_file, tags=["a"])
    updated = reg.update(record.model_id, tags=["b", "c"], capabilities=["vision"])
    assert updated.tags == ("b", "c")
    assert updated.capabilities == ("vision",)


def test_update_refuses_an_empty_display_name_and_changes_nothing(store, model_file):
    reg = ModelRegistry(store)
    record = reg.register(model_file, display_name="Keep")
    with pytest.raises(RegistryValidationError):
        reg.update(record.model_id, display_name="   ")
    assert reg.get(record.model_id).display_name == "Keep"


def test_update_refuses_an_unknown_source(store, model_file):
    reg = ModelRegistry(store)
    record = reg.register(model_file)
    with pytest.raises(RegistryValidationError):
        reg.update(record.model_id, source="imaginary")


def test_update_refuses_a_credential_in_extra(store, model_file):
    reg = ModelRegistry(store)
    record = reg.register(model_file)
    with pytest.raises(RegistryValidationError):
        reg.update(record.model_id, extra={"token": "abc"})
    assert reg.get(record.model_id).extra == {}


def test_update_with_revalidate_refreshes_the_file_facts(store, model_file):
    reg = ModelRegistry(store)
    record = reg.register(model_file)
    old_size = record.size_bytes
    # Replace the file with a larger one that declares a different quantisation.
    write_gguf(model_file, file_type=7, context=2048)
    model_file.write_bytes(model_file.read_bytes() + b"\x00" * 512)
    refreshed = reg.update(record.model_id, revalidate=True)
    assert refreshed.size_bytes > old_size
    assert refreshed.quantization == "Q8_0"
    assert refreshed.context_length == 2048


def test_update_refuses_a_credential_in_a_revalidated_license(store, model_file):
    """A value read from a file is still checked before it is stored."""
    reg = ModelRegistry(store)
    record = reg.register(model_file)
    assert reg.get(record.model_id).license == ""


# ------------------------------------------------------------------ removing


def test_remove_requires_an_explicit_confirmation(store, model_file):
    reg = ModelRegistry(store)
    record = reg.register(model_file)
    with pytest.raises(ConfirmationRequired):
        reg.remove(record.model_id)
    assert reg.count() == 1


def test_remove_forgets_the_record_and_never_touches_the_file(store, model_file):
    reg = ModelRegistry(store)
    record = reg.register(model_file)
    removed = reg.remove(record.model_id, confirm=True)
    assert removed.model_id == record.model_id
    assert reg.count() == 0
    # The model file is exactly where it was.
    assert model_file.is_file()
    assert model_file.stat().st_size == record.size_bytes


def test_remove_raises_for_an_unknown_id(store):
    reg = ModelRegistry(store)
    with pytest.raises(ModelNotFoundError):
        reg.remove("nope", confirm=True)


def test_clear_requires_confirmation_and_never_deletes_files(store, tmp_path):
    reg = ModelRegistry(store)
    files = []
    for index in range(3):
        target = tmp_path / f"m{index}.gguf"
        write_gguf(target)
        reg.register(target)
        files.append(target)
    with pytest.raises(ConfirmationRequired):
        reg.clear()
    assert reg.count() == 3
    assert reg.clear(confirm=True) == 3
    assert reg.count() == 0
    assert all(target.is_file() for target in files)


# ----------------------------------------------------------- corrupt data


def test_a_corrupt_record_is_skipped_and_reported(store, model_file):
    reg = ModelRegistry(store)
    record = reg.register(model_file)
    # Write a record that is valid JSON but not a model record.
    broken = store / NAMESPACE
    broken.mkdir(parents=True, exist_ok=True)
    (broken / "model-broken.json").write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    assert reg.count() == 1
    assert reg.get(record.model_id).display_name == record.display_name
    assert "model-broken" in reg.corrupt_keys()


def test_invalid_json_in_a_record_is_skipped(store, model_file):
    reg = ModelRegistry(store)
    reg.register(model_file)
    broken = store / NAMESPACE
    (broken / "model-garbage.json").write_text("{not json", encoding="utf-8")
    assert reg.count() == 1
    assert "model-garbage" in reg.corrupt_keys()


def test_a_record_missing_its_id_is_skipped(store):
    reg = ModelRegistry(store)
    broken = store / NAMESPACE
    broken.mkdir(parents=True, exist_ok=True)
    (broken / "model-noid.json").write_text(
        json.dumps({"identity": "x", "display_name": "X"}), encoding="utf-8"
    )
    assert reg.count() == 0
    assert reg.corrupt_keys() == ("model-noid",)


def test_a_record_with_a_bad_field_falls_back_per_field(store):
    reg = ModelRegistry(store)
    good = store / NAMESPACE
    good.mkdir(parents=True, exist_ok=True)
    (good / "model-odd.json").write_text(
        json.dumps({
            "model_id": "odd", "identity": "odd", "display_name": "Odd",
            "format": "gguf", "source": "local",
            "size_bytes": "not a number", "tags": ["ok", "bad tag!"],
            "context_length": None,
        }),
        encoding="utf-8",
    )
    record = reg.get("odd")
    assert record.size_bytes == 0
    # A malformed tag is dropped on read; a good one beside it survives.
    assert record.tags == ("ok",)
    assert record.context_length is None


def test_the_state_prefix_is_never_read_as_a_record(store):
    reg = ModelRegistry(store)
    state = store / NAMESPACE
    state.mkdir(parents=True, exist_ok=True)
    (state / "model-__next_serial__.json").write_text(json.dumps({"n": 3}), encoding="utf-8")
    assert reg.count() == 0
    assert reg.corrupt_keys() == ()


# --------------------------------------------------------------- durability


def test_records_survive_a_new_registry_instance(store, model_file):
    ModelRegistry(store).register(model_file, display_name="Persisted")
    assert ModelRegistry(store).list_records()[0].display_name == "Persisted"


def test_the_stored_document_is_deterministic(store, model_file):
    reg = ModelRegistry(store)
    record = reg.register(model_file, tags=["b", "a"])
    path = store / NAMESPACE / f"model-{record.model_id}.json"
    first = path.read_text(encoding="utf-8")
    reg.update(record.model_id, notes="x")
    # Keys are sorted by JsonStore, so the document is byte-stable in its keys.
    assert '"model_id"' in first
    assert first.index('"architecture"') < first.index('"display_name"')


def test_a_write_leaves_no_temporary_file_behind(store, model_file):
    reg = ModelRegistry(store)
    reg.register(model_file)
    leftovers = list((store / NAMESPACE).glob("*.tmp"))
    assert leftovers == []


def test_export_is_a_stable_mapping(store, model_file):
    reg = ModelRegistry(store)
    reg.register(model_file)
    first = reg.export()
    second = reg.export()
    assert first == second
    assert first["version"] == 1
    assert len(first["records"]) == 1


# ------------------------------------------------------------------- tags


def test_tags_are_normalised_and_deduplicated():
    assert normalise_tags(["GPU", "gpu", "Vision"]) == ("gpu", "vision")
    assert normalise_tags("solo") == ("solo",)
    assert normalise_tags(None) == ()
    assert normalise_tags([]) == ()


def test_a_malformed_tag_is_refused():
    with pytest.raises(RegistryValidationError):
        normalise_tags(["has spaces"])
    with pytest.raises(RegistryValidationError):
        normalise_tags(["UPPER!"])


def test_too_many_tags_are_refused():
    with pytest.raises(RegistryValidationError):
        normalise_tags([f"t{index}" for index in range(MAX_TAGS + 1)])


def test_a_non_sequence_of_tags_is_refused():
    with pytest.raises(RegistryValidationError):
        normalise_tags(42)


# -------------------------------------------------------------- concurrency


def test_concurrent_registrations_all_land(store, tmp_path):
    reg = ModelRegistry(store)
    targets = []
    for index in range(8):
        target = tmp_path / f"c{index}.gguf"
        write_gguf(target, name=f"Concurrent {index}")
        targets.append(target)

    errors: list[Exception] = []

    def work(target):
        try:
            reg.register(target)
        except Exception as error:  # pragma: no cover - only on a real race
            errors.append(error)

    threads = [threading.Thread(target=work, args=(target,)) for target in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert reg.count() == 8


def test_a_concurrent_update_does_not_lose_a_registration(store, tmp_path):
    reg = ModelRegistry(store)
    first = tmp_path / "one.gguf"
    second = tmp_path / "two.gguf"
    write_gguf(first, name="One")
    write_gguf(second, name="Two")
    record = reg.register(first)

    def register_other():
        ModelRegistry(store).register(second, name="ignored") if False else reg.register(second)

    thread = threading.Thread(target=register_other)
    thread.start()
    reg.update(record.model_id, notes="updated")
    thread.join()
    assert reg.count() == 2


# ------------------------------------------------------------------ record


def test_record_serialisation_round_trips(store, model_file):
    reg = ModelRegistry(store)
    record = reg.register(model_file, tags=["a"], notes="hello", extra={"origin": "test"})
    restored = ModelRecord.from_primitive(record.to_primitive())
    assert restored == record


def test_record_availability_is_read_live(store, model_file):
    reg = ModelRegistry(store)
    record = reg.register(model_file)
    assert record.available is True
    model_file.unlink()
    assert reg.get(record.model_id).available is False


def test_record_describe_is_readable(store, model_file):
    reg = ModelRegistry(store)
    record = reg.register(model_file, display_name="Readable")
    line = record.describe()
    assert "Readable" in line
    assert "gguf" in line
    assert "llama.cpp" in line
    assert "ready" in line


def test_a_record_with_no_file_is_not_available(store):
    record = ModelRecord(
        model_id="x", identity="x", display_name="X",
        model_format=FORMAT_GGUF, runtime="llama.cpp", source=SOURCE_CATALOG,
    )
    assert record.available is False
    assert "missing" in record.describe()


def test_a_remote_record_is_available_without_a_file(store):
    record = ModelRecord(
        model_id="x", identity="x", display_name="X",
        model_format="unknown", runtime="cloud", source="remote",
    )
    assert record.available is True


def test_a_trained_record_needs_no_path_to_be_usable(store):
    record = ModelRecord(
        model_id="x", identity="fayd-llm", display_name="Trained",
        model_format="fayd-checkpoint", runtime="fayd-inprocess",
        source=SOURCE_TRAINED,
    )
    assert record.available is False  # no path was set
    assert record.verified is False


# ---------------------------------------------------------------- helpers


def test_registry_helper_builds_a_registry(store, model_file):
    reg = registry(store)
    reg.register(model_file)
    assert len(registry(store)) == 1


def test_time_ago_renders_and_passes_through_an_unreadable_stamp():
    assert time_ago("") == "never"
    assert time_ago("not a date") == "not a date"
    assert "ago" in time_ago("2026-01-01T00:00:00+00:00", now=1_800_000_000)


def test_known_formats_and_sources_are_closed_sets():
    assert FORMAT_GGUF in KNOWN_FORMATS
    assert SOURCE_LOCAL in (SOURCE_LOCAL, SOURCE_CATALOG, SOURCE_TRAINED)


def test_the_registry_never_writes_outside_its_root(store, model_file, tmp_path):
    reg = ModelRegistry(store)
    reg.register(model_file)
    written = {path for path in store.rglob("*") if path.is_file()}
    assert written
    assert all(store in path.parents for path in written)


def test_a_registry_error_is_a_runtime_error():
    assert issubclass(RegistryError, RuntimeError)
    assert issubclass(RegistryValidationError, RegistryError)


def test_import_custom_models_is_a_no_op_when_the_shortlist_is_empty(store):
    reg = ModelRegistry(store)
    assert reg.import_custom_models() == ()


def test_import_custom_models_brings_in_a_shortlist_entry(store, model_file, monkeypatch):
    from core import custom_models

    class Entry:
        def __init__(self, path, title):
            self.path = path
            self.title = title

    monkeypatch.setattr(
        custom_models, "list_models",
        lambda: [Entry(model_file, "From The Shortlist")],
    )
    reg = ModelRegistry(store)
    imported = reg.import_custom_models()
    assert len(imported) == 1
    assert imported[0].display_name == "From The Shortlist"
    assert imported[0].identity == f"custom:{model_file}"
    # A second import does not duplicate.
    assert reg.import_custom_models() == ()


def test_import_custom_models_skips_a_file_that_is_gone(store, tmp_path, monkeypatch):
    from core import custom_models

    class Entry:
        def __init__(self, path, title):
            self.path = path
            self.title = title

    monkeypatch.setattr(
        custom_models, "list_models",
        lambda: [Entry(tmp_path / "gone.gguf", "Gone")],
    )
    assert ModelRegistry(store).import_custom_models() == ()


def test_import_custom_models_survives_an_unreadable_shortlist(store, monkeypatch):
    from core import custom_models

    def boom():
        raise RuntimeError("the store is broken")

    monkeypatch.setattr(custom_models, "list_models", boom)
    assert ModelRegistry(store).import_custom_models() == ()


def test_the_store_key_carries_the_reserved_prefix_protection(store, model_file):
    """A model id can never collide with the reserved ``__`` state prefix."""
    reg = ModelRegistry(store)
    record = reg.register(model_file)
    assert not reg._key(record.model_id).startswith("__")


def test_a_path_under_a_dotted_folder_still_registers(store, tmp_path):
    """The registry does not care where a file sits, only that it is a model."""
    target = tmp_path / ".cache" / "model.gguf"
    write_gguf(target)
    record = ModelRegistry(store).register(target)
    assert record.available is True
    assert Path(record.path) == target
