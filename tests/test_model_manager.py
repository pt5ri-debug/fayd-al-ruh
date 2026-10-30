"""The Model Manager: one API over the registry, the runtimes, and the catalog.

Everything here runs with a fake adapter and a fake catalog, so no test needs a
model, a GPU, a server, or a network. The manager's whole value is the *join* --
"what can this program run, and what is the next step?" -- so that is what the
tests pin, along with the two honesty rules: a missing model is reported with its
next step, and an operation a runtime cannot do is refused rather than faked.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from core.model_files import FORMAT_GGUF
from core.model_manager import (
    STATE_DAMAGED,
    STATE_MISSING,
    STATE_NOT_DOWNLOADED,
    STATE_READY,
    CompatibilityReport,
    ModelManager,
    OperationResult,
)
from core.model_registry import (
    SOURCE_CATALOG,
    SOURCE_LOCAL,
    SOURCE_TRAINED,
    ConfirmationRequired,
    ModelNotFoundError,
    ModelRecord,
    RegistryValidationError,
    model_id_for,
)
from core.model_runtime import (
    RUNTIME_INPROCESS_TRAINED,
    RUNTIME_LLAMA_CPP,
    RUNTIME_NAMES,
    InferenceResult,
    LoadResult,
    RuntimeCapabilities,
    RuntimeStatus,
    RuntimeUnavailable,
    UnsupportedCapability,
)
from core.model_settings import ModelSettings

# ------------------------------------------------------------------ fixtures


def _gguf_key(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _gguf_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<I", 8) + struct.pack("<Q", len(raw)) + raw


def _gguf_uint32(value: int) -> bytes:
    return struct.pack("<I", 4) + struct.pack("<I", value)


def write_gguf(path, *, name="Manager Model", architecture="llama", context=4096):
    entries = [
        ("general.name", _gguf_string(name)),
        ("general.architecture", _gguf_string(architecture)),
        ("general.file_type", _gguf_uint32(15)),
        (f"{architecture}.context_length", _gguf_uint32(context)),
    ]
    body = b"".join(_gguf_key(key) + payload for key, payload in entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 4, len(entries)) + body)
    return path


class FakeAdapter:
    """A runtime adapter a test fully controls."""

    name = "fake"

    def __init__(
        self,
        *,
        available=True,
        can_load=True,
        can_unload=True,
        can_infer=True,
        can_discover=True,
        gpu_offload=None,
        load_ok=True,
        infer_ok=True,
        validate_ok=True,
        detail="",
        hint="",
        raise_on_load=None,
        raise_on_infer=None,
    ):
        self._available = available
        self._caps = RuntimeCapabilities(
            can_load=can_load, can_unload=can_unload, can_stop=False,
            can_discover=can_discover, can_validate=True, can_infer=can_infer,
            gpu_offload=gpu_offload, resident=True,
        )
        self._load_ok = load_ok
        self._infer_ok = infer_ok
        self._validate_ok = validate_ok
        self._detail = detail
        self._hint = hint
        self._raise_on_load = raise_on_load
        self._raise_on_infer = raise_on_infer
        self._model_path: Path | None = None
        self.load_calls: list[Path] = []
        self.unload_calls = 0
        self.infer_calls: list[str] = []

    def capabilities(self):
        return self._caps

    def status(self):
        return RuntimeStatus(
            runtime=self.name, available=self._available, detail=self._detail,
            hint=self._hint, gpu_offload=self._caps.gpu_offload, capabilities=self._caps,
        )

    def validate(self, probe):
        return (self._validate_ok, "" if self._validate_ok else "the file is not usable")

    def discover(self):
        return ()

    def load(self, model_path, *, settings=None):
        if self._raise_on_load is not None:
            raise self._raise_on_load
        self.load_calls.append(Path(model_path))
        if not self._load_ok:
            return LoadResult(ok=False, runtime=self.name, detail="the load failed")
        self._model_path = Path(model_path)
        return LoadResult(ok=True, runtime=self.name, model_path=Path(model_path), seconds=0.01)

    def unload(self):
        self.unload_calls += 1
        self._model_path = None
        return LoadResult(ok=True, runtime=self.name, detail="unloaded")

    def loaded(self):
        return self._model_path is not None

    def loaded_path(self):
        return self._model_path

    def infer(self, prompt, *, settings=None):
        if self._raise_on_infer is not None:
            raise self._raise_on_infer
        if self._model_path is None:
            # A real adapter refuses to infer before a load; the fake does too, so
            # a test that forgets to load cannot pass by accident.
            return InferenceResult(
                ok=False, runtime=self.name, detail="No model is loaded. Load one first."
            )
        self.infer_calls.append(prompt)
        if not self._infer_ok:
            return InferenceResult(ok=False, runtime=self.name, detail="it did not answer")
        return InferenceResult(ok=True, runtime=self.name, text="ready", seconds=0.01)


class FakeCatalogModel:
    """The shape :func:`core.language_models.all_language_models` returns."""

    def __init__(
        self,
        key="catalog-model",
        title="Catalog Model",
        engine="ultron",
        available=False,
        detail="not downloaded",
        path=None,
        filename="catalog.gguf",
        identity="org/catalog.gguf",
        summary="a catalog model",
        download_command="python -m cli ultron download",
    ):
        self.key = key
        self.title = title
        self.engine = engine
        self.available = available
        self.detail = detail
        self.path = path
        self.filename = filename
        self.identity = identity
        self.summary = summary
        self.download_command = download_command


def manager(tmp_path, adapter=None, catalog=()):
    """A manager whose every runtime resolves to one controllable fake adapter."""
    adapters = {name: (adapter if adapter is not None else FakeAdapter()) for name in (
        RUNTIME_LLAMA_CPP, RUNTIME_INPROCESS_TRAINED,
    )}
    return ModelManager(
        tmp_path / "root",
        adapter_factory=lambda runtime: adapters.get(runtime, FakeAdapter()),
        catalog=tuple(catalog),
    )


# ------------------------------------------------------------------ listing


def test_registering_a_model_makes_it_listable(tmp_path):
    mgr = manager(tmp_path)
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"), display_name="Mine")
    listed = mgr.list_models()
    assert [item.display_name for item in listed] == ["Mine"]
    assert listed[0].ready is True
    assert entry.record.model_id == listed[0].record.model_id


def test_a_missing_file_makes_a_registered_model_not_ready(tmp_path):
    mgr = manager(tmp_path)
    target = write_gguf(tmp_path / "m.gguf")
    mgr.register(target)
    target.unlink()
    entry = mgr.list_models()[0]
    assert entry.state == STATE_MISSING
    assert entry.ready is False
    assert "gone" in entry.status_line.lower()


def test_a_damaged_file_is_reported_as_damaged(tmp_path):
    mgr = manager(tmp_path)
    # A record whose integrity check failed (as registration would store it for a
    # truncated file) reads as damaged even while the path exists.
    mgr.registry.register_record(ModelRecord(
        model_id=model_id_for("damaged"), identity="damaged", display_name="Damaged",
        model_format=FORMAT_GGUF, runtime=RUNTIME_LLAMA_CPP, source=SOURCE_LOCAL,
        path=write_gguf(tmp_path / "damaged.gguf"),
        integrity="truncated", integrity_detail="the header ends mid-block",
    ))
    entry = mgr.list_models()[0]
    assert entry.state == STATE_DAMAGED
    assert entry.ready is False


def test_a_catalog_model_that_is_absent_is_listed_as_not_downloaded(tmp_path):
    mgr = manager(tmp_path, catalog=[FakeCatalogModel(available=False)])
    entry = mgr.list_models()[0]
    assert entry.from_catalog is True
    assert entry.state == STATE_NOT_DOWNLOADED
    assert entry.ready is False
    assert "not downloaded" in entry.status_line.lower()


def test_a_present_catalog_model_is_ready(tmp_path):
    target = write_gguf(tmp_path / "catalog.gguf")
    model = FakeCatalogModel(available=True, path=target)
    mgr = manager(tmp_path, catalog=[model])
    entry = mgr.list_models()[0]
    assert entry.state == STATE_READY
    assert entry.ready is True


def test_a_catalog_model_is_described_not_stored(tmp_path):
    """The catalog is already the record of it; duplicating it would be a second truth."""
    mgr = manager(tmp_path, catalog=[FakeCatalogModel()])
    mgr.list_models()
    assert mgr.registry.count() == 0


def test_a_registered_model_wins_over_its_catalog_twin(tmp_path):
    """One line per model, and the registered one carries the user's own words."""
    target = write_gguf(tmp_path / "catalog.gguf")
    model = FakeCatalogModel(available=True, path=target, identity="org/catalog.gguf")
    mgr = manager(tmp_path, catalog=[model])
    mgr.register(target, display_name="What I Call It", identity="org/catalog.gguf")
    entries = mgr.list_models()
    assert len(entries) == 1
    assert entries[0].display_name == "What I Call It"
    assert entries[0].from_catalog is False


def test_listing_can_exclude_the_catalog(tmp_path):
    mgr = manager(tmp_path, catalog=[FakeCatalogModel()])
    assert mgr.list_models(include_catalog=False) == ()
    assert len(mgr.list_models()) == 1


def test_ready_models_returns_only_what_can_run(tmp_path):
    target = write_gguf(tmp_path / "m.gguf")
    mgr = manager(tmp_path, catalog=[FakeCatalogModel(available=False)])
    mgr.register(target, display_name="Ready One")
    ready = mgr.ready_models()
    assert [entry.display_name for entry in ready] == ["Ready One"]


def test_ready_models_are_listed_before_the_missing_ones(tmp_path):
    mgr = manager(tmp_path, catalog=[FakeCatalogModel(title="Absent", available=False)])
    mgr.register(write_gguf(tmp_path / "m.gguf"), display_name="Present")
    states = [entry.state for entry in mgr.list_models()]
    assert states.index(STATE_READY) < states.index(STATE_NOT_DOWNLOADED)


def test_the_catalog_is_read_defensively(tmp_path):
    """A catalog that cannot be read must not stop the registry half working."""

    def boom():
        raise RuntimeError("the catalog is broken")

    mgr = ModelManager(tmp_path / "root", adapter_factory=lambda runtime: FakeAdapter())
    mgr._catalog = boom
    assert mgr.catalog_entries() == ()


def test_a_catalog_entry_without_an_identity_is_skipped(tmp_path):
    model = FakeCatalogModel(identity="", filename="", path=None)
    mgr = manager(tmp_path, catalog=[model])
    assert mgr.catalog_entries() == ()


def test_the_real_catalog_pairs_are_unwrapped(tmp_path):
    """``core.language_models`` returns ``(mouth, model)`` pairs, not bare models."""
    from core import language_models

    pairs = language_models.all_language_models()
    assert pairs, "the project should ship at least one catalog model"
    assert isinstance(pairs[0], tuple)
    mgr = ModelManager(tmp_path / "root", adapter_factory=lambda runtime: FakeAdapter())
    entries = mgr.catalog_entries()
    assert entries, "the real catalog should produce entries"
    # Every entry is a real model object, not a pair.
    for entry in entries:
        assert not isinstance(entry.record, tuple)
        assert entry.record.model_id
        assert entry.from_catalog is True


def test_a_catalog_model_reports_a_runtime_that_can_run_it(tmp_path):
    mgr = ModelManager(tmp_path / "root", adapter_factory=lambda runtime: FakeAdapter())
    runtimes = {entry.runtime for entry in mgr.catalog_entries()}
    # A GGUF catalog model is a llama.cpp model; the retrieval mouth has none.
    assert runtimes & {RUNTIME_LLAMA_CPP, RUNTIME_INPROCESS_TRAINED}


# ------------------------------------------------------------ registration


def test_register_records_the_operation(tmp_path):
    mgr = manager(tmp_path)
    mgr.register(write_gguf(tmp_path / "m.gguf"), display_name="Recorded")
    events = mgr.log.entries()
    assert any(event.operation == "register" for event in events)
    assert any("Recorded" in event.display_name for event in events)


def test_a_refused_registration_is_logged_and_raised(tmp_path):
    mgr = manager(tmp_path)
    with pytest.raises(ValueError):
        mgr.register(tmp_path / "nope.gguf")
    failures = mgr.log.failures()
    assert failures
    assert "register" in failures[0].message


def test_update_records_the_operation(tmp_path):
    mgr = manager(tmp_path)
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    updated = mgr.update(entry.model_id, notes="changed")
    assert updated.record.notes == "changed"
    assert any(event.operation == "update" for event in mgr.log.entries())


def test_a_refused_update_is_logged_and_raised(tmp_path):
    mgr = manager(tmp_path)
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    with pytest.raises(RegistryValidationError):
        mgr.update(entry.model_id, display_name="  ")
    assert any("update" in event.message for event in mgr.log.failures())


def test_remove_requires_confirmation_and_is_logged_either_way(tmp_path):
    mgr = manager(tmp_path)
    target = write_gguf(tmp_path / "m.gguf")
    entry = mgr.register(target)
    with pytest.raises(ConfirmationRequired):
        mgr.remove(entry.model_id)
    assert any("refused" in event.message for event in mgr.log.failures())
    removed = mgr.remove(entry.model_id, confirm=True)
    assert removed.model_id == entry.model_id
    assert mgr.registry.count() == 0
    # The file itself was never touched.
    assert target.is_file()


def test_remove_notes_that_the_file_was_left_alone(tmp_path):
    mgr = manager(tmp_path)
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    mgr.remove(entry.model_id, confirm=True)
    removals = [event for event in mgr.log.entries() if event.operation == "remove"]
    assert removals
    assert "not touched" in removals[0].message


def test_removing_an_unknown_model_is_logged_and_raised(tmp_path):
    mgr = manager(tmp_path)
    with pytest.raises(ModelNotFoundError):
        mgr.remove("nope", confirm=True)
    assert mgr.log.failures()


# ------------------------------------------------------------------ inspect


def test_inspect_re_reads_the_file(tmp_path):
    mgr = manager(tmp_path)
    target = write_gguf(tmp_path / "m.gguf", name="Read Me")
    entry = mgr.register(target)
    probe = mgr.inspect(entry.model_id)
    assert probe.usable is True
    assert probe.gguf is not None
    assert probe.gguf.name == "Read Me"
    assert any(event.operation == "inspect" for event in mgr.log.entries())


def test_inspect_reports_a_file_replaced_in_place(tmp_path):
    mgr = manager(tmp_path)
    target = write_gguf(tmp_path / "m.gguf", name="Original")
    entry = mgr.register(target)
    write_gguf(target, name="Replaced")
    assert mgr.inspect(entry.model_id).gguf.name == "Replaced"


def test_inspect_of_a_model_with_no_file_does_not_raise(tmp_path):
    mgr = manager(tmp_path)
    record = mgr.registry.register_record(ModelRecord(
        model_id=model_id_for("remote"), identity="remote", display_name="Remote",
        model_format=FORMAT_GGUF, runtime=RUNTIME_LLAMA_CPP, source=SOURCE_CATALOG,
    ))
    probe = mgr.inspect(record.model_id)
    assert probe.exists is False
    assert "no local file" in probe.detail.lower()


def test_discover_scans_the_project_folders_by_default(tmp_path):
    mgr = manager(tmp_path)
    report = mgr.discover()
    assert report.roots
    assert any(event.operation == "discover" for event in mgr.log.entries())


def test_discover_accepts_explicit_roots(tmp_path):
    write_gguf(tmp_path / "scan" / "a.gguf")
    mgr = manager(tmp_path)
    report = mgr.discover(tmp_path / "scan")
    assert len(report) == 1


def test_default_scan_roots_stay_inside_the_project(tmp_path):
    """A scan is a filesystem read; it must not walk a user's home directory."""
    from core.paths import bundle_root, data_dir

    mgr = manager(tmp_path)
    roots = mgr.default_scan_roots()
    assert roots
    for root in roots:
        text = str(root)
        assert text.startswith(str(data_dir())) or text.startswith(str(bundle_root()))


# ----------------------------------------------------------------- settings


def test_settings_resolve_to_the_project_defaults(tmp_path):
    mgr = manager(tmp_path)
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    assert mgr.settings(entry.model_id) == ModelSettings()


def test_updating_settings_persists_and_logs(tmp_path):
    mgr = manager(tmp_path)
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    settings = mgr.update_settings(entry.model_id, temperature=0.2)
    assert settings.temperature == pytest.approx(0.2)
    assert mgr.settings(entry.model_id).temperature == pytest.approx(0.2)
    assert any(event.operation == "update" for event in mgr.log.entries())


def test_resetting_settings_returns_to_the_defaults(tmp_path):
    mgr = manager(tmp_path)
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    mgr.update_settings(entry.model_id, temperature=0.2)
    assert mgr.reset_settings(entry.model_id) == ModelSettings()


# ----------------------------------------------------------------- runtimes


def test_the_adapter_is_built_once_per_runtime(tmp_path):
    built: list[str] = []

    def factory(runtime):
        built.append(runtime)
        return FakeAdapter()

    mgr = ModelManager(tmp_path / "root", adapter_factory=factory, catalog=())
    mgr.adapter(RUNTIME_LLAMA_CPP)
    mgr.adapter(RUNTIME_LLAMA_CPP)
    assert built == [RUNTIME_LLAMA_CPP]


def test_a_missing_adapter_is_reported_not_raised(tmp_path):
    def factory(runtime):
        raise RuntimeUnavailable(f"no {runtime} here")

    mgr = ModelManager(tmp_path / "root", adapter_factory=factory, catalog=())
    status = mgr.runtime_status(RUNTIME_LLAMA_CPP)
    assert status.available is False
    assert "no llama.cpp here" in status.detail


def test_runtime_report_lists_every_adaptable_runtime(tmp_path):
    report = manager(tmp_path).runtime_report()
    # The fake factory answers for every runtime, so each name is reported once.
    assert report.count("fake:") == len(RUNTIME_NAMES)


# ------------------------------------------------------------ compatibility


def test_a_ready_model_reports_that_it_can_run(tmp_path):
    mgr = manager(tmp_path, adapter=FakeAdapter(available=True))
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    report = mgr.compatibility(entry.model_id)
    assert report.can_run is True
    assert report.runtime == RUNTIME_LLAMA_CPP
    assert report.memory_estimate_bytes is not None


def test_a_missing_file_is_a_reason_it_cannot_run(tmp_path):
    mgr = manager(tmp_path)
    target = write_gguf(tmp_path / "m.gguf")
    entry = mgr.register(target)
    target.unlink()
    report = mgr.compatibility(entry.model_id)
    assert report.can_run is False
    assert any("gone" in reason for reason in report.reasons)


def test_an_unavailable_runtime_is_a_reason_it_cannot_run(tmp_path):
    mgr = manager(tmp_path, adapter=FakeAdapter(available=False, detail="not installed"))
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    report = mgr.compatibility(entry.model_id)
    assert report.can_run is False
    assert any("not installed" in reason for reason in report.reasons)


def test_a_validation_failure_is_a_reason_it_cannot_run(tmp_path):
    mgr = manager(tmp_path, adapter=FakeAdapter(validate_ok=False))
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    report = mgr.compatibility(entry.model_id)
    assert report.can_run is False
    assert any("not usable" in reason for reason in report.reasons)


def test_an_unknown_gpu_answer_is_warned_about_not_hidden(tmp_path):
    mgr = manager(tmp_path, adapter=FakeAdapter(gpu_offload=None))
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    report = mgr.compatibility(entry.model_id)
    assert report.gpu_offload is None
    assert any("could not be determined" in warning for warning in report.warnings)
    assert "CPU" in report.describe()


def test_a_cpu_only_runtime_is_reported_as_cpu_only(tmp_path):
    mgr = manager(tmp_path, adapter=FakeAdapter(gpu_offload=False))
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    report = mgr.compatibility(entry.model_id)
    assert report.gpu_offload is False
    assert any("CPU is used" in warning for warning in report.warnings)


def test_the_memory_estimate_is_labelled_as_an_estimate(tmp_path):
    mgr = manager(tmp_path)
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    report = mgr.compatibility(entry.model_id)
    assert report.memory_estimate_bytes is not None
    assert "estimate" in report.describe()
    assert "not a measurement" in report.describe()


def test_a_record_with_no_size_has_no_memory_estimate(tmp_path):
    mgr = manager(tmp_path)
    record = mgr.registry.register_record(ModelRecord(
        model_id=model_id_for("no-size"), identity="no-size", display_name="No Size",
        model_format=FORMAT_GGUF, runtime=RUNTIME_LLAMA_CPP, source=SOURCE_CATALOG,
    ))
    report = mgr.compatibility(record.model_id)
    assert report.memory_estimate_bytes is None


def test_compatibility_of_an_unknown_model_raises(tmp_path):
    mgr = manager(tmp_path)
    with pytest.raises(ModelNotFoundError):
        mgr.compatibility("nope")


# --------------------------------------------------------------------- load


def test_load_uses_the_models_own_runtime(tmp_path):
    adapter = FakeAdapter()
    mgr = manager(tmp_path, adapter=adapter)
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    result = mgr.load(entry.model_id)
    assert result.ok is True
    assert result.load is not None
    assert adapter.load_calls


def test_load_records_the_operation(tmp_path):
    mgr = manager(tmp_path)
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    mgr.load(entry.model_id)
    assert any(event.operation == "load" and event.ok for event in mgr.log.entries())


def test_load_refuses_a_model_with_no_file(tmp_path):
    mgr = manager(tmp_path)
    record = mgr.registry.register_record(ModelRecord(
        model_id=model_id_for("remote"), identity="remote", display_name="Remote",
        model_format=FORMAT_GGUF, runtime=RUNTIME_LLAMA_CPP, source=SOURCE_CATALOG,
    ))
    result = mgr.load(record.model_id)
    assert result.ok is False
    assert "no local file" in result.detail
    assert mgr.log.failures()


def test_load_refuses_a_runtime_that_cannot_load(tmp_path):
    mgr = manager(tmp_path, adapter=FakeAdapter(can_load=False))
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    result = mgr.load(entry.model_id)
    assert result.ok is False
    assert "cannot load" in result.detail


def test_load_reports_a_runtime_failure_instead_of_raising(tmp_path):
    mgr = manager(tmp_path, adapter=FakeAdapter(raise_on_load=RuntimeError("it exploded")))
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    result = mgr.load(entry.model_id)
    assert result.ok is False
    assert "it exploded" in result.detail
    assert mgr.log.failures()


def test_loading_a_second_model_unloads_the_first(tmp_path):
    adapter = FakeAdapter()
    mgr = manager(tmp_path, adapter=adapter)
    first = mgr.register(write_gguf(tmp_path / "a.gguf", name="A"), display_name="A")
    second = mgr.register(write_gguf(tmp_path / "b.gguf", name="B"), display_name="B")
    mgr.load(first.model_id)
    mgr.load(second.model_id)
    assert adapter.unload_calls == 1
    assert any(event.operation == "unload" for event in mgr.log.entries())


def test_loading_the_same_model_twice_does_not_unload_it(tmp_path):
    adapter = FakeAdapter()
    mgr = manager(tmp_path, adapter=adapter)
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    mgr.load(entry.model_id)
    mgr.load(entry.model_id)
    assert adapter.unload_calls == 0


def test_load_passes_the_effective_settings(tmp_path):
    seen: list[object] = []

    class Recording(FakeAdapter):
        def load(self, model_path, *, settings=None):
            seen.append(settings)
            return super().load(model_path, settings=settings)

    mgr = manager(tmp_path, adapter=Recording())
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    mgr.update_settings(entry.model_id, temperature=0.3)
    mgr.load(entry.model_id)
    assert seen[0].temperature == pytest.approx(0.3)


def test_unload_of_a_model_that_can_unload(tmp_path):
    adapter = FakeAdapter()
    mgr = manager(tmp_path, adapter=adapter)
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    mgr.load(entry.model_id)
    result = mgr.unload(entry.model_id)
    assert result.ok is True
    assert adapter.unload_calls == 1


def test_unload_skips_a_runtime_that_cannot_unload(tmp_path):
    mgr = manager(tmp_path, adapter=FakeAdapter(can_unload=False))
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    result = mgr.unload(entry.model_id)
    # Nothing claimed, nothing raised.
    assert result.ok is True


def test_unload_with_no_model_id_unloads_every_registered_model(tmp_path):
    adapter = FakeAdapter()
    mgr = manager(tmp_path, adapter=adapter)
    mgr.register(write_gguf(tmp_path / "a.gguf"))
    mgr.register(write_gguf(tmp_path / "b.gguf"))
    result = mgr.unload()
    assert result.ok is True
    assert adapter.unload_calls == 2


def test_unload_handles_an_unsupported_capability(tmp_path):
    class Stubborn(FakeAdapter):
        def unload(self):
            raise UnsupportedCapability("cannot unload")

    mgr = manager(tmp_path, adapter=Stubborn())
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    assert mgr.unload(entry.model_id).ok is True


# --------------------------------------------------------------------- test


def test_test_answers_through_the_adapter(tmp_path):
    adapter = FakeAdapter()
    mgr = manager(tmp_path, adapter=adapter)
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    result = mgr.test(entry.model_id)
    assert result.ok is True
    assert result.inference is not None
    assert result.inference.text == "ready"
    assert adapter.infer_calls


def test_test_loads_the_model_first_when_nothing_is_loaded(tmp_path):
    adapter = FakeAdapter()
    mgr = manager(tmp_path, adapter=adapter)
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    mgr.test(entry.model_id)
    assert adapter.load_calls
    assert adapter.infer_calls


def test_test_does_not_reload_an_already_loaded_model(tmp_path):
    adapter = FakeAdapter()
    mgr = manager(tmp_path, adapter=adapter)
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    mgr.load(entry.model_id)
    mgr.test(entry.model_id)
    assert len(adapter.load_calls) == 1


def test_test_can_be_told_not_to_load(tmp_path):
    adapter = FakeAdapter()
    mgr = manager(tmp_path, adapter=adapter)
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    result = mgr.test(entry.model_id, load_if_needed=False)
    assert result.ok is False
    assert adapter.load_calls == []


def test_test_refuses_a_runtime_that_cannot_infer(tmp_path):
    mgr = manager(tmp_path, adapter=FakeAdapter(can_infer=False))
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    result = mgr.test(entry.model_id)
    assert result.ok is False
    assert "cannot run inference" in result.detail


def test_test_reports_an_inference_failure(tmp_path):
    mgr = manager(tmp_path, adapter=FakeAdapter(infer_ok=False))
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    result = mgr.test(entry.model_id)
    assert result.ok is False
    assert mgr.log.failures()


def test_test_reports_a_raised_inference_failure(tmp_path):
    mgr = manager(tmp_path, adapter=FakeAdapter(raise_on_infer=RuntimeError("bad model")))
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    result = mgr.test(entry.model_id)
    assert result.ok is False
    assert "bad model" in result.detail


def test_test_records_the_operation(tmp_path):
    mgr = manager(tmp_path)
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    mgr.test(entry.model_id)
    assert any(event.operation == "test" and event.ok for event in mgr.log.entries())


# --------------------------------------------------------------- download


def test_a_catalog_model_reports_its_download_command(tmp_path):
    model = FakeCatalogModel(identity="org/catalog.gguf", download_command="do-the-fetch")
    mgr = manager(tmp_path, catalog=[model])
    record = mgr.registry.register_record(ModelRecord(
        model_id=model_id_for("org/catalog.gguf"), identity="org/catalog.gguf",
        display_name="Catalog", model_format=FORMAT_GGUF, runtime=RUNTIME_LLAMA_CPP,
        source=SOURCE_CATALOG,
    ))
    assert mgr.download_command(record.model_id) == "do-the-fetch"


def test_a_model_with_no_catalog_twin_has_no_download_command(tmp_path):
    mgr = manager(tmp_path)
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"))
    assert mgr.download_command(entry.model_id) == ""


# ------------------------------------------------------------------ status


def test_status_summarises_the_subsystem(tmp_path):
    target = write_gguf(tmp_path / "m.gguf")
    mgr = manager(tmp_path, catalog=[FakeCatalogModel(available=False)])
    mgr.register(target, display_name="Ready One")
    status = mgr.status()
    assert status.registered == 1
    assert status.catalog == 1
    assert status.ready == 1
    assert status.runtimes
    assert status.summary()


def test_status_names_the_registry_folder_and_log_size(tmp_path):
    mgr = manager(tmp_path)
    mgr.register(write_gguf(tmp_path / "m.gguf"))
    status = mgr.status()
    assert status.registry_path == mgr.registry.root
    assert status.log_events >= 1
    assert "Model Manager" in status.describe()


def test_status_reports_unreadable_records(tmp_path):
    import json

    mgr = manager(tmp_path)
    broken = mgr.registry.root / "model_registry"
    broken.mkdir(parents=True, exist_ok=True)
    (broken / "model-bad.json").write_text(json.dumps({"nope": True}), encoding="utf-8")
    status = mgr.status()
    assert "model-bad" in status.corrupt_records
    assert "Unreadable" in status.describe()


def test_log_tail_returns_the_newest_first(tmp_path):
    mgr = manager(tmp_path)
    mgr.register(write_gguf(tmp_path / "m.gguf"), display_name="Only")
    tail = mgr.log_tail(1)
    assert len(tail) == 1
    assert tail[0].display_name == "Only"


# ------------------------------------------------------------------ helpers


def test_an_entry_describes_itself(tmp_path):
    mgr = manager(tmp_path)
    entry = mgr.register(write_gguf(tmp_path / "m.gguf"), display_name="Readable")
    assert "Readable" in entry.describe()
    assert "ready" in entry.describe()


def test_an_operation_result_describes_itself():
    assert "loaded" in OperationResult(
        ok=True, operation="load",
        load=LoadResult(ok=True, runtime="x", model_path=Path("/a.gguf")),
    ).describe()
    assert "failed" in OperationResult(ok=False, operation="load").describe()
    assert "done" in OperationResult(ok=True, operation="load").describe()


def test_a_compatibility_report_describes_itself():
    report = CompatibilityReport(
        model_id="m", runtime=RUNTIME_LLAMA_CPP, can_run=False, reasons=("gone",),
    )
    assert "cannot run" in report.describe()
    assert "gone" in report.describe()


def test_the_manager_helper_builds_a_manager(tmp_path):
    from core.model_manager import model_manager

    built = model_manager(tmp_path / "root", catalog=())
    assert isinstance(built, ModelManager)


def test_a_trained_record_needs_no_path_to_register(tmp_path):
    mgr = manager(tmp_path)
    mgr.registry.register_record(ModelRecord(
        model_id=model_id_for("fayd-llm"), identity="fayd-llm", display_name="Trained",
        model_format="fayd-checkpoint", runtime=RUNTIME_INPROCESS_TRAINED,
        source=SOURCE_TRAINED,
    ))
    entry = mgr.list_models()[0]
    assert entry.state == STATE_READY
    assert entry.ready is True


def test_the_manager_never_writes_outside_its_root(tmp_path):
    mgr = manager(tmp_path)
    mgr.register(write_gguf(tmp_path / "m.gguf"))
    root = mgr.registry.root
    written = {path for path in root.rglob("*") if path.is_file()}
    assert written
    assert all(root in path.parents for path in written)


def test_a_manager_with_no_catalog_still_lists_registered_models(tmp_path):
    mgr = ModelManager(tmp_path / "root", adapter_factory=lambda runtime: FakeAdapter(), catalog=())
    mgr.register(write_gguf(tmp_path / "m.gguf"))
    assert len(mgr.list_models()) == 1


def test_state_order_puts_ready_first():
    from core.model_manager import STATE_ORDER

    assert STATE_ORDER[0] == STATE_READY


# ------------------------------------------------------- task selection


def test_models_for_task_prefers_the_tasks_engine(tmp_path):
    catalog = (
        FakeCatalogModel(key="vision-model", engine="vision", available=True,
                         identity="org/vision.gguf"),
        FakeCatalogModel(key="banner-model", engine="banner", available=True,
                         identity="org/banner.gguf"),
    )
    mgr = manager(tmp_path, catalog=catalog)
    models = mgr.models_for_task("petroleum")
    assert next(entry.record.engine for entry in models) == "vision"


def test_models_for_task_falls_back_to_any_ready_model(tmp_path):
    catalog = (FakeCatalogModel(key="banner-model", engine="banner", available=True,
                                identity="org/banner.gguf"),)
    mgr = manager(tmp_path, catalog=catalog)
    models = mgr.models_for_task("petroleum")
    assert [entry.record.engine for entry in models] == ["banner"]


def test_models_for_task_ignores_models_that_cannot_run(tmp_path):
    catalog = (FakeCatalogModel(key="vision-model", engine="vision", available=False,
                                identity="org/vision.gguf"),)
    mgr = manager(tmp_path, catalog=catalog)
    assert mgr.models_for_task("petroleum") == ()


def test_models_for_task_of_an_unknown_task_is_empty(tmp_path):
    mgr = manager(tmp_path)
    assert mgr.models_for_task("no-such-task") == ()


def test_select_for_task_returns_the_best_or_none(tmp_path):
    assert manager(tmp_path).select_for_task("chat") is None
    catalog = (FakeCatalogModel(key="banner-model", engine="banner", available=True,
                                identity="org/banner.gguf"),)
    mgr = manager(tmp_path, catalog=catalog)
    chosen = mgr.select_for_task("presentation")
    assert chosen is not None
    assert chosen.record.engine == "banner"


def test_task_table_names_only_real_catalog_engines():
    from core.model_manager import TASKS

    for engines in TASKS.values():
        for engine in engines:
            assert engine in {"ultron", "vision", "banner", "natasha", "hakeem", "shared"}
