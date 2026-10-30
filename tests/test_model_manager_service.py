"""The Model Manager service: the Tk-free engine behind the window and the CLI.

Every test here runs with no display, no GPU, and no model file. The manager is
built over a temporary root and, where a runtime would be needed, a fake adapter
is injected -- the same seams :mod:`core.model_manager` already exposes, so this
suite proves the *service* layer rather than re-testing the core.
"""

from __future__ import annotations

import struct
import threading
from pathlib import Path

import pytest
from core.model_manager import STATE_ORDER, STATE_READY, ModelManager
from core.model_runtime import (
    RUNTIME_LLAMA_CPP,
    InferenceResult,
    LoadResult,
    RuntimeCapabilities,
    RuntimeStatus,
)
from gui.model_manager import (
    MANAGER_ERRORS,
    STATE_BLURBS,
    STATE_MARKERS,
    ModelManagerService,
    ModelRow,
    model_manager_service,
)
from storage.json_store import JsonStore


class FakeAdapter:
    """A runtime adapter that answers without a model or a runtime install."""

    runtime = RUNTIME_LLAMA_CPP
    backend = "fake"

    def __init__(self, *, available=True, can_infer=True):
        self._available = available
        self._can_infer = can_infer
        self._loaded = False
        self.calls: list[str] = []

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(can_load=True, can_infer=self._can_infer)

    def status(self) -> RuntimeStatus:
        return RuntimeStatus(
            runtime=self.runtime,
            available=self._available,
            detail="fake: ready" if self._available else "fake: not installed",
            hint="" if self._available else "install the fake runtime",
            gpu_offload=False,
            capabilities=self.capabilities(),
        )

    def validate(self, probe):
        return (True, "")

    def discover(self):
        return ()

    def load(self, model_path, *, settings=None, **kwargs) -> LoadResult:
        self.calls.append("load")
        self._loaded = True
        return LoadResult(ok=True, runtime=self.runtime, model_path=model_path, detail="loaded")

    def unload(self) -> LoadResult:
        self.calls.append("unload")
        self._loaded = False
        return LoadResult(ok=True, runtime=self.runtime, detail="unloaded")

    def loaded(self) -> bool:
        return self._loaded

    def loaded_path(self):
        return None

    def infer(self, prompt, *, settings=None, **kwargs) -> InferenceResult:
        self.calls.append("infer")
        return InferenceResult(ok=True, runtime=self.runtime, text="ready", seconds=0.01)


def write_gguf(path, *, name="Service Model", architecture="llama", context=4096):
    def key(text):
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    def string(text):
        raw = text.encode()
        return struct.pack("<I", 8) + struct.pack("<Q", len(raw)) + raw

    def uint32(value):
        return struct.pack("<I", 4) + struct.pack("<I", value)

    entries = [
        ("general.name", string(name)),
        ("general.architecture", string(architecture)),
        ("general.file_type", uint32(15)),
        (f"{architecture}.context_length", uint32(context)),
    ]
    body = b"".join(key(k) + payload for k, payload in entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 4, len(entries)) + body)
    return path


@pytest.fixture
def service(tmp_path):
    manager = ModelManager(tmp_path / "root", adapter_factory=lambda runtime: FakeAdapter())
    return ModelManagerService(tmp_path / "root", manager=manager)


# ---------------------------------------------------------------- snapshot


def test_a_fresh_service_lists_the_catalog(service):
    snapshot = service.snapshot()
    # The build ships models; on a fresh machine none is downloaded yet.
    assert snapshot.catalog > 0
    assert snapshot.registered == 0
    assert snapshot.rows


def test_a_fresh_service_reports_no_ready_models(service):
    snapshot = service.snapshot()
    assert snapshot.ready == 0
    assert all(not row.ready for row in snapshot.rows)


def test_the_snapshot_summary_names_the_counts(service):
    summary = service.snapshot().summary()
    assert "registered" in summary
    assert "ready to run" in summary


def test_runtime_lines_are_reported(service):
    lines = service.runtime_lines()
    assert lines
    assert any("llama.cpp" in line for line in lines)


def test_every_row_carries_a_state_marker(service):
    for row in service.snapshot().rows:
        assert row.state_marker
        assert row.state_marker == STATE_MARKERS.get(row.state, row.state)


def test_every_known_state_has_a_blurb():
    for state in (STATE_MARKERS.keys() | STATE_BLURBS.keys()):
        assert STATE_MARKERS.get(state)
        assert STATE_BLURBS.get(state)


def test_registered_only_rows_exclude_the_catalog(service):
    assert service.rows(registered_only=True) == ()


# --------------------------------------------------------------- registering


def test_registering_a_file_returns_a_ready_row(service, tmp_path):
    target = write_gguf(tmp_path / "model.gguf")
    row = service.register(target, display_name="Mine")
    assert isinstance(row, ModelRow)
    assert row.ready is True
    assert row.display_name == "Mine"
    assert row.model_id


def test_a_registered_model_appears_in_the_snapshot(service, tmp_path):
    target = write_gguf(tmp_path / "model.gguf")
    service.register(target, display_name="Mine")
    snapshot = service.snapshot()
    assert snapshot.registered == 1
    assert snapshot.ready == 1
    assert any(row.display_name == "Mine" for row in snapshot.rows)


def test_the_row_describe_is_readable(service, tmp_path):
    target = write_gguf(tmp_path / "model.gguf")
    row = service.register(target, display_name="Mine")
    described = row.describe()
    assert "Mine" in described
    assert "ready" in described


def test_a_registered_file_that_vanishes_is_reported_missing(service, tmp_path):
    target = write_gguf(tmp_path / "model.gguf")
    row = service.register(target)
    target.unlink()
    after = service.rows(registered_only=True)[0]
    assert after.model_id == row.model_id
    assert after.ready is False
    assert after.state == "missing"


# -------------------------------------------------------------------- detail


def test_detail_carries_the_record_fields(service, tmp_path):
    target = write_gguf(tmp_path / "model.gguf", name="Detailed", context=8192)
    row = service.register(target, display_name="Detailed")
    detail = service.detail(row.model_id)
    assert detail.model_id == row.model_id
    assert detail.display_name == "Detailed"
    assert detail.row.ready is True
    assert detail.identity


def test_detail_carries_the_settings(service, tmp_path):
    target = write_gguf(tmp_path / "model.gguf")
    row = service.register(target)
    detail = service.detail(row.model_id)
    assert detail.settings is not None
    assert detail.settings.temperature > 0


def test_detail_carries_a_compatibility_report(service, tmp_path):
    target = write_gguf(tmp_path / "model.gguf")
    row = service.register(target)
    detail = service.detail(row.model_id)
    assert detail.compatibility is not None
    # The fake runtime is available, so the model can run here.
    assert detail.compatibility.can_run is True


def test_detail_of_an_unknown_model_raises(service):
    from core.model_registry import ModelNotFoundError

    with pytest.raises(ModelNotFoundError):
        service.detail("not-a-model")


# ----------------------------------------------------------------- selection


def test_selection_starts_empty(service):
    assert service.selection() == ""


def test_selecting_remembers_the_model(service, tmp_path):
    target = write_gguf(tmp_path / "model.gguf")
    row = service.register(target)
    assert service.select(row.model_id) == row.model_id
    assert service.selection() == row.model_id


def test_the_selection_survives_a_new_service(tmp_path):
    root = tmp_path / "root"
    manager = ModelManager(root, adapter_factory=lambda runtime: FakeAdapter())
    first = ModelManagerService(root, manager=manager)
    first.select("some-model")

    second = ModelManagerService(root, manager=manager)
    assert second.selection() == "some-model"


def test_a_corrupt_selection_record_reads_as_empty(tmp_path):
    root = tmp_path / "root"
    store = JsonStore(root)
    store.save("models", "manager_selection", "not-a-dict")
    manager = ModelManager(root, adapter_factory=lambda runtime: FakeAdapter())
    service = ModelManagerService(root, manager=manager, store=store)
    assert service.selection() == ""


def test_selecting_a_blank_key_clears_the_selection(service, tmp_path):
    target = write_gguf(tmp_path / "model.gguf")
    row = service.register(target)
    service.select(row.model_id)
    assert service.select("") == ""
    assert service.selection() == ""


# ------------------------------------------------------------------ settings


def test_settings_can_be_changed_and_reset(service, tmp_path):
    target = write_gguf(tmp_path / "model.gguf")
    row = service.register(target)

    changed = service.update_settings(row.model_id, temperature=0.2)
    assert changed.temperature == pytest.approx(0.2)
    assert service.settings(row.model_id).temperature == pytest.approx(0.2)

    reset = service.reset_settings(row.model_id)
    assert reset.temperature == pytest.approx(0.7)


def test_an_out_of_range_setting_is_refused(service, tmp_path):
    from core.model_settings import SettingsError

    target = write_gguf(tmp_path / "model.gguf")
    row = service.register(target)
    with pytest.raises(SettingsError):
        service.update_settings(row.model_id, temperature=99)


# --------------------------------------------------------------- operations


def test_load_reports_success_through_the_fake_runtime(service, tmp_path):
    target = write_gguf(tmp_path / "model.gguf")
    row = service.register(target)
    report = service.load(row.model_id)
    assert report.ok is True
    assert report.operation
    assert report.describe()


def test_test_asks_the_model_a_question(service, tmp_path):
    target = write_gguf(tmp_path / "model.gguf")
    row = service.register(target)
    report = service.test(row.model_id, prompt="Say ready.")
    assert report.ok is True
    assert report.result is not None
    assert report.result.inference is not None
    assert report.result.inference.text == "ready"


def test_unload_with_nothing_loaded_is_still_ok(service):
    report = service.unload()
    assert report.ok is True


def test_load_of_an_unknown_model_raises(service):
    from core.model_registry import ModelNotFoundError

    with pytest.raises(ModelNotFoundError):
        service.load("not-a-model")


def test_operations_are_recorded_in_the_log(service, tmp_path):
    target = write_gguf(tmp_path / "model.gguf")
    row = service.register(target)
    service.load(row.model_id)
    events = service.log_tail(20)
    assert events
    assert any(event.operation for event in events)


def test_download_command_is_reported_for_a_catalog_model(service):
    # A catalog model the build ships has a fetch command; a plain local file does not.
    catalog_rows = [row for row in service.rows() if row.from_catalog]
    assert catalog_rows
    commands = [service.download_command(row.model_id) for row in catalog_rows]
    assert any(commands)


def test_a_local_file_has_no_download_command(service, tmp_path):
    target = write_gguf(tmp_path / "model.gguf")
    row = service.register(target)
    assert service.download_command(row.model_id) == ""


# ------------------------------------------------------------------- remove


def test_remove_needs_confirmation(service, tmp_path):
    from core.model_registry import ConfirmationRequired

    target = write_gguf(tmp_path / "model.gguf")
    row = service.register(target)
    with pytest.raises(ConfirmationRequired):
        service.remove(row.model_id)
    assert service.rows(registered_only=True)


def test_remove_with_confirmation_forgets_the_record_and_keeps_the_file(service, tmp_path):
    target = write_gguf(tmp_path / "model.gguf")
    row = service.register(target)
    service.remove(row.model_id, confirm=True)
    assert service.rows(registered_only=True) == ()
    assert target.is_file()


def test_update_changes_the_display_name(service, tmp_path):
    target = write_gguf(tmp_path / "model.gguf")
    row = service.register(target, display_name="Before")
    updated = service.update(row.model_id, display_name="After")
    assert updated.display_name == "After"
    assert service.detail(row.model_id).display_name == "After"


# --------------------------------------------------------------- discovery


def test_discover_finds_a_file(service, tmp_path):
    write_gguf(tmp_path / "scan" / "found.gguf")
    report = service.discover([tmp_path / "scan"])
    assert len(report.probes) >= 1


def test_inspect_re_reads_the_file(service, tmp_path):
    target = write_gguf(tmp_path / "model.gguf", name="Inspected")
    row = service.register(target)
    probe = service.inspect(row.model_id)
    assert probe.usable is True
    assert probe.format == "gguf"


def test_inspect_of_a_vanished_file_is_not_usable(service, tmp_path):
    target = write_gguf(tmp_path / "model.gguf")
    row = service.register(target)
    target.unlink()
    probe = service.inspect(row.model_id)
    assert probe.usable is False


def test_import_custom_models_with_nothing_to_import(service):
    assert service.import_custom_models() == ()


# ------------------------------------------------------------ background work


def test_run_in_background_hands_the_result_back(service):
    done = threading.Event()
    captured: dict[str, object] = {}

    def on_done(result, error):
        captured["result"] = result
        captured["error"] = error
        done.set()

    service.run_in_background(lambda: "finished", on_done=on_done)
    assert done.wait(5)
    assert captured["result"] == "finished"
    assert captured["error"] is None


def test_run_in_background_hands_an_error_back_rather_than_raising(service):
    done = threading.Event()
    captured: dict[str, object] = {}

    def boom():
        raise RuntimeError("the runtime refused")

    def on_done(result, error):
        captured["result"] = result
        captured["error"] = error
        done.set()

    service.run_in_background(boom, on_done=on_done)
    assert done.wait(5)
    assert captured["result"] is None
    assert isinstance(captured["error"], RuntimeError)


def test_run_in_background_reports_progress(service):
    done = threading.Event()
    seen: list[str] = []

    service.run_in_background(
        lambda: None,
        on_done=lambda result, error: done.set(),
        on_progress=seen.append,
    )
    assert done.wait(5)
    assert seen


# ------------------------------------------------------------------- errors


def test_the_error_tuple_covers_the_manager_failures():
    from core.model_files import ModelFileError
    from core.model_registry import ConfirmationRequired, ModelNotFoundError, RegistryError
    from core.model_runtime import RuntimeUnavailable
    from core.model_settings import SettingsError

    for error in (
        ModelFileError,
        ModelNotFoundError,
        RegistryError,
        RuntimeUnavailable,
        SettingsError,
        ConfirmationRequired,
    ):
        assert error in MANAGER_ERRORS


def test_the_factory_builds_a_service(tmp_path):
    service = model_manager_service(tmp_path / "root")
    assert isinstance(service, ModelManagerService)
    assert service.snapshot().rows


# ------------------------------------------------------- task selection


def test_task_names_come_from_the_manager(service):
    from core.model_manager import TASKS

    assert service.task_names() == tuple(TASKS)


def test_models_for_task_returns_rows(service):
    rows = service.models_for_task("petroleum")
    assert all(isinstance(row, ModelRow) for row in rows)


def test_select_for_task_is_none_when_nothing_can_run(service):
    # On a fresh machine no catalog model is downloaded, so nothing is ready.
    assert service.select_for_task("petroleum") is None


def test_models_for_task_of_an_unknown_task_is_empty(service):
    assert service.models_for_task("no-such-task") == ()


# ------------------------------------------------------- search + dashboard


def test_search_matches_the_display_name(service, tmp_path):
    service.register(write_gguf(tmp_path / "m.gguf"), display_name="Findable Model")
    rows = service.search("findable")
    assert [row.display_name for row in rows] == ["Findable Model"]


def test_search_matches_the_path_case_insensitively(service, tmp_path):
    service.register(write_gguf(tmp_path / "Unique.gguf"), display_name="X")
    assert service.search("unique.GGUF")


def test_search_with_a_blank_query_keeps_everything(service):
    all_rows = service.rows()
    assert service.search("") == all_rows
    assert service.search("   ") == all_rows


def test_search_filters_by_state(service, tmp_path):
    service.register(write_gguf(tmp_path / "ready.gguf"), display_name="Ready One")
    ready = service.search(state=STATE_READY)
    assert any(row.display_name == "Ready One" for row in ready)
    assert all(row.state == STATE_READY for row in ready)


def test_search_of_nothing_returns_nothing(service):
    assert service.search("no-such-model-name-anywhere") == ()


def test_search_never_re_reads_the_registry(service):
    # Passing an explicit ``rows`` tuple must be honoured, so a caller can filter
    # a listing it already holds without a second read.
    rows = service.rows()
    assert service.search("", rows=rows) == rows


def test_state_names_come_from_the_manager_order():
    manager = ModelManager(Path("/tmp/nonexistent-root"), adapter_factory=lambda r: FakeAdapter())
    assert ModelManagerService(Path("/tmp/nonexistent-root"), manager=manager).state_names() == (
        STATE_ORDER
    )


def test_the_snapshot_carries_the_machine_resources(service):
    snapshot = service.snapshot()
    assert snapshot.resources is not None
    assert snapshot.resources.system


def test_the_snapshot_reports_the_active_models(service, tmp_path):
    row = service.register(write_gguf(tmp_path / "m.gguf"), display_name="Mine")
    service.load(row.model_id)
    snapshot = service.snapshot()
    assert snapshot.active == (row.model_id,)


def test_the_snapshot_has_no_active_models_before_a_load(service, tmp_path):
    service.register(write_gguf(tmp_path / "m.gguf"), display_name="Mine")
    assert service.snapshot().active == ()


def test_diagnostics_comes_straight_from_the_manager(service):
    report = service.diagnostics()
    assert report.status.registered == 0
    assert report.resources is not None

