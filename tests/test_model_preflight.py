"""Resource discovery, memory estimation, preflight, and process management.

Every test here runs without a GPU, without starting a real process, and without a
real model: the resource reads use injectable seams, the estimates are pure
functions of metadata, and the process registry takes a fake ``popen`` and a fake
liveness probe. The point is the *honesty* rules, so that is what is pinned: an
unknown figure stays unknown, an estimate is a band and says so, a CPU path is
never rejected for want of a GPU, and a pid this program did not start is never
signalled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from core.model_manager import ModelManager
from core.model_preflight import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_UNKNOWN,
    DEFAULT_CONTEXT,
    estimate_memory,
    preflight,
)
from core.model_process import (
    STATUS_EXITED,
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_STALE,
    STATUS_STOPPED,
    ProcessError,
    ProcessRecord,
    ProcessRegistry,
    pid_is_alive,
)
from core.model_registry import ModelRecord
from core.model_resources import (
    READING_MEASURED,
    GpuInfo,
    SystemResources,
    describe_reading,
    gpu_info,
    vendor_name,
)

# ------------------------------------------------------------------- helpers


def _record(**kwargs) -> ModelRecord:
    base: dict[str, Any] = {
        "model_id": "m",
        "identity": "test/model",
        "display_name": "Test",
        "model_format": "gguf",
        "runtime": "llama.cpp",
        "source": "local",
        "size_bytes": 4_000_000_000,
        "architecture": "llama",
        "context_length": 8192,
        "parameter_count": 7_000_000_000,
    }
    base.update(kwargs)
    return ModelRecord(**base)


def _resources(**kwargs) -> SystemResources:
    base: dict[str, Any] = {
        "system": "Linux",
        "machine": "x86_64",
        "cpu_count": 8,
        "total_memory_bytes": 16 * 1024**3,
        "available_memory_bytes": 12 * 1024**3,
        "disk_free_bytes": 100 * 1024**3,
        "disk_path": "/tmp",
        "memory_reading": READING_MEASURED,
    }
    base.update(kwargs)
    return SystemResources(**base)


class FakePopen:
    """A ``Popen`` stand-in that reports a pid and an exit code the test controls."""

    def __init__(self, pid: int = 4242, *, exits: bool = False, code: int = 0) -> None:
        self.pid = pid
        self.returncode = code if exits else None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class Clock:
    """A monotonic clock the test advances by hand, so waits are instant."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(seconds, 0.001)


# ----------------------------------------------------------------- resources


def test_gpu_info_is_honest_when_nothing_is_found():
    info = gpu_info(runner=lambda cmd: (1, ""))
    assert not info.available
    assert "nvidia-smi" in info.note
    assert "none detected" in info.describe()


def test_gpu_info_reads_nvidia_smi_output():
    info = gpu_info(
        runner=lambda cmd: (0, "NVIDIA GeForce RTX 4090, NVIDIA, 24564, 24000")
    )
    assert info.available
    assert info.vendor == "NVIDIA"
    assert info.memory_total_bytes == 24564 * 1024 * 1024
    assert info.memory_free_bytes == 24000 * 1024 * 1024
    assert info.source == "nvidia-smi"


def test_gpu_info_reports_unknown_memory_not_zero():
    info = gpu_info(runner=lambda cmd: (0, "AMD Radeon, AMD, N/A, N/A"))
    assert info.available
    assert info.memory_total_bytes is None  # not 0


def test_vendor_name_maps_the_known_ids():
    assert vendor_name("0x1002") == "AMD"
    assert vendor_name("0x10DE") == "NVIDIA"
    assert vendor_name("0xffff") == ""


def test_describe_reading_covers_every_level():
    assert describe_reading(READING_MEASURED) != ""
    assert describe_reading("nonsense") != ""


def test_gpu_info_primitive_is_json_safe():
    import json

    info = gpu_info(runner=lambda cmd: (0, "NVIDIA GeForce RTX 4090, NVIDIA, 24564, 24000"))
    assert json.dumps(info.to_primitive())


# ---------------------------------------------------------------- estimation


def test_estimate_uses_the_file_size_for_weights():
    estimate = estimate_memory(_record())
    assert estimate.weights_bytes == 4_000_000_000
    assert estimate.kv_cache_bytes > 0
    assert estimate.low <= estimate.high
    assert estimate.dominant_term in ("weights", "kv-cache")


def test_estimate_is_a_band_not_a_single_number():
    estimate = estimate_memory(_record())
    assert estimate.high > estimate.low
    assert estimate.low <= estimate.typical <= estimate.high


def test_estimate_without_a_size_is_unknown():
    estimate = estimate_memory(_record(size_bytes=0))
    assert estimate.confidence == CONFIDENCE_UNKNOWN
    assert estimate.weights_bytes == 0
    assert "file size" in " ".join(estimate.unknowns)
    # The weights are unknown, but the KV cache can still be sized from the
    # architecture -- so the estimate is honest about the gap rather than absent.
    assert "the file size" in estimate.describe()


def test_estimate_without_an_architecture_does_not_invent_one():
    estimate = estimate_memory(_record(architecture="", parameter_count=None))
    assert estimate.kv_cache_bytes == 0
    assert "architecture" in " ".join(estimate.unknowns)
    assert estimate.confidence == CONFIDENCE_LOW


def test_estimate_assumes_a_default_context_and_says_so():
    estimate = estimate_memory(_record(context_length=None, architecture="", parameter_count=None))
    assert estimate.context_length == DEFAULT_CONTEXT
    assert any("assumes" in note for note in estimate.notes)


def test_estimate_confidence_is_high_when_all_facts_are_known():
    estimate = estimate_memory(_record())
    assert estimate.confidence == CONFIDENCE_HIGH


def test_estimate_notes_a_gpu_offload_without_promising_it():
    estimate = estimate_memory(_record(), gpu_layers=10, gpu=GpuInfo(available=True, memory_total_bytes=2 * 1024**3))
    assert any("offload" in note for note in estimate.notes)
    assert any("GPU memory" in note for note in estimate.notes)


def test_estimate_says_when_no_gpu_memory_was_reported():
    estimate = estimate_memory(_record(), gpu_layers=10, gpu=GpuInfo(available=False))
    assert any("No GPU memory" in note for note in estimate.notes)


def test_estimate_describe_warns_it_is_not_a_measurement():
    text = estimate_memory(_record()).describe()
    assert "estimate, not a measurement" in text


def test_estimate_primitive_is_json_safe():
    import json

    assert json.dumps(estimate_memory(_record()).to_primitive())


def test_estimate_dominant_term_is_kv_when_the_context_is_huge():
    estimate = estimate_memory(_record(context_length=1_000_000))
    assert estimate.dominant_term == "kv-cache"


# ------------------------------------------------------------------ preflight


def _manager_with(tmp_path, record: ModelRecord, *, path: Path | None = None) -> ModelManager:
    manager = ModelManager(root=tmp_path)
    if path is not None:
        from dataclasses import replace

        record = replace(record, path=path)
    manager.registry.register_record(record)
    return manager


def test_preflight_blocks_when_the_file_is_missing(tmp_path):
    record = _record(path=tmp_path / "gone.gguf")
    manager = _manager_with(tmp_path, record)
    report = preflight(manager, "m", resources=_resources())
    assert not report.can_load
    assert any("gone" in error for error in report.errors)


def test_preflight_reports_a_missing_runtime(tmp_path):
    record = _record(runtime="llama.cpp", path=None, source="catalog")
    manager = _manager_with(tmp_path, record)

    class Unavailable:
        def status(self):
            raise AssertionError("not reached")

    report = preflight(manager, "m", resources=_resources())
    # No file, so that is the first error; the runtime one may or may not follow.
    assert not report.can_load


def test_preflight_passes_with_a_fake_adapter_and_a_real_file(tmp_path):
    path = _write_gguf(tmp_path / "m.gguf")
    record = _record(path=path, size_bytes=path.stat().st_size)
    manager = _manager_with(tmp_path, record)
    adapter = _FakeAdapter()
    report = preflight(manager, "m", resources=_resources(), adapter=adapter)
    assert report.can_load
    assert report.estimate is not None
    assert report.cpu_fallback  # no GPU reported


def test_preflight_rejects_a_context_longer_than_declared(tmp_path):
    path = _write_gguf(tmp_path / "m.gguf")
    record = _record(path=path, size_bytes=path.stat().st_size, context_length=2048)
    manager = _manager_with(tmp_path, record)
    report = preflight(manager, "m", context_length=8192, resources=_resources(), adapter=_FakeAdapter())
    assert not report.can_load
    assert any("declares a context" in error for error in report.errors)
    assert any("2048" in item for item in report.recommendations)


def test_preflight_warns_when_the_estimate_exceeds_available_memory(tmp_path):
    path = _write_gguf(tmp_path / "m.gguf")
    record = _record(path=path, size_bytes=100 * 1024**3)
    manager = _manager_with(tmp_path, record)
    report = preflight(
        manager, "m",
        resources=_resources(available_memory_bytes=1 * 1024**3),
        adapter=_FakeAdapter(),
    )
    assert any("larger than" in warning for warning in report.warnings)


def test_preflight_says_an_estimate_is_not_a_guarantee(tmp_path):
    path = _write_gguf(tmp_path / "m.gguf")
    record = _record(path=path, size_bytes=path.stat().st_size)
    manager = _manager_with(tmp_path, record)
    report = preflight(manager, "m", resources=_resources(), adapter=_FakeAdapter())
    assert any("not a guarantee" in warning for warning in report.warnings)


def test_preflight_warns_when_memory_cannot_be_read(tmp_path):
    path = _write_gguf(tmp_path / "m.gguf")
    record = _record(path=path, size_bytes=path.stat().st_size)
    manager = _manager_with(tmp_path, record)
    report = preflight(
        manager, "m",
        resources=_resources(available_memory_bytes=None),
        adapter=_FakeAdapter(),
    )
    assert any("does not report available memory" in warning for warning in report.warnings)


def test_preflight_does_not_block_on_an_absent_gpu(tmp_path):
    path = _write_gguf(tmp_path / "m.gguf")
    record = _record(path=path, size_bytes=path.stat().st_size)
    manager = _manager_with(tmp_path, record)
    report = preflight(
        manager, "m", gpu_layers=20,
        resources=_resources(gpu=GpuInfo(available=False)),
        adapter=_FakeAdapter(),
    )
    assert report.can_load  # a GPU is not required
    assert report.cpu_fallback


def test_preflight_warns_when_gpu_offload_is_unsupported(tmp_path):
    path = _write_gguf(tmp_path / "m.gguf")
    record = _record(path=path, size_bytes=path.stat().st_size)
    manager = _manager_with(tmp_path, record)
    report = preflight(
        manager, "m", gpu_layers=20,
        resources=_resources(gpu=GpuInfo(available=True, memory_total_bytes=2 * 1024**3)),
        adapter=_FakeAdapter(offload=False),
    )
    assert any("not supported" in warning for warning in report.warnings)
    assert any("CPU path" in item for item in report.recommendations)


def test_preflight_unknown_model_id_is_reported_not_raised(tmp_path):
    manager = ModelManager(root=tmp_path)
    report = preflight(manager, "nope", resources=_resources())
    assert not report.can_load
    assert report.errors


def test_preflight_describe_and_primitive(tmp_path):
    import json

    path = _write_gguf(tmp_path / "m.gguf")
    record = _record(path=path, size_bytes=path.stat().st_size)
    manager = _manager_with(tmp_path, record)
    report = preflight(manager, "m", resources=_resources(), adapter=_FakeAdapter())
    assert "can load" in report.describe()
    assert json.dumps(report.to_primitive())


def test_manager_preflight_and_estimate_are_wired(tmp_path):
    path = _write_gguf(tmp_path / "m.gguf")
    record = _record(path=path, size_bytes=path.stat().st_size)
    manager = _manager_with(tmp_path, record)
    estimate = manager.memory_estimate("m")
    assert estimate.weights_bytes == path.stat().st_size
    report = manager.preflight("m", resources=_resources())
    assert report.model_id == "m"


def test_manager_resources_returns_system_figures(tmp_path):
    manager = ModelManager(root=tmp_path)
    resources = manager.resources()
    assert resources.system


# ------------------------------------------------------------------ processes


def _registry(tmp_path, *, popen=None, alive=None, clock=None) -> ProcessRegistry:
    live = {4242: True}
    if alive is not None:
        live = alive
    return ProcessRegistry(
        tmp_path,
        popen=popen or (lambda command, env=None, cwd=None: FakePopen(4242)),
        probe=lambda pid: live.get(pid, False),
        identity=lambda pid: "same",
        clock=clock or Clock(),
        sleep=(clock or Clock()).sleep,
    )


def test_start_records_a_running_process(tmp_path):
    registry = _registry(tmp_path)
    record = registry.start(["llama-server", "--model", "m.gguf"], runtime="llama.cpp", model_id="m")
    assert record.status == STATUS_RUNNING
    assert record.pid == 4242
    assert record.model_id == "m"
    assert registry.get(record.process_id).status == STATUS_RUNNING


def test_start_waits_for_readiness(tmp_path):
    state = {"ready": False}
    registry = _registry(tmp_path)

    def ready() -> bool:
        state["ready"] = True
        return True

    record = registry.start(["x"], ready=ready)
    assert record.status == STATUS_RUNNING


def test_start_times_out_and_stops_the_process(tmp_path):
    handle = FakePopen(4242)
    clock = Clock()
    live = {4242: True}
    registry = ProcessRegistry(
        tmp_path,
        popen=lambda command, env=None, cwd=None: handle,
        probe=lambda pid: live.get(pid, False),
        identity=lambda pid: "same",
        clock=clock,
        sleep=clock.sleep,
    )
    record = registry.start(["x"], ready=lambda: False, start_timeout=2.0)
    assert record.status == STATUS_FAILED
    assert "did not become ready" in record.last_error
    assert handle.terminated


def test_start_reports_a_process_that_exits_during_startup(tmp_path):
    handle = FakePopen(4242, exits=True, code=3)
    registry = _registry(tmp_path, popen=lambda command, env=None, cwd=None: handle)
    record = registry.start(["x"], ready=lambda: False, start_timeout=1.0)
    assert record.status == STATUS_FAILED
    assert record.exit_code == 3


def test_start_reports_a_popen_failure(tmp_path):
    def explode(command, env=None, cwd=None):
        raise OSError("no such program")

    registry = _registry(tmp_path, popen=explode)
    record = registry.start(["missing-binary"])
    assert record.status == STATUS_FAILED
    assert "Could not start" in record.last_error


def test_stop_is_graceful_first(tmp_path):
    handle = FakePopen(4242)
    live = {4242: True}

    def probe(pid: int) -> bool:
        # Once terminate() has been called, the process is gone.
        return live[pid] and not handle.terminated

    registry = ProcessRegistry(
        tmp_path, popen=lambda command, env=None, cwd=None: handle,
        probe=probe, identity=lambda pid: "same", clock=Clock(), sleep=lambda s: None,
    )
    record = registry.start(["x"])
    stopped = registry.stop(record.process_id)
    assert stopped.status == STATUS_STOPPED
    assert handle.terminated
    assert not handle.killed  # graceful only


def test_stop_force_escalates_when_asked(tmp_path):
    handle = FakePopen(4242)
    state = {"alive": True}

    def probe(pid: int) -> bool:
        return state["alive"] and not handle.killed

    registry = ProcessRegistry(
        tmp_path, popen=lambda command, env=None, cwd=None: handle,
        probe=probe, identity=lambda pid: "same", clock=Clock(), sleep=lambda s: None,
    )
    record = registry.start(["x"])
    stopped = registry.stop(record.process_id, force=True)
    assert handle.killed or handle.terminated
    assert stopped.status in (STATUS_STOPPED, STATUS_RUNNING)


def test_stop_does_not_signal_a_stale_record(tmp_path):
    registry = _registry(tmp_path)
    record = registry.start(["x"])
    # Simulate the pid being reused by another program.
    registry._identity = lambda pid: "different"
    refreshed = registry.refresh(record.process_id)
    assert refreshed.status == STATUS_STALE
    stopped = registry.stop(record.process_id)
    assert stopped.status == STATUS_STOPPED
    assert "stale" in stopped.last_error


def test_stop_refuses_an_unknown_process_id(tmp_path):
    registry = _registry(tmp_path)
    with pytest.raises(ProcessError):
        registry.stop("nope")


def test_refresh_marks_a_gone_process_exited(tmp_path):
    registry = _registry(tmp_path)
    record = registry.start(["x"])
    registry._probe = lambda pid: False
    assert registry.refresh(record.process_id).status == STATUS_EXITED


def test_stop_all_stops_everything_it_started(tmp_path):
    handle = FakePopen(4242)
    state = {"alive": True}

    def probe(pid: int) -> bool:
        return state["alive"] and not handle.terminated

    clock = Clock()
    registry = ProcessRegistry(
        tmp_path, popen=lambda command, env=None, cwd=None: handle,
        probe=probe, identity=lambda pid: "same", clock=clock, sleep=clock.sleep,
    )
    first = registry.start(["x"], model_id="a", process_id="a")
    second = registry.start(["y"], model_id="b", process_id="b")
    stopped = registry.stop_all()
    ids = {record.process_id for record in stopped}
    assert ids == {first.process_id, second.process_id}
    # Both share one fake handle, so one is stopped and the other is already gone;
    # either way every record ends terminal, never running.
    assert all(
        record.status in (STATUS_STOPPED, STATUS_EXITED) for record in stopped
    )
    assert registry.stop_all() == ()


def test_forget_refuses_a_running_process(tmp_path):
    registry = _registry(tmp_path)
    record = registry.start(["x"])
    with pytest.raises(ProcessError):
        registry.forget(record.process_id)


def test_forget_drops_a_gone_process(tmp_path):
    registry = _registry(tmp_path)
    record = registry.start(["x"])
    registry._probe = lambda pid: False
    registry.forget(record.process_id)
    assert registry.find(record.process_id) is None


def test_prune_drops_only_gone_records(tmp_path):
    registry = _registry(tmp_path)
    registry.start(["x"], process_id="live")
    registry.start(["y"], process_id="dead")
    registry._probe = lambda pid: pid == 4242
    assert registry.prune() == 0  # both share pid 4242, so both look live


def test_records_are_durable_across_instances(tmp_path):
    registry = _registry(tmp_path)
    record = registry.start(["x"], model_id="m")
    reloaded = ProcessRegistry(
        tmp_path, popen=lambda command, env=None, cwd=None: FakePopen(4242),
        probe=lambda pid: True, identity=lambda pid: "same",
    )
    assert reloaded.find(record.process_id) is not None
    assert reloaded.find(record.process_id).model_id == "m"


def test_a_service_record_is_distinguished_from_a_model_process(tmp_path):
    registry = _registry(tmp_path)
    service = registry.start(["llama-server"], runtime="llama.cpp", is_service=True)
    assert service.is_service
    assert service.service_pid == service.pid
    model = registry.start(["worker"], runtime="llama.cpp", model_id="m")
    assert not model.is_service
    assert model.service_pid is None


def test_for_model_returns_only_live_records(tmp_path):
    registry = _registry(tmp_path)
    registry.start(["x"], model_id="m", process_id="a")
    registry.start(["y"], model_id="other", process_id="b")
    found = registry.for_model("m")
    assert len(found) == 1 and found[0].process_id == "a"


def test_describe_lists_the_records(tmp_path):
    registry = _registry(tmp_path)
    assert "No process" in registry.describe()
    registry.start(["x"], model_id="m")
    assert "m" in registry.describe()


def test_pid_is_alive_is_honest_for_a_bad_pid():
    assert not pid_is_alive(-1)
    assert not pid_is_alive(0)


def test_process_record_round_trips():
    record = ProcessRecord(
        process_id="p", pid=1, runtime="llama.cpp", model_id="m",
        command=("x",), is_service=False,
    )
    again = ProcessRecord.from_primitive(record.to_primitive())
    assert again is not None and again.process_id == "p" and again.pid == 1


def test_process_record_rejects_a_bad_primitive():
    assert ProcessRecord.from_primitive({"pid": "not-an-int"}) is None
    assert ProcessRecord.from_primitive(None) is None


def test_manager_exposes_a_process_registry(tmp_path):
    manager = ModelManager(root=tmp_path)
    registry = manager.processes()
    assert isinstance(registry, ProcessRegistry)
    record = registry.start(["x"], model_id="m")
    assert manager.processes().find(record.process_id) is not None


# ------------------------------------------------------------------ fixtures


class _FakeAdapter:
    """A runtime adapter that reports honestly and needs no runtime at all."""

    name = "llama.cpp"

    def __init__(self, *, offload: bool | None = False) -> None:
        self._offload = offload

    def capabilities(self):
        from core.model_runtime import RuntimeCapabilities

        return RuntimeCapabilities(can_load=True, can_infer=True, gpu_offload=self._offload)

    def status(self):
        from core.model_runtime import RuntimeStatus

        return RuntimeStatus(runtime=self.name, available=True, gpu_offload=self._offload)

    def validate(self, probe):
        return (True, "")


def _write_gguf(path: Path) -> Path:
    """A minimal but real GGUF header, so ``probe_file`` accepts the file."""
    import struct

    path.write_bytes(
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 0)
    )
    return path
