"""The resource governor arbitrates from real figures and never guesses."""

from __future__ import annotations

import pytest
from core.model_resources import SystemResources
from core.resource_governor import (
    DEFAULT_HEADROOM,
    GovernorReason,
    ResourceGovernor,
    ResourceRequest,
    estimate_model_memory,
)

GB = 1024**3


def _machine(available: int | None = 8 * GB, total: int | None = 16 * GB) -> SystemResources:
    return SystemResources(cpu_count=8, total_memory_bytes=total, available_memory_bytes=available)


# -- construction ---------------------------------------------------------


def test_headroom_must_be_a_fraction():
    with pytest.raises(ValueError):
        ResourceGovernor(headroom=1.0)


def test_at_least_one_model_slot_is_required():
    with pytest.raises(ValueError):
        ResourceGovernor(max_concurrent_models=0)


def test_at_least_one_task_slot_is_required():
    with pytest.raises(ValueError):
        ResourceGovernor(max_concurrent_tasks=0)


def test_a_request_needs_a_label():
    with pytest.raises(ValueError):
        ResourceRequest(label="  ")


def test_a_negative_memory_estimate_is_refused():
    with pytest.raises(ValueError):
        ResourceRequest(label="x", memory_bytes=-1)


# -- grants and deferrals -------------------------------------------------


def test_a_modest_request_is_granted():
    governor = ResourceGovernor(resources=lambda: _machine())
    decision = governor.request(ResourceRequest("ocr", memory_bytes=1 * GB))
    assert decision.granted is True
    assert decision.reason is GovernorReason.GRANTED
    assert "granted" in decision.describe()


def test_a_request_beyond_the_headroom_budget_is_deferred():
    governor = ResourceGovernor(resources=lambda: _machine(available=8 * GB))
    # budget = 8 GB * 0.75 = 6 GB
    decision = governor.request(ResourceRequest("big model", memory_bytes=7 * GB, model=True))
    assert decision.granted is False
    assert decision.reason is GovernorReason.LOW_MEMORY
    assert "deferred" in decision.describe()


def test_unknown_available_memory_defers_a_request_that_needs_it():
    """An unknown is not plenty -- a need that cannot be judged is deferred."""
    governor = ResourceGovernor(resources=lambda: _machine(available=None))
    decision = governor.request(ResourceRequest("needs memory", memory_bytes=1 * GB))
    assert decision.granted is False
    assert decision.reason is GovernorReason.UNKNOWN_MEMORY


def test_unknown_available_memory_does_not_block_a_request_with_no_need():
    governor = ResourceGovernor(resources=lambda: _machine(available=None))
    decision = governor.request(ResourceRequest("cheap task"))
    assert decision.granted is True


def test_a_disabled_governor_defers_with_a_reason():
    governor = ResourceGovernor(resources=lambda: _machine(), enabled=False)
    decision = governor.request(ResourceRequest("anything"))
    assert decision.granted is False
    assert decision.reason is GovernorReason.DISABLED


# -- occupancy ------------------------------------------------------------


def test_acquire_reserves_a_model_slot_and_release_frees_it():
    governor = ResourceGovernor(resources=lambda: _machine(), max_concurrent_models=1)
    granted = governor.acquire(ResourceRequest("qwen", memory_bytes=1 * GB, model=True))
    assert granted.granted is True
    assert governor.active_models == ("qwen",)
    second = governor.can_load_model(1 * GB, label="other")
    assert second.granted is False
    assert second.reason is GovernorReason.TOO_MANY_MODELS
    governor.release("qwen")
    assert governor.active_models == ()
    assert governor.can_load_model(1 * GB).granted is True


def test_a_deferral_reserves_nothing():
    governor = ResourceGovernor(resources=lambda: _machine(available=1 * GB), max_concurrent_models=1)
    deferred = governor.acquire(ResourceRequest("big", memory_bytes=4 * GB, model=True))
    assert deferred.granted is False
    assert governor.active_models == ()


def test_release_is_idempotent():
    governor = ResourceGovernor(resources=lambda: _machine())
    governor.release("never-taken")
    assert governor.active_models == ()


def test_too_many_tasks_is_deferred():
    governor = ResourceGovernor(resources=lambda: _machine(), max_concurrent_tasks=1)
    governor.acquire(ResourceRequest("task one"))
    decision = governor.can_run_task(label="task two")
    assert decision.granted is False
    assert decision.reason is GovernorReason.TOO_MANY_TASKS


# -- observation ----------------------------------------------------------


def test_an_observer_sees_every_decision_and_never_breaks_it():
    seen = []

    def observer(decision):
        seen.append(decision.label)
        raise RuntimeError("observer failed")

    governor = ResourceGovernor(resources=lambda: _machine(), on_decision=observer)
    decision = governor.request(ResourceRequest("x"))
    assert decision.granted is True
    assert seen == ["x"]


def test_snapshot_and_describe_report_the_state():
    governor = ResourceGovernor(resources=lambda: _machine())
    governor.acquire(ResourceRequest("m", model=True, memory_bytes=1 * GB))
    snapshot = governor.snapshot()
    assert snapshot["active_models"] == ["m"]
    assert snapshot["enabled"] is True
    assert snapshot["headroom"] == DEFAULT_HEADROOM
    assert "Resource governor" in governor.describe()


def test_decisions_are_recorded():
    governor = ResourceGovernor(resources=lambda: _machine())
    governor.request(ResourceRequest("one"))
    governor.request(ResourceRequest("two"))
    assert [d.label for d in governor.decisions] == ["one", "two"]


def test_decision_to_primitive_is_json_safe():
    governor = ResourceGovernor(resources=lambda: _machine())
    payload = governor.request(ResourceRequest("x", memory_bytes=1 * GB)).to_primitive()
    assert payload["granted"] is True
    assert payload["reason"] == "granted"


def test_estimate_model_memory_is_none_for_an_uninspectable_file(tmp_path):
    """An uninspectable file yields ``None`` -- unknown, never a zero standing in."""
    assert estimate_model_memory(tmp_path / "nope.gguf") is None
