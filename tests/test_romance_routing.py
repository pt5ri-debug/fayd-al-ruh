"""Tests for the companion model router (Phase 3).

Every test drives the router with a **fake** runtime adapter and a **fake**
catalog through the project's own seams, so the whole router is exercised with no
model, no GPU, and no network. The real Model Manager is used (with the fake
adapter), so the registration, listing, load, and unload paths are the real ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from core.model_manager import ModelManager
from core.model_runtime import (
    InferenceResult,
    LoadResult,
    RuntimeCapabilities,
    RuntimeStatus,
)
from core.romance.routing import (
    CompanionModelError,
    CompanionModelRouter,
)


@dataclass
class FakeModel:
    """A catalog entry, shaped like ``core.language_models.LanguageModel``."""

    key: str
    title: str = ""
    engine: str = "natasha"
    path: Path | None = None
    available: bool = False
    detail: str = ""
    download_command: str = ""


class FakeAdapter:
    """A runtime adapter that loads and answers without any real engine."""

    name = "fayd-inprocess"

    def __init__(self, *, answer: str = "ready", fail_load: bool = False) -> None:
        self._loaded: Path | None = None
        self._answer = answer
        self._fail_load = fail_load

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(can_load=True, can_unload=True, can_infer=True)

    def status(self) -> RuntimeStatus:
        return RuntimeStatus(runtime=self.name, available=True)

    def validate(self, probe) -> tuple[bool, str]:
        return True, ""

    def discover(self) -> tuple[str, ...]:
        return ()

    def load(self, model_path: Path, *, settings=None) -> LoadResult:
        if self._fail_load:
            return LoadResult(ok=False, runtime=self.name, detail="simulated load failure")
        self._loaded = Path(model_path)
        return LoadResult(ok=True, runtime=self.name, model_path=Path(model_path), seconds=0.01)

    def unload(self) -> LoadResult:
        self._loaded = None
        return LoadResult(ok=True, runtime=self.name, detail="unloaded")

    def loaded(self) -> bool:
        return self._loaded is not None

    def loaded_path(self) -> Path | None:
        return self._loaded

    def infer(self, prompt: str, *, settings=None) -> InferenceResult:
        if self._loaded is None:
            return InferenceResult(ok=False, runtime=self.name, detail="nothing is loaded")
        return InferenceResult(ok=True, runtime=self.name, text=self._answer, seconds=0.02)


def _gguf(tmp_path: Path, name: str = "model.gguf") -> Path:
    path = tmp_path / name
    path.write_bytes(b"GGUF\x00\x00\x00\x00" + b"\x00" * 64)
    return path


def _router(tmp_path: Path, models: tuple[FakeModel, ...], *, adapter: FakeAdapter | None = None):
    adapter = adapter or FakeAdapter()
    manager = ModelManager(tmp_path / "data", adapter_factory=lambda runtime: adapter)
    router = CompanionModelRouter(
        manager,
        catalog=lambda: models,
        default_key=models[0].key if models else "",
    )
    return router, adapter


# ------------------------------------------------------------------ choices


def test_choices_are_the_catalog_with_honest_readiness(tmp_path: Path):
    ready = _gguf(tmp_path)
    models = (
        FakeModel("natasha-default", title="Companion", path=ready, available=True),
        FakeModel("natasha-other", title="Other", available=False, detail="not installed"),
    )
    router, _ = _router(tmp_path, models)
    choices = router.choices()
    assert [c.key for c in choices] == ["natasha-default", "natasha-other"]
    assert choices[0].available is True
    assert choices[1].available is False
    assert choices[1].detail == "not installed"


def test_resolve_falls_back_to_an_available_model(tmp_path: Path):
    ready = _gguf(tmp_path)
    models = (
        FakeModel("natasha-default", path=ready, available=True),
        FakeModel("natasha-other", available=False),
    )
    router, _ = _router(tmp_path, models)
    # A stale/unknown key resolves to the first available model, not a crash.
    resolved = router.resolve("does-not-exist")
    assert resolved.key == "natasha-default"
    assert resolved.available is True


def test_resolve_reports_honestly_when_nothing_is_installed(tmp_path: Path):
    models = (FakeModel("natasha-default", available=False, detail="not installed"),)
    router, _ = _router(tmp_path, models)
    resolved = router.resolve()
    assert resolved.available is False
    assert resolved.path is None
    health = router.status()
    assert health.ready is False
    assert "not installed" in health.detail or health.hint


# ------------------------------------------------------------------ status / health


def test_status_reports_ready_for_an_installed_model(tmp_path: Path):
    ready = _gguf(tmp_path)
    router, _ = _router(tmp_path, (FakeModel("natasha-default", path=ready, available=True),))
    health = router.status("natasha-default")
    assert health.ready is True
    assert health.loaded is False


def test_status_reflects_a_load(tmp_path: Path):
    ready = _gguf(tmp_path)
    router, _ = _router(tmp_path, (FakeModel("natasha-default", path=ready, available=True),))
    result = router.load("natasha-default")
    assert result.ok is True
    health = router.status("natasha-default")
    assert health.loaded is True


# ------------------------------------------------------------------ load / unload


def test_load_and_unload(tmp_path: Path):
    ready = _gguf(tmp_path)
    router, adapter = _router(
        tmp_path, (FakeModel("natasha-default", path=ready, available=True),)
    )
    assert router.load("natasha-default").ok is True
    assert adapter.loaded_path() == ready
    assert router.unload("natasha-default").ok is True
    assert adapter.loaded_path() is None


def test_load_without_a_file_is_refused(tmp_path: Path):
    router, _ = _router(tmp_path, (FakeModel("natasha-default", available=False),))
    with pytest.raises(CompanionModelError):
        router.load("natasha-default")


def test_a_failed_load_is_reported_not_raised(tmp_path: Path):
    ready = _gguf(tmp_path)
    router, _ = _router(
        tmp_path,
        (FakeModel("natasha-default", path=ready, available=True),),
        adapter=FakeAdapter(fail_load=True),
    )
    result = router.load("natasha-default")
    assert result.ok is False
    assert "simulated load failure" in result.detail


# ------------------------------------------------------------------ settings / estimates


def test_safe_settings_are_cpu_first_and_bounded():
    from core.romance.routing import SAFE_CONTEXT, SAFE_MAX_TOKENS

    router = CompanionModelRouter(ModelManager())
    settings = router.safe_settings()
    assert settings.context == SAFE_CONTEXT
    assert settings.max_tokens == SAFE_MAX_TOKENS
    assert settings.gpu_layers == 0


def test_memory_estimate_is_available_for_a_registered_model(tmp_path: Path):
    ready = _gguf(tmp_path)
    router, _ = _router(tmp_path, (FakeModel("natasha-default", path=ready, available=True),))
    estimate = router.memory_estimate("natasha-default")
    assert estimate is not None
    # The estimate is an estimate: it always says so.
    assert estimate.high >= estimate.low
    assert estimate.confidence


def test_preflight_is_none_without_a_model(tmp_path: Path):
    router, _ = _router(tmp_path, (FakeModel("natasha-default", available=False),))
    assert router.preflight("natasha-default") is None


# ------------------------------------------------------------------ ranking / download


def test_rank_and_fallback_plan_run_against_the_manager(tmp_path: Path):
    ready = _gguf(tmp_path)
    router, _ = _router(tmp_path, (FakeModel("natasha-default", path=ready, available=True),))
    ranked = router.rank()
    assert isinstance(ranked, tuple)
    plan = router.fallback_plan("natasha-default")
    assert plan is not None


def test_download_hint_reports_the_governed_command(tmp_path: Path):
    models = (
        FakeModel(
            "natasha-default",
            available=False,
            download_command="python -m cli natasha download --approve",
        ),
    )
    router, _ = _router(tmp_path, models)
    assert router.download_hint("natasha-default") == "python -m cli natasha download --approve"


# ------------------------------------------------------------------ cancellation


def test_generation_handle_cancels_and_times_out(tmp_path: Path):
    import time

    ready = _gguf(tmp_path)
    router, _ = _router(tmp_path, (FakeModel("natasha-default", path=ready, available=True),))
    handle = router.begin_generation("natasha-default", timeout_seconds=5)
    assert router.should_stop(handle) is False
    handle.cancel()
    assert router.should_stop(handle) is True

    expired = router.begin_generation("natasha-default", timeout_seconds=0.01)
    time.sleep(0.02)
    assert router.should_stop(expired) is True


# ------------------------------------------------------------------ assignment


def test_assignment_persists_through_the_injected_writer(tmp_path: Path):
    saved: dict[str, str] = {}
    router = CompanionModelRouter(
        ModelManager(tmp_path / "data"),
        catalog=lambda: (),
        character_model_key=lambda cid, key: saved.__setitem__(cid, key),
        mode_model_key=lambda mode, key: saved.__setitem__(mode, key),
    )
    router.assign_character("CHR-1", "natasha-default")
    router.assign_mode("romantic", "natasha-default")
    assert saved == {"CHR-1": "natasha-default", "romantic": "natasha-default"}


def test_assignment_without_a_writer_is_refused(tmp_path: Path):
    router = CompanionModelRouter(ModelManager(tmp_path / "data"), catalog=lambda: ())
    with pytest.raises(CompanionModelError):
        router.assign_character("CHR-1", "x")


# ------------------------------------------------------------------ generate


def test_generate_runs_through_the_manager_and_returns_the_result(tmp_path: Path):
    ready = _gguf(tmp_path)
    router, _adapter = _router(
        tmp_path, (FakeModel("natasha-default", path=ready, available=True),),
        adapter=FakeAdapter(answer="Hello from the model."),
    )
    result = router.generate("Say hello.", key="natasha-default")
    assert result.ok is True
    assert result.text == "Hello from the model."
    # The one inference call goes through the manager's own runtime -- the
    # runtime name is the manager's, never a second engine's.
    assert result.runtime == "fayd-inprocess"


def test_generate_reports_honestly_when_the_model_is_not_installed(tmp_path: Path):
    router, _adapter = _router(tmp_path, (FakeModel("natasha-default", available=False),))
    result = router.generate("Say hello.", key="natasha-default")
    assert result.ok is False
    assert "not installed" in result.detail


def test_generate_honours_a_cancelled_handle_before_it_starts(tmp_path: Path):
    ready = _gguf(tmp_path)
    router, _adapter = _router(
        tmp_path, (FakeModel("natasha-default", path=ready, available=True),)
    )
    handle = router.begin_generation("natasha-default", timeout_seconds=5)
    handle.cancel()
    result = router.generate("Say hello.", key="natasha-default", handle=handle)
    assert result.ok is False
    assert "cancel" in result.detail
