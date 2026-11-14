"""Tests for the companion model-evaluation facility (Phase 3).

The evaluator is driven through its ``runner`` seam (a fake generator) and a fake
catalog, so the whole facility is exercised with no model and no GPU. One test
proves the *honest* case: with no model installed the report says "not measured"
rather than printing a fabricated score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.model_manager import ModelManager
from core.model_runtime import (
    InferenceResult,
    LoadResult,
    RuntimeCapabilities,
    RuntimeStatus,
)
from core.romance.model_eval import (
    DEFAULT_PROBES,
    NOT_MEASURED,
    ModelEvaluator,
    Probe,
)
from core.romance.routing import CompanionModelRouter


@dataclass
class FakeModel:
    key: str
    title: str = ""
    engine: str = "natasha"
    path: Path | None = None
    available: bool = False
    detail: str = ""
    download_command: str = ""


class FakeAdapter:
    name = "fayd-inprocess"

    def __init__(self) -> None:
        self._loaded: Path | None = None

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(can_load=True, can_unload=True, can_infer=True)

    def status(self) -> RuntimeStatus:
        return RuntimeStatus(runtime=self.name, available=True)

    def validate(self, probe) -> tuple[bool, str]:
        return True, ""

    def discover(self) -> tuple[str, ...]:
        return ()

    def load(self, model_path: Path, *, settings=None) -> LoadResult:
        self._loaded = Path(model_path)
        return LoadResult(ok=True, runtime=self.name, model_path=Path(model_path))

    def unload(self) -> LoadResult:
        self._loaded = None
        return LoadResult(ok=True, runtime=self.name)

    def loaded(self) -> bool:
        return self._loaded is not None

    def loaded_path(self) -> Path | None:
        return self._loaded

    def infer(self, prompt: str, *, settings=None) -> InferenceResult:
        return InferenceResult(ok=True, runtime=self.name, text="hello there", seconds=0.01)


def _gguf(tmp_path: Path) -> Path:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF\x00\x00\x00\x00" + b"\x00" * 64)
    return path


def _router(tmp_path: Path, *, available: bool = True):
    models = (
        FakeModel(
            "natasha-default",
            title="Companion",
            path=_gguf(tmp_path) if available else None,
            available=available,
            detail="" if available else "not installed",
        ),
    )
    manager = ModelManager(tmp_path / "data", adapter_factory=lambda runtime: FakeAdapter())
    return CompanionModelRouter(manager, catalog=lambda: models, default_key="natasha-default")


def _score(report, dimension: str):
    """The named dimension's score, asserted present (narrows the Optional)."""
    score = report.score_for(dimension)
    assert score is not None, f"{dimension} missing from {report.scores!r}"
    return score


# ------------------------------------------------------------------ honest case


def test_no_model_reports_not_measured(tmp_path: Path):
    router = _router(tmp_path, available=False)
    report = ModelEvaluator(router).evaluate()
    assert report.measured is False
    assert report.overall() == 0.0
    assert "not measured" in report.describe().lower()
    # Every dimension is present but not measured -- never a fabricated score.
    assert all(s.status == NOT_MEASURED for s in report.scores)


# ------------------------------------------------------------------ measured case


def test_a_runner_measures_every_dimension(tmp_path: Path):
    router = _router(tmp_path)
    report = ModelEvaluator(router, runner=lambda prompt: "a warm and friendly line").evaluate()
    assert report.measured is True
    dimensions = {s.dimension for s in report.scores}
    assert {
        "arabic_quality",
        "english_quality",
        "personality_consistency",
        "emotional_coherence",
        "long_context",
        "roleplay_continuity",
        "latency",
        "stability",
    } <= dimensions
    # Every probe produced a word-bearing reply, so the quality dimensions are 1.0.
    assert _score(report, "english_quality").score == 1.0
    assert report.overall() > 0.0


def test_an_empty_reply_is_not_wordbearing(tmp_path: Path):
    router = _router(tmp_path)
    report = ModelEvaluator(router, runner=lambda prompt: "   ").evaluate()
    assert report.measured is True
    # A stopping run is not a pass: the quality dimensions fall.
    assert _score(report, "english_quality").score == 0.0
    assert _score(report, "stability").score < 1.0


def test_a_raising_runner_is_a_failed_probe_not_a_crash(tmp_path: Path):
    router = _router(tmp_path)

    def boom(prompt: str) -> str:
        raise RuntimeError("engine exploded")

    report = ModelEvaluator(router, runner=boom).evaluate()
    assert report.measured is True
    assert _score(report, "stability").score == 0.0
    assert all(not r.ok for r in report.results)


def test_latency_is_measured_lower_is_better(tmp_path: Path):
    router = _router(tmp_path)
    report = ModelEvaluator(router, runner=lambda prompt: "ok reply here").evaluate()
    latency = _score(report, "latency")
    assert latency.measured is True
    assert 0.0 < latency.score <= 1.0


def test_custom_probes_are_used(tmp_path: Path):
    router = _router(tmp_path)
    probe = Probe("only", "english_quality", "say hi", min_chars=1)
    report = ModelEvaluator(router, runner=lambda prompt: "hi", probes=(probe,)).evaluate()
    assert len(report.results) == 1
    assert report.results[0].probe.name == "only"


def test_default_battery_covers_the_required_dimensions():
    dimensions = {p.dimension for p in DEFAULT_PROBES}
    assert "arabic_quality" in dimensions
    assert "english_quality" in dimensions
    assert "roleplay_continuity" in dimensions


def test_compare_names_every_model(tmp_path: Path):
    router = _router(tmp_path)
    text = ModelEvaluator(router, runner=lambda prompt: "hello").compare(("natasha-default",))
    assert "natasha-default" in text or "Companion" in text
    assert "Comparative" in text


def test_compare_with_no_model_is_honest(tmp_path: Path):
    router = _router(tmp_path, available=False)
    text = ModelEvaluator(router).compare(("natasha-default",))
    assert "not measured" in text.lower()
    assert "Install a companion model" in text
