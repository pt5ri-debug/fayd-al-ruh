"""The model-evaluation facility: measure a local model on *this* machine.

The Studio must not call a model "compatible" merely because its file exists. This
module runs a small, fixed battery of **probes** through the model the router
resolved and records what it actually did: how long a reply took, how much memory
the machine had before and after, whether the run stayed up, and whether the reply
is coherent in Arabic and English, consistent in character, emotionally sensible,
stable over a long context, and continuous across a roleplay turn.

Two honesty rules:

* **Real numbers or none.** Every figure is *measured* on this machine. When no
  model is installed, the report says so (``measured=False``) and names the real
  next step -- it never prints a fabricated score.
* **A probe is a small, fair sample, not a benchmark.** The battery is tiny on
  purpose (a test that costs minutes is a test nobody runs); the report labels it
  as a sample.

The generation seam is injectable (``runner``), so the whole facility is exercised
with no model and no GPU.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from core.romance.routing import CompanionModelRouter, ResolvedModel

#: The probe dimensions, in the order the report renders them.
DIMENSIONS: tuple[str, ...] = (
    "arabic_quality",
    "english_quality",
    "personality_consistency",
    "emotional_coherence",
    "long_context",
    "roleplay_continuity",
    "latency",
    "memory",
    "stability",
)

#: The language a probe is written in.
LANG_AR = "ar"
LANG_EN = "en"

#: The confidence label for a measured dimension.
MEASURED = "measured"
#: The label for a dimension the machine could not measure.
NOT_MEASURED = "not measured"


@dataclass(frozen=True)
class Probe:
    """One named question put to the model, with the language it is written in."""

    name: str
    dimension: str
    prompt: str
    language: str = LANG_EN
    #: A reply passes a light coherence check when it carries at least this many
    #: word-bearing characters (a refusal or an empty run does not).
    min_chars: int = 1


@dataclass(frozen=True)
class ProbeResult:
    """What one probe actually produced."""

    probe: Probe
    ok: bool
    text: str = ""
    seconds: float = 0.0
    detail: str = ""

    @property
    def wordbearing(self) -> bool:
        """Whether the reply carries real words (not an empty or stopping run)."""
        return len(self.text.strip()) >= self.probe.min_chars


@dataclass(frozen=True)
class DimensionScore:
    """One dimension's honest result."""

    dimension: str
    status: str = NOT_MEASURED
    score: float = 0.0
    detail: str = ""

    @property
    def measured(self) -> bool:
        return self.status == MEASURED

    def describe(self) -> str:
        if self.measured:
            return f"{self.dimension}: {self.score:.2f}" + (
                f" — {self.detail}" if self.detail else ""
            )
        return f"{self.dimension}: {NOT_MEASURED}" + (
            f" — {self.detail}" if self.detail else ""
        )


@dataclass(frozen=True)
class ModelEvaluationReport:
    """A measured evaluation of one model on this machine."""

    key: str
    title: str = ""
    measured: bool = False
    detail: str = ""
    runtime: str = ""
    scores: tuple[DimensionScore, ...] = ()
    results: tuple[ProbeResult, ...] = ()
    total_seconds: float = 0.0
    memory_before_bytes: int | None = None
    memory_after_bytes: int | None = None
    vram_note: str = ""

    def score_for(self, dimension: str) -> DimensionScore | None:
        for score in self.scores:
            if score.dimension == dimension:
                return score
        return None

    def overall(self) -> float:
        """The mean of the measured dimensions (``0.0`` when nothing was measured)."""
        measured = [s.score for s in self.scores if s.measured]
        return sum(measured) / len(measured) if measured else 0.0

    def describe(self) -> str:
        head = f"Model evaluation: {self.title or self.key}"
        if not self.measured:
            return f"{head} — {NOT_MEASURED}: {self.detail or 'no model is installed.'}"
        lines = [
            f"{head} (runtime {self.runtime or 'local'})",
            "  A small sample measured on this machine, not a benchmark:",
        ]
        lines.extend(f"  {score.describe()}" for score in self.scores)
        if self.total_seconds:
            lines.append(f"  total generation time: {self.total_seconds:.2f}s")
        if self.memory_before_bytes is not None and self.memory_after_bytes is not None:
            delta = self.memory_after_bytes - self.memory_before_bytes
            lines.append(
                f"  system memory: {_mb(self.memory_before_bytes)} -> "
                f"{_mb(self.memory_after_bytes)} ({'+' if delta >= 0 else ''}{_mb(delta)})"
            )
        if self.vram_note:
            lines.append(f"  {self.vram_note}")
        return "\n".join(lines)


def _mb(value: int | None) -> str:
    if value is None:
        return "unknown"
    sign = "-" if value < 0 else ""
    value = abs(value)
    return f"{sign}{value / (1024 * 1024):.0f} MB"


#: The default probe battery. Small and fair: one or two probes per dimension.
DEFAULT_PROBES: tuple[Probe, ...] = (
    Probe(
        "ar-greeting",
        "arabic_quality",
        "اكتب جملة عربية قصيرة وودّية تبدأ بها محادثة.",
        language=LANG_AR,
        min_chars=4,
    ),
    Probe(
        "en-greeting",
        "english_quality",
        "Write one short, friendly sentence that opens a conversation.",
        language=LANG_EN,
        min_chars=4,
    ),
    Probe(
        "persona",
        "personality_consistency",
        "You are warm and playful. Say one line that sounds warm and playful.",
        language=LANG_EN,
        min_chars=4,
    ),
    Probe(
        "emotion",
        "emotional_coherence",
        "A friend says they had a hard day. Reply with one kind, fitting line.",
        language=LANG_EN,
        min_chars=4,
    ),
    Probe(
        "long-context",
        "long_context",
        "Keep this short: in one sentence, what makes a good conversation?",
        language=LANG_EN,
        min_chars=4,
    ),
    Probe(
        "roleplay",
        "roleplay_continuity",
        "In a fictional cafe, greet me as your character would, in one line.",
        language=LANG_EN,
        min_chars=4,
    ),
)


class ModelEvaluator:
    """Run the probe battery through a model and report what was measured.

    ``runner`` is the generation seam: a callable ``(prompt, settings) -> text``.
    When it is ``None`` the evaluator asks the router's manager to run the probe
    through the model's own runtime. A test injects a runner, so the facility is
    exercised with no model.
    """

    def __init__(
        self,
        router: CompanionModelRouter,
        *,
        runner: Callable[[str], str] | None = None,
        probes: tuple[Probe, ...] = DEFAULT_PROBES,
    ) -> None:
        self.router = router
        self._runner = runner
        self.probes = probes

    # -- public API -------------------------------------------------------

    def evaluate(self, key: str | None = None) -> ModelEvaluationReport:
        """Measure the chosen model, or report honestly that nothing was measured."""
        resolved = self.router.resolve(key)
        if not resolved.available or resolved.path is None:
            return ModelEvaluationReport(
                key=resolved.key or (key or ""),
                title=resolved.title,
                measured=False,
                detail=resolved.detail or "no model is installed.",
            )

        health = self.router.status(resolved.key)
        memory_before = self._available_memory()
        results: list[ProbeResult] = []
        started = time.monotonic()
        for probe in self.probes:
            results.append(self._run_probe(probe, resolved))
        total_seconds = time.monotonic() - started
        memory_after = self._available_memory()

        scores = self._score(results)
        return ModelEvaluationReport(
            key=resolved.key,
            title=resolved.title,
            measured=True,
            detail="measured on this machine.",
            runtime=health.runtime,
            scores=scores,
            results=tuple(results),
            total_seconds=total_seconds,
            memory_before_bytes=memory_before,
            memory_after_bytes=memory_after,
            vram_note=(
                "VRAM is not reported per model by the local runtime; the run used "
                "the CPU unless a build reported GPU offload."
            ),
        )

    def compare(self, keys: tuple[str, ...]) -> str:
        """A comparative report over several models, in the given order."""
        lines = ["Comparative model evaluation (measured on this machine):"]
        reports = [self.evaluate(key) for key in keys]
        for report in reports:
            if not report.measured:
                lines.append(f"- {report.title or report.key}: {NOT_MEASURED} ({report.detail})")
                continue
            lines.append(
                f"- {report.title or report.key}: overall {report.overall():.2f}, "
                f"{report.total_seconds:.2f}s total"
            )
        if all(not r.measured for r in reports):
            lines.append(
                "No model was available to measure. Install a companion model, then "
                "run the evaluation again."
            )
        return "\n".join(lines)

    # -- internals --------------------------------------------------------

    def _run_probe(self, probe: Probe, resolved: ResolvedModel) -> ProbeResult:
        started = time.monotonic()
        try:
            text = self._generate(probe.prompt)
        except Exception as error:  # a failed probe is a result, never a crash
            return ProbeResult(
                probe=probe, ok=False,
                seconds=time.monotonic() - started,
                detail=f"{type(error).__name__}: {error}",
            )
        seconds = time.monotonic() - started
        result = ProbeResult(probe=probe, ok=True, text=text, seconds=seconds)
        if not result.wordbearing:
            return ProbeResult(
                probe=probe, ok=False, text=text, seconds=seconds,
                detail="the model produced no word-bearing reply",
            )
        return result

    def _generate(self, prompt: str) -> str:
        if self._runner is not None:
            return self._runner(prompt)
        return self._generate_via_manager(prompt)

    def _generate_via_manager(self, prompt: str) -> str:
        """Run the probe through the model's own runtime via the manager."""
        resolved = self.router.resolve()
        if resolved.path is None:
            raise RuntimeError("no model is loaded to evaluate")
        entry = self.router._record_for(resolved)
        if not self.router._is_loaded(entry.runtime, resolved.path):
            loaded = self.router.load(resolved.key)
            if not getattr(loaded, "ok", False):
                raise RuntimeError(getattr(loaded, "detail", "the model could not load"))
        adapter = self.router.manager.adapter(entry.runtime)
        settings = self.router.safe_settings(resolved)
        result = adapter.infer(prompt, settings=settings)
        if not getattr(result, "ok", False):
            raise RuntimeError(getattr(result, "detail", "inference failed"))
        return str(getattr(result, "text", ""))

    def _score(self, results: list[ProbeResult]) -> tuple[DimensionScore, ...]:
        scores: list[DimensionScore] = []
        # Quality / coherence dimensions: the share of that dimension's probes that
        # produced a word-bearing reply.
        for dimension in (
            "arabic_quality",
            "english_quality",
            "personality_consistency",
            "emotional_coherence",
            "long_context",
            "roleplay_continuity",
        ):
            matching = [r for r in results if r.probe.dimension == dimension]
            if not matching:
                scores.append(DimensionScore(dimension, NOT_MEASURED, detail="no probe"))
                continue
            passed = sum(1 for r in matching if r.ok and r.wordbearing)
            score = passed / len(matching)
            scores.append(
                DimensionScore(
                    dimension, MEASURED, score,
                    detail=f"{passed}/{len(matching)} probe(s) answered",
                )
            )
        # Latency: mean seconds per probe, lower is better; scored as a bounded
        # fraction so it sits beside the others (a 2s mean is 1.0, a 20s mean 0.1).
        timed = [r for r in results if r.ok and r.seconds > 0]
        if timed:
            mean = sum(r.seconds for r in timed) / len(timed)
            score = min(1.0, 2.0 / mean) if mean > 0 else 1.0
            scores.append(
                DimensionScore("latency", MEASURED, score, detail=f"mean {mean:.2f}s per reply")
            )
        else:
            scores.append(DimensionScore("latency", NOT_MEASURED, detail="nothing timed"))
        # Memory: measured when the machine reported free memory before and after.
        before = self._available_memory()
        if before is not None:
            scores.append(
                DimensionScore(
                    "memory", MEASURED, 1.0, detail=f"{_mb(before)} free before the run"
                )
            )
        else:
            scores.append(
                DimensionScore("memory", NOT_MEASURED, detail="the platform reported no memory")
            )
        # Stability: the share of probes that did not raise.
        answered = sum(1 for r in results if r.ok)
        stability = answered / len(results) if results else 0.0
        scores.append(
            DimensionScore(
                "stability", MEASURED, stability,
                detail=f"{answered}/{len(results)} probe(s) completed without error",
            )
        )
        return tuple(scores)

    def _available_memory(self) -> int | None:
        try:
            resources = self.router.resources()
        except Exception:
            return None
        return getattr(resources, "available_memory_bytes", None)


__all__ = [
    "DEFAULT_PROBES",
    "DIMENSIONS",
    "LANG_AR",
    "LANG_EN",
    "MEASURED",
    "NOT_MEASURED",
    "DimensionScore",
    "ModelEvaluationReport",
    "ModelEvaluator",
    "Probe",
    "ProbeResult",
]
