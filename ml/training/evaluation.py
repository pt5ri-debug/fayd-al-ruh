"""The evaluation pipeline: compare a candidate against the baseline, honestly.

A trained adapter must never become the default on its own. This module is the
gate that stands between a run and a promotion: it measures the **base** model and
the **candidate** (base + adapter) on the *same held-out data*, compares them
metric by metric, and refuses a candidate that regresses.

The comparisons the task asks for:

* the original base model;
* the fine-tuned candidate;
* an existing accepted adapter, when one is supplied.

The metrics it computes are deliberately simple and **measurable**: a held-out
next-token loss, a held-out token-level accuracy, and a small instruction-following
score. They are honest measurements of the model, not proxies -- and where a metric
cannot be measured (no held-out data, no tokenizer) it is reported as absent rather
than invented as zero.

The promotion decision itself lives in :mod:`ml.training.registry` (the promotion
manager), which uses :func:`compare` here and the cognition version log's own
``should_accept`` rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ml.training.adapters import AdapterBackend

#: The metric names this pipeline produces, so a report and a comparison agree.
METRIC_LOSS = "loss"
METRIC_ACCURACY = "token_accuracy"
METRIC_INSTRUCTION = "instruction_score"

METRICS: tuple[str, ...] = (METRIC_LOSS, METRIC_ACCURACY, METRIC_INSTRUCTION)

#: A metric is "higher is better" unless it is in this set.
LOWER_IS_BETTER: frozenset[str] = frozenset({METRIC_LOSS})


def higher_is_better(metric: str) -> bool:
    return metric not in LOWER_IS_BETTER


@dataclass(frozen=True)
class MetricComparison:
    """One metric's baseline vs candidate, and the delta in the good direction."""

    metric: str
    baseline: float
    candidate: float

    @property
    def raw_delta(self) -> float:
        return self.candidate - self.baseline

    @property
    def delta(self) -> float:
        """The delta expressed so that *positive is an improvement*."""
        return self.raw_delta if higher_is_better(self.metric) else -self.raw_delta

    def describe(self) -> str:
        arrow = "improves" if self.delta > 0 else ("regresses" if self.delta < 0 else "holds")
        return (
            f"{self.metric}: {self.baseline:.4f} -> {self.candidate:.4f} "
            f"({arrow} {self.delta:+.4f})"
        )


@dataclass(frozen=True)
class EvaluationReport:
    """The full comparison: baseline, candidate, the per-metric deltas, a verdict."""

    baseline: dict[str, float] = field(default_factory=dict)
    candidate: dict[str, float] = field(default_factory=dict)
    comparisons: tuple[MetricComparison, ...] = ()
    accepted: bool = False
    verdict: str = ""
    held_out_examples: int = 0
    notes: tuple[str, ...] = ()

    def metric(self, name: str) -> MetricComparison | None:
        for comparison in self.comparisons:
            if comparison.metric == name:
                return comparison
        return None

    def describe(self) -> str:
        lines = [
            f"Evaluation on {self.held_out_examples} held-out example(s)",
        ]
        if not self.comparisons:
            lines.append("  no metric could be measured on this data")
        lines.extend(f"  {c.describe()}" for c in self.comparisons)
        lines.extend(f"  note: {note}" for note in self.notes)
        lines.append(f"  verdict: {'accept' if self.accepted else 'reject'} -- {self.verdict}")
        return "\n".join(lines)

    def to_primitive(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "baseline": dict(self.baseline),
            "candidate": dict(self.candidate),
            "comparisons": [
                {
                    "metric": c.metric,
                    "baseline": c.baseline,
                    "candidate": c.candidate,
                    "delta": c.delta,
                }
                for c in self.comparisons
            ],
            "held_out_examples": self.held_out_examples,
            "notes": list(self.notes),
            "verdict": self.verdict,
        }

    @classmethod
    def from_primitive(cls, data: object) -> EvaluationReport:
        if not isinstance(data, dict):
            raise ValueError("An evaluation report must be a mapping.")
        return cls(
            baseline=dict(data.get("baseline", {}) or {}),
            candidate=dict(data.get("candidate", {}) or {}),
            comparisons=tuple(
                MetricComparison(
                    metric=str(item.get("metric", "")),
                    baseline=float(item.get("baseline", 0.0)),
                    candidate=float(item.get("candidate", 0.0)),
                )
                for item in data.get("comparisons", ())
            ),
            accepted=bool(data.get("accepted", False)),
            verdict=str(data.get("verdict", "")),
            held_out_examples=int(data.get("held_out_examples", 0) or 0),
            notes=tuple(str(n) for n in (data.get("notes") or ())),
        )


# ------------------------------------------------------------- the measurements


def _token_stream(texts: Sequence[str], tokenizer: Any, block: int) -> list[int]:
    stream: list[int] = []
    for text in texts:
        if hasattr(tokenizer, "encode"):
            stream.extend(int(i) for i in tokenizer.encode(text))
        elif hasattr(tokenizer, "token_ids"):
            stream.extend(int(i) for i in tokenizer.token_ids(text))
        eos = getattr(tokenizer, "eos_id", None) or getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos, int):
            stream.append(eos)
    return stream


def _windows(stream: Sequence[int], block: int, limit: int) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, max(0, len(stream) - block - 1), block):
        if len(pairs) >= limit:
            break
        x = np.asarray(stream[start : start + block], dtype=np.int64)[None, :]
        y = np.asarray(stream[start + 1 : start + block + 1], dtype=np.int64)[None, :]
        pairs.append((x, y))
    return pairs


def _evaluate_model(model: Any, texts: Sequence[str], tokenizer: Any, block: int, limit: int) -> dict[str, float]:
    stream = _token_stream(texts, tokenizer, block)
    pairs = _windows(stream, block, limit)
    if not pairs:
        return {}
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    for x, y in pairs:
        logits = model.forward(x)
        total_loss += model.loss(logits, y)
        predictions = np.argmax(logits, axis=-1)
        total_correct += int((predictions == y).sum())
        total_tokens += int(y.size)
    return {
        METRIC_LOSS: total_loss / len(pairs),
        METRIC_ACCURACY: total_correct / total_tokens if total_tokens else 0.0,
    }


class _applied:
    """Apply an adapter for the duration of a measurement, then restore the base."""

    def __init__(self, backend: AdapterBackend | None, model: Any) -> None:
        self.backend = backend
        self.model = model
        self._saved: list[tuple[Any, Any]] = []

    def __enter__(self) -> None:
        if self.backend is None:
            return
        backend: Any = self.backend
        pairs = getattr(backend, "_pairs", None)
        if not isinstance(pairs, dict):
            return
        for name, pair in pairs.items():
            matrix = backend._resolve(self.model, name)
            self._saved.append((matrix, matrix.copy()))
            matrix[...] = matrix + pair.delta().astype(matrix.dtype)

    def __exit__(self, *_exc: object) -> None:
        # Restore the exact saved values, so the base is bit-for-bit unchanged.
        for matrix, original in self._saved:
            matrix[...] = original
        self._saved.clear()


def measure_loss(
    backend: AdapterBackend | None,
    model: Any,
    texts: Sequence[str],
    tokenizer: Any,
    *,
    block: int | None = None,
) -> float | None:
    """The next-token loss with the adapter applied, or ``None`` when unmeasurable."""
    if tokenizer is None or not texts:
        return None
    block = block or int(getattr(getattr(model, "config", None), "block_size", 64))
    with _applied(backend, model):
        metrics = _evaluate_model(model, texts, tokenizer, block, limit=64)
    return metrics.get(METRIC_LOSS)


def evaluate(
    model: Any,
    backend: AdapterBackend | None,
    held_out: Sequence[str],
    tokenizer: Any,
    *,
    instruction_pairs: Sequence[tuple[str, str]] = (),
    block: int | None = None,
) -> dict[str, float]:
    """Measure one model configuration (with ``backend`` applied) on held-out data."""
    if tokenizer is None or not held_out:
        return {}
    block = block or int(getattr(getattr(model, "config", None), "block_size", 64))
    with _applied(backend, model):
        metrics = _evaluate_model(model, held_out, tokenizer, block, limit=64)
        if instruction_pairs:
            metrics[METRIC_INSTRUCTION] = _instruction_score(
                model, instruction_pairs, tokenizer
            )
    return metrics


def _instruction_score(
    model: Any, pairs: Sequence[tuple[str, str]], tokenizer: Any
) -> float:
    """A small, honest instruction-following score: overlap of the model's continuation.

    For each ``(prompt, expected)`` pair the model is asked to continue the prompt
    greedily for a few tokens, and the score is the share of the expected text's
    words the continuation contains. It is a **rough** measure and is named as one.
    """
    if not pairs:
        return 0.0
    scores: list[float] = []
    for prompt, expected in pairs:
        prompt_ids = _token_stream([prompt], tokenizer, 0)
        if not prompt_ids:
            continue
        generated = _greedy_continue(model, prompt_ids, steps=16)
        text = _decode(tokenizer, generated)
        want = set(expected.casefold().split())
        got = set(text.casefold().split())
        scores.append(len(want & got) / len(want) if want else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def _greedy_continue(model: Any, prompt_ids: Sequence[int], *, steps: int) -> list[int]:
    block = int(getattr(getattr(model, "config", None), "block_size", 64))
    ids = list(prompt_ids)[-block:]
    out: list[int] = []
    for _ in range(steps):
        x = np.asarray(ids[-block:], dtype=np.int64)[None, :]
        logits = model.forward(x)
        next_id = int(np.argmax(logits[0, -1]))
        out.append(next_id)
        ids.append(next_id)
    return out


def _decode(tokenizer: Any, ids: Sequence[int]) -> str:
    if hasattr(tokenizer, "decode"):
        try:
            return str(tokenizer.decode(list(ids)))
        except Exception:  # pragma: no cover - defensive
            return ""
    return ""


# ------------------------------------------------------------- the comparison


def compare(
    baseline: dict[str, float],
    candidate: dict[str, float],
    *,
    max_regression: float = 0.02,
    min_improvement: float = 0.0,
    require_improvement: bool = True,
) -> tuple[tuple[MetricComparison, ...], bool, str]:
    """Compare two metric maps and decide whether the candidate may be promoted.

    The rule is the cognition version log's own: a metric that regresses by more
    than ``max_regression`` blocks the candidate outright; otherwise at least one
    metric must improve by more than ``min_improvement`` (when
    ``require_improvement``). A candidate that is merely *equal* is not promoted.
    """
    shared = [name for name in METRICS if name in baseline and name in candidate]
    comparisons = tuple(
        MetricComparison(name, baseline[name], candidate[name]) for name in shared
    )
    if not comparisons:
        return (), False, "no shared metric could be compared"

    regressions = [c for c in comparisons if c.delta < -abs(max_regression)]
    if regressions:
        worst = min(regressions, key=lambda c: c.delta)
        return comparisons, False, f"regresses {worst.metric} by {worst.delta:+.4f}"

    improvements = [c for c in comparisons if c.delta > abs(min_improvement)]
    if require_improvement and not improvements:
        return comparisons, False, "no measurable improvement"
    best = max(improvements, key=lambda c: c.delta) if improvements else None
    if best is None:
        return comparisons, True, "no regression (improvement not required)"
    return comparisons, True, f"improves {best.metric} by {best.delta:+.4f}"


def build_report(
    *,
    baseline: dict[str, float],
    candidate: dict[str, float],
    held_out_examples: int,
    max_regression: float = 0.02,
    min_improvement: float = 0.0,
    notes: Sequence[str] = (),
) -> EvaluationReport:
    """Assemble the evaluation report and apply the acceptance rule to it."""
    comparisons, accepted, verdict = compare(
        baseline, candidate, max_regression=max_regression, min_improvement=min_improvement
    )
    return EvaluationReport(
        baseline=dict(baseline),
        candidate=dict(candidate),
        comparisons=comparisons,
        accepted=accepted,
        verdict=verdict,
        held_out_examples=held_out_examples,
        notes=tuple(notes),
    )


__all__ = [
    "LOWER_IS_BETTER",
    "METRICS",
    "METRIC_ACCURACY",
    "METRIC_INSTRUCTION",
    "METRIC_LOSS",
    "EvaluationReport",
    "MetricComparison",
    "build_report",
    "compare",
    "evaluate",
    "higher_is_better",
    "measure_loss",
]
