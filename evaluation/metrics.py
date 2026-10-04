from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Iterable


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


@dataclass(frozen=True)
class BrierSummary:
    """Brier score and Murphy decomposition for binary forecasts."""

    count: int
    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    base_rate: float


def brier_summary(forecasts: list[tuple[float, float]]) -> BrierSummary:
    """Compute Brier score and its Murphy decomposition.

    ``forecasts`` are ``(confidence, outcome)`` pairs where ``outcome`` is 1.0
    when the event happened and 0.0 otherwise. The identity
    ``brier = reliability - resolution + uncertainty`` always holds.
    """
    if not forecasts:
        return BrierSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    brier = mean((confidence - outcome) ** 2 for confidence, outcome in forecasts)
    base_rate = mean(outcome for _, outcome in forecasts)
    uncertainty = base_rate * (1.0 - base_rate)

    buckets: dict[int, list[tuple[float, float]]] = {}
    for confidence, outcome in forecasts:
        index = min(9, max(0, int(confidence * 10)))
        buckets.setdefault(index, []).append((confidence, outcome))

    reliability = 0.0
    resolution = 0.0
    total = len(forecasts)
    for group in buckets.values():
        weight = len(group) / total
        group_confidence = mean(confidence for confidence, _ in group)
        group_outcome = mean(outcome for _, outcome in group)
        reliability += weight * (group_confidence - group_outcome) ** 2
        resolution += weight * (group_outcome - base_rate) ** 2

    return BrierSummary(
        count=total,
        brier=round(brier, 6),
        reliability=round(reliability, 6),
        resolution=round(resolution, 6),
        uncertainty=round(uncertainty, 6),
        base_rate=round(base_rate, 6),
    )


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    empirical_accuracy: float


def calibration_bins(forecasts: list[tuple[float, float]], bins: int = 10) -> list[CalibrationBin]:
    """Reliability-diagram bins for ``(confidence, outcome)`` forecasts."""
    if bins < 1:
        raise ValueError("Number of bins must be at least 1.")
    if not forecasts:
        return []

    groups: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for confidence, outcome in forecasts:
        index = min(bins - 1, max(0, int(confidence * bins)))
        groups[index].append((confidence, outcome))

    result: list[CalibrationBin] = []
    for index, group in enumerate(groups):
        if not group:
            continue
        result.append(
            CalibrationBin(
                lower=round(index / bins, 6),
                upper=round((index + 1) / bins, 6),
                count=len(group),
                mean_confidence=round(mean(confidence for confidence, _ in group), 6),
                empirical_accuracy=round(mean(outcome for _, outcome in group), 6),
            )
        )
    return result


def expected_calibration_error(forecasts: list[tuple[float, float]], bins: int = 10) -> float:
    """Weighted average gap between confidence and empirical accuracy."""
    total = len(forecasts)
    if total == 0:
        return 0.0
    return round(
        sum(
            (bucket.count / total) * abs(bucket.mean_confidence - bucket.empirical_accuracy)
            for bucket in calibration_bins(forecasts, bins)
        ),
        6,
    )


def metric_names(cls: type) -> tuple[str, ...]:
    return tuple(field.name for field in fields(cls))
