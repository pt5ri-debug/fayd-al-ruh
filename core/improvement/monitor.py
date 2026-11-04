"""The performance monitor: the loop's only source of operational numbers.

Section 4 of the phase asks for structured, timestamped, provenance-carrying
metrics that are local, offline, configurable, and free of unnecessary sensitive
data. This module is that collector, and it is deliberately small:

* it **records** :class:`~core.improvement.metrics.MetricSample` objects into a
  bounded ring (so a long session cannot grow without limit) and, optionally,
  into a durable store;
* it **summarises** them by kind -- a counter is totalled, a ratio or a gauge is
  averaged, a duration also reports a p95 -- and never mixes measured and
  estimated samples in one figure without saying how many of each it used;
* it **subscribes** to the Phase-1 event bus, so the core's own lifecycle events
  (a cycle, a belief revision, a tool decision) become metrics without the core
  knowing anything about this module; and
* it reads the machine's **real** figures for the resource gauges through the
  existing :mod:`core.model_resources`, and records them as measured -- but only
  when asked, so a caller that does not want a resource probe pays nothing.

Honesty rules it enforces rather than documents: a sample must name a metric; a
counter is never averaged into a ratio; and an unknown metric name is recorded
but flagged with a neutral kind rather than silently dropped.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from core.best_effort import best_effort
from core.improvement.metrics import (
    METRIC_CATALOGUE,
    MetricKind,
    MetricSample,
    MetricSpec,
    Provenance,
    is_known,
    metric_names,
)

#: How many samples the monitor keeps in memory. A whole session's telemetry
#: fits comfortably; the bound is what stops a runaway producer from growing the
#: process.
DEFAULT_MAX_SAMPLES = 4000

#: How many of the newest samples a durable save writes, so a store stays small.
DEFAULT_SAVED_SAMPLES = 2000


@dataclass(frozen=True)
class MonitorConfig:
    """What the monitor is allowed to collect -- monitoring is configurable.

    ``include`` is an allow-list of metric names; empty means "every catalogue
    metric". ``collect_resources`` gates the one reading that touches the
    machine, so a headless or restricted caller can switch it off.
    """

    enabled: bool = True
    max_samples: int = DEFAULT_MAX_SAMPLES
    saved_samples: int = DEFAULT_SAVED_SAMPLES
    collect_resources: bool = True
    include: frozenset[str] = frozenset()

    def allows(self, name: str) -> bool:
        if not self.enabled:
            return False
        if self.include and name not in self.include:
            return False
        return True


@dataclass(frozen=True)
class MetricSummary:
    """An honest roll-up of one metric's samples."""

    name: str
    kind: MetricKind
    component: str
    unit: str
    count: int
    measured: int
    estimated: int
    total: float
    mean: float
    minimum: float
    maximum: float
    last: float
    p95: float

    def describe(self) -> str:
        unit = f" {self.unit}" if self.unit else ""
        provenance = f"{self.measured} measured"
        if self.estimated:
            provenance += f", {self.estimated} estimated"
        if self.kind is MetricKind.COUNTER:
            return f"{self.name}: total {self.total:g}{unit} ({provenance})"
        return (
            f"{self.name}: mean {self.mean:.4g}{unit} "
            f"[{self.minimum:.4g}..{self.maximum:.4g}] ({provenance})"
        )


def _percentile(values: list[float], fraction: float) -> float:
    """A simple, deterministic nearest-rank percentile (no interpolation)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


class PerformanceMonitor:
    """A bounded, provenance-aware collector over the metric catalogue."""

    def __init__(
        self,
        *,
        config: MonitorConfig | None = None,
        store: Any | None = None,
    ) -> None:
        self.config = config or MonitorConfig()
        self._store = store
        self._samples: deque[MetricSample] = deque(maxlen=max(1, self.config.max_samples))
        self._unknown: set[str] = set()
        self._subscriptions: list[str] = []

    # -- recording ----------------------------------------------------------
    def record(
        self,
        name: str,
        value: float,
        *,
        provenance: Provenance = Provenance.MEASURED,
        labels: dict[str, str] | None = None,
    ) -> MetricSample | None:
        """Record one observation, or ``None`` when the config refuses it.

        A non-numeric or non-finite value is refused by :class:`MetricSample`
        itself; the monitor lets that raise so a programming error is visible
        rather than silently dropped.
        """
        if not self.config.allows(name):
            return None
        return self.record_sample(MetricSample.of(name, value, provenance=provenance, labels=labels or {}))

    def record_sample(self, sample: MetricSample) -> MetricSample:
        """Record a pre-built sample (kind/component/unit already set)."""
        if not is_known(sample.name):
            self._unknown.add(sample.name)
        self._samples.append(sample)
        return sample

    # -- reading ------------------------------------------------------------
    def samples(self, *, name: str | None = None, limit: int | None = None) -> tuple[MetricSample, ...]:
        """Samples oldest-first, optionally filtered by name and capped."""
        chosen = [s for s in self._samples if name is None or s.name == name]
        if limit is not None and limit >= 0:
            chosen = chosen[-limit:]
        return tuple(chosen)

    def latest(self, name: str) -> MetricSample | None:
        for sample in reversed(self._samples):
            if sample.name == name:
                return sample
        return None

    def total(self, name: str) -> float:
        """The sum of a counter's samples (0.0 when none were recorded)."""
        return sum(s.value for s in self._samples if s.name == name)

    def summary(self, name: str) -> MetricSummary | None:
        chosen = [s for s in self._samples if s.name == name]
        if not chosen:
            return None
        kind = chosen[-1].kind
        values = [s.value for s in chosen]
        total = sum(values)
        return MetricSummary(
            name=name,
            kind=kind,
            component=chosen[-1].component,
            unit=chosen[-1].unit,
            count=len(chosen),
            measured=sum(1 for s in chosen if s.provenance is Provenance.MEASURED),
            estimated=sum(1 for s in chosen if s.provenance is Provenance.ESTIMATED),
            total=total,
            mean=total / len(values),
            minimum=min(values),
            maximum=max(values),
            last=values[-1],
            p95=_percentile(values, 0.95) if kind is MetricKind.DURATION else max(values),
        )

    def summaries(self) -> tuple[MetricSummary, ...]:
        """One summary per metric that has at least one sample, in catalogue order."""
        present = {s.name for s in self._samples}
        ordered = [name for name in metric_names() if name in present]
        ordered += sorted(name for name in present if not is_known(name))
        return tuple(summary for name in ordered if (summary := self.summary(name)) is not None)

    def unknown_metrics(self) -> tuple[str, ...]:
        """Names recorded that the catalogue does not list, so a typo is visible."""
        return tuple(sorted(self._unknown))

    # -- resource probe -----------------------------------------------------
    def sample_resources(self) -> tuple[MetricSample, ...]:
        """Read the machine's real figures into the resource gauges (best-effort).

        A figure the platform cannot report is simply not recorded -- an unknown
        is never turned into a zero, which would read as "no memory in use".
        """
        if not self.config.collect_resources:
            return ()
        recorded: list[MetricSample] = []
        with best_effort("sample system resources"):
            from core.model_resources import system_resources

            resources = system_resources()
            total = getattr(resources, "total_memory_bytes", None)
            available = getattr(resources, "available_memory_bytes", None)
            if isinstance(total, int) and isinstance(available, int) and total > 0:
                used_mb = (total - available) / (1024 * 1024)
                sample = self.record("resource.memory_usage", used_mb)
                if sample is not None:
                    recorded.append(sample)
                self.record("resource.consumption", (total - available) / total)
            disk = getattr(resources, "disk_free_bytes", None)
            if isinstance(disk, int) and disk > 0:
                sample = self.record("resource.disk_free", disk / (1024 * 1024))
                if sample is not None:
                    recorded.append(sample)
        return tuple(recorded)

    # -- event bus ----------------------------------------------------------
    def attach(self, bus: Any, *, events: Iterable[str] = ()) -> tuple[str, ...]:
        """Subscribe to an event bus, turning each event into a counter sample.

        The mapping is deliberately tiny: every subscribed event increments a
        counter named ``event.<name>``. A caller that wants a specific event
        mapped to a specific catalogue metric records it itself; this is only the
        catch-all, so *something* is observed without the core being changed.

        The bus's own prefix rule is a *segment* match, so ``"cycle"`` matches
        ``cycle.started`` but not ``cyclone``; an empty prefix subscribes to
        everything. A caller that passes no ``events`` subscribes to all events.
        """
        subscribed: list[str] = []
        prefixes = tuple(events) or ("",)

        def _handler(event: Any) -> None:
            name = getattr(event, "name", "")
            if not name:
                return
            self.record(f"event.{name}", 1.0, provenance=Provenance.MEASURED, labels={"event": str(name)})

        subscribe = getattr(bus, "subscribe", None)
        if callable(subscribe):
            for prefix in prefixes:
                with best_effort("subscribe the monitor to an event"):
                    subscribe(_handler, prefix)
                    subscribed.append(prefix or "*")
        self._subscriptions.extend(subscribed)
        return tuple(subscribed)

    # -- persistence --------------------------------------------------------
    def save(self) -> None:
        """Persist the newest samples to the durable store, if one was supplied."""
        if self._store is None:
            return
        newest = list(self._samples)[-self.config.saved_samples:]
        with best_effort("save metric samples"):
            self._store.save(
                "metrics",
                "samples",
                {"samples": [s.to_primitive() for s in newest]},
            )

    def load(self) -> int:
        """Restore samples from the durable store; returns how many were read."""
        if self._store is None:
            return 0
        data: object = None
        with best_effort("load metric samples"):
            data = self._store.load("metrics", "samples")
        if not isinstance(data, dict):
            return 0
        raw = data.get("samples")
        if not isinstance(raw, list):
            return 0
        restored = 0
        for item in raw:
            if not isinstance(item, dict):
                continue
            self.record_sample(MetricSample.from_primitive(item))
            restored += 1
        return restored

    def clear(self) -> None:
        """Forget every in-memory sample (a test, or a fresh diagnostic window)."""
        self._samples.clear()
        self._unknown.clear()

    def describe(self) -> str:
        """A short, readable status line for a window or a CLI."""
        lines = [f"Performance monitor: {len(self._samples)} sample(s), "
                 f"{len(self.summaries())} metric(s) tracked."]
        if self._unknown:
            lines.append(f"Unknown metric names recorded: {', '.join(self.unknown_metrics())}")
        return "\n".join(lines)


def catalogue() -> tuple[MetricSpec, ...]:
    """The metric catalogue, re-exported so a caller need not import two modules."""
    return METRIC_CATALOGUE


def subscribe_controller(bus: Any, monitor: PerformanceMonitor) -> tuple[str, ...]:
    """Convenience: subscribe a monitor to the controller's lifecycle events."""
    return monitor.attach(bus, events=("cycle.", "belief.", "memory.", "tool.", "plan.", "execution."))


__all__ = [
    "DEFAULT_MAX_SAMPLES",
    "DEFAULT_SAVED_SAMPLES",
    "MetricSummary",
    "MonitorConfig",
    "PerformanceMonitor",
    "catalogue",
    "subscribe_controller",
]
