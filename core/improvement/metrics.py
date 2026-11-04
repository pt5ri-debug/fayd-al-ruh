"""The one structured metric vocabulary the self-improvement loop reads.

A self-improvement system is only as honest as the numbers it reasons over, so
this module fixes two things and invents nothing:

* **what a metric is** -- a :class:`MetricSpec` (name, kind, unit, component,
  and a one-line description) drawn from one closed catalogue, so no surface
  names a metric another does not know; and
* **where a number came from** -- every :class:`MetricSample` carries a
  :class:`Provenance`, so a *measured* figure (a real latency, a real success
  count) is never confused with an *estimated* one (a projection, a heuristic).
  Section 4 of the phase requires that distinction, and section 5 forbids
  treating a correlation as a cause; keeping provenance on every sample is how
  both rules are made structural rather than advisory.

The catalogue covers exactly the operational dimensions the phase lists -- task
success and failure, reasoning accuracy, retrieval quality, knowledge
integration and duplication, retrieval and inference latency, tool execution,
planning efficiency, repeated failures, resource consumption (memory, CPU),
model load/unload behaviour, and user corrections and feedback.

Dependency-free, Tk-free, and import-safe: nothing here reads the machine, a
model, or the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class MetricKind(str, Enum):
    """How a metric's value behaves, so an aggregator can summarise it honestly."""

    COUNTER = "counter"      # a running count; summarised as a total
    GAUGE = "gauge"          # a momentary reading; summarised as last/mean/max
    RATIO = "ratio"          # a 0..1 share; summarised as a mean
    DURATION = "duration"    # milliseconds; summarised as mean/p95


class Provenance(str, Enum):
    """Where a sample's number came from -- the honesty rule made explicit."""

    MEASURED = "measured"    # read from the system or timed directly
    ESTIMATED = "estimated"  # projected, sampled, or inferred


#: The components a metric can belong to. A closed set, so a weakness can name
#: the affected component without inventing a new name each time.
COMPONENTS: tuple[str, ...] = (
    "assistant",
    "retrieval",
    "memory",
    "reasoning",
    "planning",
    "execution",
    "knowledge",
    "learning",
    "training",
    "resource",
    "model",
    "interface",
    "system",
)


@dataclass(frozen=True)
class MetricSpec:
    """One metric the loop knows about: its name, kind, unit, and meaning."""

    name: str
    kind: MetricKind
    component: str
    unit: str = ""
    description: str = ""
    #: True when a *higher* value is the good outcome. Kept so a reader (and a
    #: weakness detector) never has to guess a metric's direction.
    higher_is_better: bool = True


#: The held-in catalogue. Names are stable dotted identifiers so they can be used
#: as JSON keys and as a store's key without escaping.
METRIC_CATALOGUE: tuple[MetricSpec, ...] = (
    # -- tasks ---------------------------------------------------------------
    MetricSpec("task.success_rate", MetricKind.RATIO, "execution", "ratio",
               "Share of tasks that completed successfully.", True),
    MetricSpec("task.failure_rate", MetricKind.RATIO, "execution", "ratio",
               "Share of tasks that failed.", False),
    MetricSpec("task.repeated_failure", MetricKind.COUNTER, "execution", "count",
               "How often the same task failed again after a retry.", False),
    # -- reasoning -----------------------------------------------------------
    MetricSpec("reasoning.accuracy", MetricKind.RATIO, "reasoning", "ratio",
               "Share of reasoning conclusions that verified as supported.", True),
    MetricSpec("reasoning.unsupported_claim", MetricKind.COUNTER, "reasoning", "count",
               "Claims the groundedness gate refused to publish.", False),
    MetricSpec("reasoning.inference_latency", MetricKind.DURATION, "reasoning", "ms",
               "Wall time one inference or reasoning call took.", False),
    # -- retrieval -----------------------------------------------------------
    MetricSpec("retrieval.quality", MetricKind.RATIO, "retrieval", "ratio",
               "Share of queries whose top passage was relevant.", True),
    MetricSpec("retrieval.latency", MetricKind.DURATION, "retrieval", "ms",
               "Wall time one retrieval call took.", False),
    MetricSpec("memory.retrieval_latency", MetricKind.DURATION, "memory", "ms",
               "Wall time one memory retrieval took.", False),
    # -- knowledge -----------------------------------------------------------
    MetricSpec("knowledge.integration_error", MetricKind.COUNTER, "knowledge", "count",
               "Claims that failed to integrate into the graph.", False),
    MetricSpec("knowledge.duplicate_rate", MetricKind.RATIO, "knowledge", "ratio",
               "Share of incoming knowledge found to be a duplicate.", False),
    MetricSpec("knowledge.contradiction", MetricKind.COUNTER, "knowledge", "count",
               "Stored contradictions observed.", False),
    # -- planning ------------------------------------------------------------
    MetricSpec("planning.efficiency", MetricKind.RATIO, "planning", "ratio",
               "Share of planned tasks that were needed (no wasted steps).", True),
    MetricSpec("planning.mistake", MetricKind.COUNTER, "planning", "count",
               "Plans that failed validation or needed replanning.", False),
    # -- tools ---------------------------------------------------------------
    MetricSpec("tool.execution_success", MetricKind.RATIO, "execution", "ratio",
               "Share of permitted tool calls that succeeded.", True),
    MetricSpec("tool.unnecessary_use", MetricKind.COUNTER, "execution", "count",
               "Tool calls that were not required for the goal.", False),
    # -- resources -----------------------------------------------------------
    MetricSpec("resource.memory_usage", MetricKind.GAUGE, "resource", "MB",
               "Process/system memory in use.", False),
    MetricSpec("resource.cpu_utilization", MetricKind.GAUGE, "resource", "percent",
               "CPU utilisation.", False),
    MetricSpec("resource.disk_free", MetricKind.GAUGE, "resource", "MB",
               "Free disk space.", True),
    MetricSpec("resource.consumption", MetricKind.GAUGE, "resource", "ratio",
               "Overall share of the machine the run used.", False),
    # -- models --------------------------------------------------------------
    MetricSpec("model.load", MetricKind.COUNTER, "model", "count",
               "How many times a model was loaded.", False),
    MetricSpec("model.unload", MetricKind.COUNTER, "model", "count",
               "How many times a model was unloaded.", False),
    # -- user signal ---------------------------------------------------------
    MetricSpec("user.correction", MetricKind.COUNTER, "interface", "count",
               "Answers the user corrected.", False),
    MetricSpec("user.feedback", MetricKind.COUNTER, "interface", "count",
               "Feedback signals received.", False),
    # -- latency (end to end) ------------------------------------------------
    MetricSpec("assistant.latency", MetricKind.DURATION, "assistant", "ms",
               "Wall time one answer took.", False),
    MetricSpec("system.error", MetricKind.COUNTER, "system", "count",
               "Recorded best-effort failures (a swallowed error made visible).", False),
)

#: Fast lookup by name; built once.
_SPEC_BY_NAME: dict[str, MetricSpec] = {spec.name: spec for spec in METRIC_CATALOGUE}


def metric_spec(name: str) -> MetricSpec | None:
    """The spec for a metric name, or ``None`` for one not in the catalogue."""
    return _SPEC_BY_NAME.get(name)


def metric_names() -> tuple[str, ...]:
    """Every metric name the loop knows, in catalogue order."""
    return tuple(spec.name for spec in METRIC_CATALOGUE)


def is_known(name: str) -> bool:
    """True when a name is in the catalogue (so a caller can refuse a typo)."""
    return name in _SPEC_BY_NAME


def _as_enum(enum: type[Enum], value: object, fallback: Enum) -> Enum:
    """Read an enum from a stored string, falling back rather than raising."""
    if isinstance(value, str):
        try:
            return enum(value)
        except ValueError:
            return fallback
    return fallback


@dataclass(frozen=True)
class MetricSample:
    """One observation: a value for a named metric, with its provenance.

    ``labels`` is a small, bounded set of string tags (``question``, ``tool``)
    used to group samples; ``at`` preserves the timestamp so a trend is
    reconstructable and an audit can say *when* something happened.
    """

    name: str
    value: float
    provenance: Provenance = Provenance.MEASURED
    kind: MetricKind = MetricKind.GAUGE
    component: str = "system"
    unit: str = ""
    at: datetime = field(default_factory=lambda: datetime.now(UTC))
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A metric sample must name its metric.")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError("A metric sample's value must be a real number.")

    @classmethod
    def of(cls, name: str, value: float, **kwargs: object) -> MetricSample:
        """Build a sample, filling kind/component/unit from the catalogue.

        An unknown metric name is still recorded (a caller may measure something
        the catalogue does not yet list), but it is given a neutral kind so an
        aggregator never misreads it.
        """
        spec = metric_spec(name)
        if spec is not None:
            kwargs.setdefault("kind", spec.kind)
            kwargs.setdefault("component", spec.component)
            kwargs.setdefault("unit", spec.unit)
        return cls(name=name, value=float(value), **kwargs)  # type: ignore[arg-type]

    def to_primitive(self) -> dict[str, object]:
        """A JSON-safe form, so a sample can be stored or exported."""
        return {
            "name": self.name,
            "value": float(self.value),
            "provenance": self.provenance.value,
            "kind": self.kind.value,
            "component": self.component,
            "unit": self.unit,
            "at": self.at.isoformat(),
            "labels": dict(self.labels),
        }

    @classmethod
    def from_primitive(cls, data: dict[str, object]) -> MetricSample:
        """Rebuild a sample from :meth:`to_primitive` output, tolerating junk."""
        raw_at = data.get("at")
        at = datetime.now(UTC)
        if isinstance(raw_at, str):
            try:
                at = datetime.fromisoformat(raw_at)
            except ValueError:
                at = datetime.now(UTC)
        raw_value = data.get("value", 0.0)
        value = float(raw_value) if isinstance(raw_value, (int, float)) else 0.0
        labels = data.get("labels")
        return cls(
            name=str(data.get("name", "unknown")),
            value=value,
            provenance=_as_enum(Provenance, data.get("provenance"), Provenance.MEASURED),  # type: ignore[arg-type]
            kind=_as_enum(MetricKind, data.get("kind"), MetricKind.GAUGE),  # type: ignore[arg-type]
            component=str(data.get("component", "system")),
            unit=str(data.get("unit", "")),
            at=at,
            labels={str(k): str(v) for k, v in labels.items()} if isinstance(labels, dict) else {},
        )


__all__ = [
    "COMPONENTS",
    "METRIC_CATALOGUE",
    "MetricKind",
    "MetricSample",
    "MetricSpec",
    "Provenance",
    "is_known",
    "metric_names",
    "metric_spec",
]
