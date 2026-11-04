# Performance Monitoring (Phase 9, section 4)

The performance monitor is the loop's **only** source of operational numbers. It
collects structured, timestamped, provenance-carrying metrics locally, so the
weakness detector reasons over measurements rather than assumptions.

Module: `core/improvement/metrics.py` (the vocabulary) and
`core/improvement/monitor.py` (the collector).

## The metric catalogue

Every metric the phase names is in `METRIC_CATALOGUE`. A `MetricSpec` carries its
`component`, its `kind` (`ratio`, `counter`, `duration`, `gauge`), its `unit`,
whether `higher_is_better`, a `description`, and a `tolerance`.

| metric | component | kind | better | meaning |
|---|---|---|---|---|
| `task.success_rate` | execution | ratio | higher | Share of tasks that completed successfully. |
| `task.failure_rate` | execution | ratio | lower | Share of tasks that failed. |
| `task.repeated_failure` | execution | counter | lower | How often the same task failed again after a retry. |
| `reasoning.accuracy` | reasoning | ratio | higher | Share of reasoning conclusions that verified as supported. |
| `reasoning.unsupported_claim` | reasoning | counter | lower | Claims the groundedness gate refused to publish. |
| `reasoning.inference_latency` | reasoning | duration | lower | Wall time one inference or reasoning call took. |
| `retrieval.quality` | retrieval | ratio | higher | Share of queries whose top passage was relevant. |
| `retrieval.latency` | retrieval | duration | lower | Wall time one retrieval call took. |
| `memory.retrieval_latency` | memory | duration | lower | Wall time one memory retrieval took. |
| `knowledge.integration_error` | knowledge | counter | lower | Claims that failed to integrate into the graph. |
| `knowledge.duplicate_rate` | knowledge | ratio | lower | Share of incoming knowledge found to be a duplicate. |
| `knowledge.contradiction` | knowledge | counter | lower | Stored contradictions observed. |
| `planning.efficiency` | planning | ratio | higher | Share of planned tasks that were needed (no wasted steps). |
| `planning.mistake` | planning | counter | lower | Plans that failed validation or needed replanning. |
| `tool.execution_success` | execution | ratio | higher | Share of permitted tool calls that succeeded. |
| `tool.unnecessary_use` | execution | counter | lower | Tool calls that were not required for the goal. |
| `resource.memory_usage` | resource | gauge | lower | Process/system memory in use. |
| `resource.cpu_utilization` | resource | gauge | lower | CPU utilisation. |
| `resource.disk_free` | resource | gauge | higher | Free disk space. |
| `resource.consumption` | resource | gauge | lower | Overall share of the machine the run used. |
| `model.load` | model | counter | lower | How many times a model was loaded. |
| `model.unload` | model | counter | lower | How many times a model was unloaded. |
| `user.correction` | interface | counter | lower | Answers the user corrected. |
| `user.feedback` | interface | counter | lower | Feedback signals received. |
| `assistant.latency` | assistant | duration | lower | Wall time one answer took. |
| `system.error` | system | counter | lower | Recorded best-effort failures (a swallowed error made visible). |

## Measured vs. estimated

`MetricSample.provenance` is a `Provenance` value: `MEASURED` (a real reading),
`ESTIMATED` (an inference the monitor did not itself observe), or `REPORTED` (a
value a component handed in). `MetricSummary` counts each, so a reader can always
tell a measurement from an estimate — the phase's rule that the two must not be
conflated is structural, not a convention.

## How a summary is computed

- A **ratio** summarises as a `mean` (its value is a share).
- A **counter** summarises as a `total` (its value accumulates).
- A **duration** and a **gauge** report `mean`, `minimum`, `maximum`, and a
  nearest-rank `percentile`.

`MetricSummary.describe()` names the figure and its provenance counts.

## Bounded, configurable, local

`MonitorConfig` makes monitoring configurable:

- `enabled` — when `False`, nothing is collected (`samples()` returns `()`).
- `max_samples` — a bounded ring; the oldest sample is dropped, so a long run
  cannot grow without limit.
- `sample_resources` — whether the machine's own figures are read.

An **unknown** metric name is still recorded, and reported by
`unknown_metrics()`, rather than silently dropped — so a typo is visible instead
of losing a measurement.

## The event bus

`PerformanceMonitor.attach(bus, events=(...))` subscribes to the project's event
bus, turning each subscribed event into a counter named `event.<name>`. The
mapping is deliberately tiny: a caller that wants a specific event mapped to a
specific catalogue metric records it itself. The bus's own prefix rule is a
segment match, so `"cycle"` matches `cycle.started` but not `cyclone`.

`subscribe_controller(controller, monitor)` is the convenience that attaches a
monitor to a `CognitiveController`'s bus.

## Resource readings

`PerformanceMonitor.sample_resources()` reads the machine's own figures —
memory, CPU, disk — with each reading's provenance. A figure the platform cannot
report is an honest `None` rather than a zero, so a missing reading is never
mistaken for a good one. The readings are local and offline; nothing is sent
anywhere.

## The honesty rule

A high task completion rate is **not** correctness. The monitor records
`task.success_rate` and `reasoning.accuracy` as *separate* metrics, and
`reasoning.unsupported_claim` counts the answers the groundedness gate refused,
so a run that completes many tasks but publishes unsupported claims is visible as
exactly that. The health analyser never reports "healthy" from a completion rate
alone.

## The CLI

```
python -m cli improve metrics                 # the catalogue
python -m cli improve monitor --record retrieval.quality=0.4
python -m cli improve monitor --record memory.retrieval_latency=0.2 --estimated
python -m cli improve health
```

`--estimated` marks a sample as an estimate rather than a measurement.
