# Weakness Detection (Phase 9, section 5)

The weakness detector turns the monitor's measurements and reported failures into
**evidence-backed findings** — never guesses, and never an automatic change.

Module: `core/improvement/weakness.py`.

## What a weakness carries

`Weakness` is a frozen record with every field the phase requires:

| field | meaning |
|---|---|
| `weakness_id` | a stable id (`WK-…`) |
| `kind` | one of the `WeaknessKind` values |
| `description` | what was observed, in words |
| `evidence` | the raw observations the finding rests on |
| `component` | the affected component |
| `frequency` | how often it was seen |
| `severity` | `low` / `medium` / `high` / `critical` |
| `confidence` | how sure the diagnosis is (always `< 1.0`) |
| `impact` | the potential impact |
| `suggested_investigation` | the next step a human should take |
| `related_metrics` | the performance metrics it reads |

`Weakness.score` is a transparent priority *hint* (severity × log(frequency) ×
confidence), not a decision — the prioritization engine ranks with named criteria
and the user may override it.

## The kinds it detects

| kind | raised when |
|---|---|
| `repeated_failure` | the same operation failed `minimum_frequency` (default 2) or more times |
| `frequent_correction` | the user corrected answers repeatedly |
| `poor_retrieval` | `retrieval.quality` sits below its floor |
| `incorrect_reasoning` | `reasoning.accuracy` sits below its floor |
| `unsupported_claim` | the groundedness gate refused claims |
| `planning_mistake` | `planning.mistake` accumulates |
| `unnecessary_tool_use` | `tool.unnecessary_use` accumulates |
| `excessive_latency` | a latency metric sits above its ceiling |
| `memory_duplication` | `knowledge.duplicate_rate` is high |
| `resource_exhaustion` | a resource gauge crosses its limit |
| `regression` | a post-deployment comparison regressed |
| `knowledge_error` | `knowledge.integration_error` accumulates |

## The two honesty rules

**A single failure is not a weakness.** `detect_weaknesses(...,
minimum_frequency=2)` reports nothing below the floor, so one transient error
cannot drive a change. The gate pins this with the negative control
`detect-single-refused`.

**Correlation is not causation.** A finding states *what was seen* and *what to
investigate*; it never says a cause. `confidence` is always strictly less than
`1.0`, so no finding is ever presented as certain. The gate pins this with
`detect-not-causal`, which asserts the finding's text contains neither "caused
by" nor "because of".

## How a finding is built

`detect_weaknesses(monitor_or_summaries, *, failures=(), minimum_frequency=2)`
runs an ordered set of rules. Each rule reads the monitor's summaries and the
reported failures, and emits a `Weakness` whose `evidence` names the actual
observations (for a repeated failure, the individual failure records; for a low
metric, the metric name and its value). `describe_weaknesses(findings)` renders
the list readably.

## The CLI

```
python -m cli improve detect
python -m cli improve detect --failure execution:read_file \
                             --failure execution:read_file \
                             --failure execution:read_file
python -m cli improve detect --min-frequency 3
```

A single `--failure` prints that no weakness was found, which is the intended
behaviour.
