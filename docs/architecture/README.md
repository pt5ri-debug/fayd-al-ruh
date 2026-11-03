# Fayd Al-Ruh — Architecture Documentation Index (Phase 0)

**Version:** 2.3.0 · **Commit:** `9c0b8f5` · **Phase 0 completed:** 2026-11-02

This folder is the **Phase 0 engineering audit** of the Fayd Al-Ruh repository,
produced before any implementation work on the target Cognitive Intelligence
System. Every figure in these documents was **measured** on the checkout (source
inspection *and* execution), never copied from the README.

**Phase 0 made no change to source code, model weights, or personality learning
datasets.** The only additions are these documents.

---

## Start here

- **`PHASE_0_ENGINEERING_REPORT.md`** — the final engineering report: project
  status, capabilities, architecture, missing capabilities, weaknesses,
  hardware/offline/security assessments, the test baseline, the prioritized
  roadmap, and the exact first task for Phase 1.

## The nine required deliverables

| # | Document | Contents |
| ---: | --- | --- |
| 1 | `CURRENT_ARCHITECTURE.md` | What exists today, subsystem by subsystem, with measured figures |
| 2 | `TARGET_ARCHITECTURE.md` | The target design, mapped onto existing modules (extend, never replace) |
| 3 | `COMPONENT_INVENTORY.md` | Every module and its key API; the duplication analysis |
| 4 | `DEPENDENCY_MAP.md` | Every third-party and system dependency, classified |
| 5 | `TECHNICAL_GAPS.md` | 18 gaps (G-1…G-18) with evidence, priority, and fix |
| 6 | `IMPLEMENTATION_ROADMAP.md` | Nine phases with effort, risks, tests, acceptance criteria |
| 7 | `TEST_BASELINE.md` | The reproducible baseline and coverage analysis |
| 8 | `RESOURCE_ASSESSMENT.md` | Measured environment and target-machine guidance |
| 9 | `SECURITY_ASSESSMENT.md` | The security posture and eight findings |

---

## Baseline (all four gates green)

```
python -m ruff check .     All checks passed!
python -m mypy             Success: no issues found in 872 source files
python -m pytest -q        5824 passed, 15 skipped in 1812.23s   (under xvfb-run)
python -m evaluation       overall 0.987721, 55 metrics
```

## The headline findings

- The project is ≈ **80%** of the target already; **do not rebuild**.
- **Strongest:** the grounded answer path (retrieval + a fail-closed
  groundedness gate + a visible reasoning trace), the reasoning layer, Arabic
  support, and governance (deny-by-default permissions, human approval, audit,
  structural learning isolation).
- **Four P0 gaps:** no LoRA/PEFT controlled weight training; no event bus; memory
  not fully durable; offline operation not enforced/tested — plus the automated
  (human-gated) self-improvement loop.
- **The generative path is the weak link:** the in-project LLM (842,624 params,
  byte tokenizer) produces incoherent English (measured).
- **Hardware:** the target machine's GPUs (Radeon R7 M360, Intel HD 520) cannot
  be used; plan for CPU inference, one large model at a time on 16 GB.

## Delivered phases (post-Phase 0)

Each delivered phase has its own implementation report beside these documents:

- `PHASE_1_IMPLEMENTATION_REPORT.md` — cognitive core enhancement (events,
  resource governor, validation service, uncertainty).
- `PHASE_2_IMPLEMENTATION_REPORT.md` — long-term memory architecture.
- `PHASE_3_IMPLEMENTATION_REPORT.md` — cognitive and experience-based learning.
- `PHASE_6_IMPLEMENTATION_REPORT.md` — **controlled model-weight training**
  (`ml/training/`, `cli/train.py`; metric `training_accuracy` = 1.0000).

## First task for Phase 1

> Implement `core/event_bus.py` and publish the controller's lifecycle events
> through it. See `IMPLEMENTATION_ROADMAP.md` § "Recommended first task".
