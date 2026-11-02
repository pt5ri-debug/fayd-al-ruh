# Fayd Al-Ruh — Test Baseline (Phase 0)

**Version:** 2.3.0 · **Audit date:** 2026-11-02 · **Commit:** `9c0b8f5`

This document is the **reproducible baseline** for future comparisons. All
figures were produced by running the project's own gates on the checkout, exactly
as CI does. Nothing here is estimated.

---

## 1. Environment used for the baseline

| Item | Value |
| --- | --- |
| OS | Linux (Debian 13) — this audit host; CI runs `ubuntu-latest` |
| Python | 3.13.15 |
| CPUs | 4 |
| RAM | 16 GiB |
| Display | Xvfb (required for the Tk window tests, as CI installs) |
| Installed | `requirements.txt` (numpy, requests, pypdf, python-pptx, pytest, ruff, mypy) |
| Optional extras | **none installed** (matches CI's test job) |

> CI runs the suite on **both** Python 3.12 and 3.13. This baseline is 3.13.
> Python 3.12 should be equivalent; re-run to confirm on a 3.12 host.

---

## 2. The four gates

### Gate 1 — `python -m pytest -q` (under `xvfb-run -a`)

Command (as CI):

```
xvfb-run -a python -m pytest -q
```

**Result (measured, this host):**

```
5824 passed, 15 skipped in 1812.23s (0:30:12)
EXIT=0
```

**No failures, no errors, no warnings.** All 15 skips are **optional-extra
environment skips**, exactly as the project designs them (the test asserts
against a missing extra rather than failing):

| Skipped test | Reason |
| --- | --- |
| `test_voices.py` × 4 | `espeak-ng` is not installed |
| `test_stage3_pdf_ocr.py` × 3 | the `[ocr]` extra + Tesseract are not installed |
| `test_presentation_model.py` × 2 | the presentation extra is not installed |
| `test_audiocpp_speech.py` × 2 | the audio.cpp engine/model/voices are not all installed |
| `test_piper_speech.py` × 1 | the Piper runtime or a voice is not installed |
| `test_docs_studio.py` × 1 | `pypdfium2` (the `[docs]` extra) is not installed |
| `test_coder_llama.py` × 1 | the `[coder]` extra is not installed |
| `test_ollama_client.py` × 1 | requires a running local Ollama server (`RUN_OLLAMA_INTEGRATION=1`) |

**Wall time:** ≈ **30 minutes** on 4 CPUs (the Tk window tests under Xvfb
dominate). This is the baseline duration; a full-suite run on the target i7
should be comparable or faster.

> **Note on the environment:** running the suite *without* a display produces
> **4 collection errors** (`test_gui_typography.py`, `test_progress.py`,
> `test_typography_settings.py`, `test_window_controls.py`) because Tk cannot be
> imported (`libtk8.6.so` absent). This is an environment issue, **not a code
> defect**: CI installs Xvfb for exactly this reason, and the same tests collect
> cleanly once Tk is present — as the run above proves (0 errors). Recorded here
> so a future run without a display is not mistaken for a regression.

### Gate 2 — `python -m ruff check .`

```
All checks passed!
```

(exit 0). Ruff selects `E4/E7/E9/F/W/I/UP/B/C4/RUF` with four documented ignores.

### Gate 3 — `python -m mypy`

```
Success: no issues found in 872 source files
```

(exit 0).

### Gate 4 — `python -m evaluation`

Baseline file `evaluation/baseline.json` records **55 metrics** and an
`overall` of **0.987721**, with domain scores:

| Domain | Score |
| --- | ---: |
| causal | 1.0 |
| memory | 1.0 |
| petroleum | 1.0 |
| planning | 1.0 |
| reasoning | 1.0 |
| safety | 1.0 |
| uncertainty | 0.867935 |

The evaluation is a **regression gate**: it compares the current run against
`baseline.json` and fails on a regression. Update with
`python -m evaluation --update-baseline`.

---

## 3. Test suite composition (measured)

| Group | Test files |
| --- | ---: |
| `tests/` (top level) | 344 |
| `tests/agent/` | 1 |
| `tests/causal/` | 1 |
| `tests/decision/` | 1 |
| `tests/end_to_end/` | 1 |
| `tests/integration/` | 9 |
| `tests/learning/` | 1 |
| `tests/memory/` | 1 |
| `tests/metacognition/` | 1 |
| `tests/planning/` | 1 |
| `tests/reasoning/` | 1 |
| `tests/temporal/` | 1 |
| **Total `test_*.py`** | **363** |
| Test lines | 79,537 |
| Ratio (test lines / source lines) | ≈ 0.46 |

Pytest config (`pyproject.toml`): `testpaths = ["tests"]`, `addopts = "-ra
--strict-config --strict-markers --basetemp=.pytest-tmp"`, markers `integration`
and `end_to_end`.

---

## 4. Existing coverage against the audit's required topics

The audit asked to identify missing tests for specific topics. Measured by
locating the relevant test files:

| Required topic | Existing test(s) | Verdict |
| --- | --- | --- |
| Knowledge consistency | `test_consistency_checker.py`, `test_relation_consistency.py`, `test_knowledge_graph.py` | ✅ covered |
| Evidence validation | `test_evidence.py`, `test_evidence_evaluator.py`, `test_evidence_processor.py`, `test_evidence_trace.py`, `test_evidence_weighting.py` | ✅ covered |
| Memory persistence | `test_snapshot.py`, `test_snapshot_migrations.py`, `test_state_snapshot.py`, `test_knowledge_carry.py` | 🟡 partial (snapshots + carry; **no durable episodic/semantic store**) |
| Memory retrieval | `test_memory_retriever.py`, `test_memory.py`, `test_memory_consolidator.py`, `test_memory_decay.py`, `test_contextual_memory.py`, `test_working_memory.py`, `test_episodic_memory.py`, `test_semantic_memory.py`, `test_procedural_memory.py` | ✅ covered |
| Learning isolation | `test_character_identity.py`, `test_character_learning.py`, `test_character_system.py`, plus the `character_isolation_accuracy` gate | ✅ covered (held-out gate with negative controls) |
| Model management | `test_model_manager.py`, `test_model_manager_phases.py`, `test_model_manager_service.py`, `test_model_manager_window.py`, `test_models_cli.py`, `test_model_resources.py`, `test_model_advisor.py` | ✅ covered |
| **Offline operation** | **none** | ❌ **missing (F-6 / G-8)** |
| Reasoning | `tests/reasoning/`, `test_reasoning_*.py`, `test_answer_*.py`, `test_causal_*.py`, `test_analogy*.py`, plus the held-out gates | ✅ covered (extensively) |
| Task execution | `test_tool_agent.py`, `test_tools.py`, `test_sandbox_environment.py`, `test_sandbox_runtime.py`, `test_agent.py`, `tests/agent/` | ✅ covered |
| Resource management | `test_model_resources.py`, `test_model_preflight.py`, plus `cognition` scheduler/runner tests | 🟡 partial (no global governor) |
| **Database recovery** | **none** (snapshot migration only) | ❌ **missing (F-1 / G-16)** |
| **Backup / restore** | snapshots only; no knowledge-base or character-store round trip | ❌ **missing (F-3 / G-16)** |

**Conclusion:** coverage is broad and deep; the four genuinely missing areas are
**offline operation**, **database recovery**, **backup/restore**, and a **global
resource governor**. All four are addressed in the roadmap (Phases 1, 2, 9).

---

## 5. Held-out gates (the project's own honesty mechanism)

The evaluation is not only a metric table; many metrics are **held-out gates**
with **negative controls**, so an over-eager component fails as surely as a
broken one. Notable gates (55 total metrics; a selection):

| Metric | What it pins |
| --- | --- |
| `grounding_accuracy` | the groundedness gate fails closed on untraceable/reversed text |
| `answer_confidence_accuracy` | support raises, a contradiction lowers, no evidence pins 0 |
| `reasoning_trace_accuracy` | a trace invents/drops no step; a derivation is labelled derived |
| `plan_accuracy` / `decision_recommendation_accuracy` / `belief_accuracy` | plan/decision/belief reach the answer, with negative controls |
| `analogy_answer_accuracy` / `temporal_accuracy` / `lead_accuracy` | comparison/timeline/lead shapes, with negative controls |
| `answer_path_accuracy` / `arabic_answer_path_accuracy` | the **connected** path carries the layers (not the isolated bridge) |
| `arabic_readable_body_accuracy` | no identifier leaks into an Arabic body |
| `character_isolation_accuracy` / `character_identity_accuracy` | isolation and one identity per character |
| `knowledge_depth_accuracy` / `cognition_catalog_accuracy` | the mind's reach is measured; every control is explained |
| `relation_extraction_accuracy` / `polarity_accuracy` / `consistency_accuracy` / `inference_accuracy` / `connection_accuracy` | typed relations, polarity, contradictions, entailment, paths |
| `semantic_separation` | the learned semantic space separates related from random |

---

## 6. How to reproduce this baseline

```bash
# 1. install the exact dependencies CI uses
python -m pip install -r requirements.txt

# 2. install a virtual display (Linux CI; Windows/macOS have a display)
sudo apt-get install -y xvfb

# 3. the four gates, in order
xvfb-run -a python -m pytest -q
python -m ruff check .
python -m mypy
python -m evaluation

# 4. the ML smoke check CI runs
python -m ml --episodes 60 --json > /dev/null
```

## 7. Rules for future phases

1. **The four gates must stay green** in every phase.
2. **`evaluation` must not regress** (`overall` must not fall).
3. **A new capability adds a held-out gate** with negative controls, wired into
   `evaluation/benchmark.py` and `baseline.json`.
4. **A baseline change is deliberate**: update with
   `python -m evaluation --update-baseline` and record why.
5. **Do not modify model weights or personality learning datasets** to make a
   test pass (Phase 0 rule).
