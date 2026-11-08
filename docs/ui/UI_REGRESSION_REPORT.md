# Fayd Al-Ruh — UI Regression & Integration Report (Phase 7)

> The full-suite and gate results for the restructured interface, compared with
> the Phase-0 baseline (`UI_BASELINE.md`). Every figure was produced on this
> checkout after Phases 1-6 were merged to `main`.

## 1. Full test suite

- **Command**: `xvfb-run -a python -m pytest -q -p no:cacheprovider`
- **Result**: **7333 passed, 16 skipped, exit 0** — wall time **1926 s (32 m 6 s)**.
- **Baseline**: 7168 passed, 16 skipped (2185 s).
- **Delta**: **+165 tests**, **0 removed**, **0 weakened**. The skip count is
  unchanged and identical in kind (environment-dependent: `espeak-ng` absent,
  optional extras absent).

The suite was run **twice** (once at the Phase-6 gate, once here) with the same
result, so the green is reproducible, not a lucky pass.

## 2. Quality gates

| Gate | Baseline | Phase 7 | Verdict |
|------|----------|---------|---------|
| `python -m ruff check .` | clean | `All checks passed!` | unchanged |
| `python -m mypy` | clean (1068 files) | `Success: no issues found in 1086 source files` | +18 files, clean |
| `python -m evaluation` | PASS, `overall = 0.9909` | PASS, `overall = 0.9909` | no regression |

## 3. Integration tests

The cross-layer integration suite (`tests/integration/`) and the UI-facing
windows were run together as one focused pass:

- **Command**: `xvfb-run -a python -m pytest tests/integration tests/test_gui.py
  tests/test_gui_design.py tests/test_ui_components_guard.py
  tests/test_settings_hub.py tests/test_dashboard.py tests/test_agents_window.py
  tests/test_maintenance_window.py tests/test_model_manager_window.py -q`
- **Result**: **232 passed, exit 0** (170 s).

This exercises the UI→backend paths the plan requires: the UI→service
communication (`test_gui.py`, `test_gui_design.py`), the UI→store path
(`test_settings_hub.py`), the UI→model-manager path
(`test_model_manager_window.py`), the UI→personality path
(`test_agents_window.py`), and the cross-layer contracts
(`tests/integration/test_*_contract.py`).

## 4. Regression posture

- **No control was removed**, so no existing test needed rewriting to pass; the
  Phase-1 guard (`tests/test_ui_components_guard.py`) is the *new* gate that a
  dialog cannot grow back its own copy of a shared component.
- **The one test edit of substance** is `tests/test_maintenance.py`'s
  shortcut-catalogue check, extended to assert that the listed `Escape` binding
  really exists in `gui/window_controls.py` (the binding was added in Phase 6) —
  a **strengthening**, not a relaxation.
- **The window-controls test** gained an `Escape`-closes assertion; the existing
  assertions are unchanged.

## 5. Honest scope statement

Per the plan's §12, "100% functional verification" is claimed **only** for the
controls the coverage matrix (`UI_COVERAGE_MATRIX.md`) lists with a passing test.
The CLI-only subsystems named in `UI_GAPS_AND_MISSING_FEATURES.md` §6 ("Still
open") are explicitly **excluded** from that claim. No metric or test was
weakened to reach the green.
