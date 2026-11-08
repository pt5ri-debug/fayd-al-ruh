# Fayd Al-Ruh — UI Baseline (Phase 0)

> The measured "before" the restructuring is compared against. Every figure was
> produced on the untouched checkout (branch `main`, commit `34e19d6`, version
> `2.3.0`), so it is a real baseline, not an estimate.

## Baseline test run

- **Command**: `xvfb-run -a python -m pytest -q -p no:cacheprovider`
- **Result**: **7168 passed, 16 skipped, exit 0** — wall time **2185 s (36 m 25 s)**.
- **Skips**: 16, all environment-dependent (e.g. `espeak-ng is not installed`,
  optional-extra absences) — the same skips the project documents in `AGENTS.md`.
- **Interpretation**: the suite is green before any change, so a later failure is
  a regression the restructuring introduced, not a pre-existing one.

## Baseline quality gates

| Gate | Command | Baseline |
|------|---------|----------|
| Lint | `ruff check .` | clean |
| Types | `mypy` | clean (1068 files) |
| Evaluation | `python -m evaluation` | **PASS**, `overall = 0.9909` |

## Baseline UI figures (from the audit)

- 18 dialogs + the main window; 44 modules, 41 519 lines, 845 widgets.
- 59 sidebar actions (45 visible), 135 CLI menu entries.
- 17 duplicated `_button`, 13 duplicated `_build_header`, 12 duplicated worker
  pumps (see `UI_DUPLICATION_AUDIT.md`).

## How later phases are compared

Each phase re-runs the same three gates and the same test command, and records
its own numbers beside these. A phase is only merged when:

1. the test count does not drop (unless a test was demonstrably obsolete and
   replaced by a stronger one, which is stated explicitly);
2. `ruff` and `mypy` stay clean;
3. `evaluation` still PASSes with `overall` no lower than the baseline.
