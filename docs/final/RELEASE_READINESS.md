# Release Readiness

**Status: PARTIALLY READY.**

---

## 1. Acceptance criteria — scored

| # | Criterion | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Major subsystems demonstrably integrated | **Met** | Probes + `evaluation` overall 0.9903 |
| 2 | Critical end-to-end workflows pass | **Met** | `END_TO_END_TEST_RESULTS.md` (A–F) |
| 3 | Knowledge provenance and evidence preserved | **Met** | `core.evidence`/`core.provenance`; perception bridge |
| 4 | Memory persists correctly | **Met** | restart + backup/restore verified |
| 5 | Reasoning handles uncertainty and contradictions | **Met** | `answer_confidence`; `relation_consistency` |
| 6 | Planning and execution respect permissions | **Met** | execution gate + approval; probe |
| 7 | Learning is versioned and recoverable | **Met** | `learning_store` version + append-only rollback |
| 8 | All seven personalities respect isolation | **Met** | 16/16 probe; gate 1.0000 |
| 9 | Natasha excluded from aggregated training data | **Met** | `training_keys()` excludes her; 3 layers |
| 10 | Offline operation verified for every claimed-offline capability | **Partial** | Core paths verified with 0 attempts; optional engines absent → not tested |
| 11 | Backup and restoration work | **Met** | bad backup refused, good preserved |
| 12 | Security tests reveal no unresolved critical vulnerability | **Met** | 17/17 security; no critical/high |
| 13 | Resource usage documented and acceptable | **Met (this machine)** | ~60–69 MB, ms-scale warm answers; target unmeasured |
| 14 | Regression tests pass | **Met** | suite run + `evaluation` PASS |
| 15 | Remaining limitations explicitly documented | **Met** | `KNOWN_LIMITATIONS.md` |

**Met: 13 · Partial: 1 · Not met: 0.**

## 2. Why not fully "RELEASE READY"

- Criterion 10 is **partial**: the core paths are verified offline, but the
  optional engines (OCR, Piper, Qwen, audio.cpp, VL, diffusion) are absent on this
  machine and their offline behaviour is **not tested here**.
- The intended **target hardware** could not be measured (criterion 13 met only on
  the audit machine).
- The GUI could not be exercised under a display here.

These are **environmental**, not architectural: the code paths exist, are gated,
and are seam-tested. Re-running the probe on the target with the optional engines
installed is the remaining step to move from *partially* to *fully* ready.

## 3. Blocking issues

**None.** There is no unresolved critical or high security finding, no failing
acceptance gate, and no data-integrity defect.

## 4. Release checklist

- [x] All held-out gates green (`evaluation` PASS, 30/30)
- [x] `ruff` clean
- [x] `mypy` clean
- [x] Offline enforcement + test (G-8 closed)
- [x] Store-recovery test (F-1/G-16 closed)
- [x] Security audit (no critical/high)
- [x] Personality isolation (7/7, Natasha excluded)
- [x] Backup/restore verified
- [x] Full suite run under a display (`xvfb-run`): 6692 passed, 16 skipped
- [ ] Target-hardware performance recorded *(pending the Windows target)*
- [ ] Optional engines verified offline *(pending their installation)*
- [ ] Human visual/interactive GUI verification on a real desktop *(pending)*

## 5. Recommendation

Ship as a **local cognitive software system** with the documented limitations.
Before a "full" release, complete the three unchecked items on the real target.
Do **not** describe the system as AGI, conscious, or human-like — it is a local
cognitive software system with only the capabilities demonstrated by
implementation and testing.
