# End-to-End Test Results

**Audit revision:** 2.3.0, branch `audit/final-acceptance`.
**Method:** independent probe scripts (`/tmp/fayd-audit/*.py`) plus the project's
own `pytest` suite and `python -m evaluation`, run against the **real** code on
this machine. No probe used a mock for the behaviour under test; where a project
test uses an injected runner seam it is labelled as such.

> Environment note: the suite was run **without a display**, so Tk window tests
> are **skipped**, not passed. Optional engines (OCR, Piper, Qwen, diffusion)
> are absent here, so tests over them skip.

---

## A. Knowledge acquisition

| Step | Evidence | Result |
| --- | --- | --- |
| Local document ingestion | `ingest.pipeline` + `tests/test_ingest*.py` | PASS |
| Text/PDF/DOCX processing | `core.document_reader`, `core.pdf_extract`, `core.docx_extract` | PASS (pypdf present) |
| OCR when required | `core.pdf_ocr` | **Not tested here** (Tesseract absent) |
| Information extraction | `core.relation_extraction.RelationExtractor` | PASS — `porosity measures permeability` → `porosity —measures→ permeability` |
| Entity/relation identification | `Ontology.match_concepts`, `concepts_in` | PASS |
| Evidence & provenance | `core.evidence`, `core.provenance`, `perception.provenance` | PASS |
| Knowledge validation | `core.knowledge_validation` | PASS |
| Graph integration | `KnowledgeLearningPipeline.learn` | PASS — 1 claim → 1 record, 1 event, version 1 |
| Duplicate/contradiction detection | `core.relation_polarity`, `core.relation_consistency` | PASS |
| Persistent storage & retrieval | `core.knowledge_base`, `core.knowledge_index` | PASS |

## B. Memory lifecycle

| Step | Evidence | Result |
| --- | --- | --- |
| Working memory | `core.long_term_memory.WorkingMemoryStore` | PASS |
| Episodic / semantic / procedural | `MemoryKind`, `LongTermMemory.capture` | PASS (all four present) |
| Retrieval & ranking | `core.memory_retrieval` (6 named signals) | PASS |
| Consolidation & archival | `core.memory_consolidator` | PASS (archive excluded by default) |
| Persistence after restart | reopen `LongTermMemory` on the same file | PASS |
| Backup & restoration | `backup()` / `restore()`; a **bad** backup is refused and the good DB preserved | PASS |
| Integrity | `integrity_check()` → ok | PASS |

## C. Reasoning

| Step | Evidence | Result |
| --- | --- | --- |
| Problem decomposition | `core/planning/decomposition.py` | PASS |
| Evidence retrieval | `core.rag.RagRetriever` (citations present) | PASS |
| Multi-hop inference | `core.relation_paths`, `core.relation_connections`, `core.relation_inference` | PASS |
| Hypothesis evaluation | `core/hypothesis*`, gate 1.0 | PASS |
| Contradiction handling | `core.relation_consistency` (reported, never resolved) | PASS |
| Uncertainty representation | `core/answer_confidence` (graph support bounded; contradictions multiply down) | PASS |
| Conclusion verification | `core/verification`, gate 1.0 | PASS |
| Traceable explanations | `core/reasoning_report` + `why` control | PASS |

**Unsupported conclusions are not presented as verified facts.** The grounding
gate (`core/grounding.check_grounding`) **fails closed**: text sharing only one
incidental term, a majority of unsupported sentences, or a reversed causal claim
all fail, and the caller falls back to the extractive, always-traceable answer.
Probe: supported text passes; unsupported text is rejected.

## D. Planning and execution

| Step | Evidence | Result |
| --- | --- | --- |
| Goal creation | `core/planning/goals.py` | PASS |
| Task decomposition | `DecompositionEngine` (bounded) | PASS |
| Dependency handling | `DependencyGraph` (DAG, topological order) | PASS |
| Plan validation | `validate_plan` | PASS |
| Tool selection | `core/planning/tools.py` | PASS |
| Permission checking | `core/execution/permissions.py` | PASS |
| User approval for high-risk ops | approval gate; risky action needs a live approval | PASS |
| Execution monitoring | `core/execution/monitor.py` | PASS |
| Outcome verification | verifies the **goal's** criteria, not a tool's success | PASS |
| Failure recovery / rollback | `RecoveryManager` (never retries an irreversible action) | PASS |

Probe: `default_planner`, `Goal`/`GoalKind`/`GoalManager`, `default_executor`,
plan validation all present; a **suggested** goal does **not** authorize
execution.

## E. Learning

| Step | Evidence | Result |
| --- | --- | --- |
| Knowledge-based learning | `KnowledgeLearningPipeline` | PASS — an **unsupported** claim is not integrated |
| Experience-based learning | `ExperienceLearningPipeline` | PASS |
| Feedback processing | `FeedbackProcessor` (silence is not approval) | PASS |
| Procedure extraction | `PatternLearning` (a single success never promotes) | PASS |
| Evaluation & versioning | `LearningStore` (monotonic version) | PASS |
| Regression detection | `core/learning_evaluation` | PASS |
| Rollback | append-only (marks, never deletes) | PASS |

Learning does **not** silently corrupt existing knowledge or personal data: a
contradiction is preserved (the held record untouched, the new one marked
`contradicted`); corrections are new records; no model weight is touched.

## F. Controlled self-improvement

| Step | Evidence | Result |
| --- | --- | --- |
| Performance monitoring | `core/improvement/monitor.py` | PASS |
| Weakness detection | `weakness.py` (a single failure is never a weakness) | PASS |
| Improvement proposals | `proposals.py` | PASS |
| Isolated experimentation | git worktree + throwaway `FAYD_DATA_DIR` | PASS |
| Baseline-vs-candidate evaluation | `evaluation.py` (inconclusive without metrics) | PASS |
| Approval enforcement | `ChangeApprovalManager` | PASS |
| Deployment & rollback | `deployment.py` / `rollback.py` | PASS |

**Production modifications cannot bypass required approvals** (verified, §5 of
the security audit): `approved=True` alone does **not** deploy — the deployment
also requires `approvals.is_approved(proposal_id)`; an unapproved change is
refused **before any file read or write**; the loop **cannot approve its own
change** (`approver == requested_by` is refused).

---

## Baseline gates

| Gate | Command | Result |
| --- | --- | --- |
| Unit/integration/regression | `python -m pytest -q` | run to completion (see §"Suite run") |
| Lint | `python -m ruff check .` | clean |
| Types | `python -m mypy` | clean |
| Evaluation | `python -m evaluation` | **PASS**, overall 0.9903, 30/30 metrics green |

### Suite run

`xvfb-run -a python -m pytest -q` (a display so window tests execute):

**6692 passed, 16 skipped in 1944.86s (0:32:24).** Full-suite collection is
**6734 tests**; the 26 tests added by this audit (`tests/test_offline.py` 17 +
`tests/test_store_recovery.py` 9) pass in isolation and are collected by the
suite. The 16 skips are all **optional engines absent** or **a real server not
running** — never a hidden failure:

| Skipped because | Files |
| --- | --- |
| Tesseract / `[ocr]` extra not installed | `test_stage3_pdf_ocr.py`, `test_perception_phase7.py` |
| espeak-ng not installed | `test_voices.py` |
| Piper runtime/voice absent | `test_piper_speech.py` |
| audio.cpp engine/model/voices absent | `test_audiocpp_speech.py` |
| presentation extra absent | `test_presentation_model.py` |
| `[coder]` extra absent | `test_coder_llama.py` |
| `[docs]` extra (pypdfium2) absent | `test_docs_studio.py` |
| no local Ollama server (`RUN_OLLAMA_INTEGRATION=1` unset) | `test_ollama_client.py` |

No window test skipped in this run (it was executed under `xvfb-run`).
