# Fayd Al-Ruh — Security Assessment (Phase 0)

**Version:** 2.3.0 · **Audit date:** 2026-11-02

Scope: data isolation, file access, database integrity, backup/recovery, model
file validation, training-data provenance, execution sandboxing, permissions,
audit logging, and protection against accidental deletion/corruption. Each
finding is grounded in the source.

---

## 1. Executive summary

The project's governance layer is **strong and deliberately deny-by-default**.
Permissions, human approval, an append-only audit log, a three-gate runtime, a
root-confined tool sandbox, and structural learning isolation all exist and are
tested. The main weaknesses are **operational**, not architectural: offline
operation is not *enforced* (only off by default), there is no backup/restore or
store-recovery test, and there is no unified migration framework.

| Area | Rating |
| --- | --- |
| Permissions / capability gate | ✅ strong |
| Human approval | ✅ strong |
| Audit logging | ✅ strong |
| Execution sandbox | ✅ strong |
| Learning isolation | ✅ strong (structural + gated) |
| Model download gating | ✅ strong |
| Data isolation on disk | ✅ good |
| Database integrity | 🟡 good, untested recovery |
| Backup / restore | 🟡 partial (snapshots only) |
| Model file validation | 🟡 partial (GGUF magic check only) |
| Training-data provenance | ✅ good |
| Offline enforcement | ❌ not enforced/tested |
| Migration framework | 🟡 partial |

---

## 2. Permissions and capability gate

**Implementation:** `core/permissions.py`.

- `PermissionPolicy` is an **allow-list**; the default set contains only
  `SYMBOLIC_SIMULATION` and `SANDBOX_TASK`.
- `deny_external_by_default = True`; `Capability.EXTERNAL_TOOL` is refused unless
  the policy explicitly opts in **and** the target is in `allowed_external_tools`.
- `allows_external` is true only when the external capability was granted.

**Assessment:** ✅ correct and safe by default. No finding.

---

## 3. Human approval gate

**Implementation:** `core/human_approval.py`.

- An `ApprovalRequest` is `PENDING` until a human decides; it carries
  `decided_by`, `decided_at`, and an optional `expires_at`.
- The runtime proceeds only when a matching, non-expired approval exists.

**Assessment:** ✅ a human decision, not an agent one. No finding.

---

## 4. Three-gate runtime and audit log

**Implementation:** `core/sandbox_runtime.GovernedRuntime` (capability →
approval → audit) and `core/audit_log.AuditLog`.

- The audit log is **append-only** (`append`/`record_decision`/`records`; no edit
  or delete API), so it is a tamper-evident trail.
- Every governed decision — allowed or denied — is recorded.

**Assessment:** ✅ correct. **Observation:** the audit log's `records()` returns
in-memory records; persistence of the audit trail across restarts depends on a
caller saving it. **Recommendation (P2):** persist the audit log through
`JsonStore`/SQLite by default so the trail survives a restart. (The learning
audit, `core/learning_audit.py`, is persisted under `data/learning/.../audit`.)

---

## 5. Tool sandbox

**Implementation:** `core/tools.py` (`ToolSandbox`) and `core/tool_agent.py`.

Enforced: (1) allow-list of tools; (2) **root confinement** — every path is
resolved under one root and an escape (`..`, an absolute path elsewhere, a
symlink out) is refused; (3) read/write separation — a read-only policy refuses
any mutating tool; (4) capability + approval gates. Tools never raise into the
caller; a failure is a `ToolResult`.

**Assessment:** ✅ strong. The Code Studio adds its own confinement
(`core/code_studio.py`: `../escape.py`, absolute paths, and path climbs all
raise; a text-extension allow-list and a size cap).

---

## 6. Learning isolation (characters)

**Implementation:** `core/learning_isolation.py`, `core/knowledge_sharing.py`,
`core/learning_pipeline.py`, `core/character_learning_store.py`.

- Isolation is **structural**: `store_dir(owner)` resolves the directory from the
  policy, and every read/write path calls `decide`/`assert_read`/`assert_owner`
  **before** resolving a path. A caller cannot point a store at another owner.
- `PRIVATE_OWNERS = {natasha}` (a deny-list of one, so a new character is
  shareable by default — the safe direction).
- `KnowledgeSharing.learn_from(character)` refuses Natasha **before it resolves
  any path** and logs the refusal.
- The **knowledge is shared but the private exchange is not**: a source's
  `question`/`answer` are copied only when the source's own configuration sets
  `share_conversation_history`, and the check **fails closed** (an unreadable
  configuration exposes nothing).
- `effective_sharing` is `flag and not is_private`, so a hand-edited config can
  never enable Natasha's sharing.
- The held-out gate `character_isolation_accuracy` runs real scenarios **with
  negative controls** (a cross-character read, a read of Natasha's records, a
  "share all" that must not reach Natasha, a hand-edited config that must not
  enable her sharing).

**Assessment:** ✅ excellent — the strongest part of the security posture.

---

## 7. Data isolation on disk

- A source checkout writes user data to a git-ignored `data/` beside the code;
  `FAYD_DATA_DIR` overrides; a frozen build writes next to the executable
  (`core/paths.py`).
- Per-owner directories: `data/learning/fayd_al_ruh/main` for the model,
  `.../characters/<key>` for a character; per-mouth models under
  `data/{ultron,vision,presentation,natasha,hakeem,shared_models,...}`.
- Secrets: the Cloud mouth reads its API key from the **environment at call
  time** and never writes it to the repository, a config file, or a log;
  `redact_secret` scrubs it from every message (`core/cloud_model.py`). The SQL
  import redacts passwords/tokens and credential-bearing URIs
  (`gui/sql_import.py`).

**Assessment:** ✅ good. **Observation:** `data/` is git-ignored, which protects
user data from being committed — verify `.gitignore` stays correct as new stores
are added (it currently ignores `data/`, `documents/`, `knowledge_base/`).

---

## 8. Database integrity and recovery

**Stores:** `storage/json_store.JsonStore` (atomic: write `.tmp` then `replace`;
percent-encoded keys are Windows-safe), `storage/sqlite_store.SqliteStore`,
`core/knowledge_index.py` (SQLite FTS5), `core/snapshot.py` +
`snapshot_migrations.py` (schema v2, ordered side-effect-free migrations).

**Findings:**

- **F-1 (P0):** no test corrupts a store and asserts recovery. Character stores
  tolerate a corrupt file (read as empty, documented), but this is not proven by
  a test for every store.
- **F-2 (P2):** no unified migration framework — `snapshot_migrations` covers
  snapshots only.
- **F-3 (P2):** no backup/restore round-trip test for the knowledge base or the
  character stores.

**Recommendations:** add `tests/test_store_recovery.py` (corrupt each store type
→ the reader tolerates or reports, never crashes) and a backup/restore round
trip; extend `snapshot_migrations` into a small registry all stores can use.

---

## 9. Model file validation

**Implementation:** `core/model_files.inspect_model_file` reads a GGUF header
(architecture/context/parameter count) and never executes the file;
`core/audiocpp_speech.is_valid_gguf` reads the `GGUF` magic and `find_model_file`
skips a file that fails it.

**Findings:**

- **F-4 (P2):** GGUF validation is a **magic check**, not an integrity/digest
  check. `core/model_sources.py` separates a source's *claim* from a
  user-pinned digest, and `core/model_download.py` verifies a **pinned** digest
  before the rename — but a model with no pinned digest is not verified beyond
  the magic.
- **Positive:** a partial download is written to `.part` and **never promoted**
  to the real name; a torn download is not a model.

**Recommendation:** surface an optional SHA-256 for a downloaded model in the
registry (already stored where the source provides one) and offer a verify
action.

---

## 10. Training-data provenance

**Implementation:** `core/learning_records.LearningRecord` (frozen; unique id,
owner, source type + reference, question, answer, knowledge, timestamps,
confidence, verification, evidence, status, and provenance fields
`source_character`/`source_record_id`/`source_date`). `core/learning_pipeline.py`
is the one controlled path (receive → extract → identify facts → check
contradictions → confidence → store → optional bounded graph update → preserve
original + provenance → allow later correction). It **never auto-verifies** and
**preserves contradictions**.

**Assessment:** ✅ good. Provenance is explicit and a revision is a *new* record,
so an original is never mutated.

---

## 11. Protection against accidental deletion/corruption

- `JsonStore.delete` removes both the encoded and the legacy path and returns
  whether one was removed.
- `ingest/delete.py` removes a source's chunks by an **all-or-nothing** shard
  rewrite and drops the stale FTS index.
- The Code Studio refuses to merge when `main` has moved on, so a merge cannot
  overwrite newer work.
- The updater refuses to fast-forward when local changes touch a file the update
  also changes, names the conflicting files, and uses git's `--ff-only` as the
  backstop so a failed merge leaves the tree untouched.

**Findings:**

- **F-5 (P1):** `forget_facts`/`reset_learning` require confirmation (good), but
  there is no automatic backup before a destructive operation on the character
  stores or the knowledge base.

**Recommendation:** take a cheap snapshot (copy the JSONL/SQLite file) before a
destructive operation, and offer restore. This is a small, bounded change.

---

## 12. Offline operation

- Web search is **off by default** (`ai/jarvis.build_session`; Phase 53), the
  preference persists, and `web=True` is the explicit opt-in.
- Every model download requires `approved=True`.
- **Finding F-6 (P0):** nothing *enforces* offline operation. There is no test
  that a base-install answer path opens no socket, and `requests` is a base
  dependency (importable everywhere).

**Recommendation:** a `FAYD_OFFLINE=1` hard-off switch that makes any network
attempt raise, plus `tests/test_offline.py` that monkeypatches the network to
raise and asserts the answer path, a memory write, a training smoke step, and a
personality exchange all succeed.

---

## 13. Execution sandboxing (code)

- `core/code_studio.py` runs every change in a **git worktree** under
  `data/studio/workspaces`, never the running checkout; `run_checks` runs the
  project's own test command **inside the workspace**; `can_merge` is a gate.
- `core/code_agent.py` validates every action's fields and confines paths through
  the Studio's checks.
- `core/updater.py` never copies or moves the user's own data in `data/`.

**Assessment:** ✅ strong. Nothing is executed by the Studio except the declared
check command.

---

## 14. Network egress inventory (what can leave the machine)

| Path | Gate | Default |
| --- | --- | --- |
| `core/web_source.py` (search) | permission + approval + audit | off |
| `core/web_verifier.py` (verify) | same | off unless enabled |
| `core/knowledge_base.py` (bulk build) | explicit CLI command | not built |
| `core/updater.py` (update check) | explicit CLI/GUI action | not run |
| every `download_model` | `approved=True` + permission | not run |
| `core/cloud_model.py` | permission + approval, off by default | off |
| `ingest/connectors/web.py` | registered only with `allow_web=True` | off |

**Assessment:** ✅ every egress is gated and off by default. The gap is
*enforcement/testing* (F-6), not the design.

---

## 15. Findings and recommendations

| ID | Finding | Priority | Recommendation |
| --- | --- | --- | --- |
| F-6 | Offline operation not enforced/tested | P0 | `FAYD_OFFLINE=1` hard-off + `tests/test_offline.py` |
| F-1 | No store-recovery test | P0 | `tests/test_store_recovery.py` (corrupt → tolerate/report) |
| F-5 | No auto-backup before a destructive op | P1 | cheap snapshot before delete/reset; offer restore |
| F-2 | No unified migration framework | P2 | extend `snapshot_migrations` to all stores |
| F-3 | No backup/restore round-trip test | P2 | add one for the knowledge base and character stores |
| F-4 | GGUF validated by magic only | P2 | record/verify an optional SHA-256 |
| F-7 | Audit log persistence depends on a caller | P2 | persist by default |
| F-8 | `data/` ignore must stay correct as stores are added | P3 | a test asserting every store path is under `data/` |

**No finding requires weakening or removing an existing control.** Every
recommendation is additive.
