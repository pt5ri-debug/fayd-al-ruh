# Personality Isolation Verification

**Method:** an independent probe (`/tmp/fayd-audit/isolation_probe.py`, **16/16
checks**) over the **real** `core.learning_isolation` policy, plus the project's
own held-out gates `character_isolation_accuracy` (**1.0000**, 19 judgements) and
`character_identity_accuracy` (**1.0000**).

---

## 1. The seven personalities

`CHARACTER_OWNERS = (jarvis, ultron, vision, banner, natasha, hakeem, cloud)`;
the project's own model is `MAIN_MODEL_OWNER = fayd_al_ruh`.

- **Private owner:** `natasha` (`PRIVATE_OWNERS = {natasha}`).
- **Shareable owners:** the other six (`SHAREABLE_OWNERS`).
- Each owner's records live in its **own** directory, resolved from the policy
  (`data/learning/fayd_al_ruh/main` for the model; `.../characters/<key>` for a
  character) **before** any read or write.

## 2. Required rules — verified

| # | Rule | Result |
| --- | --- | --- |
| 1 | Each personality has its own isolated private memory/learning data | **PASS** — one directory per owner, resolved structurally |
| 2 | One personality cannot read/modify another's private data without authorization | **PASS** — `assert_read` raises `LearningAccessError`; the guard runs before any path |
| 3 | Shared general semantic knowledge follows the sharing rules | **PASS** — `decide`/`assert_read` honour `PRIVATE_OWNERS` vs `SHAREABLE_OWNERS` |
| 4 | Private experiences and training datasets are not accidentally mixed | **PASS** — `learning_scope` keeps scopes separate; `aggregated_dataset_is_clean()` |
| 5 | The primary model's aggregate may include eligible personalities | **PASS** — `training_keys()` = the six shareable characters |
| 6 | **Natasha's private learning data must never enter the primary model's aggregate** | **PASS** — excluded at three levels (ceiling, scope, corpus) |
| 7 | Agent collaboration does not bypass isolation | **PASS** — a character may never read another's memory even via the agent runtime |

## 3. The Natasha guarantee (rule 6) — three independent layers

1. **Capability ceiling** — her ceiling omits `CONTRIBUTE_TRAINING`.
2. **Learning scope** — her scope is non-contributing.
3. **Corpus** — `core.character_learning.training_keys()` **excludes** her, and
   `character_learning_units` defaults to it; `aggregated_dataset_owners()` never
   contains her.

Verified directly: `training_keys()` → `('jarvis','ultron','vision','banner',
'hakeem','cloud')` (no `natasha`). `learn_from_all()` iterates
`SHAREABLE_OWNERS` only, so "share all" can never reach her.

## 4. Sharing fails closed

- `KnowledgeSharing.learn_from(character)` refuses a private character **before
  it resolves any path** (Natasha's records are never opened) and logs the refusal.
- The **knowledge is shared but the private exchange is not**: `question`/`answer`
  are copied only when the source's own configuration sets
  `share_conversation_history`, and an unreadable configuration exposes nothing
  (fails closed).
- A hand-edited configuration cannot enable Natasha's sharing:
  `LearningSettings.effective_sharing` is `flag AND not is_private`.

## 5. Identity is one, across two vocabularies

`core.character_identity.identity_for(key)` resolves **either** vocabulary
(`sources`/`jarvis`, `model`/`fayd_al_ruh`) to one identity, so a fact taught to
"Jarvis" in the Characters window is the same fact the live Jarvis mouth reads.
The identity gate proves the two windows share one memory and one name.

## 6. Conclusion

All seven personalities respect the isolation rules. **Natasha remains excluded
from the primary model's aggregated training data.** No cross-personality read or
write succeeded without authorization in any probe or gate.
