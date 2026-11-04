# Rollback (Phase 9, section 14)

A rollback restores the **prior accepted state** without deleting unrelated user
data or learning history. It deletes nothing, so a failed change and its
evaluation stay auditable.

Module: `core/improvement/rollback.py`.

## What a rollback does

`RollbackManager.rollback(deployment, reason=)` restores **only** the files a
deployment actually changed, from that deployment's own backup. It:

- never touches anything outside those files, so user data and learning records
  are untouched;
- records the rollback as a **new** history entry and marks the deployment rolled
  back — it deletes nothing;
- **verifies** the restore afterwards and reports whether it succeeded, so a
  rollback that itself failed is visible rather than assumed;
- reuses `cognition.versions.ModelRegistry.rollback` for the one case that
  belongs there (a trained model version), rather than re-implementing it.

## The outcomes

`RollbackStatus`:

| status | meaning |
|---|---|
| `rolled_back` | the files were restored and verified |
| `nothing_to_roll_back` | the deployment was not applied (a `MANUAL_MERGE_REQUIRED` source change, say) |
| `refused` | the deployment has no backup, so it cannot be reversed safely |
| `failed` | the restore was attempted but did not verify |

A deployment that was never applied returns `nothing_to_roll_back` with
`verified=False` — honest, because there is nothing to restore.

## What a rollback preserves

`RollbackRecord` carries the rollback id, the deployment it reversed, the
proposal, the restored files, the backup directory, the reason, and the
verification result. The **deployment record and its evaluation remain** in the
append-only history: the gate pins this with `rollback-preserves-history`, which
asserts both a `DEPLOYMENT` and a `ROLLBACK` entry survive.

## Rollback triggers

- A **post-deployment health check** that fails: `DeploymentCoordinator` restores
  the backup automatically and records `FAILED`.
- A **post-deployment regression**: a comparison against the pre-deployment state
  that regresses raises a `WeaknessKind.REGRESSION` finding, which is the signal
  to roll back.
- An **explicit user request**: `python -m cli improve rollback <deployment-id>`.

## The CLI

```
python -m cli improve rollback DEP-abc12345
```

Exit code `0` when the rollback succeeded, `4` when it refused or found nothing
to reverse.

## Tests

The gate pins three rollback judgements:

- `rollback-restores` — the pre-deployment file content is back;
- `rollback-preserves-history` — the deployment and rollback both remain in the
  history;
- `rollback-verifies` — the restore verified itself.

`tests/test_self_improvement_phase9.py` also exercises the full
deploy-then-rollback path through `ImprovementLoop`.
