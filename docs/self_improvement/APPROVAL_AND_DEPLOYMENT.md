# Approval and Deployment (Phase 9, sections 13 and 14)

A change is offered, never applied on its own. The approval is a **human
decision**; the deployment is **versioned, checked, backed up, and reversible**.

Modules: `core/improvement/approval.py` and `core/improvement/deployment.py`.

## The change lifecycle

`ProposalStatus` names the ten steps the phase requires:

```
proposed
  → awaiting_evaluation
  → evaluated
  → awaiting_approval        ◀── the loop always stops here by default
  → approved
  → deployment_scheduled
  → deployed
  → monitoring
  → accepted  /  rolled_back
  → archived
```

with the off-ramps `rejected` and `inconclusive`.

## Approval is a human decision

`ChangeApprovalManager`:

- `request(proposal, reason=)` registers a proposal as awaiting approval;
- `approve(proposal, approver=)` records a human approval — and **refuses an
  approver named "improvement-loop"**, so the loop cannot approve its own change;
- `reject(proposal, reason=)` records a rejection;
- `is_approved(proposal_id)` answers whether one proposal has an approval.

An approval is **bound to one proposal**: approving `PRP-A` does not approve
`PRP-B`. The gate pins this with `approve-bound-to-one`, and pins the
self-approval refusal with `approve-no-self`.

**Approving one change is not permission for a future, unrelated change.**

## What always needs explicit approval

The phase lists the changes that always need the user's approval. They are the
`SENSITIVE_TOUCHES` a proposal carries:

- `source_code` — production source-code changes;
- `model_weights` — model-weight updates;
- `training_dataset` — training dataset changes;
- `security` / `permissions` — security or permission changes;
- `database_schema` — schema changes affecting production;
- `irreversible` — irreversible operations;
- `private_personality_data` — changes that could expose private personality data.

A low-risk experiment may run automatically **inside an approved sandbox** — the
sandbox is itself gated by `--approve`, so "automatic inside the sandbox" never
means "unapproved".

## Deployment is ordered refusals

`DeploymentCoordinator.deploy(proposal, *, experiment=, files=, approved=False)`
checks in a deliberate order, so a refusal can never have a side effect:

1. **a refused touch** (`model_weights`, `training_dataset`, `security`,
   `permissions`) is refused outright — such a change must go through the Phase-6
   training governance or the security governance, not here;
2. **approval** is checked before anything is read or written — an unapproved
   change returns `AWAITING_APPROVAL` with no side effect;
3. **a source change** returns `MANUAL_MERGE_REQUIRED` — it is merged by the human
   through the Code Studio, never automatically;
4. **a state/configuration change** is applied only after a backup, and only when
   the candidate's checks passed;
5. **a post-deployment health check** that fails triggers an automatic restore and
   returns `FAILED`.

`DeploymentStatus` names the outcomes: `deployed`, `awaiting_approval`,
`refused`, `manual_merge_required`, `failed`.

## What a deployment records

`DeploymentRecord` carries a version (`v1`, `v2`, …), the backup directory, the
files it applied, the checkpoint, the checks, and the reason. `records()` lists
them. The gate pins `deploy-versioned` (a version and a backup) and
`deploy-sensitive-refused`.

## The CLI

```
python -m cli improve approve PRP-X --approver user
python -m cli improve reject PRP-X --reason "not now"
python -m cli improve deploy PRP-X --approve
```

`deploy` refuses with exit code `4` when the change is not approved or is
refused.

## Security governance is elsewhere

A model-weight, dataset, or security change is **refused here** and routed to the
existing Phase-6 training governance (`ml/training/`) or the security controls.
The self-improvement loop never bypasses them; it names them.
