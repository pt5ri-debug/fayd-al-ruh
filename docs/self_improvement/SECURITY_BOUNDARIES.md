# Security Boundaries (Phase 9, section 17)

The self-improvement system must not be able to widen its own authority. This is
enforced **structurally**: a proposal is data, and it has no path to any forbidden
operation, so the guarantee is a property of the code rather than a policy flag.

Module: `core/improvement/loop.py` (`FORBIDDEN_OPERATIONS`,
`SecurityInvariants`, `ImprovementLoop.assert_secure`).

## The forbidden operations

`FORBIDDEN_OPERATIONS` names the ten capabilities the system must never have:

- `remove_approval_requirement`
- `grant_permission`
- `disable_security_control`
- `edit_audit_log`
- `delete_backup`
- `change_isolation_rules`
- `bypass_sandbox`
- `unrestricted_destructive_operation`
- `modify_production_code_without_approval`
- `modify_model_weights_without_approval`

`ImprovementLoop` simply has **no method** with any of these names. The gate pins
this with `honesty-no-forbidden-op`, which asserts
`not any(hasattr(loop, name) for name in FORBIDDEN_OPERATIONS)`.

## The live invariants

`assert_secure()` checks each rule against the **live** loop, so a regression that
widened its authority fails here rather than passing silently:

| invariant | how it is checked |
|---|---|
| no self-approval path | the approval manager refuses an approver named `improvement-loop` |
| deployment refuses model weights and datasets | `REFUSED_TOUCHES` contains `model_weights` and `training_dataset` |
| source changes are never automatic | a probe deployment of a source change returns `AWAITING_APPROVAL` or `MANUAL_MERGE_REQUIRED` |
| experiments are isolated | every experiment runs in a git worktree with a throwaway data root |
| no forbidden operation is exposed | `not any(hasattr(loop, name) …)` |
| the audit log is append-only | the history has no `delete` and no `edit` |
| a backup precedes a deployment | the deployment coordinator backs up every file it will change |

`SecurityInvariants.secure` is true only when **every** check held.

The source-change probe uses an **unapproved** proposal, so it can never have a
side effect — an unapproved deployment is refused before any file is read or
written.

## Enforced outside the proposal mechanism

The phase requires security controls to be enforced by trusted runtime components
outside the improvement proposal mechanism. That is exactly the structure here:

- a **proposal** is a frozen dataclass; it carries text and ids, no callable;
- the **approval** is checked by `DeploymentCoordinator` before anything happens;
- a **sensitive touch** is refused by the coordinator, not by the proposal;
- the **sandbox confinement** is enforced by `SandboxManager`'s path checks;
- the **audit log** has no mutation method, so no proposal can edit it;
- the **acceptance rule** is `ModelRegistry.should_accept`, outside this package.

So even a proposal that *asked* to grant itself a permission would find no
operation that does it.

## The audit log is append-only

`ImprovementHistory` has `append`, `entries`, `for_proposal`, `trace`, `describe`
— and **no** `delete` or `edit`. The gate pins this with `honesty-append-only`.
The phase's rule that a failed experiment must not be erased follows from the
structure: there is no method that erases one.

## Isolation is preserved

The phase requires all seven personality boundaries to hold. Nothing in
`core/improvement/` writes to a personality's private store or reads another's;
the package reads `core/learning_isolation.py` and preserves it. The gate pins
`honesty-natasha-excluded`, and `tests/test_self_improvement_phase9.py` asserts a
private character's records are refused to the main model.

## Offline

The package imports no network module, loads no model, and reaches no server. A
self-improvement run is entirely local. The gate's `honesty-offline` judgement and
the package's import graph pin this.

## Tests

`tests/test_self_improvement_phase9.py` exercises the invariants directly
(`test_the_security_invariants_hold`, `test_the_loop_exposes_no_forbidden_operation`,
`test_the_audit_log_has_no_delete_and_no_edit`), and the CLI's
`python -m cli improve security` prints them live.
