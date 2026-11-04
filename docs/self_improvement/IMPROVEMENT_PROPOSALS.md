# Improvement Proposals (Phase 9, sections 6 and 7)

A proposal is a **structured, reviewable change request**. It is data: it lives
apart from production configuration until a human approves it, and it has no path
to any operation the system must not perform on itself.

Module: `core/improvement/proposals.py` (generation) and
`core/improvement/prioritization.py` (ranking).

## What a proposal carries

`ImprovementProposal` has every field the phase lists:

| field | meaning |
|---|---|
| `proposal_id` | a stable id (`PRP-…`) |
| `title` | a one-line name |
| `category` | a `ProposalCategory` value |
| `target_component` | the component it would change |
| `problem` | the identified problem |
| `evidence` | the supporting evidence |
| `proposed_change` | what would change |
| `expected_benefit` | the benefit it expects |
| `risks` | the potential risks |
| `required_resources` | the resources it needs |
| `dependencies` | what must be in place first |
| `expected_side_effects` | what else it might touch |
| `testing_strategy` | how it would be tested |
| `rollback_strategy` | how it would be reversed |
| `approval_requirements` | what approval it needs |
| `status` | a `ProposalStatus` value |
| `source_weakness_id` | the finding it answers |
| `touches` | the sensitive categories it touches |
| `complexity` | 1–5 |

`requires_explicit_approval` is true when `touches` intersects
`SENSITIVE_TOUCHES` — `source_code`, `model_weights`, `training_dataset`,
`security`, `permissions`, `database_schema`, `private_personality_data`,
`irreversible`. `is_code_change` is true for a source-code touch or a code-level
category.

## The categories

`knowledge_schema`, `retrieval_optimization`, `memory_ranking`,
`reasoning_rule`, `procedure`, `planning`, `tool_integration`,
`resource_management`, `bug_fix`, `documentation`, `model_or_adapter`,
`configuration`.

## The lifecycle

`ProposalStatus` names the ten-step lifecycle the phase requires:

```
proposed → awaiting_evaluation → evaluated → awaiting_approval → approved
        → deployment_scheduled → deployed → monitoring → accepted / rolled_back
        → archived
```

plus the off-ramps `rejected` and `inconclusive`.

## Generation

`generate_proposals(weaknesses)` maps each finding to a proposal through a
per-kind planner. **A kind with no planner yields no proposal** — nothing is
invented to fill a gap. `proposal_from_weakness(weakness)` is the single-finding
form. Every generated proposal names the finding it answers in
`source_weakness_id`, so an improvement is always traceable to the problem it
addresses.

## Prioritization

`rank_proposals(proposals, *, severities=, overrides=)` orders proposals by a
transparent score. `score_proposal(proposal, ...)` computes it from nine **named
factors**, each with a documented weight:

| factor | weight | reads |
|---|---|---|
| `severity` | 0.20 | the finding's severity |
| `frequency` | 0.10 | how often it was seen |
| `evidence` | 0.10 | how strong the evidence is |
| `benefit` | 0.15 | the expected benefit |
| `complexity` | 0.10 | implementation complexity (lower is better) |
| `regression_risk` | 0.10 | the risk of a regression (lower is better) |
| `resource_cost` | 0.05 | the resource cost (lower is better) |
| `reversibility` | 0.10 | how reversible it is |
| `user_impact` | 0.10 | the user impact |

`PriorityScore.factors` exposes each factor's contribution, so a ranking is
**explained**, not asserted. `rank_proposals(..., overrides={"PRP-X": 0.99})` lets
the user pin a proposal's priority; a pinned proposal sorts first and
`PriorityScore.pinned` is true.

**The loop never prioritizes its own work over the user's.** A proposal competes
only with other proposals; it never pre-empts a user-requested task, and the user
can override any ordering.

## The CLI

```
python -m cli improve propose
python -m cli improve propose --failure execution:read_file --failure execution:read_file --failure execution:read_file
python -m cli improve propose --json
```
