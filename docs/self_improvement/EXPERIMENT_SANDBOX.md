# Experiment Sandbox (Phase 9, section 8)

Every candidate improvement is tested in an **isolated environment** before it can
be offered for approval. The sandbox reuses the project's own git worktrees
(`core/code_studio.py`) rather than inventing a second isolation mechanism.

Module: `core/improvement/sandbox.py` (the sandbox) and
`core/improvement/experiments.py` (the experiment manager).

## What the sandbox guarantees

| requirement | how it is met |
|---|---|
| isolate candidate code from production | one `git worktree` on its own `si-<slug>` branch, cut from the base branch |
| use cloned or temporary databases | the run gets a throwaway `FAYD_DATA_DIR`, so a learned-data write cannot reach real state |
| prevent unintended production writes | every path goes through a confinement check; the checkout itself is never edited |
| restrict filesystem access | a path escaping the worktree, or a forbidden file, raises `SandboxError` |
| restrict network access | the sandbox makes no network call; the check command runs offline |
| enforce resource limits | `SandboxLimits` bounds the timeout, the file count, and the check's wall time |
| support execution timeouts | `SandboxSession.check()` raises `SandboxError` once the budget is spent |
| capture logs and outputs | the check's output and the diff are recorded on the session |
| support cancellation | the session is checked before every step, so a cancelled or over-budget run stops |
| preserve experiment artifacts | the diff and the recorded outputs are kept in the history |
| clean up temporary resources safely | `cleanup()` removes the worktree and the throwaway data root |

## The confinement rules

`FORBIDDEN_PATHS` names the files a candidate may never touch — `.git`,
`pyproject.toml`, `requirements.txt`, `.github` — because changing them could
alter how the program is built or run. `SandboxManager.write` resolves every
relative path and refuses:

- an absolute path,
- a path that climbs out of the worktree (`../escape.py`),
- a forbidden file.

The gate pins the last two with the negative controls `sandbox-escape-refused`
and `sandbox-forbidden-refused`.

## Resource limits

`SandboxLimits` bounds a run:

- `timeout_seconds` (default 900) — the wall-clock budget for the check;
- `max_files` (default 4000) — a cap on how many files the sandbox will list.

A limit is validated on construction: a negative timeout raises `ValueError`. The
gate pins `sandbox-timeout` by creating a session with a near-zero budget and
asserting the next `check()` raises.

## The throwaway data root

`SandboxSession.environment()` returns the process environment with
`FAYD_DATA_DIR` pointed at the sandbox's own root. `CodeStudio.run_checks` accepts
that environment, so the project's own checks run **inside** the sandbox with a
throwaway data folder — a check that writes learned data cannot touch the real
store. The gate pins this with `sandbox-data-throwaway`.

## An experiment

`ExperimentManager.run(proposal_id=, problem=, change=, runner=, check_command=)`
runs one candidate change:

1. create the sandbox;
2. write the candidate change (`CandidateChange`, a mapping of path → content);
3. run the project's own checks inside the sandbox;
4. run the candidate's `runner` (a seam — a test injects its own, so no model and
   no real workload is needed) and collect its metrics;
5. evaluate the candidate against the baseline;
6. record the experiment (its diff, its check output, its evaluation) in the
   append-only history;
7. clean the sandbox up.

`Experiment.passed` is true only when the checks passed **and** the evaluation
found a measured improvement. A failed or regressing experiment is recorded, not
erased.

## The CLI

```
python -m cli improve experiment PRP-X --approve \
    --file core/x.py='y = 2' --overall 0.95 --check-command "python -c pass"
```

An experiment requires `--approve`; without it the command refuses with exit code
`4`, because running a candidate change is itself an action the user must confirm.
