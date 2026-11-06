# Code Studio — Universal Developer Environment

The Code Studio is the project's **own** code host and IDE, in one window and one
CLI, with **no cloud service anywhere in the loop**. It edits the project's own
source in an isolated `git worktree` per change, tests it, and merges it only when
the user approves. Every capability the Studio offers is local: the workspace, the
diff, the merge, the coding model, and the project's own checks.

## Why it exists

A developer working on Fayd Al-Ruh should not have to leave the program to change
the program. The Studio is that surface: it branches, edits, searches, reviews git,
runs the project's own tests, and merges — all against the running checkout, and
all reversible because a change never reaches the running tree until it is merged.

## The safety model (unchanged and non-negotiable)

- **A change lives in a workspace, never in the checkout.** Each change gets its
  own `git worktree` under `data/studio/workspaces/<slug>` on its own
  `studio/<slug>` branch cut from the base branch. The running project is not
  touched by a single edit.
- **Merge is gated.** `can_merge` refuses while the checks have not passed, while
  the workspace is empty, when the base branch has moved on (so a merge can never
  overwrite newer work), when the checkout is on another branch, and with
  uncommitted changes. `merge` fast-forwards the base branch to the workspace's
  commit.
- **The network is opt-in.** `git fetch`/`pull`/`push` require an explicit
  `--approve` (CLI) or a confirmation dialog (window). No read action ever touches
  the network.
- **A path cannot escape.** Every file operation goes through the engine's
  deny-by-default confinement, a text-extension allow-list, and a size cap.
- **The model is local.** The coding model is an in-process GGUF or a local Ollama
  model; nothing is sent to a hosted service.

## The window (`gui/code_studio_dialog.py`)

Built the way every other window is built (the shared `Theme` tokens,
`RoundedBox`, `gui.rich_text` Markdown, the shared `gui.chat_composer.ChatComposer`,
the three `gui.window_controls` title-bar buttons, and a worker-thread/queue the UI
thread drains). The left column is a `SidebarColumn` of collapsible groups, one per
part of a developer's day:

| Section | What it does |
| --- | --- |
| **Workspaces** | New change (`PR1`, `PR2`, … named for you), the workspace list, discard. |
| **Explorer** | The workspace file tree; new file/folder, rename, copy, delete. |
| **Editor** | Open and save a file; outline a file's symbols; blame a file; discard a file's changes. |
| **Find & replace** | Literal search across the tree; preview a project-wide replace, then apply it. |
| **Source control** | Status, stage/unstage, commit, history, branches, create branch, stash/pop, conflicts, remotes, fetch/pull/push (gated), export a patch. |
| **Tasks** | The project's own commands (tests, lint, types, evaluation, build, run…), plus a custom command; output streams live; cancel. |
| **Notes** | A TODO/FIXME/BUG/XXX/HACK scan of the whole workspace. |
| **Review and apply** | Run the checks, show the diff, and **Apply to the project** (the user's approval). |

The centre is the transcript — the request echoed as a user bubble, the model's
steps streamed live, the checks, and the diff — and the shared composer at the
bottom. The editor slides in above the composer when a file is opened.

## The CLI (`cli/studio.py`)

The headless twin drives the **same** `gui.code_studio.CodeStudioService`, so the
CLI and the window share one set of workspaces, one diff, and one merge gate — no
logic is reimplemented.

```text
python -m cli studio status | coder | backend | models | download --approve
python -m cli studio new ["fix the flag"] [--slug name] | list
python -m cli studio files <slug> | tree <slug> | read <slug> <path>
python -m cli studio write <slug> <path> --file f | edit <slug> <path> --old X --new Y
python -m cli studio search <slug> "text" | replace <slug> --old X --new Y [--apply]
python -m cli studio outline <slug> <path> | notes <slug>
python -m cli studio touch | mkdir | rename | cp | rm <slug> …
python -m cli studio git <slug> status|log|branches|stash|conflicts|remotes
python -m cli studio git <slug> stage|unstage|commit|restore|stash-save|stash-pop
python -m cli studio git <slug> show|blame
python -m cli studio git <slug> fetch|pull|push --approve     # the network is gated
python -m cli studio tasks <slug> | run <slug> <task> [--command "…"]
python -m cli studio ask <slug> "…" | diff | check | merge | discard <slug>
```

The shared flags (`--root`, `--base-branch`, `--max-steps`, `--check-command`,
`--json`) are accepted **before or after** the subcommand, so `studio list --json`
and `studio --json list` behave identically.

Exit codes: `0` success, `2` invalid arguments, `3` the coding model is not
available, `4` a workspace/gate refused the operation, `5` an unexpected failure.

## The engines

- `core/studio_git.py` — `GitEngine`: a dependency-free wrapper over the project's
  own `git` for status/log/branches/remotes/tags/stash/diff/show/blame/config,
  staging, commit/amend, restore, stash, branch create/switch/delete, patch
  apply (with `--check`), revert, and fetch/pull/push. Every read is read-only;
  the three network actions are explicit calls. Non-interactive, credential-
  stripping, and it supplies a commit identity only for its own commits.
- `core/studio_files.py` — `FileEngine`: the workspace file tree, create/rename/
  copy/delete for files and folders, a literal project-wide replace (preview by
  default, bounded), a Python/generic outline, and a note scan. Confinement and
  the size cap are enforced here.
- `core/studio_tasks.py` — the project's own task catalog: `detect_project` reads
  the checkout's markers (python/node/rust/go), `catalog_for` builds the runnable
  commands per kind (plus the project's own check command), and `CommandRunner`
  runs one with streamed output, a timeout, and cooperative cancellation. A
  command is always an argv — never a shell — so metacharacters are inert.
- `gui/code_studio.py` — `CodeStudioService`: the Tk-free brain that drives all
  three plus `core.code_studio.CodeStudio` and `core.code_agent.CodingAgent`. It
  never merges on its own and never edits the main checkout.

## Tests

`tests/test_code_studio.py`, `tests/test_code_studio_engines.py`,
`tests/test_code_studio_cli.py`, `tests/test_code_studio_ide.py`, and
`tests/test_code_studio_window.py` build a real throwaway git repository and
exercise the real engines, the real service, the real CLI, and the real Tk window
(the window tests run under `xvfb-run` in CI).
