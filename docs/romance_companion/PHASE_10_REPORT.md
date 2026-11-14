# Romance & Companion Studio — Phase 10 Report (Orchestrator, Window & CLI)

Phase 10 turns the Studio's layers into a **single conversation turn** and gives it
its two user-facing surfaces — a standalone professional window and a headless CLI.
It is additive: the orchestrator, the Tk-free studio service, the window, the CLI,
the menu/action wiring, the Phase-10 gate judgements, tests, and docs. No existing
application behaviour changes.

## What was built

| Module | Responsibility |
|---|---|
| `core/romance/orchestrator.py` | `CompanionOrchestrator` — compose safety → conversation → generation → relationship → memory into one turn; `send` (single) and `send_multi` (a cast, sequential generation) |
| `core/romance/routing.py` (edit) | `CompanionModelRouter.generate()` — the one inference seam, delegating to the existing `ModelManager` runtime; no second runtime, no network |
| `gui/romance.py` | `RomanceStudioService` — the Tk-free surface the window and the CLI **both** drive (characters, conversations, relationship, memory, scenarios, voice, safety, status, and the **Discover** example library) |
| `gui/romance_dialog.py` | `RomanceStudioDialog` — the standalone Studio window (sidebar · centre · right panel) |
| `cli/romance.py` | The headless twin (`python -m cli romance …`), driving the same service |
| `gui/app.py`, `gui/actions.py`, `cli/menu.py`, `cli/__main__.py` (edit) | The navigation entry: a sidebar action, the composer's **Companion** pill, the ＋ menu entry, and the CLI dispatch |
| `core/romance/gate.py` (edit) | 7 new judgements → 74 total, still 1.0 |
| `tests/test_romance_orchestrator.py`, `tests/test_romance_studio.py`, `tests/test_romance_window.py` | Orchestrator, service + CLI, and the real window |

## The turn, in order

`CompanionOrchestrator.send` is deliberate about ordering, because each step is a
guarantee:

1. **The safety floor is checked first** — before anything is written. A refusal
   returns a `TurnResult(refused=True)` and the conversation is untouched.
2. **The user's message is appended** before the model runs, so a generation
   failure still keeps what the user said.
3. **Generation** runs through the one seam (`router.generate`), which resolves the
   chosen model and runs the project's existing runtime — never a second runtime.
4. **A reply is appended only when one was really produced** — no empty turn is
   stored.
5. **The relationship and the memory are observed** from the exchange, with the
   message as evidence.

`send_multi` draws a **cast**: the user speaks once, each character answers from its
own prompt, and generation is **sequential** (one character, then the next), so a
low-memory machine never runs two models at once. Each reply carries its own
character's name as its speaker, so a character never impersonates another.

## Safety fails closed

`_safety_refusal` wraps the safety manager's verdict in a guard: a check that
**raises** is reported as a refusal (`"the safety floor could not be evaluated"`),
never allowed through and never crashed at the caller. The gate's
`negative-orchestrator-safety-fails-closed` judgement breaks the *underlying*
manager and proves the orchestrator still refuses the turn — the negative control
that a naive "wrap the call and pass on error" implementation fails.

## The window

`RomanceStudioDialog` follows the project's window conventions exactly (the shared
`Theme`, `DialogHeader` with its window controls, `SidebarColumn`, the worker-thread
/ queue pattern, `ThemedDialog` for live re-theming). It is a three-column page:

- **Left sidebar** — the character library, the archive, **Discover Characters**
  (the clearly-labelled example companions this build ships, added one at a time),
  recent conversations, the conversation-mode selector, the scenario library, the
  voice picker, and the privacy/safety controls (adult mode, local-only state).
- **Centre** — the selected character's header, the live relationship indicator,
  the conversation transcript, and the message composer.
- **Right panel** — the profile, personality traits and interests, the relationship
  state and its milestones, the important memories, the voice configuration, and the
  active scenario. It **collapses** (the header's *Right panel* control) to give the
  transcript the full width.

The application's honest disclosure — *these are fictional AI companions, not real
people; everything stays on this machine; adult content is off by default and every
fictional participant must be an adult* — is shown in the transcript from the first
moment, **outside the fiction**. The character's own words stay in character.

## The CLI

`python -m cli romance` drives the *same* `RomanceStudioService`, so the CLI and the
window share one database and one character library. Commands: `status`, `modes`,
`characters`, `new`, `show`, `discover`, `add-example NAME`,
`favorite`/`unfavorite`, `archive`/`restore`, `delete
--choice keep|purge|anonymize`, `export`/`import`, `chat`/`send`, `history`,
`conversations`, `recent`, `relationship`, `milestones`, `reset-relationship`,
`memories`, `recall`, `remember`, `forget`, `export-memories`, `scenarios`,
`scenario`, `voices`, `set-voice`, `safety`, `adult --on --confirm`, `boundary`.
Exit codes: `0` success, `2` invalid arguments, `3` the Studio or a model is
unavailable, `4` an operation was refused, `5` an unexpected failure.

## Honesty by construction

- **No second runtime, downloader, or registry.** The orchestrator's only inference
  call is `CompanionModelRouter.generate`, which delegates to the existing
  `ModelManager`. The Studio never reaches the network for inference.
- **The status never overstates.** `status()` reports the real character,
  conversation, memory, and scenario counts, whether a model is ready **and a real
  file is on disk**, and the real next step when it is not; the voice line names the
  engine to install.
- **The service is the one surface.** The window and the CLI call the same methods,
  so they cannot disagree about a character, a conversation, or a relationship.

## What this phase does **not** do

- It adds no model and no weights: with no model installed the Studio opens, lists
  characters, stores memories, and reports the real next step; a turn honestly
  reports that no model answered.
- It does not change the chat, Code Studio, or any other workflow.
