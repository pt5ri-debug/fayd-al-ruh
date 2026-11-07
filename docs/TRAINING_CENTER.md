# Training Center — the learn-and-train workspace

The Training Center is the project's **one professional workspace for learning,
building datasets, training, and evaluating** — reached from the composer's
**Training** control (beside **Code Studio**), from the sidebar's **Training
Center** entry, from `python -m cli training-center …`, and from four CLI menu
entries.

It is deliberately a **thin composition over the machinery the project already
has**, never a second implementation. The window and the CLI drive the *same*
Tk-free service (`gui/training_center.py`'s `TrainingCenterService`), so they
share one review queue, one dataset registry, one run log, one knowledge base,
and one history — no figure is computed twice and no surface can disagree with
another.

## Why it exists

The project already had every piece of a learning pipeline — the knowledge base
and retrieval, the document readers, the governed dataset builder, the
controlled Phase-6 trainer, the Model Manager, the characters, the cloud client,
the web search — but no single surface that put them together. The Training
Center is that surface. It adds **no new model, no new store, and no new
training loop**: it orders what exists and makes it usable.

## The window (`gui/training_center_dialog.py`)

Built the way every other window is built: the shared `Theme` tokens,
`RoundedBox`, `gui.rich_text` Markdown, the shared `gui.chat_composer.ChatComposer`,
the three `gui.window_controls` title-bar buttons, a `SidebarColumn` of
collapsible sections, and a worker-thread/queue the UI thread drains (Tk is not
thread-safe). It adopts `ThemedDialog`, so a theme or font change restyles it in
place without closing it.

Every button's **label, one-line hint, and hover tooltip come from the
catalogue** (`core/training_center/catalog.py`), so the window and the headless
twin can never describe a control differently. A control that is *governed*
(asks for confirmation and/or writes) says so; a control that is *unavailable*
says so with the real next step rather than pretending.

The fourteen sidebar sections are:

1. **Dashboard** — every figure read from the real stores, with charts.
2. **Learning Sources** — databases, the internet, Gemini/cloud, local models,
   uploaded documents, books, characters.
3. **Knowledge Acquisition** — fold documents into the project's own knowledge
   base and retrieval; search, inspect, reindex, back up, export.
4. **Question-and-Answer Learning** — ask, generate questions, compare, review,
   approve; export only approved answers.
5. **Dataset Studio** — build, validate, list, export governed datasets.
6. **Training Jobs** — start, cancel, resume, retry, evaluate; live progress.
7. **Model Fine-Tuning** — which methods the selected model and this machine
   really support (a GGUF is never a training base; QLoRA needs a CUDA GPU).
8. **Pretraining and Continued Training** — from-scratch and continued training
   with compatibility validation and resource estimates.
9. **Evaluation and Benchmarking** — held-out evaluation with honest metrics.
10. **Knowledge Base Management** — the bulk knowledge base: build, index,
    search, report.
11. **Training History** — the joined, searchable timeline.
12. **Resource Monitor** — this machine's own CPU/RAM/disk, honestly.
13. **Settings** — the Training Center's own preferences.
14. **Pipelines** — the visual learning-pipeline builder and runner.

## Live training details in the chat area

Every long operation streams its detail into the window's own chat area *while it
runs*, not only when it ends -- exactly the pattern the main window uses for a
training action. A worker thread never touches a widget: it queues each line on a
`queue.Queue`, and the UI thread's pump drains it into **one growing message**,
coalesced to a bounded rate (~0.15 s) so a fast training loop cannot freeze the
window.

What streams, and from where:

- **A training run** (`Start a training job`, `Continue training the checkpoint`,
  `Resume from a checkpoint`, `Retry a failed job`) wires the pipeline's own
  `on_step` callback to the chat area: the first step, the last, and about every
  tenth in between, each line naming the run, `step N/max (P%)`, the loss,
  steps/second, and an ETA. The run still ends with its full result message.
- **A pipeline run** streams each stage's outcome (`[ok] import: …`) as it
  completes, through the runner's own `on_stage` callback.
- **Building the bulk knowledge base** streams the build's own progress lines
  (category fetches and failures), through `build_knowledge_base(progress=…)`.

`_start_streamed_job` is the one place the wiring lives, so every training
control streams identically; the CLI is unaffected (it prints to stdout as
before).

## The CLI (`python -m cli training-center …`)

```bash
python -m cli training-center dashboard          # the real figures
python -m cli training-center catalog            # what every control does
python -m cli training-center settings           # show / --set key=value
python -m cli training-center import <path…>     # files or folders
python -m cli training-center sources --url <u> --approve   # governed web
python -m cli training-center knowledge --query <text>
python -m cli training-center ask --topic <t> --count 5 --provider gemini
python -m cli training-center review --approve <id>
python -m cli training-center export --dataset <id>
python -m cli training-center datasets --validate <id>
python -m cli training-center jobs --cancel <run_id>
python -m cli training-center evaluate <run_id>
python -m cli training-center pipeline --run <id> --approve
python -m cli training-center history --kind import
python -m cli training-center resources
python -m cli training-center web --on             # the internet switch
python -m cli training-center providers            # the choices a dropdown offers
python -m cli training-center gate               # the held-out gate report
```

Exit codes: `0` success, `2` invalid arguments, `3` a source or engine is
unavailable, `4` a gate refused the operation, `5` an unexpected failure.

## Choices, not free typing

Every control that used to ask for a name in a plain text box now offers an
**editable dropdown**: a user picks a real value from the list *or* types one.
`gui/choice_dialog.py`'s `ask_choice` is the one helper behind it, and the service
(`gui/training_center.py`) is the one place that computes the choices, so the
window and the CLI can never disagree about what is on offer:

- **provider** — `manual`, `document`, every cloud provider by name
  (`gemini`, `openai`, …), a local model per mouth (`local:ultron`), and a
  character per key (`character:hakeem`);
- **mouth / character** — the local mouths and the seven characters by name;
- **dataset / run / pipeline id** — read from the real stores, newest first (a
  dataset may still be *typed*, because it is created by building it);
- **learning topic** — the project's own disciplines and corpus-entry titles, so
  a topic is picked from what the project really knows (and typed when it is
  new);
- **setting key and value** — the preference keys and their sensible values.

## The internet switch

The Training Center has **one** internet switch (`web`, off on a fresh install
and persisted). It governs every path that leaves this machine — a cloud
provider, a URL import, and the bulk knowledge-base build — so turning it on *is*
the human approval, exactly as the chat's web toggle is for a search. It is
reachable three ways, all driving the same setting:

- the **Web** switch beside the message box in the window;
- the *Settings → Turn the internet on or off* entry;
- `python -m cli training-center web --on|--off` (or `settings --set web=true`).

When an option that needs the internet is chosen while the switch is off, the
window asks once and turns it on for the user; if the user declines, nothing runs
and the reason is stated. The CLI's `sources` command refuses with the real next
step rather than silently doing nothing.

## The safety model (non-negotiable)

- **Nothing is trained on unreviewed data.** A gathered answer is a `pending`
  `LearningExample`; only `approved` examples can be exported into a dataset, and
  the dataset builder refuses anything else. This is a negative control in the
  held-out gate.
- **A governed stage is refused without approval.** Training, a web crawl, and a
  cloud call each pass the same permission + human-approval gate every other
  external action uses. A pipeline that reaches a governed stage without
  `approved=True` stops there.
- **An unimplemented capability is reported unavailable**, never run as a silent
  no-op or substituted with a different method.
- **Nothing is fabricated.** Every dashboard figure is read from a real store; an
  unreportable resource (a GPU the platform cannot see) reads as ``None``, never a
  zero; a training-time or memory estimate is labelled an estimate.
- **Imported content is data, not instructions.** A web page or a document can
  never change a system prompt, run a command, or alter the configuration.
- **No secret is exposed.** A cloud provider reads its key from the environment
  at call time and never logs, stores, or prints it.
- **No model weight is touched** except through the project's own Phase-6
  controlled trainer, and that only ever produces an *adapter*, never a base.
- **A character's personality is never silently changed.** A character may take
  part as a learning participant, but its persona and system prompt are untouched
  unless the user edits them explicitly in the Characters window.

## Supported learning sources

| Source | Backed by |
| --- | --- |
| Local databases (SQLite) | `gui.sql_import.SqlImportService` over `ingest.sql_job` |
| Tabular/structured files (CSV, JSON, JSONL, XML) | `ingest/` + `core.document_reader` |
| The internet (URLs, crawl) | `ingest.discovery` + `core.web_source`, governed |
| Gemini / any OpenAI-compatible cloud model | `core.cloud_model` (no third-party dep) |
| Local models (Ultron, Vision, Banner, the companion, the checkpoint) | `ai.mouths.build_mouth` |
| Uploaded documents (TXT, MD, PDF, DOCX, DOCM, CSV, JSON, JSONL, XML, HTML, ZIP) | `gui.import_center` + `core.document_reader` |
| Books and collections (EPUB, PDF, TXT, MD, folders) | the same readers, grouped by chapter |
| Characters | `ai.characters.CharacterMouth` (read-only) |

## Supported training methods

Whatever the project's own Phase-6 pipeline supports, and no more — the method
compatibility check reports the truth per model and per machine:

- **Adapter fine-tuning (LoRA)** — the in-process numpy LoRA backend, always
  available for the project's own transformer.
- **PEFT/transformers LoRA** — the optional `[training]` extra.
- **QLoRA** — only with a CUDA GPU (this machine's GPUs have none).
- **Continued training / resume** — `--resume-from <adapter>` continues a run.
- **Pretraining from scratch** — the in-process transformer's own trainer.
- **A GGUF is never a training base** — the compatibility check refuses it by
  name.

## Opening Training from the composer

The composer's bottom-left toolbar carries a **Training** pill (🎓) directly
beside **Code Studio**. Clicking it — or the *Open the Training Center…* item in
its menu — opens the Training Center window without disturbing the chat session
(one window is kept and reused, so a second click raises the open one). The
sidebar's **Training Center** entry opens the same window.

## Running a complete learning pipeline

```bash
# 1. Learn — gather answers into the review queue (nothing is trained yet).
python -m cli training-center ask --topic "reservoir simulation" --count 5

# 2. Review — approve only what is good.
python -m cli training-center review --pending
python -m cli training-center review --approve <example_id>

# 3. Build a governed dataset from the approved examples.
python -m cli training-center export --dataset learned

# 4. Validate the dataset before training.
python -m cli training-center datasets --validate learned

# 5. Check feasibility, then train (explicit approval required).
python -m cli train feasibility --model fayd-inprocess-model --dataset learned
python -m cli train run --dataset learned --approve

# 6. Evaluate on the held-out split.
python -m cli training-center evaluate <run_id>
```

In the window, the same sequence is the *Question-and-Answer Learning* →
*Dataset Studio* → *Training Jobs* → *Evaluation* sections, and the *Pipelines*
section chains the stages into one named, re-runnable workflow.

## Known limitations

- A cloud provider and the web need explicit approval (and, for a cloud model, a
  key in the environment); without them the surface reports the real next step.
- QLoRA and GPU training require a CUDA GPU; this machine has none, and the
  compatibility check says so rather than offering a method it cannot run.
- Fine-tuning only ever produces an **adapter**; a base GGUF is never modified.
- Parquet import is offered only when a compatible dependency is present; the
  readers otherwise cover JSON/JSONL/CSV/TXT/MD.

## Tests

- `tests/test_training_center.py` — the core records, providers, review queue,
  dataset studio, jobs, dashboard, resources, pipelines, history, catalogue, and
  the held-out gate.
- `tests/test_training_center_cli.py` — the CLI surface and exit codes.
- `tests/test_training_center_window.py` — the real Tk window (skips with no
  display): the sidebar sections, the catalogue-driven buttons, and the **live
  stream** (the step throttle's first/last/about-ten rule, `_start_streamed_job`
  wiring `on_step`, and the growing message the pump draws).
- `tests/test_training_center_choices.py` — the editable dropdown: the label/value
  resolution (headless) and the window's switch and dropdown wiring (skips with no
  display).
