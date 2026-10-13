# AGENTS.md — Fayd Al-Ruh

Evidence-aware symbolic cognitive architecture framed as a petroleum-engineering
mind. English is the primary language for the code, menus, answers, and docs;
Arabic is kept as secondary aliases where useful.

## Workflow (hard rules)

- ALWAYS work on a feature branch; never commit to `main`.
- Before finishing: `python -m pytest -q`, `python -m ruff check .`, `python -m mypy`,
  `python -m evaluation` (must print PASS, no regression).
- Push branches and open PRs with the `create_pr` tool. Merge only on the user's
  explicit approval ("نعم قم بالدمج").

## Layout

- `core/` — cognitive architecture. Key seeds:
  - `core/domain_knowledge.py` — the petroleum *frame*: 17 disciplines, concepts
    + relations, `seed_ontology`, `discipline_coverage`, `discipline_labels`.
  - `core/curriculum_knowledge.py` — the *curriculum*: 48 courses across
    foundations / earth sciences / engineering science / petroleum disciplines
    (including an advanced block), `seed_curriculum`, `curriculum_coverage`,
    `curriculum_coverage_score`.
  - `core/petroleum_corpus.py` — the knowledge *content*: 48 entries across 18
    disciplines with definitions, relations, formulas, procedures, standards,
    and glossary; `seed_petroleum_corpus`, `corpus_documents`,
    `corpus_sentences`, `corpus_coverage`, `corpus_coverage_score`.
    All three seeds run in `CognitiveController.__init__` and share nodes
    (idempotent), so frame + curriculum + corpus form one graph.
- `ml/` — learning. `ml/petroleum.py` trains the Q-policy + predictive brain on
  the discipline tasks in `core/petroleum_tasks.py`. `ml/release.py` trains every
  layer (base pipeline, petroleum tasks, language model) and writes
  `ml/artifacts/release_manifest.json` (with `knowledge.curriculum_coverage` and
  `knowledge.corpus_coverage`). Trained artifacts in `ml/artifacts/` and
  `ml/llm/checkpoints/fayd-llm.npz` are committed on purpose for a release;
  regenerate with `python -m cli release`.
- `core/paths.py` — resolves bundled assets vs. writable data (source vs. frozen
  build). A source checkout keeps learned data (artifacts, LLM checkpoints,
  knowledge base, learned documents) in a git-ignored `data/` folder beside the
  code, so retraining never dirties the working tree; the committed `ml/artifacts`
  and `ml/llm/checkpoints` copies stay as the read-only fallback. `ensure_data_layout`
  copies any pre-upgrade in-repo data into `data/` on first run.
  `core/version.py` — the single version string.
- `core/updater.py` — program updates for a source checkout. `check_for_updates`
  fetches the remote (never touching the working tree) and reports how many
  commits the checkout is behind; `apply_update` fast-forwards the tracked branch
  only when the tree is clean and the branch is not ahead, so local work is never
  destroyed. It reports when a commit touches the committed training files, but
  never copies or moves the user's own data in `data/` — a fast-forward cannot
  touch the git-ignored folder, so it is left exactly as it is. It is a thin
  wrapper over the project's own `git`, carries no extra dependency, strips
  embedded credentials from every message, and tells a frozen portable build to
  download a new bundle instead. Surfaced as the CLI command
  `python -m cli update [--apply]` and the GUI/CLI "Search for program updates
  (GitHub)" entry.
- `core/progress.py` — live progress for long runs. `ProgressReporter` (stdlib
  only) is callable, prints percentage/elapsed/eta to `sys.stderr` (so stdout
  stays clean for `--json`), and every training loop takes an optional
  `progress` callback. `ml`, `ml.petroleum`, `ml.llm`, and `ml.release` expose
  `--quiet` to suppress it; `ml.release` streams its child process's stderr
  line by line, and the GUI (`gui/app.py`) redirects stderr into the transcript
  so a long run never looks frozen.
- `ml/llm/corpus.py` builds the language-model text, including sentences generated
  from every discipline and course (`curriculum_sentences`), from the petroleum
  corpus (`corpus_sentences`), and numbered procedures (`procedural_sentences`);
  `corpus_units()` returns the distinct documents for an honest document-level
  split, and `build_corpus(..., exclude=...)` drops held-out units from training.
- `ml/llm/transformer.py` — decoder-only transformer with manual backprop.
  `TransformerConfig.preset(name)` selects a capacity tier (`small` default,
  `release-1`, `capable`) with optional weight tying and dropout. Training uses a
  warmup+cosine learning-rate schedule, gradient clipping, and periodic held-out
  validation (`train_language_model` returns `TrainReport` with `val_losses` and
  `learning_rates`). New config fields persist in save/load. `train_language_model`
  also takes `optimizer`/`start_step`, and `train_and_save` exposes `--resume`: it
  loads the checkpoint weights + saved tokenizer and continues instead of starting
  over, extending the schedule to `start_step + steps`.
- `ml/llm/schedule.py` — the sharded, resumable schedule. By default a `--resume`
  replays the same corpus, so more steps deepen the fit rather than reaching new
  material; `--llm-shards N` (in `ml/llm` and `ml.release`) instead splits the
  document universe into `N` deterministic shards and trains one per run,
  recording the consumed shards in `<checkpoint>.schedule.json` (gitignored run
  state) so a later `--resume` trains the next shard and cycles once all are
  consumed. `ShardPlan.shards`/`next_index`, `ScheduleState`, `load_schedule`/
  `save_schedule`/`schedule_path`, `select_shard`, and `describe_schedule` (used
  by the release manifest); the shard is recorded only after a run succeeds.
- `ml/llm/state.py` — run state for resuming: `save_training_state` /
  `load_training_state` persist Adam's moments and the cumulative step count to
  `<checkpoint>.state.npz` (gitignored as run state, not release content). A
  mismatched sidecar is a hard error; a missing one resumes weights with fresh
  moments. `load_step_count` reads just the step figure for honest manifests.
  The GUI sidebar's `llm`/`release` actions and the CLI `release` menu entry pass
  `--resume`, so the menus accumulate training rather than restarting; `ml/release`
  records `total_steps` and `resumed` (true only when a checkpoint was actually
  loaded).
- `ml/llm/evaluate.py` — honest measurement: `document_split` (whole documents to
  train/val/test), `perplexity` (held-out, dropout off), `aggregate` (mean±std
  across seeds), and bits-per-byte for cross-tokenizer comparison. CLI:
  `python -m ml.llm.evaluate --checkpoint <path>`.
- `ml/llm/bpe.py` — byte-level BPE tokenizer trained in-project. `ml/llm/tokenizer.py`
  has `train_tokenizer(kind, text, vocab_size)` / `save_tokenizer` / `load_tokenizer`;
  the trainer writes `<checkpoint>.tokenizer.json` and `LocalLanguageModel` loads it
  automatically. BPE compresses the corpus ~3x vs raw bytes; merges are learned from
  a bounded 300k-char prefix. `--tokenizer-kind byte` restores the old byte behaviour.
- `core/conversation.py` — multi-turn memory (sessions, roles, bounded window,
  `JsonStore` persistence). `core/documents.py` — citable documents + knowledge
  corpus from the seeds (disciplines, courses, and the petroleum corpus).
  `core/rag.py` — fuses lexical coverage (Arabic-normalised),
  hashed semantic, and multi-hop ontology signals into `RetrievedPassage` with
  citations.   `core/assistant.py` — `GroundedAssistant` / `build_default_assistant`;
  answers with citations, extractive fallback with no model, and a groundedness gate
  that rejects ungrounded model output. CLI: `python -m cli chat` (text-only
  conversation) and `python -m cli jarvis` (listen/act/speak; `python -m ai.voice`
  and `python -m ai.jarvis` also use it). Both build their session through
  `ai/jarvis.py:build_session`, so the assistant, learner, and sandbox wiring
  lives in one place.
- `ai/mind.py` — the model mouth: `MindAssistant` answers from the **trained
  language model** directly, with no retrieval and no citations, so the quality of
  what the model itself learned is visible. The raw continuation is tidied by
  `core.composition.compose_model_answer` — it keeps the whole sentences the
  model finished (in order), drops the leading fragment, a trailing clause that
  ends on a conjunction/article, a repeated sentence, and a bare section label
  ("Step 2 of ...") when a cleaner statement survives; it centres the answer on
  the question by opening with the sentence that names the question's words; and
  it collapses repeated punctuation/doubled words *and short repeated phrases*
  ("the same the same") while leaving a hyphen run between word characters
  ("10--20", "5CT--spec") intact, capitalises and punctuates. The phrase collapse
  is applied until it stops changing, so a *run* the byte model stalls on ("the
  casing the casing the casing the casing") collapses all the way to one copy
  rather than leaving every other copy; a stall the model joins with a
  conjunction ("the stage complex and stage complex") collapses the same way,
  while a real pair of different phrases ("porosity and permeability") is left
  alone. It also closes the
  two ragged boundaries the byte model leaves: a sentence that ran on past a
  boundary in lower case ("...mixed. and the slurry is pumped.") is capitalised
  (presentation only), and a space before a full stop or a glued-on next sentence
  ("flows .Then it stops.") is tidied to one boundary -- while a decimal, a
  hyphenated range, and a formula ("q / A") keep their spacing. A *stuttered
  separator* ("2,,3", "3:1::1", "saturation, , are") is trimmed to the model's own
  mark rather than turned into a full stop, so the model's number or list is not
  corrupted. The mouth also
  forwards the checkpoint's `temperature`/`seed` to the generator, so
  `--mind-temperature 0` really is greedy and reproducible (the generator's own
  0.8/unseeded defaults no longer override it).
  The mouth therefore reads as connected sentences while still speaking the
  model's own words; `MindResult.raw_text` keeps the untouched continuation for
  inspection. When the model produced nothing wordbearing (an empty continuation
  or a stopping run such as `")))))"`) the tidied text is empty and `ask`
  substitutes `MIND_EMPTY`, so the answer is never a header with no body — the
  GUI would otherwise draw a blank bubble carrying only its Copy button.
  `load_mind_model` /
  `mind_status` load and describe the checkpoint. Wired **only** into
  `python -m cli chat --mind` (and the `from the model` / `from the sources`
  switch, matched in `ai/chat.py`; the portable menu reaches it through
  `cli.menu.run_mind_chat`, command `python -m cli mind-chat`); it deliberately never touches
  `ai/jarvis.py`, `core/jarvis.py`, `JarvisMode`, `build_session`, or the shared
  `JarvisControls`, so Jarvis is unaffected. Every model answer is labelled as
  ungrounded on purpose.
- `core/composition.py` — the deterministic composer: dependency-free rules that
  render evidence (retrieved passages, web sources, causal/reasoning/decision
  traces) into clean attributed prose. Shared by the assistant's extractive
  fallback, the web verifier's answer, and the causal/reasoning/decision
  explainers, so surface wording lives in one auditable place. `compose_model_answer`
  is the model-output half: given an optional `question` it keeps the whole
  sentences the model finished, drops fragments/dangling tails/repeats and bare
  section labels, centres the answer on the question, collapses repeats, and reads
  an ontology identifier the model echoed ("pore_pressure") as the phrase it names
  — without rewriting the model's words — and is used by `ai/mind.py`.
- `core/speech.py` — speech I/O contracts (ASR/TTS), dependency-free `AudioClip`
  (PCM/WAV), transcript + silent fallbacks, `VoiceAssistant` voice loop. CLI:
  `python -m ai.voice` / `python -m cli voice`.
- `core/jarvis.py` — the unified Jarvis loop over the assistant, the tool router,
  and speech; it keeps one `Conversation` per run (persisted through the memory's
  `JsonStore` when one is configured) and a mute state.
  `core/jarvis_controls.py` matches the conversation controls (stop/speak/repeat/
  new conversation/save/sessions/confirm/correct/help) exactly; a confirmation or
  correction is folded into an optional `ContinuousLearner`, and the retriever is
  re-indexed so the change is visible to the next answer.
  `core/preferences.py` persists user preferences. CLI: `python -m cli jarvis`
  (`--session <id>`, `--list-sessions`, `--no-feedback`).
- `core/tools.py` — deny-by-default tool sandbox: allow-list, root confinement,
  read-only mode, capability + human-approval gates, built-in file/search/calculator
  tools returning `ToolResult`. `core/tool_agent.py` — `ToolPlan`/`ToolAgent`
  coordination with a full trace. CLI: `python -m ai.agent_demo` / `python -m cli agent`.
- `core/knowledge_update.py` — folds confirmed statements into the document store
  with provenance and decay (`KnowledgeUpdater`). `core/continuous_learning.py` —
  `ContinuousLearner` (explicit feedback only) + `RegressionMonitor` for baseline
  comparisons. CLI: `python -m ai.learn_demo` / `python -m cli learn`.
- `core/web_source.py` — governed web search: `WebSearchProvider` contract with a
  keyless `WikipediaSearchProvider` (real internet, used by default), an offline
  `StaticSearchProvider`, an `HttpSearchProvider`, and a `FallbackSearchProvider`
  that drops to offline when the network is unavailable; `WebSearchService` runs
  only through the permission/approval/audit gates (denied by default) and
  `estimate_credibility` scores a source by its domain. `core/web_verifier.py`
  `WebVerifier.verify` searches, cites sources, compares them with the internal
  corpus, and learns an answer via `KnowledgeUpdater` only when its trust clears
  the threshold and it does not contradict known knowledge. CLI:
  `python -m ai.web_verify` / `python -m cli web` (pass `--approve` to grant the
  external capability; omit the question to be prompted for your own).
  The conversation loop uses the same engine: `GroundedAssistant` takes optional
  `web`/`web_enabled`/`web_threshold`/`on_learn` and, when its top internal
  passage is weak, searches, compares, and learns a trusted answer into the store
  it retrieves from (immediately retrievable). `build_default_assistant` wires a
  default internet verifier when `web_enabled=True`. Jarvis toggles it with the
  `web on`/`web off` control; the CLI exposes `--no-web` and `--web-threshold`.
- `core/internet_learner.py` — the *Continuous Internet Learner*: batch learning
  over corpus-derived petroleum topics from trusted domains only (SPE, OnePetro,
  API, ISO, ASTM, Wikipedia, `.gov`/`.edu`), folding accepted statements into the
  document store through `KnowledgeUpdater`. Denied by default and dry-run by
  default; `--write` persists with provenance. `commit(report)` saves a previewed
  dry-run report without searching again. CLI: `python -m cli internet-learner`.
- `core/knowledge_navigator.py` — `science_branches()` exposes the curriculum as
  an ordered tree of **science branch → course → concept** with readable concept
  phrases; `InteractiveLearner` (in `ai/internet_learner.py`) walks it with
  numbered menus, searches at any level, reviews results, then saves or ignores.
  Menu entry (5) opens this interactive mode.
- `core/vision.py` — optional multimodal perception: dependency-free `ImageBuffer`
  (PPM P6), measurable `ImageFeatures`, `HeuristicVisionEncoder`, and a
  `MultimodalAssistant` that grounds an image description in the RAG store. A real
  encoder can implement the `VisionEncoder` protocol without touching the assistant.
- `core/imagegen.py` — optional **text-to-image**, the counterpart to
  `core/vision.py`. Dependency-free: `encode_png` writes a valid PNG with only
  `zlib`+`struct`, and the default `ProceduralImageGenerator` renders a
  deterministic abstract composition from a prompt-derived seed (`ImageRequest` /
  `GeneratedImage`, `ImageGenerator` protocol, `load_image_generator`,
  `image_backend_names` / `image_backend_status`, `image_filename`,
  `default_image_dir`). It is a *renderer, not a model*, and every
  `GeneratedImage` says so (`learned=False`, an honest `description`).
  `load_image_generator("diffusion", ...)` builds the optional learned backend.
  `gui/image_gen.py` adds the Tk-free `ImageGenerationService`
  (`parse_size` reads a trailing `WxH`; `ImageReport`; `for_backend`,
  `backend_status`, `download_model`) and `gui/image_gen_dialog.py` is the
  window's page (＋ menu → "Generate an image from text…", a sidebar action, a
  backend selector, and `python -m cli image`).
- `core/imagegen_diffusion.py` — the optional **learned** text-to-image backend:
  a real Diffusers diffusion model (`stabilityai/sd-turbo` by default, 1-4 steps)
  that runs on CPU. Everything heavy is lazy: torch/diffusers are imported only
  when generating, so a base install never needs them (`pip install .[image]`).
  Weights download only through the project's permission + human-approval gates
  (`DOWNLOAD_TOOL`, denied by default; `python -m cli image download --approve`)
  and are cached under `data/models` (git-ignored). `DiffusionImageGenerator`
  implements `ImageGenerator` with `learned=True` and a `description` that names
  the model, steps, and seed; `DiffusionUnavailable` is raised when the extra or
  the weights are missing so the caller can fall back to the procedural renderer.
  `DiffusionPipelineRunner` is the seam a test injects (no torch needed).
- `core/knowledge_base.py` — the **bulk knowledge base**: a keyless
  `WikipediaFetcher` (batched MediaWiki API, `Retry-After`-aware), `chunk_text`,
  and a sharded JSONL `KnowledgeBaseStore`. `build_knowledge_base(store_dir,
  target_mb=...)` walks seed categories **recursively through subcategories**
  (`depth=2`), de-dupes titles, and stores until the byte target, resumably via
  `checkpoint.json`. Lives under the data directory (`data/knowledge_base`,
  git-ignored), built on demand — never committed. CLI:
  `python -m cli knowledge build|build-index|stats|search` (`cli/knowledge.py`).
- `core/knowledge_index.py` — a SQLite **FTS5** index (stdlib, no new dependency)
  over the shards, built once with `build-index` (~5 s for 81k chunks). Turns a
  query from a ~35 s streaming scan into milliseconds. `KnowledgeTextIndex` owns
  its `KnowledgeBaseStore` (`read_chunks` fetches only the winning chunks);
  contentless FTS keeps the index ~60 MB without duplicating the text.
- `core/rag.py`'s `BulkCorpusRetriever` serves queries from that index when
  present and streams otherwise (bounded memory either way); wired into
  `RagRetriever(bulk=...)` and `build_default_assistant(bulk=True)`. Voice and
  Jarvis call `bulk_knowledge_available()` and enable it automatically once the
  index exists. It is *reference material only*: never fed to the acceptance
  policy or belief layers.
- `core/version.py` is the single source of the version string.
- `ingest/` — the **local-first ingestion subsystem**: a separate pipeline that
  turns *sources* into the same on-disk knowledge store without ever loading a
  whole dataset into RAM. `ingest.records` holds the contracts (`SourceSpec`,
  `SourceRecord`, `IngestItem`, `IngestionReport`, `Connector`); `ingest.registry`
  resolves a source to a connector by scheme/suffix (files is the path/dir
  fallback; the internet connector is registered only with `allow_web=True`);
  `ingest.connectors.{files,sql,web}` stream TXT/Markdown/CSV/JSON/JSONL and
  digital PDFs (**Stage 2A adds `.pdf`**) and Word `.docx`/`.docm` (**Stage 2B**
  adds Word); a scanned PDF is read with the opt-in OCR path
  (**Stage 3**, `allow_ocr`/`--ocr`, via `core.pdf_ocr`), read-only
  SQLite/driver DBs, and
  (denied by default, host-allow-listed or explicitly authorized) one HTTP(S)
  page. `ingest.pipeline.IngestionPipeline`
  streams `plan`→`stream_item`, chunk-appends to `core.knowledge_base`
  (`INGESTED_SOURCE="ingested"`), skips unchanged sources via `ingest.dedup`
  (`DedupLedger`, content-addressed `ingest:<blake2b>` chunk ids), captures a
  bad source in `ingest.errors` (never raises a batch), and (optionally) records
  resumable progress in `ingest.jobs`. `ingest.delete` removes a source's chunks
  by an all-or-nothing shard rewrite and drops the stale FTS index. Ledger and
  jobs live under `<store>/ingest/`; auto-excluded from its own walk. It never
  touches training artifacts. Phase-1 interfaces and bookkeeping, all additive
  and stdlib/SQLite-only: `ingest.interfaces` names the explicit contracts
  (`SourceConnector`, `IngestionJob`, `JobStateStore`, all `runtime_checkable`);
  `ingest.source_registry` persists *text-free* source registrations
  (`SourceRegistration`/`SourceRegistry` via `storage.sqlite_store.SqliteStore`,
  with provenance, `trusted`/`enabled` flags, and last-run state);
  `ingest.metadata` stores per-document metadata only (`DocumentMetadata`/
  `DocumentMetadataStore`, hashes/sizes/chunk ids, no text); `ingest.config`
  (`IngestConfig`) resolves tunables as explicit arg → `FAYD_INGEST_<FIELD>` env
  → stored `JsonStore` value → default, and never enables web on its own. File
  imports are normalized before chunking by `core/text_normalize.normalize_text`
  (Unicode NFC + structure only; Arabic diacritics and letter distinctions are
  **preserved** by default). `--fold-arabic` is the opt-in that also unifies
  Arabic letter forms, and `--no-normalize` stores text verbatim. A transient
  local read is retried up to `--retries` times
  (`IngestionPipeline(retries=...)`) with the report counters rolled back so a
  retry never double-counts. A Word `.docx`/`.docm` (Stage 2B) is read through
  `core/docx_extract` and grouped into records along its own structure (a heading
  opens a record). Scanned PDF pages (**Stage 3**) are OCR'd only when
  `allow_ocr` is set (`--ocr`), through `core.pdf_ocr`: only the pages with no
  text layer are sent to OCR (a page with real text is never replaced), the work
  is bounded by `ocr_dpi`/`ocr_batch_size`/`ocr_max_pages`/`ocr_time_budget`
  limits, each record keeps its page number and an `ocr` flag, and a missing
  engine/extra is an actionable error rather than a silent skip. Each ingested
  record also writes text-free
  per-document metadata (source path, filename/extension, content hash, media
  type, chunk ids, ingestion timestamp) through `ingest.metadata`, wired into the
  pipeline/CLI best-effort so a metadata-write failure never loses the ingest.
  `IngestionPipeline(..., on_item=...)` emits one additive `ingest.records.ItemOutcome`
  per source item (filename, status ok/unchanged/failed, chunks, message), which
  the front ends collect for a per-file result. CLI:
  `python -m cli ingest add|list-connectors|
  stats|build-index|search|delete|jobs` (`cli/ingest.py`); the two knowledge
  actions mirror into the GUI sidebar.
- `gui/import_center.py` — the **Knowledge Import Center** engine: the
  user-facing front end over `ingest/`, shared by the window's ＋ control and the
  CLI. `KnowledgeImporter` (`import_files`, `import_folder(recursive=, job_id=)`,
  `retry_failed`, `resume_job`, `list_jobs`, `stats`) drives
  `IngestionPipeline` with an `on_item` collector and returns an `ImportReport`
  (counts + per-file `ImportOutcome`s, `failures`, `summary()`, `as_dict()`).
  `SUPPORTED_SUFFIXES` / `UNSUPPORTED_SUFFIXES` / `picker_filetypes()` are the one
  place the accepted formats are declared. It reuses `core.document_reader` and
  `core.knowledge_base` rather than reimplementing extraction. Reachable headless
  as `python -m cli import-center [interactive]|files|folder|resume|jobs|stats|formats`
  (`cli/import_center.py`); with no subcommand it prompts for a file or folder
  (the menu's item 10 runs `interactive`, so it imports a typed path rather than
  only listing formats). Scanned PDFs are reachable end to end: `--ocr` /
  `--ocr-languages` / `--ocr-dpi` / `--ocr-max-pages` select the opt-in OCR path,
  and any run that skipped a scanned PDF offers to retry it with OCR on a
  terminal. Mirrored as GUI action `import-center` — which the
  window routes to the dialog below — and opened from the composer's ＋ menu.
- `gui/import_dialog.py` — the Import Center **dialog** (Tk, imported lazily so
  `cli/` never imports Tk): Upload files / Import from a local folder / Import
  jobs, with a background worker thread, per-file results, retry, and a progress
  line. A **Read scanned PDFs (OCR)** checkbox (with a language field) turns on
  the OCR path for both import buttons, and a **Retry with OCR** button appears
  when a run failed on scanned PDFs. An **Install OCR support…** button appears
  when the OCR packages are missing, running the bundled requirements file so a
  desktop user never has to open a terminal. A **3. Import a SQL database**
  section opens the SQL screen below. `core/document_reader.py` reads
  TXT/Markdown (streamed above 4 MB), CSV/TSV, JSON/JSONL, digital PDF, and DOCX;
  scanned PDFs go through the opt-in OCR path in `core/pdf_ocr.py`. Every reader
  only parses bytes — no macro or embedded instruction is ever executed.
- `gui/sql_import.py` — the **SQL Database Import** engine (Tk-free): the
  user-facing front end over the resumable SQL ingestion service the CLI's
  `ingest sql-import` command uses. `SqlImportService` (`test_connection`,
  `inspect`, `preview`, `run_import`, `resume_job`, `retry_job`, `job`,
  `list_jobs`, `forget_job`, `stats`) drives `ingest.sql_job.SqlImportRunner`
  unchanged, so the window, the CLI, and the pipeline share one read-only
  connector, one dedup ledger, one durable checkpoint, and one job file under
  `<store>/ingest/jobs-sql`. `DATABASE_TYPES` declares the enabled engines
  (SQLite only, at this stage); `picker_filetypes()` is the one place the suffix
  list lives. A column uncheck becomes the connector's `exclude_columns`, so an
  excluded column is never in the projection, a preview, or the store.
  `redact_secrets` strips a password/token or a credential-bearing URI from every
  message, so no status line or dialog can leak one. `SqlImportJob.options` (in
  `ingest/sql_job.py`) records the connector options, so a resume reproduces the
  original table/column selection instead of a fresh guess.
- `gui/sql_import_dialog.py` — the SQL Database Import **screen** (Tk, imported
  lazily): 1. Database file (choose + read-only connection test + schema list) /
  2. Tables and columns (table multi-select, per-column include/exclude with a
  bounded preview) / 3. Start import (explicit, with a batch-size field) /
  4. Import jobs (durable status, Resume / Retry / Forget). Long imports run on
  a worker thread with readable progress (a determinate bar when a percent is
  measurable) and a **Cancel** button that trips the service's cooperative
  `should_cancel`; every status and result is secret-redacted before display.
  Reachable from the composer's ＋ menu (**Import a SQL database (SQLite)…**) and
  from the Import Center dialog.
- `gui/image_gen_dialog.py` — the **Generate an image from text** screen (Tk,
  imported lazily): a prompt box (a trailing `WxH` sets the size), optional
  width/height/seed fields, **Generate** (worker thread, then a `tk.PhotoImage`
  preview), and **Save a copy…**. It says plainly that this build has no learned
  text-to-image model and the picture is a geometric composition. Reachable from
  the composer's ＋ menu (**Generate an image from text…**) and the sidebar
  (**Generate an image from text**).
- `cli/ingest.py` — the ingestion CLI, including the **SQLite import** commands
  (`sql-drivers`, `sql-inspect`, `sql-preview`, `sql-import`, `sql-status`,
  `sql-resume`, `sql-cancel`, `sql-jobs`). They are the headless twin of the
  window's SQL screen and drive the *same* `gui.sql_import.SqlImportService`, so
  the CLI and GUI share one read-only connector, one ledger, one checkpoint, and
  one job file — the CLI reimplements no import logic. Options: `--tables` /
  `--exclude-tables` / `--columns` / `--exclude-columns` (exclusion wins),
  `--batch-size` / `--row-group` / `--page` / `--timeout` (validated before the
  source is touched), `--job`, `--schema-change`, `--quiet`, `--json`. A missing
  or unreadable database fails safely (never created or written). Exit codes:
  `0` success, `2` invalid configuration, `3` connection error, `4` incomplete
  (cancelled/interrupted, resumable), `5` unrecoverable failure. A cross-process
  cancel is a marker file (`SqlImportService.request_cancel`) that a running
  import polls, so `sql-cancel` works from a separate invocation.
- `evaluation/` — regression gate. `evaluation/runner.py` assembles metrics,
  `benchmark.py` declares `MetricSpec`s, `baseline.json` is the reference.
  Update the baseline with `python -m evaluation --update-baseline`.
- `cli/` — portable front end. `menu.py` holds actions + `MENU`; `__main__.py`
  dispatches commands and reconfigures stdout/stderr to UTF-8 (Arabic-safe on
  legacy Windows consoles). `cli/image.py` is the text-to-image command:
  `python -m cli image generate "prompt" [--width/--height/--seed/--output/--json]`
  and `python -m cli image backends`.
- `gui/` — ChatGPT-style desktop window, an alternative *front end* to `cli/`,
  never a replacement. `service.py` reuses the text `chat` session
  (`ai.jarvis.build_session(text_only=True)`) + `ai.chat.ModelSwitcher` and adds
  `list_sessions`/`resume`, `saved_session_summaries` (dated `HistoryItem`s for
  the sidebar), `set_operating_mode`, and `preview_execution` (the confirmation
  gate for a mutating tool); `actions.py` maps the sidebar to `cli.menu.run_*` —
  the **Actions** list mirrors numbered CLI items 5..11 (never repeating the
  window's own controls); `options.py` has the argument parser (`--mode ai|jarvics`);
  `app.py` is the Tkinter window (imported lazily, so `cli/` never imports Tk).
  Reachable as `python -m cli gui` and `python -m gui`.
- `gui/modes.py` — the two operating modes. **AI Mode** (`allows_tools=False`)
  answers conversationally; **Jarvics Mode** (`allows_tools=True`) also runs the
  sandboxed tools. `AnswerOnlyRouter` is the brain of AI Mode: it subclasses the
  real `NaturalLanguageRouter` and always decides *answer*, so a command-shaped
  sentence can never run a tool there. The chosen key persists through `JsonStore`
  (`load_mode_key`/`save_mode_key`), `resolve_mode` accepts a key or a label and
  falls back safely, and `HistoryItem`/`group_by_date`/`search_items` feed the
  sidebar's dated, searchable history. Tk-free, so it is testable headless.
- `gui/markdown.py` — a dependency-free block/segment renderer for the
  transcript: headings, lists, block quotes, fenced code (with a light token
  highlighter), tables, horizontal rules, and inline bold/code, returned as
  `StyledLine`/`Segment`s the window paints with Tk text tags. No Markdown library
  is added.
- `gui/app.py` — a professional desktop front end styled after the **OpenHands
  chat UI**: the same cool-grey token scale (base `cool-grey-950`, surface
  `cool-grey-925`, raised `cool-grey-900`), the gold primary accent, and
  HeroUI's small/large radii, defined once as the `Theme` dataclass
  (`DARK_THEME`/`LIGHT_THEME`) and mapped back to OpenHands' semantic names
  (`surface`, `foreground`, `text_dim`, `border_subtle`, `interactive_hover`
  …). `RoundedBox` paints a rounded-rectangle surface (Tk has no corner radius)
  and hosts real widgets, used for the rounded composer card and the right-side
  user bubbles. The transcript reads as a centred column (`_COLUMN_MAX`, with
  `_column_pad` recentring it on resize): user messages are small rounded
  bubbles on the right, assistant/note messages are full-width rows, and an
  assistant row carries a gold avatar. A palette that switches between a
  **dark** and a **light** theme (the Appearance control toggles it, the choice
  is saved through `JsonStore` and reloaded on the next launch, and the whole
  window is rebuilt for the new palette with the transcript preserved), a
  collapsible sidebar (brand + version, a filled **New chat** button, a search
  box, **Conversations** grouped by date, the sandbox tools, **Settings** with
  the answering-mouth switch and the **Actions** list) behind a scrollbar that
  appears only when the content overflows, a header showing the active mode by
  name (one switch) plus the model selector, and a centred hero/empty state that
  gives way to the first message (and returns when the transcript is cleared).
  On a narrow (mobile-sized) window the sidebar becomes a drawer that
  `_toggle_sidebar` shows or hides and that is restored when the window widens.
  Assistant answers are rendered as **Markdown** through `gui/markdown.py`
  (`_insert_rich`/`_insert_styled_line`/`_inline_tags`). A **＋** button beside
  the input attaches a Word (`.docx`/`.docm`) or text file: the file's text is
  indexed into the source assistant's store and previewed in the transcript, a
  removable chip tracks it, and detaching (the chip's ✕, or a new chat) removes
  it from the store. Each message body is a read-only `tk.Text`, so answers and
  live training output can be selected and copied — per-message Copy button,
  Ctrl+C copies the selection, Ctrl+A selects all, Delete clears, right-click
  offers copy / copy-conversation / select-all / clear; Ctrl+N new chat, Ctrl+S
  save. A small animated "Thinking…" indicator shows while an answer is pending.
  Live training output is queued and rendered at a bounded rate (coalesced into
  one widget update every ~0.15 s, fully drained before the result), so a fast
  training loop never freezes the window.
- `core/document_reader.py` — read text out of a Word file with no third-party
  dependency: a `.docx`/`.docm` is an Office Open XML package (a ZIP holding
  `word/document.xml`), read through **`core/docx_extract.py`** (Stage 2B), which
  owns the untrusted-ZIP handling — archive resource limits (member count,
  over-large part/member/total), encrypted/legacy OLE detection, malformed-package
  reporting — and returns ordered **paragraphs and table rows** (a heading carries
  a level) into a `DocumentText`. Legacy `.doc`
  and `.rtf` are refused with a "save as .docx" hint instead of being scrambled.
  Plain `.txt`/`.md` are read with **safe encoding detection** (a BOM, then
  strict UTF-8, then a short list of legacy encodings; a mis-decoded file raises
  rather than filling with U+FFFD). Large files are streamed line by line up to
  `MAX_TEXT_CHARS` with an incrementally-decoded, strictly-checked codec, and
  `DocumentText.truncated` / `DocumentText.encoding` report a capped read and the
  codec honestly. A digital `.pdf` is read **page by page** through
  `core.pdf_extract` (the optional `pypdf` extra); `DocumentText.page_numbers`,
  `DocumentText.scanned`, and `DocumentText.ocr_pages` carry the page split and
  the OCR provenance (which pages came from OCR). A scanned PDF with no
  extractable text falls back to the separate optional OCR path — either the
  Stage 3 per-page pipeline (`core.pdf_ocr`, enabled with an `OcrConfig` or an
  `ocr_engine`) or the simple whole-file `ocr` callable — and a
  `NormalizationError` is raised only when a requested normalization fails.
  `read_document(..., normalize=..., encoding=..., password=..., ocr_config=...,
  ocr_engine=...)` applies
  `core/text_normalize`, pins the codec, and decrypts a protected PDF. All lazy
  imports; the base install pulls nothing new and no document is ever executed.
- `core/pdf_ocr.py` — the **optional OCR path for scanned PDFs** (Stage 3), off
  unless a caller enables it, so the base install imports nothing from the OCR
  extra. `OcrConfig` is the one validated set of knobs (`enabled`, `languages`,
  `dpi`, `batch_size`, `max_pages`, `max_pixels_per_page`, `max_page_dimension`,
  `time_budget_seconds`, `max_temp_bytes`, `min_chars_per_page`, `retries`).
  `page_needs_ocr` flags a page whose native text is thinner than
  `min_chars_per_page` (a scan, or a bare page number); `extract_pdf_pages` runs
  native extraction first and OCRs **only** those pages — a page with real text
  is never sent to OCR and never overwritten — and returns an `OcrPdfResult`
  carrying every page's text plus `ocr_pages` (provenance), `failures`
  (per-page, never fatal), `skipped_pages` (a cap or the time budget was hit),
  and `languages_used`. The default engine is `TesseractOcrEngine` (maintained
  Tesseract): `pypdfium2` rasterises in memory (no external poppler) and
  `pytesseract` drives the engine. Limits are honest — the raster is refused
  *before* allocation if it exceeds the pixel/dimension caps, the temp PNG is
  measured against `max_temp_bytes` before it is written and removed after, and
  a transient page failure is retried `retries` times. `check_raster_limits` and
  `_write_bounded_png` are split out so the guards are testable without the
  optional packages. Language data is required separately: `usable_languages` /
  `resolve_languages` intersect the requested codes with what Tesseract has and
  report the missing model by name, so Arabic + English work only when
  `tesseract-ocr-ara` / `tesseract-ocr-eng` are installed. OCR reads pixels; it
  never executes the document. (System packages are listed in
  `OCR_SYSTEM_PACKAGES`; the extra is `[ocr]`.)
- `core/optional_deps.py` — the **optional-feature installer**, so a user who
  hits "OCR/PDF is not installed" has a next step the app can actually perform.
  It owns, per feature (`pdf`, `ocr`), the bundled requirements file, the import
  names to check, and the pip command that works from a source checkout
  (`python -m pip install -r requirements-<feature>.txt`) — because the project
  is **not on PyPI**, the old `pip install fayd-al-ruh[ocr]` hint could never
  work. `status` separates a missing *Python* package (pip can fix) from a
  missing *system* package (Tesseract; pip cannot), so the app never promises an
  install it cannot do; `install` runs the command and returns an
  `InstallResult`, and takes an injected `runner` so it is tested without the
  network. Dependency-free and import-safe. `requirements-pdf.txt` /
  `requirements-ocr.txt` mirror the extras. Surfaced as the CLI
  `python -m cli deps [--install pdf|ocr]` and the window's **＋ → Install OCR
  support…** entry and the Import Center's **Install OCR support…** button.
- `core/pdf_extract.py` — read text out of a **digital** PDF page by page with
  `pypdf` (maintained, pure-Python, BSD; the optional `[pdf]` extra), imported
  lazily so the base install stays dependency-light. `PdfPageStream` opens and
  validates the document up front (missing dependency, missing file, encrypted
  without a usable password, unreadable page tree, zero pages all raise
  `PdfExtractionError` with a specific reason) and then yields one `PdfPage`
  (its 1-based number, text, `rtl_hint`) at a time, so a 1000-page PDF is never
  held whole; the accumulated text is capped at `DEFAULT_MAX_CHARS`.
  `extract_pdf` is the materialising convenience. Encrypted, malformed, empty,
  and image-only (scanned) PDFs are distinguished, not lumped together: an
  image-only PDF raises a message that points at the OCR stage. `pypdf` only
  *parses* the file -- embedded JavaScript/launch actions are never executed and
  instructions inside the text are never followed. Arabic is recovered as far as
  the format allows (a `ToUnicode` map decodes it correctly; PDFs store glyphs in
  visual order so it is often reversed -- a format limitation, flagged by
  `rtl_hint`, not a bug here).
- `core/docx_extract.py` — read text out of a Word `.docx`/`.docm` (Stage 2B) with
  the standard library only (`zipfile` + `xml.etree`), treating the file as an
  **untrusted ZIP package**. `DocxReader`/`extract_docx` validate up front
  (existence, not a ZIP, an encrypted/legacy OLE header, the archive resource
  limits, a readable `word/document.xml`) and return an ordered `DocxDocument` of
  `DocxBlock`s: paragraphs (a heading style maps to `level` 1-9) and table rows
  rendered as TSV (`table` numbers them), so a table's rows keep their place in
  the flow. Resource limits are honest and configurable: a member-count cap
  (`DEFAULT_MAX_ENTRIES`), a per-member and total *declared*-size cap
  (`DEFAULT_MAX_ENTRY_BYTES`/`DEFAULT_MAX_TOTAL_BYTES`, checked before anything is
  inflated), a cap on `word/document.xml` (`DEFAULT_MAX_PART_BYTES`), and a
  character cap (`DEFAULT_MAX_CHARS`, reporting `truncated`). Only the main body
  is read, so **headers, footers, footnotes/endnotes, comments, text boxes, and
  images are deliberately excluded**; tracked deletions (`<w:delText>`) are not
  read. Macros (`word/vbaProject.bin`) and embedded content are never executed,
  and instructions inside the text are inert. Arabic/English are preserved as
  Unicode characters (no visual-order problem, unlike PDF). A non-ZIP file, a ZIP
  without the document part, malformed XML, an OLE/encrypted file, an
  archive-bomb, an over-large part, and a document with no text each raise
  `DocxExtractionError` with a specific reason.
- `core/text_encoding.py` — dependency-free, explainable text decoding for
  imported files. `detect_encoding` honours a BOM, trusts strictly-valid UTF-8
  only when it decodes to plausible text (so NUL-filled UTF-16 is not mistaken
  for UTF-8), then tries a small ordered set of legacy encodings (Windows Arabic
  `cp1256`, `cp1252`, ISO-8859-6/1, KOI8-R, BOM-less UTF-16), each strict and
  plausibility-checked; `decode_bytes` never uses `errors="replace"`. When
  nothing fits it raises `EncodingDetectionError`, so a decode failure is explicit
  and never silently corrupts a large amount of text. `allow_truncated` tolerates
  a bounded stream prefix that ends mid-character.
- `core/text_normalize.py` — one deterministic, pure pass over imported text:
  Unicode NFC, unified line endings, control-char removal, and whitespace
  collapse. It never removes punctuation, diacritics, or Arabic letter
  distinctions; the Arabic diacritic/tatweel removal + letter folding matching
  `core.rag` is the *opt-in* `fold_arabic` (`--fold-arabic`). Idempotent and
  hash-stable; normalization is applied at the ingestion boundary because it
  changes the content hash (so it is opt-out).
- `gui/attachments.py` — the `+` composer button's brain: `attach_document`
  reads a file through the reader, chunks it, and returns an `AttachmentNotice`
  (success + preview, or the reader's error verbatim); the chunks become
  `Document`s under the `attachment:` namespace and are indexed into the
  assistant's own store, so an answer can retrieve and **cite** them like the
  built-in corpus. `ocr=True` reads a scanned PDF through the same opt-in OCR
  path, and a scanned-PDF failure names the real next step (enable OCR, or use
  the Import Center) instead of the reader's CLI-only flag. `remove_documents`
  detaches them again. Kept out of the Tk window so the logic is testable without
  a display.
- `fayd-al-ruh.spec` — PyInstaller; uses `collect_submodules` for `ml`,
  `evaluation`, `ai`, `storage`, `gui` (and `core` is bundled as an imported
  package).

## Conventions

- Concept labels and relations are English (`reservoir`, `is_a`, `measures`,
  `contains`, `drives`, ...); Arabic terms are kept as secondary aliases.
- Seeded confidence is modest and revisable.
- Keep seeds declarative tuples (`CourseKnowledge` / `DisciplineKnowledge`);
  coverage helpers report what an ontology actually carries.
- Add a test per new seed or CLI action; keep the evaluation gate green.

## Jarvis roadmap (delivered)

Layers added on top of the core, each dependency-free and testable on a fresh
clone, each optional so the base system still runs without it:

- Phase 1 — BPE subword tokenizer, conversation memory, RAG documents/retriever,
  grounded assistant with citations and an extractive fallback.
- Phase 2 — speech I/O contracts + voice loop (`core/speech.py`).
- Phase 3 — sandboxed real tools + tool-agent coordination (`core/tools.py`,
  `core/tool_agent.py`).
- Phase 4 — continuous learning from explicit feedback + regression monitor
  (`core/knowledge_update.py`, `core/continuous_learning.py`).
- Phase 5 — optional multimodal perception (`core/vision.py`) and its
  counterpart, optional text-to-image (`core/imagegen.py`): a dependency-free
  PNG renderer with an honest procedural default and a swappable `ImageGenerator`.
- Phase 6 — Jarvis mode: wake-word gate (`core/wake_word.py`), natural-language
  command routing to sandboxed tools (`core/command_router.py`), persistent user
  preferences (`core/preferences.py`), real system speech engines
  (`core/speech.py`), and the unified listen/route/answer/act/speak loop
  (`core/jarvis.py`, `ai/jarvis.py`, `python -m cli jarvis`).
- Phase 7 — conversational Jarvis: one conversation session per run so follow-ups
  keep context (`core/jarvis.py` + `core/conversation.py`'s
  `last_assistant_message`), plus conversation controls — stop / speak / repeat /
  new conversation / help — matched exactly so a real question is never hijacked
  (`core/jarvis_controls.py`), with a mute state that gates synthesis.
- Phase 8 — persistent Jarvis sessions: the loop uses a `ConversationMemory`
  backed by `JsonStore` under the data directory, so a session survives a restart.
  `JarvisMode.save_session` / `resume_session` / `list_sessions` and the `save` /
  `sessions` controls expose it; the CLI adds `--session <id>` (autosaving) and
  `--list-sessions`.
- Phase 9 — learn from explicit feedback in the loop: `JarvisMode` takes an
  optional `ContinuousLearner` and handles the `that's right` (confirm) and
  `that's wrong, it's ...` (correct) controls. Confirmations and corrections feed
  the last exchange to the learner, and the retriever is re-indexed so a corrected
  statement changes the very next answer. The `--no-feedback` flag disables it, and
  a session with no learner refuses honestly instead of recording dead feedback.

Every phase keeps the four hard gates green and never lets the model widen its own
authority: the sandbox, the groundedness gate, the explicit-feedback rule, and the
regression monitor each can refuse a change.
