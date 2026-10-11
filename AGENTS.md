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
  it collapses repeated punctuation/doubled words, capitalises and punctuates.
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
  section labels, centres the answer on the question, and collapses repeats —
  without rewriting the model's words — and is used by `ai/mind.py`.
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
  digital PDFs (**Stage 2A adds `.pdf`**; DOCX and scanned-PDF OCR remain
  deliberately out of scope, and an explicit `.docx` path is refused), read-only
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
  retry never double-counts. Each ingested record also writes text-free
  per-document metadata (source path, filename/extension, content hash, media
  type, chunk ids, ingestion timestamp) through `ingest.metadata`, wired into the
  pipeline/CLI best-effort so a metadata-write failure never loses the ingest.
  CLI:
  `python -m cli ingest add|list-connectors|
  stats|build-index|search|delete|jobs` (`cli/ingest.py`); the two knowledge
  actions mirror into the GUI sidebar.
- `evaluation/` — regression gate. `evaluation/runner.py` assembles metrics,
  `benchmark.py` declares `MetricSpec`s, `baseline.json` is the reference.
  Update the baseline with `python -m evaluation --update-baseline`.
- `cli/` — portable front end. `menu.py` holds actions + `MENU`; `__main__.py`
  dispatches commands and reconfigures stdout/stderr to UTF-8 (Arabic-safe on
  legacy Windows consoles).
- `gui/` — ChatGPT-style desktop window, an alternative *front end* to `cli/`,
  never a replacement. `service.py` reuses the text `chat` session
  (`ai.jarvis.build_session(text_only=True)`) + `ai.chat.ModelSwitcher` and adds
  `list_sessions`/`resume`; `actions.py` maps the sidebar to `cli.menu.run_*` —
  the **Actions** list mirrors numbered CLI items 5..11 (never repeating the
  window's own controls) and `saved_session_items` builds the **Saved chats**
  list; `options.py` has the argument parser; `app.py` is the Tkinter window
  (imported lazily, so `cli/` never imports Tk). Reachable as `python -m cli gui`
  and `python -m gui`.
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
  scrollable sidebar (brand + version, a filled **New chat** button, source
  switch, **Conversations**, **Actions**) behind a scrollbar that appears only
  when the content overflows, a toolbar showing the current answering source,
  and a centred hero/empty state that gives way to the first message (and
  returns when the transcript is cleared). A **＋** button beside the input
  attaches a Word (`.docx`/`.docm`) or text file: the file's text is indexed into
  the source assistant's store and previewed in the transcript, a removable chip
  tracks it, and detaching (the chip's ✕, or a new chat) removes it from the
  store. Each message body is a read-only `tk.Text`, so answers and live training
  output can be selected and copied — per-message Copy button, Ctrl+C copies the
  selection, Ctrl+A selects all, Delete clears, right-click offers copy /
  copy-conversation / select-all / clear; Ctrl+N new chat, Ctrl+S save. A small
  animated "Thinking…" indicator shows while an answer is pending. Live training
  output is queued and rendered at a bounded rate (coalesced into one widget
  update every ~0.15 s, fully drained before the result), so a fast training loop
  never freezes the window.
- `core/document_reader.py` — read text out of a Word file with no third-party
  dependency: a `.docx`/`.docm` is an Office Open XML package (a ZIP holding
  `word/document.xml`), so `zipfile` + `xml.etree` extract its paragraphs
  (joining runs, honouring tabs and breaks) into a `DocumentText`. Legacy `.doc`
  and `.rtf` are refused with a "save as .docx" hint instead of being scrambled.
  Plain `.txt`/`.md` are read with **safe encoding detection** (a BOM, then
  strict UTF-8, then a short list of legacy encodings; a mis-decoded file raises
  rather than filling with U+FFFD). Large files are streamed line by line up to
  `MAX_TEXT_CHARS` with an incrementally-decoded, strictly-checked codec, and
  `DocumentText.truncated` / `DocumentText.encoding` report a capped read and the
  codec honestly. A digital `.pdf` is read **page by page** through
  `core.pdf_extract` (the optional `pypdf` extra); `DocumentText.page_numbers`
  and `DocumentText.scanned` carry the page split and the OCR provenance. A
  scanned PDF with no extractable text falls back to the separate optional OCR
  extra (`[ocr]` = pypdfium2 + pytesseract, engine installed separately), and a
  `NormalizationError` is raised only when a requested normalization fails.
  `read_document(..., normalize=..., encoding=..., password=...)` applies
  `core/text_normalize`, pins the codec, and decrypts a protected PDF. All lazy
  imports; the base install pulls nothing new and no document is ever executed.
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
  built-in corpus. `remove_documents` detaches them again. Kept out of the Tk
  window so the logic is testable without a display.
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
- Phase 5 — optional multimodal perception (`core/vision.py`).
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
