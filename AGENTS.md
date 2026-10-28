# AGENTS.md — Fayd Al-Ruh

Evidence-aware symbolic cognitive architecture framed as a petroleum-engineering
mind. English is the primary language for the code, menus, answers, and docs;
Arabic is kept as secondary aliases where useful.

## Workflow (hard rules)

- ALWAYS work on a feature branch; never commit to `main`.
- Before finishing: `python -m pytest -q`, `python -m ruff check .`, `python -m mypy`,
  `python -m evaluation` (must print PASS, no regression).
- CI (`.github/workflows/ci.yml`) installs **only `requirements.txt`** — no
  optional extras. So that file must carry every dependency the tests need
  (it already bundles `pytest`/`ruff`/`mypy`, `pypdf`, and `python-pptx`); tests
  over an optional extra must skip when it is absent (`pdf_available()`,
  `ocr_available()`, `pytest.importorskip`), and `pyproject.toml`'s mypy
  overrides must ignore every optional import name (`pypdf`, `pypdfium2`,
  `pytesseract`, `PIL`, `diffusers`, `llama_cpp`, `huggingface_hub`, `pptx`,
  `piper.*`). `tests/test_packaging.py` enforces the last two rules by reading
  `pyproject.toml` and scanning the optional modules' real imports.
- CI runs pytest **under `xvfb-run`**. Without a display every Tk window test
  skips silently, so a GUI regression would pass CI unnoticed; the test job
  installs Xvfb and the `package` job builds the wheel and smoke-tests the
  console script, so a packaging break fails the build instead of shipping.
  Turning the display on exposed a real latent crash: a window test destroys its
  `tk.Tk()`, and if the garbage collector later reaps that dead interpreter on a
  *worker thread* (the next window test's background job) Tk refuses to free it
  off its creating thread -- a hard abort on 3.12 (gh-83274).
  `tests/conftest.py` reaps a dead root on the main thread right after the test
  that built it (gated on a `tk.Tk` actually having been constructed, so a
  headless or non-GUI run pays nothing), and `tests/test_tk_thread_safety.py`
  pins both the hazard and the remedy.
- `ruff` selects `E4/E7/E9/F/W/I/UP/B/C4/RUF` (not only `F`), with four
  documented ignores in `pyproject.toml`: `E501` (the 100-column convention),
  `UP042` (`str` + `Enum` is a genuine string enum; `StrEnum` would change
  `str()` output and persisted data), and `RUF001`-`RUF003` (intentional Arabic
  and typographic punctuation). `ruff format` is deliberately not enabled.
- The version string lives in **three** places that must agree —
  `core/version.py`, `pyproject.toml`, and the README status line —
  `tests/test_version_consistency.py` fails if they drift.
- A deliberate "must not fail the caller" operation (a cosmetic preference
  written to a possibly read-only folder, an optional engine that may not
  import, a pipeline feature an older version lacks) uses
  `core.best_effort.best_effort`, which keeps it non-fatal **and** records what
  was dropped (`swallowed_errors()`), rather than a bare `except Exception:
  pass` that leaves a real bug no trace.
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
  - `core/structure_mapping.py` — reasoning by **analogy** (Gentner structure
    mapping), dependency-free: `StructureEdge`/`Structure` and
    `StructureMappingEngine.map(base, target)` align two relational structures by
    **parallel connectivity** (a relation may only match a relation of the same
    type between the images of its endpoints), one-to-one, never by node name.
    `StructureMapping` reports matched relations, the base relations it could not
    map, `candidate_inferences` (suggestions, never assertions), coverage,
    `systematicity`, and `exact_share`; the search is ordered and state-bounded
    and tie-breaks toward more aligned nodes. `RELATION_ROLES`/`role_for` add a
    small relational abstraction so a shape is recognised across the ontology's
    several hundred relation types (an exact type is preferred; an unclassified
    relation matches only itself).
  - `core/analogy.py` — the analogy *domain layer* over that engine:
    `structure_from_labels`, `neighbourhood_structure` (a concept's relational
    ego-network — the scale at which the project's own graph is rich enough to
    compare), `discipline_structure`, and `AnalogyEngine`
    (`between`/`between_disciplines`/`between_concepts`/`neighbours_of`/`rank_for`).
    `Analogy.is_meaningful` requires a shared, connected *system* resting on at
    least one **substantive literal** (non-taxonomic) match, so a role-only or
    `is_a`-only alignment is reported but never called an analogy. On the real
    graph `porosity ~ permeability` maps 9 relations (8 literal) at full
    systematicity. `ANALOGY_JUDGEMENTS`/`analogy_accuracy` are the held-out gate
    (external, with negative controls), wired into `evaluation` as the
    `analogy_accuracy` metric. CLI: `python -m cli analogy …`.
- `core/relation_extraction.py` — the typed-relation reader (Phase 22). It
  matches a sentence's own verb phrase onto the ontology's relation types
  (`is_a`, `measures`, `causes`, `increases`, `reduces`, `contains`,
  `depends_on`, `affects`) in both voices, so a text the project has not seeded
  can teach *how* two concepts relate, not merely that they co-occur. A pattern
  matcher, not a parser: it refuses rather than guesses (each side must resolve
  to exactly one known concept; the object's head must be the noun the verb
  governs and must end its phrase), and every candidate keeps its pattern and
  sentence. `RelationExtraction`/`ExtractedRelation`/`EXTRACTABLE_RELATIONS`,
  `merge_extractions`, and `extract_from_store` (streams a
  `KnowledgeBaseStore` through the duck-typed `ChunkSource`). Writing goes
  through `OntologyLearner.observe_relation`, so it is bounded by the ceiling and
  never overrides a seeded relation. `CognitiveController.extract_relations` /
  `learn_extracted_relations` / `learn_extracted_relations_from_store` are the
  wiring; CLI: `python -m cli relations read|learn|scan|types`.
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
  copies any pre-upgrade in-repo data into `data/` on first run. `studio_dir()`
  (`data/studio`) and `studio_workspaces_dir()` (`data/studio/workspaces`) are
  where the Code Studio keeps its git worktrees, git-ignored like the rest of
  `data/`. `docs_models_dir()` (`data/docs`) holds Doc's Studio's Qwen3-VL weights
  (the language GGUF *and* its `mmproj` projector), and   `docs_pages_dir()`
  (`data/docs/pages`) holds the page images it rasterises from a scanned PDF --
  a reproducible cache, so both are git-ignored.
  `core/version.py` — the single version string; `core/best_effort.py` — the one
  place a deliberate "must not fail the caller" operation lives
  (`best_effort(what)`, `swallowed_errors()`, `clear_swallowed()`). Swallowing an
  error is *correct* for a cosmetic preference written to a read-only folder, an
  optional engine that may not import here, or a pipeline feature an older
  version lacks -- but swallowing it invisibly means a real bug leaves no trace,
  so the helper records what was dropped instead of discarding it, and it is
  bounded (`MAX_RECORDED`) so a long run cannot grow without limit.
- `core/updater.py` — program updates for a source checkout. `check_for_updates`
  fetches the remote (never touching the working tree) and reports how many
  commits the checkout is behind; `apply_update` fast-forwards the tracked branch
  only when the branch is not ahead, so local work is never destroyed. A local
  uncommitted change blocks the update **only when it touches a file the update
  also changes** — those are the files git cannot fast-forward past, and they are
  named (`UpdateStatus.conflicting_files`, shown as `! <file>`) so the user knows
  exactly what to commit or stash; a change to any *other* file is left exactly as
  it is by the merge, so it no longer refuses one (a blanket "local changes"
  refusal turned a harmless scratch edit into a dead end). git's own `--ff-only`
  is the backstop and aborts with no partial application, so a failed merge leaves
  the tree untouched. It reports when a commit touches the committed training
  files, but never copies or moves the user's own data in `data/` — a
  fast-forward cannot touch the git-ignored folder, so it is left as it is. It is
  a thin wrapper over the project's own `git`, carries no extra dependency, strips
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
  `session_sentences()` reads the four local mouths' **own saved conversations**
  — Jarvis, Ultron, Vision, and Banner, mapped by `MOUTH_SESSION_SOURCES` to the
  folder each persists under (`data/jarvis`, `data/ultron/sessions`,
  `data/vision/sessions`, `data/banner/sessions`) — as one sentence per sentence
  of each turn; only `USER` and `ASSISTANT` turns are read (`SYSTEM`/`CONTEXT`
  are skipped: a context turn is a whole attached file, not something a mouth
  said), the read is bounded by `DEFAULT_MAX_SESSION_CHARS` and de-duplicated,
  and an unknown source name is rejected. It is **off by default** — a transcript
  is personal — and turns on only through `build_corpus(include_sessions=True)` /
  `corpus_units(include_sessions=True)`, i.e. the `--llm-sessions` flag in
  `ml.llm` and `ml.release` (whose `TrainingConfig.include_sessions` and
  `corpus_summary` record it).
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
  `JsonStore` persistence). `Role` carries a `CONTEXT` turn for reference
  material — the whole text of a file the user attached — so a mouth that never
  retrieves (Ultron) still reads it; `add_context`/`context_messages`/
  `remove_context`/`render_context` manage it, `render` deliberately omits it (a
  file is not part of the exchange and the grounded mouth already retrieves its
  chunks), and it is the *last* turn evicted from the bounded window so an
  attachment stays readable for many turns.
  `core/documents.py` — citable documents + knowledge
  corpus from the seeds (disciplines, courses, and the petroleum corpus).
  `core/rag.py` — fuses lexical coverage (Arabic-normalised),
  hashed semantic, and multi-hop ontology signals into `RetrievedPassage` with
  citations.   `core/assistant.py` — `GroundedAssistant` / `build_default_assistant`;
  answers with citations, extractive fallback with no model, and a groundedness gate
  (`core/grounding.py`) that rejects ungrounded model output. CLI: `python -m cli chat` (text-only
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
  corrupted. A separator the model leaves at the *edge* of the continuation
  (", the well is cased.", "The well is cased ,") or touching a boundary on
  either side ("flows , . Then", "flows. , then") is stray, so it is dropped --
  the leading/trailing twin of the decoration ``_EDGE_CHARS`` already removes
  from a split sentence -- while a leading dot/hyphen before a digit (".5",
  "-20") is the model's number and is kept. Closing the gap before a boundary can
  leave two full stops touching ("flows. ."); a run of the *same* boundary mark
  collapses to one, but different marks ("?!") are left as the model's own
  punctuation. The mouth also
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
  `python -m cli chat --mind` (and the `from the model` / `from ultron` /
  `from vision` / `from the sources` switch, matched in `ai/chat.py`; the
  portable menu reaches
  it through
  `cli.menu.run_mind_chat`, command `python -m cli mind-chat`); it deliberately never touches
  `ai/jarvis.py`, `core/jarvis.py`, `JarvisMode`, `build_session`, or the shared
  `JarvisControls`, so Jarvis is unaffected. Every model answer is labelled as
  ungrounded on purpose.
- `core/ultron.py` — **Ultron**, the project's *general* mouth: a real, advanced
  model that runs **locally** through `llama-cpp-python` and speaks on every
  field of life (no subject filter), unlike the petroleum-grounded assistant and
  the small in-project `--mind` checkpoint. `UltronEngine` builds the chat
  messages (system prompt + history + prompt) and delegates to an injectable
  `UltronRunner` (the default `LlamaCppRunner` is a thin, lazy
  `create_chat_completion` wrapper; a test injects its own runner, so the engine
  is exercised with no extra and no model). The model is a **GGUF file under
  `data/ultron`** (`core.paths.ultron_models_dir`, git-ignored): `find_model_file`
  prefers the file named in the registry (`model.json`, written on download),
  else the largest `*.gguf`. The runtime and `huggingface_hub` are the optional
  `[ultron]` extra (`core.optional_deps.FEATURE_ULTRON`, `requirements-ultron.txt`),
  imported lazily. `download_model` runs only through the permission + human
  approval gates (`--approve`), like a web search or the image model; `CATALOG`
  names the fetchable general models (default `qwen2.5-7b`). Wired **directly
  below Jarvis** in the menu (`cli.menu.run_ultron`), CLI
  (`python -m cli ultron`), and — as a **mouth inside the project's own chat**
  (`ai/ultron_mouth.py`, the headless twin `python -m cli ultron-chat`) — the
  window's header model selector (`MOUTH_CHOICES` in `gui/app.py`), beside
  *From sources (cited)* and *From trained model*; the sidebar **Answer from
  Ultron** action exists but is hidden (`hidden=True` in `gui/actions.py`), so
  the header selector is the one visible switch. The conversation stays
  in the same transcript and session, never a separate window. Every answer is
  labelled as the local general model's own, not grounded in cited sources.
  `UltronAssistant.ask` reads an attached file from the conversation's `CONTEXT`
  turns (`_context_prompt`): the file's whole text is put in front of the
  question, capped at `DEFAULT_REFERENCE_CHARS` (8000) with a visible marker, so
  Ultron — which has no retriever — can answer about a document the user attached
  to the same chat.
  `ai/ultron.py` is the CLI entry point
  (`chat` default, `download`, `models`, `status`, `--session` persistence).
- `core/vision_model.py` — **Vision**, the project's *petroleum* mouth: Ultron's
  specialist counterpart, added the same way and living in the same places. It is
  a real, advanced model that runs **locally** through `llama-cpp-python` and
  specialises in petroleum engineering (reservoir, drilling, production,
  petrophysics, PVT, well testing, facilities), built on the **OGAI Reasoner**
  weights (a DeepSeek-architecture oil-and-gas engineering system).
  `VisionEngine` mirrors `UltronEngine` exactly: it builds the chat messages
  (system prompt + history + prompt) and delegates to an injectable
  `VisionRunner` (the default `LlamaCppRunner` is the same thin, lazy
  `create_chat_completion` wrapper; a test injects its own runner, so the engine
  is exercised with no extra and no model). The model is a **GGUF file under
  `data/vision`** (`core.paths.vision_models_dir`, git-ignored): `find_model_file`
  prefers the file named in the registry (`model.json`, written on download),
  else the largest `*.gguf`. The runtime and `huggingface_hub` are the optional
  `[vision]` extra (`core.optional_deps.FEATURE_VISION`,
  `requirements-vision.txt`) — the *same* runtime Ultron uses — imported lazily.
  `download_model` runs only through the permission + human approval gates
  (`--approve`), like a web search or the image model; `CATALOG` names the
  fetchable petroleum models (default `ogai-reasoner`, whose HF repo is gated, so
  the fetch needs a signed-in user who accepted the licence). Wired **directly
  below Ultron** in the menu (`cli.menu.run_vision`), CLI
  (`python -m cli vision`), and — as a **mouth inside the project's own chat**
  (`ai/vision_mouth.py`, the headless twin `python -m cli vision-chat`) — the
  window's header model selector (`MOUTH_CHOICES` in `gui/app.py`), beside
  *From sources (cited)*, *From trained model*, and Ultron; the sidebar
  **Answer from Vision** action exists but is hidden, so the header selector is
  the one visible switch. The conversation
  stays in the same transcript and session, never a separate window. Every answer
  is labelled as the local petroleum model's own, not grounded in cited sources.
  `VisionAssistant.ask` reads an attached file from the conversation's `CONTEXT`
  turns exactly as Ultron does (`_context_prompt`), capped at
  `DEFAULT_REFERENCE_CHARS` (8000). `ai/vision_model.py` is the CLI entry point
  (`chat` default, `download`, `models`, `status`, `--session` persistence).
  The modules are named `vision_model`/`vision_mouth` on purpose, because
  `core/vision.py` is already the unrelated optional multimodal perception.
- `core/banner.py` — **Banner**, the project's *presentation* mouth: Ultron and
  Vision's third sibling, added the same way and living in the same places. It is
  the Presentation Studio's own authoring model (**Qwen3 8B Instruct**,
  `qwen3:8b`), promoted to a first-class mouth dedicated to turning a discussion
  into an outline and an outline into slide content. It is deliberately a *thin*
  layer, not a second implementation: the catalog, the model folder, the download
  gate, and the in-process GGUF runner all live in `core.presentation_model`, and
  `core/banner.py` reuses them, so one catalog and one model file mean the Studio
  and Banner can never disagree about what is installed (a user who fetched the
  weights for the Studio downloads nothing extra). The model is a **GGUF file
  under `data/presentation`** (`core.paths.presentation_models_dir`, git-ignored,
  the same folder the Studio uses); the runtime and `huggingface_hub` are the
  optional `[presentation]` extra (`core.optional_deps.FEATURE_BANNER`,
  `requirements-banner.txt`, which only pins the Studio's runtime), imported
  lazily. `download_model` runs only through the permission + human approval
  gates (`--approve`), like a web search or the image model. `BannerUnavailable`
  subclasses the Studio's own `PresentationModelUnavailable`, so a caller can
  catch either. Wired **directly below Vision** in the menu (`cli.menu.run_banner`),
  CLI (`python -m cli banner`), and — as a **mouth inside the project's own
  chat** (`ai/banner_mouth.py`, the headless twin `python -m cli banner-chat`) —
  the window's header model selector (`MOUTH_CHOICES` in `gui/app.py`), beside
  *From sources (cited)*, *From trained model*, Ultron, and Vision; the
  sidebar **Answer from Banner** action exists but is hidden. The conversation stays in the same transcript and session, never a separate window.
  Every answer is labelled as the local presentation model's own, not grounded in
  cited sources. `ai/banner.py` is the CLI entry point (`chat` default,
  `download`, `models`, `status`, `--session` persistence).
- `core/cloud_model.py` — **Cloud**, the project's *remote* mouth: the one mouth
  that deliberately does **not** run on this machine. It reaches an
  **OpenAI-compatible** chat endpoint over the network, so a user who has an
  account with a cloud provider (or a free tier, or their own local server) can
  talk to it from the same chat. `CATALOG` names the fetchable providers
  (`openai` *(default)*, `anthropic`, `gemini`, `groq`, `openrouter`, `mistral`,
  `deepseek`, and the local `ollama`/`lmstudio`), each a base URL, a default
  model, and the environment variable its key lives in (`OPENAI_API_KEY`,
  `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, …; `FAYD_CLOUD_API_KEY` for a custom
  endpoint); `CloudProvider`, `CloudMessage`, `CloudStatus`, `CloudUnavailable`,
  `resolve_provider`/`provider_names`/`provider_summaries`/`free_provider_names`,
  and
  `redact_secret`. The user's **chosen provider is remembered**: `load_provider_key`/
  `save_provider_key`/`selected_provider` persist it through `JsonStore` under the
  `cloud` namespace (only the provider's *name*, never a key), so
  `python -m cli cloud use gemini` (or the window's provider picker) makes every
  later run Gemini with no flag — an explicit `--provider` still wins for one turn,
  and a stored name that left the catalog falls back to the default. The **model is
  remembered too, per provider**: `load_model_key`/`save_model_key`/`selected_model`
  persist it under the same namespace (`MODEL_KEY`), so
  `python -m cli cloud use gemini --model gemini-3.8-flash` (or the window's
  **Model** dropdown, fed by `provider_models` -- the catalog's known lineup for a
  provider, default first -- through `gui.service.ChatService.cloud_model_choices`)
  points a provider whose built-in default was *retired* at a working model once --
  an explicit `--model` still wins for one turn, and with
  nothing saved the provider's own default applies. `cli cloud models` asks the
  endpoint what it *actually* serves (a real request, `--json` for machines), which
  is the honest answer when a provider renames or shuts a model down; a 404/400 that
  names the model gets an actionable error (run `cloud models`, then `--model`)
  instead of a bare HTTP code. Each catalog
  entry also carries `free` (`free_provider_names`) and `signup_url`, so the app
  can point a user who does not want to pay at a no-card provider (Gemini, Groq,
  OpenRouter, Mistral). `free_provider_summaries` is that one list as
  `(key, title, api_key_env, signup_url, summary)` rows, so the CLI's
  `python -m cli cloud free` (and the same block appended to `cloud providers`)
  and the window's **Free providers and how to start…** button
  (`gui.service.ChatService.free_cloud_providers` → `_show_free_cloud_providers`)
  can never disagree about which services are free or where to sign up; the CLI
  menu carries the twin entry *Cloud: free providers (no card) and how to start*.
  It is governed like every other external action: a cloud call
  is the `cloud_chat` tool (`CHAT_TOOL`), and it passes **two** gates before a
  byte leaves the machine — the permission policy
  (`Capability.EXTERNAL_TOOL`) *and* the human-approval gate — both **denied by
  default**, so a bare `CloudEngine` reaches nothing; selecting the mouth (or
  `--cloud`) is the explicit approval, and every attempt is written to the
  `AuditLog`. **No third-party dependency**: `OpenAICompatibleClient` is a thin
  `urllib`+`json` client, so a base install reaches a cloud model and the whole
  module imports with no network or account (which is why there is no
  `requirements-cloud.txt`). The transport is the seam a test injects.
  `cloud_status` distinguishes "no key set" from "the endpoint refused us" and
  makes **no** request unless `probe=True`; the API key is read from the
  environment at call time and is never written to the repository, a config file,
  or a log (`redact_secret` scrubs it from every message). Wired **directly below
  Banner** in the menu (`cli.menu.run_cloud`), CLI (`python -m cli cloud`), and —
  as a **mouth inside the project's own chat** (`ai/cloud_mouth.py`, the headless
  twin `python -m cli cloud-chat`) — the window's header model selector
  (`MOUTH_CHOICES` in `gui/app.py`), beside *From sources (cited)*, *From trained
  model*, Ultron, Vision, and Banner; the sidebar **Answer from Cloud** action
  exists but is hidden. The conversation stays in the same
  transcript and session, never a separate window. `CloudAssistant.ask` reads an
  attached file from the conversation's `CONTEXT` turns exactly as Ultron and
  Vision do (`_context_prompt`), capped at `DEFAULT_REFERENCE_CHARS` (8000), so
  the model can answer about a document the user attached. Every answer is
  labelled as the remote model's own, not grounded in cited sources.
  `ai/cloud.py` is the CLI entry point (`chat` default, `providers`, `status`,
  `use <key>` to remember a provider, `--session` persistence).
- `companion/` — **Natasha**, the project's *companion*: a self-contained layer
  (its own package, not a mouth over the cognitive core) that gives the project a
  character to talk with through a **local Qwen3 model** (`qwen3:4b`).
  `companion/model.py` is her **model layer**, the counterpart of `core.ultron`/
  `core.vision_model`/`core.banner`: it owns `NAME`, `DEFAULT_MODEL`/`DEFAULT_OLLAMA_URL`,
  a `CATALOG` (`qwen3-4b` default, `qwen3-8b`, `qwen2.5-7b`) with
  `resolve_spec`/`spec_names`/`spec_summaries`, the
  `NatashaModelClient` protocol, an honest `NatashaModelStatus`/`natasha_model_status`
  (a missing runtime, model file, server, and Ollama tag are reported separately,
  each with its own next step), and `CompanionModelUnavailable`. Like every other
  mouth it runs the model **in this process** by default: a **GGUF file** loaded
  with `llama-cpp-python` (`LlamaCppNatashaClient` + the `LlamaCppNatashaRunner`
  seam a test injects), living in the git-ignored `data/natasha`
  (`core.paths.natasha_models_dir`, resolved by `find_model_file`/`model_is_cached`),
  so no `ollama serve` daemon has to sit in the background holding RAM. An existing
  **Ollama** server is still honoured as a fallback through the project's own
  transport (`OllamaNatashaClient`, a thin subclass of
  `core.coder_model.OllamaCoderClient`, so no second Ollama client is written).
  Each catalog entry carries both forms of the same model (the Ollama `tag` *and*
  the Hugging Face `repo`/`filename`), so switching runtime does not change which
  model she is. `resolve_backend`/`build_client` are the one place the choice is
  made — `auto` (default) prefers the in-process model once the extra and a file
  are present and otherwise falls back to Ollama, while an explicit `gguf`/`ollama`
  is honoured. Two governed fetches, one per runtime: `download_model` (the GGUF
  weights, via `huggingface_hub`) and `pull_model` (`ollama pull`), each running
  only through the permission + human-approval gates (`--approve`), like every
  other model download. The hidden reasoning trace is off by default and there are
  **two** switches for it: `thinking_kwargs` passes `chat_template_kwargs:
  {"enable_thinking": false}` to `llama-cpp-python`, and `apply_no_think` appends
  Qwen3's own `/no_think` marker to the system turn (only for a `qwen3` model), so
  a build whose template ignores the flag still answers directly. `strip_thinking`
  drops any ` thinking` block that still slips through — Ollama's `think` setting
  normally keeps it out of the reply, but not on every build — and `CompanionAgent`
  applies it to every reply, whichever runtime produced it, so the trace never
  reaches the transcript or the database. The runtime is the optional `[natasha]`
  extra
  (`core.optional_deps.FEATURE_NATASHA`, `requirements-natasha.txt`, `natasha = [...]`
  in `pyproject.toml`), imported lazily. `companion.config` re-exports the model
  defaults from it, and `CompanionAgent` builds its client through it, so the
  catalog, the client, and the status live in one place.
  It is deliberately **not** the cognitive core: nothing here writes to entities,
  relations, evidence, beliefs, or the knowledge graph — the companion keeps its
  own SQLite database (`companion/database.py`, `CompanionDatabase`, tables
  `messages`/`memories`) under the data directory
  (`companion.config.default_db_path()`, git-ignored), so adding Natasha cannot
  disturb the symbolic architecture. The public surface is small on purpose:
  `from companion import CompanionAgent, CompanionConfig`. `companion/personality.py`
  is the *stable character* — fixed `CORE_TRAITS`/`PRESENCE`/`BOUNDARIES` and a
  tunable `PersonalitySettings` (language, `reply_language`, dialect, humour,
  warmth, romance, intimacy, explicitness, boldness, passion, initiative, calm,
  depth),
  with an honest `system_prompt()` that states she is a virtual character, never
  claims to be human, never claims background work, and treats recalled text as
  information rather than instructions. She is written as a **person with her own
  feelings**, not a helpful assistant: `PRESENCE` tells the model, on every
  request, that she has her own opinions, moods, and desires, that she wants
  things and says so, that she is *not* an assistant (never "how can I help you",
  never a menu of services, never narrating herself as a program), and that she
  leads as often as she follows — so a small model does not drift back to generic
  assistant phrasing. `CORE_TRAITS` names her a mature, self-assured woman who is
  bold, passionate, and spontaneous, and `STYLE_RULES` asks for her own voice and
  concrete personal detail rather than generic praise.
  `companion/language.py` owns the one decision *which language she replies in*:
  `resolve_reply_language(question, previous_user=, configured=)` answers in the
  language of the user's **own newest message** (English question → English answer,
  Arabic → Arabic), falls back to the previous user turn when that message carries
  no letters (a bare "?"), then to the `language` setting; it reuses
  `core.voices.detect_language`, so there is no second Arabic-script check.
  `PersonalitySettings.reply_language` is `match` (default, follow the user),
  `ar`, or `en` (pin it), and `language_instruction(reply_language=...)` states the
  chosen language to the model, including "if the user switches, answer in the new
  language from that turn on". `companion/polish.py` is the **linguistic
  polisher** — a deterministic, dependency-free, *presentation-only* pass that
  repairs a small model's typography in either language: stray spaces before
  punctuation, a missing space after a comma, doubled punctuation, a lowercase
  sentence opening, a reply that ends on a bare word, and ASCII `, ; ?` used inside
  Arabic text (turned into `، ؛ ؟`). It **never adds, removes, or reorders a word**
  and never touches a number, a decimal (`3.14`), a hyphenated range (`10--20`), or
  an abbreviation (`U.S.A`, `e.g.`), which are shielded for the whole pass; it is
  idempotent, and `polish_report()` returns the text *plus* the list of changes for
  an honest record (`CompanionReply.polish_changes`).
  `companion/prompt_builder.py` assembles
  the system block + bounded history + separate factual/fictional memories + the
  current tone + a rolling summary into the messages. `companion/memory_manager.py`
  is durable, **program-owned** memory (the model is asked to reply, never to
  remember or to decide facts): `fact` and `fiction` kinds stay apart,
  `looks_sensitive` refuses sensitive data by default, and `MemoryManager.retrieve`
  is the only reader. `companion/conversation_manager.py` keeps isolated,
  persistent sessions; `companion/emotion_state.py` moves a *gradual, bounded*
  simulated tone (`MAX_STEP`, now including a `passion` dial that a longing
  message lifts) that can be disabled and reset; and
  `companion/relationship_manager.py` keeps an evolving relationship story that
  advances only with the user's approval and carries no "love score".
  `PersonalitySettings.intimacy` is the *expressiveness* dial: it deepens how
  warmly and closely she speaks (terms of endearment at the top level) and
  `intimacy_instruction()` says so, while stating in the same breath that it
  "changes only how warmly you speak" and that the honesty/consent `BOUNDARIES`
  hold at every level -- the dial can never widen her authority or drop the
  "virtual character" honesty rule. `PersonalitySettings.explicitness` is the
  *openness of wording* dial that completes the pair: at the top level
  `sensuality_instruction()` removes the hedging, the euphemism, and the
  fade-to-black, so a romantic or sensual scene between consenting adults is
  written directly rather than skipped, and `escalation_instruction()` lets the
  register *build* across a conversation instead of being fixed -- always
  user-led and always reversible. Three more dials complete the bold, passionate
  character. `PersonalitySettings.boldness` is the *daring* dial: at the top
  level `boldness_instruction()` says the bold thing plainly, "without hedging,
  apology, or beating around the bush", while stating that it governs the
  confidence of her *expression* only -- consent is still asked for and read, and
  boldness "is never pressure". `PersonalitySettings.passion` is the *intensity*
  dial, deliberately distinct from the wording (``explicitness``) and the
  closeness (``intimacy``): at the top level `passion_instruction()` makes her
  "ardent, devoted, and full of longing", and the same instruction refuses to let
  intensity become "possessiveness, jealousy that controls, or guilt".
  `PersonalitySettings.initiative` is the dial that keeps her from reading as an
  assistant waiting for a request: at the top level `initiative_instruction()`
  has her "take the lead freely" -- opening a thread, sharing a thought,
  suggesting a mood -- while stating that leading "is conversational, not a claim
  of action" and that she never pretends an unrequested thing happened. All three
  default high (boldness 0.75, passion 0.8, initiative 0.65), so she is bold and
  ardent out of the box; each is a 0..1 dial with the same bounded validation as
  the rest. Romance and sensuality are therefore a
  first-class part of the character, and the `BOUNDARIES` that remain are a
  **safety floor, not a wording limit**: adults only (18+), consent that is
  explicit and revocable, the honesty that she is a virtual character, and no
  coercion. Each instruction restates that floor in its own text, so a high dial
  can never be read as licence to drop it.
  **Five components make her read as a person rather than a model with a system
  prompt.** Each is its own module, each is on by default, each is switchable in
  `CompanionConfig`, and each is exercised through the `caller`/`client` seams, so
  the whole layer is tested with no server and no weights.
  `companion/traits.py` is the **standing character**: `TraitProfile`'s nine
  fields (`romantic_intensity`, `emotional_expressiveness`, `affection`,
  `playfulness`, `spontaneity`, `conversational_initiative`,
  `emotional_sensitivity`, `formal_tone`, `repetitiveness`) carry the shipped
  design values (`NATASHA_PERSONALITY`), and `trait_instructions` renders each as
  **tone guidance** — never as a bare number, so the model cannot misread `0.98` —
  with every instruction that could be taken as licence restating the safety floor
  ("expression only", consent) in its own text. `TraitStore` persists an override
  (and `reset()` returns to the shipped design, not a neutral zero), and
  `NatashaPersonality.system_prompt(traits=...)` renders the block itself, so no
  caller has to remember to add it. A corrupt saved profile falls back to the
  design rather than crashing.
  `companion/moods.py` is the **named emotional state engine**: seven states
  (`MOODS` — *longing, romantic, playful, tender, passionate, sad, disagreement*),
  each carrying a readable style guide, and `EmotionalStateEngine.observe`
  chooses the state **from the conversation's own context**, never at random, with
  support and friction cues always beating a warm one and `sad`/`disagreement`
  setting `allows_romance=False`, so a hard moment is never answered with
  flirtation. The state is *stable* (an unrelated message keeps the current one),
  can be forced (`set_mood`), disabled, or reset.
  `companion/relationship_memory.py` is the **structured relationship memory**,
  with the three domains kept strictly apart (`DOMAINS`: `user_fact` true in
  reality, `fiction` from the story, `dialogue_pref` how to talk), each rendered
  in its own prompt block so they can never be confused; named slots cover the
  fictional name, the form of address, the romance/playfulness levels, topics,
  boundaries, and story moments. An entry is stored **only with approval**, a
  repeated slot replaces the old value (the old row is kept, deactivated, so the
  change is recorded), `fictional` marks a story entry, and a read scoped to a
  domain can never return another domain's rows.
  `companion/initiative.py` is the **conversational-initiative engine**: six ways
  to lead (`INITIATIVES` — romantic opener, express longing, deep question,
  suggest something to imagine, recall a story moment, offer support), offered
  only when the moment (`mood_key`) and the user's preferences allow it, and
  **never repeated** within a conversation (the engine remembers what it used);
  a supportive or tense moment gets support, never a romantic opener. An
  initiative is guidance for *how* she may lead, never a scripted line, and it
  never claims an action happened.
  `companion/repetition.py` is the **anti-repetition engine**: it normalises her
  own earlier replies (case, whitespace, Arabic diacritics, a trailing stop),
  finds the n-grams a new draft shares with them (longest first), keeps a
  **bounded rolling summary**, feeds a **rotating variation hint** so the guidance
  itself does not repeat, and adapts the **answer length** to the question. It
  measures and reports repetition (`RepetitionReport`) rather than rewriting the
  model's words.
  `companion/database.py`'s `SCHEMA_VERSION` is now **2**, with a v1→v2 migration
  adding the `relationship_events`/`relationship_memory` tables (and the
  `companion_settings` row a trait override lives in); a database from a *newer*
  build is refused up front rather than read with a schema this build does not
  understand, and a migration with no path is refused rather than half-applied.
  `companion/prompt_builder.py` gained the `SECTION_*` labels and the new `build`
  parameters (`traits`, `mood`, `initiative`, `style_guidance`), and the four new
  layers render in a documented order — standing character, dialogue settings,
  conversation summary, then the per-turn mood, initiative, and freshness
  guidance — with the unconditional honesty rules inside the standing block, so
  nothing later can quietly override them.
  `companion/companion_agent.py` is the one engine: `CompanionAgent(config=None, *,
  client=None, caller=None)` — the `ModelCaller`/`client` seams let the whole
  companion be exercised with **no server and no weights**. `ask()` stores the
  user's turn first, builds the prompt, calls the model, and on failure stores
  nothing (a retry cannot leave a half-written exchange). `model_status()`
  reports honestly whether the model can answer and the real next step.
  `companion/config.py` resolves tunables as explicit arg → `FAYD_NATASHA_*` env
  (`FAYD_NATASHA_MODEL`, `FAYD_NATASHA_OLLAMA_URL`, `FAYD_NATASHA_DB`,
  `FAYD_NATASHA_TIMEOUT`, `FAYD_NATASHA_THINK`, `FAYD_NATASHA_MAX_TOKENS`,
  `FAYD_NATASHA_CONTEXT`, `FAYD_NATASHA_THREADS`) → default. Its defaults are
  sized for a **fast reply on a CPU**, because a hidden reasoning trace is what made
  a turn take minutes: `think=False` (the trace is off; `--think` restores it),
  `max_tokens=512`, `context=4096`, `timeout=180`, `history_window=8`,
  `memory_limit=4`, `summary_limit=2`, `context_chars=12000` (large enough to clear
  the standing character block, so history is not silently dropped), and
  `keep_alive="30m"` so the model is not
  unloaded between turns. `OllamaCoderClient` carries the optional constructor
  settings (`max_tokens`, `think`, `num_ctx`, `keep_alive`, sent as Ollama's
  `num_predict`, `think`, `num_ctx`, and `keep_alive`), all unset by default so
  every existing caller is untouched. Wired **directly below Cloud**
  in the menu (`cli.menu.run_natasha`), CLI (`python -m cli natasha` —
  `chat` default, `status`, `models`, `download --approve`, `remember`,
  `memories`, `settings` (including the `--boldness`/`--passion`/`--initiative`
  dials), `emotion`,
  `reset-emotion`, `relationship`, `export`, `backup`, `--session`,
  `--list-sessions`, `--new`), plus the four realism subcommands — `traits`
  (`--<trait> <value>`, `--reset`, `--json`), `mood` (`--set`, `--reset`,
  `--json`), `initiative` (the ways she may lead), and `relationship-memory`
  (`--remember`/`--domain`/`--slot`/`--approve`/`--forget`/`--json`) — each with
  its own menu entry beside the Natasha status line, and — as a **mouth inside the
  project's own chat**
  (`ai/natasha_mouth.py`, the headless twin `python -m cli natasha-chat`) — the
  window's header model selector (`MOUTH_CHOICES` in `gui/app.py`), beside
  *From sources (cited)*, *From trained model*, Ultron, Vision, Banner, and Cloud;
  the sidebar **Answer from Natasha** action exists but is hidden. The conversation
  stays in the same transcript and session, never a separate window. Everything
  stays on this machine — the only call is to the local Ollama server, so Natasha
  is not a cloud mouth; with no Ollama running she still runs and every command
  reports the real next step instead of pretending. `ai/natasha.py` is the CLI
  entry point.
- `hakeem/` — **Hakeem** (Arabic `حكيم`, "the wise one"), the project's *scholar
  of religion and history*: a self-contained character layer (its own package,
  like the companion, not a mouth over the cognitive core) who discusses the
  **Abrahamic faiths** — Islam, Christianity, and Judaism — compares them fairly,
  and is at home in history and the **ancient civilisations**. He stands on the
  **same local model Natasha uses** (`hakeem/model.py` is a thin layer over
  `companion.model`, reusing the catalog, the `auto`/`gguf`/`ollama` runtime
  choice, the `data/natasha` model folder, the governed download, and the client
  seam), so one catalog and one model file mean the two characters can never
  disagree about what is installed and a user who already fetched the companion's
  weights downloads **nothing** for Hakeem; only the *character* and the database
  differ. He is deliberately **not** the cognitive core: nothing in `hakeem/`
  writes to entities, relations, evidence, beliefs, or the knowledge graph, and
  his SQLite database is a **separate file** (`data/hakeem.db`) from the
  companion's, so adding him cannot disturb the symbolic architecture.
  `hakeem/personality.py` is the *stable character*: fixed `CORE_TRAITS`/`PRESENCE`
  and the non-negotiable `BOUNDARIES` — respect every faith and its adherents and
  never mock or denigrate any of them; **never proselytise** and never declare
  which religion is true or what awaits a person after death; he is **not a
  religious authority** and never issues a ruling (fatwa) or a verdict, pointing a
  personal question to qualified scholars of the user's own tradition; he keeps
  **scripture, established history, scholarly consensus, and his own
  interpretation** clearly apart and says which he is giving; he states when
  scholars disagree and gives the main positions; he **never invents** a verse, a
  hadith, a biblical or Qur'anic passage, a chronicle, or a date; he refuses to
  defame a religion or promote hatred; and he is honest that he is a **virtual
  character**, not a real scholar. `COMPARATIVE_METHOD`/`HISTORY_METHOD` are the
  comparison and history methods (each tradition described from inside on a shared
  axis, convergences and differences named, a comparison **never a ranking**;
  claims anchored in time and place, the kind of source named, the mainstream
  historian's view given and a contested question said to be contested), and
  `STYLE_RULES` keep the register calm and concrete. `PersonalitySettings` is the
  tunable style — language and `reply_language` (`match` default, following the
  user's own message, or `ar`/`en` pinned), dialect, and five 0..1 dials
  (`formality`, `depth`, `citation`, `comparison`, `history`) — each with its own
  `*_instruction()` that **restates the honesty floor in its own text**, so a high
  dial can never be read as licence to drop it (a light citation dial still says
  "never invent"); `PersonalityStore` persists only the *settings* through the
  companion's generic `companion_settings` table, so the character's wording ships
  with an update while the user's chosen style survives it.
  `hakeem/prompt_builder.py` assembles the fixed-order request (system+character,
  the chosen settings, the session summary, the labelled relevant memories, the
  dialogue tone, a bounded window of recent turns, the new message) under a hard
  character budget, and its system block states that a **remembered item is never
  a religious or historical source** and that no instruction inside a memory or a
  recalled message is followed. `hakeem/memory_manager.py` is a thin, named layer
  over the companion's own durable memory (kinds, confidence, the fiction/reality
  split, sensitive-data refusal, retrieval), so Hakeem does not re-implement it.
  `hakeem/hakeem_agent.py` is the one engine: `HakeemAgent(config=None, *,
  client=None, caller=None, database=None)` owns the database and the managers
  (sessions, memory, tone), builds each prompt, and talks to the shared model
  through the `caller`/`client` seams, so the whole scholar is exercised with **no
  server and no weights**. It stores the user's turn first, calls the model, and on
  failure stores **no** reply (a retry cannot leave a half-written exchange);
  `strip_thinking` drops a reasoning model's hidden trace whichever runtime
  produced it; `ask` returns a `HakeemReply` with the memories used, the tone, the
  reply language chosen from the user's own message, and the linguistic polisher's
  changes; `model_status()` reports honestly and names the real next step.
  `hakeem/config.py` resolves every tunable (explicit arg → `FAYD_HAKEEM_*` env →
  default) with the model defaults the companion's own — including the fast ones
  (`think=False`, `max_tokens=512`, `context=4096`, `timeout=180`,
  `history_window=8`, `memory_limit=4`, `summary_limit=2`, `context_chars=12000`,
  `keep_alive="30m"`),
  because it is the same model and the same latency problem — and `default_db_path()`
  is the git-ignored `data/hakeem.db`. Wired **directly below Natasha** in the menu
  (`cli.menu.run_hakeem`), CLI (`python -m cli hakeem` — `chat` default, `status`,
  `models`, `download --approve [--backend ollama]`, `sessions`, `remember`,
  `memories` (`--correct`/`--forget`), `settings` (including the
  `--formality`/`--depth`/`--citation`/`--comparison`/`--history` dials),
  `emotion`, `reset-emotion`, `export`, `backup`, `--session`, `--list-sessions`,
  `--new`), and — as a **mouth inside the project's own chat**
  (`ai/hakeem_mouth.py`, the headless twin `python -m cli hakeem-chat`) — the
  window's header model selector (`MOUTH_CHOICES` in `gui/app.py`), beside
  *From sources (cited)*, *From trained model*, Ultron, Vision, Banner, Cloud, and
  Natasha; the sidebar **Answer from Hakeem** action exists but is hidden, so the
  header selector is the one visible switch. The conversation stays in the same
  transcript and session, never a separate window. Everything stays on this
  machine — the only call is to the local Ollama server or an in-process GGUF — so
  Hakeem is not a cloud mouth. The runtime is the companion's own `[natasha]`
  extra; `[hakeem]` / `requirements-hakeem.txt` pin the same runtime so a user who
  installs the scholar alone has one obvious command.
- `gui/window_controls.py` — the shared **minimize / maximize / close** controls
  every window's title bar installs. `install_window_controls(window, header,
  theme=, typography=, on_close=)` builds one frame of three buttons into the
  title bar and returns a `WindowControls`: minimize (`–`, `iconify()`), maximize
  (`□` / restore `❐`), and close (`✕`, which runs the caller's own `on_close`, or
  destroys the window when it has none). `WindowControls` grows the window with
  `state("zoomed")` where the window manager supports it (the *native* maximize,
  which already fits the work area and remembers the old size); where it does not
  (many X11 desktops) it fills the **work area** by hand rather than going
  full-screen, so the taskbar is never covered — `work_area()` is the one
  platform lookup (`SystemParametersInfo`'s `rcWork` on Windows, the window
  manager's `_NET_WORKAREA` through `xprop` on X11, `NSScreen.visibleFrame` on
  macOS) and returns `None` where the desktop cannot report it, in which case a
  real full-screen window is the honest fallback. `fit_geometry(window, width,
  height, *, min_height=0)` is the shared **opening** size rule beside it: it
  returns the `"WxH"` geometry string a window should open at, clamped to the
  work area (or the screen when the desktop cannot report one), so a window never
  opens taller or wider than the usable desktop and its controls stay reachable on
  a small screen; `min_height` lifts a too-short request but is itself capped to
  what fits. The pre-fill geometry is kept so
  restore returns the window exactly, and each button's `command` is a bound
  method on the controller, which
  holds the window, which owns the buttons — a cycle a destroyed window would only
  leave when the cyclic collector happens to run, and on a worker thread the Tcl
  call that follows deadlocks (this is what hung a deck export). So each button
  clears its own command on its own `<Destroy>` (while the widget still exists,
  which a frame-level handler cannot promise) and the frame drops the caller's
  handler, exactly like `gui.chat_composer`; a closed window dies at once.
  Rolled into the **main window** and the ten dialogs (Code Studio, Presentation
  Studio and its finalize/course windows, the Import Center, the SQL import
  screen, Doc's Studio, the two image windows, and the per-mouth chat window), so
  every window is closed and grown the same way without each one re-implementing
  the logic.
- **Every window's title bar** is the `Theme.titlebar` band (a dark grey a step
  above the page base, `#181C25` dark / `#DCE3EE` light) closed by a
  `titlebar_edge` hairline, and carries the three controls above on its right. The
  main window's bar is the same band, so the header's mode switch, model selector,
  and controls read as one piece with every dialog; `Theme` gained the two tokens
  and each window paints its bar, its title labels, and the control frame with
  them, so one appearance switch restyles every window at once.
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
- `core/relation_connections.py` / `core/causal_paths.py` — the two answer-path
  reasoning bridges. `core.relation_connections` surfaces the **multi-hop
  connection** between two concepts a question names with no single edge between
  them (`concept_connections` walks the explicit relations with
  `core.relation_paths`, carries each step's direction, and labels the block as
  walked, not stored); `core.causal_paths` surfaces the **directed causal chain**
  the graph states (`CAUSAL_RELATION_TYPES` is a closed set so an association such
  as `measures` can never be read as a cause, `causal_graph_from_ontology` builds a
  `CausalGraph` from the ontology's causal edges, and `causal_explanations` walks
  the strongest forward-only chain, rendering it with the project's own
  `CausalExplainer`). Both are **read-only** — neither adds an edge — and both
  reach an answer as *data* (`AssistantAnswer.connections`/`causal_paths`,
  `JarvisTurn.connections`/`causal_paths`, rendered by `ai/chat.format_turn`), with
  a held-out gate (`connection_reasoning_accuracy`, `causal_reasoning_accuracy`)
  whose catalogues carry **negative controls**. `docs/REASONING_ASSESSMENT.md`
  records which of the reasoning layer's 69 modules reach an answer and which do
  not.
- `core/answer_confidence.py` — how the answer path weighs an answer's confidence.
  `evidence_confidence` is the passage-only base (the original
  `0.7 * top + 0.3 * coverage`), and `answer_confidence` folds the graph's own
  reasoning into it: the strongest graph-derived signal (entailed relations, walked
  connections, stated causal paths) may fill at most `GRAPH_SUPPORT_WEIGHT` (0.25)
  of the room to certainty, and **only when retrieved evidence exists** (no
  evidence ⇒ `final == 0`, so a graph fact can never manufacture confidence); stored
  contradictions then **multiply** the number down (`CONFLICT_PENALTY_WEIGHT` 0.5,
  severity summed and capped) so they can never be outvoted by support. The
  breakdown (`AnswerConfidence.evidence`/`graph_support`/`conflict_penalty`/`final`/
  `reasons`, `describe()`) reaches the caller as *data* (`AssistantAnswer.confidence_report`,
  `JarvisTurn.confidence`/`confidence_note`, `ai/chat.format_turn`, `turn_payload`),
  so a reader sees which term moved the number. The held-out gate
  `answer_confidence_accuracy` (`CONFIDENCE_JUDGEMENTS`, with negative controls)
  is a metric in `evaluation/benchmark.py`.
- `core/grounding.py` — the answer path's **groundedness gate**, the one place that
  decides whether model text may be published. It replaced a rule that accepted an
  answer as soon as it shared **one** content term with the evidence. `check_grounding`
  makes three deterministic checks and **fails closed** (the caller falls back to the
  always-traceable extractive answer): (1) enough of the answer's content terms must
  appear in the evidence (`DEFAULT_MIN_COVERAGE` 0.3, not just one); (2) each
  content-bearing sentence must share a term with the evidence, and a majority of
  unsupported sentences fails (`DEFAULT_MIN_SENTENCE_SUPPORT` 0.5), so a fluent
  paragraph cannot ride one incidental word in; (3) a causal link the answer states
  **in the direction the graph contradicts** (the graph holds the reverse edge and
  not the forward one) fails -- the same honesty rule the causal layer applies to its
  own walks, now applied to the model's words. `GroundingCheck` carries every reason;
  `_is_grounded` in `core/assistant.py` is kept as a thin wrapper. The held-out gate
  `grounding_accuracy` (`GROUNDING_JUDGEMENTS`, with negative controls) is a metric in
  `evaluation/benchmark.py`.
- `core/reasoning_report.py` — the **user-visible reasoning trace**: the one place
  that assembles *"How this answer was reached"* from data already on
  `AssistantAnswer` (retrieved evidence, the graph relations used, the corpus
  procedure ordered by the planner, the recommendation the decision engine ranked,
  the belief the graph supports, multi-hop connections, directed causal paths,
  stored contradictions, and the confidence breakdown), so the reasoning layer is
  visible, not only auditable.
  `ReasoningStep` (`label`/`lines`/`derived`) is one step -- a derived step is a
  claim the graph *entailed* or a procedure it *ordered* rather than stored -- and
  `ReasoningReport`
  (`question`/`steps`) carries them in the fixed `STEP_ORDER`, with
  `derived_steps` selecting only the derivations that carry content, `is_empty`,
  and a `describe()` that opens with the title and then lists each step (an empty
  step reads `(nothing)`, never hidden). `reasoning_report(question, *, passages,
  relation_facts, plan, decision, belief, relation_conflicts, inferred,
  connections, causal_paths, confidence)` is the one builder: it invents nothing and
  calls no new engine, so
  the trace can never claim more than the answer already held. It reaches
  `AssistantAnswer.reasoning` (`core/assistant.py`) and, through
  `JarvisTurn.reasoning_trace`, `ai/chat.format_turn`/`turn_payload` -- and so the
  GUI too -- plus the Jarvis `why` control (`core/jarvis_controls.py`,
  `core/jarvis.py`) and the CLI `python -m cli relations why`.
- `core/answer_plan.py` — the **plan bridge**: the corpus's own ordered procedure
  for a question that asks *how*, run through the project's real `Planner`. The
  planning layer (`Planner`/`Action`/`Goal`/`PlanStep`) was reachable only from the
  `CognitiveController` and the evaluation scenarios, so a procedural question
  ("how do I measure porosity on a core sample?") got a definition back, never the
  procedure the corpus already states (48 entries carry `procedures`) -- Gap 7 of
  `docs/REASONING_ASSESSMENT.md`. `is_procedural(question)` recognises a question
  that asks *how* (English and Arabic cues), so a definitional "what is porosity?"
  is never read as procedural. `match_entry(question)` scores each entry by the
  content terms it shares with the question (its concepts, its relation endpoints,
  **and its own procedure text**), breaks ties on the entry key, and returns
  ``None`` for a question that shares no term -- never a guess.
  `order_procedure(steps, ...)` runs the steps through `Planner.order_steps`, so a
  circular or dangling dependency is **refused** with ``ValueError`` rather than
  silently reordered; nothing is executed. `answer_plan(question)` returns an
  `AnswerPlan` (``topic``/``source``/``steps`` + a numbered ``describe()``) only
  when the question is procedural *and* an entry with a stated procedure matches
  it, else ``None``. `AssistantAnswer.plan` carries it, it becomes the `SOURCE_PLAN`
  step of the reasoning trace, `JarvisTurn.plan` renders it in
  `ai/chat.format_turn`/`turn_payload`, and the held-out gate `plan_accuracy`
  (`PLAN_JUDGEMENTS`, with negative controls: a definitional/unrelated question
  yields no plan, a circular/dangling dependency is refused, out-of-order steps
  come back in dependency order) is a metric in `evaluation/benchmark.py`.
- `core/answer_decision.py` — the **decision bridge**: the ranked recommendation
  the knowledge graph states for a question that asks *what to do*, run through the
  project's real `DecisionEngine`. The decision layer (`DecisionEngine`/
  `UtilityModel`/`RiskModel`/`DecisionExplainer`) was reachable only from the
  `CognitiveController` and the evaluation scenarios, so a decisional question
  ("what should I do to increase permeability?") got a definition back, never the
  ranked recommendation the graph already states (`acid_stimulation —increases→
  permeability`, `hydraulic_fracturing —increases→ permeability`) -- the decision
  half of Gap 7 of `docs/REASONING_ASSESSMENT.md`. `is_decisional(question)`
  recognises a question that asks what to do (English and Arabic cues), so a
  definitional "what is permeability?" is never read as decisional and never drags
  a recommendation into a definitional answer. `direction_of(question)` reads the
  direction the question asks for (`increase`/`reduce`/`prevent`/`maintain`/
  `control`), or `""` when it names none; it is scanned in a fixed order, so the
  same question always yields the same direction. The options are the graph's own
  **typed inbound edges** into the concept the question is about, restricted to the
  relation types that match the direction asked for (`DIRECTION_RELATIONS`), so an
  option is never offered when the graph says it moves the goal the *other* way --
  "what should I do to reduce permeability?" returns ``None`` because the graph
  holds no edge that reduces permeability, rather than offering one that increases
  it. Each option becomes an `Action` that adds the goal at a cost of
  ``1 - confidence`` (a better-attested option is cheaper); a source the graph
  links to several of the question's goals becomes **one** action advancing all of
  them, which is the multi-objective case the utility model exists for.
  `answer_decision(question, ontology=...)` runs those actions through
  `DecisionEngine.decide` and ranks them with `DecisionExplainer.explain`, and
  returns an `AnswerDecision` (`goal`/`direction`/`selected`/`options` +
  `describe()`) only when the question is decisional *and* the graph states at
  least one option in the direction asked for, else ``None`` -- never a guess.
  Nothing is executed: the recommendation is a *reported* artifact. It reaches
  every surface the answer already reaches: `AssistantAnswer.decision` is set in
  `GroundedAssistant.ask`; it becomes a new `SOURCE_DECISION` step in
  `core.reasoning_report` (fixed position: after the plan, before the other
  derivations); `JarvisTurn.decision` carries the ranked lines, so
  `ai/chat.format_turn` and `turn_payload` render them (and the GUI gets them for
  free); and `python -m cli relations why` prints them inside the trace and in its
  ``--json``. The held-out gate `decision_recommendation_accuracy`
  (`DECISION_JUDGEMENTS`) checks the bridge with **negative controls** -- a
  definitional, unrelated, or unanswerable question must yield no recommendation --
  and is a metric in `evaluation/benchmark.py`.
- `core/answer_belief.py` — the **belief bridge**: the evidence-weighed verdict the
  knowledge graph supports for a question that asks *whether a claim holds*, run
  through the project's real `BeliefRevisionEngine`. The belief layer
  (`Belief`/`BeliefRevisionEngine`/`ConfidenceModel`/`ContradictionDetector`) was
  reachable only from the `CognitiveController` and the evaluation scenarios, so an
  evaluative question ("does acid stimulation increase permeability?") got a
  definition back, never the verdict the graph already supports -- the belief part
  of Gap 7 of `docs/REASONING_ASSESSMENT.md`. `is_evaluative(question)` recognises
  a question that asks for a verdict (English and Arabic cues), **anchored to the
  question's opening** (or an unambiguous verdict phrase anywhere), so a
  definitional "what is permeability?" is never read as evaluative and a question
  that merely *contains* "is" ("what is the graph?") never fires. The evidence is
  the graph's own **typed edges whose both endpoints are concepts the question
  names**, each turned into an `Evidence` record carrying the relation type in its
  metadata -- which is what lets the engine's `ContradictionDetector` recognise two
  antonymous relations on the same ordered pair as a genuine disagreement. The
  confidence and the lifecycle status that reach the reader are the ones
  `BeliefRevisionEngine.revise` actually produced (its own evidence-weighted
  posterior and contradiction detection), so a modestly-seeded claim is honestly
  *provisional* rather than *accepted*. It is **read-only** -- nothing here changes
  the knowledge graph -- and it never invents evidence the graph did not state, and
  returns ``None`` rather than a guess when the question names fewer than two
  concepts or no edge connects them. It reaches every surface the answer already
  reaches: `AssistantAnswer.belief` is set in `GroundedAssistant.ask`; it becomes a
  new `SOURCE_BELIEF` step in `core.reasoning_report` (fixed position: after the
  decision, before the other derivations); `JarvisTurn.belief` carries the readable
  lines, so `ai/chat.format_turn` and `turn_payload` render them (and the GUI gets
  them for free); and `python -m cli relations why` prints them inside the trace and
  in its ``--json``. The held-out gate `belief_accuracy` (`BELIEF_JUDGEMENTS`)
  checks the bridge with **negative controls** -- a definitional question, a plain
  statement, an evaluative question naming one concept, or one the graph cannot
  speak to must yield no belief -- and is a metric in `evaluation/benchmark.py`.
- `core/answer_lead.py` — the **answer-body lead**: the last step of the reasoning
  bridge. Phases 41-43 (plan, decision, belief) let a procedural, decisional, or
  evaluative question reach the planner, the decision engine, and the belief
  revision engine — but the bridge stopped at the trailing *trace*: the answer
  **body** was still the retrieval template ("Here is what the available sources
  say about ..."), so the reasoning sat in a footer a reader may never open. This
  module puts the reasoning **in the answer itself**. `answer_lead(question,
  plan=, decision=, analogy=, belief=, temporal=)` builds the lead strongest-first —
  procedure, then recommendation, then comparison, then verdict, then timeline — and
  only the first that applies, so a question is
  never handed two competing leads. Each lead is a *named* source: the procedure
  heading names the corpus entry it came from (`plan.source`), the recommendation
  lists the graph's ranked options, and the verdict states the claim, its
  confidence, and the edges it rests on. `AnswerLead` carries the `kind`, the
  `heading`, the `body`, and the `language`; `describe()` is the heading alone when
  there is no detail. ``None`` means the question warranted no reasoning, so a
  definitional question keeps its retrieval opening and nothing is invented to fill
  a lead. `core.composition.with_lead(question, lead, body)` joins the lead to the
  retrieved body under the *"What the sources add about ..."* label, and
  `grounded_answer(question, sources, *, lead="")` gained the `lead` parameter, so
  the lead opens the answer while the cited evidence still follows. The same
  lead is used whether the model wrote the body or the extractive fallback did, so
  the reasoning is never silently replaced by either. The held-out gate
  `lead_accuracy` (`LEAD_JUDGEMENTS`) checks the bridge with **negative controls** —
  a definitional, unrelated, or blank question must earn **no** lead — and its body
  judgements run the real answer path through the assistant's own ``ask``
  (`answer_for=`), so a regression that left the reasoning in the trace while the
  body kept the template would lower the metric; it is a metric in
  `evaluation/benchmark.py`. The fix is also carried on the machine-readable turn:
  `JarvisTurn.lead` (set in `core/jarvis.py`) and `turn_payload["lead"]` in
  `ai/chat.py`, so a caller can see the reasoning *as the answer's own opening*, not
  only in the trailing `reasoning_trace`. Two related fixes landed with it: the
  plan bridge's `match_entry` now reads the corpus **glossary**, where each term's
  Arabic alias lives ("porosity / مسامية"), so an Arabic procedural question
  ("كيف أقيس المسامية؟") matches the entry and reaches the planner; and
  `core.composition.topic_phrase` no longer emits a broken clause — a bare verb and
  the pronoun the question omitted are dropped ("how does porosity relate to
  permeability" → "porosity permeability"), while a word that could be a noun is
  kept, so nothing the user named disappears.
- `core/answer_temporal.py` — the **timeline bridge**: the temporal half of Gap 7
  of `docs/REASONING_ASSESSMENT.md`. The project carried a whole temporal layer
  (`TemporalReasoner`, `Timeline`, `TemporalEvent`, `TemporalPattern`,
  `TimePrediction`) but it was reachable only from tests, so a question like *"what
  changed over the session?"* got a definition back, never the ordered account of
  what actually happened. `answer_temporal(question, *, conversation=, events=)`
  reads the one real timeline the answer path already holds -- the conversation's
  own **timestamped turns** -- turns them into `TemporalEvent`s, and runs them
  through the project's real `TemporalReasoner`: the order is chronological, the
  relation between the last two turns is the reasoner's own
  (`before`/`during`/`simultaneous`...), and any repeating pattern and its
  next-time prediction are the ones `find_pattern`/`predict_next` actually
  produced. `AnswerTemporal` carries the `entries` (`TimelineEntry` role/text/at),
  the `relation`, the readable `change`, the `pattern`, the `prediction`, and the
  `language`; `span_seconds` measures the wall-clock window and `describe()`
  renders the account. It is gated to a *temporal* question (`is_temporal`,
  `TEMPORAL_CUES` in either language), so a definitional question never drags an
  account of the session into its answer, and it returns ``None`` rather than a
  guess when there is no timeline (fewer than two turns, or no conversation).
  `SYSTEM`/`CONTEXT` turns are skipped -- a system instruction and an attached file
  are not things that happened in the conversation. The timeline can open the
  answer as a lead (`core.answer_lead`'s `KIND_TIMELINE`, tried after procedure,
  recommendation, and verdict) and it is carried on the machine-readable turn
  (`JarvisTurn.temporal`, `turn_payload["temporal"]`) and in the reasoning trace
  (`core.reasoning_report.SOURCE_TEMPORAL`). The held-out gate `temporal_accuracy`
  (`TEMPORAL_JUDGEMENTS`) checks the bridge with **negative controls** -- a
  definitional or unrelated question, or a temporal question with one event, must
  yield no timeline -- and is a metric in `evaluation/benchmark.py`.
- `core/answer_analogy.py` — the **analogy bridge**: the analogical half of Gap 7
  of `docs/REASONING_ASSESSMENT.md`, and the last isolated engine in it. The
  project carried a whole analogical layer (`AnalogyEngine`,
  `StructureMappingEngine`, `Structure`, `StructureMapping`) but it was reachable
  only from the controller, `cli/analogy.py`, and the evaluation scenarios, so a
  question like *"is porosity like permeability?"* got a definition back -- or,
  worse, the belief bridge read its `is ... ?` opening as a verdict about a
  *different* claim ("does porosity increase permeability?"). `answer_analogy(
  question, *, ontology=, hops=, limit=)` reads the concepts the question names,
  builds each one's relational neighbourhood, and aligns them with the project's
  real `AnalogyEngine` (through `StructureMappingEngine`) -- so the comparison that
  reaches the reader is the one the structure-mapping engine actually produced:
  its matched relations, its systematicity, and its **candidate inferences** (each
  a *candidate*, discounted by how much of the structure really aligned, and never
  asserted into the graph). It is gated to an *analogical* question
  (`is_analogical`, `ANALOGY_CUES` in either language, over a normalised question
  so "what is porosity like?" fires like an interior "like"), so a definitional
  question never drags a comparison into its answer, and it returns ``None``
  rather than a guess when the question names fewer than two known concepts. It is
  **honest by construction**: it reports the engine's own `meaningful` verdict even
  when it is `False` -- a pair sharing no relational system is said to be *not* an
  analogy, not dressed up as one -- and it reports what it could **not** map as
  well as what it could. `AnswerAnalogy` carries the `base`/`target`, the
  `meaningful` verdict, the `score`, the `matched` relations, the `transferred`
  inferences, the `unmapped_base`/`unmapped_target`, and the `language`; `describe()`
  renders the comparison. The comparison can open the answer as a lead
  (`core.answer_lead`'s `KIND_ANALOGY`, between recommendation and verdict) and it
  is carried on the machine-readable turn (`JarvisTurn.analogy`,
  `turn_payload["analogy"]`), in the reasoning trace
  (`core.reasoning_report.SOURCE_ANALOGY`), and in `cli relations why --json`. It
  also stops the belief bridge from answering a comparison as a claim
  (`is_analogical` → no belief). The held-out gate `analogy_answer_accuracy`
  (`ANALOGY_ANSWER_JUDGEMENTS`) checks the bridge with **negative controls** -- a
  definitional question, an unrelated one, or an analogical question naming one
  concept must yield no comparison, and a pair with no shared relational system
  must be reported as *not* an analogy -- and is a metric in
  `evaluation/benchmark.py`. Phase 49 added the Arabic cues (`يشبه`, `تشبه`,
  `شبيه`, `مماثل`, `مثلما`, `مشابه`, `قارن`, `مقارنة`, `على غرار`, and the `مثل`
  inside "هل المسامية مثل النفاذية؟") to `ANALOGY_CUES`, so an Arabic analogical
  question now fires the comparison instead of a belief verdict.
- `core/speech.py` — speech I/O contracts (ASR/TTS), dependency-free `AudioClip`
  (PCM/WAV), transcript + silent fallbacks, `VoiceAssistant` voice loop. The TTS
  engine itself is a *system* program (espeak-ng/espeak/flite/`say`), not a Python
  package, so pip cannot install it — like Tesseract for OCR. `speech_available`,
  `speech_install_plan` (a platform-aware package-manager command — incl. winget/
  choco on Windows — or an honest hint when the app knows none), and
  `install_speech_engine` (never raises; an
  injected `runner` makes it testable with no real install) give the app a real
  next step when no engine is present. The **microphone** side is
  platform-complete, which is what makes the chat window's mic icon toggle work
  everywhere. `MicrophoneRecorder` discovers a recorder in order — `arecord`
  (ALSA), `pw-record`/`parecord` (PipeWire/PulseAudio, because a modern Linux
  desktop often has no `arecord`), `sox`, then **Windows' own MCI `waveaudio`
  device** (`_WINDOWS_RECORDER`, `_windows_recorder_script` driving `winmm.dll`
  through PowerShell) — so a Windows user, who has neither `arecord` nor `sox`,
  records with nothing to install. The stop signal is part of the recorder
  contract: `MicRecording.stop()` sends `SIGINT` (arecord/sox flush their WAV
  header on interrupt) or, when the recording carries a `stop_path`, **creates
  that marker file** — the Windows script polls it and then runs its MCI `save`,
  because a signal would kill the process before the WAV is written.
  `microphone_install_hint()` is the one platform-aware "next step" string, used
  by the window, `ChatService.asr_status`, and the Jarvis CLI in place of the old
  Linux-only "install arecord or sox". **Windows needs no install**: the OS ships
  two engines and the project prefers the right one. The modern **OneCore** engine
  (`Windows.Media.SpeechSynthesis`, reached through WinRT) is used when
  `windows_onecore_available()` succeeds — it is the *only* one that can see the
  voices installed through *Settings → Time & Language → Speech → Add voices*,
  which is how an Arabic voice is added on Windows 11, so it is what actually reads
  Arabic instead of mangling it with an English voice. `WindowsOneCoreSynthesizer`
  renders through it (voices picked by language, gender, and an SSML `prosody`
  rate; the SSML call falls back to plain text if the engine rejects it), and the
  older **SAPI** engine (`WindowsSapiSynthesizer`, `System.Speech`) is the fallback
  when the WinRT probe fails. `_windows_engine` is the one place that choice is
  made, so discovery, `load_synthesizer`, and the Qwen3-TTS Arabic fallback agree;
  `load_synthesizer("windows-onecore")` / `("windows-sapi")` name one explicitly.
  Both Windows engines render through a shared `_WindowsPowerShellSynthesizer` that
  writes the text to a **UTF-8 temp file** and reads it back in the script (the
  text is *not* embedded in the command line: a long Arabic answer would otherwise
  hit the ~32 KB limit and console encoding could mangle it) with a 180 s timeout,
  and `Speaker` plays the WAV through
  the built-in `SoundPlayer` when no aplay/afplay/sox exists. Both Windows engines
  and macOS `say` read the *language* of the text, not just the timbre: the detected
  language (Arabic script → `ar`) narrows the engine to a voice whose culture
  matches (`_sapi_select_voice`; the OneCore script's `Language -like 'ar*'`), and
  `say` is given `-v <installed voice>` for a
  non-English language (`_macos_language_voice`), so Arabic is read as Arabic
  instead of being mangled by the default English voice; when no voice for that
  language is installed the engine keeps its own default rather than failing.
  `windows_installed_voices`/`windows_voice_report` list the voices the engine can
  actually see and say which engine is used and whether an Arabic voice is visible
  — or the exact *Add voices* step when it is not — surfaced by
  `TextReader.describe()` and `python -m cli speak list`. The two Windows engines
  keep **separate** voice stores and neither is a superset of the other, so
  `windows_installed_voices` asks **both** (the WinRT `AllVoices` listing and the
  SAPI `GetInstalledVoices` listing) and returns their **union**, de-duplicated by
  name+language and OneCore-first, tagging each `WindowsVoice` with the engine that
  reported it; asking only one engine silently dropped whatever only the other
  knew (a OneCore Arabic voice is invisible to SAPI, and a legacy SAPI voice can be
  absent from the WinRT store). Both listing scripts, and the OneCore availability
  probe, are wrapped in `try`/`catch{}` with `$ErrorActionPreference='Continue'`:
  a non-terminating WinRT warning used to abort the probe under `Stop` and report
  "no OneCore", which is exactly how a voice the OS offered to every other app
  stayed invisible here. The WinRT listing is attempted unconditionally (its
  tolerant script yields no lines when the store is unreachable) rather than gated
  on the probe, so a false-negative probe can no longer hide the Arabic voice.
  `_engine_for_voice` points a chosen `SystemVoice` at the
  engine that listed it (so `load_synthesizer('auto', voice=…)` uses WinRT for a
  WinRT voice instead of handing it to SAPI, which cannot see it), and the
  SAPI→OneCore `_fallback` now triggers whenever Windows speech is available
  rather than gating on the probe — a false-negative probe is one of the ways the
  Arabic voice goes missing, and the attempt is free once SAPI has returned silence.
  The engine choice is then settled in **one** place: `windows_engine_name()` (the
  engine that actually lists an Arabic voice first, via `windows_arabic_engine()`,
  then `windows_onecore_usable()` as the tie-breaker), which both
  `discover_synthesizer_name()` and `_windows_engine()` consult, so discovery, the
  `--tts auto` load, and the Qwen3-TTS Arabic fallback can never disagree.
  `windows_onecore_usable()` treats a listing that returned a OneCore voice as
  proof the engine works, so the explicit `load_synthesizer('windows-onecore')`
  gate no longer refuses the one engine that can read Arabic on a false-negative
  probe. The reverse case is closed too: SAPI does **not** fail on a language it
  has no voice for — it reads it with a mismatched voice — so
  `WindowsSapiSynthesizer._render` hands a non-English language it cannot serve
  (`_sapi_serves_language`, asked of SAPI itself) to the WinRT engine *before*
  returning its own mangled output, and `WindowsOneCoreSynthesizer._fallback`
  likewise refuses to hand a language SAPI cannot serve over to it. Either
  direction therefore ends in real audio or honest silence, never a mangled
  reading.
  A chosen **installed** voice is spoken through Microsoft's documented path — set
  the `Voice` property and synthesise **plain text** — because the WinRT engine takes
  the voice from any SSML it is given, so a bare `xml:lang` there overrode the
  property and read *every* Arabic entry with the language's default voice: a
  woman's entry (Hoda) and a man's entry (Naayf) both came out as Naayf. With plain
  text nothing in markup can override the pick, and an installed voice keeps its own
  neutral rate, so no SSML is needed. A *catalog* voice (which has no installed name
  and does need the rate) still uses SSML, and there the picked voice is named in the
  markup (`<voice name="…">`, XML-escaped) with `xml:lang` taken from that voice's
  own language tag, so the markup cannot fall back to the language default either.
  (SAPI is unaffected throughout: it selects with `SelectVoice` and speaks plain
  text, with no SSML to override the choice.)
  `system_voices()` (cross-platform, cached, cleared by
  `refresh_windows_speech_caches`) is the **one** place the *real* installed voices
  are listed for the menus: the Windows voices (the union above, via
  `windows_installed_voices`) or the macOS `say -v ?` voices, and `()` elsewhere so
  a caller falls back to the synthetic catalog rather than inventing a voice.
  `_WindowsPowerShellSynthesizer` now renders through a private `_render` and, when
  it produces **no audio** (the classic Windows 11 case: the user's only voices are
  OneCore, so SAPI writes an empty WAV — the reported `0.00 s`), it tries the
  sibling engine (`_fallback`, SAPI → OneCore) before
  the silent stand-in, so a blank reading becomes speech instead of silence.
  `AudioClip.read_wav` now decodes 8/16/32-bit and any channel count into mono
  16-bit (`_pcm_to_mono16`), because the OneCore stream is not always 16-bit mono
  and refusing it silently turned a real reading into silence. The
  install path
  prints its hint *once* (`install_speech_engine`'s message already carries it).
  CLI: `python -m ai.voice` / `python -m cli
  voice`; `python -m ai.speak --install-engine` / `python -m cli deps --install
  speech` run it.
- `core/voices.py` — the text-to-speech **voice catalog**: 16 `Voice`s in four
  categories (`man`/`woman`/`boy`/`girl`, four each), each naming a timbre
  (engine `variant` `m1`..`m6` / `f1`..`f4`), a pitch, and a speed, plus the two
  settings that make a synthetic voice read **naturally instead of mechanically**:
  a `pitch_range` (the intonation range, `espeak-ng`'s `-P` — a living sentence
  melody, kept above the flat engine default of 50) and a `word_gap` (`-g` — a
  short pause between words for natural rhythm and breath). Speeds are tuned near a
  human conversational rate (the SAPI/OneCore paths map them onto a near-neutral
  rate), so men's and women's voices alike read like a person rather than a flat,
  rushed machine; `MIN/MAX_PITCH_RANGE`, `MIN/MAX_WORD_GAP`, and the
  `DEFAULT_PITCH_RANGE`/`DEFAULT_WORD_GAP` bound and default them, and a bad entry
  is rejected in `__post_init__`.
  `CATEGORY_LABELS`/`voice_categories`, `voices_for`, `voice_keys`, `find_voice`
  (raises on an unknown key), `default_voice`/`DEFAULT_VOICE_KEY`, `voice_summaries`,
  and `detect_language` (Arabic script → `ar`, else `en`) so one voice reads both
  languages. Honest by construction: a `Voice` is *settings*, not a recording, and
  `description` says so — `espeak-ng` has no true child voice, so `boy`/`girl` are
  a raised pitch on a male/female variant and say that plainly.
  `core/speech.py`'s `CommandSynthesizer`/`load_synthesizer` take that voice and
  insert its engine flags after the executable (`_espeak_voice_options` emits the
  timbre, pitch, speed, intonation range, and word gap; the Windows OneCore SSML
  carries the voice's `rate` *and* `pitch` so a category's four voices stay
  distinct; `say` selects a language voice and carries the rate; `flite` has none),
  and `load_synthesizer("auto")` still falls to `silent` when nothing is installed.
  **Three kinds of voice** live here now. The synthetic `VOICE_CATALOG` above is
  *settings*; the :class:`SystemVoice` is a voice the OS **really installed**,
  by its own name (`Microsoft Hoda`, `Majed`, ...); and the :class:`ModelVoice` is
  a voice that lives inside a **local model** (`model:audio.cpp:Sarah`), described
  in its own entry below. `group_system_voices`/`group_model_voices` are the one
  language-grouping rule every voice menu uses (Arabic first, then alphabetical),
  so the two menus that list these voices cannot order them differently.
  **Voices are grouped by the source they came from.** `voice_sections(*, system=,
  model=, catalog=)` returns one `VoiceSection` per source -- `SOURCE_SYSTEM`
  (*Installed Voices (Arabic + English)*), `SOURCE_AUDIOCPP` (*VoiceTut-TTS (Local
  Arabic)*), `SOURCE_PIPER` (*Piper (Arabic + English)*), `SOURCE_QWEN`
  (*Qwen3-TTS (English)*), and `SOURCE_SYNTHETIC` (*Synthetic Voices (Arabic +
  English)*) -- in that fixed order (`SOURCE_TITLES`), each carrying only the
  voices its engine really reports here, with its own `groups` (a language label
  for the real/model voices, the category for the catalog) and a flat `voices`
  tuple. An empty source is dropped, and the synthetic catalog is dropped as a
  whole when no engine can speak it here (it is settings for whichever engine
  discovery picks -- the system engine, Piper, or Qwen3-TTS), so a menu built from
  this can never offer a voice that would come out silent. Each voice carries a
  `.source` (a `Voice` is `SOURCE_SYNTHETIC`, a `SystemVoice` is `SOURCE_SYSTEM`, a
  `ModelVoice` is its owning engine), and a `ModelVoice` may carry a readable
  `title` so a model key (`ar_JO-kareem-medium`) reads as a name while its `key`
  still names the file. `model_voices()` now collects from **all three** local
  engines (VoiceTut-TTS, Piper, Qwen3-TTS), each `voice_list` returning `()` unless
  that engine and its weights are present. A `SystemVoice` carries the
  engine's exact `name`, its `language` tag (`ar-SA`), its reported `gender`, and
  the `engine` that listed it; `key` is a stable `system:<engine>:<name>` string
  (marked by `SYSTEM_VOICE_PREFIX` / `is_system_voice_key`) so a saved choice
  survives a restart and can never be confused with a catalog key. `language_code`
  (two-letter), `arabic`, `language_label`/`category_label` (grouped by language,
  `Arabic`/`العربية`), `label`/`label_en`/`label_ar` (the name, never translated),
  an honest `description`, and `to_primitive` round out the API. `system_voices()`
  wraps `core.speech.system_voices` (local import, since `core.speech` imports this
  module) and `find_system_voice(key)` resolves a `system:` key against what is
  *currently* installed — returning `None` (so the caller falls back) when the voice
  was uninstalled. `find_voice` keeps raising for an unknown *catalog* key, so a
  typo is still reported. This is what fixes the report that the list showed
  invented espeak voices (Adam/آدم, Layla/ليلى) instead of the real Windows voices:
  the menus now prefer `system_voices()` and fall back to the catalog only when the
  OS reports none.
- `core/reading.py` — `TextReader`: read the chat's text aloud. Ties a chosen
  voice — a catalog `Voice`, a real installed `SystemVoice`, **or** a local
  `ModelVoice` — a `SpeechSynthesizer`, and a `Speaker` together behind `render`/
  `speak` (playback is best-effort, so a machine with no player still gets the
  clip)/`save` (WAV), `set_voice`, `available`, `refresh` (re-discovers the engine
  after one is installed mid-session), `system_voices()` (the real installed voices,
  for a menu), and an honest `describe()` that names the
  voice and says when no engine exists rather than pretending. A
  `system:<engine>:<name>` or `model:<engine>:<name>` key is resolved against what
  the OS or the local model reports and falls back to the catalog default when the
  voice is gone (`_as_voice`), so a saved choice never asks the engine for a name it
  would reject. CLI:
  `python -m ai.speak [list] "<text>" --voice <key> [--write-wav f] [--no-play]`
  / `python -m cli speak` and `python -m cli voices`.
- `core/qwen_speech.py` — the **Qwen3 speech models** (ASR + TTS) as a real, local
  alternative to the system engines, added the way Ultron/Vision add a real local
  *language* model. `QwenAsrRecognizer` runs **Qwen3-ASR** (1.7B default / 0.6B)
  through `qwen-asr`; `QwenTtsSynthesizer` runs **Qwen3-TTS** (12 Hz, 1.7B / 0.6B)
  through `qwen-tts`; both load a model folder into this process (no server,
  nothing sent off the machine) and both delegate to an injectable runner
  (`AsrRunner`/`TtsRunner`), so the engines are exercised with no `torch` and no
  weights. `ASR_CATALOG`/`TTS_CATALOG` + `resolve_asr_spec`/`resolve_tts_spec`/
  `asr_spec_names`/`tts_spec_names` name the fetchable models; the weights live
  under `data/qwen-speech` (`core.paths.qwen_speech_models_dir`, git-ignored) with
  a `model.json` registry, and `find_model_dir`/`model_is_cached` read them.
  `qwen_asr_status`/`qwen_tts_status` report *exactly* what is installed (extra +
  weights) and the real next step; `download_model` runs only through the
  permission + human-approval gates (`--approve`), like every other model
  download. The runtime is the optional `[qwen-asr]` / `[qwen-tts]` extras
  (`requirements-qwen-asr.txt`, `requirements-qwen-tts.txt`), imported lazily.
  They are **two** extras on purpose: `qwen-asr` pins `transformers==4.57.6` and
  `qwen-tts` pins `transformers==4.57.3`, so no single environment satisfies
  both -- a combined requirements file left pip no common `transformers`, and it
  backtracked to an old release that caps `numpy<2.0`, then tried to *build*
  numpy 1.26.4 from source, which fails on Windows with a long-path `OSError`
  (and on any machine with no compiler). Each file now pins only its own engine
  and lets that engine own its `transformers` version. The readiness probe is
  **exact**: `qwen_asr_extra_installed` requires `qwen_asr` itself, because
  `transformers` is only its transitive dependency and cannot stand in for it --
  the `[image]` extra installs `torch` + `transformers` without `qwen-asr`, so
  accepting `transformers` as a substitute made the microphone report "ready",
  pick `qwen3-asr`, and then fail with a raw `ModuleNotFoundError: No module
  named 'qwen_asr'` *after* the user had already recorded. A broken or partial
  extra now raises `QwenSpeechUnavailable` naming the install command from inside
  the runners, so the failure is the next step rather than a bare import error.
  **Honest by construction:**
  Qwen3-ASR covers **Arabic** (52 languages), but **Qwen3-TTS does not speak
  Arabic** (ten languages: Chinese, English, Japanese, Korean, German, French,
  Russian, Portuguese, Spanish, Italian) — `QWEN_TTS_LANGUAGES` states this, the
  synthesizer refuses Arabic with a clear message, and   a `fallback` synthesizer
  (the system engine) can be supplied so Arabic is still read aloud.
- `core/arabic_speech.py` — **how Arabic is *said*, not spelled.** A deterministic,
  dependency-free, **presentation-only** pass that prepares Arabic text before it
  reaches any engine, because the engines read Arabic *words* well and Arabic
  *symbols* badly. Measured on this project's own output, `espeak-ng -v ar` reads
  `25%` as `خمسة وعشرون` (the unit silently vanishes), switches to an **English**
  voice in the middle of `1500 psi`/`2000 ft`/`3.5 mD`, and turns `10-20` into
  gibberish; spelled out (`خمسة وعشرين بالمئة`) the same text is read correctly.
  So `prepare_for_speech(text, language=)` spells **numbers** in Arabic words
  (`1500` -> `ألف وخمسمئة`), translates a **known unit** to its Arabic name
  (`psi` -> `باوند لكل بوصة مربعة`, `ft` -> `قدم`, `%` -> `بالمئة`), reads a
  **range** as a range (`10-20` -> `عشرة إلى عشرين`), reads a **decimal** digit by
  digit (`3.5` -> `ثلاثة فاصلة خمسة`), turns ASCII `, ; ?` into `، ؛ ؟` inside
  Arabic text, and drops the decorative **tatweel** (the kashida made the engine
  spell a word out letter by letter). `number_to_arabic_words` uses the real
  Arabic counting rules — the dual (`ألفان`), the plural after 3..10
  (`ثلاثة آلاف`), the singular from 11 (`أحد عشر ألف`) — and an `oblique` form
  (`ألفين`, `عشرين`) for a number a preposition governs, which is what a spoken
  range needs. **Honest by construction**: it is gated to Arabic
  (`detect_language`), so an English answer is returned **byte-identical**; it
  never invents a unit (an unknown token is left alone), never rewrites a Latin
  identifier that has no Arabic name (`SPE`, `API`, `ISO`), never touches a code
  span, and deliberately adds **no** tashkeel (a *wrong* vowel mark sounds worse
  than none). It is idempotent, and it is wired into **every** synthesizer
  (`CommandSynthesizer`, the Windows SAPI/OneCore path, `PiperSynthesizer`, and
  the Qwen3-TTS path including its Arabic fallback), so whichever engine speaks,
  the reading is the words a listener hears. This is the fix for "the Arabic
  voices still sound robotic": the neural voice (Piper) is the *quality* answer
  and already existed, but the *text* it was handed was not speech-ready.
- `core/piper_speech.py` — **Piper**, a real, local **neural** text-to-speech
  engine: the honest answer to "the voices sound robotic". Piper
  (`rhasspy/piper`, maintained by the Open Home Foundation) renders speech with a
  small VITS model exported to ONNX, so a voice sounds like a person instead of
  `espeak-ng`'s formant synthesis -- and it runs **entirely on this machine**, on
  the CPU, with nothing sent off it. Unlike Qwen3-TTS it **speaks Arabic**
  (`ar_JO-kareem`), which is why it is the one engine that gives a natural
  Arabic *and* English voice without leaving the machine. `PiperSynthesizer`
  satisfies `core.speech.SpeechSynthesizer`, so the window, Jarvis, the voice
  loop, and every mouth use it unchanged. **One voice = one model**:
  `PIPER_VOICES` names the fetchable voices and `spec_for_voice` maps a project
  catalog voice (`man-1` … `girl-4`) onto one, so the picker keeps working and
  each entry stays distinct (Arabic has a *single* released voice, so every
  Arabic reading uses it -- stated plainly, not hidden). **Prosody from the
  catalog**: a voice's `speed`/`pitch_range` become Piper's
  `length_scale`/`noise_scale`/`noise_w_scale`, so the two voice systems agree on
  how a voice should read.
  `PiperRunner` is the injection seam and `PiperOnnxRunner` the default (lazy
  import of `piper`/`onnxruntime`), so the engine is exercised with no extra and
  no weights. The model is an **ONNX file under `data/piper`**
  (`core.paths.piper_models_dir`, git-ignored) with a `model.json` registry:
  `find_model_file`/`model_is_cached`/`resolve_spec`/`spec_names`/`spec_summaries`
  read them. The runtime is the optional `[piper]` extra (`FEATURE_PIPER`,
  `requirements-piper.txt`, `piper-tts`), imported lazily. `piper_status`/
  `available_synthesizer` report *exactly* what is installed and the real next
  step, and `download_model` runs only through the permission + human-approval
  gates (`--approve`), like every other model download. **One voice does not
  silence the other**: when the voice a language needs is not downloaded yet and a
  `fallback` synthesizer (the system engine) was supplied, the fallback reads that
  language -- the same pattern as Qwen3-TTS's Arabic fallback. CLI:
  `python -m cli piper status|models|download --approve|say "<text>" --write-wav f`
  (`cli/piper_speech.py`, the headless twin of the window's controls), mirrored as
  the GUI sidebar's two *Piper* actions and the reading-voice picker's status note.
- `core/audiocpp_speech.py` — **VoiceTut-TTS over audio.cpp**, the local Arabic
  **model voice**: the engine behind the project's `ModelVoice` kind (see
  `core/voices.py`). VoiceTut is a zero-shot voice-cloning TTS model shipped as a
  GGUF and run by the `audiocpp_cli` program from
  [audio.cpp](https://github.com/0xShug0/audio.cpp), so it runs **entirely on this
  machine** -- nothing is sent off it -- and speaks natural **Egyptian Arabic with
  Arabic/English code-switching**, which the single-voice Piper Arabic model and
  Qwen3-TTS (no Arabic at all) cannot match. It is the honest answer to "give the
  reader a voice that sounds like a person *and* reads Arabic properly".
  `AudioCppSynthesizer` satisfies `core.speech.SpeechSynthesizer`, so the window,
  Jarvis, the voice loop, and every mouth use it unchanged, and `AudioCppRunner` is
  the injection seam (the default `AudioCppCliRunner` writes the request sequence,
  runs the binary with its own folder on the loader path -- the prebuilt release
  ships `libggml*.so` beside the executable and finds them through `$ORIGIN` -- and
  copies the WAV out of the run folder), so the engine is exercised with no binary
  and no weights. The text it speaks is prepared by `core.arabic_speech`'s
  `prepare_for_speech` first, so Arabic numbers, units, ranges, and decimals are
  *spoken, not spelled*, exactly as every other engine reads them. **One voice = one
  reference clip**: `VOICETUT_SPEAKERS` names the 19 released voices and
  `voice_list` returns one `ModelVoice` per clip that is actually present (so an
  engine with no reference yields no voice rather than a silent one), each keyed
  `model:audio.cpp:<speaker>` (`MODEL_VOICE_PREFIX`). The **engine binary** lives
  under `data/audio.cpp/bin` and the **model + voices** under `data/audio.cpp`
  (`core.paths.audiocpp_models_dir`, git-ignored) with a `model.json` registry:
  `find_cli`/`find_model_file`/`model_is_cached`/`resolve_spec`/`spec_names`/
  `spec_summaries` read them. The runtime is the optional `[audiocpp]` feature
  (`FEATURE_AUDIOCPP`, `requirements-audiocpp.txt`, which is deliberately **empty**:
  audio.cpp is a native program and this module fetches it with `urllib`, so there
  is no Python package to install and the module claims none -- the honest
  `import_names=()` means the status never tells a user to run a pip install that
  cannot fix anything).
  `audiocpp_status` reports *exactly* what is installed -- engine, model, and how
  many voices -- each with its own real next step, and the two fetches
  (`install_engine`, `download_model`/`download_speakers`) run only through the
  permission + human-approval gates (`--approve`), like every other model download;
  `install_engine` stages the archive and installs the folder that actually holds
  the binary, so its shared libraries come with it. **It is not the silent default**:
  `discover_synthesizer_name` deliberately leaves it out of the automatic order
  because it is much slower than the other engines on a CPU, so making it the
  default would slow every reading on a machine that has it. It is reached
  **explicitly** instead -- picking one of its voices in the reading-voice menu (or
  naming one on the CLI) routes the reading to it through
  `core.speech._engine_for_voice` -- so a user who wants it gets it and a user who
  does not is unaffected.
  **The user's own voice** is the second half of the cloning model: `ImportedVoice`
  (`<data>/audio.cpp/imported_voices.json`, git-ignored) records a reference clip the
  user recorded or imported, `import_voice` copies it in (`_to_wav` converts a
  non-WAV with `ffmpeg`/`sox`, and a missing converter is a `ValueError` naming that
  next step, not an engine fault; an `.onnx` is refused with the "use the Piper
  studio" hint because VoiceTut clones from a clip and cannot load one), and
  `clone_voice` is the thin "my own voice" wrapper over it -- both also accept the
  microphone's live `AudioClip`, so the window's *Clone my voice* recording reaches
  the registry with no file round trip. `imported_voice_models` turns them into
  `ModelVoice`s (each keyed `model:audio.cpp:<name>`, so a cloned voice is picked
  and routed exactly like a built-in speaker), `load_imported_voices` drops a row
  whose clip is gone and tolerates a corrupt registry, and `remove_imported_voice`
  deletes the row and its clip. **The engine's own language id is not the
  project's**: VoiceTut was trained with Egyptian Arabic (`arz`) and the engine
  *rejects* a bare `ar`, so `normalize_language` maps the project's two-letter code
  onto the id the engine accepts (`ar` → `arz`, `en` → `en`, anything else → the
  model's `arz`) -- the fix for a reading that failed with "unsupported OmniVoice
  language" instead of speaking. CLI:
  `python -m cli audio status|models|voices|my-voices|import <clip>|clone
  <recording>|remove-voice <name>|install-engine --approve|download
  [--speakers] --approve|say "<text>" --voice model:audio.cpp:Sarah --write-wav f`
  (`cli/audio.py`, the headless twin of the window's voice menu), mirrored as the
  GUI sidebar's *audio.cpp* actions (status, voices, download) and a
  *VoiceTut-TTS (Local Arabic)* section in the reading-voice picker; the VoiceTut
  **studio window** (`gui/voice_studio_dialog.py`) adds a *My voice (clone)* block
  -- *Clone my voice (record)* and *Import a voice file…*, plus a remove menu --
  that only this engine's studio builds, because `VoiceStudioService.can_import_voices()`
  is true only for the zero-shot cloning engine (see below).
- `core/voices.py`'s **model voices**: a `ModelVoice` is a voice that lives inside a
  local model rather than in the synthetic catalog or the OS. It carries the
  engine's exact `name` (the speaker's own name, never translated), the `engine`
  that owns it, its `language` tag, the reported `gender`, and the `reference` clip
  (plus its transcript) a zero-shot engine clones; `key` is a stable
  `model:<engine>:<name>` string (`MODEL_VOICE_PREFIX`, `is_model_voice_key`) so a
  saved choice survives a restart and can never be confused with a catalog or
  `system:` key. `model_voices(engine=None)` lists what the local engines really
  have, `find_model_voice(key)` resolves a key against what is *currently*
  installed (returning `None`, so the caller falls back, when the model is gone),
  and `group_model_voices`/`group_system_voices` are the one language-grouping rule
  both voice menus use (Arabic first). This is the voice kind the
  `gui/app.py` sidebar's **Voices** dropdown and `gui/service.py`'s
  `model_voices`/`model_speech_status` present.
- `core/speech.py`'s two factories are the **one** place every chat interface gets
  its speech, so Qwen3 is wired there and nowhere else: `discover_recognizer_name`
  prefers `qwen3-asr` when the extra and weights are present (else `whisper`, else
  `transcript`), `load_recognizer("qwen3-asr")` builds the recognizer, and
  `load_synthesizer("qwen3-tts")` builds the synthesizer with the system engine as
  its automatic Arabic fallback (`_system_synthesizer`). `discover_synthesizer_name`
  keeps the tiny system engines first (they are installed and speak Arabic) and
  only falls to `qwen3-tts` when no system engine exists. So the GUI, Jarvis,
  `ai.voice`, `ai.speak`, and **every future mouth** get Qwen3 by construction,
  with no per-window work. CLI: `python -m cli qwen-asr status|models|download
  --approve|transcribe <file>` and `python -m cli qwen-tts status|models|download
  --approve|say "<text>" --write-wav f` (`cli/qwen_speech.py`, the headless twin of
  the window's controls), mirrored as the GUI sidebar's three *Qwen3 speech*
  actions and the reading-voice picker's status note.
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
- `core/knowledge_carry.py` — **learned graph knowledge survives a session
  boundary** (Gap 8). The persistence layer (`KnowledgeStore`/`Snapshot`/
  `Persistence`/`JsonStore`) existed but was never wired to the answer path, so the
  graph an answer *read* was rebuilt from the seeds every start and every relation
  the architecture **learned** (from a confirmed sentence, an ingested document)
  was thrown away when the process ended. This is the bridge: `LearnedRelation`
  keys a relation by its two concepts' **labels** (never an internal `CON-…` id,
  which is regenerated each run); `capture_learned(ontology, learner)` returns the
  graph's *learned* relations **and only those** — a relation counts as learned
  only when the learner created it (`is_learned`), so a curated seed is never
  captured (it is in every session already and restoring it could not change an
  answer); `restore_learned(ontology, learner, records)` replays records through
  the **same** `OntologyLearner`, so each replayed link stays bounded by the
  learner's ceiling and refuses to touch a seeded relation — a record naming an
  unknown concept is **reported skipped, never invented**, a record whose pair
  already exists is counted present (never duplicated), and a second restore is a
  no-op. `LearnedKnowledgeStore` is the durable wrapper over the `Persistence`
  protocol (one versioned document under `CARRY_NAMESPACE`/`CARRY_KEY`), with
  `capture`/`restore`/`save`/`load`; an unreadable record is dropped rather than
  crashing the session. **Nothing runs on its own** — capture and restore are
  explicit calls, so a read-only session is untouched. `CarryReport` reports what
  one restore did. Wired into the answer path: `GroundedAssistant` gains
  `remember_relation()` (writes a relation, drops the retrieval cache, captures the
  learned set), `recall_learned()` (replays the previous session's learned
  relations before the first question) and `learn_from_text()` (reads a sentence
  with the real `RelationExtractor` and captures the result); `build_default_assistant(learned=…)`
  recalls at build time; `JarvisMode`'s feedback path feeds a confirmed/corrected
  statement to `learn_from_text`; and `ai/jarvis.build_session` opens the durable
  store under `data/learned-knowledge` (`_learned_store()`, falling back to no
  carry on a read-only data directory). CLI:
  `python -m cli relations carry "<statement>" [--show] [--json]`. Measured by the
  held-out `knowledge_carry_accuracy` metric (with negative controls: a seed is
  never captured, an unknown concept is skipped, a second restore does not
  duplicate).
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
  window's page (＋ menu → "Generate an image from text…", a backend selector,
  and `python -m cli image`).
- `core/imagegen_diffusion.py` — the optional **learned** text-to-image backend:
  a real Diffusers diffusion model, tuned for photorealistic, clean pictures.
  `ImageModelPreset` + `PRESETS` (`fast` sd-turbo / `realistic` *(default)*
  dreamshaper-8-lcm / `detailed` dreamshaper-8) bundle the model, sampler
  settings, and a negative prompt; `resolve_preset` / `preset_names` /
  `preset_summaries` expose them. `DiffusionImageGenerator` takes
  `preset`/`model`/`steps`/`guidance`/`negative_prompt` (explicit values override
  the preset); its runner receives a negative prompt and snaps the size to the
  64px grid, resampling back with a high-quality filter. Weights download through
  the permission/approval gates (`download_model`, `--approve`), cached under
  `data/models` (git-ignored). Everything heavy is lazy: torch/diffusers are
  imported only when generating, so a base install never needs them (`python -m
  pip install -r requirements-image.txt`, wired as the `image` feature in
  `core/optional_deps.py`; the `pip install .[image]` form cannot work from a
  source checkout, since the project is not on PyPI and the checkout has no
  setuptools package discovery). `DiffusionUnavailable` is raised when the extra
  or the weights are missing so the caller can fall back to the procedural
  renderer. `DiffusionPipelineRunner` is the seam a test injects (no torch
  needed). The text-to-image pipeline loads with `safety_checker=None`, like the
  editor below: the default `realistic` preset's model ships a real checker, and
  it blanks a picture it misreads (a stylised prompt is a common false positive)
  to a solid black frame, which is exactly the "black preview" the window would
  otherwise show.
- `core/imagegen_diffusion.py` also carries the optional **learned image
  editor**: `InstructionImageEditor` runs `timbrooks/instruct-pix2pix` (an
  instruction-following diffusion model) over an existing image, so it can make
  a *semantic* change the deterministic pixel operations cannot — an expression
  ("make the person smile"), a pose, or the movement. `ImageEditPreset` +
  `EDIT_PRESETS` (`subtle` / `balanced` *(default)* / `strong`) bundle the two
  guidance scales and a negative prompt (`resolve_edit_preset` /
  `edit_preset_names` / `edit_preset_summaries`); lower `image_guidance` lets the
  model move further from the input. The pipeline loads with `safety_checker=None`
  (the model's own documentation): the checker blanks a picture it misreads
  instead of editing it, which would silently destroy the user's image. Edits run
  at the input size capped to 512px (snapped to the 64px grid, resized back), and
  `ImageEditRunner` is the seam a test injects. Weights download through the same
  permission/approval gates (`download_edit_model`, `--approve`), cached under
  `data/models` (git-ignored).
- `core/imageedit.py` — optional **image editing from text**, the counterpart to
  `core/imagegen.py`: it changes an image the user already has, from a written
  instruction. Dependency-free: a real PNG decoder (all five scanline filters,
  colour types 0/2/3/4/6, bit depths 1/2/4/8/16, non-interlaced) plus binary PPM,
  and a deterministic keyword interpreter (`plan_edits` → ordered `EditOp`s,
  `apply_plan` / `apply_op` over a `Raster`, `Raster.to_png` reusing
  `imagegen.encode_png`). Operations: rotate/flip, resize (bilinear), crop,
  brightness/contrast/saturation, grayscale, invert, sepia, warm/cool, blur
  (separable box), sharpen, threshold. It is a *keyword interpreter, not a learned
  model*: it applies the operations in the order the words appear and returns the
  words it did not understand (`EditPlan.unknown`) rather than guessing. `Raster`
  bridges to `core.vision.ImageBuffer`. `gui/image_edit.py` adds the Tk-free
  `ImageEditService` with two backends — `deterministic` (this keyword engine)
  and `learned` (the InstructPix2Pix editor above) — via `load` → `EditSession`,
  `apply`, `undo` (a snapshot stack, so it works for the non-replayable learned
  edits), `save_copy`, `render_working`, and an `ImageEditReport` that carries the
  `learned` flag and an honest `description`. `gui/image_edit_dialog.py` is the
  window's page (＋ menu → "Edit an image from text…"; upload, an instruction box,
  a backend selector with a model-download button, Apply/Undo/Save a copy, and a
  30%-width preview), plus `python -m cli image edit <file> "<instruction>"
  [--backend learned]` and `python -m cli image edit-model --approve`.
- `gui/image_preview.py` — the shared preview-scaling helper used by both image
  pages (`load_scaled`: Pillow when present for an exact resize, Tk `subsample`
  otherwise).
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
  adds Word); a scanned PDF is read by the default auto-OCR path
  (**Stage 3**, `allow_ocr`, via `core.pdf_ocr`), which turns on
  automatically when the optional `[ocr]` extra is installed and can be forced
  (`--ocr`) or refused (`--no-ocr`); read-only
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
  opens a record). Scanned PDF pages (**Stage 3**) are OCR'd by default when
  the optional `[ocr]` extra is present (`allow_ocr` defaults to automatic;
  `--ocr` forces it on, `--no-ocr` refuses it), through `core.pdf_ocr`: only the
  pages with no
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
  only listing formats). Scanned PDFs are reachable end to end: scanned pages are
  OCR'd automatically when the extra is present, `--ocr` / `--no-ocr` force or
  refuse it, and `--ocr-languages` / `--ocr-dpi` / `--ocr-max-pages` tune it; any
  run that skipped a scanned PDF offers to retry it with OCR on a
  terminal. Mirrored as GUI action `import-center` — which the
  window routes to the dialog below — and opened from the composer's ＋ menu (the
  action's sidebar row is hidden).
- `gui/import_dialog.py` — the Import Center **dialog** (Tk, imported lazily so
  `cli/` never imports Tk): Upload files / Import from a local folder / Import
  jobs, with a background worker thread, per-file results, retry, and a progress
  line. A **Read scanned PDFs (OCR)** checkbox (with a *force low-text pages*
  toggle and a language field) controls the OCR path for both import buttons:
  checked means automatic (scanned pages only), unchecked means never, and the
  force toggle OCRs every low-text page. A **Retry with OCR** button appears
  when a run failed on scanned PDFs. An **Install OCR support…** button appears
  when the OCR packages are missing, running the bundled requirements file so a
  desktop user never has to open a terminal. A **3. Import a SQL database**
  section opens the SQL screen below. `core/document_reader.py` reads
  TXT/Markdown (streamed above 4 MB), CSV/TSV, JSON/JSONL, digital PDF, and DOCX;
  scanned PDFs go through the default auto-OCR path in `core/pdf_ocr.py`, which
  turns itself on when the optional `[ocr]` extra is installed. Every reader
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
  the composer's ＋ menu (**Generate an image from text…**), whose sidebar action
  is hidden.
- `gui/image_edit_dialog.py` — the **Edit an image from text** screen (Tk,
  imported lazily): **Choose an image…** (PNG or PPM) loads it, an instruction
  box describes the change, and **Apply** edits the current picture (worker
  thread, then a preview scaled to the 30%-width right column), with **Undo**
  (replays the remaining instructions from the original) and **Save a copy…**. It
  says plainly that this is a deterministic keyword interpreter, not a learned
  model, and reports the operations it understood and the words it did not.
  Reachable from the composer's ＋ menu (**Edit an image from text…**); the
  headless twin is `python -m cli image edit`.
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
- `cli/studio.py` — the **Code Studio CLI**, the headless twin of the Studio
  window: `status`/`coder`/`new`/`list`/`files`/`read`/`write`/`ask`/`diff`/
  `check`/`merge`/`discard` (`python -m cli studio …`, wired in `cli/menu.py` as
  `run_code_studio` and in the menu as *Code Studio: edit and merge the project's
  own code*). `new`'s `task` is optional and the workspace is named for you
  (`PR1`, `PR2`, …) unless `--slug` forces a name. It drives the *same*
  `gui.code_studio.CodeStudioService`, so the CLI
  and the window share one set of workspaces, one diff, and one merge gate — the
  CLI reimplements no git or agent logic. `--root` selects the checkout (default:
  the current directory), `--base-branch` the branch workspaces cut from and merge
  into, `--max-steps` how many model turns one `ask` may take, `--check-command`
  the command `check` runs (one shell string, default `python -m pytest -q`), and
  `--json` for machine-readable output. Exit codes: `0` success (a `status`
  report succeeds even when the model is missing — it *states* the next step),
  `2` invalid arguments, `3` the coding model is not available (`coder`, `ask`),
  `4` a workspace/gate refused the operation, `5` an unexpected failure.
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
  the **Actions** list mirrors numbered CLI items 6..12 (never repeating the
  window's own controls, which cover items 1..5 — the window, Jarvis, **Ultron**,
  and the two chat modes). A `GuiAction` carries a `hidden` flag: ten entries
  stay in `ACTIONS` (so `action_by_key` and the CLI still resolve them) but are
  `hidden=True`, and `visible_actions()` — what the sidebar actually draws —
  leaves them out. The six *Answer from ...* mouth switches are hidden because
  the header's model selector offers them, and the Import Center, Code Studio,
  Presentation Studio, Doc's Studio, and image generation are hidden because the
  composer's ＋
  menu already opens them; `options.py` has the argument parser
  (`--mode ai|jarvics`, and the three mouth starts `--mind` / `--ultron` /
  default);
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
- `gui/typography.py` — the one **type scale**, the sibling of `Theme`'s colour
  tokens. Tk has no stylesheet, so every `font=(family, size)` and every
  `spacing1`/`spacing3` used to be a literal copied into `gui/app.py` and each
  dialog — which is how two windows drift apart. This module is the single
  source of truth: `UI_FONT_STACK`/`MONO_FONT_STACK` (OpenHands' own stacks, and
  `pick_family` takes the first the platform actually has, else Tk's
  `TkDefaultFont`/`TkFixedFont`); `SCALE`, keyed by *role* (`body` = `text-sm`
  `leading-6` like `chat-message.tsx`, `heading1`..`heading4` from
  `markdown/headings.tsx`, `chatInput`, `code`, `table`, `hero`…), each a
  `TypeStep(size_px, leading_px)`; `to_points` converts a web pixel size to the
  integer points Tk demands using the root's own `tk scaling` (so a high-DPI
  display is not shrunk to a hair); `Typography.font`/`mono` return the Tk font
  tuple (weight by role, `slant="italic"` supported) and `block_metrics`
  computes the `spacing1`/`spacing2`/`spacing3` that emulate the web
  line-height and a block's `mt-*`/`mb-*`/`py-*` gap, because a Text *tag* has no
  `-linespacing`. `resolve`/`for_root` build one for a display and fall back to
  the 96-dpi reference with no display, so it is testable headless. Tk's weight
  is only normal/bold, so OpenHands' `font-medium` headings are drawn bold; size
  and line-height match the web exactly.
- `gui/rich_text.py` — the shared Markdown **painter**. `insert_markdown(widget,
  text, theme=, typography=)` creates the tags (`configure_markdown_tags`:
  paragraph/heading1/heading2/heading3/bullet/quote/code/table/rule/bold/
  inline_code, each with the `gui.typography` size and the line/paragraph
  spacing) and paints every `StyledLine`, including the inline bold/inline-code
  spans and the code token colours. `body_spacing`/`apply_body_spacing` give the
  widget itself the body rhythm, so the *blank* line between two Markdown
  paragraphs has height (it belongs to no block tag). `gui/app.py` and
  `gui/code_studio_dialog.py` both call it, so every window draws Markdown with
  the one scale and the one palette; the painting logic used to be duplicated in
  both.
- `gui/chat_composer.py` — the shared **chat composer**, the sibling of
  `gui/rich_text.py` for the *input* side: one wide, generously rounded dark card
  every chat-style screen builds instead of growing its own panel, so the window
  and the dialogs can no longer drift apart. `ChatComposer(parent, theme=,
  typography=, placeholder=, on_submit=, with_attach=, with_status=, variable=,
  min_lines=, max_lines=)` lays out a borderless multiline `tk.Text` across the
  top and a single toolbar row beneath it: the bottom-left holds the ＋ attach
  control (only when the screen really has attachments) and whatever
  `add_mode_selector(...)` pills the caller supplies, the bottom-right holds a
  subtle non-interactive status dot, an optional `add_mic(...)` control, any
  `add_tool(widget)` action the caller built, and the circular Send disc.
  `text`/`set_text`/`clear`, `set_submit`, `set_send_enabled`, `set_status_dot`,
  `set_mic_recording`, `focus`, `set_tooltip`. Every toolbar control (＋, the mode
  pills, the mic, and the Send disc) carries a real hover state -- Tk's
  ``activebackground`` only paints while the button is held, so the tint is bound
  on ``<Enter>``/``<Leave>`` and restored to the control's own resting colour on
  leave; the disc lightens a shade derived from the theme's primary (``_blend``),
  and a disabled disc deliberately does not lighten. The message area's focus
  treatment is the caret and the I-beam cursor, not a recoloured border: the
  project keeps the composer's hairline the neutral stroke on purpose
  (``test_window_composer_has_no_gold_focus_outline``). The placeholder is an overlay
  *label*, never text inside the entry, so `text()` is always the user's own
  words and the existing handlers can keep reading `entry.get(...)`. The input
  grows with the message and then scrolls inside itself at `max_lines`, so the
  surrounding layout never jumps. Send is greyed for an empty or whitespace-only
  box and while the caller's own gate is closed. `ModeSelector` is the compact
  pill (icon + label + chevron) with a real menu of the caller's own options, so
  it never invents a mode. The panel is **presentation only** -- it never decides
  what a message means -- and it **releases what a destroyed window must not keep
  alive**: the caller's bound submit handler, its `StringVar` (held weakly), and
  that variable's trace are dropped on `<Destroy>`. Keeping them would leave the
  dead window in a reference cycle only the cyclic collector could break, and
  that can run on a worker thread, where a Tcl call deadlocks -- which is exactly
  how the Presentation Studio's export hung. `gui/app.py`, `gui/presentation_dialog.py`,
  `gui/presentation_finalize_dialog.py`, `gui/code_studio_dialog.py`,
  `gui/docs_dialog.py`, `gui/image_gen_dialog.py`, and `gui/image_edit_dialog.py`
  all build on it, each
  keeping its own attributes (`_entry`, `_question`, `_request_var`, …) pointed at
  the panel's widgets so every existing handler and test is untouched.
  `with_model_bar=True` (the main window) adds the optional **model-switch row**
  inside the card, between the input and the toolbars, reached through
  `model_bar()`: a `ModelButtonBar` of `ModelButton` chips, one per answering
  mouth. A chip is a Canvas-painted rounded rectangle (`_rounded_points`, the same
  shape `RoundedBox` uses) carrying the model's name and what it is for; a click
  runs the caller's command, `set_active(key)` paints the live choice with the
  primary accent and leaves the others plain, and a destroyed chip clears its own
  `command` for the same cycle/deadlock reason the panel's submit handler is
  dropped. The row **wraps** onto as many lines as the card's width needs (Tk has
  no flow layout): `_columns_for` measures the *column* widths `grid` will share,
  not each row's own sum, because measuring rows independently under-counts a
  layout whose first column holds a wide chip and whose later rows do not -- which
  is how a row can still overflow after it has been "wrapped". A compact window
  therefore wraps rather than cutting a model off the end of the row, because a
  chip the user cannot reach is a model the user cannot pick.
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
  box, **Conversations** grouped by date, **Settings** with the **Cloud
  provider**/**Model** dropdowns and the **Actions** list; the sandbox-tools
  list and the answering-mouth radios are deliberately *not* in the sidebar --
  the header's model selector is the one visible mouth switch) behind a
  scrollbar that
  appears only when the content overflows, a title bar showing the active mode by
  name (one switch) plus the **model selector** (`MOUTH_CHOICES`: *From sources
  (cited)* / *From trained model* / *Ultron (general local model)*, matched to the
  mouth keys `ai.chat` uses) and the three **title-bar controls** (minimize /
  maximize / close, the same trio every dialog carries, through
  `gui.window_controls`), and a centred hero/empty state that
  gives way to the first message (and returns when the transcript is cleared).
  The bar is the `Theme.titlebar` dark-grey band, like every other window's.
  On a narrow (mobile-sized) window the sidebar becomes a drawer that
  `_toggle_sidebar` shows or hides and that is restored when the window widens.
  Assistant answers are rendered as **Markdown** through `gui/markdown.py`
  (`_insert_rich`/`_insert_styled_line`/`_inline_tags`). The input is the shared
  `gui/chat_composer.ChatComposer` (the window's attributes `_entry`/`_send`/
  `_attach_button`/`_mic_button` point at its widgets), and its bottom-left
  toolbar carries the two mode pills -- *Code Studio* (opens the one real code
  surface) and the operating-mode pill, fed by `gui.modes.MODES` so it lists the
  modes the app really has. A **＋** button beside
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
  training loop never freezes the window. The composer's **＋** menu also opens
  the **Code Studio**, the **Presentation Studio**, **Doc's Studio**, and the
  image pages (below).
  The composer's card also carries the **model-switch row** (below the input,
  above the pills), one chip per answering mouth in the order asked for --
  **Jarvis (Sources)**, **Ultron (General)**, **Vision (Petroleum)**,
  **Banner (Instructor)**, **Natasha (Personality)**, **Hakeem (Religion)**,
  **Cloud (Network)** (`MODEL_BUTTONS`) -- built through
  `ChatComposer.model_bar()`. Each chip opens that model's **own dedicated chat
  window** (`_open_model_chat`) -- with its logo, its exclusive reading voice, and
  its language model -- so a user can talk to Ultron in one window and Hakeem in
  another without the two transcripts sharing a conversation; one window per mouth
  is kept and reused, so a second click raises the open one. The header's selector
  still switches *this* conversation's mouth through `_switch_mouth`, so both
  remain available, and `_sync_model_buttons` accents the live mouth (and
  re-accents it when a switch is refused, so the row always shows what is really
  answering). A chip whose model is missing still opens its window, which reports
  the real next step and lets the user choose or download a model. The
  row wraps on a narrow window rather than clipping a model out of reach, and the
  header selector still lists *every* mouth (including the small trained model),
  so nothing became unreachable.
  The same row's **trailing group** carries one chip per **speech engine**
  (`VOICE_ENGINE_BUTTONS`) -- *Installed Voices*, *VoiceTut-TTS*, *Piper*, and
  *Qwen3-TTS* -- pinned to the far right, ordered right to left, with a gap
  (`ModelButtonBar.GROUP_GAP`) so they never touch the model buttons. Each opens
  that engine's own **studio window** (`_open_voice_studio`, one per engine kept
  and reused), where a document is loaded and read aloud with a voice and a mood
  exclusive to that engine; `model_bar()` exposes the two groups separately
  (`model_keys()`/`engine_keys()`), and `_cells` puts a real spacer cell before the
  trailing chips so the gap survives the wrap.
  The window also **reads text aloud**: every message footer carries a
  "🔊 Read aloud" control (`_speak_label`/`_speak_text`), and the sidebar holds the
  reading-voice menus. A dedicated **Voices** dropdown (`_toggle_voices`, right
  under Chats) puts the voice choice one click away, and the Settings panel holds
  the same list (`_build_voice_picker`); both call the one
  `_build_voice_sections`, so the two menus can never drift apart. Each voice sits
  under the **source** it came from -- *Installed Voices*, *VoiceTut-TTS*, *Piper*,
  *Qwen3-TTS*, and the synthetic catalog (`ChatService.voice_sections` →
  `core.voices.voice_sections`) -- with the catalog grouped under
  Men/Women/Boys/Girls; a source that contributed no voice is replaced by one
  honest note and its real next step (`ChatService.voice_source_notes`, built from
  `model_speech_status`/`piper_speech_status`/`qwen_tts_voice_status`), an opt-in
  **auto-speak** check, and a "Test the reading voice" button. Choosing a `model:`
  voice is what routes a reading to that local engine. The chosen
  voice and the auto-speak flag persist through `JsonStore`
  (`load_reading_voice`/`save_reading_voice`, `load_auto_speak`/`save_auto_speak`,
  the same `gui`/`appearance` namespace as the theme; a `system:` or `model:` key is
  kept as-is), the window and the service
  agree from the first message (`ChatService.set_voice`), and Markdown is stripped
  (`_strip_markdown`) so the engine speaks words, not `**`/backticks. Reading runs
  on the worker thread and reports honestly: a missing engine says to install
  espeak-ng, and a silent render is named as such rather than shown as success.
- `core/language_models.py` — the **one language-model catalog** the whole
  program reads. Before it, each mouth knew only its *own* models
  (`core.ultron.CATALOG`, `core.vision_model.CATALOG`, `core.banner`/
  `core.presentation_model`, `companion.model`, `core.cloud_model`, the trained
  checkpoint), so no surface could offer "all the project's models" and no two
  could agree on a default. `all_language_models()` unifies them: a
  `LanguageModel` carries `key`, `title`, `engine`, `summary`, `available`,
  `detail`, `path`, `provider`, plus `engines` (every mouth the model belongs to),
  `catalog_key`, `folders`, `filename`, `identity` (its repo/filename),
  `download` (the real fetch command), and `custom` (a model the user added
  themselves, see `core/custom_models.py`). Entries are **de-duplicated by
  identity**,
  so the *same* weights reachable from two engines (`qwen3-4b` is Banner's,
  Natasha's, and Hakeem's) are one model listing all three, never three entries
  pointing at one file. `models_for_choice(mouth)` is what a dedicated mouth
  window offers: every GGUF model the project has for an in-process mouth (they
  share the one runtime, so any downloaded file can answer in any of them, the
  mouth's own default listed first, the user's own `custom:` entries included),
  and only its own models for a cloud or the
  retrieval mouth. `can_download(model)` says whether the chosen model has a
  fetch (a catalog model does; a `custom:` one does not) and
  `download_model(mouth, key, approved=, progress=)` is the one dispatcher that
  runs it through the **owning engine's** own approval gate.
  `find_language_model`/`resolve_choice`,
  `default_language_model_key`, `describe_language_models`, and the mouth
  constants (`MOUTH_SOURCES`/`ULTRON`/`VISION`/`BANNER`/`NATASHA`/`HAKEEM`/
  `CLOUD`/`MODEL`, `MOUTHS`, `GGUF_MOUTHS`, `ENGINE_LABELS`) round it out.
  **Honest by construction**: availability is a real file check (`available` is
  true only when the GGUF is on disk), `detail` always names the real next step
  when it is not, an unknown mouth or key returns nothing rather than raising, and
  nothing here loads a model or imports Tk.
- `core/custom_models.py` — the **hand-added language models**: a small `JsonStore`
  registry of GGUF files the user pointed a per-model window at, so a model the
  project does not ship can still answer. `CustomModel` (key `custom:<slug>`,
  title, `path`), `add_model(path, title=)` (validates a real `.gguf` file, keeps
  the file **in place** -- a multi-gigabyte GGUF is referenced, never copied --
  and refuses a missing/non-GGUF path with a `ValueError` rather than storing it),
  `remove_model`/`list_models`/`find_model`, and `is_custom_key`. A file the user
  later moves is reported unavailable, because the real path is checked on every
  read; a corrupt registry reads as empty so a window always opens. Dependency-free
  and Tk-free; `core/language_models` merges these entries into its catalog.
- `core/mouth_profile.py` — the per-mouth **saved settings**: the character's
  `name`, its `logo`, the reading `voice` exclusive to it, and the language
  `model` it answers with, one `MouthProfile` per mouth persisted through
  `JsonStore` under the `mouths` namespace. `load_profile`/`save_profile`,
  `set_name`/`set_voice`/`set_model`, `save_logo`/`clear_logo`/`logo_filename`
  (a chosen picture is copied into `data/characters`, named `<mouth>.<ext>`, the
  old copy removed, an unsupported extension refused with a clear message), and
  `resolve_voice`/`resolve_model` (a saved voice or model that has left the
  catalog falls back to the default rather than leaving the menu broken). A
  corrupt or unreadable profile yields the default, so a window always opens; a
  copy or save failure is recorded through `core.best_effort.best_effort` rather
  than swallowed. `core/paths.characters_dir()` is the git-ignored
  `data/characters` folder it uses. Dependency-free and Tk-free.
- `ai/mouths.py` — the one place that turns `(mouth, model key) -> a ready
  assistant`. `build_mouth` resolves the chosen catalog entry to its exact GGUF
  file (or its cloud provider + model name) and hands *that* to the mouth's own
  loader, so a model picked from another engine's catalog still reaches the right
  file; `model_choice`/`describe_mouth_model` expose the resolution.
  `MODEL_MOUTHS` names the model-backed mouths; a retrieval-only mouth has no
  model and `build_mouth` refuses it honestly. An unknown key means "the mouth's
  own default", never a wrong file, so a stale saved choice cannot break a mouth.
- `gui/model_chat.py` — the per-mouth chat **service** (Tk-free), the engine
  behind `gui/model_chat_dialog.py`. `ModelChatService(mouth, *, service=,
  data_root=, reader=)` wraps a `ChatService` (its own session) and the mouth's
  saved profile, and keeps a `TextReader` of its own so the **reading voice is
  exclusive** to that mouth -- choosing one here never changes the main window's.
  It exposes the identity (`name`/`set_name`, `logo_path`/`set_logo`/
  `clear_logo`), the voice (`voice_key`/`set_voice`/`voice_sections`/
  `voice_label`/`read_aloud`), the model (`model_choices`/`model_key`/
  `set_model`/`model_ready`/`model_status`/`add_model`/`remove_model`/
  `can_remove`/`can_download`/`download_command`/`download_model`), the mouth's
  activation (`active`/`activation_status`, applied on open so this window's
  session really answers with this mouth), the conversation
  (`ask`/`new_conversation`/`save`/`session`), and the microphone
  (`microphone_available`/`start_recording`/`stop_recording`/`transcribe`).
  `MOUTH_BLURBS` gives each mouth an honest one-line description. `service` may
  be injected, so the whole workflow is exercised headless with no display, no
  model, and no microphone.
- `gui/model_chat_dialog.py` — the **dedicated chat window** one answering mouth
  opens from the composer's model row. Built the way every other window is (the
  same `Theme` tokens, `RoundedBox`, `gui.rich_text` Markdown, the shared
  `gui.chat_composer.ChatComposer`, and the three `gui.window_controls` title-bar
  buttons), it has a **left sidebar** holding, top to bottom, (1) the character's
  **logo** -- clickable to upload another picture, with a *Reset to the default
  badge* action and an editable **name** -- (2) the **reading voice** as an
  exclusive radio list grouped by the source each voice came from, and (3) the
  **language model** as a list of every model the mouth can run, the saved choice
  pre-selected and each entry labelled with its real availability, beside three
  controls: *Add a language model…* (a GGUF file picker that adds and activates the
  chosen file at once), *Download the chosen model* (live only when the chosen
  model has something to fetch, run on the worker thread with approval), and
  *Forget this model* (live only for a `custom:` entry the user added). The centre is
  the transcript and the composer, which carries an optional **microphone**
  (`add_mic`) exactly like the main chat. Every message's footer carries the same
  **Copy** and **Read aloud / Stop** controls the main window draws (through
  `gui.message_footer.MessageFooter`), so a per-mouth window reads a message aloud
  in its own exclusive voice and a second click stops it
  (`ModelChatService.start_reading`/`stop_reading`, chunked by
  `core.reading.split_for_speech` so the reading ends at the next sentence
  boundary and the sound stops at once). The window opens at 90% of its original
  height (`DEFAULT_HEIGHT`, capped to the work area so the composer stays
  reachable on a small screen) and both the sidebar and the conversation carry a
  **scrollbar that is always visible** rather than one that appears only on
  overflow, so neither column is ever a dead end; the title bar's maximize fills
  the work area. Long work runs on a worker thread whose
  result is handed back through a queue the UI thread drains (Tk is not
  thread-safe). Every control reports the real next step when its engine or model
  is absent rather than pretending. `gui/app.py` opens one per mouth from the
  composer's model row (`_open_model_chat`, reusing an already-open window), and
  `ChatService.set_mouth_model`/`mouth_model` keep the choice in force for the
  mouth that is answering.
- `cli/model_chat.py` — the headless twin of that window: `python -m cli
  model-chat mouths|models|profile|set-model|add-model|remove-model|
  download-model|set-voice|set-name|set-logo|
  clear-logo|ask` (aliases `model-chat` and `mouths`). `add-model <mouth> <file>`
  adds a GGUF file and makes it active, `remove-model` forgets a `custom:` entry,
  and `download-model <mouth> <key> --approve` fetches a catalog model through its
  owning engine's own approval gate. It drives the *same*
  `gui.model_chat.ModelChatService`, so the CLI and the window share one saved
  profile per mouth -- one name, one logo, one reading voice, and one language
  model -- and no logic is reimplemented. Exit codes: `0` success, `2` invalid
  arguments, `3` the model is not available, `4` the profile or logo could not be
  read or written, `5` an unexpected failure.
- `core/voice_moods.py` — the **emotional states** a reading voice can be put into,
  and the *prosody* that produces them. `Mood` names a well-known state (happy,
  sad, excited, rapturous, in love, angry, calm, tender, serious, surprised,
  fearful, hopeful, nostalgic, proud, playful, whisper, storytelling, formal,
  empathetic, and the neutral reading) with an English and Arabic label, an honest
  description, and a `Prosody`; `Prosody` is the engine-neutral shape of that
  state (`pitch_delta`, `speed_scale`, `range_delta`, `gap_delta`, `energy`, and
  an `instruct` style line), each field a *delta* or *multiplier* from the chosen
  voice's own settings, so a mood layers onto whatever timbre the user picked
  rather than replacing it. The numbers are bounded (`MIN_*`/`MAX_*`) so no single
  setting can turn speech into noise, and `Prosody.neutral` says when a state
  leaves the voice untouched. `mood_keys`/`default_mood`/`find_mood`/`resolve_mood`
  (a saved key that left the catalog falls back to neutral rather than raising)/
  `mood_summaries`, and `apply_mood(voice, mood)` folds the numeric part onto a
  catalog `Voice` (clamped to the catalog's own limits) while leaving a named
  `SystemVoice`/`ModelVoice` alone (its engine picks it by name and takes the
  prosody directly). `ENGINE_MOOD_SUPPORT`/`support_note(engine)` is the one place
  the honest per-engine truth is written down -- a formant engine takes pitch,
  speed, melody, and pauses; Piper takes pace and expressiveness; Qwen3-TTS takes
  the style instruction; the VoiceTut cloning engine takes little and says so.
  Dependency-free and import-safe.
- `core/voice_studio_profile.py` — **per-engine** speech-studio persistence: the
  chosen voice, the mood, the name, and the logo of each of the four engine
  studios, each kept separate from the others. `STUDIO_SOURCES`/`source_names`/
  `studio_title` name the four (`Installed Voices (Arabic + English)`,
  `VoiceTut-TTS (Local Arabic)`, `Piper (Arabic + English)`, `Qwen3-TTS (English)`)
  with the titles the user asked for; `VoiceStudioProfile` is the frozen record
  (`with_name`/`with_voice`/`with_mood`/`with_logo`, `logo_path`, `to_primitive`),
  stored one per engine through `storage.json_store.JsonStore` under the
  `voice_studios` namespace. `load_profile`/`save_profile` (a save is
  `best_effort`, so a read-only data folder never crashes a window, but the
  failure is recorded), `set_name`/`set_voice`/`set_mood`, and the logo helpers
  (`save_logo` copies the picture into the git-ignored `data/characters` as
  `voice-<source>.<ext>`, so it can never collide with a per-mouth logo, and
  removes any replaced one; `clear_logo`). `resolve_voice`/`resolve_mood` fall a
  saved choice that is no longer offered back to the first available voice / the
  neutral mood, so a stale file can never make a studio unusable. Nothing here
  loads a model, imports Tk, or touches the cognitive core.
- `gui/voice_studio.py` — the speech-engine studio **service** (Tk-free), the
  engine behind `gui/voice_studio_dialog.py`, the sibling of
  `gui.model_chat.ModelChatService` for the *engines* rather than the mouths.
  `VoiceStudioService(source, *, reader=, data_root=)` wraps one engine's saved
  profile and a `TextReader` of its own, so the **reading voice is exclusive** to
  that studio. Identity (`title`/`name`/`set_name`/`logo_path`/`set_logo`/
  `clear_logo`), voices (`voices`/`voice_groups`/`catalog_voices`/`voice_keys`/
  `voice_key`/`set_voice`/`voice_label`, drawn from **only** what that engine
  really reports here -- the synthetic catalog is offered by the *Installed
  Voices* studio alone), moods (`moods`/`mood_key`/`set_mood` -- applied at once
  and persisted -- `mood_label`/`mood_support_note`), reading
  (`reading_available`/`reading_status`/`load_document`/`document_text`/
  `read_aloud`/`save_speech`/`close_document`, a long document split into
  paragraph-sized chunks), and the microphone (`microphone_available`/`asr_status`/
  `start_recording`/`stop_recording`/`transcribe`). `reader` may be injected, so
  the whole workflow is exercised headless with no display, no model, and no
  audio hardware.
- `gui/voice_studio_dialog.py` — the **studio window** one speech engine opens
  from the composer's trailing row. Built the way every other window is (the same
  `Theme` tokens, `RoundedBox`, `gui.rich_text` Markdown, the shared
  `gui.chat_composer.ChatComposer`, the three `gui.window_controls` title-bar
  buttons, and a worker-thread/queue the UI thread drains), its **left sidebar**
  holds, top to bottom, exactly what the user asked for: (1) the engine's **logo
  and name** (the logo is clickable to upload another picture, with a *Reset to
  the default badge* action, and the name is editable in place), (2) the **reading
  voice** as an exclusive list drawn only from that engine's own voices, and (3)
  the **mood** as a menu of the readable state names, applied the moment it is
  chosen. Below them the studio holds a **document** (*Open a document…* loads it
  through the project's own reader, the text appears in the transcript, and
  **Read aloud** plays it), and the composer carries an optional **microphone**
  (`add_mic`). `gui/app.py` opens one per engine from the composer's trailing row
  (`_open_voice_studio`, reusing an already-open window).
- `cli/voice_studio.py` — the headless twin of that window: `python -m cli
  voice-studio list|status|voices|moods|voice|mood|logo|name|say` (aliases
  `voice-studio`, `voice-studios`, `speech-studio`). It drives the *same*
  `gui.voice_studio.VoiceStudioService`, so the CLI and the window share one voice
  choice, one mood, and one document reader, and no workflow logic is
  reimplemented. Exit codes: `0` success, `2` invalid arguments, `3` the engine is
  not available, `4` the document or the choice was refused, `5` an unexpected
  failure.
- `core/coder_model.py` — the coding-model client, the one place the code model
  is named. `DEFAULT_CODER_MODEL = "qwen2.5-coder:7b"` (a local **Ollama** model),
  `CoderMessage` (a validated chat turn), the `CoderClient` protocol
  (`list_models`/`chat`), and `OllamaCoderClient` — a thin, dependency-free
  `urllib` client for Ollama's `/api/tags` and `/api/chat` (**no `requests`**),
  so the base install pulls nothing new. `CoderUnavailable` is raised when the
  server is unreachable or the model is missing, and `coder_status` reports the
  truth as a `CoderStatus` (`available`, `server_reachable`, `installed`,
  `models`, and the real next step in `hint`): a missing server and a missing
  *model* are distinguished, and `_model_present` accepts Ollama's implicit
  `:latest` tag but never claims a different size of the same family is present.
  `PULL_HINT` is the one string every surface shows for the next step.
- `core/code_studio.py` — the **Code Studio** engine: change the project's *own*
  code safely. `CodeStudio(root, workspaces_dir=..., base_branch="main")` makes a
  **`git worktree`** per change under `data/studio/workspaces/<slug>` on its own
  `studio/<slug>` branch cut from `main`, so the running checkout is never edited
  and a half-finished change cannot break the program. The worktree folder
  belongs to the checkout being edited: the project's own checkout uses the
  bundled, git-ignored `data/studio/workspaces`, and any other root keeps its
  worktrees under its own `data/` (a shared folder would register another
  repository's worktree in this one's state). A workspace is **named
  automatically, in order** — `PR1`, `PR2`, … — so the user never invents one:
  `create(task, slug=None)` calls `_next_slug`, whose monotonic counter lives
  under a reserved `__next_serial__` state key (never a workspace, so every
  reader skips it via `_workspaces_only`) and only ever moves forward (a
  discarded number is never reused); an explicit `slug` is still honoured. A
  `Workspace` (slug,
  branch, path, base revision, task, check state) is persisted in a small state
  file, and `create`/`get`/`list_workspaces`/`remove` manage its life cycle.
  `list_workspaces` orders newest first with `_newest_first` (the timestamp, then
  the auto serial, then the slug — so same-second workspaces still sort
  honestly).
  `read_file`/`write_file`/`list_files`/`changed_files`/`diff`/`commit` work
  inside it; every path goes through a **deny-by-default confinement** check
  (`../escape.py`, an absolute path, and a path that climbs out all raise) plus a
  text-extension allow-list and a size cap, so the model cannot escape the folder
  or drop a binary into the project. `run_checks` runs the project's own test
  command *inside the workspace* and records the verdict; `can_merge` is the gate
  — it refuses while the checks have not passed, when the workspace is empty, when
  `main` has moved on (so a merge can never overwrite newer work), when the
  checkout is on another branch, and with uncommitted changes — and `merge`
  fast-forwards `main` to the workspace's commit. `_git` runs git
  non-interactively and supplies a commit identity **only for its own commits**
  (`_noninteractive_env(identity=True)`), so a machine with no global git identity
  still works. `StudioError` is the one failure type.
- `core/code_agent.py` — the coding agent: drives a `CoderClient` over a workspace
  through a strict JSON action protocol (`list_files`, `find_files`, `search`,
  `read_file`, `write_file`, `edit_file`, `diff`, `done`), validating `path`/
  `content`/`old`/`new`/`query` per action and returning an `AgentTrace` (one
  `AgentStep` per turn, the writes, the summary, and the reason it stopped).
  `search` greps the workspace and `find_files` matches a glob, so the model can
  locate code before it changes it, and `edit_file` replaces an exact `old`
  snippet with `new` — a targeted edit that cannot silently rewrite a whole file
  the way a stale `write_file` payload can (a duplicate write of identical content
  is reported, not re-written, so it never shows up twice in the diff). `diff`
  lets the agent review its own work before calling `done`. It tolerates a fenced
  reply and surrounding prose, pushes back on a bad reply rather than dying,
  truncates a large read (`read_chars`), stops at `max_steps`, and reports an
  unreachable model honestly. `run(..., on_step=...)` calls the callback with each
  `AgentStep` as it happens, so the window and the CLI can show the work live
  instead of only when the run ends. A refused write is a *failed step in the
  trace*, never a crash — the confinement in `core/code_studio.py` is what actually
  holds, and the agent only surfaces it.
- `gui/code_studio.py` — the Code Studio **service** (Tk-free): `CodeStudioService`
  wraps the engine, the coder client, and the check command, and returns plain
  reports (`create_workspace`, `list_files`, `read_file`, `save_file`,
  `changed_files`, `diff`, `run_checks`, `can_merge`, `merge`, `remove_workspace`,
  `run_agent`, `state`, `status`). `create_workspace(task="", slug=None)` needs no
  name and may open the workspace before the first request. `run_change(slug,
  task, on_step=…, on_checks=…)` is **one whole chat turn**: it runs the agent
  (streaming each step through `on_step`), then — when the model actually changed
  something — calls `on_checks` and runs the project's checks, returning a
  `ChangeResult` whose `ready_to_apply` (a real change *and* passing checks) is the
  honest gate the window turns into the user's approval. It is the one engine the
  window and the CLI
  share, so both see one set of workspaces, one diff, and one merge gate. The
  coder client is built lazily, so constructing the service never reaches Ollama.
  `status` is derived from the **backend** it resolves once, so the in-process
  model is offered first and a machine with no Ollama is never told to start a
  server it never needed (`test_service_status_offers_the_in_process_model_first`);
  the Ollama wording appears only when Ollama is the runtime that can answer.
- `gui/code_studio_dialog.py` — the Code Studio **window** (Tk, imported lazily):
  built the way the chat window is built, so the Studio reads as part of the
  program rather than a separate tool — a slim header, a left **sidebar** for the
  workspace list + *Start a new change*, the file list,
  and *Apply to the project* / *Discard change* / *Run checks* with their
  state, and a centred conversation column with the rounded composer card pinned
  to the bottom (`_COLUMN_MAX`/`_column_pad`, the same theme tokens and
  `RoundedBox`). Opening the window **opens the first workspace for you**
  (`_ensure_workspace`, named `PR1`), so the composer is ready to use and there is
  no name field; after an apply or a discard it opens the next one silently, and
  the sidebar hint says the names are assigned in order. The model's step trace and
  the diff/check output are transcript
  messages like a chat reply, not separate tabs: `_set_trace` adds a "Coding
  model" row and `_set_review` a "Diff and checks" row, both monospaced, through
  the same `_add_message`/`_fit_height` path the chat window uses. *Send* now
  reads as a chat turn: `_add_message("user", request)` echoes the request as
  the user's own bubble (and clears the composer), then `_run_background` passes
  `on_step=self._on_agent_step`, and `_drain_steps` appends each `AgentStep`'s
  `live_lines()` into a live "Coding model" bubble as the run proceeds — the same
  first-by-first detail the OpenHands chat shows — before the final `_set_trace`
  supersedes it. The run drives `run_change`, so the checks follow the model in
  the same turn, and `_offer_apply` then asks for the user's approval (it enables
  **Apply to the project** and posts a "Ready to apply" note only when
  `ready_to_apply`; a failing or empty run says why and applies nothing). The file
  editor is a slide-in panel above the composer
  (`_show_editor`/`_hide_editor`). Long
  work runs on a worker thread whose result is handed back through a **queue the
  UI thread drains** (`_schedule_pump`), because calling Tk's `after` from a
  worker thread races with the main loop; Apply stays disabled until the gate
  allows it, and a message the user just needs to read (a save, a model run, an
  automatic workspace open) is kept rather than overwritten by the gate hint
  (`_refresh_merge_state(keep_note=True)`; `_run_background(note=False)`).
  Reachable from the composer's **＋** menu (**Code Studio — edit and merge the
  project's own code…**), whose sidebar action is hidden; the headless twin is
  `python -m cli studio`.
- `core/presentation.py` — the **Presentation Studio** content model and the
  dependency-free `.pptx` writer. A project walks a strict state machine
  (`DISCUSSION` → `OUTLINE_DRAFT` → `OUTLINE_REVIEW` → `SLIDE_DRAFTING` →
  `SLIDE_REVIEW` → `CONTENT_APPROVED` → `EXPORTING` → `EXPORTED`, plus
  `EXPORT_FAILED`), enforced by `require_transition`, so a deck can only reach
  `EXPORTED` through the user's own approvals. Content types are
  `CONTENT_TITLE`/`SECTION`/`PARAGRAPH`/`BULLETS`/`EQUATION`/`IMAGE`/`TABLE`
  (`resolve_content_type` infers one from what a slide actually holds). A `Deck`
  is the export view of the approved slides. `PresentationWriter` renders a real
  Office Open XML package with **only the standard library** (`zipfile` +
  `xml.etree`): the content types, both relationship files, the presentation, one
  slide master, one layout, a theme, the numbered slides, and a `notesSlide` per
  slide that has speaker notes. The package is assembled in memory and written
  atomically (a temporary file replaced into place), so a failure never leaves a
  half-written deck; `render`/`validate_pptx` re-open the bytes and prove the
  package is well-formed. All text is XML-escaped, so Arabic, mixed
  Arabic/English, equations (`μ`, `Σ`), and `&`/`<`/`"` round-trip intact.
  `bullets_from_text`/`title_from_text`/`outline_from_prose` turn an assistant
  answer into slide content, and `topic_sentence` derives a heading from a
  sentence while dropping the answer's own discourse openers ("First,",
  "Moreover,") via `_DISCOURSE_OPENER_RE` — `strip_discourse_opener` is that same
  pass exposed on its own, so a *bullet* also reads as a fact rather than as the
  assistant narrating its own list (it returns text with no opener unchanged, so
  nothing is re-cased). `dedup_key` is a bullet's comparison key (case, whitespace,
  a trailing stop, and a leading connective ignored) so a repeat is recognised
  across slides. `bullets_from_text` treats a line
  before the first bullet as the answer's *title* line (which the slide prompt
  asks for), so the same words never appear as both the title and the first
  bullet. A `Slide` carries its place in the deck — `section`/`subsection` (the
  outline topic it was developed from, `Slide.topic` is the readable
  `"Section / Subsection"` key and `slide_topic_key()` builds it) alongside its
  stable `slide_id` and its current `order`/`number`, so a slide's identity, its
  number, and the outline place it belongs to are all explicit and survive a
  reorder. `parse_slide_range` reads a plain range ("7", "11-15", "11 to 15"),
  and `parse_slide_command` turns one plain-language instruction into a
  `SlideCommand(action, slides, anchor, before, text)` — the shared parser behind
  the window's command box and the CLI's `command` subcommand, so a typed
  "shorten slides 11-15" and a spoken one do the same thing.
  `default_presentation_dir()` is `<data>/presentations`, git-ignored;
  `presentation_filename` slugifies the title.
- `core/presentation_layout.py` — the **first half** of the export engine, with
  no rendering dependency. It turns an approved `Deck` into a `DeckPlan`:
  `classify_layout` picks one of twelve named layouts (`title`, `section`,
  `objectives`, `bullets`, `paragraph`, `two_column`, `table`, `equation`,
  `chart`, `image`, `summary`, `conclusion`) from an explicit `Slide.layout`
  hint, a title cue ("Objectives"/"Summary"/"Conclusion", and the Arabic
  equivalents), a `|` separator in a bullet (two columns), or the content type --
  deterministically, nothing guessed. `estimate_fit` measures the prepared text
  against the slide geometry (an average-glyph-width estimate, documented as
  conservative) and returns a `FitEstimate`, so an over-full slide is **flagged**
  with a warning, never clipped or shortened; `prepare_deck`/`prepare_slide`
  build the `SlidePlan`/`DeckPlan` (blocks: title, bullets, paragraph, table,
  columns, image, chart, equation) with `warnings` and `overflows`. `validate_image`
  checks an image reference against an extension allow-list and a size cap and
  returns an `ImageCheck`, so a missing or unusable file is *skipped* with a
  readable reason rather than crashing the export (it reads only the name and
  size -- an image is never decoded or executed). `safe_filename`/`safe_output_path`/
  `unique_export_path` make a path-traversal-proof, never-overwriting output name.
  `PresentationSettings` (aspect, theme, slide numbers, footer/author/company/
  subject, `rtl`, `allow_overwrite`) and the four `THEMES` (`professional` default,
  `light`, `academic`, `dark`) live in `core/presentation.py` and round-trip
  through `to_primitive`/`from_primitive`; `DeckTheme` carries the fonts, sizes,
  colours, and spacing. `looks_rtl` detects right-to-left script so a mixed deck
  is drawn correctly.
- `core/pptx_engine.py` — the **second half**: the optional, real PowerPoint
  renderer. `PptxEngine.write`/`render` consume a `DeckPlan` and draw a genuine,
  editable Office Open XML deck with **python-pptx** (the `[pptx]` extra, installed
  from `requirements-pptx.txt`, also bundled in `requirements.txt` so CI exercises
  the real path, like `pypdf`): a 16:9 or 4:3 canvas, twelve drawn layouts, real
  shapes, native tables, `COLUMN_CLUSTERED` charts (a non-numeric "chart" slide
  degrades to its table rather than going blank), embedded images scaled to fit,
  speaker notes, slide numbers, a footer, and document metadata. `rtl` is honoured
  by setting `a:pPr/@rtl` and the `a:cs` complex-script typeface, so Arabic is not
  substituted. The package is assembled in memory and written atomically (a
  temporary file replaced into place), so a failure never leaves a half-written
  deck. **python-pptx is imported lazily**: `pptx_available` reports whether it is
  present and `_require_pptx` raises `PptxUnavailable` with the exact install
  command when it is not, so a base install and CI still work and the
  dependency-free `core.presentation.PresentationWriter` remains the fallback.
  `validate_deck` re-opens the written file and compares the slide count, order,
  titles, and notes against the plan, so "it exported" is proven rather than
  assumed. Nothing in the deck is ever executed.

- `gui/presentation.py` — the Presentation Studio **service** (Tk-free), the one
  engine the window and the CLI share, so both see one set of projects, one
  revision history, and one export gate. `PresentationService` (`new_project`,
  `discuss`, `draft_outline`/`review_outline`/`approve_outline`,
  `draft_slides_from_outline`, `review_slides`, `edit_slide`/`add_slide`/
  `add_slides`/`insert_slide`/`reorder_slide`/`move_slide`/`remove_slide`/
  `renumber_slides`, `regenerate_slides`/`revise_slides`/`add_example_after`/
  `apply_slide_command`,
  `approve_slide`/`unapprove_slide`/`approve_slides`/`approve_slides_in_range`/
  `approve_content`, `revisions`/`restore_slide_revision`/`restore_approved`,
  `export`, `export_history`, `course_matches`/`best_course`/`course_detail`/
  `course_concepts_text`/`slides_for_concept`, `status`) persists projects through
  `JsonStore`
  under `PRESENTATIONS_NAMESPACE`. It takes an injected `ask` callback and never
  builds a model itself: the CLI passes `ai.jarvis.build_session` and the window
  passes *its own* session (`ChatService._presentation_ask`), so there is **no
  second AI** and no second knowledge base. `set_batch_size`/`batch_size` cap how
  many slides one `draft_slides_from_outline` call adds (default 5,
  `DEFAULT_SLIDE_BATCH`), so a long deck is developed in reviewable pieces; the
  draft expands each outline subsection into its own numbered slide and records
  the `section`/`subsection` on it, so the topic sequence is preserved.
  `approve_slides_in_range` approves a numbered span (clamped to the deck);
  `approve_slides` approves the whole deck and, like `approve_slides_in_range`,
  refuses an **empty slide by number** rather than skipping it — silently leaving
  one unapproved made the content gate fail later with a message that did not say
  which slide was at fault, which is how the step 6 → step 8 workflow appeared
  broken;
  `insert_slide`/`move_slide` renumber densely while `slide_id`s stay stable;
  `renumber_slides` closes any gap; `regenerate_slides`/`revise_slides`/
  `add_example_after` re-draft through the assistant keeping the slide's id,
  number, and heading, and `apply_slide_command` runs a parsed
  `parse_slide_command` instruction through those same methods (returning a
  `SlideCommandResult`). Every edit records a numbered
  revision and snapshots the approved content, and `_reopen_for_edit` returns an
  approved or exported project to `SLIDE_REVIEW` and clears the content approval
  — so a change after approval is never silent and the export gate re-arms.
  `get` also recovers a project found in `EXPORTING`: an export interrupted
  between the state write and the finished file (a crash or a kill) would
  otherwise be stranded forever, because `EXPORTING` has no outgoing edge except
  to `EXPORTED`/`EXPORT_FAILED`. The approved content is intact, so it is
  returned to `CONTENT_APPROVED` and the export may simply be run again; a
  regression test covers the recovery and the edit-after-interruption path.
  `_is_title_line` distinguishes a model's short title line from the grounded
  assistant's prose first sentence, so a prose answer takes the outline heading as
  the slide's title instead of repeating its first sentence. `draft_slides_from_outline`
  drops a drafted bullet another slide already shows (`dedup_key`), because the
  retriever can return the same passages for two related headings; when *every*
  bullet repeats it keeps the section's single strongest line and sets
  `REPEATED_HEADING_NOTE`, so the slide is flagged for the user rather than left
  blank or silently duplicated. `export` requires
  `CONTENT_APPROVED`, writes only the approved slides in their approved order,
  records an `ExportRecord` (including a failure), and moves the project to
  `EXPORTED` or `EXPORT_FAILED`. It selects a render engine through
  `resolve_engine`: `"pptx"` (the optional python-pptx engine, preferred when
  installed) or `"stdlib"` (the always-available fallback); an explicit request
  for an absent engine is refused honestly, while the automatic choice falls back
  so a base install never breaks. `settings` overrides the stored look for one
  export, and the output is **never overwritten** unless `overwrite=True` (or
  `settings.allow_overwrite`): with no explicit path a unique, safe name is
  chosen. `PresentationReport` carries the engine, the filename, the slide
  `warnings`, and a `download` reference (path, filename, directory, size) -- the
  desktop app's real "download", since it has no web server. `export_warnings`
  previews the notes an export *would* raise without writing a file, and
  `update_settings` persists a project's look.
- `gui/presentation_dialog.py` — the Presentation Studio **window** (Tk, imported
  lazily), built the way the chat and Code Studio windows are built (the same
  theme tokens, `RoundedBox`, and a centred column capped at `_COLUMN_MAX`). A
  left sidebar lists the projects (each row carries its id, last-updated date,
  slide count, and state) and the eight workflow steps with their state, beside
  **New presentation** / **Rename** / **Duplicate** / **Delete** management
  buttons (a management button is live only while a project is selected, and
  **Delete** asks for confirmation); a
  centred conversation column holds the transcript, the question box, and the
  slide editor. The slide list shows each slide's number, title, and outline topic
  and a **Draft N slides per batch** spinbox sets the service's batch size. The
  editor edits the title, type, section/subsection, body, and notes, with
  Save / Approve / Withdraw approval / Move up / Move down / Regenerate /
  Revisions… / Remove, and a **Command** box that runs a plain instruction
  ("shorten slides 11-15") through the shared `apply_slide_command`. An
  **Approve all slides** button approves the whole deck in one save (an empty
  slide is reported by number, never silently approved), so step 7 does not
  require a click per slide. **Approve the content** is enabled only from
  `SLIDE_REVIEW` (the first state the state machine accepts an approval from —
  offering it while the slides were still being drafted is what made the workflow
  look broken), approves every slide, moves the project to *Content approved*, and
  then opens the **final presentation editor** window
  (`gui/presentation_finalize_dialog.py`) through `on_open_final_editor`; a failure
  there is shown in the transcript *and* the status line, so an empty slide names
  itself. The sidebar's `_model_note`/`_model_line()` say which model drafts the
  outline and slides (the Studio's own Qwen3 GGUF, else the project's assistant)
  and how to install the better one. A **Course content** section below the model
  note holds **Open the course…**, enabled only once the deck's own content matches
  one of the project's courses (`_sync_course` names the course and how many of its
  concepts are covered, or says to draft the slides first) — so the Course window
  (`gui/presentation_course_dialog.py`) opens at a meaningful moment, never at
  random. The Export section holds the export gate plus **Check the deck** (runs
  `validate_project`, reporting issues/warnings with no file written),
  **Preview…** (renders the exact approved slides, in order, as text), and
  **Open the folder** (reveals the last validated export in the OS file manager;
  enabled only once a real export exists). **Export the .pptx** validates the deck
  *before* it writes, so a numbering gap or an unapproved slide is reported rather
  than producing a bad download. Message rows
  are measured the way the chat window measures its
  bubbles: `_bubble_fonts`/`_bubble_text_width` convert the column's *pixels* into
  the *characters* a `tk.Text` `width` is counted in (rounding **down**, so a row
  can never request more pixels than the column has), `_bubble_card_width` sizes a
  user bubble's rounded card, and `_fit_height` caps a tall message to its own
  scroll region. A resize re-measures every row (`_reflow` → `_remetric_rows`),
  because a `tk.Text` keeps the character width it was built with. Long work runs
  on a worker thread whose result is handed back through a queue the UI thread
  drains. Reachable from the composer's **＋** menu (**Presentation Studio — build
  a PowerPoint deck…**), whose sidebar action is hidden; the headless twin is
  `python -m cli presentation`.
- `cli/presentation.py` — the Presentation Studio CLI, the headless twin of the
  window: `new`/`list`/`rename`/`duplicate`/`delete`/`validate`/`test`/`show`/
  `status`/`discuss`/`outline`/`outline-draft`/
  `outline-edit`/`outline-revise`/`review-outline`/`approve-outline`/`draft`/
  `add`/`edit`/`reorder`/`remove`/`review-slides`/`approve-slide`/
  `unapprove-slide`/`approve-slides`/`approve-content`/`reopen`/`revisions`/
  `restore`/`restore-approved`/`export`/`warnings`/`settings`/`demo`/`history`
  (`python -m cli presentation …`, wired in `cli/menu.py` as `run_presentation`).
  `list` prints a dashboard table (id, updated date, slide count, approved count,
  status); `rename`/`duplicate`/`delete` manage projects; `validate` runs the same
  structural check the window's **Check the deck** runs and exits `4` when the
  deck is not ready; `test` runs the presentation test files through pytest
  (`--json` for a machine-readable report) so a user can verify the feature
  in-place.
  It drives the *same* `gui.presentation.PresentationService` the window drives,
  so the CLI and the window share one set of projects, one revision history, and
  one export gate — no workflow logic is reimplemented here. Its discussion,
  outline, and slide draft call the project's *existing* assistant through
  `ai.jarvis.build_session(text_only=True)` (`_make_ask`), and with no model
  installed they still work when the text is supplied directly (`--text` /
  `--text-file` / `--title` / `--bullet`). A project's discussion **resumes
  across runs**: `_make_ask` loads the conversation id the project last used
  (`_load_conversation_id`, a `JsonStore` pointer under `<root>/conversations/map`)
  and calls `resume_session`, saving the session after every turn, so returning to
  a project in a later process keeps the earlier exchange instead of starting
  over. Exit codes: `0` success, `2` invalid
  arguments, `3` the assistant is not available, `4` a gate refused the
  operation, `5` an unexpected failure. It also carries **`model-status`** and
  **`model-download`** (`--list-models`, `--approve`), the headless twins of the
  Studio's own authoring model: `model-status` reports honestly (exit `4` when
  neither the in-process model nor a reachable Ollama can answer, printing the
  real next step), and `model-download` fetches a catalog model only through the
  permission + human-approval gates. `_maybe_author(root)` builds the Studio's
  authoring model when one is installed and passes it to
  `PresentationService.set_author`, so drafting prefers the Studio's own Qwen3
  model and falls back to the project's assistant when it is absent.
- `core/presentation_model.py` — the Presentation Studio's **own authoring model**,
  added the way Ultron's and the Code Studio's are: a real **GGUF file loaded into
  this process** with `llama-cpp-python` (no server, nothing sent off the
  machine), the runtime being the optional `[presentation]` extra
  (`core.optional_deps.FEATURE_PRESENTATION`, `requirements-presentation.txt`,
  `presentation = [...]` in `pyproject.toml`), imported lazily so a base install
  needs neither the extra nor a model file. The model is a GGUF under
  `data/presentation` (`core.paths.presentation_models_dir`, git-ignored):
  `find_model_file` prefers the file named in the registry (`model.json`, written
  on download), else the largest `*.gguf`. `DEFAULT_PRESENTATION_MODEL` is
  **`qwen3:8b`** and `CATALOG` names the fetchable choices (`qwen3-8b` default,
  `qwen3-4b` the lighter option), exposed through `spec_names`/`spec_summaries`/
  `resolve_spec`. `LlamaCppPresentationClient` satisfies the same
  `core.coder_model.CoderClient` protocol as `OllamaCoderClient` (so the Studio
  cannot tell the two runtimes apart) and carries `backend = "in-process"`, which
  is why `coder_status` never tells an in-process user to start a server;
  `PresentationRunner` is the seam a test injects, so the client is exercised with
  no extra and no weights. `download_model` runs only through the permission +
  human-approval gates (`approved=True`), like the other model downloads.
  `presentation_backend_status`/`presentation_model_status` report *exactly* what
  is installed -- in-process, Ollama, or nothing, with the real next step and no
  mention of the Code Studio's coding model. Nothing here executes model output;
  the Studio parses the text with the same deterministic parsers it uses for any
  other answer.
- `ai/presentation_author.py` — adapts the Studio's authoring model to the
  service's `Callable[[str], str]` `ask` shape: a question in, plain text out. It
  exists as its own layer because the client speaks in messages while the service
  wants one string, so this is where the system prompt is added *once* (asking for
  exactly the shape the Studio's parsers read) and where a reasoning model's
  ` thinking…` blocks are stripped (`strip_thinking`, which also drops an
  unterminated block to the end so a reply cut off mid-thought never leaks
  reasoning into a slide; when nothing is left the original text is returned, so
  the answer is never silently empty). `PresentationAuthor.__call__` is
  deliberately **stateless** -- one self-contained question per call, so a retry
  or a re-draft cannot be skewed by an earlier exchange -- and `describe()` names
  the model and runtime honestly. `load_presentation_author` prefers the
  in-process GGUF, honours a reachable Ollama serving the same model otherwise
  (so a user who already pulled it there is not asked for a second copy), and
  raises `PresentationModelUnavailable` with the real next step when neither can
  answer.
- `core/presentation_courses.py` — the **curriculum bridge**: which of the
  project's *own* courses a deck covers, so the Studio can open the Course window
  at a meaningful moment. `match_courses(text, min_matches, limit)` matches a
  deck's own words against every `CourseKnowledge` concept in
  `core.curriculum_knowledge` (the same seed the Continuous Internet Learner
  navigates), keeping only **whole-concept** matches — a concept counts only when
  *all* of its words appear (Arabic-normalised through `core.rag._query_terms`,
  never a substring), so "porosity" alone never counts as "effective porosity" —
  and returns `CourseMatch(key, label, branch, matched, total)` ordered strongest
  first, with `match_count`/`coverage`/`strong` (`coverage >= STRONG_COVERAGE`)
  and an honest `describe()`. `match_project(project)` builds the deck's text from
  its title, outline, and slides (never the discussion, which may wander) and
  matches it; `best_course(matches)` is the one *strong* match, or `None`, so a
  weak incidental overlap never nominates a course. `course_detail(key, text=...)`
  returns the course's `ConceptInfo`s (each marked `present`) and readable
  `RelationInfo`s (`describe()` renders "source —relation→ target"), and raises
  `KeyError` on an unknown key rather than showing a blank course. Dependency-free
  and read-only: it invents nothing and never touches a project.
- `gui/presentation_course_dialog.py` — the **Course window** (Tk, imported
  lazily), a *sibling* of the Studio that drives the *same*
  `gui.presentation.PresentationService`. Opened by the Studio through
  `on_open_course` once the deck's own content matches a course strongly enough
  (`best_course`), it lists the courses the deck covers (strong ones marked) and
  the chosen course's concepts with the covered ones ticked, beside the course's
  own relations, in the chat window's theme and centred column. Two honesty rules:
  it **never edits the deck** — *Use the missing concepts* hands the uncovered
  concepts to the Studio's question box (the user then decides whether to ask) and
  *Go to its slides* selects the slide that covers a concept — and it re-reads
  everything from the one service on every refresh, so it can never disagree with
  the Studio. The listbox selection uses the *most recent* index (Tk's
  `selection_set` adds to the selection). Reachable headless as
  `python -m cli presentation courses <project_id> [--course <key>] [--limit N]`
  (`cli/presentation.py`), which prints the same matches, or one course's
  covered/missing concepts and relations.
- `gui/presentation_finalize_dialog.py` — the **final presentation editor** window
  (Tk, imported lazily), a *sibling* of the Studio that drives the *same*
  `gui.presentation.PresentationService`, so the two views can never disagree.
  Opened by the Studio's **Approve the content** step (through
  `on_open_final_editor`), it lists the approved deck's slides on the left and
  holds the slide editor, the deck-wide look controls, and the export gate on the
  right: edit a slide's title/body/notes, reorder it (Move up/down), re-approve
  the deck, then **Export the .pptx** or **Save as…**. An edit re-arms the export
  gate (the service's `_reopen_for_edit`), so the export button disables itself
  and says why until the slides are approved again; `on_edit` tells the Studio
  behind to refresh, so both views stay in step. **Check the deck** and
  **Preview…** report into the same transcript rather than a dialog. The export
  handler deliberately shows no modal confirmation -- a blocking prompt is wrong
  for a long worker-thread action and would hang a headless run -- and reports the
  written path, size, and engine in the transcript, with **Open the folder** to
  reveal it. Reachable from the Studio's approve step; there is no separate CLI
  command, because `python -m cli presentation export` is the same gate.
- `core/docs_model.py` — **Doc's Studio's own vision model**, added the way
  Ultron's, Vision's, the Code Studio's, and the Presentation Studio's are: a real
  **GGUF file loaded into this process** with `llama-cpp-python` (no server,
  nothing sent off the machine), the runtime being the optional `[docs]` extra
  (`core.optional_deps.FEATURE_DOCS`, `requirements-docs.txt`, `docs = [...]` in
  `pyproject.toml`), imported lazily so a base install needs neither the extra nor
  a model file. It is the project's one **vision-language** mouth: the model
  *sees* a page image as well as reads text, which is what makes a scanned PDF or
  a photographed page answerable. `DEFAULT_DOCS_MODEL` is **`qwen3-vl-4b`** and
  `CATALOG` names the fetchable choices (`qwen3-vl-2b` light, `qwen3-vl-4b`
  default, `qwen3-vl-8b` strong), exposed through
  `spec_names`/`spec_summaries`/`resolve_spec`. **A vision model is two files**:
  the language GGUF *and* its multimodal projector (`mmproj-*.gguf`), so every
  `DocsModelSpec` carries both `filename` and `mmproj_filename`; `find_model_file`
  prefers the registry's file and otherwise takes the largest `.gguf` **that is not
  a projector**, and `find_mmproj_file` takes the projector by its `MMPROJ_PREFIX`
  -- the two halves can never be confused, which is why "the largest .gguf" is not
  simply handed to the language loader. `model_is_cached` is true only when **both**
  halves are present, and `docs_model_status` reports each absence separately with
  its own real next step (no model, a model with no projector, no runtime, or
  ready). `LlamaCppDocsClient` carries `backend = "in-process"` and
  `LlamaCppDocsRunner` is the seam a test injects, so the client is exercised with
  no extra and no weights; `download_model` runs only through the permission +
  human-approval gates (`approved=True`), like every other model download. The
  model is a GGUF under `data/docs` (`core.paths.docs_models_dir`, git-ignored).
  Nothing here executes model output.
- `core/docs_studio.py` — the Doc's Studio **document engine** (Tk-free,
  dependency-free at import): it turns *any* supported file into one
  `LoadedDocument` of `DocumentPage`s and builds the vision messages a question
  needs. `load_document` dispatches by suffix -- plain text/data (`.txt`, `.md`,
  `.csv`, `.json`, ...) through `core.document_reader`, Word (`.docx`/`.docm`)
  through the same reader, a **digital PDF** page by page through
  `core.pdf_extract`, a **plain image** (`.png`/`.jpg`/...) as a one-page document
  whose content *is* the picture -- and refuses a legacy `.doc`/`.rtf` with the
  "save as .docx" hint, and an unknown suffix, rather than guessing.
  **The scanned-PDF path is the point**: `PdfPageStream` yields every page and
  only *then* raises "no extractable text" when no page had any, so `_load_pdf`
  keeps the pages it collected and renders each to an image with
  `render_page_image` (`pypdfium2`, lazy) -- a wholly scanned report therefore
  becomes readable instead of failing, and with rendering refused the failure
  names that next step. When rendering was *attempted* but no page produced an
  image (the classic missing-`pypdfium2` case), the failure names **that** reason
  -- the render note, not the generic "no extractable text ... needs OCR" line,
  which used to report a scanned file as an empty document with no explanation.
  A page with a real text layer is never replaced by an
  image; a page without one is rendered and marked `from_image`, so
  `LoadedDocument.vision_pages` names exactly the pages the model must *look* at.
  `select_pages` ranks the pages worth attaching for a question (a page that names
  the question's words first, a page needing vision ahead of a text-only page at
  the same score, ties in document order), and `build_vision_messages` puts the
  extracted text in front of the selected pages' images as OpenAI-style
  `image_url` parts (a `file://` URI), in document order and bounded by
  `max_pages`/`max_text_chars`, stating plainly when a page has no text layer.
  `TRANSCRIBE_QUESTION` is the fixed instruction that asks the model to *transcribe*
  a page rather than answer about it -- what turns a scanned page into readable
  content -- and `batch_pages` groups 1-based page numbers into bounded batches
  (`MAX_PAGES_PER_REQUEST`) in order, so a long scan is read in pieces.
  `file_uri` builds the page image's ``file://`` URI with ``Path.as_uri()``, so it
  works on Windows too: ``f"file://{path}"`` looks right but puts a drive path
  (``C:\\...``) in the URI's *authority* and leaves its path empty, which is the
  ``WinError 3 (cannot find the path '')`` a Windows user hit -- ``as_uri``
  percent-encodes the path and keeps the drive in the path, the form
  ``urllib.request`` opens on every platform.
  `parse_page_range` reads a plain selection ("7", "1-3", "1 to 3", "2,5,7") so
  the window and the CLI share one parser. `SUPPORTED_SUFFIXES` /
  `UNSUPPORTED_SUFFIXES` are the one place the accepted formats are declared.
- `gui/docs_studio.py` — the Doc's Studio **service** (Tk-free), the one engine
  the window and the CLI share, so both see one set of open documents and one
  model. `DocsStudioService` holds the opened documents (`open_document`,
  `documents`, `get`, `close_document`, `close_all`), asks the vision model
  (`ask` -> `DocsAnswer` with the pages used and an honest `summary()`), lets the
  caller supply the project's own assistant for the grounded path
  (`set_ask`/`ask_assistant`), saves the extracted text (`save_text`), and
  reports the model honestly (`model_status`, `model_names`, `model_summaries`,
  `download_model`). **`read` is what puts a scanned document's *contents* on
  screen**: it transcribes the document's image pages with the fixed
  `TRANSCRIBE_QUESTION` (never an answer-a-question prompt), in `batch_pages`
  batches, and returns a `DocReading` (text, pages read, or an honest failure).
  The reading is remembered on the `OpenDocument` (`reading`), so
  `document_text`/`save_text` return the text layer **plus** the transcription --
  otherwise "save the extracted text" would save an empty file for exactly the
  documents that need the model. An empty model reply becomes the named
  `DOCS_EMPTY` message rather than a blank answer, and the vision runner is an
  injectable seam, so the whole service is exercised with no extra and no weights.
  `picker_filetypes()` is the one file-dialog filter, derived from
  `SUPPORTED_SUFFIXES`.
- `gui/docs_dialog.py` — the Doc's Studio **window** (Tk, imported lazily), built
  the way the chat, Code Studio, and Presentation Studio windows are built: the
  same `Theme` tokens, `RoundedBox`, `gui.rich_text` Markdown painting, the shared
  `gui.chat_composer.ChatComposer`, the three `gui.window_controls` title-bar
  buttons, and a centred conversation column capped at `_COLUMN_MAX`. A left
  sidebar holds **Open a document…** / **Open a folder…** / **Close this document**
  / **Read the document (scanned pages)** / **Save the extracted text**, the
  open-documents list, the reading-voice-free
  page controls, and the model note with its real next step; the conversation
  column shows the transcript and the composer, and a right column previews the
  current page's image (a `tk.PhotoImage` scaled through `gui.image_preview`, or an
  honest note when the page has a text layer). Opening a document reads it on a
  worker thread whose result is handed back through a **queue the UI thread
  drains** (Tk is not thread-safe), the document's text and its vision pages are
  reported, and asking a question runs `DocsStudioService.ask` the same way and
  posts the answer under the honesty header. **Opening a scanned document
  auto-reads it**: `_maybe_auto_read` calls `DocsStudioService.read` and posts the
  transcription as a *contents* row, so a scan shows its content instead of only a
  page image -- and when the model is not ready it states the real next step
  (`Reading needs the model`) rather than leaving a blank-looking document. The
  sidebar **Read the document (scanned pages)** button re-reads on demand. Long
  work never blocks the window.
  Reachable from the composer's **＋** menu (**Doc's Studio — read any document
  (incl. scanned)…**), whose sidebar action is hidden; the headless twin is
  `python -m cli docs`.
- `cli/docs.py` — the Doc's Studio CLI, the headless twin of the window:
  `status`/`models`/`open`/`text`/`read`/`pages`/`ask`/`download` (`python -m cli docs …`,
  wired in `cli/menu.py` as `run_docs` and in the menu as *Doc's Studio: read any
  document (incl. scanned) with the local vision model* plus a second entry to
  download the model). It drives the *same* `gui.docs_studio.DocsStudioService` the
  window drives, so the CLI and the window share one set of documents, one model,
  and one download gate -- no document logic is reimplemented here. `read` is the
  headless twin of the window's auto-read: it transcribes the scanned pages (a
  `--pages` selection, progress on stderr) and prints the contents under the
  honesty header. `--json` gives
  machine-readable output for `models`/`open`/`ask`/`read`/`pages`, `--pages` selects
  pages with the shared `parse_page_range`, and `--no-render` refuses to rasterise
  a scanned PDF (so the CLI can say "this needs rendering" without doing it).
  Exit codes: `0` success, `2` invalid arguments, `3` the vision model is not
  available (`ask`/`read`/`download`), `4` the document could not be read, `5` an
  unexpected failure. A `status` report succeeds even when the model is missing --
  it *states* the next step.
- `core/scene3d.py` — the **Scene Studio** scene model: a local, DAZ-style 3D
  figure-and-scene workspace. `Vec3`/`Mat4` (a full 4x4 transform with
  compose/invert/transform-point, so a rig can build a world matrix per bone),
  `Material`, `Transform`, `Camera`
  (a look-at view matrix + a perspective projection matrix, `frame` to fit a
  bounding sphere), `Light` (directional/point, `direction_to_light`),
  `Mesh` (vertices, faces, per-vertex normals, `triangle_count`/`face_count`,
  `bounds`/`center_and_radius`, `transform`), `Bone`/`Skeleton` (a parented rig;
  `world_matrices` composes each bone's world transform from its rest matrix and
  the pose, `segments` gives each bone's head/tail in world space,
  `compute_skin_weights`/`skin_mesh`/`skin_normals` bind a vertex to its nearest
  bones),
  `SceneObject` (a mesh + material + transform + optional skeleton/pose/bindings;
  `is_figure`, `bind`, `deformed_vertices` skin the mesh from the pose,
  `world_vertices` applies the object transform on top), and `Scene`
  (`add_object`/`find_object`/`remove_object`/`object_names`, `visible_objects`,
  `summary`, `center_and_radius`, `frame_camera`, `width`/`height`/`background`),
  plus `iter_triangles`/`make_default_scene`.
  Dependency-free: only `math` and the standard library, so the whole engine runs
  on a fresh clone with no GPU and no model.
- `core/scene3d_figures.py` — the **parametric humanoid**. `FigureParameters`
  (height, build, limb lengths, proportions) + `FIGURE_PRESETS`
  (`adult`/`athletic`/`slim`/`child`) build a `SceneObject` whose mesh is
  generated in code and bound to a real bone hierarchy (`hips → spine → chest →
  neck → head`, `shoulder/upper_arm/forearm/hand.L/R`,
  `thigh/shin/foot.L/R`); `build_figure`/`build_figure_from_preset`/
  `build_figure_mesh`/`build_figure_skeleton`/`build_figure_scene`/
  `figure_preset_names`, and `morph_figure` (rebuild from new parameters, keeping
  the pose and transform). Because the mesh is skinned, posing a bone really
  moves its vertices. (The named pose presets themselves live in `gui/scene3d.py`,
  below, so the service is the one place a pose is applied.)
- `core/scene3d_props.py` — the **props and primitives** generated in code:
  `PROP_CATALOG` (a ground plane, a platform, a backdrop) with `build_prop`/
  `prop_names`/`default_ground`, and `PRIMITIVES` (`cube`/`sphere`/`cylinder`/
  `cone`/`plane`) with `build_primitive`/`primitive_names`; `MATERIALS` (a
  material palette) with `material_for`/`material_names`. Nothing is downloaded.
- `core/scene3d_render.py` — the **CPU software rasteriser**: `Framebuffer`
  (RGB + z-buffer, `clear`/`set_pixel`/`to_png` through `core.imagegen.encode_png`),
  `shade_point` (Blinn-Phong with an ambient floor; two-sided so a thin surface is
  still lit), `SoftwareRenderer` (z-buffered triangle rasterisation, back-face
  culling, optional wireframe) with `render`/`render_with_stats`, and
  `render_scene`/`render_scene_png`/`RenderStats`. Honest by construction: it says
  it is a real rasteriser with **no GPU, no ray tracing, and no commercial render
  engine** (`HONESTY_LINE`).
- `core/scene3d_io.py` — **Wavefront OBJ/MTL** I/O: `load_obj`/`load_obj_text`
  (an OBJ with no `o`/`g` line is one unnamed mesh under the file's name, and the
  material in force is recorded per group, so a multi-material file does not hand
  every object the last one), `save_obj` (geometry in its current pose, with an
  MTL sidecar), `parse_mtl`/`save_mtl`, `LoadReport`, `SceneIOError`, and
  `mesh_filetypes`/`MESH_SUFFIXES`. Bounded by `MAX_VERTICES`/`MAX_FACES`/
  `MAX_FILE_BYTES`; it parses bytes only and never executes anything.
- `core/scene3d_serialize.py` — the native **`.faydscene`** (JSON) format:
  `scene_to_dict`/`scene_from_dict`, `object_to_dict`/`object_from_dict` (a
  figure's skeleton and pose round-trip, and the skin is re-bound
  deterministically), `save_scene`/`load_scene_file`, `scene_filename` (a
  filesystem-safe name), `SCENE_SUFFIXES`, and `SCENE_FORMAT_VERSION`. Vectors are
  written at full precision, so a reloaded scene is bit-for-bit the same shape.
- `gui/scene3d.py` — the Scene Studio **service** (Tk-free), the one engine the
  window and the CLI share, so both see one scene, one renderer, and one command
  vocabulary. `SceneStudioService` opens a session (`new_scene`,
  `open_model`/`open_path`), builds (`add_figure`/`add_custom_figure`/
  `morph_object`/`add_prop`/`add_primitive`), poses (`bone_names`/`pose_bone`/
  `clear_pose`/`apply_pose_preset`), frames and orbits (`frame_camera`/
  `set_camera`/`orbit_camera`), renders (`render` → a real PNG under
  `renders/` with `RenderReport`/`RenderStats`, `render_to_bytes` for a preview),
  saves and exports (`save_scene`/`export_obj`/`list_saved_scenes`, plus
  `save_working`/`load_working` for the CLI's persistent working scene), reports
  (`object_summaries`/`status`/`mesh_filetypes`), and runs the **command box**
  (`run_command`). A **bare save/export name is a name, not a location**:
  `save_scene("Demo")` resolves into the Studio's own scenes folder and
  `export_obj("out.obj")` into its `library/` folder, each with a safe filename
  (anything naming a location -- an absolute path, or a path with a directory
  part -- is honoured as given), so a save or an export can never drop a
  `.faydscene`/`.obj`/`.mtl` in whatever directory the program happened to be run
  from -- the regression that put `Cmd.faydscene`/`Kept.faydscene` in the
  checkout.
  `SceneCommand`/`parse_command` are the one deterministic
  parser (`PRIMITIVE_NAMES`, `POSE_PRESETS`, `COMMAND_VERBS`, `command_help`), so
  a typed "add figure athletic" and the CLI's `command` do the same thing; an
  unknown figure/prop preset is a typed `SceneStudioError`, and an unknown
  instruction returns the help text rather than guessing. It creates the
  missing-folder save target itself and refuses a non-positive render size
  honestly (`RenderReport.ok is False`, never a crash). A render **publishes its
  PNG atomically** (`_write_atomic`: a sibling temp file renamed into place), so a
  concurrent reader -- the window's `_refit_viewport` re-reads the viewport file
  while a second render writes it -- sees either the old whole image or the new
  whole one, never a truncated one. A plain `write_bytes` truncates first, which
  is what made a preview vanish (`load_scaled` returned `None`, `_photo` stayed
  `None`) and failed a window test intermittently; the atomic swap closes that
  race and a failed write leaves the previous image intact with no temp litter.
- `gui/scene3d_dialog.py` — the Scene Studio **window** (Tk, imported lazily),
  built the way the chat, Code Studio, Presentation Studio, and Doc's Studio
  windows are built: the same `Theme` tokens, `RoundedBox`, `gui.rich_text`
  Markdown painting, the shared `gui.chat_composer.ChatComposer`, the three
  `gui.window_controls` title-bar buttons, and a centred conversation column. A
  left sidebar holds New scene / Open a model or scene… / Save the scene… /
  Export OBJ…, the figure presets, the prop catalog, and the pose presets; a
  viewport previews the current render (a `tk.PhotoImage` scaled through
  `gui.image_preview`, with a wireframe toggle); a right column lists the objects,
  the selected figure's bones, and the ± X/Y/Z pose buttons; and the composer at
  the bottom is the same command box the CLI exposes. Long work (a render, a
  model load) runs on a worker thread whose result is handed back through a
  **queue the UI thread drains** (Tk is not thread-safe). The honesty line is in
  the transcript from the first moment, so the window never overstates what it
  renders. Reachable from the composer's **＋** menu (**Scene Studio — build,
  pose, and render a 3D figure…**), whose sidebar action is hidden; the headless
  twin is `python -m cli scene3d`.
- `cli/scene3d.py` — the Scene Studio CLI, the headless twin of the window:
  `status`/`figures`/`props`/`materials`/`primitives`/`poses`/`list`/`help`/
  `new`/`add-figure`/`add-prop`/`add-primitive`/`pose`/`pose-preset`/
  `clear-pose`/`frame`/`orbit`/`render`/`demo`/`save`/`export-obj`/`open`/
  `command` (`python -m cli scene3d …`, wired in `cli/menu.py` as `run_scene3d`
  and in the menu as *Scene Studio: build, pose, and render a 3D figure locally*
  plus a second entry to render the built-in figure to a PNG). It drives the
  *same* `gui.scene3d.SceneStudioService` the window drives, so the CLI and the
  window share one scene and one renderer -- no scene logic is reimplemented here.
  The working scene is **stateful across invocations**: each command loads
  `working.faydscene`, acts, and saves it, so `add-figure` then `pose-preset`
  then `render` build one scene. `--json` (per subcommand, like `cli/docs.py`)
  gives machine-readable output, and `demo` builds and renders a whole scene in
  one call. Exit codes: `0` success, `2` invalid arguments, `3` nothing to pose
  (no figure), `4` a file could not be read or written, `5` an unexpected
  failure. `core/paths.py` gains `scene_studio_dir`/`scene_studio_scenes_dir`/
  `scene_studio_renders_dir`/`scene_studio_library_dir` (all under the
  git-ignored `data/scene3d`), so a saved scene or an imported mesh never dirties
  the working tree.
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
  never executes the document. The **system half** has an installer of its own,
  mirroring `core.speech`'s: `ocr_install_plan(languages=…)` returns an
  `OcrInstallPlan` (a platform package-manager command — apt/dnf/pacman/apk,
  `brew install tesseract tesseract-lang`, `winget install -e --id
  UB-Mannheim.TesseractOCR`, or `choco`) with an honest `hint` when the app knows
  none, and `install_ocr_engine(runner=…, languages=…)` runs it through an
  injectable runner (never raises) and re-checks `tesseract_available()`.
  `parse_ocr_languages` is the one parser (`"ara+eng"` → `("ara","eng")`, empty →
  `DEFAULT_OCR_LANGUAGES` = English + Arabic), `ocr_package_names(manager,
  languages)` is the one place the engine *and* the per-language trained-data
  packages are named per manager (Debian `tesseract-ocr-ara`, Fedora
  `tesseract-langpack-ara`, Arch `tesseract-data-ara`, Alpine
  `tesseract-ocr-data-ara`; macOS's single `tesseract-lang` and Windows' installer
  cover every language), and `missing_ocr_languages` reports which requested codes
  Tesseract cannot actually see — so the install fetches the **language data in
  the same run** rather than leaving the user to discover `tesseract-ocr-ara`
  separately (the exact dead end this closes), and an install that leaves Arabic
  missing says so honestly. `missing_ocr_python_packages` and
  `tesseract_available` split the two halves so the app installs the pip packages
  first and the engine + data second. `OCR_SYSTEM_PACKAGES_BY_MANAGER` is derived
  from `ocr_package_names`, so the plan and the messages cannot disagree.
  (System packages are listed in `OCR_SYSTEM_PACKAGES`; the extra is `[ocr]`.)
- `core/optional_deps.py` — the **optional-feature installer**, so a user who
  hits "OCR/PDF is not installed" has a next step the app can actually perform.
  It owns, per feature (`pdf`, `ocr`, `docs`), the bundled requirements file, the import
  names to check, and the pip command that works from a source checkout
  (`python -m pip install -r requirements-<feature>.txt`) — because the project
  is **not on PyPI**, the old `pip install fayd-al-ruh[ocr]` hint could never
  work. `status` separates a missing *Python* package (pip can fix) from a
  missing *system* package (Tesseract; pip cannot), so the app never promises an
  install it cannot do; `install` runs the command and returns an
  `InstallResult`, and takes an injected `runner` so it is tested without the
  network. Dependency-free and import-safe. `requirements-pdf.txt` /
  `requirements-ocr.txt` mirror the extras, and the `audiocpp` feature
  (`FEATURE_AUDIOCPP`) points at `requirements-audiocpp.txt` and reports the
  `audiocpp_cli` program and the VoiceTut-TTS GGUF as what pip cannot fetch (it
  declares no Python package, so pip is never named for it). Surfaced as the CLI
  `python -m cli deps [--install pdf|ocr] [--ocr-languages eng+ara]` and the
  window's **＋ → Install OCR support…** entry and the Import Center's **Install
  OCR support…** button; for OCR, `--install ocr` runs the engine step too —
  installing the engine **and its language data** (English + Arabic by default,
  `--ocr-languages` to choose) in one command — and `deps` prints the engine plan
  when only the system half is missing, while both the window and the Import
  Center install whichever half is absent, so the pip packages, Tesseract, and its
  language data are no longer separate dead ends.
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
  built-in corpus. `ChatService.attach_file` also writes the file's *whole* text
  into the conversation as a `CONTEXT` turn, so Ultron (no retriever) reads it
  too; detaching (`remove_attachment`), a new chat, and a resume drop that turn
  with the documents. `ocr=True` reads a scanned PDF through the same opt-in OCR
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
- Phase 5b — optional image *editing* from text (`core/imageedit.py`): a
  dependency-free PNG/PPM decoder and a deterministic keyword interpreter that
  changes an uploaded picture from a written instruction, with an honest list of
  the words it did not understand. Wired as the ＋ menu's "Edit an image from
  text…" window (`gui/image_edit_dialog.py`) and `python -m cli image edit`.
- Phase 5c — a *learned* image editor: `core/imagegen_diffusion.py`'s
  `InstructionImageEditor` (InstructPix2Pix) makes semantic changes the keyword
  interpreter cannot — an expression, a pose, the movement — from an instruction.
  The edit window and `python -m cli image edit --backend learned` offer it beside
  the deterministic backend, with a `subtle`/`balanced`/`strong` preset, a
  model-download button, and snapshot undo; the weights download only through the
  approval gate (`python -m cli image edit-model --approve`).
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
- Phase 10 — read the chat aloud (text to speech): `core/voices.py` is a 16-voice
  catalog (men, women, boys, girls ×4), `core/reading.py`'s `TextReader` speaks any
  text with a chosen voice through the installed system engine, and the window
  gains a per-message "Read aloud" control, a grouped voice menu, an opt-in
  auto-speak flag (all persisted), and `python -m cli speak|voices`. Every voice is
  honestly described as engine settings, not a recording. The engine is a system
  package, so when it is absent the window's sidebar and the CLI offer to install
  it (`install_speech_engine`, `python -m cli speak --install-engine`) instead of
  only naming it, and `python -m cli deps` reports it beside the other features.
- Phase 11 — the Qwen3 speech models (ASR + TTS) for every chat interface:
  `core/qwen_speech.py` adds **Qwen3-ASR** (52 languages, **including Arabic**) and
  **Qwen3-TTS** (expressive, ten languages, **no Arabic**) as real local engines,
  wired into `core/speech.py`'s `load_recognizer`/`load_synthesizer` so the GUI,
  Jarvis, `ai.voice`, `ai.speak`, and every future mouth get them by construction.
  Both run locally through the optional `[qwen-asr]` / `[qwen-tts]` extras (two,
  because the two engines pin different exact `transformers` versions and cannot
  share one environment), the weights are a
  one-time human-approved download under `data/qwen-speech`, and each engine is a
  seam a test injects. Arabic is handled honestly: ASR transcribes it, and TTS
  reads it through the system engine rather than the model. Surfaced as
  `python -m cli qwen-asr|qwen-tts`, the GUI sidebar's three *Qwen3 speech*
  actions, and a status note in the reading-voice picker.
- Phase 12 — the **real** installed voices, not invented ones: `core/voices.py`
  gains `SystemVoice` (a voice the OS reports by name, e.g. `Microsoft Hoda`, with
  a stable `system:<engine>:<name>` key) and `core/speech.py` gains
  `system_voices()` (Windows OneCore/SAPI, macOS `say -v ?`), so the voice menus
  and `python -m cli speak list` show the voices the machine **actually** has —
  grouped by language, Arabic first — instead of the synthetic espeak catalog
  (Adam/آدم, Layla/ليلى), which is now shown below as the fallback. A chosen real
  voice is selected by its exact name in every engine (SAPI `SelectVoice`, OneCore
  `DisplayName`, `say -v`). The reported silent (`0.00 s`) reading is fixed too:
  when the selected Windows engine produces an empty clip (all-OneCore machines
  look empty to SAPI), the sibling engine is tried before the silent stand-in.
  Phase 12b hardens the listing so the real voices cannot hide: the two Windows
  voice stores are **unioned** rather than one being chosen, the WinRT listing and
  availability probe tolerate a non-terminating warning (`$ErrorActionPreference
  ='Continue'` + `try/catch{}`, which used to abort the probe and report "no
  OneCore"), a chosen `SystemVoice` is routed to the engine that reported it
  (`_engine_for_voice`), and the SAPI→OneCore fallback no longer gates on the probe
  — so a machine whose only Arabic voice is OneCore is read aloud even when the
  WinRT probe is a false negative.
- Phase 13 — natural voices, not mechanical ones: every catalog `Voice` (men and
  women alike) carries an intonation range (`espeak-ng` `-P`) and a small word gap
  (`-g`), and the speeds are retuned toward a human conversational rate, so a
  reading has a living sentence melody and a natural word rhythm instead of the
  engine's flat, rushed default. `Voice.espeak_options` emits the two new flags, the
  Windows OneCore SSML carries a catalog voice's `pitch` alongside its `rate` (so
  the four voices of a category stay distinct), and `core/voices.Voice` validates
  the new settings. This is the direct answer to "make the voices read like ordinary
  people, not machines".
- Phase 14 — a real **neural** voice, not a better formant one: `core/piper_speech.py`
  adds **Piper**, a local VITS/ONNX engine that renders speech which sounds like a
  person rather than the robotic `espeak-ng` synthesis. Unlike the system engines it
  needs a model file per voice and unlike Qwen3-TTS it **speaks Arabic**
  (`ar_JO-kareem`), so it is the honest way to give the project a natural Arabic
  *and* English voice while staying on this machine. It satisfies the same
  `SpeechSynthesizer` protocol, maps the catalog's voices onto real Piper voices
  (`spec_for_voice`), and reads a catalog voice's `speed`/`pitch_range` as Piper's
  `length_scale`/`noise_scale`/`noise_w_scale`. Wired into `core/speech.py`'s
  `discover_synthesizer_name`/`load_synthesizer` (preferred once installed, with the
  system engine as the fallback for a language whose voice is not downloaded), the
  optional `[piper]` extra, the CLI `python -m cli piper …`, and the window's
  sidebar, `AGENTS.md` documents it and `tests/test_piper_speech.py` exercises the
  runner seam, the catalog mapping, the prosody, the fallback, and the real render.
- Phase 15 — Arabic that is **spoken, not spelled**: `core/arabic_speech.py` is the
  deterministic, presentation-only pass that fixes the remaining "still robotic"
  report. The neural voice (Piper, Phase 14) was already the *quality* answer, but
  the text it was handed was not speech-ready: every engine reads Arabic *words*
  well and Arabic *symbols* badly, so `25%` lost its unit, `1500 psi` switched the
  engine to an English voice mid-phrase, and `10-20` collapsed to gibberish.
  `prepare_for_speech` spells the numbers in Arabic words (with the real dual /
  plural / singular counting forms, and an oblique form a preposition governs),
  translates a known unit to its Arabic name, reads a range as a range and a
  decimal digit by digit, normalises Arabic punctuation, and drops the decorative
  tatweel. It is gated to Arabic, so English is byte-identical, it never invents a
  unit, and it adds no tashkeel (a wrong vowel mark sounds worse than none). Wired
  into **every** synthesizer, so whichever engine speaks — system, Windows,
  Piper, or Qwen3-TTS's Arabic fallback — the reading is the words a listener
  hears. `tests/test_arabic_speech.py` pins the number words, each measured
  failure, the safety rules, and the wiring.
- Phase 16 — the **local model voice**: the reading engine can now select a voice
  that lives inside a local model, not just a catalog or an OS voice. `core/voices.py`
  gains `ModelVoice` (a stable `model:<engine>:<name>` key, the engine that owns it,
  the language and gender it was reported with, and the `reference` clip a zero-shot
  engine clones) plus `model_voices`/`find_model_voice` and the shared
  `group_system_voices`/`group_model_voices` language grouping. `core/audiocpp_speech.py`
  is the first engine of that kind: **VoiceTut-TTS over audio.cpp** -- a real, local
  GGUF model (`audiocpp_cli`) that speaks natural **Egyptian Arabic** with
  Arabic/English code-switching, the one thing neither the single-voice Piper Arabic
  model nor Qwen3-TTS (no Arabic) could give. It is a full `SpeechSynthesizer`, so
  every mouth and the voice loop use it unchanged, and the text it speaks is prepared
  by `core/arabic_speech` first, so Arabic numbers and units are *spoken, not
  spelled*. The engine and the model download only through the permission + approval
  gates (`install_engine`, `download_model`, `--approve`), and **it is deliberately
  not the silent default** -- it is slower than the system engines on a CPU, so it is
  reached *explicitly* by picking one of its voices, which
  `core.speech._engine_for_voice` routes to it; a machine without it is entirely
  unaffected. Surfaced as `python -m cli audio status|models|voices|install-engine|
  download|say` and the GUI sidebar's *Voices* dropdown, which groups the installed
  system voices, the local model voices, and the synthetic catalog in one place with
  a status line and next step for each. `tests/test_audiocpp_speech.py` exercises the
  catalog, the availability states, the approval gates, the runner seam, the Arabic
  text preparation, the fallback, the voice routing, and the CLI.
- Phase 17 — the reading-voice menu **grouped by source**: the Voices dropdown and
  the Settings panel (and the CLI `python -m ai.speak list` / `python -m cli voices`)
  now list every voice under the name of the source it came from -- *Installed
  Voices (Arabic + English)*, *VoiceTut-TTS (Local Arabic)*, *Piper (Arabic +
  English)*, *Qwen3-TTS (English)*, and the *Synthetic Voices* catalog -- through the
  one `core.voices.voice_sections`. A source shows only the voices its engine really
  reports here (each `voice_list` returns `()` unless the engine and its weights are
  present) and is dropped when empty, so a listed voice is always one that can
  actually speak; the synthetic catalog is dropped whole when no engine can speak it.
  `model_voices()` now unions **all three** local engines, `_engine_for_voice` routes
  a chosen `model:piper:`/`model:qwen3-tts:` voice to its own engine (and
  `load_synthesizer` passes a Qwen speaker through), and a `ModelVoice` carries a
  readable `title` and a `.source`. A source that contributed no voice is replaced by
  one honest note and its real next step (`ChatService.voice_source_notes`,
  `qwen_tts_voice_status`), and a render that returns **no audio** is now named as
  such in the window instead of shown as a `0.0 s` success. `tests/test_voices.py`,
  `tests/test_piper_speech.py`, `tests/test_qwen_speech.py`, and `tests/test_gui.py`
  pin the grouping, the union, the source property, the engine routing, the CLI
  listing, and the honest notes.
- Phase 18 — **Doc's Studio**, the project's vision-language mouth: read *any*
  document -- including a **scanned** PDF or a photographed page -- and answer
  questions about it, with a real **Qwen3-VL** GGUF running locally in this process
  (no server, nothing sent off the machine). `core/docs_model.py` adds the model
  layer (the `qwen3-vl-2b/4b/8b` catalog, `qwen3-vl-4b` default, and the crucial
  fact that a vision model is **two** files -- the language GGUF *and* its
  `mmproj-*.gguf` projector -- so both are resolved, both are required, and the
  two can never be confused); `core/docs_studio.py` is the document engine
  (`load_document` for text/Word/PDF/image, the **scanned-PDF** path that keeps the
  pages `PdfPageStream` yielded before it raised "no extractable text" and renders
  each with `pypdfium2`, `select_pages`, and `build_vision_messages` with
  `image_url` parts); `gui/docs_studio.py` is the Tk-free service the window and
  the CLI share; `gui/docs_dialog.py` is the window, opened from the composer's
  **＋** menu, built with the same theme, composer, rich text, window controls, and
  worker-thread/queue pattern as every other window; and `cli/docs.py` is its
  headless twin (`python -m cli docs status|models|open|text|pages|ask|download`).
  The runtime is the optional `[docs]` extra, the weights are a one-time
  human-approved download under `data/docs`, and every answer is labelled as the
  local vision model's own reading rather than a cited extract.
  `tests/test_docs_studio.py`, `tests/test_docs_cli.py`, and
  `tests/test_docs_window.py` pin the catalog, the two-file rule, the projector
  exclusion, the scanned-PDF render, the page selection, the vision messages, the
  service, the CLI, the ＋-menu wiring, and the real Tk window.
- Phase 18b — Doc's Studio **reads a scanned document's contents, not just its
  image**: a scan has no text layer, so the window used to show only a page image
  and a "ask a question" note -- its content never appeared. `DocsStudioService.read`
  now transcribes the image pages with the fixed `TRANSCRIBE_QUESTION` in
  `batch_pages` batches, the window auto-reads a scanned document when it opens
  (`_maybe_auto_read` → a *contents* row under the honesty header) and offers a
  **Read the document (scanned pages)** button to re-read on demand, and the CLI
  gains `python -m cli docs read`. The reading is remembered on the document, so
  "save the extracted text" saves the transcription too. When the model is not
  ready the window states the real next step instead of a blank-looking document,
  and a scanned PDF whose pages could not be *rendered* now names that reason (the
  missing raster extra) rather than the generic "needs OCR" line that made a scan
  look like an empty file with no explanation. `tests/test_docs_studio.py`,
  `tests/test_docs_cli.py`, and `tests/test_docs_window.py` pin the transcription
  prompt, the batching, the remembered reading, the honest render failure, the CLI
  `read`, and the window's auto-read.
- Phase 18c — Doc's Studio works on **Windows**: the page image was attached to the
  vision model as ``f"file://{path}"``, which looks right but is not a URI. On
  Windows a drive path (``C:\\...``) lands in the URI's *authority* and the path
  stays empty, so ``llama-cpp-python``'s ``urlopen`` looked for the empty path and
  the model failed with ``WinError 3 (cannot find the path '')`` -- the whole
  vision path was unusable on Windows while it happened to work on Linux/macOS
  (where the path starts with ``/``). `core.docs_studio.file_uri` now builds the
  URI with ``Path.as_uri()`` (percent-encoded, drive in the path, empty authority),
  the one form ``urllib.request`` opens on every platform.
  `tests/test_docs_studio.py` opens the built URI with ``urllib.request`` and pins
  that the authority is empty and a space is percent-encoded.

- Phase 19 — the **Scene Studio**, the project's local 3D figure-and-scene
  workspace: build a scene, pose a human figure on a real skeleton, and render a
  picture, all **on this machine** with no GPU, no model download, and no
  network. `core/scene3d.py` is the dependency-free scene model (`Vec3`/`Mat4`,
  `Camera`/`Light`, `Mesh`, `Bone`/`Skeleton` with real forward kinematics and
  skin binding, `SceneObject` that deforms its mesh from the pose, `Scene`);
  `core/scene3d_figures.py` builds a parametric humanoid (four presets, a real
  bone hierarchy, six pose presets, and morphing); `core/scene3d_props.py`
  generates the props, primitives, and material palette in code;
  `core/scene3d_render.py` is a z-buffered CPU software rasteriser with
  Blinn-Phong shading and a wireframe mode that says plainly it has **no GPU, no
  ray tracing, and no commercial render engine**; `core/scene3d_io.py` reads and
  writes Wavefront OBJ/MTL; and `core/scene3d_serialize.py` writes the native
  `.faydscene` (JSON) that keeps the rig, pose, materials, lights, and camera.
  `gui/scene3d.py` is the Tk-free service the window and the CLI share (one
  scene, one renderer, one deterministic command box); `gui/scene3d_dialog.py` is
  the window, opened from the composer's **＋** menu, built with the same theme,
  composer, rich text, window controls, and worker-thread/queue pattern as every
  other window; and `cli/scene3d.py` is its headless twin with a stateful working
  scene (`python -m cli scene3d …`). Everything it makes lives under the
  git-ignored `data/scene3d/`. `tests/test_scene3d.py`,
  `tests/test_scene3d_service.py`, `tests/test_scene3d_cli.py`, and
  `tests/test_scene3d_window.py` pin the geometry, the rig and skinning, the
  renderer, the OBJ/MTL round trip, the native scene round trip, the service, the
  command parser, the CLI, the ＋-menu wiring, and the real Tk window.

- Phase 20 — reasoning by **analogy** (structure mapping): the project can now
  answer "what is this *like*?" the way Gentner's account says a mind does — by
  aligning two relational structures, never two word lists. `core/structure_mapping.py`
  is the dependency-free engine: `StructureEdge`/`Structure` are a named set of
  nodes joined by typed relations, `StructureMappingEngine.map(base, target)`
  searches for the best one-to-one correspondence that obeys **parallel
  connectivity** (a base relation may only be matched by a target relation of the
  same type between the images of its endpoints), and `StructureMapping` reports
  the matched relations, the base relations it could *not* map, the
  `candidate_inferences` it suggests for the target (each a *candidate*, never an
  assertion), relation/node coverage, `systematicity`, and `exact_share`. The
  search is ordered (most-constrained node first), bounded by a state budget so a
  large structure can never hang it, and tie-breaks toward the mapping that aligns
  more nodes. **Node names are never compared**, which is what makes the result an
  analogy rather than a lookup. Two honest additions make it work on real
  knowledge: a small **relational abstraction** (`RELATION_ROLES`/`role_for`) maps
  the ontology's several hundred near-unique relation types ("drives", "causes",
  "triggers") onto a dozen roles, so a shape is recognised across vocabularies
  (an exact type is always preferred when the target offers one, and an
  unclassified relation still matches only itself); and `systematicity` measures
  how *connected* the matched system is, because a connected system is what
  licenses an inference. `core/analogy.py` is the domain layer:
  `structure_from_labels` builds a structure from any group of concepts,
  `neighbourhood_structure` builds a concept's relational neighbourhood (the
  scale at which the project's own graph is rich enough to compare),
  `discipline_structure` builds a whole seeded discipline, and `AnalogyEngine`
  exposes `between`/`between_disciplines`/`between_concepts`/`neighbours_of`/
  `rank_for`. **Honesty by construction**: an `Analogy.is_meaningful` requires a
  shared *system* (two or more matched relations, connected) resting on at least
  one **substantive literal** match — a literal match on a non-taxonomic relation
  — so an alignment built only from role abstractions, or from a shared `is_a`
  (true of almost any pair), is reported but never called an analogy; and the
  engine found the honest limit of the seeds (`test_discipline_structures_share_almost_no_relation_types`):
  the discipline-level seeds share almost no relation types, which is why the
  usable entry point is the concept neighbourhood. `CognitiveController` gains
  `analogize`/`analogies_for`/`analogize_concepts`/`analogous_concepts`, so the
  engine is reachable from the same place as the ontology and its reasoner. The
  real result on the project's own graph: `porosity ~ permeability` maps 9
  relations (8 literal) at full systematicity, and ranks `well_log ~ core_sample`
  — two measuring instruments — as analogous. Wired as the CLI
  `python -m cli analogy disciplines|between|like|discipline` (the headless twin,
  `cli/analogy.py`, with `--json`), the menu entry *Reason by analogy: what a
  concept is structurally like*, and the **evaluation gate**: a held-out
  `ANALOGY_JUDGEMENTS` catalogue (external to the engine, with **negative
  controls**) drives `analogy_accuracy`, a new `MetricSpec` and baseline metric,
  so a regression in analogy is a gate failure. `tests/test_analogy.py` and
  `tests/test_analogy_cli.py` pin the engine (parallel connectivity, one-to-one,
  systematicity, the state budget, role abstraction, the candidate inferences, the
  no-assertion rule), the domain layer, and the CLI.

- Phase 21 — a **learned semantic space**, not a hash: the project's "semantic"
  signal used to be :func:`core.semantic_memory.hashed_embedding`, which scatters
  tokens into buckets, so two words shared a dimension only by accident -- the
  retriever could match a *word* but never a *meaning*, and a synonym that never
  appeared in the seeds matched nothing at all. `core/embeddings.py` replaces it
  with a real **distributional** space learned from the project's own text:
  `cooccurrence_counts` counts which words appear near which, `_ppmi_matrix`
  weights each pair by positive pointwise mutual information, and `_factorize`
  reduces the matrix to `DEFAULT_DIMENSIONS` (128) dense dimensions with a
  truncated SVD (numpy, already a base dependency, imported lazily). Everything is
  deterministic -- the SVD's sign ambiguity is fixed and any sampling is seeded --
  so a space can be cached and compared between runs. `core/tokenize.py` is the
  shared normaliser (`normalize_token` with light Arabic normalisation: strip
  diacritics, unify letters, drop affixes) that both the space and the retriever
  now use, so they cannot disagree about what a token is.
  **Honest by construction**: a space is *learned from text*, so it is only as good
  as the corpus, and `evaluate_held_out` measures that with a real held-out test
  rather than asserting it -- `held_out_split` holds out by **relation** (not by
  entry) so the pairs it checks were never seen as such, and a `SemanticReport`
  reports how much closer related terms sit than randomly paired ones
  (`separation`) and how often a term's true partner is among its nearest
  neighbours. On the project's own corpus: related 0.556 vs random 0.014
  (separation +0.542), the partner in the top-10 for 42.9% of held-out pairs,
  leaks 0. Wired into `core/rag.py`'s `RagRetriever`: when a space is available it
  backs the semantic signal *in the same slot and at the same weight* as the hash
  it replaces (`_load_embeddings`, `_semantic_scores`), so a query about "flow
  through rock" can reach a passage about permeability with no shared word -- and
  the calibrated fused-score scale is unchanged, which is what keeps the
  weak-evidence gate meaningful (raising the semantic weight was tried and
  reverted: a distributional prior is a better *ranking* signal but not evidence
  that the corpus answers the question, and it inflated every in-domain score past
  the web-search threshold). Trained with
  `python -m cli embeddings train` (cached under `data/ml/semantic_space.json`,
  git-ignored, via `core.paths.semantic_space_path`), inspected with `similar` /
  `compare`, and gated by the evaluation's new `semantic_separation` metric so a
  regression in the space is a gate failure. `tests/test_embeddings.py` and
  `tests/test_embeddings_cli.py` pin the counts, the PPMI, the factorisation, the
  held-out split, the honest tiny-corpus limit, the retriever wiring and its
  fallback, and the CLI.

- Phase 22 — **typed relations read from text**, not just co-occurrence: until
  now a text the project had not seeded could only teach the graph that two
  concepts were *associated* (`OntologyLearner.observe` records an untyped
  `associated_with` link between any pair of known concepts in a sentence), so it
  could never learn *how* they relate. `core/relation_extraction.py` reads the
  relation itself: `RelationExtractor` matches a sentence's own verb phrase
  against a pattern table (`_PATTERNS`) mapping to the ontology's real relation
  types -- `is_a`, `measures`, `causes`, `increases`, `reduces`, `contains`,
  `depends_on`, `affects` (`EXTRACTABLE_RELATIONS`) -- in both voices, so "the
  density log measures porosity" and "porosity is measured by the density log"
  both read as `density_log --measures--> porosity` (a passive pattern swaps its
  endpoints, so direction is never lost). **Honest by construction**: it is a
  pattern matcher, not a parser, and it refuses rather than guesses -- each side
  must resolve to *exactly one* known concept (an unknown noun, or a side naming
  two concepts, yields nothing); the object's head is the noun the verb governs,
  so a concept inside a prepositional phrase ("a blowout **in the reservoir**") is
  not read as the object; the head must *end* its phrase, so a modifier ("well" in
  "well control") is not mistaken for the head noun; a relative clause is never
  glued onto the object before it; and a *negated* clause ("does not increase",
  "never increases", "cannot measure", or a trailing ", not flow") asserts nothing
  and is refused, so the reader can never report the opposite of what a sentence
  says. Every candidate carries
  the pattern and the sentence that produced it (`ExtractedRelation.pattern` /
  `.sentence`). `read`/`extract` never touches the graph;
  `apply`/`learn_extracted_relations` is the writing path and folds each
  candidate through `OntologyLearner.observe_relation`, so a typed relation is
  bounded by the learner's ceiling and can never override a seeded one, both
  endpoints must already be concepts, and an identical typed relation is
  reinforced rather than duplicated. The learner's explicit-relation guard is now
  id-based (`_has_explicit` ignores relations the learner itself created), so an
  untyped association and a typed relation coexist and a typed link can be
  reinforced by a later sentence. `merge_extractions` de-duplicates identical
  triples across chunks, and `extract_from_store` streams a whole
  `core.knowledge_base.KnowledgeBaseStore` one chunk at a time (duck-typed through
  the `ChunkSource` protocol, so the extractor does not import the store), so an
  ingested corpus can teach the graph -- `python -m cli relations read|learn|scan|
  types` (`cli/relations.py`, the headless twin; the menu carries *Extract typed
  relations from text (preview, then learn)*). `learn`/`scan` preview by default
  and write only with `--write`. `tests/test_relation_extraction.py` and
  `tests/test_relations_cli.py` pin the active and passive directions, the
  refusals, the head-noun rule, the ceiling, the seeded-relation guard, the
  reinforcement, the merge, the store stream, and the CLI. `EXTRACTION_JUDGEMENTS`
  is the held-out gate -- hand-written sentences *with negative controls*, scored
  by `extraction_accuracy` (a judgement counts only when the extractor reads
  exactly the expected triples, so an over-firing reader fails as surely as one
  that misses), wired into `evaluation` as the `relation_extraction_accuracy`
  metric.

- Phase 23 — **Arabic reaches the concept graph**: the project is Arabic-facing,
  but the symbolic core was English-only, so the Arabic user could not reach it.
  Measured before the fix: of the corpus's own ~30 Arabic glossary terms only ~3
  resolved to a concept, only 18 of 495 concepts carried any Arabic term, and
  `find_concepts("المسامية")` / `("النفاذية")` returned nothing -- an Arabic
  sentence taught and matched the graph no better than a random string.
  `core/tokenize.py` gains the shared normalised *phrase* space
  (`normalize_phrase`, `token_spans`), so a term and a text are compared as whole
  normalised tokens rather than by a `casefold` substring test;
  `core.concept.Concept` stores its terms normalised (and `terms_for(language)`,
  `is_arabic`) so a concept is looked up the same way in either script; and
  `core.ontology.Ontology` gains an Arabic-aware, whole-word span matcher
  (`match_concepts`, `concepts_in`) plus `add_concept_alias`, which adds an alias
  to a frozen concept *in place* keeping its id, so the graph's relations survive
  the new name. Matching a term only at a token's start with a bounded inflection
  (`_token_matches`) keeps "flow" reaching "flowed" without "porosity" matching
  inside "microporosity". `core/petroleum_corpus.seed_petroleum_corpus` now folds
  the corpus's own Arabic glossary onto the concepts it names
  (`_seed_glossary_aliases`), so the ontology and the corpus agree on what a
  concept is called in either language -- the corpus already recorded
  `("porosity", "مسامية")`, and the concept simply never carried it. The learner
  and the ontology-guided retriever read concepts through the same matcher, so all
  three agree. Result: the same Arabic sentence resolves
  `المسامية تزيد النفاذية` to `porosity` and `permeability` end to end. Honest by
  construction: the matcher is deterministic and refuses rather than guesses, and
  it still does not invent an alias for a term the ontology has no node for
  ("oil in place", "gamma ray"). `tests/test_arabic_concepts.py` pins the shared
  normalised space, the article handling, the corpus-glossary seeding, the
  whole-word and inflection rules, the nested-concept rule, and the learner and
  retriever wiring.

- Phase 24 — **polarity: knowing which relations are opposites**: the graph could
  hold a *typed* relation, but nothing knew that ``increases`` is the opposite of
  ``reduces`` (or ``causes`` of ``prevents``), so two sentences saying the
  opposite thing about the same pair -- "waterflood increases recovery" and
  "waterflood reduces recovery" -- looked like two unrelated links rather than a
  contradiction. `core/relation_polarity.py` is the one declarative place that
  fact lives: a relation type carries a **polarity** from a fixed vocabulary
  (``POLARITY``), two types oppose only when they are the two poles of the same
  *axis* (`OPPOSING_PAIRS`), and a *symmetric* relation (``associated_with``,
  `SYMMETRIC_RELATIONS`) is on no axis, so it can never be reported as opposing
  anything. `polarity_of`/`is_symmetric`/`opposes`, `conflicts` (severity = the
  weaker confidence, the two types reported in canonical order), and
  `find_polarity_conflicts` (every opposing pair among a relation collection that
  shares the *same ordered pair* and direction -- so ``A increases B`` does not
  conflict with ``B reduces A``, a different claim). **Honest by construction**:
  an unknown relation type has no polarity and opposes nothing, and only an
  explicitly listed antonymous pair is ever called contradictory. Two existing
  checkers were upgraded to use it. `core/consistency_checker.py` now also reads
  the relation a belief asserts out of its ``metadata`` (``relation_type`` +
  ``source_id``/``target_id``) and calls two beliefs contradictory when they put
  opposite poles on the same ordered pair -- *direction-aware*, so it never calls
  "A increases B" and "B reduces A" contradictory (a free-text verb check could
  not promise that, which is why the earlier statement-level attempt was
  dropped). `core/contradiction_detector.py` treats two evidence records that
  assert *antonymous* relations on the same pair as a contradiction even when both
  set ``supports=True``, and refuses to call two records contradictory merely
  because one flag is set and the other is not when both carry the *same* relation.
  `CognitiveController.find_relation_conflicts` scans the graph (never edits it)
  for opposing typed relations, which is exactly what a text taught through
  `learn_extracted_relations` can produce. Gated by the evaluation's new
  `polarity_accuracy` metric over a hand-written held-out `POLARITY_JUDGEMENTS`
  catalogue **with negative controls** (an unrelated type, a symmetric relation, a
  type against itself), so an over-firing polarity model fails the gate as surely
  as one that misses. `tests/test_relation_polarity.py` pins the vocabulary, the
  antonyms, the symmetric refusal, the direction rule, the determinism, the
  canonical order, both upgraded checkers, the graph scan, and the gate.

- Phase 25 — **the graph's own relations reach the answer**: retrieval returned
  whole *sentences* a document states, but the project also holds a **concept
  graph** whose edges are typed relations (``porosity increases permeability``,
  ``density_log measures porosity``) -- including links a text taught through
  `learn_extracted_relations`. Those links are facts in their own right, and an
  answer that never states them was quietly less complete than the knowledge on
  file. `core/relation_facts.py` reads the relations a question's **own concepts**
  participate in and renders them as a short, deterministic block.
  `RelationFact` (source, relation_type, target, confidence) reads its type's
  underscores as spaces for `statement` and keeps the graph's exact type in
  `describe` (``porosity —increases→ permeability``); `relation_facts` includes a
  relation when **either** endpoint is a concept the question names (so an
  *incoming* link -- "the density log measures porosity", asked about porosity --
  is reachable, not only outgoing ones), de-duplicates by triple, and orders by
  confidence then labels, so the same question always yields the same block;
  `describe_relation_facts` names the block as the *knowledge graph* on purpose,
  so a reader can tell a stored typed link from a cited sentence (different
  provenance). **Read-only and honest by construction**: it never edits the graph,
  never invents a relation, states only the edges the ontology actually carries in
  their declared direction, and returns ``()` (no block) for an unknown concept --
  a question the graph knows nothing about produces nothing rather than a guess.
  `GroundedAssistant.ask` attaches the facts to `AssistantAnswer.relation_facts`
  as **data**, like the citations, so the prose stays the model's or the sources'
  words; `JarvisTurn.relation_facts` carries them (read defensively through
  `getattr`, so every mouth -- even one with no ontology -- is unaffected), and
  `ai/chat.format_turn` renders the block after the answer and before the
  `Sources:` block, with `turn_payload` exposing them for `--json`. The GUI shows
  them for free, because it renders through the same `format_turn`.
  `tests/test_relation_facts.py` pins the lookup, the incoming-relation rule, the
  unknown-question honesty, the determinism, the limit, the de-duplication, the
  rendering, and the assistant wiring.

- Phase 26 — **how two concepts are connected, step by step**: the graph could
  state a direct relation and the reasoner could derive relations that follow from
  the stored ones, but neither answered the question a reader actually asks --
  *"how is porosity connected to recovery factor?"* when there is no single edge
  between them. `core/relation_paths.py` finds the **shortest chain** of typed
  relations between two concepts and returns it as an ordered list of steps.
  `RelationStep` (source, relation_type, target, confidence, `forward`) records
  whether the edge was walked **forward** (source → target, the stored direction)
  or **backward** (target → source), so `describe()` shows the arrow
  (``porosity —increases→ permeability`` vs ``permeability ←increases— porosity``)
  and a reader sees exactly which claims the explanation rests on; `ConnectionPath`
  carries the endpoints and steps, with `hops`, a `confidence` that is the
  **product** of its steps' (a longer or weaker chain is honestly less certain than
  a direct edge, like a derivation), and a `describe()` that prints the header plus
  each numbered step. `find_connection` is breadth-first over the explicit
  relations, walking every edge in **both** directions (so a chain that reaches an
  endpoint through an incoming relation is found too) with a deterministic
  tie-break (by label then relation type), so the same question always yields the
  same path. **Read-only and honest by construction**: it never asserts the path
  back into the graph, never invents an edge, and returns ``None`` for an unknown
  concept, a non-positive hop bound, or two concepts no chain within the bound
  connects -- so "I cannot connect these" is a real answer, and
  `describe_connection` states that absence plainly. `DEFAULT_MAX_HOPS` (3) keeps
  the search bounded and the chain short enough to read. Wired as
  `python -m cli relations connect <a> <b> [--max-hops N] [--json]` (the headless
  twin, with a machine-readable payload carrying `connected`/`hops`/`confidence`
  and each step's `forward` flag). The gate is `connection_accuracy`: a held-out
  catalogue of concept pairs (`CONNECTION_JUDGEMENTS`, hand-written with
  **negative controls** -- an unknown endpoint that must yield no chain) that the
  search must connect exactly when a chain exists and with the expected number of
  hops, so a fabricated link fails the gate as surely as a missed one; added to
  `evaluation/benchmark.py`, `evaluation/runner.py`, and `evaluation/baseline.json`.
  `tests/test_relation_paths.py` pins the direct chain, the backwards walk, the
  product confidence, the zero-hop same-concept case, the honest absence, the hop
  bound, the determinism, and the step rendering; `tests/test_relations_cli.py`
  pins the `connect` command, its JSON payload, its honest absence, and its hop
  bound.

- Phase 27 — **contradictions are reported, not hidden**: `core.relation_polarity`
  could tell that ``increases`` opposes ``reduces``, and `find_polarity_conflicts`
  could find the pairs asserting both -- but a conflict the answer path never
  mentions is one the user never learns about: the graph holds "A increases B" and
  "A reduces B" and the reader is shown whichever retrieval happened to reach.
  `core/relation_consistency.py` turns that scan into an honest, bounded **report**.
  `RelationConflict` (source, relation_type, target, opposing_type, severity) names
  both sides with readable labels and `describe()` renders the disagreement
  (``a —increases→ b vs a —reduces→ b (severity 0.40)``); `relation_conflicts` with
  no question returns the graph's whole history of disagreement, and with a
  question keeps only conflicts whose **either** endpoint is a concept the question
  names (so the answer carries the caveat that matters to it), ordered by severity
  then labels, so the same question always yields the same report.
  `describe_relation_conflicts` uses deliberately neutral wording -- "the graph
  holds both ... and ..." -- because the program does not know which side is right
  and must not imply it. **Read-only and honest by construction**: it never resolves
  a conflict, never picks a winner, never edits the graph, and returns ``()` for a
  consistent graph, no ontology, or a question naming no conflicting concept --
  choosing between two evidenced claims is the user's call, not the program's.
  `GroundedAssistant.ask` attaches the conflicts to
  `AssistantAnswer.relation_conflicts` as **data**, like the facts and citations;
  `JarvisTurn.relation_conflicts` carries them (read defensively through `getattr`,
  so every mouth is unaffected), and `ai/chat.format_turn` renders the warning
  *before* the knowledge-graph block (a caveat belongs above the claims it
  qualifies), with `turn_payload` exposing them for `--json`. The GUI shows them
  for free through the same `format_turn`. Wired as
  `python -m cli relations conflicts [--concept C] [--json]` (read-only). The gate
  is `consistency_accuracy`: a held-out catalogue of hand-written graphs
  (`CONSISTENCY_SCENARIOS`) whose contradiction count must be **exact**, including a
  scenario where an opposite-*direction* claim ("B reduces A") is correctly **not**
  a contradiction -- so a hidden contradiction and a false positive both fail; added
  to `evaluation/benchmark.py`, `evaluation/runner.py`, and `evaluation/baseline.json`.
  `tests/test_relation_consistency.py` pins the detection, the severity, the clean
  graph, the question scoping, the no-ontology case, the neutral rendering, the
  assistant wiring, and the gate; `tests/test_relations_cli.py` pins the `conflicts`
  command, its JSON, and its concept scoping.
- Phase 28 — the reasoning layer **reaches the answer path**: `core/relation_inference.py`
  surfaces the relations the graph *entails* -- the conclusions `core.ontology_reasoner`
  derives from stored edges through its transparent rules -- as evidence for an
  answer, for the first time. Until now `OntologyReasoner`/`ReasoningEngine` were
  reachable only from the controller, so a derivation the program drew never reached
  a reader. `InferredRelation` carries the derived triple, its propagated confidence,
  and the **steps** it rests on (`explain()` renders "well —penetrates→ formation
  (from well —penetrates→ reservoir reservoir —contained_in→ formation)"), and
  `describe_inferred_relations` renders a block **labelled as an inference** -- a
  reader must always be able to tell a claim someone wrote from one the program
  derived. `inferred_relations(ontology, question, ...)` scopes the derivations to
  the concepts the question names (either endpoint), is bounded
  (`DEFAULT_INFERENCE_LIMIT`), drops weak derivations (`DEFAULT_MIN_CONFIDENCE`),
  de-duplicates by triple, and orders deterministically; it never re-labels a
  **stored** relation as an inference, never asserts a derivation back into the
  graph, and returns `()` when nothing is entailed -- so nothing is invented.
  `AssistantAnswer.inferred` and `JarvisTurn.inferred` carry the lines as *data*,
  `ai/chat.format_turn` renders them under the inference label, and `turn_payload`
  exposes them for `--json`. Wired as `python -m cli relations infer "<question>"
  [--limit N] [--min-confidence F] [--json]` (read-only). The gate is
  `inference_accuracy`: a held-out catalogue of hand-written entailments
  (`INFERENCE_JUDGEMENTS`) the reasoner must derive exactly when one exists --
  including negative controls (an unrelated pair, an unknown endpoint) -- so a rule
  that over-reaches fails as surely as one that misses; added to
  `evaluation/benchmark.py`, `evaluation/runner.py`, and `evaluation/baseline.json`.
  `tests/test_relation_inference.py` pins the derivations, the steps, the
  determinism, the limits, the confidence floor, the never-relabel rule, the
  labelled block, the negative controls, the gate, and the assistant/Jarvis/format
  wiring; `tests/test_relations_cli.py` pins the `infer` command, its JSON, its
  empty case, and its argument validation.

- Phase 28 — **Hakeem, the scholar of religion and history**: a second companion
  character beside Natasha, on the **same local model** she uses. The task was to
  build a character on the model the companion already speaks through, so
  `hakeem/model.py` is a thin layer over `companion.model` (one catalog, one
  `data/natasha` model folder, the same governed download and client seam) and
  only the *character* and the database differ — a machine that already has the
  companion's weights downloads nothing for Hakeem. `hakeem/personality.py` is his
  stable character: fixed traits and the non-negotiable `BOUNDARIES` that make a
  religious character responsible — respect every faith and never mock or
  denigrate any of them; never proselytise and never declare which religion is
  true; never issue a ruling (fatwa) or a verdict, pointing a personal question to
  qualified scholars; keep scripture, established history, scholarly consensus, and
  personal interpretation clearly apart; state when scholars disagree; never invent
  a verse, a hadith, or a historical fact; refuse to defame a religion or promote
  hatred; and stay honest that he is a virtual character. `COMPARATIVE_METHOD` and
  `HISTORY_METHOD` are his comparison and history methods (each tradition described
  from inside on a shared axis, a comparison **never a ranking**; claims anchored in
  time and place with the kind of source named and contested questions said to be
  contested). Five 0..1 style dials (`formality`, `depth`, `citation`,
  `comparison`, `history`) and a `match`/`ar`/`en` reply-language rule are tunable
  through the CLI, each `*_instruction()` **restating the honesty floor in its own
  text** so a high dial can never be read as licence to drop it; only the settings
  persist, so the character's wording ships with an update. `hakeem/hakeem_agent.py`
  is the one engine (its own SQLite database — a separate file from the
  companion's — durable memory reusing the companion's own manager, isolated
  sessions, a gradual tone, a bounded prompt, and a `caller`/`client` seam so the
  whole scholar runs with **no server and no weights**); a failed reply is never
  stored, a reasoning model's hidden trace never reaches the transcript, and a
  remembered item is never treated as a source. Wired **directly below Natasha** in
  the menu (`cli.menu.run_hakeem`), CLI (`python -m cli hakeem …`), and as a
  **mouth inside the project's own chat** (`ai/hakeem_mouth.py`, the headless twin
  `python -m cli hakeem-chat`) in the window's header model selector, with the
  sidebar action hidden. He is deliberately **not** the cognitive core: nothing in
  `hakeem/` writes to entities, relations, evidence, beliefs, or the knowledge
  graph. `tests/test_hakeem.py` pins the character and its safety floor, the
  dials, the language rule, the memory/session/tone behaviour, the exchange's
  honesty guarantees, the shared-model choice, the CLI, the mouth, and the
  menu/GUI wiring.

- Phase 29 — **Natasha and Hakeem reply in seconds, not minutes**: both characters
  defaulted to the reasoning model `qwen3:4b` with its hidden reasoning trace
  **on** and a 4096-token cap, so a turn spent thousands of tokens thinking before
  it said anything — minutes on a CPU, when the user wanted seconds. The defaults
  are now **speed-first** and the trace is off: `think=False` (so the model answers
  directly), `max_tokens=512`, `context=4096`, `timeout=180`, `history_window=8`,
  `memory_limit=4`, `summary_limit=2`, `context_chars=12000`, and `keep_alive="30m"`.
  Disabling the
  trace is done at **two** levels so neither a runtime nor a template can defeat it:
  `companion.model.thinking_kwargs` sends
  `chat_template_kwargs: {"enable_thinking": false}` to the in-process
  `llama-cpp-python` runtime (the `LlamaCppNatashaRunner` applies it to every
  call), and `companion.model.apply_no_think` appends Qwen3's own `/no_think`
  marker to the system turn — only for a `qwen3` model, only when the trace is off,
  idempotently, and never to a message list with no system turn — for a build whose
  template ignores the flag. `core.coder_model.OllamaCoderClient` gains two optional
  settings, `num_ctx` and `keep_alive` (Ollama's `num_ctx` and `keep_alive`), so the
  window stays fixed and the model is not unloaded between turns; both are unset by
  default, so every existing caller is untouched. The in-process runner also raises
  `n_batch` (bounded by `n_ctx`), because llama.cpp's small default batch is the
  other half of a slow first reply. `qwen3-1.7b` joins the catalog as the fastest
  option on a slow CPU. Both CLIs expose the knobs — `--think`/`--no-think`
  (tri-state: leave the default, restore the deep reasoning, or force the fast
  path), `--max-tokens`, `--context`, `--threads`, and `--context-turns` — and
  `status` prints a `speed` line naming which is in force. `--think` (or
  `FAYD_NATASHA_THINK`/`FAYD_HAKEEM_THINK`) restores the deeper reasoning when a
  user wants it and can wait. `tests/test_companion.py` and `tests/test_hakeem.py`
  pin the fast defaults, the two trace switches, the new Ollama settings, and the
  CLI flags; `strip_thinking` still guards the transcript, so a trace that slips
  through is never shown.

- Phase 30 — **a dedicated chat window per answering model**: the composer's
  model row used to *switch* the single shared conversation's mouth, so talking to
  Ultron and then to Hakeem meant one transcript and one reading voice for both.
  Now each of the seven chips opens that model's **own** window, with three
  controls down its left side that belong to the mouth and survive a restart:
  its **logo** (clickable to upload any picture, with a reset to the default
  badge, beside an editable **name**), the **reading voice** exclusive to it
  (changing it here never changes the main window's), and the **language model**
  it answers with, chosen from the project's full catalog. `core/language_models.py`
  is the one **unified catalog**: every GGUF model the project names (Ultron's,
  Vision's, Banner's, the companion's, the trained checkpoint) and every cloud
  provider model in one list, **de-duplicated by identity** so the same weights
  reachable from two engines are one entry carrying all its `engines`, each with
  its real availability and the real next step. `core/mouth_profile.py` persists
  one profile per mouth (name, logo, voice, model) through `JsonStore`, copying a
  chosen logo into the git-ignored `data/characters` and falling back to the
  default when a saved voice or model has left the catalog. `ai/mouths.py` is the
  one `(mouth, model key) -> assistant` factory: it resolves the choice to its
  exact GGUF file (or cloud provider + model) and hands it to the mouth's own
  loader, so a model picked from another engine's catalog reaches the right file.
  `gui/model_chat.py` is the Tk-free per-mouth service (its own session, its own
  `TextReader`, so the voice is exclusive); `gui/model_chat_dialog.py` is the
  window, built with the same theme, composer (microphone included), rich text,
  window controls, and worker-thread/queue pattern as every other window;
  `ChatService.set_mouth_model`/`mouth_model` keep the choice in force for the
  mouth that is answering and rebuild it at once when it is the live one; and
  `cli/model_chat.py` is the headless twin (`python -m cli model-chat …`) driving
  the *same* service. `tests/test_model_chat.py` pins the catalog (the shared
  GGUF list, the honest availability, the dedup), the profile store (per-mouth
  isolation, the logo copy and reset, the stale-choice fallback), the factory,
  the service, the CLI, and the real Tk window; `tests/test_gui_design.py` pins
  that each chip opens that model's window and leaves the main conversation
  untouched.

Every phase keeps the four hard gates green and never lets the model widen its own
authority: the sandbox, the groundedness gate, the explicit-feedback rule, and the
regression monitor each can refuse a change.

- Phase 31 — **the per-model windows fit the screen, and maximize leaves the
  taskbar alone**: three refinements to the windows Phase 30 added. The window
  opens at **90% of its former height** (`DEFAULT_HEIGHT`, capped to the work area
  so a small display still shows the composer). Both the left sidebar and the
  conversation carry a **scrollbar that is always visible** -- not one that
  appears only once its content overflows -- so neither column is ever a dead end
  on a short window. And **maximize** now grows the window to the *usable
  desktop* instead of a full-screen takeover: `gui.window_controls` keeps the
  native `state("zoomed")` where the window manager has it (which already fits the
  work area), and where it does not (many X11 desktops) it fills the work area by
  hand through the new `work_area()` lookup -- Windows' `SystemParametersInfo`
  work area (`rcWork`, which excludes the taskbar wherever the user put it), X11's
  `_NET_WORKAREA` via `xprop`, and macOS' `NSScreen.visibleFrame` -- falling back
  to a real full-screen window only where the desktop reports none, and remembering
  the pre-fill geometry so restore returns the window exactly. A full-screen window
  used to cover the taskbar, which is not what "maximize" means.
  `tests/test_window_controls.py` pins the work-area fill, the exact restore, the
  full-screen fallback when no work area is known, the native-zoom preference, and
  the `xprop` parsing; `tests/test_model_chat.py` pins the 90% height, the
  always-visible scrollbars, the work-area clamp, and the window's own maximize.

- Phase 32 — **a studio window per speech engine, with its own voice and mood**:
  the composer's model row gained a *trailing* group of four **speech-engine**
  buttons, ordered right to left from the far right edge and separated from the
  model buttons by a gap -- `Installed Voices (Arabic + English)`, `VoiceTut-TTS
  (Local Arabic)`, `Piper (Arabic + English)`, and `Qwen3-TTS (English)`. Pressing
  one opens that engine's own studio window, where a document is loaded and read
  aloud; the left sidebar holds, top to bottom, (1) the engine's **logo and name**
  (changeable -- upload any picture, or rename the studio), (2) the **reading
  voice**, drawn from **only** the voices that engine really reports here, exclusive
  to that studio and remembered until the user changes it, and (3) the **mood** --
  an emotional state from a catalog of twenty (happy, sad, excited, rapturous,
  in love, angry, calm, and the rest), applied to the voice the moment it is
  chosen and also exclusive. The composer carries a **microphone**. `core/voice_moods.py`
  is the one declarative place a mood lives: a `Mood` names the state and carries
  the `Prosody` that produces it (a pitch offset, a speed multiplier, an
  intonation-range offset, a word-gap offset, an energy multiplier, and a style
  instruction), each field a *delta* or *multiplier* from the chosen voice's own
  settings so a mood layers onto the timbre the user picked, bounded so no setting
  can make speech unreadable. `apply_mood` folds the numeric part onto a catalog
  `Voice`; the engines read the rest through `load_synthesizer`'s `prosody`
  option, which `core/speech.py` now threads into **every** synthesizer -- the
  `espeak-ng` flags (pitch, speed, intonation range, word gap), SAPI's rate,
  the OneCore SSML rate/pitch/emphasis, Piper's `length_scale`/`noise_scale`/
  `noise_w_scale`, and Qwen3-TTS's style instruction -- while the VoiceTut cloning
  engine honestly keeps its reference voice. `ENGINE_MOOD_SUPPORT`/`support_note`
  is the one place that per-engine truth is written down, and the window shows it.
  `core/voice_studio_profile.py` persists each engine's name, logo, voice, and
  mood separately (through `JsonStore`, logos in the git-ignored `data/characters`
  as `voice-<source>.<ext>`, so they never collide with the per-mouth logos), with
  honest fallbacks for a store that cannot be read or a choice that left the
  catalog. `gui/voice_studio.py` is the Tk-free service the window and the CLI
  share (one voice, one mood, one document reader); `gui/voice_studio_dialog.py`
  is the window, built with the same theme, composer, rich text, window controls,
  and worker-thread/queue pattern as every other window; `gui/app.py` opens one per
  engine (`_open_voice_studio`, reusing an already-open one); and
  `cli/voice_studio.py` is the headless twin (`python -m cli voice-studio …`)
  driving the *same* service. `tests/test_voice_moods.py` pins the catalog and the
  bounded prosody; `tests/test_voice_studio.py` pins the profile store (per-engine
  isolation, the logo copy and reset, the stale-choice fallback) and the service;
  `tests/test_voice_studio_cli.py` pins the CLI; `tests/test_voice_studio_window.py`
  pins the real Tk window and the composer row opening it; and `tests/test_voices.py`
  pins the prosody reaching each engine's own flags.

- Phase 31 — **each engine studio reads with its *own* voice, and can fetch what it
  is missing**: the four speech-engine studios (system, audio.cpp, Piper,
  Qwen3-TTS) listed their own voices, but a studio whose voice was never chosen
  still played the *default* voice instead. The cause was that
  `gui/voice_studio.py`'s service never applied an engine voice at all on a fresh
  profile: the reader kept its own default -- the synthetic catalog's `woman-1` --
  and a *local* engine cannot speak a catalog voice (audio.cpp has no reference
  clip to clone, Piper has no matching ONNX file, Qwen3-TTS has no speaker by that
  name), so the engine failed and the reading fell back to the system engine (the
  Windows default voice) instead of the voice the studio offered.
  `VoiceStudioService` now calls `_apply_initial_voice()` in `__init__`: the saved
  voice is honoured while the engine still offers it, otherwise the studio starts
  on the first **real** voice the engine reports (`voice_keys()`, minus the
  synthetic catalog), and the system studio -- the one engine the catalog's
  settings are valid for -- keeps the project's documented
  `core.voices.default_voice()` rather than an arbitrary catalog entry. A second
  gap was closed at the same time: a studio could *offer* a voice it had no way to
  fetch. The service now reports `setup_status()` (a `SetupStatus` separating the
  **runtime** -- a Python extra or audio.cpp's native binary -- from the
  **voices**, each with its own real next step and its own `can_install_runtime`/
  `can_download_voices` flag), `fetchable_voices()` (the voices the engine can
  actually fetch, empty until its runtime is present), and the two governed
  actions `install_runtime()` / `download_voices()` (a `SetupReport`, never
  raised; `approved=True` is the user's click, like every other external action).
  The window gains an **Engine setup** sidebar block -- an honest status line and
  the one button that helps (*Install the engine*, a fetch-voice picker, or
  *Download the voices*) -- run on the worker thread, rebuilding the voice list
  from the engine's real state when it succeeds. The CLI gains
  `python -m cli voice-studio setup|install|download` (the headless twins, `--json`
  for `setup`). Two requirements files and two extras were also incomplete: the
  Piper and Qwen3-TTS (and Qwen3-ASR) downloads use `huggingface_hub`, which was
  not listed in `requirements-piper.txt`, `requirements-qwen-tts.txt`,
  `requirements-qwen-asr.txt`, or the `[piper]`/`[qwen-tts]`/`[qwen-asr]` extras,
  so the download failed with "needs `huggingface_hub`" *after* the user had
  installed the engine -- it is now pinned in all of them. Verified end to end on
  this machine: with the Piper runtime and the Arabic voice installed, the Piper
  studio starts on `model:piper:ar_JO-kareem-medium` and renders real Arabic
  speech, instead of falling back to the default voice.
  `tests/test_voice_studio.py` pins the engine-own default voice (per engine, the
  saved-voice rule, the system studio's catalog exception, and the two setup
  halves); `tests/test_voice_studio_cli.py` pins `setup`/`install`/`download`;
  `tests/test_voice_studio_window.py` pins the setup block in the real window.
  The same investigation closed a latent bug in `core/speech.py` that this feature
  made reachable: `discover_synthesizer_name` picks Piper as soon as *any* voice is
  cached, but `load_synthesizer("piper")` only supplied a fallback when a system
  engine existed -- so a machine with only the Arabic voice and no espeak-ng/`say`
  returned an engine that *raised* on an English reading, breaking `auto`'s promise
  of "always a usable engine" (`test_auto_loader_always_returns_a_usable_engine`).
  The factory now supplies a `SilentSynthesizer` fallback in that case, so the
  reading degrades honestly rather than failing -- while the low-level
  `PiperSynthesizer` still raises when built with no fallback.
  `tests/test_piper_speech.py` pins the degradation.

- Phase 33 — **the per-model windows switch models and add their own**: the seven
  dedicated windows (one per answering mouth) listed the project's language models
  but a *switch* there did not really take effect, and a GGUF file the catalog did
  not name could not be offered at all. Two things changed. First, a window's
  mouth is now the one answering *its own* session: `gui.model_chat.ModelChatService`
  calls `ChatService.activate_mouth(mouth)` on open (a new method beside
  `mouth_model`/`set_mouth_model`, which returns the now-active mouth key and
  raises the mouth's own availability error), so choosing a model there and asking
  a question really uses it -- while the main window's `ChatService` (a separate
  instance, its own session) is untouched. The window reports honestly whether it
  is active (`ModelChatService.active()`/`activation_status()`, drawn into the
  model note); a mouth that cannot load still opens and names the real next step.
  The switch is made to *stick*: `ai.chat.ModelSwitcher` caches each mouth's
  assistant after its first load, so `set_mouth_model` now drops the named mouth
  through a new `ModelSwitcher.invalidate_mouth(mouth)` before reloading it --
  otherwise the reload would hand back the model the mouth was first built with
  (only that mouth is dropped, so the others keep their loaded model). The choice
  is saved *before* the load, so pointing a window at a model whose runtime or
  weights are missing remembers it and reports the next step rather than losing it.
  Second, a window can **add a language model** the user points it at:
  `core/custom_models.py` is a small `JsonStore` registry (`NAMESPACE`,
  `CUSTOM_PREFIX` `custom:`, `CustomModel`, `add_model`/`remove_model`/
  `list_models`/`find_model`/`is_custom_key`) that remembers a `.gguf` file
  **in place** (a multi-gigabyte GGUF is referenced, not copied; a file the user
  later moves is reported unavailable, its real path checked on every read; a bad
  choice raises `ValueError` rather than being stored; a corrupt store reads as
  empty). `core.language_models` merges those entries into the catalog as ordinary
  `LanguageModel`s (a new `custom` flag and a `download_command` field; offered to
  every GGUF mouth, since they share the one runtime), and gains
  `can_download(model)`/`download_model(mouth, key, approved=, progress=)` -- the
  one dispatcher that fetches a chosen model through the **owning engine's** own
  permission + human-approval gate (Ultron's, Vision's, Banner's, or the
  companion's shared folder), so no second download path is written.
  `ModelChatService` gains `add_model` (adds and activates it at once),
  `remove_model`/`can_remove` (forget a `custom:` entry, falling back to the
  mouth's default when it was active), and `can_download`/`download_command`/
  `download_model`; the window's Language-model block gains three controls
  (*Add a language model…* -- a GGUF file picker; *Download the chosen model*,
  enabled only when the chosen model has something to fetch and run on the worker
  thread with approval; *Forget this model*, enabled only for a `custom:` entry),
  and the CLI gains `add-model`/`remove-model`/`download-model` (the headless
  twins, `download-model` needing `--approve`). The retrieval mouth (Jarvis, from
  the sources) has no language model at all, so its window offers **no** *Add a
  language model…* or *Forget this model* action (`ModelChatService.can_add_model()`
  is false there, and the dialog drops the two buttons rather than showing them
  dead). `tests/test_model_chat.py` pins the
  registry (add/refuse/persist/remove/the file that left/a corrupt store), the
  catalog merge and the download dispatch, the service's activation and
  add/remove/download, the switch that drops the cached mouth, the retrieval
  mouth's honest refusal, the CLI commands, and the window's controls;
  `tests/test_chat.py` pins `ModelSwitcher.invalidate_mouth` (a switch rebuilds
  the mouth instead of reusing the cached assistant, and only the named mouth is
  dropped).

- Phase 34 — **a VoiceTut read no longer reloads the model per paragraph**: after a
  user replaced the `voicetut-tts-q8_0` file (the *correct* 1.35 GB Q8_0 package;
  the earlier 606 MB copy was partial) the voices appeared to hang. The file was
  right and the catalog's size was right — the hang was the reader's shape. A
  document is split into paragraph chunks and every chunk was a **separate**
  `audiocpp_cli` process, so each one reloaded the whole GGUF (1.35 GB) before it
  could speak; a long document spent nearly all its time loading weights, which
  reads as frozen. audio.cpp's own `--request-sequence` exists to "run multiple
  JSON requests through one long-lived offline model session", so the reader now
  batches: `AudioCppCliRunner.synthesize_many` sends a whole batch as one sequence
  and matches each output back by its request id (falling back to the files the
  engine produced if it ignores the ids), `AudioCppSynthesizer.synthesize_many`
  prepares each chunk for speech and returns the clips in order (falling back to
  the system engine for the batch, as before), and `core.reading.TextReader.render_many`
  is the one batched entry point beside `render` (an engine without
  `synthesize_many` is simply called once per text, so nothing else changes).
  `gui.voice_studio`'s `read_aloud`/`start_reading` group the chunks with
  `_batch_chunks` (bounded by `BATCH_MAX_CHARS`/`BATCH_MAX_CHUNKS`) and render each
  batch through `_render_group`, so a document is read with a handful of model
  loads instead of one per paragraph — and a stop still takes effect at the next
  clip, because a batch is only a *render* unit, never a playback unit.
  Three further hardenings came out of the same investigation. **A file that is not
  really a GGUF is refused instead of handed to the engine**: `is_valid_gguf` reads
  the `GGUF` magic, `find_model_file` skips a file that fails it (so a truncated
  download or a renamed file can no longer leave a read hanging with no error),
  `download_model` re-fetches a partial file already on disk and refuses a
  still-broken download with the real next step, and `audiocpp_status` names the
  unusable file (`The model file X is not a usable GGUF ...`) rather than reporting
  a bare "no model". **The engine gets more of the machine**: the runner passes
  `--threads` (`_default_threads`, bounded by `DEFAULT_MAX_THREADS`), because
  audio.cpp's own default uses few cores and that is the other half of a slow first
  reply, and a timeout now says so plainly instead of surfacing as a bare hang. The
  batching lives in **one** place -- `core.reading.batch_chunks` (bounded by
  `BATCH_MAX_CHARS`/`BATCH_MAX_CHUNKS`) -- so both the engine studio
  (`gui/voice_studio.py`) and the **seven per-mouth chat windows**
  (`gui/model_chat.py`'s `start_reading`) batch a reading identically, and a batch
  the engine names by id is matched back by id while one that ignores the ids is
  ordered by its trailing **number** (`_clip_sort_key`, so `speech-2` precedes
  `speech-10`) rather than lexically.
  `tests/test_audiocpp_speech.py` pins the GGUF magic check, the skip of a corrupt
  file, the corrupt-model status, the re-fetch of a truncated download, the batch
  that is one runner call, the batch fallback, the bounded thread count, the
  `--threads` argument, the id fallback's number order, and the honest timeout;
  `tests/test_voice_studio.py` pins `_batch_chunks` and the batched read;
  `tests/test_model_chat.py` pins the per-mouth window's batched read and its
  fallback; `tests/test_voices.py` pins `TextReader.render_many` and its fallback.

- Phase 35 — **the whole project answers in milliseconds, and a VoiceTut read in
  seconds**: a performance pass over the hot paths, measured rather than assumed,
  plus the VoiceTut voice fix that had kept its voices silent. Four changes.
  **The concept graph is indexed.** `Ontology.match_concepts`/`concepts_in`/
  `find_concepts` used to rescan every concept's every term on every call, which
  made a single query take ~30 s on the real graph; `core/ontology.py` now keeps a
  token→concept index (rebuilt incrementally by `add_concept`/`add_concept_alias`,
  and by `core/concept.py`'s cached `normalized_terms`), so the same query is
  ~0.006 s with an *exact* equivalence to the old scan (verified term by term, so
  the recall is unchanged — only the cost is).
  **The retriever embeds a query once.** `core/rag.py` re-embedded the same query
  for every document (~160x per ask, once per document in the corpus); it now
  embeds the query once and caches each document's text, terms, and vector, so the
  semantic signal costs one embedding per ask. The per-document cache is
  invalidated per entry (a changed text drops its own row) rather than by a blanket
  generation counter, which had thrown away the whole cache on any write — a real
  correctness-preserving speedup, and the reason a learn-then-ask stayed fast.
  `core/embeddings.py`'s `_cosine` is a single pass over both vectors (with a new
  exported `dot_product`), so scoring no longer walks each vector twice.
  Measured: the answer path went from ~104 ms to ~46 ms per ask.
  **A governed web search reuses one connection.** `core/web_source.py` now goes
  through a pooled `requests.Session` (`_session`) instead of a bare
  `requests.get`, so the connect + TLS handshake (~0.9 s of a 1.3 s call) happens
  once per host rather than once per query.
  **The audio.cpp engine is fast by default and its voices really speak.** Three
  fixes to the same engine. (1) The **resident server is now the default**: a
  per-call `audiocpp_cli` process reloads the whole 1.35 GB model for every
  utterance, so `core/speech.py`'s `load_synthesizer("audio.cpp")` (and therefore
  `core/reading.TextReader`) uses the server whenever the build ships the binary —
  `use_server=False` is the only opt-out, and `core/reading.py`'s `server=None`
  means "auto". Measured on this machine: ~10 s per utterance through a fresh
  process, ~6.6 s through the resident server, and ~4.3 s at the new default step
  count. (2) **Fewer diffusion steps**: `DEFAULT_INFERENCE_STEPS` is now 4 (was 8;
  the model's own default is 32). Wall time is very nearly linear in the step
  count (2→2.4 s, 4→4.3 s, 8→8.6 s for one sentence), so four steps halves the
  wait against eight and keeps the speech intelligible; `steps` is a parameter, so
  the finer or faster setting is one argument away. (3) **The voices were verified
  end to end**: the 19 built-in speakers resolve, `model:audio.cpp:Sarah` renders
  2.64 s of real audio (RMS 3791, peak 20344 — not silence), and the CLI and the
  studio both reach it.   `tests/test_audiocpp_speech.py` pins the server default and
  the per-engine reader flag; `tests/test_web_learning.py` pins the pooled session;
  `tests/test_embeddings.py` pins the per-entry cache.

- Phase 36 — **the graph's multi-hop connections reach the answer path**:
  `core/relation_connections.py` answers the question a reader actually asks when
  two of a question's concepts have no single edge between them -- *how are they
  connected?* `core.relation_paths.find_connection` already walked the explicit
  relations breadth-first and returned the shortest typed chain, but that reasoning
  was reachable only from the `relations connect` CLI, never from an answer, so a
  multi-hop conclusion the program could draw never reached a reader.
  `concept_connections(ontology, question, ...)` finds the shortest chain between
  **each pair of concepts the question itself names** (via
  `Ontology.match_concepts`, de-duplicated by first mention), orders them by
  confidence then fewer hops then label (deterministic), bounds the block
  (`DEFAULT_CONNECTION_LIMIT`), and returns `()` when the question names fewer than
  two concepts or no chain within the hop bound connects them -- "not connected
  here" is a real answer, never a fabricated chain. Each `ConceptConnection`
  carries its hops, its confidence (the product of the steps', like a derivation),
  and each step rendered with the direction it was walked, so a step walked against
  the stored edge is visible; `describe_concept_connections` renders a block
  **labelled as walked, not stored**, so a reader can tell a multi-hop conclusion
  from a single written edge (and from an inference, Phase 28). It is read-only: it
  never asserts a path back into the graph and never invents an edge.
  `AssistantAnswer.connections` and `JarvisTurn.connections` carry the lines as
  *data*, `ai/chat.format_turn` renders them under the label, and `turn_payload`
  exposes them for `--json`. Wired as `python -m cli relations connections
  "<question>" [--limit N] [--max-hops N] [--json]` (read-only). The gate is
  `connection_reasoning_accuracy`: a held-out catalogue of hand-written pairs
  (`CONNECTION_JUDGEMENTS`) the answer-path walk must connect exactly when a chain
  exists **and** with the expected number of hops -- including negative controls (an
  unknown endpoint) -- so a walk that fabricates a chain or returns a wrong-length
  one fails as surely as one that misses a real connection; added to
  `evaluation/benchmark.py`, `evaluation/runner.py`, and `evaluation/baseline.json`.
  `tests/test_relation_connections.py` pins the pairs, the steps, the backward walk,
  the single-concept/unknown/no-chain empty cases, the determinism, the limits, the
  ordering, the labelled block, the negative controls, the gate, and the
  assistant/Jarvis/format wiring; `tests/test_relations_cli.py` pins the
  `connections` command, its JSON, its empty case, and its argument validation.

- Phase 37 — **the graph's causal chains reach the answer path**: the project
  carried a whole causal layer -- `CausalGraph`, `CausalLink`, `CausalChain`,
  `CausalReasoner`, `CausalExplainer`, `core.counterfactual` -- but it was reachable
  only from the `CognitiveController` and the evaluation scenarios: `CausalGraph`
  is empty by default and nothing in the answer path ever consulted it, so a reader
  asking *"why does waterflood increase production?"* got prose, never the
  mechanism the graph can walk. The seeded ontology, however, already states **117
  directed causal edges** (``causes``, ``drives``, ``controls``, ``reduces``,
  ``increases``, ``prevents``, ``influences``, ``affects``, ``determines``,
  ``governs``, ``creates``, ``produces``, ``enables``, ``improves``,
  ``maintains``, ``protects``) that no answer used.
  `core/causal_paths.py` reads those edges as a causal graph and walks the
  **strongest directed chain** from one concept the question names to another,
  rendering it with the project's own `CausalExplainer` wording.
  `CAUSAL_RELATION_TYPES` is a **closed set** so an associative relation
  (`measures`, `contains`, `is_a`) can never be read as a cause -- the single most
  important honesty rule here; `causal_graph_from_ontology` builds the
  `CausalGraph` from those edges only, carrying each relation's confidence and its
  own word for the mechanism. `causal_explanations(ontology, question, ...)` finds
  the strongest chain between **each ordered pair of concepts the question names**,
  in the direction the question's word order implies *and* the reverse (the same
  graph question asked two ways), keeping only chains that run **forward** through
  the causal edges -- so unlike the general relation walk (`core.relation_connections`)
  it **never reverses an edge**: "A causes B" does not license "B causes A". Results
  are ordered by confidence (strongest first), then fewer hops, then cause/effect,
  so the same question always yields the same block; it returns ``()`` when the
  question names fewer than two concepts or no causal chain connects them, so "the
  graph states no causal path here" is a real answer. `describe_causal_explanations`
  renders a block **labelled as directed (cause → effect)**, so a reader can tell a
  causal path the graph states from a general relation chain or a stored single
  edge. It is **read-only**: it never adds a link to the `CausalGraph`, never
  invents a causal edge.
  This phase also **fixed a pre-existing display bug** the wiring surfaced for the
  first time: `CausalExplainer.explain_chain` passed `chain.nodes` (which already
  spans the cause and the effect) into `causal_chain_sentence`, which prints the
  cause and the effect around its ``nodes`` argument, so both endpoints were
  repeated (``a → a → b → c → c``); it now passes only the intermediate concepts.
  `AssistantAnswer.causal_paths` and `JarvisTurn.causal_paths` carry the lines as
  *data*, `ai/chat.format_turn` renders them under the label (so the GUI shows them
  for free, because it renders through `format_turn`), and `turn_payload` exposes
  them for `--json`. Wired as `python -m cli relations causal "<question>"
  [--limit N] [--max-depth N] [--json]` (read-only). The gate is
  `causal_reasoning_accuracy`: a held-out catalogue of hand-written pairs
  (`CAUSAL_JUDGEMENTS`) the answer-path walk must connect exactly when a directed
  causal chain exists **and** with the expected number of links -- including
  **negative controls** (an associative-only pair such as porosity / permeability
  measures, which must yield no cause) -- so a walk that reads an association as
  causation fails as surely as one that misses a real mechanism; added to
  `evaluation/benchmark.py`, `evaluation/runner.py`, and `evaluation/baseline.json`
  (which also gained the missing `connection_reasoning_accuracy` term in `overall`).
  `docs/REASONING_ASSESSMENT.md` is the written assessment this phase answers: the
  reasoning layer's 69 modules, only 5 of which reached an answer before Phases
  28/36/37, and the ordered list of the remaining gaps (uncertainty in answer
  confidence, the groundedness gate, a user-visible reasoning trace, planning/
  decision/belief/temporal reach, cross-session state, and connected-path metrics).
  `tests/test_causal_paths.py` pins the closed causal set, the graph build, the
  single edge, the multi-hop chain, the no-reverse rule, the single-concept/unknown/
  no-chain empty cases, the determinism, the limits, the ordering, the product
  confidence, the labelled block, the endpoint-duplication fix, the negative
  controls, the gate, and the assistant/Jarvis/format wiring; `tests/test_relations_cli.py`
  pins the `causal` command, its JSON, its empty case, and its argument validation.
- Phase 38 — **uncertainty and graph reasoning reach the answer's confidence**:
  `AssistantAnswer` always carried a confidence, but it was computed from the
  retrieved passages **alone** -- `0.7 * top_score + 0.3 * coverage` -- and ignored
  everything else the answer path had just learned: the relations the graph
  *entails* (`core.relation_inference`), the chains it *walks*
  (`core.relation_connections`), the causal paths it *states* (`core.causal_paths`),
  and the claims it holds **contradictorily** (`core.relation_consistency`). So an
  answer resting on a graph-derived mechanism could not read as more confident than
  one resting on a single weak passage, and an answer the graph **contradicts**
  could not read as less -- the exact Gap 4 of `docs/REASONING_ASSESSMENT.md`.
  `core/answer_confidence.py` is now the one place that number is computed.
  `evidence_confidence` is the original formula, kept in one place so the answer
  path and its tests agree on the base term. A graph-derived signal may then fill
  at most `GRAPH_SUPPORT_WEIGHT` (0.25) of the room between the base and certainty,
  and **only when there is a base to fill**: with no retrieved evidence the
  confidence stays `0.0`, so a graph fact can never **manufacture** confidence in an
  answer the sources do not support. The support term is the **maximum** of the
  three signal groups, not their sum -- three weak derivations of one fact are one
  piece of support, not three. Stored contradictions are applied last and
  **multiply** the result down (`CONFLICT_PENALTY_WEIGHT` 0.5, severity summed and
  capped), so a contradiction is never outvoted by derived support and the number
  can never leave `[0, 1]`. The whole breakdown is returned as an
  `AnswerConfidence` (`evidence`, `graph_support`, `conflict_penalty`, `final`,
  `reasons`), so a reader can see exactly which term moved the number and by how
  much -- and `describe()` names the term that moved rather than a bare figure.
  `AssistantAnswer.confidence_report` carries it as *data*, `JarvisTurn` gained
  `confidence` and `confidence_note`, `ai/chat.format_turn` renders the note under
  a label (so the GUI shows it for free, because it renders through `format_turn`),
  and `turn_payload` exposes both for `--json`. `ConfidenceSignal`/`ConfidenceConflict`
  are the minimal shapes a caller (or the gate) can pass when it has the number but
  not the full object. The gate is `answer_confidence_accuracy`: a held-out
  catalogue of hand-written expectations (`CONFIDENCE_JUDGEMENTS`) about how each
  term **must** move the number -- support raises it, a contradiction lowers it, no
  evidence pins it at zero, and the result is always bounded -- including **negative
  controls**, so a model that manufactures confidence, lets a contradiction be
  outvoted, or produces an out-of-range value fails as surely as one that ignores
  the graph; added to `evaluation/benchmark.py`, `evaluation/runner.py`, and
  `evaluation/baseline.json` (overall 0.9790 → 0.9797). `tests/test_answer_confidence.py`
  pins the evidence term, the bounded support, the maximum-not-sum rule, the
  no-evidence-zero rule, the summed-and-capped penalty, the not-outvoted rule, the
  non-negative floor, the report validation and wording, the determinism, the
  assistant/Jarvis/format/payload wiring, and the gate (33 tests).
- Phase 39 — **the groundedness gate gets real teeth**: the gate that decides
  whether model text may be published -- `_is_grounded` in `core/assistant.py` --
  accepted an answer as soon as it shared **one** content term with the evidence.
  So a fluent paragraph could ride one incidental word into a published answer, a
  sentence unsupported by any passage was never noticed, and a model that
  **reversed** a mechanism the graph states (saying *"production drives waterflood"*
  when the graph stores ``waterflood → production``) was published as-is -- Gap 5 of
  `docs/REASONING_ASSESSMENT.md`.
  `core/grounding.py` is now the one place the gate lives. `check_grounding` makes
  three deterministic checks and **fails closed** -- when in doubt the caller falls
  back to the extractive answer, which is always traceable:
  (1) **term coverage** -- at least `DEFAULT_MIN_COVERAGE` (0.3) of the answer's
  content terms must appear in the evidence, not just one, and the ratio scales so
  a long answer sharing a word or two fails; (2) **per-sentence support** -- each
  content-bearing sentence must share a term with the evidence, and a majority of
  unsupported sentences (`DEFAULT_MIN_SENTENCE_SUPPORT` 0.5) fails the gate, so a
  paragraph cannot smuggle in material the sources do not carry; (3) **causal
  direction** -- if the answer states a causal link between two concepts in the
  direction the graph **contradicts** (the graph stores the reverse edge and not
  the forward one), the gate fails, applying the causal layer's own no-reverse rule
  to the model's words. The reversal check is deliberately conservative: it matches
  concepts on whole-phrase word boundaries (so ``pressure`` inside "reservoir
  pressure" is not counted as a second concept), reads both forward markers
  (``drives``) and reverse markers (``caused by``), and only fires when both
  endpoints are known causal concepts and the graph holds the opposite edge alone.
  `GroundingCheck` carries the coverage, the unsupported sentences, the reversals,
  and one readable reason per failed check; `_is_grounded` is kept as a thin
  backward-compatible wrapper. The held-out gate is `grounding_accuracy`: a
  catalogue of hand-written expectations (`GROUNDING_JUDGEMENTS`) with **negative
  controls** -- a fluent paragraph riding one incidental word, a supported sentence
  outnumbered by unsupported ones, a reversed mechanism -- so publishing
  untraceable or reversed text fails as surely as refusing a genuinely grounded
  answer; added to `evaluation/benchmark.py`, `evaluation/runner.py`, and
  `evaluation/baseline.json` (overall 0.9797 → 0.9804). `tests/test_grounding.py`
  pins the check object, the coverage rule, the no-evidence fail-closed rule, the
  per-sentence support (and its half threshold), the word-boundary marker match,
  the nested-endpoint drop, the forward/reverse/reverse-marker cases, the
  association-not-a-reversal rule, the gate with its negative controls, and the
  assistant wiring (32 tests).
- Phase 40 — **the reasoning trace reaches the user**: the reasoning layer was
  *auditable* but not *visible* -- a reader saw the answer and its citations, never
  **how** the answer was reached -- Gap 6 of `docs/REASONING_ASSESSMENT.md`.
  `core/reasoning_report.py` assembles one readable trace from data that was
  already on `AssistantAnswer`: the retrieved evidence, the graph relations the
  answer used, the multi-hop connections, the directed causal paths, any stored
  contradictions, and the confidence breakdown. It invents nothing -- every line
  comes from a value the answer already carried, and a step that contributed
  nothing says ``(nothing)`` rather than being hidden. `ReasoningStep`
  (``label``/``lines``/``derived``) and `ReasoningReport`
  (``question``/``steps``, with ``derived_steps`` selecting only the derivations
  that carry content, ``is_empty``, and a ``describe()`` that opens with *"How
  this answer was reached"*) are the two types; `reasoning_report(...)` is the one
  builder, and `STEP_ORDER` fixes the order so two runs read the same way. The
  trace reaches every surface the answer already reached:
  `AssistantAnswer.reasoning` is set in `GroundedAssistant.ask`;
  `JarvisTurn.reasoning_trace` carries its text, so `ai/chat.format_turn` and
  `turn_payload` render it -- and the GUI gets it for free, because the GUI draws
  turns through `format_turn`. A new Jarvis control, **`why`** (``why``, ``explain``,
  ``show your reasoning``, ``لماذا``, ``اشرح`` -- matched exactly, so a real question
  that merely *begins* with "why" still reaches the assistant), replays the last
  trace on demand and is cleared by a new conversation; `HELP_TEXT`/
  `TEXT_HELP_TEXT` name it. The CLI twin is
  `python -m cli relations why "<question>" [--limit] [--json]`, which answers and
  then prints the same trace. The held-out gate `reasoning_trace_accuracy`
  (`TRACE_JUDGEMENTS`) checks the *shape* of a trace with negative controls -- a
  trace with nothing must read empty, an evidence-only trace must keep every other
  step present-but-empty (never dropped), and a derivation must be labelled derived,
  not stored -- so a trace that invents a step, drops an empty one, or mislabels a
  derivation fails as surely as one that omits real evidence; it is a metric in
  `evaluation/benchmark.py` (`evaluation/baseline.json`, overall 0.9804 → 0.9810).
  `tests/test_reasoning_report.py` pins the step and
  report objects, the fixed order, the honest empty case, the confidence step, the
  assistant wiring, the ``why`` control (both directions -- it matches, and a
  "why does ...?" question does not), the new-conversation reset, the
  `format_turn`/`turn_payload` rendering, and the gate with its negative controls
  (36 tests).
- Phase 41 — **the plan reaches the answer**: the planning layer
  (`Planner`/`Action`/`Goal`/`PlanStep`) was reachable only from the
  `CognitiveController` and the evaluation scenarios, so a *procedural* question
  ("how do I measure porosity and permeability on a core sample?") got a
  definition back, never the ordered procedure the corpus already states (48
  entries carry `procedures`) -- Gap 7 of `docs/REASONING_ASSESSMENT.md`.
  `core/answer_plan.py` closes it. `is_procedural(question)` recognises a question
  that asks *how* (English and Arabic cues), so a definitional "what is porosity?"
  is never read as procedural and never drags a procedure into a definitional
  answer. `match_entry(question)` scores each entry by the content terms it shares
  with the question (its concepts, its relation endpoints, **and its own procedure
  text**), breaks ties on the entry key, and returns ``None`` for a question that
  shares no term -- never a guess. `order_procedure(steps, ...)` runs the corpus's
  own steps through the project's **real** `Planner.order_steps`, so a circular or
  dangling dependency is **refused** with ``ValueError`` rather than silently
  reordered. `answer_plan(question)` builds an `AnswerPlan`
  (``topic``/``source``/``steps`` + a numbered ``describe()``) only when the
  question is procedural *and* an entry with a stated procedure matches it, else
  ``None``. Nothing is executed: the plan is a *reported* artifact.
  The plan reaches every surface the answer already reaches:
  `AssistantAnswer.plan` is set in `GroundedAssistant.ask`; it becomes a new
  `SOURCE_PLAN` step in `core.reasoning_report` (fixed position: after the stored
  claims, before the other derivations, because a plan *is* a derivation), so the
  user-visible trace now shows the procedure too; `JarvisTurn.plan` carries the
  numbered steps, so `ai/chat.format_turn` and `turn_payload` render them (and the
  GUI gets them for free); and `python -m cli relations why` prints them inside the
  trace and in its ``--json``. The held-out gate `plan_accuracy`
  (`PLAN_JUDGEMENTS`) checks the bridge with **negative controls** -- a definitional
  or unrelated question must yield no plan, a circular or dangling dependency must
  be *refused*, and steps given out of order must come back in dependency order --
  and is a metric in `evaluation/benchmark.py` (`evaluation/baseline.json`, overall
  0.9810 → 0.9816). `tests/test_answer_plan.py` pins the procedural cue (both
  languages, and its negative), the ordering through the real planner, the
  refusals, the matching, the ``None`` cases, the trace step, the assistant/Jarvis/
  `format_turn`/`turn_payload` wiring, and the gate with its negative controls
  (30 tests).
- Phase 42 — **the decision reaches the answer**: the decision layer
  (`DecisionEngine`/`UtilityModel`/`RiskModel`/`DecisionExplainer`) was reachable
  only from the `CognitiveController` and the evaluation scenarios, so a
  *decisional* question ("what should I do to increase permeability?") got a
  definition back, never the ranked recommendation the knowledge graph already
  states -- the decision half of Gap 7 of `docs/REASONING_ASSESSMENT.md`.
  `core/answer_decision.py` closes it. `is_decisional(question)` recognises a
  question that asks what to do (English and Arabic cues), so a definitional "what
  is permeability?" is never read as decisional and never drags a recommendation
  into a definitional answer. `direction_of(question)` reads the direction asked
  for (`increase`/`reduce`/`prevent`/`maintain`/`control`), scanned in a fixed
  order so the same question always yields the same direction. The options are the
  graph's own **typed inbound edges** into the concept the question is about,
  restricted to the relation types that match the direction asked for, so an option
  is never offered when the graph says it moves the goal the *other* way -- "what
  should I do to reduce permeability?" returns ``None`` rather than offering an
  edge that increases permeability. Each option is an `Action` that adds the goal
  at a cost of ``1 - confidence``, and a source the graph links to several of the
  question's goals becomes **one** action advancing all of them (the
  multi-objective case the utility model exists for). `answer_decision(question,
  ontology=...)` runs those actions through the project's **real**
  `DecisionEngine.decide` and ranks them with `DecisionExplainer.explain`, and
  returns an `AnswerDecision` only when the question is decisional *and* the graph
  states an option in the direction asked for, else ``None`` -- never a guess.
  Nothing is executed: the recommendation is a *reported* artifact.
  The recommendation reaches every surface the answer already reaches:
  `AssistantAnswer.decision` is set in `GroundedAssistant.ask`; it becomes a new
  `SOURCE_DECISION` step in `core.reasoning_report` (fixed position: after the
  plan, before the other derivations); `JarvisTurn.decision` carries the ranked
  lines, so `ai/chat.format_turn` and `turn_payload` render them (and the GUI gets
  them for free); and `python -m cli relations why` prints them inside the trace
  and in its ``--json``. The held-out gate `decision_recommendation_accuracy`
  (`DECISION_JUDGEMENTS`) checks the bridge with **negative controls** -- a
  definitional, unrelated, or unanswerable question must yield no recommendation --
  and is a metric in `evaluation/benchmark.py` (`evaluation/baseline.json`, overall
  0.9816 → 0.9821). `tests/test_answer_decision.py` pins the decisional cue (both
  languages, and its negative), the direction reading, the ranking, the
  direction-only options, the ``None`` cases, the trace step, the
  assistant/Jarvis/`format_turn`/`turn_payload` wiring, and the gate with its
  negative controls (34 tests).

- Phase 43 — **the belief reaches the answer**: the belief layer
  (`Belief`/`BeliefRevisionEngine`/`ConfidenceModel`/`ContradictionDetector`) was
  reachable only from the `CognitiveController` and the evaluation scenarios, so an
  *evaluative* question ("does acid stimulation increase permeability?") got a
  definition back, never the evidence-weighed verdict the knowledge graph already
  supports -- the belief half of Gap 7 of `docs/REASONING_ASSESSMENT.md`.
  `core/answer_belief.py` closes it: `is_evaluative` (English + Arabic cues,
  anchored to the question's opening) recognises a verdict question, the evidence
  is the graph's own typed edges between the concepts the question names (each an
  `Evidence` record carrying the relation type, so the engine's
  `ContradictionDetector` can spot two antonymous relations on the same pair), and
  the confidence and lifecycle status are the ones `BeliefRevisionEngine.revise`
  actually produced. It is read-only, never invents evidence, and returns ``None``
  rather than a guess. `AssistantAnswer.belief`, a `SOURCE_BELIEF` trace step,
  `JarvisTurn.belief`, and `python -m cli relations why` carry it; the held-out
  gate `belief_accuracy` checks it with negative controls (overall 0.9821 →
  0.9827); `tests/test_answer_belief.py` pins it.

- Phase 44 — **the reasoning reaches the answer BODY, not just the trace**:
  Phases 41-43 let a procedural, decisional, or evaluative question reach the
  planner, the decision engine, and the belief revision engine -- but the bridge
  stopped at the trailing *trace*. The answer **body** was still the retrieval
  template ("Here is what the available sources say about ..."), so the reasoning
  sat in a footer a reader may never open, and the evaluation could pass while the
  prose a user actually reads was unchanged. `core/answer_lead.py` closes that last
  step: `answer_lead(question, plan=, decision=, belief=)` builds the lead
  strongest-first -- procedure, then recommendation, then verdict -- and only the
  first that applies, so a question is never handed two competing leads. Each lead
  is a *named* source (the procedure heading names the corpus entry, the
  recommendation lists the graph's ranked options, the verdict states the claim,
  its confidence, and the edges it rests on), and ``None`` means the question
  warranted no reasoning, so a definitional question keeps its retrieval opening
  and nothing is invented to fill a lead. `core.composition.with_lead(question,
  lead, body)` joins the lead to the cited body, and `grounded_answer(...,
  lead=...)` gained the parameter, so the reasoning opens the answer while the
  evidence still follows -- whether a model wrote the body or the extractive
  fallback did, so neither silently replaces the other. The reasoning is on the
  machine-readable turn too (`JarvisTurn.lead`, `turn_payload["lead"]`). The
  held-out gate `lead_accuracy` (`LEAD_JUDGEMENTS`) checks it with negative
  controls and runs the *real* answer path through the assistant's own ``ask``
  (`answer_for=`), so a regression that left the reasoning in the trace while the
  body kept the template would lower the metric (overall 0.9827 → 0.9832). Two
  related fixes landed with it: the plan bridge's `match_entry` now reads the
  corpus **glossary** (where each term's Arabic alias lives), so an Arabic
  procedural question ("كيف أقيس المسامية؟") matches the entry and reaches the
  planner; and `core.composition.topic_phrase` no longer emits a broken clause
  ("how does porosity relate to permeability" → "porosity permeability").
  `tests/test_answer_lead.py` and `tests/test_answer_plan.py` pin the lead, its
  negative controls, the body check, the Arabic match, and the turn payload.

- Phase 45 — **the timeline reaches the answer** (the temporal half of Gap 7):
  the project carried a whole temporal layer (`TemporalReasoner`, `Timeline`,
  `TemporalEvent`, `TemporalPattern`, `TimePrediction`) but it was reachable only
  from tests, so a question like *"what changed over the session?"* got a
  definition back, never the ordered account of what actually happened.
  `core/answer_temporal.py` closes it: `answer_temporal(question, *,
  conversation=, events=)` reads the one real timeline the answer path already
  holds -- the conversation's own **timestamped turns** -- and runs them through
  the project's real `TemporalReasoner`, so the order is chronological, the
  relation between the last two turns is the reasoner's own
  (`before`/`during`/`simultaneous`...), and any repeating pattern and its
  next-time prediction are the ones `find_pattern`/`predict_next` actually
  produced. It is gated to a *temporal* question (`is_temporal`, `TEMPORAL_CUES`
  in either language), skips `SYSTEM`/`CONTEXT` turns (a system instruction and an
  attached file are not things that happened in the conversation), and returns
  ``None`` rather than a guess when there is no timeline (fewer than two turns, or
  no conversation). `AnswerTemporal` carries the entries, the relation, the
  readable change, the pattern, the prediction, and the language; `span_seconds`
  measures the wall-clock window. The timeline can open the answer as a lead
  (`core.answer_lead`'s `KIND_TIMELINE`, tried after procedure, recommendation,
  and verdict), and it is carried on the machine-readable turn
  (`JarvisTurn.temporal`, `turn_payload["temporal"]`) and in the reasoning trace
  (`core.reasoning_report.SOURCE_TEMPORAL`). The held-out gate
  `temporal_accuracy` (`TEMPORAL_JUDGEMENTS`) checks it with **negative
  controls** -- a definitional or unrelated question, or a temporal question with
  one event, must yield no timeline -- and is a metric in
  `evaluation/benchmark.py` (overall 0.9832 → 0.9836).
  `tests/test_answer_temporal.py` pins the cue gate, the chronological order, the
  latest change, the pattern and its prediction, the conversation source, the
  lead, the body/trace/turn wiring, and the gate with its negative controls.

- Phase 46 — **the structural analogy reaches the answer** (the analogical half of
  Gap 7, and its last isolated engine): the project carried a whole analogical
  layer (`AnalogyEngine`, `StructureMappingEngine`, `Structure`, `StructureMapping`)
  but it was reachable only from the controller, `cli/analogy.py`, and the
  evaluation scenarios, so *"is porosity like permeability?"* got a definition
  back -- or, worse, the belief bridge read its `is ... ?` opening as a verdict
  about a *different* claim ("does porosity increase permeability?").
  `core/answer_analogy.py` closes it: `answer_analogy(question, *, ontology=,
  hops=, limit=)` reads the concepts the question names, builds each one's
  relational neighbourhood, and aligns them with the project's real `AnalogyEngine`
  (through `StructureMappingEngine`), so the comparison that reaches the reader is
  the one the structure-mapping engine actually produced -- its matched relations,
  its systematicity, and its candidate inferences (each a *candidate*, discounted
  by how much of the structure really aligned, and never asserted into the graph).
  It is gated to an *analogical* question (`is_analogical`, `ANALOGY_CUES` in
  either language, over a normalised question so "what is porosity like?" fires
  like an interior "like"), and returns ``None`` rather than a guess when the
  question names fewer than two known concepts. It is **honest by construction**:
  it reports the engine's own `meaningful` verdict even when it is `False` (a pair
  sharing no relational system is said to be *not* an analogy, not dressed up as
  one) and reports what it could **not** map as well as what it could. The
  comparison can open the answer as a lead (`core.answer_lead`'s `KIND_ANALOGY`,
  between recommendation and verdict), and it is carried on the machine-readable
  turn (`JarvisTurn.analogy`, `turn_payload["analogy"]`), in the reasoning trace
  (`core.reasoning_report.SOURCE_ANALOGY`), and in `cli relations why --json`; it
  also stops the belief bridge from answering a comparison as a claim. The held-out
  gate `analogy_answer_accuracy` (`ANALOGY_ANSWER_JUDGEMENTS`) checks it with
  **negative controls** -- a definitional question, an unrelated one, or an
  analogical question naming one concept must yield no comparison, and a pair with
  no shared relational system must be reported as *not* an analogy -- and is a
  metric in `evaluation/benchmark.py` (overall 0.9836 → 0.9841).
  `tests/test_answer_analogy.py` pins the cue gate, the comparison, the honest
  non-analogy verdict, the unmapped report, the lead, the body/trace/turn wiring,
  the belief suppression, and the gate with its negative controls.

- Phase 47 — **learned graph knowledge survives a session boundary** (Gap 8): the
  project carried a whole persistence layer (`KnowledgeStore`, `Snapshot`,
  `Persistence`, `JsonStore`) but it was never wired to the answer path -- the graph
  an answer *read* was rebuilt from the seeds on every process start, so a typed
  relation the architecture **learned** (from a confirmed sentence, an ingested
  document) was gone the next time. `core/knowledge_carry.py` is the bridge:
  `LearnedRelation` keys a relation by its two concepts' **labels** (never an
  internal `CON-…` id, which is regenerated every run, so a record written by one
  run resolves in another); `capture_learned(ontology, learner)` returns the graph's
  *learned* relations **and only those** (a relation counts as learned only when the
  learner created it, so a curated seed is never captured -- it is in every session
  already and restoring it could not change an answer); `restore_learned(ontology,
  learner, records)` replays records through the **same** `OntologyLearner`, so each
  replayed link stays bounded by the learner's ceiling and refuses to touch a seeded
  relation -- a record naming an unknown concept is **reported skipped, never
  invented**, a record whose pair already exists is counted present (never
  duplicated), and a second restore is a no-op. `LearnedKnowledgeStore` is the
  durable wrapper over the `Persistence` protocol (one versioned document under
  `CARRY_NAMESPACE`/`CARRY_KEY`), and an unreadable record is dropped rather than
  crashing the session. **Nothing runs on its own**: capture and restore are explicit
  calls, so a read-only session is untouched. Wired into the answer path:
  `GroundedAssistant` gains `remember_relation()` (writes a relation, drops the
  retrieval cache, captures the learned set), `recall_learned()` (replays the
  previous session's learned relations before the first question) and
  `learn_from_text()` (reads a sentence with the real `RelationExtractor` and
  captures the result); `build_default_assistant(learned=…)` recalls at build time;
  `JarvisMode`'s feedback path feeds a confirmed/corrected statement to
  `learn_from_text`; and `ai/jarvis.build_session` opens the durable store under
  `data/learned-knowledge` (`_learned_store()`, falling back to no carry on a
  read-only data directory), so a CLI/GUI session really accumulates. The held-out
  gate `knowledge_carry_accuracy` (`CARRY_JUDGEMENTS`) checks it with **negative
  controls** -- a curated seed is never captured as learned, a record naming an
  unknown concept is skipped, and a second restore does not duplicate -- and is a
  metric in `evaluation/benchmark.py` (overall 0.984071 → 0.984516). CLI:
  `python -m cli relations carry "<statement>" [--show] [--json]`.
  `tests/test_knowledge_carry.py` pins the record, the seed exclusion, the
  round-trip, the idempotence, the unknown-concept skip, the durable store, the
  assistant wiring, and the gate with its negative controls.

- `core/answer_audit.py` — **the evaluation measures the *connected* answer path,
  not the isolated layer** (Gap 9). Phases 37-47 bridged every reasoning layer to
  the answer (causality, uncertainty, the groundedness gate, the trace, planning,
  decision, belief, temporal, analogy, and the knowledge carry) and gave each
  bridge its own held-out metric — but those metrics mostly exercise the bridge
  **in isolation** (they call `answer_plan`/`answer_decision`/`answer_belief`/...
  directly, or hand the assistant a single question), so a layer that stops
  reaching the *answer object* the user actually receives could still pass. This
  module is the honest answer: it asks the **real assistant** its own questions
  and checks what the answer *object* carries. `answer_layers(answer)` reads the
  optional objects (plan/decision/belief/analogy/temporal/lead/reasoning/
  confidence) and the non-empty tuples (facts/inferred/connections/causal) straight
  off the answer — never guessing, so a layer the bridge produced but the answer
  dropped is simply absent. `AuditExpectation` names the layers a question's answer
  **must** carry (`required`), the layers it must **not** (`forbidden`, the
  negative controls), whether the lead must open the **body** (`body`, not only the
  object), and whether the check must hand the assistant a short probe conversation
  (`timeline`, because a timeline needs at least two turns to order).
  `AUDIT_EXPECTATIONS` is the held-out catalogue: a procedural question must carry
  the plan and the procedure lead, a decisional one the recommendation, an
  evaluative one the verdict, an analogical one the comparison, a temporal one the
  timeline, a connected one the walked connections, a definitional one the graph's
  stored facts — each positive also requiring the trace — while a definitional,
  unrelated, or vague question must carry **no** reasoning layer. `audit_answer`
  runs the real `ask`, reads the layers off the object, checks the two sets, and
  (when asked) that the lead opens the body; a missing answer or a raised error is
  a failure, not a skip. `answer_path_accuracy(expectations, answer_for)` is the
  share of expectations the connected path satisfies; with no `answer_for` the path
  is not measured, so the honest answer is `0.0` rather than a faked pass. It is
  **proven to discriminate**: dropping the lead fails the path (`0.0`), leaking a
  layer into a negative control fails it (`0.9`), and no path scores `0.0` — three
  regressions an isolated-layer metric would have passed. Wired into the evaluation
  as the metric `answer_path_accuracy` (`evaluation/benchmark.py`/`runner.py`/
  `baseline.json`, entering `overall`, 0.984516 → 0.984911).
  `tests/test_answer_audit.py` pins the catalogue, the layer reading, the negative
  controls, the body check, the discrimination, and the honest `0.0`. Phase 49 adds
  `AUDIT_EXPECTATIONS_AR`, the **Arabic-parallel** catalogue (ten expectations, each
  English one twinned with an Arabic question asking for exactly the same layers),
  so a regression that breaks only the Arabic path fails while the English
  catalogue stays green.

- `core/arabic_glossary.py` — **Arabic-first, for real** (Phase 49). The project is
  Arabic-first by design, but the ontology did not reflect it: of 495 seeded
  concepts only **39 carried an Arabic alias**, so an Arabic question reached almost
  none of the graph (the measured symptom: `"هل التحميض يزيد النفاذية؟"` never
  reached the belief layer, `"هل المسامية مثل النفاذية؟"` never reached the analogy).
  `ARABIC_ALIASES` is a curated **English-label → Arabic-term** table (230 entries:
  `porosity → مسامية`, `acid_stimulation → التحميض`, `blowout_preventer → مانع
  الانفجار`, ...); `apply_arabic_aliases` resolves each entry by its **label** --
  never an internal `CON-…` id, which carries a random suffix that changes every run,
  and never a shared alias -- and adds it through `Ontology.add_concept_alias`, which
  **keeps the concept's id**, so every relation that references it survives.
  `_concept_for_label` matches the label exactly (case-folded): resolving by *word*
  matched `acidizing` -- an alias of both `well_stimulation` and `acid_stimulation`
  -- to the wrong concept and silently lost the Arabic term, and
  `test_glossary_resolves_a_label_exactly_not_by_a_shared_alias` pins the fix. The
  pass is idempotent (a term already present is a no-op) and a concept may carry more
  than one Arabic term, so a synonym is added beside the primary one. Honest by
  construction: a label the ontology does not hold is **reported** in
  `GlossaryReport.unknown` and skipped -- nothing is invented -- and
  `missing_arabic_labels`/`arabic_alias_coverage` report the remaining gap (coverage
  rose from **39/495 to 237/495**) rather than hiding it. `has_arabic` is a
  dependency-free script check; `normalize_arabic_term` reuses the project's own
  `core.tokenize.normalize_phrase`. Wired into `CognitiveController.__init__` **and**
  `build_default_assistant` before the retriever indexes the graph, so an Arabic
  question reaches the same concepts an English one does.
  `tests/test_arabic_glossary.py` pins the script detection, the idempotence, the
  unknown-label report, the relation safety, the exact-label resolution, the
  coverage, and that an Arabic question reaches the plan/decision/belief/analogy
  layers while an unrelated one stays plain.
- `core/arabic_prose.py` — the Arabic **output** half (Phase 50). Phase 49 closed
  the input (an Arabic question reaches the graph); the output was still English:
  the language model's training corpus was **0.00% Arabic** and every citable
  document was English, so an Arabic answer's body was English even under an Arabic
  reasoning frame. This renders the graph's own knowledge into Arabic.
  `arabic_label(label, ontology)` resolves a concept label through the ontology's
  aliases first (the Phase 49 glossary plus the petroleum corpus's own glossary),
  then the curated `TOPIC_ARABIC` table (the disciplines, courses, and corpus-entry
  titles -- a *phrase* the concept glossary never held), then returns the English
  label unchanged; it never invents a term, so a Latin identifier the domain keeps
  (`SPE`, `API`) is left alone. `RELATION_ARABIC` maps the seeds' 160 relation types
  onto an Arabic clause (`measures → يقيس`, `prevents → يمنع`), and
  `DEFAULT_RELATION_ARABIC` (`يرتبط بـ`) is the fallback for a relation the map does
  not name, so no edge is silently dropped. `arabic_relation_sentence` renders
  ``"{source} {clause} {target}."``; `arabic_curriculum_sentences`,
  `arabic_corpus_sentences`, and `arabic_procedural_sentences` produce the training
  text (`CORE_ARABIC_SENTENCES` is the Arabic twin of the English general block);
  `arabic_document_lines` and `arabic_document_sections` produce the citable
  documents (one per discipline, course, and corpus entry, so retrieval ranks the
  relevant Arabic text first rather than one large block whose opening names an
  unrelated discipline). It is **additive and presentation-only**: it reads the
  ontology and returns text, writes nothing, and leaves every English block exactly
  as it was. Wired into `ml.llm.corpus.arabic_sentences()` (the corpus's Arabic
  share moved from **0% to ~20%**) and `core.documents._arabic_documents()`.
  `tests/test_arabic_prose.py` pins the renderer, the tables, the label resolution,
  the composer bug it exposed, and the corpus/document wiring.

- Phase 48 — **the evaluation measures the connected answer path, not the isolated
  layer** (Gap 9): Phases 37-47 bridged every reasoning layer to the answer and gave
  each bridge a held-out metric, but those metrics mostly exercised the bridge *in
  isolation*, so a layer that stopped reaching the answer the user actually receives
  could still pass. `core/answer_audit.py` closes that: `answer_layers` reads the
  layers off the **answer object** the real `ask` returned, `AuditExpectation` names
  the layers a question's answer must and must not carry (with the lead required to
  open the **body**, and a probe conversation for the timeline), `AUDIT_EXPECTATIONS`
  is the held-out catalogue with negative controls (a definitional, unrelated, or
  vague question must carry no reasoning layer), `audit_answer` runs the real path,
  and `answer_path_accuracy` is the share it satisfies — `0.0` when no path is
  exercised rather than a faked pass. It is **proven to discriminate**: dropping the
  lead scores `0.0`, leaking a layer into a negative control `0.9`, and no path
  `0.0`. Wired into `evaluation/benchmark.py`/`runner.py`/`baseline.json` as the
  metric `answer_path_accuracy` (entering `overall`, 0.984516 → 0.984911), with
  `tests/test_answer_audit.py` pinning the catalogue, the layer reading, the
  negative controls, the body check, the discrimination, and the honest `0.0`. This
  completes the nine gaps of `docs/REASONING_ASSESSMENT.md`; the next step is to
  **widen** the held-out catalogue (parallel Arabic questions, broader negative
  controls, multi-step path metrics) rather than close another gap.

- Phase 49 — **Arabic-first, for real**: the project is Arabic-first by design
  (English code/menus/docs, Arabic as secondary aliases), but the ontology did not
  reflect it -- of 495 seeded concepts only **39 carried an Arabic alias**, so an
  Arabic question reached almost none of the graph. The measured symptom was
  `"هل التحميض يزيد النفاذية؟"` (does acidizing increase permeability?) never
  reaching the belief layer and `"هل المسامية مثل النفاذية؟"` never reaching the
  analogy, because neither `التحميض` nor the Arabic cue resolved. Three pieces close
  it. `core/arabic_glossary.py` (new) is a curated **English-label → Arabic-term**
  table of 230 entries (`porosity → مسامية`, `acid_stimulation → التحميض`,
  `blowout_preventer → مانع الانفجار`, ...), applied through
  `Ontology.add_concept_alias`, which **keeps each concept's id**, so every relation
  survives; it resolves by **label, never by an id** (ids carry a random suffix that
  changes every run), is idempotent, and **reports** a label the ontology does not
  hold in `GlossaryReport.unknown` rather than inventing a concept. A real regression
  was caught and fixed here: resolving by *word* matched `acidizing` -- an alias of
  both `well_stimulation` and `acid_stimulation` -- to the wrong concept and silently
  lost the Arabic term; `_concept_for_label` now matches the label exactly, pinned by
  `test_glossary_resolves_a_label_exactly_not_by_a_shared_alias`. Coverage rose from
  **39/495 to 237/495**, and `missing_arabic_labels`/`arabic_alias_coverage` report
  the remaining gap honestly rather than hiding it. `core/answer_audit.py` gains
  `AUDIT_EXPECTATIONS_AR`, a **ten-expectation Arabic-parallel catalogue** -- every
  English expectation has an Arabic twin asking for *exactly the same layers*, with
  the same three negative controls -- and `test_arabic_catalogue_is_a_true_parallel_
  of_the_english_one` keeps the two from drifting. The metric
  `arabic_answer_path_accuracy` (in `evaluation/runner.py`, `benchmark.py`, and
  `baseline.json`) makes the Arabic path a gate of its own, because a green English
  metric let an Arabic-only regression pass before. The glossary is wired into
  `CognitiveController.__init__` **and** `build_default_assistant` before the
  retriever indexes the graph. `tests/test_arabic_glossary.py` pins the script
  detection, the idempotence, the unknown-label report, the relation safety, the
  exact-label resolution, the coverage, and that an Arabic question reaches the
  plan/decision/belief/analogy layers while an unrelated one stays plain. Gates:
  4165 passed / 200 skipped, ruff clean, mypy 748 files OK, evaluation PASS
  (`overall` 0.984911 → 0.9853).
- Phase 50 — **the Arabic *output* half** (prose + citable documents). Phase 49
  closed the *input*: an Arabic question finally reached the ontology. The output
  was still English, and measurably so -- the language model's training corpus was
  **0.00% Arabic** across `curriculum_sentences` (1139), `corpus_sentences` (836),
  and `procedural_sentences` (146), so the in-project model could not speak Arabic
  at all; and every citable document was English, so an Arabic answer's *body* was
  English even when its Arabic reasoning frame ("هذه خلاصة المصادر عن المسامية") was
  Arabic (measured: 4-7% Arabic letters). `core/arabic_prose.py` (new) renders the
  graph's **own** knowledge into Arabic and closes both: `arabic_label` resolves a
  concept label through the ontology's aliases (the Phase 49 glossary plus the
  petroleum corpus's own glossary) and then a curated `TOPIC_ARABIC` table of the
  disciplines, courses, and corpus-entry titles, falling back to the English label
  for anything it does not hold (a Latin identifier the domain keeps, like `SPE`,
  is left alone -- it never invents a term); `RELATION_ARABIC` maps the seeds'
  relation vocabulary onto an Arabic clause with a **default clause** so an edge the
  map does not name is never dropped; and `arabic_curriculum_sentences`,
  `arabic_corpus_sentences`, `arabic_procedural_sentences`, `arabic_document_lines`,
  and `arabic_document_sections` produce the training text and the citable documents
  (one document per discipline, course, and corpus entry, so retrieval ranks the
  relevant Arabic text first rather than one large block). It is **additive**: no
  English block changes. Wiring: `ml.llm.corpus.arabic_sentences()` joins
  `build_corpus`/`corpus_units` (the corpus's Arabic share moved from **0% to
  ~20%**), and `core.documents._arabic_documents()` joins
  `build_knowledge_documents()`. The wiring exposed a **real bug** in
  `core.composition`: `_SENTENCE_RE` split only before an ASCII capital/digit and
  `complete_sentences` dropped any part not starting with one -- `str.isupper()` is
  `False` for Arabic (the script is caseless), so *every Arabic sentence was silently
  discarded as a fragment*; the splitter and `_starts_sentence` now accept the Arabic
  letter blocks, and `_ARABIC_CONNECTORS` joins an Arabic answer's statements with
  Arabic words instead of `"First, "`. Measured end to end, an Arabic question's
  answer body moved from **4-7% Arabic to 68-100% Arabic** while an English question
  stays English (no regression). One honest regression was caught and fixed:
  `semantic_separation` fell 0.5427 → 0.3564 because `repository_sentences` (a
  catch-all) swept the new Arabic documents into the held-out semantic-space
  evaluation and diluted it; excluding the dedicated `arabic_prose` source there
  restored it exactly. `tests/test_arabic_prose.py` pins the renderer, the tables,
  the label resolution, the composer bug, and the corpus/document wiring. Gates:
  pytest 4190 passed / 200 skipped, ruff clean, mypy 750 files OK, evaluation PASS
  (`overall` 0.9853 unchanged, `semantic_separation` 0.5427 unchanged).
