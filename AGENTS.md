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
  `pytesseract`, `PIL`, `diffusers`, `llama_cpp`, `huggingface_hub`, `pptx`).
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
  copies any pre-upgrade in-repo data into `data/` on first run. `studio_dir()`
  (`data/studio`) and `studio_workspaces_dir()` (`data/studio/workspaces`) are
  where the Code Studio keeps its git worktrees, git-ignored like the rest of
  `data/`.
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
- `gui/window_controls.py` — the shared **minimize / maximize** controls every
  dialog window installs. `install_window_controls(window, header, theme=,
  typography=, on_close=)` builds one frame of two buttons into the window's
  header and returns a `WindowControls`; `WindowControls` grows the window with
  `state("zoomed")` where the window manager supports it and falls back to a real
  full-screen window where it does not (many X11 desktops), restoring exactly
  what it did. Rolled into the eight dialog windows (Code Studio, Presentation
  Studio and its finalize/course windows, the Import Center, the SQL import
  screen, and the two image windows) so a secondary window behaves like the main
  one without each dialog re-implementing the logic.
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
  (PCM/WAV), transcript + silent fallbacks, `VoiceAssistant` voice loop. The TTS
  engine itself is a *system* program (espeak-ng/espeak/flite/`say`), not a Python
  package, so pip cannot install it — like Tesseract for OCR. `speech_available`,
  `speech_install_plan` (a platform-aware package-manager command — incl. winget/
  choco on Windows — or an honest hint when the app knows none), and
  `install_speech_engine` (never raises; an
  injected `runner` makes it testable with no real install) give the app a real
  next step when no engine is present. **Windows needs no install**: the OS ships
  SAPI, so `WindowsSapiSynthesizer` renders through `System.Speech` in PowerShell
  (`windows_speech_available`; `discover_synthesizer_name` falls back to it, and
  `load_synthesizer("windows-sapi")` loads it), and `Speaker` plays the WAV through
  the built-in `SoundPlayer` when no aplay/afplay/sox exists. The install path
  prints its hint *once* (`install_speech_engine`'s message already carries it).
  CLI: `python -m ai.voice` / `python -m cli
  voice`; `python -m ai.speak --install-engine` / `python -m cli deps --install
  speech` run it.
- `core/voices.py` — the text-to-speech **voice catalog**: 16 `Voice`s in four
  categories (`man`/`woman`/`boy`/`girl`, four each), each naming a timbre
  (engine `variant` `m1`..`m6` / `f1`..`f4`), a pitch, and a speed.
  `CATEGORY_LABELS`/`voice_categories`, `voices_for`, `voice_keys`, `find_voice`
  (raises on an unknown key), `default_voice`/`DEFAULT_VOICE_KEY`, `voice_summaries`,
  and `detect_language` (Arabic script → `ar`, else `en`) so one voice reads both
  languages. Honest by construction: a `Voice` is *settings*, not a recording, and
  `description` says so — `espeak-ng` has no true child voice, so `boy`/`girl` are
  a raised pitch on a male/female variant and say that plainly.
  `core/speech.py`'s `CommandSynthesizer`/`load_synthesizer` take that voice and
  insert its engine flags after the executable (`_espeak_voice_options`; `say`
  carries only the rate; `flite` has none), and `load_synthesizer("auto")` still
  falls to `silent` when nothing is installed.
- `core/reading.py` — `TextReader`: read the chat's text aloud. Ties a catalog
  `Voice`, a `SpeechSynthesizer`, and a `Speaker` together behind `render`/
  `speak` (playback is best-effort, so a machine with no player still gets the
  clip)/`save` (WAV), `set_voice`, `available`, `refresh` (re-discovers the engine
  after one is installed mid-session), and an honest `describe()` that names the
  voice and says when no engine exists rather than pretending. CLI:
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
  download. The runtime is the optional `[qwen-speech]` extra
  (`requirements-qwen-speech.txt`), imported lazily. **Honest by construction:**
  Qwen3-ASR covers **Arabic** (52 languages), but **Qwen3-TTS does not speak
  Arabic** (ten languages: Chinese, English, Japanese, Korean, German, French,
  Russian, Portuguese, Spanish, Italian) — `QWEN_TTS_LANGUAGES` states this, the
  synthesizer refuses Arabic with a clear message, and a `fallback` synthesizer
  (the system engine) can be supplied so Arabic is still read aloud.
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
  and the two chat modes). A `GuiAction` carries a `hidden` flag: eight entries
  stay in `ACTIONS` (so `action_by_key` and the CLI still resolve them) but are
  `hidden=True`, and `visible_actions()` — what the sidebar actually draws —
  leaves them out. The four *Answer from ...* mouth switches are hidden because
  the header's model selector offers them, and the Import Center, Code Studio,
  Presentation Studio, and image generation are hidden because the composer's ＋
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
  appears only when the content overflows, a header showing the active mode by
  name (one switch) plus the **model selector** (`MOUTH_CHOICES`: *From sources
  (cited)* / *From trained model* / *Ultron (general local model)*, matched to the
  mouth keys `ai.chat` uses), and a centred hero/empty state that
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
  training loop never freezes the window. The composer's **＋** menu also opens
  the **Code Studio** (below).
  The window also **reads text aloud**: every message footer carries a
  "🔊 Read aloud" control (`_speak_label`/`_speak_text`), and the Settings panel
  holds the reading-voice menu (`_build_voice_picker`) — every catalog voice,
  grouped under Men/Women/Boys/Girls with its English and Arabic name — plus an
  opt-in **auto-speak** check and a "Test the reading voice" button. The chosen
  voice and the auto-speak flag persist through `JsonStore`
  (`load_reading_voice`/`save_reading_voice`, `load_auto_speak`/`save_auto_speak`,
  the same `gui`/`appearance` namespace as the theme), the window and the service
  agree from the first message (`ChatService.set_voice`), and Markdown is stripped
  (`_strip_markdown`) so the engine speaks words, not `**`/backticks. Reading runs
  on the worker thread and reports honestly: a missing engine says to install
  espeak-ng, and a silent render is named as such rather than shown as success.
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
  Both run locally through the optional `[qwen-speech]` extra, the weights are a
  one-time human-approved download under `data/qwen-speech`, and each engine is a
  seam a test injects. Arabic is handled honestly: ASR transcribes it, and TTS
  reads it through the system engine rather than the model. Surfaced as
  `python -m cli qwen-asr|qwen-tts`, the GUI sidebar's three *Qwen3 speech*
  actions, and a status note in the reading-voice picker.

Every phase keeps the four hard gates green and never lets the model widen its own
authority: the sandbox, the groundedness gate, the explicit-feedback rule, and the
regression monitor each can refuse a change.
