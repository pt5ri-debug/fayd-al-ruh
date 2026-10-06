# AGENTS.md — Fayd Al-Ruh

Evidence-aware symbolic cognitive architecture framed as a petroleum-engineering
mind. User-facing docs and text are in Arabic.

## Workflow (hard rules)

- ALWAYS work on a feature branch; never commit to `main`.
- Before finishing: `python -m pytest -q`, `python -m ruff check .`, `python -m mypy`,
  `python -m evaluation` (must print PASS, no regression).
- Push branches and open PRs with the `create_pr` tool. Merge only on the user's
  explicit approval ("نعم قم بالدمج").

## Layout

- `core/` — cognitive architecture. Key seeds:
  - `core/domain_knowledge.py` — the petroleum *frame*: 14 disciplines, concepts
    + relations, `seed_ontology`, `discipline_coverage`, `discipline_labels`.
  - `core/curriculum_knowledge.py` — the *curriculum*: 34 courses across
    foundations / geoscience / engineering science / petroleum disciplines,
    `seed_curriculum`, `curriculum_coverage`, `curriculum_coverage_score`.
    Both seeds run in `CognitiveController.__init__` and share nodes (idempotent).
- `ml/` — learning. `ml/petroleum.py` trains the Q-policy + predictive brain on
  the discipline tasks in `core/petroleum_tasks.py`. `ml/release.py` trains every
  layer (base pipeline, petroleum tasks, language model) and writes
  `ml/artifacts/release_manifest.json`. Trained artifacts in `ml/artifacts/` and
  `ml/llm/checkpoints/fayd-llm.npz` are committed on purpose for a release;
  regenerate with `python -m cli release`.
- `ml/llm/corpus.py` builds the language-model text, including sentences generated
  from every discipline and course (`curriculum_sentences`).
- `ml/llm/bpe.py` � byte-level BPE tokenizer trained in-project. `ml/llm/tokenizer.py`
  has `train_tokenizer(kind, text, vocab_size)` / `save_tokenizer` / `load_tokenizer`;
  the trainer writes `<checkpoint>.tokenizer.json` and `LocalLanguageModel` loads it
  automatically. BPE compresses the corpus ~5x vs raw bytes. `--tokenizer-kind byte`
  restores the old byte behaviour.
- `core/conversation.py` � multi-turn memory (sessions, roles, bounded window,
  `JsonStore` persistence). `core/documents.py` � citable documents + knowledge
  corpus from the seeds. `core/rag.py` � fuses lexical coverage (Arabic-normalised),
  hashed semantic, and multi-hop ontology signals into `RetrievedPassage` with
  citations. `core/assistant.py` � `GroundedAssistant` / `build_default_assistant`;
  answers with citations, extractive fallback with no model, and a groundedness gate
  that rejects ungrounded model output. CLI: `python -m ai.ask` / `python -m cli ask`.
- `core/speech.py` � speech I/O contracts (ASR/TTS), dependency-free `AudioClip`
  (PCM/WAV), transcript + silent fallbacks, `VoiceAssistant` voice loop. CLI:
  `python -m ai.voice` / `python -m cli voice`.
- `core/version.py` is the single source of the version string.
- `evaluation/` — regression gate. `evaluation/runner.py` assembles metrics,
  `benchmark.py` declares `MetricSpec`s, `baseline.json` is the reference.
  Update the baseline with `python -m evaluation --update-baseline`.
- `cli/` — portable front end. `menu.py` holds actions + `MENU`; `__main__.py`
  dispatches commands and reconfigures stdout/stderr to UTF-8 (Arabic-safe on
  legacy Windows consoles).
- `fayd-al-ruh.spec` — PyInstaller; uses `collect_submodules` for `ml`,
  `evaluation`, `ai`, `storage` (and `core` is bundled as an imported package).

## Conventions

- Concepts are Arabic labels; relations are English verbs (`is_a`, `measures`,
  `contains`, `drives`, ...). Seeded confidence is modest and revisable.
- Keep seeds declarative tuples (`CourseKnowledge` / `DisciplineKnowledge`);
  coverage helpers report what an ontology actually carries.
- Add a test per new seed or CLI action; keep the evaluation gate green.
