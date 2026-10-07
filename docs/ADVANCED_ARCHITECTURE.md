# Advanced cognitive architecture

This project now provides practical, extensible layers — not a claim to possess
consciousness or general human intelligence.

## Components

- `core/model_interface.py`: a unified contract for connecting any language or
  multimodal model. The client is passed by the application and no API keys are
  stored inside the project.
- `core/semantic_memory.py`: a deterministic local semantic index (cosine
  similarity). `hashed_embedding` can be swapped for a real embeddings provider
  later.
- `core/memory.py`: episodic memory with retrieval that fuses lexical relevance,
  semantic relevance, and importance.
- `core/goals.py`: goals and plans with explicit dependencies; the planner never
  executes its own steps.
- `core/safe_learning.py`: runs recorded simulations only; no access to a shell,
  the network, or files.
- `core/evaluation_suite.py`: measures knowledge-graph consistency, memory use,
  and forecast calibration.
- `core/ontology_reasoner.py`: transparent inference over the ontology with
  composition rules (`RelationRule`); derives relations (transitive
  `contained_in`, `measures`+`has_property`) without stuffing them into the graph.
- `core/ontology_retrieval.py`: retrieval that crosses links instead of matching
  words; walks explicit and derived relations with a hop decay per step, and
  returns the link that reached each result.
- `core/ontology_learner.py`: grows the ontology from observations with weak
  `associated_with` relations between co-bridged concepts, reinforces them by
  repetition, decays them by disuse below the seed ceiling, and credits or blames
  a link by the outcome of the recall — without creating concepts or overwriting
  an explicit relation.
- `evaluation/`: a measurable scientific evaluation package — fixed scenarios,
  metrics, version comparison, and regression gating.

## Progressive agent runtimes

Agent execution moves from the simplest and safest to the richest, without
opening external tool execution before the guards are complete:

- **Level 1 — replayable symbolic simulation** (`core/symbolic_simulation.py`):
  pure state transitions; each run yields a serialisable `SimulationTrace`, and
  `replay` can reconstruct the final state literally.
- **Level 2 — sandbox environments with explicit tasks**
  (`core/sandbox_environment.py` + `core/sandbox_tasks.py`): each task declares
  an allowed action list, a success condition, and a step budget; any action
  outside the list is refused and logged as a violation, and its steps are
  exported as a replayable trace.
- **Level 3 — external tools (locked by default)**
  (`core/progressive_environment.py` + `core/sandbox_runtime.py`): opened only
  once three guards are present together:
  1. **Permission policy** (`core/permissions.py`): an allow-list; external tools
     are refused by default.
  2. **Audit log** (`core/audit_log.py`): an append-only record of every
     allow/deny decision.
  3. **Human approval** (`core/human_approval.py`): the requester cannot approve
     their own request, and approval expires after a timeout.

Every run passes through `GovernedRuntime`, which enforces permission, then
approval (for external tools only), then logs to the audit trail, and refuses
execution if any guard is missing.

## Jarvis mode

Jarvis mode is the user-facing loop that composes the layers above; it is a
front end, not a new reasoning core. It keeps every guard in force and adds
nothing the sandbox does not already permit.

- `core/wake_word.py`: `WakeWordDetector` matches an English or Arabic wake
  phrase (longest phrase wins, so "hey jarvis" is consumed whole);
  `AudioSpeechGate` rejects silence and clips that are too short or too quiet to
  be speech. Both are pure and testable without a microphone.
- `core/command_router.py`: `NaturalLanguageRouter` maps a clear sentence to a
  `ToolPlan` (calculate, search, list, read, write) and leaves everything else to
  the grounded assistant. Every candidate tool must already be on the sandbox
  allow-list, and a write intent is only routed when the sandbox is writable, so
  a sentence can never widen the agent's authority.
- `core/speech.py`: `CommandSynthesizer` drives an installed system TTS engine
  (`espeak-ng`, `espeak`, `flite`, `say`) through a fixed argv and falls back to a
  silent clip on any failure; `WhisperRecognizer` drives `whisper` when present;
  `MicrophoneRecorder` uses `arecord`/`sox`; `Speaker` plays a clip through
  `aplay`/`afplay`/`sox`. `load_synthesizer("auto")` / `load_recognizer("auto")`
  pick the best installed engine; a named engine that is missing raises rather
  than faking audio. `JarvisMode.play` is best-effort and reports, rather than
  raises, when no player is present.
- `core/preferences.py`: `UserPreferences` persists the user's name, language,
  spoken-answer preference, and wake word with `JsonStore`; it only changes on an
  explicit `update`.
- `core/jarvis.py`: `JarvisMode` runs listen → route → answer/act → speak and
  returns a `JarvisTurn` recording what was heard, how it was routed, the answer
  or agent run, and the spoken clip.

The loop is deliberately honest: when it does not confidently understand a
sentence it answers from knowledge instead of guessing a tool call, and when a
step is refused it says so rather than inventing a result.

## Machine learning (ml/)

The project used to set belief confidence with hand-written rules only. Now a
real statistical learning layer sits on top of the symbolic substrate, using
`numpy` only:

- `ml/dataset.py`: an experience store (replay buffer), deterministic hashed
  feature encoding, train/validation split.
- `ml/models.py`: a softmax classifier, a back-propagation MLP, linear
  regression (world model), an Adam optimiser, and `.npz` save/load.
- `ml/trainer.py`: a mini-batch training loop with metrics (accuracy, Brier),
  early stopping, and best-weight restore.
- `ml/learner.py`: symbolic reinforcement learning (linear Q-learning with
  epsilon-greedy) over sandbox tasks.
- `ml/cognitive_brain.py`: a predictive-coding loop (perception -> prediction ->
  learning -> memory -> reflection) with self-calibrating learning rate.

### The progressive path toward a learning mind

1. **Level 1 — symbolic reinforcement learning**: an agent starts ignorant and
   masters the tasks through reward (100% success on both tasks).
2. **Level 2 — predictive-coding brain**: learns a world model that predicts the
   next state, reduces prediction error, stacks experiences in memory, and
   self-tunes its learning rate (learning how to learn).
3. **Level 3 — supervised classification**: a classifier learns from the
   stacked experiences and is measured on a separate validation set.

### Running

```bash
python -m ml                 # train policies + brain + classifier, print a report
python -m ml --episodes 400  # a longer learning session
python -m ml --json          # JSON summary
python -m ml --save-dir ml/artifacts
```

### Wiring learning into the agent loop

The learned policy is wired into the real loop through the
`core/action_policy.py` protocol (`ActionPolicy`) and the
`ml/policy_adapter.py` adapter (`LearnedActionPolicy`). This way `core` does not
depend on numpy, and the learning layer does not know the internal decision
details:

- `Agent` and `AgentLoop.run_once` accept an optional `policy`; when absent, the
  original behaviour stays exactly as it was.
- The policy only chooses among the actions offered by the caller, and the chosen
  action passes through the same executor and precondition checks, so it cannot
  invent new effects.

### Memory, replay, and continuous learning

The replay buffer became **active** rather than a mere log: after each episode
the brain re-learns the world model from a random batch of stored experiences
(`CognitiveBrain.replay`). Measurement on the `wet-garden` task (mean of 3 seeds)
shows that memory speeds up reaching a low error:

| Error threshold | Without replay | With replay (32) |
|---|---|---|
| 0.002 | 9.7 episodes | 6.3 episodes |
| 0.001 | 15.0 episodes | 8.0 episodes |
| 0.0005 | 18.7 episodes | 11.0 episodes |

Sessions can now be **paused and resumed**: `CognitiveBrain.save/load` stores the
world model, policy, memory, and episode counter, so training continues where it
stopped:

```bash
python -m ml --episodes 300 --save-dir ml/artifacts            # first session
python -m ml --episodes 300 --save-dir ml/artifacts --resume   # continue
python -m ml --replay-batch 0                                   # disable replay
```

### Reward from real interaction instead of a sandbox condition

The task success condition (`_success_facts`) is a **proxy** reward: it says "the
desired fact is present", not "the choice worked". A reward source built on
**observed** outcomes was added:

- `core/interaction_reward.py`: `InteractionRewardModel` learns a calibration
  table (confidence -> observed success rate) and a contextual action-value table
  (context -> action -> success rate). Two real sources: the **evaluation corpus**
  (`evaluation.scenarios`, where every prediction carries its observed outcome)
  and **human review** via `outcomes_from_reviews`.
- `ml/interaction_learner.py`: a `reset/step` environment whose reward is the
  learned action success rate, **not** the task condition. Reward is granted only
  when the action actually changes the state (direct observation needs no success
  condition), so the system learns not to repeat an action that changed nothing.

The same `QLearningAgent` trains unchanged; only the **reward source** changes:

```bash
python -m ml --episodes 400 --real-reward   # observed reward + comparison with the sandbox condition
```

The result on all six tasks (including held-out ones): success `>= 0.98`, and the
calibration measurement for the learned reward appears in the output (`ece`,
`brier`) instead of being hidden.

### Planning with a learned world model (Dyna-Q)

Before: planning in `core/planner.py` was purely symbolic (reading hand-written
transitions), and the brain's learned world model was used for prediction error
only, with decisions made without a model.

Now the learned model is **usable in decisions**:

- `ml/world_model.py`: `SymbolicWorldModel` learns each action's effect from
  **observed** transitions only (it does not read declared effects). It stores
  under the exact state and under (action, satisfied preconditions) for
  generalisation, returns `unknown` for anything never seen, and also learns that
  an action does nothing where it does nothing.
- `ml/model_based_planner.py`:
  - `plan_with_world_model`: search inside the **learned** model — solves every task.
  - `DynaQAgent`: learns from **real and imagined** transitions together (Dyna-Q).

Success is always measured by the real goal; the model only chooses and imagines,
it never decides the meaning of success:

```bash
python -m ml --episodes 40 --model-based --seeds 3
```

Comparison at a small budget (mean of 3 seeds, 40 episodes):

| Task | Without a model | Dyna-Q |
|---|---|---|
| `wet-garden` | 1.00 | 1.00 |
| `stack-blocks` | 1.00 | 0.96 |
| `plant-seed` | 0.42 | **0.71** |
| `boil-water` | 0.67 | **0.92** |
| `light-candle` | 1.00 | 1.00 |
| `recharge-lamp` | 0.83 | **0.96** |

The gain concentrates in multi-step tasks; on easy tasks the agent without a
model already saturates, so there is no difference. We show the whole picture
(including the slight regression on `stack-blocks`) rather than cherry-picking.

### Sandbox task catalogue

| Task | Role | Solution chain | Step budget |
|---|---|---|---|
| `wet-garden` | training | `water` | 4 |
| `stack-blocks` | training | `pick` -> `place` | 4 |
| `plant-seed` | training | `till` -> `plant` -> `water` -> `wait` | 6 |
| `boil-water` | training (has a decoy action `stir`) | `fill` -> `heat` -> `steep` | 6 |
| `light-candle` | held-out | `strike` -> `light` | 4 |
| `recharge-lamp` | held-out | `open` -> `swap` -> `switch` | 6 |

The new training tasks are longer (4 steps) and include decoy actions, forcing
the policy to learn over a longer horizon instead of solving a single step. The
extra held-out tasks (`recharge-lamp`) test that the same machinery works without
special tuning.

### Generalising to new tasks

Two tasks were added to `core/sandbox_tasks.py` as `HELD_OUT_TASKS`
(`light-candle` and `recharge-lamp`): the agent never trains on them in the base
training path, then we verify that the same learning machinery solves them with
no task-specific tuning. The integration test now runs on all held-out tasks,
not just the first.

### The in-project language model

The project carries its own language model in `ml/llm/`, written from scratch in
`numpy` and trained inside the repository, so it needs no external service and no
API keys:

- `tokenizer.py`: byte-level encoding (257 tokens) that is safe for any script
  with no training, and every text round-trips exactly.
- `transformer.py`: a decoder-only transformer (multi-head causal attention +
  GELU + LayerNorm) with manual backprop and Adam. We verified the gradients
  against a numerical estimate.
- `corpus.py`: in-repo training text (English about the project's world and about
  petroleum engineering, plus the seeded petroleum corpus and the curriculum, with
  JSON examples in the `MODEL_OUTPUT_SCHEMA` format).
- `generator.py` + `ai/local_model.py`: reproducible seeded text generation, and
  adaptation to the `CognitiveModel` interface, so it replaces Ollama anywhere a
  model is accepted.
- `schema.py`: schema-guided decoding. It fixes the structure from the schema and
  lets the model suggest only leaf content, so the output is always valid JSON.

```bash
python -m ml.llm --steps 4000 --generate "water"
```

The checkpoint is saved at `ml/llm/checkpoints/fayd-llm.npz` and committed in the
repository, so any user gets a working language model immediately without
training. Usage:

```python
from ai.local_model import LocalLanguageModel
model = LocalLanguageModel(temperature=0.0)
response = controller.reason(model, request)
```

### Wiring the in-project language model into the decision path

When the request carries a `response_schema` (the structured-suggestion path),
`LocalLanguageModel` directs generation to `StructuredDecoder`: the JSON
structure is fixed from `MODEL_OUTPUT_SCHEMA`, and the model suggests leaf
content (concepts, evidence, hypotheses, questions, plans). The output passes
through `controller.analyze_with_model`, so the same `LLMReasoner` and
`parse_structured_output` that validate any external model validate it too:

```python
analysis = controller.analyze_with_model(model, request)
```

The model only suggests: it does not modify beliefs or their state, does not run
tools, and does not mark a step complete. Any forbidden key (`beliefs`,
`tool_calls`, ...) is rejected.

> Explicit disclaimer: a real transformer but a small one (3 layers, width 96)
> trained on a small corpus. It produces coherent text about the project's
> world, not a general model at frontier-model scale.

### Domain identity: a petroleum-engineering mind

The project explicitly defines itself as a **petroleum-engineering mind**
(`core/self_model.py`), with its initial domain knowledge in
`core/domain_knowledge.py`:

- `SelfModel`: the name, identity (`petroleum engineering mind`), domain,
  capabilities, limits, and areas of expertise (reservoirs, drilling, production,
  petrophysics, logs...).
- `seed_ontology`: seeds 13 petroleum concepts and 13 relations (such as
  `reservoir contained_in formation`, `pressure drives production`) with modest,
  revisable confidence, **idempotently**, so it does not duplicate on restart or
  restore.
- `CognitiveController`: seeds the ontology at startup, adds `domain_summary`
  (the identity frame and expertise areas) to every model request's context, and
  exposes the identity via `controller.identity`.

```python
controller = CognitiveController()
controller.identity                                  # the identity frame
controller.ontology.find_concepts("permeability")    # permeability
```

> Explicit disclaimer: the identity is a **domain frame**, not a claim to
> consciousness. The seed is small and auditable, it never changes a belief
> automatically, and it does not replace field data or the judgement of a
> certified engineer. Every seed relation is revisable with evidence like any
> other knowledge.

### The knowledge content: the petroleum corpus

The frame and the curriculum declare *which* concepts exist; the corpus in
`core/petroleum_corpus.py` holds what they *mean*. Each `CorpusEntry` carries a
definition plus concepts, relations, formulas, procedures, standards, and
bilingual glossary terms, organised by discipline. It currently holds **48
entries across 18 disciplines**, spanning the core petroleum disciplines and an
advanced block of field-engineering frontiers: petroleum systems and seismic
interpretation, gas processing and LNG, process safety management, unconventional
resources, reservoir simulation, advanced EOR and waterflood management,
integrated field development, geomechanics, flow assurance, multiphase flow,
subsea and deepwater production, data analytics, carbon capture and storage, and
well decommissioning. It is seeded into every knowledge consumer so the content
is inseparable from the project:

- `seed_petroleum_corpus(ontology)`: plants the concepts and relations
  **idempotently**, reusing nodes the frame and the curriculum already created,
  so the three seed layers form one graph.
- `corpus_documents()`: turns each entry into a `Document`, so the corpus is part
  of the retrieval store (`core/documents.py`) and any claim drawn from it is
  citable as `[petroleum_corpus:corpus:<key>]`.
- `corpus_sentences()`: generates training prose merged into the language-model
  corpus (`ml/llm/corpus.py`), so the model learns the same content.
- `corpus_coverage()` / `corpus_coverage_score()`: per-discipline and whole-corpus
  coverage, recorded in the release manifest (`ml/release.py`) as
  `knowledge.corpus_coverage`, so losing any part of the content is a regression.

```python
from core.petroleum_corpus import corpus_coverage, corpus_coverage_score, seed_petroleum_corpus

seed_petroleum_corpus(controller.ontology)          # idempotent
corpus_coverage_score(controller.ontology)          # 1.0
```

### Wiring a real language model to actions

`core/model_advisor.py` (`ModelActionAdvisor`) wraps any `CallableModel` as a
policy: it sends the current facts and the allowed action list only, and accepts
the model's reply only if it names an action from the list; otherwise it returns
`None` so the decision stays with the caller. A model failure never breaks the
loop.

> Explicit disclaimer: this gives the project a **measurable learning
> capability** (improved performance, reduced prediction error, self-calibration),
> not human consciousness. True self-awareness is not a goal achievable by a
> software construction, and the approach here is inspired by known principles of
> biological learning (predictive coding, memory, metacognition).

## Connecting a real model

Create a `CallableModel` that wraps the model client of your choice, then pass it
to `CognitiveController.reason`. The wrapper must return text only; tools and
external actions stay under the control of the application and the user.

```python
model = CallableModel("my-model", lambda request: client.generate(request))
response = controller.reason(model, request)
```

## Answering from the web under governance

The system can answer a question it does not know by searching the internet, but
never blindly: search is an external tool, so it passes the same permission,
approval, and audit gates as every other governed action, and a web answer is
folded into the corpus only when it is trusted.

- `core/web_source.py`:
  - `WebResult` is one source (URL, title, snippet, credibility).
  - `WebSearchProvider` is the contract; `WikipediaSearchProvider` is keyless and
    reaches the real internet by default, `StaticSearchProvider` is deterministic
    and offline (tests, no-network fallback), and `HttpSearchProvider` is a JSON
    client for a caller-supplied endpoint (the application owns the URL and key).
    `FallbackSearchProvider` tries the internet first and drops back to the
    offline corpus on failure or an empty result.
  - `WebSearchService` consults a `PermissionPolicy` and a `HumanApprovalGate`
    before it runs, and records every decision in the `AuditLog`; a provider
    failure is contained and never crashes the caller.
  - `estimate_credibility` is a transparent heuristic over the domain suffix
    (government, standards bodies, and academic hosts rank above generic blogs).
- `core/web_verifier.py`:
  - `WebVerifier.verify` retrieves the project's own passages, fetches web
    sources, and rates the answer with a transparent trust score that blends
    **source credibility**, **source-to-source agreement**, and **internal
    support** (how much of the web content the project already knows).
  - Every source keeps its own support score and matched terms, and a source that
    negates a known term is flagged as contradictory.
  - The answer is learned into the citable corpus via `KnowledgeUpdater` (with
    provenance and decay) only when its trust clears the threshold, it does not
    contradict internal knowledge, and it connects to something already known.

```bash
python -m cli web --approve "What is the relationship between porosity and permeability?"
python -m cli web --approve            # prompts: type your own question
python -m cli web "What is a gamma ray log?"          # denied by default
python -m ai.web_verify --json --approve "What is a gas kick?"
```

`FAYD_SEARCH_ENDPOINT` (and optionally `FAYD_SEARCH_API_KEY`) point the command
at a real search API; with none set it uses the keyless Wikipedia API and drops
back to a small offline corpus when there is no network, so the whole path is
testable on a fresh clone.

## Measurable scientific evaluation

The `evaluation/` package runs fixed scenarios across six domains, measures
system performance, and compares versions to prevent regression:

- **Scenarios** (`evaluation/scenarios.py`): fixed cases for memory, reasoning,
  causality, planning, and uncertainty (including contradiction detection and
  decision explanation), plus the **petroleum** domain (`Domain.PETROLEUM`) with
  ten scenarios: causality, petrophysics/retrieval, well-completion /
  production-decline planning, the "is production declining?" uncertainty, a
  check of the seeded ontology relations, inference over the ontology (derived
  relations not stored), multi-hop retrieval that crosses derived links with no
  shared words, ontology learning from a field observation (new weak relations
  while preserving the seeded pair), link weighting by experience (reinforcement
  by repetition and decay by disuse within a ceiling), and learning from a
  recall's outcome (credit for a useful link, blame otherwise).
- **Metrics** (`evaluation/metrics.py`): prediction accuracy, Brier score with
  the Murphy decomposition, calibration error (ECE), and contradiction
  consistency.
- **Runner** (`evaluation/runner.py`): executes the scenarios through
  `CognitiveController` and assembles: prediction accuracy, calibration,
  contradiction rate, retrieval quality (accuracy and MRR), planning success,
  decision explanation, causal accuracy/quality, and the eight petroleum metrics
  (`petroleum_causal_accuracy`, `petroleum_retrieval_accuracy`,
  `petroleum_ontology_accuracy`, `petroleum_inference_accuracy`,
  `petroleum_multihop_retrieval_accuracy`,
  `petroleum_ontology_learning_accuracy`,
  `petroleum_ontology_reinforcement_accuracy`,
  `petroleum_ontology_feedback_accuracy`).
- **Regression gate** (`evaluation/benchmark.py` + `baseline.json`): compares the
  current version's metrics against the stored baseline and refuses any
  regression (with an optional `--tolerance` margin).

```bash
python -m evaluation                    # run + compare against baseline (fails on regression)
python -m evaluation --json             # full JSON report
python -m evaluation --update-baseline  # update the baseline after an intended improvement
python -m evaluation --tolerance 0.01   # allow a noise margin
```

The gate runs automatically in CI after the tests, so no change that weakens
performance can be merged without an intended baseline update.

## Verification

Run `python main.py` for a safe local cycle. With the project requirements
installed, run `python -m pytest -q`.
