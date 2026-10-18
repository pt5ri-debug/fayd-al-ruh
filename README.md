# Fayd Al-Ruh

Fayd Al-Ruh is an **evidence-aware symbolic cognitive architecture**. It turns
the knowledge graph from a store of entities and relations into an explainable,
evolving cognitive model: every piece of knowledge carries its evidence, its
confidence, its source, its history, and its status (provisional or stable).

The project defines itself as a **petroleum-engineering mind**
(`core/self_model.py`): it reasons across **all petroleum-engineering
disciplines** — reservoirs, drilling, production, petrophysics, well logging,
well completion, enhanced recovery, well testing, artificial lift, well
stimulation, well cementing, well control, petroleum economics, health, safety
and environment, facilities engineering, process safety, and unconventional
resources — from evidence, not from guesswork. The identity is a
domain frame, not a claim to consciousness. The initial domain knowledge is an
auditable seed in `core/domain_knowledge.py`, organised by discipline, and is
seeded into the project ontology at startup. On top of it sits a **full
curriculum** in `core/curriculum_knowledge.py` that gathers the seed knowledge a
petroleum-engineering student receives during their studies, course by course.

> The project does not claim consciousness or general human intelligence. It is
> a set of practical, extensible layers, and every inference in it is
> explainable and reviewable.

## Core principles

For every belief in the system:

- What do I know? — `core/belief.py`
- Why do I believe it? — `core/evidence.py`, `core/evidence_processor.py`
- When did I learn it and from where? — `core/cognitive_context.py`
- How confident am I? — `core/confidence_model.py`, `core/uncertainty.py`
- Is there evidence against it? — `core/contradiction_detector.py`, `core/belief_conflict.py`
- Is the knowledge provisional or stable? — `core/belief.py` (`BeliefStatus`)

## Domain identity: a petroleum-engineering mind

The project defines itself in `core/self_model.py`, and the initial domain
knowledge lives in `core/domain_knowledge.py`:

```python
from core.cognitive_controller import CognitiveController

controller = CognitiveController()
controller.identity          # the identity frame passed with every model request
controller.ontology.find_concepts("reservoir")   # reservoir, formation, porosity, permeability...
```

- `SelfModel`: the name, identity, domain, capabilities, limits, and areas of
  expertise.
- `seed_ontology`: seeds the petroleum concepts and relations **by discipline**
  (such as `reservoir contained_in formation`, `water_drive maintains
  reservoir_pressure`) with modest, revisable confidence, **idempotently**.
- `discipline_coverage`: reports how much of each discipline (concepts and
  relations) is present in an existing ontology, so the breadth of the frame is
  measured instead of merely claimed.
- `CognitiveController`: seeds the ontology at startup, adds the identity frame
  to every model request's context, and exposes it via `controller.identity`.

This is a domain frame only: it does not grant the system consciousness, does not
change beliefs automatically, and every seed relation is correctable with
evidence like any other knowledge.

## Measuring the identity: a petroleum evaluation suite

The identity is not an unverifiable claim; the `evaluation/` package measures it.
The `Domain.PETROLEUM` domain was added with fixed scenarios:

- **Causality** (`petroleum-causal`): the chain `pressure -> production ->
  revenue` is discovered and explained.
- **Reasoning/petrophysics** (`petroleum-petrophysics`): retrieves well-log
  readings and links them to the seeded ontology concepts (`well_log measures
  porosity/permeability`).
- **Planning** (`petroleum-planning`): the sequence "complete the well" (casing
  then cement) and the sequence "treat the production decline" (water injection
  then stimulation).
- **Uncertainty** (`petroleum-uncertainty`): "is production declining?" is
  revised from field evidence and stays calibrated.
- **Identity** (`petroleum-identity`): verifies that the seeded ontology carries
  the declared domain frame (such as `reservoir contained_in formation`, `cement
  seals well`).
- **Inference over the ontology** (`petroleum-ontology-inference`): verifies that
  relations are derived and not stored, through transparent composition rules
  (`core/ontology_reasoner.py`): transitive `contained_in`, `well penetrates
  formation`, `reservoir has_property porosity`, and `well_log measures
  reservoir` (composition of `measures` with `has_property`).
- **Multi-hop retrieval** (`petroleum-multihop-retrieval`): a query about a log
  reaches a reservoir observation, and a query about drilling reaches a well
  observation, despite **sharing no word** between query and observation — by
  walking the derived links (`core/ontology_retrieval.py`).
- **Ontology learning** (`petroleum-ontology-learning`): a field observation
  bridges three known concepts, so weak `associated_with` relations are recorded
  between them, while the seeded pair stays explicit without duplication or
  overwriting (`core/ontology_learner.py`).
- **Link weighting by experience** (`petroleum-ontology-reinforcement`): a
  repeated observation strengthens its learned links step by step up to a ceiling
  below the seed, and a neglected observation decays its links until they vanish
  — without touching the seeded relations.
- **Learning from a recall's outcome** (`petroleum-ontology-feedback`): a useful
  recall strengthens the learned link that produced it, and an unhelpful recall
  weakens it until it vanishes — so the graph is shaped by the outcomes of its
  actions, not only by its observations.

Nine petroleum metrics are measured in the baseline: `petroleum_causal_accuracy`,
`petroleum_retrieval_accuracy`, `petroleum_ontology_accuracy`,
`petroleum_inference_accuracy`, `petroleum_multihop_retrieval_accuracy`,
`petroleum_ontology_learning_accuracy`,
`petroleum_ontology_reinforcement_accuracy`,
`petroleum_ontology_feedback_accuracy`, and `petroleum_learning_accuracy`, plus
`domain_scores["petroleum"]` and `curriculum_coverage` (full curriculum coverage).
The identity thus becomes measurable and protected against regression in CI.

## The curriculum: a complete petroleum-engineering college

On top of the domain frame, `core/curriculum_knowledge.py` provides the
**curriculum** — the seed knowledge a petroleum-engineering student receives
during their studies, organised by the course that teaches it. It is not a
textbook, but a compact, evidence-aware map of the programme: each course
contributes its concepts and relations with modest, revisable confidence.

The curriculum covers **48 courses** in four blocks:

- **Foundations** (6 courses): mathematics, physics, chemistry, statistics and
  data analysis, computing and programming, numerical methods.
- **Earth sciences** (7 courses): general geology, structural geology,
  sedimentology, petroleum geology, geophysics, basin analysis and petroleum
  systems, seismic interpretation.
- **Engineering science** (6 courses): fluid mechanics, thermodynamics, mechanics
  of materials, rock mechanics, petroleum geomechanics, multiphase flow.
- **Petroleum disciplines** (29 courses): petrophysics, reservoir fluid
  properties, reservoir engineering, drilling engineering, production
  engineering, well logging, well completion, well cementing, well stimulation,
  artificial lift, well testing, enhanced recovery, well control, facilities,
  transport and storage, offshore drilling, petroleum economics, project
  management, health, safety and environment, ethics and professionalism, plus
  the advanced block: reservoir simulation and modelling, advanced enhanced oil
  recovery, unconventional resources engineering, subsea and deepwater
  engineering, flow assurance, process safety engineering, petroleum data
  analytics and machine learning, carbon capture and storage, well plugging and
  decommissioning, and integrated field management.

```python
from core.cognitive_controller import CognitiveController
from core.curriculum_knowledge import curriculum_coverage, course_labels

controller = CognitiveController()
coverage = curriculum_coverage(controller.ontology)
labels = course_labels()
for key, counts in coverage.items():
    print(labels[key], counts["concepts"], counts["relations"])
```

- `seed_curriculum`: seeds the curriculum courses, concepts, and relations,
  **idempotently**, and shares nodes with the domain seed (it does not duplicate
  `reservoir` or `porosity`).
- `curriculum_coverage`: reports the coverage of each course (concepts and
  relations) in an existing ontology.
- `curriculum_coverage_score`: a single number over the whole curriculum, merged
  into the evaluation package as the `curriculum_coverage` metric, so no
  regression in any part of the curriculum goes unseen.

To display it from the command line:

```bash
python -m cli curriculum    # show courses and their coverage by block
```

The curriculum is seeded automatically at `CognitiveController` startup, so it
becomes an **inseparable part** of the project's knowledge: any reasoning or
retrieval works on top of it with no extra setup.

## The corpus: the knowledge content behind the curriculum

The frame and the curriculum say *which* concepts exist and how they relate. The
**corpus** in `core/petroleum_corpus.py` says what they *mean*: definitions,
formulas, procedures, standards, and bilingual glossary terms, organised by
discipline. It is the content layer that makes the project's petroleum identity
concrete, and it is seeded into all three knowledge consumers:

- the **ontology**, via `seed_petroleum_corpus` (idempotent, and it shares nodes
  with the frame and the curriculum, so all three form one graph);
- the **citable documents**, via `corpus_documents`, so a retrieved answer can
  quote the corpus and point back to it;
- the **language-model training text**, via `corpus_sentences`, so the model is
  trained on the same knowledge the graph carries.

The corpus currently covers **48 entries across all 18 disciplines**, including
reservoir rock properties and drive mechanisms, volumetric reserves and material
balance, formation evaluation and saturation-height, the logging suite and its
interpretation workflow, drilling fluids and casing design, directional drilling,
production optimisation and artificial lift, completions, hydraulic fracturing
and matrix acidizing, primary cementing, well control, enhanced oil recovery,
pressure-transient testing, project economics and field development planning,
well integrity, process safety, offshore operations, surface facilities, project
management, professional ethics, reservoir fluid properties, rock mechanics, and
flow in porous media. A second, advanced block adds the frontiers of field
engineering: the petroleum system and seismic interpretation, surface facilities
and gas processing/LNG, process safety management, unconventional resources,
reservoir simulation and modelling, advanced EOR, waterflood management,
integrated field development planning, geomechanics and wellbore stability, flow
assurance, multiphase flow, subsea and deepwater production, data analytics and
machine learning, carbon capture and storage, and well plugging and
decommissioning.

```python
from core.cognitive_controller import CognitiveController
from core.petroleum_corpus import corpus_coverage, corpus_coverage_score

controller = CognitiveController()
coverage = corpus_coverage(controller.ontology)
print(corpus_coverage_score(controller.ontology))  # 1.0
```

- `seed_petroleum_corpus`: plants the corpus concepts and relations
  **idempotently**, reusing nodes the frame and the curriculum already created.
- `corpus_documents`: turns each entry into a `Document`, so the corpus is part
  of the retrieval corpus and every claim drawn from it is citable.
- `corpus_sentences`: generates the training prose merged into the
  language-model corpus (`ml/llm/corpus.py`).
- `corpus_coverage` / `corpus_coverage_score`: per-discipline and whole-corpus
  coverage, recorded in the release manifest (`ml/release.py`) so losing any part
  of the content shows up as a regression.

To display it from the command line (the curriculum view now lists corpus
coverage too):

```bash
python -m cli curriculum    # show courses, then corpus coverage by discipline
```

## Learning petroleum-engineering tasks

Identity alone is not enough to measure: the system must also *learn* the tasks
of each discipline. `core/petroleum_tasks.py` provides a catalogue of engineering
tasks — one or more per discipline — and each task is a short, explicit action
chain ending in a real accomplishment: reserve estimation, drilling a well,
restoring production, core analysis, log interpretation, completing a well,
enhanced recovery, pressure testing, installing gas lift, matrix acidizing,
primary cementing, controlling a gas kick, computing net present value,
verifying well integrity, gas processing, a process-safety review, and
completing a shale well.

`ml/petroleum.py` trains — on these tasks — the same two layers used in
`python -m ml`: a symbolic Q-learning policy and the predictive-coding brain. For
each discipline it reports its success rate, with a generalisation section on two
disciplines **never trained on** (`geoscience`, `flow_assurance`) to verify that
the same machinery works on what it has not seen.

```bash
python -m cli petroleum --episodes 120    # train every petroleum-engineering discipline
python -m cli petroleum --json            # machine-readable summary
```

Beyond the hand-written catalog, `ml/procedural_tasks.py` derives a task from
**every procedure recorded in the corpus** — one ordered action chain per entry,
ending in the recorded outcome. These are opt-in so the default run stays fast:

```bash
python -m cli petroleum --episodes 120 --procedural
```

This also runs from the operator menu (the "Train petroleum-engineering tasks"
option) and is measured via `petroleum_learning_accuracy` in the evaluation
package.

## Release training: storing every training artifact inside the project

Before releasing, a single command gathers **every learning layer** and pins its
outputs inside the project:

```bash
python -m cli release                       # train everything and write the manifest
python -m cli release --episodes 120 --llm-steps 4000
python -m ml.release --manifest-only        # rewrite the manifest without retraining
```

`ml/release.py` runs, in order:

1. **The base cognitive pipeline** (`python -m ml`): policies, predictive brain,
   experience classifier.
2. **Petroleum-engineering tasks** (`ml/petroleum.py`): every discipline +
   generalisation, plus the **procedural tasks** derived from the corpus
   (`ml/procedural_tasks.py`) when enabled.
3. **The in-project language model** (`ml/llm`): trained on a corpus that
   includes the whole curriculum **and every repository's knowledge** — the
   authored documents and (by default) the runtime documents, plus a
   relevance-filtered sample of the bulk knowledge base.

Because the full corpus is large, release training uses the **byte** tokenizer by
default (encoding is linear); BPE gives better compression but is slow to train on
a multi-megabyte corpus, so it is opt-in via `--tokenizer-kind bpe`.

```bash
python -m ml.release --kb-max-chunks 2000          # smaller knowledge-base sample
python -m ml.release --no-procedural               # skip corpus-derived procedures
python -m ml.release --no-runtime-docs             # authored documents only
```

It then writes a **manifest** at `ml/artifacts/release_manifest.json` recording:

- The project version (`core/version.py`), the training time, and the git revision.
- **The knowledge** trained on: number of disciplines, courses, concepts, and
  relations, the curriculum coverage (1.0), and the authored documents per
  repository.
- **The corpus**: number of curriculum sentences, repository and knowledge-base
  sentences, built examples, and characters.
- **The configuration**: episodes, language steps, seed, replay size, the
  language-model capacity preset, and the knowledge-base sample and procedural
  options.
- **The results**: petroleum task success (and procedural coverage), the
  language-model parameter count and capacity tier, and — with
  `--measure-perplexity` — its **held-out perplexity**.

All outputs are stored inside the project: `ml/artifacts/` (policies, brains,
classifier, and `petroleum/`) and `ml/llm/checkpoints/fayd-llm.npz`. These files
are committed to git on purpose for a release, so the repository carries a
ready-trained state that needs no retraining on the user's machine. Training is
deterministic for a fixed seed, so the manifest can be reproduced and verified.

## Inference over the ontology

The seeded ontology is not static data. `core/ontology_reasoner.py` derives
relations that follow from stored relations through transparent composition rules
(`RelationRule`):

- `containment_transitive`: transitivity of `contained_in`.
- `penetrates_through_containment`: `well penetrates reservoir` + `reservoir
  contained_in formation` => `well penetrates formation`.
- `containment_inherits_property`: `reservoir contained_in formation` +
  `formation has_property porosity` => `reservoir has_property porosity`.
- `measure_property_of_host`: `well_log measures porosity` + `formation
  has_property porosity` => `well_log measures formation`, then `well_log
  measures reservoir`.

Derived relations are **not stuffed into the ontology**; they are returned as
`OntologyInference` with the composition steps (`steps`) and a multiplied
confidence, so they stay distinct from explicit, reviewable relations. They are
available via `controller.infer_ontology_relation(...)` and
`controller.derive_ontology_relations()`.

## Ontology-connected retrieval

Ordinary retrieval (words + semantic fingerprint) only reaches items that share
words with the query. `core/ontology_retrieval.py` makes retrieval cross links
instead: it bridges each item to the ontology concepts, then walks the explicit
**and derived** relations in their declared direction for a bounded distance,
reaching an item that mentions only a related concept.

- `OntologyGuidedRetriever.concepts_in(text)`: the concepts whose terms appear in
  the text.
- `reachable(seeds)`: bounded breadth-first walk (`max_hops`) with a geometric
  decay (`hop_decay`) per step, so a direct match precedes a one-hop match.
- `rank(items, query, base_scores)`: ranks items by the best of their base score
  and their relational score, so ordinary retrieval loses nothing.
- `controller.recall_with_ontology(query)`: fuses ordinary retrieval with this
  ranking.

Example: a query about "well_log" reaches an observation about the reservoir via
the derived `well_log measures reservoir`, despite sharing no word. This is what
the `petroleum-multihop-retrieval` scenario measures and the
`petroleum_multihop_retrieval_accuracy` metric protects.

## Ontology learning from experience

The ontology does not stay confined to its seed. `core/ontology_learner.py` makes
it expand from observations: when known concepts are bridged in a single text, a
weak `associated_with` relation is recorded between them, so the acquired meaning
becomes retrievable later.

- `OntologyLearner.observe(text)`: records a weak relation between every pair of
  known concepts present in the text, and reinforces any learned relation it
  repeats.
- `learn_from_observations(texts)`: merges several observations into one report.
- `controller.learn_ontology_from_observation(text)`: teaches the ontology and
  then invalidates the retrieval cache, so subsequent retrieval operations see
  the new links.

Learning preserves cognitive honesty with three constraints: it does not create
concepts from raw text (it only links known ones), its confidence is lower than
the seeded relations (so it never overrides organised knowledge), and it never
touches a pair that already has an explicit relation (so it neither duplicates
nor overwrites what is known). It stays distinct from inference: a learned
relation is **stored**, while a derived one stays computed and not stored. This
is what `petroleum-ontology-learning` measures and the
`petroleum_ontology_learning_accuracy` metric protects.

## Link weighting by experience

Experience does not only add links, it weights them. `core/ontology_learner.py`
makes each learned link's confidence follow usage:

- **Repetition reinforces**: each observation that bridges the same pair raises
  its confidence step by step (`reinforcement`), so what repeats in experience
  gains confidence.
- **The ceiling protects the seed**: learned confidence never exceeds `ceiling`
  (below the seeded relations), so it never competes with organised knowledge.
- **Disuse decays**: `decay()` weakens every learned link step by step, and what
  falls to zero is deleted, so no transient associations remain.
- **The seed is fixed**: decay touches only `associated_with`, and the seeded
  relations stay as they are.

Available via `controller.reinforce_ontology_relation(...)` and
`controller.decay_ontology_learning(...)`. This is what
`petroleum-ontology-reinforcement` measures and the
`petroleum_ontology_reinforcement_accuracy` metric protects.

## Learning from a recall's outcome

The graph does not learn only from observations, but from the **outcomes of its
actions**. `recall_with_ontology` returns, with each result, the learned link
that reached it in `linked_relation`, so the link that produced the recall can be
credited or blamed:

- **Useful recall**: `feedback_ontology_recall(recall, useful=True)` strengthens
  the learned link that reached it, so what proves useful gains weight.
- **Unhelpful recall**: `useful=False` weakens the link, and if it vanishes it is
  deleted, so the recall path that did not help disappears.
- **No blame for the seed**: direct recalls or those crossing a seeded relation
  (`linked_relation` with no learned link) are neither strengthened nor weakened.

This is how the graph is shaped by the outcomes of retrieval, not only by its
structure. This is what `petroleum-ontology-feedback` measures and the
`petroleum_ontology_feedback_accuracy` metric protects.

## Learning the whole knowledge graph

Identity and tasks are not the whole picture: the system also learns from its
**knowledge graph** — the concepts and relations seeded across the disciplines,
the curriculum, and the corpus. `ml/general_learning.py` treats this as **link
prediction**: it holds out a fifth of the edges, then asks the model to rank each
true target against decoys, using concept vocabulary alone.

- **Honest baseline**: the reported `hit_at_1` is measured against a random
  baseline (one right answer among four candidates = 0.25), with mean rank and a
  `generalisation_gain`, so the number is never a training-set artefact.
- **No hand-coded answers**: the model never sees the held-out edge; it must
  generalise from how concepts are used elsewhere.

It runs as part of `python -m ml` and reports one line:

```text
General knowledge (link prediction): relations=467 concepts=460 types=177
held-out Hit@1=0.41 baseline=0.25 gain=+0.16 mean-rank=2.02
```

## Layout

```
core/        the cognitive core: reasoning, causality, memory, learning, planning, explanation
             (self_model.py and domain_knowledge.py: the "petroleum-engineering mind" identity and its domain seed)
storage/     storage adapters (JSON, SQLite, Graph)
ai/          the in-project language model (transformer) and an optional Ollama client
tests/       contract, integration, and full-cycle tests
main.py      a safe local demo of a full cognitive cycle
```

## The grounded assistant (RAG + conversation memory)

The assistant answers a question in English, retrieves the passages that support
the answer, **and cites its sources**. It combines the project's existing
components instead of duplicating them:

- `core/conversation.py` — multi-turn conversation memory: sessions, roles, a
  bounded context window, and persistence in `storage.json_store.JsonStore`.
- `core/documents.py` — citable documents, a document store, and conversion of the
  discipline and course seed into quotable text.
- `core/rag.py` — fuses three signals into one ranking: query-term coverage (with
  light normalisation), hashed semantic similarity, and multi-hop ontological
  proximity.
- `core/assistant.py` — `GroundedAssistant`: retrieves, then answers, then
  attaches citations; if no language model is available it answers
  **extractively** from the passages.
- `core/composition.py` — the deterministic **composer**: a rule-based
  verbaliser that turns structured evidence (retrieved passages, web sources,
  causal/reasoning/decision traces) into clean, attributed English prose. It is
  the dry, auditable counterpart to free generation: every word is placed by a
  rule and every claim keeps its citation. The assistant's extractive fallback,
  the web verifier's answer, and the causal/reasoning/decision explainers all
  compose through it, so the project speaks with one voice in pure Python.

Two deliberate guarantees: the language model receives only a rendered prompt (it
never touches the knowledge graph or runs tools), and if no model is available
the assistant stays useful with no external service. If the model's answer looks
ungrounded in the passages, it is rejected and replaced by the composed,
extractive answer.

```bash
python -m cli jarvis "What is the relationship between porosity and permeability?"   # cited answer
python -m cli jarvis --no-web --session demo                                        # multi-turn, internal knowledge only
```

## The subword tokenizer (BPE)

The in-project language model used to treat every UTF-8 byte as a separate token,
so a common word cost several tokens. We added a byte-level BPE tokenizer trained
inside the project (`ml/llm/bpe.py`): it merges frequent byte pairs, so common
words and English terms and punctuation become single tokens. On the current
corpus it compresses the text by about **3x**, i.e. an effective context three
times longer. The tokenizer is optional: `--tokenizer-kind byte` restores the old
behaviour.

BPE merge learning scans the corpus, so it reads a bounded prefix (300k
characters) rather than the full repeated text; encoding the whole corpus with
those merges is linear. On a release-sized corpus the byte tokenizer is still the
default because training stays faster; measure the trade-off honestly with
`python -m ml.llm.evaluate`, which reports **bits per byte** so a byte model and a
BPE model can be compared on equal terms.

```bash
python -m ml.llm --tokenizer-kind bpe --vocab-size 1024 --steps 4000
```

## Voice and conversation

Voice is a front end for the assistant, not a new reasoning layer. The contracts
are in `core/speech.py`:

- `SpeechRecognizer` (ASR): turns an audio clip into text.
- `SpeechSynthesizer` (TTS): turns text into audio.
- `AudioClip`: a simple PCM store with no dependencies, so the whole path can be
  tested with no audio hardware and no external package.

Engines are loaded by name on demand. Real engines are ordinary system programs,
so no pip package and no network access is needed: TTS uses `espeak-ng`,
`espeak`, `flite`, or macOS `say`; ASR uses `whisper` when it is installed. On
**Windows** the project uses the speech engine the OS already ships (SAPI, reached
through `System.Speech` in PowerShell), so read-aloud works with nothing to
install; Windows also plays the rendered WAV through the built-in `SoundPlayer`.
If none is present, the project can still run a complete *text loop*: the
recognizer returns supplied text, and the synthesizer produces a real silent WAV
clip. This
way the whole voice path is tested on a fresh clone, and a real ASR/TTS engine
can be plugged in later without touching the core. Use `--tts auto` / `--asr auto`
to pick the best installed engine, or name one explicitly (`--tts espeak`); an
explicitly named engine that is missing raises rather than faking audio.

The TTS engine itself is an ordinary *system* program, not a Python package, so
`pip` cannot install it — exactly like Tesseract for OCR. On a machine with no
engine the window's **Read aloud** control and the CLI report that honestly and
offer the next step. Install it with the platform package manager:

```bash
sudo apt install espeak-ng                          # Debian/Ubuntu (also Fedora/Arch/Alpine)
winget install -e --id eSpeak-NG.eSpeak-NG          # Windows (or: choco install espeak-ng)
```

On Windows this is usually unnecessary: the built-in SAPI engine is used when no
command-line engine is on `PATH`, so **Read aloud** just works. The project can
also run the install for you: `python -m cli speak --install-engine` runs the
platform's package manager, and `python -m cli deps` reports the speech engine
beside the other optional features. macOS already has `say`; the project also
accepts `espeak`, `flite`, and `say`.

```bash
python -m cli voice "What is the relationship between porosity and permeability?"  # a voice turn
python -m ai.voice --write-wav out.wav "permeability"                              # and write a WAV
python -m ai.voice --tts auto "permeability"                                       # speak with a real engine
python -m ai.speak list                                                            # the reading-voice catalog
python -m ai.speak "Hello there" --voice girl-2 --write-wav out.wav                # read aloud in a chosen voice
python -m ai.speak --install-engine                                                # install a real speech engine
```

## Tools and agents

The core reasons but does not *do* anything outside memory. `core/tools.py`
gives it real tools — read a file, write a file, list a folder, search text,
calculate — and a safety sandbox that **every** call passes through, under four
rules:

1. **Explicit allow-list**: no tool except what the policy names.
2. **Root confinement**: every path resolves under a single root; any path that
   leaves it (`..`, an absolute path elsewhere, an external symlink) is refused.
3. **Read/write separation**: a read-only policy refuses every mutating tool.
4. **Capability and approval gates**: `PermissionPolicy` decides whether to grant
   a capability, and external tools need live human approval
   (`HumanApprovalGate`).

Tools return a `ToolResult` and never raise to the caller, so the agent can
continue and log what happened. `core/tool_agent.py` adds coordination:
`ToolPlan` (a sequence of calls) is executed by `ToolAgent` through the sandbox,
stops at the first failure, and records a full auditable trace. The plan is
declared by the caller; if the model proposes it, it cannot widen its own
authority, because the sandbox alone decides what runs.

```bash
python -m ai.agent_demo          # a full plan inside a temporary sandbox
python -m cli agent
```

## Jarvis mode

Jarvis mode is the layer the previous ones were building toward: one loop that
*listens, understands, acts, and speaks*. It adds no new reasoning core; it
orchestrates what already exists, and every guarantee still holds.

- `core/wake_word.py` gates the loop: `WakeWordDetector` matches an English or
  Arabic wake phrase ("jarvis", "hey jarvis", "يا جارفيس"), and
  `AudioSpeechGate` rejects silence and clips too short or too quiet to be speech.
- `core/command_router.py` (`NaturalLanguageRouter`) turns a clear sentence into a
  sandboxed tool plan — calculate, search, list, read, write — and routes
  everything else to the assistant. A proposed tool call must already be
  on the sandbox allow-list, so a sentence can never widen the agent's authority.
- `core/jarvis.py` (`JarvisMode`) runs the loop and records every turn. It keeps
  one conversation session per run, so follow-up questions stay in context, and it
  records tool turns in the same session.
- `core/jarvis_controls.py` (`JarvisControls`) recognises a few conversation-level
  commands — `stop`, `speak`, `repeat`, `new conversation`, `save`, `sessions`,
  `that's right`, `that's wrong, it's ...`, `help` — as exact phrases, never
  hijacking a real question like "how do I stop the pump?".
- `core/preferences.py` (`UserPreferences`) remembers the user's name, language,
  spoken-answer preference, and wake word, persisted with `JsonStore`.
- Sessions persist under the data directory: `--session <id>` resumes a saved
  conversation (autosaving after each turn), and `--list-sessions` shows what is
  stored.
- Learning from explicit feedback: `that's right` confirms the last answer and
  `that's wrong, it's ...` corrects it. The assistant and the learner share one
  document store, so a corrected statement is visible to the very next answer.
  Pass `--no-feedback` to disable it.

```bash
python -m cli jarvis "what is permeability?"          # answer, with citations
python -m cli jarvis "list the files in notes/"       # run a sandboxed tool
python -m cli jarvis "calculate 2 * (3 + 4)"          # route to the calculator
python -m cli jarvis --read-write "write to file notes/a.txt: permeability is high"
python -m cli jarvis --wake "hey jarvis what is porosity?"   # require the wake word
python -m cli jarvis --tts auto --play "what is porosity?"   # synthesize and play the answer
python -m cli jarvis --tts auto --write-wav out.wav "what is porosity?"  # save the spoken answer
python -m cli jarvis                                  # interactive text loop
python -m cli jarvis --list-sessions                  # show saved sessions
python -m cli jarvis --session CNV-xxxxxxxx           # resume a saved session
```

In the interactive loop, follow-up questions share one conversation, and the
conversation controls work by voice or text:

```
You: what is porosity?
You: and how is it measured?      # same conversation, context preserved
You: repeat                       # replay the last answer with its citations
You: stop                         # mute the voice until you say 'speak'
You: new conversation             # start over
You: save                         # persist this session
You: sessions                     # list saved sessions
You: that's right                 # confirm the last answer, record it
You: that's wrong, it's ...       # correct the last answer with new wording
You: help                         # list the controls
```

With no speech engine installed, Jarvis still runs as a complete text loop — the
same fallback philosophy as the rest of the project. `--mic` records from the
system microphone when `arecord` or `sox` is available, and `--play` uses
`aplay`/`afplay`/`sox` when one is installed.

### Chatting in text (Jarvis without the voice)

`chat` is the plain conversation entry point: the same grounded loop as Jarvis —
one session, follow-up questions keep context, every answer cites its sources —
but it stays in text. No microphone, no speaker, no wake word, so it behaves the
same on a machine with no audio at all.

```bash
python -m cli chat                          # interactive conversation
python -m cli chat "what is porosity?"      # one question, then exit
python -m cli chat --no-web                 # answer only from what it knows
python -m cli chat --mind                   # answer from the trained model, not the sources
python -m cli chat --ultron                 # answer from Ultron, the general local model
python -m cli chat --vision                 # answer from Vision, the petroleum local model
python -m cli chat --banner                 # answer from Banner, the presentation local model
python -m cli chat --cloud                  # answer from a remote model (needs an API key)
python -m cli chat --cloud --cloud-provider groq --cloud-model llama-3.3-70b-versatile
python -m cli chat --session demo           # resume/autosave a named session
python -m cli chat --list-sessions          # show saved conversations
python -m cli chat --json "what is porosity?"   # machine-readable turn
```

The spoken controls still work as typed commands — `new conversation`, `save`,
`sessions`, `repeat`, `web on`/`web off`, `that's right`, `that's wrong, it's ...`,
and `help` — only the wording changes so it never mentions muting or replaying
audio. Type `quit` to leave.

### Answering from the trained model (`--mind`)

By default `chat` answers the way Jarvis does: it retrieves the passages that
support an answer and cites them. With `--mind` it instead asks the **trained
language model** (`ml/llm/checkpoints/fayd-llm.npz`) directly, with no retrieval
and no citations, so the quality of what the model itself has learned is visible:

```bash
python -m cli chat --mind "what is porosity?"          # answer from the model only
python -m cli chat --mind --mind-tokens 160 "explain permeability"   # longer generation
python -m cli chat --mind --model-checkpoint path/to/fayd-llm.npz "hi"  # a specific checkpoint
python -m cli mind-chat                    # the same thing, from the menu/CLI command
```

In the portable numbered menu, entry **“Chat with the AI (trained model, no
sources)”** opens exactly this mode, so the portable build (which has no
command line) can reach it too. The plain **“Chat with the AI”** entry is
unchanged and still answers from the sources with citations.

Every such answer is labelled `[mind] answered from the trained model only; not
grounded in cited sources.` — it is the model's own prose, not a cited extract,
and it is never checked against sources. That is the point: it measures the
model. A missing checkpoint is reported plainly, with the command to train one
(`python -m cli llm`).

Inside an interactive session you can switch mouths without restarting: type
`from the model` to answer from the model, `from ultron` for the general local
model, `from vision` for the petroleum local model, `from banner` for the
presentation local model (all three below), `from cloud` for a remote model (see
below), or `from the
sources` for cited, retrieved answers again. When
the sources cannot answer a question, the chat prints a short hint pointing at
the other mouths. Those phrases are recognised by `chat` alone — they are
**not** Jarvis controls, so the Jarvis command's wording is unchanged.

The model mouth is wired only in `ai/chat.py` (`ai/mind.py` holds the logic) and
never touches `ai/jarvis.py`, `core/jarvis.py`, or the session builder, so Jarvis
mode is unaffected. The Ultron (`ai/ultron_mouth.py`), Vision
(`ai/vision_mouth.py`), Banner (`ai/banner_mouth.py`), and Cloud
(`ai/cloud_mouth.py`) mouths follow the same
rule, so all six mouths answer in one session.

### Ultron: a general, advanced local model (optional)

The two mouths above are both *narrow*: the grounded assistant answers only from
the petroleum corpus it was seeded with, and `--mind` speaks the small in-project
checkpoint. **Ultron** is a third mouth that is deliberately different — a real,
general, advanced model that runs **locally** and talks about *every field of
life*: science, history, technology, art, everyday advice, anything the user
raises, with no subject filter.

```bash
python -m cli chat --ultron                       # chat answering from Ultron (same window, same session)
python -m cli ultron                              # Ultron on its own (type 'quit' to exit)
python -m cli ultron "explain how a jet engine works"     # one question, answered locally
python -m cli ultron --session demo "and who invented it?"   # keep context across runs
python -m cli ultron --no-stream --max-tokens 800 "write a short poem about rain"
python -m cli ultron models                       # list the models it can fetch
python -m cli ultron status                       # what is installed and present
python -m cli ultron download --approve           # one-time: fetch a model (~5 GB)
```

It is a **mouth inside the project's own chat**: the window's model selector
(the one above the transcript, beside the mode switch) lists **Ultron (general
local model)** beside *From sources (cited)* and *From trained model*, and the
sidebar's **Answer from Ultron** entry selects it. Whichever way you pick it, the conversation continues in the
**same transcript, the same session, and with the same conversation controls** as
the other two mouths — there is no separate window. In the text chat, type
`from ultron` (or `from the sources` to switch back). It is still wired
**directly below Jarvis** in the numbered menu as its own program
(`python -m cli ultron`).

The model is a **GGUF file under the writable data folder** (`data/ultron` by
default), not an in-project checkpoint: drop any `.gguf` there and Ultron uses
it, or run `download --approve` to fetch a known-good general model
(`qwen2.5-7b` by default; `qwen2.5-3b` and `llama-3.2-3b` are lighter choices).
The folder is git-ignored, so a multi-gigabyte model never dirties the working
tree or bloats a program update.

Ultron runs entirely on this machine through `llama-cpp-python`; nothing is sent
to a remote server. That runtime and `huggingface_hub` are an **optional extra**
(`pip install -r requirements-ultron.txt`, or the `[ultron]` extra), imported
lazily, so a base install keeps working and reports clearly when they are absent.
Downloading the weights is an external action that leaves the machine, so it
runs only through the same permission and human-approval gates as a web search or
the image model — denied by default, and granted with `--approve` (or the
window's **Download model…** button).

Every Ultron answer is labelled `[ultron] answered by the local general model;
not grounded in cited sources.` It is the model's own prose, not a cited extract,
and it is never checked against the corpus — the same honesty rule the other
mouths follow. `core/ultron.py` holds the engine (an injectable runner, the
download catalog, and the gates); `ai/ultron.py` is the CLI entry point; and
`ai/ultron_mouth.py` is the chat mouth that lets Ultron answer inside the
project's own conversation (`from ultron` in text, the model selector in the
window).

Ultron has no retriever, so a file you attach is read the other way: the file's
whole text is written into the conversation as reference material (a `CONTEXT`
turn in `core/conversation.py`) and put in front of your question for the model.
That is what makes "attach a document, then ask Ultron about it" work inside the
same chat. The grounded mouth is unaffected — it already retrieves the file's
chunks and cites them — and the conversation's own transcript (`render`) leaves
reference material out, so a file is never counted twice. Reference material is
the last thing evicted from the bounded window, so an attachment stays readable
for many turns; the prompt is capped at `ai.ultron_mouth.DEFAULT_REFERENCE_CHARS`
(8000) with a visible marker, so a very long file cannot overflow the model's
context window.

### Vision: a petroleum-engineering local model (optional)

**Vision** is Ultron's *specialist* counterpart, added the same way and living in
the same places. Where Ultron is a general model that talks about every field of
life, Vision is a real, advanced model that runs **locally** and specialises in
**petroleum engineering** — reservoir, drilling, production, petrophysics, PVT,
well testing, and oil-and-gas facilities — built on the **OGAI Reasoner** weights
(a DeepSeek-architecture engineering system for oil and gas operations).

```bash
python -m cli chat --vision                       # chat answering from Vision (same window, same session)
python -m cli vision                              # Vision on its own (type 'quit' to exit)
python -m cli vision "what is the material balance equation?"   # one question, answered locally
python -m cli vision --session demo "and for a gas cap?"   # keep context across runs
python -m cli vision models                       # list the models it can fetch
python -m cli vision status                       # what is installed and present
python -m cli vision download --approve           # one-time: fetch a model (~5 GB)
```

Like Ultron, it is a **mouth inside the project's own chat**: the window's model
selector (the one above the transcript, beside the mode switch) lists
**Vision (petroleum local model)** beside *From sources (cited)*, *From trained
model*, and *Ultron (general local model)*, and the sidebar's **Answer from
Vision** entry selects it. Whichever way you pick
it, the conversation continues in the **same transcript, the same session, and
with the same conversation controls** — there is no separate window. In the text
chat, type `from vision` (or `from the sources` to switch back). It is wired
**directly below Ultron** in the numbered menu as its own program
(`python -m cli vision`).

The model is a **GGUF file under the writable data folder** (`data/vision` by
default), not an in-project checkpoint: drop any petroleum `.gguf` there and
Vision uses it, or run `download --approve` to fetch a known-good model
(`ogai-reasoner` by default; `ogai-8x7b` is the larger mixture-of-experts
choice). The folder is git-ignored, so a multi-gigabyte model never dirties the
working tree or bloats a program update. The default `ogai-reasoner` repo is
**gated** on Hugging Face: accept its licence and sign in (`huggingface-cli
login`) before downloading, and the CLI says so if the fetch is refused.

Vision runs entirely on this machine through `llama-cpp-python`; nothing is sent
to a remote server. That runtime and `huggingface_hub` are an **optional extra**
(`pip install -r requirements-vision.txt`, or the `[vision]` extra) — the *same*
runtime Ultron uses, so a machine that already has Ultron needs nothing new;
it is imported lazily, so a base install keeps working and reports clearly when
it is absent. Downloading the weights is an external action that leaves the
machine, so it runs only through the same permission and human-approval gates as
a web search or the image model — denied by default, and granted with `--approve`.

Every Vision answer is labelled `[vision] answered by the local petroleum model;
not grounded in cited sources.` It is the model's own prose, not a cited extract,
and it is never checked against the corpus — the same honesty rule the other
mouths follow. `core/vision_model.py` holds the engine (an injectable runner, the
download catalog, and the gates); `ai/vision_model.py` is the CLI entry point;
and `ai/vision_mouth.py` is the chat mouth that lets Vision answer inside the
project's own conversation (`from vision` in text, the model selector in the
window).

Vision has no retriever, so a file you attach is read the other way: the file's
whole text is written into the conversation as reference material (a `CONTEXT`
turn in `core/conversation.py`) and put in front of your question for the model.
That is what makes "attach a document, then ask Vision about it" work inside the
same chat. The grounded mouth is unaffected — it already retrieves the file's
chunks and cites them.

### Banner: a presentation local model (optional)

**Banner** is the Presentation Studio's own model, promoted to a first-class
mouth so the same model that drafts a deck can also be talked to directly. It is
a real, advanced model that runs **locally** — the Studio's authoring model,
**Qwen3 8B Instruct** (`qwen3:8b`) — dedicated to turning a discussion into an
outline and an outline into slide content.

```bash
python -m cli chat --banner                       # chat answering from Banner (same window, same session)
python -m cli banner                              # Banner on its own (type 'quit' to exit)
python -m cli banner "outline a talk on hydraulic fracturing"   # one prompt, answered locally
python -m cli banner --session demo "expand slide 3"   # keep context across runs
python -m cli banner models                       # list the models it can fetch
python -m cli banner status                       # what is installed and present
python -m cli banner download --approve           # one-time: fetch the model
```

Like Ultron and Vision, it is a **mouth inside the project's own chat**: the
window's model selector lists **Banner (presentation local model)**, and the
sidebar's **Answer from Banner** entry selects it. Whichever way you pick it, the conversation continues in the
**same transcript, the same session, and with the same conversation controls** —
there is no separate window. In the text chat, type `from banner` (or `from the
sources` to switch back). It is wired **directly below Vision** in the numbered
menu as its own program (`python -m cli banner`).

Banner is deliberately a *thin* layer, not a second implementation: the catalog,
the model folder, the download gate, and the in-process GGUF runner all live in
`core/presentation_model.py`, and `core/banner.py` reuses them. One catalog and
one model file mean the Studio and Banner can never disagree about what is
installed, and a user who already fetched the weights for the Studio downloads
nothing extra for Banner. The model is a **GGUF file under `data/presentation`**
(the same folder the Studio uses, git-ignored), it shares the **`[presentation]`
runtime extra** (`llama-cpp-python` + `huggingface_hub`, imported lazily), and
downloading the weights runs only through the permission and human-approval gates
(`--approve`), denied by default.

Every Banner answer is labelled `[banner] answered by the local presentation
model; not grounded in cited sources.` It is the model's own prose, not a cited
extract, and it is never checked against the corpus — the same honesty rule the
other mouths follow. `core/banner.py` holds the engine (an injectable runner, the
catalog, and the gates); `ai/banner.py` is the CLI entry point; and
`ai/banner_mouth.py` is the chat mouth that lets Banner answer inside the
project's own conversation (`from banner` in text, the model selector in the
window).

### Cloud: a remote model you choose (optional)

Every mouth above runs **on this machine**. **Cloud** is the one deliberate
exception: an *opt-in* mouth that reaches a remote, OpenAI-compatible service —
**ChatGPT (OpenAI)**, **Claude (Anthropic)**, **Gemini (Google)**, Groq,
OpenRouter, Mistral, DeepSeek, or a local server (Ollama, LM Studio) — using
**your own API key**. The project ships **no** key and talks to no AI service on
its own; the mouth stays closed until you set a key *and* select it.

```bash
python -m cli cloud status                        # is a key present, and is it reachable?
python -m cli cloud providers                     # the catalog: OpenAI, Anthropic, Gemini, ...
python -m cli cloud models                        # the models the provider serves *now*
python -m cli cloud use gemini                    # remember a provider (Gemini *or* Groq, ...)
python -m cli cloud use gemini --model gemini-3.8-flash   # also remember its model
python -m cli cloud "who wrote Hamlet?"           # one question, answered remotely
python -m cli cloud --provider anthropic "explain PVT"
python -m cli cloud --provider ollama --model llama3.2 "hi"   # a local server, no key
python -m cli cloud-chat                          # the cloud mouth inside `python -m cli chat`
python -m cli chat --cloud "what is porosity?"    # the same thing
```

### Choose one provider or the other (remembered)

The provider is a plain either/or choice, and it is **remembered**: run
`python -m cli cloud use gemini` once and every later `cloud` command, the window,
and the next launch use Gemini — no flag needed. Run `python -m cli cloud use groq`
to switch to Groq instead; `python -m cli cloud use` with no name lists the
catalog and asks. A one-off `--provider <key>` still wins for a single turn. In
the window, open **Settings → Cloud provider** and pick from the menu; the choice
is stored at once and shown under the menu (including whether that provider's key
is already set). Only the provider's *name* is saved, never a key.

The **model** is remembered the same way, per provider. If a provider retires the
model the project defaults to (Google shut `gemini-2.0-flash` down), you get a
clear error and two ways out:

```bash
python -m cli cloud models                        # what the endpoint really serves
python -m cli cloud use gemini --model gemini-3.8-flash   # fix it once, remembered
python -m cli cloud --model gemini-3.8-flash "hi"         # or just for one turn
```

The window's **Settings → Cloud provider → Model** dropdown does the same thing:
pick a model from the list — the current provider's known lineup, with the
project's default first (for Gemini: `gemini-2.5-flash`, `gemini-2.5-flash-lite`,
`gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite`) — and every later
answer, in the window and the CLI, uses it. You can still type a name the catalog
does not list yet, and it stays in the menu. With nothing saved, the provider's own
default applies.

Two of the catalog entries need **no card** — both have a free tier:

| Provider | Key variable | Where to get a free key |
|---|---|---|
| **Gemini** (Google) | `GEMINI_API_KEY` | <https://aistudio.google.com/apikey> → *Create API key* |
| **Groq** | `GROQ_API_KEY` | <https://console.groq.com/keys> → *Create API Key* |

OpenRouter and Mistral also have free tiers, and `ollama`/`lmstudio` run on your
own machine with no key at all.

**Gemini, step by step**

1. Sign in at <https://aistudio.google.com/apikey> with a Google account.
2. Click **Create API key** and copy it.
3. Set it in your shell: `export GEMINI_API_KEY="your key"` (Windows PowerShell:
   `setx GEMINI_API_KEY "your key"`, then open a new terminal).
4. `python -m cli cloud use gemini` — remembers Gemini.
5. `python -m cli cloud status` — should print **Cloud ready**.
6. Ask: `python -m cli cloud "اشرح المسامية في المكامن"`.

**Groq, step by step**

1. Sign in at <https://console.groq.com/keys> (email or Google; no card).
2. Click **Create API Key** and copy it.
3. `export GROQ_API_KEY="your key"` (Windows: `setx GROQ_API_KEY "your key"`).
4. `python -m cli cloud use groq` — remembers Groq.
5. `python -m cli cloud status` — should print **Cloud ready**.
6. Ask: `python -m cli cloud "what is a kick-off meeting in drilling?"`.

To go back to Gemini, run `python -m cli cloud use gemini` again; to use the paid
OpenAI models, `python -m cli cloud use openai` after setting `OPENAI_API_KEY`.

The key is read from the environment — `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GEMINI_API_KEY`, and so on (or `FAYD_CLOUD_API_KEY` for any provider) — and is
**never** written to a file, a message, or a log: an error string is redacted
before it is shown. A local server (Ollama, LM Studio) needs no key at all.

The layer is governed like every other external action. Reaching a remote model
is the `cloud_chat` tool, and it passes **two** gates before a byte leaves the
machine: the permission policy (`Capability.EXTERNAL_TOOL`) and the
human-approval gate. Both are **denied by default**, so a bare engine answers
nothing; choosing the Cloud mouth in the chat, or passing `--cloud`, is the
explicit approval. Every attempt — allowed or refused — is written to the audit
log. There is **no third-party dependency**: the client is a thin `urllib`
wrapper, so a base install reaches a cloud model.

Like the other mouths, Cloud is a **mouth inside the project's own chat**: the
window's model selector lists **Cloud (remote model)**, and the sidebar's
**Answer from Cloud** entry selects it; the conversation stays in the **same transcript and the same session**. The
settings panel also carries the **Cloud provider** menu, so you pick Gemini *or*
Groq (or any provider) right in the window, and the choice is remembered. In
the text chat, type `from cloud` (or `from the sources` to switch back).
`core/cloud_model.py` holds the engine, the catalog, and the gates;
`ai/cloud.py` is the CLI entry point; and `ai/cloud_mouth.py` is the chat mouth.

Every Cloud answer is labelled `[cloud] answered by a remote model over the
network; not grounded in cited sources.` — it is a remote model's own prose, sent
over the network, and it is never checked against the corpus. That is the honest
trade you accept by selecting the mouth.

## Graphical interface (ChatGPT-style)

A desktop window that looks and reads like ChatGPT ships **next to** the command
line, not instead of it. The CLI is untouched: `python -m cli` still opens the
numbered menu, and every `python -m cli <command>` behaves exactly as before.

```bash
python -m cli gui                       # from the menu/CLI command
python -m gui                           # the same window, directly
python -m cli gui --mind                # start answering from the trained model
python -m cli gui --ultron              # start answering from Ultron (general local model)
python -m cli gui --vision              # start answering from Vision (petroleum local model)
python -m cli gui --banner              # start answering from Banner (presentation local model)
python -m cli gui --cloud               # start answering from a remote model (needs an API key)
python -m cli gui --session demo        # resume a saved conversation
python -m cli gui --no-web              # answer only from internal knowledge
python -m cli gui --mode jarvics        # start with sandboxed tools enabled
```

The numbered menu shows it as the first entry, **“Open the graphical
interface (ChatGPT-style)”**; choosing it launches the window.

The window is a thin shell over the same code the CLI already runs:

- It builds the **same text-only Jarvis session** `python -m cli chat` builds
  (`ai.jarvis.build_session(..., text_only=True)`), and reuses `ai.chat`'s
  `ModelSwitcher` to swap the answering mouth. A GUI answer and a CLI answer come
  from the same path, so there is no second implementation to keep in sync.
- **Two operating modes** replace the old Chat/Work controls. **AI Mode** answers
  conversationally and never runs a tool — it uses an `AnswerOnlyRouter` that
  always decides *answer*, even for a sentence the Jarvics router would read as a
  command. **Jarvics Mode** keeps the real `NaturalLanguageRouter`, so an
  explicit request (“list the files in notes/”, “calculate 2 * (3 + 4)”) can run
  in the sandbox under every existing guarantee. The header shows the active mode
  by name and switches with one control, and beside it sits the **model
  selector** — *From sources (cited)*, *From trained model*, *Ultron (general
  local model)*, *Vision (petroleum local model)*, *Banner (presentation
  local model)*, and *Cloud (remote model: ChatGPT, Claude, Gemini)* — so the
  answering mouth
  changes inside the *same* chat. The sandbox tools still run in Jarvics Mode;
  they are no longer listed in the sidebar. The chosen mode
  persists (see `gui/modes.py`), and switching never clears the transcript.
- The **composer** is a rounded, ChatGPT-like card: a multiline field (Enter
  sends, Shift+Enter inserts a newline), a send button, and a **＋** control that
  opens the evidence and import choices — **Attach a file to this chat** (the
  file's text joins this conversation's evidence *and* is written into the
  conversation itself as reference material, so a question can cite it like the
  built-in corpus and Ultron — which never retrieves — can read it too; it shows
  as a removable chip), **Attach a scanned PDF
  (OCR)…** (the same, for a PDF with no text layer; needs the optional `[ocr]`
  extra), **Open Knowledge Import Center…** (the persistent path below), **Import
  a SQL database (SQLite)…**, and **Generate an image from text…** (below).
- In Jarvics Mode a message that would **change files** (a `write_file` plan) is
  previewed and held for confirmation — the plan's steps are shown with **Run** /
  **Cancel** — so a destructive tool never runs unreviewed. Read-only tools run
  immediately, and every tool step is rendered with its result or error, so the
  window never claims an operation succeeded unless the backend reported it.
- **Markdown is rendered** in the transcript (headings, lists, quotes, code
  fences with syntax highlighting, tables, rules, and inline bold/code), through
  the dependency-free renderer in `gui/markdown.py`.
- The **sidebar** carries the logo, **New chat**, a **search** box, the saved
  conversations grouped by date (*Today* / *Yesterday* / *Previous 7 days* /
  *Previous 30 days* / *Older*) and a **Settings** panel with the **Cloud
  provider**/**Model** dropdowns, **Save conversation**, and the **Actions** list.
  The sidebar collapses to a drawer on a narrow (mobile-sized) window and is
  restored when the window widens.
- The sidebar's **Actions** buttons and the **Saved chats** list call the
  `cli.menu` `run_*` functions the numbered menu calls. The **Actions** section
  adds numbered CLI items **6 through 12** — learn, insight, Internet learner,
  web verify, the two knowledge-base entries, and the Import Center — with the
  same labels and default arguments, so nothing is offered twice: the window's
  own controls cover the first five menu items (the window, Jarvis, **Ultron**,
  **Vision**, and the two chat modes).
  The training and evaluation entries that already lived in the sidebar are kept
  here too, even though they are hidden from the numbered CLI menu.
- Answers run on a **worker thread** and are posted back through a queue, so the
  window never freezes while the model thinks; output from a sidebar action is
  streamed into the transcript.

Tkinter is used for the window because it ships with CPython — **no new
dependency** is added, keeping the runtime at pure Python plus numpy/requests. It
is imported lazily by the `gui` command, so a headless machine without Tk can
still run every CLI command; a missing display is reported with the suggestion to
use `python -m cli chat`. The GUI code lives in `gui/` (`service.py`,
`modes.py`, `markdown.py`, `actions.py`, `options.py`, `app.py`) and is included
in the portable bundle.

## Generating an image from text (optional)

The window and the CLI can turn a written prompt into an image file. This is
**optional and additive**, the counterpart to the perception layer in
`core/vision.py`, and it adds **no dependency**: the default backend renders a
valid PNG using only the standard library (`zlib` and `struct`).

Being honest matters more than looking impressive, so the default is stated
plainly: it is a *procedural renderer, not a learned text-to-image model*. It
derives a seed from the prompt and paints a deterministic abstract composition
from it — the same prompt always gives the same picture, and the result is a
geometric pattern, not a depiction of the words.

### The learned backend (optional, real model)

For a genuinely learned text-to-image result, install the optional extra and
download a real diffusion model once:

```bash
python -m pip install -r requirements-image.txt   # torch (CPU) + Diffusers, ~1 GB
python -m cli image download --approve            # one-time, ~4 GB from HuggingFace
python -m cli image generate "a drilling rig at dusk" --backend diffusion
```

The project is not published on PyPI, so `pip install fayd-al-ruh[image]` cannot
work from a source checkout; install the bundled requirements file instead (or,
in the window, the **＋** menu's *Install optional PDF/OCR/image support* entry,
which runs the same command). The requirements file pulls **CPU** torch on
purpose — the GPU/CUDA build would add several gigabytes of NVIDIA packages that
the default model does not need. To use a GPU instead, install the CUDA build of
torch from <https://pytorch.org> first, then the requirements file.

**The default is tuned for photorealistic, clean pictures, not sketches.** The
`realistic` preset (the default) uses the photorealistic SD 1.5 fine-tune
`Lykon/dreamshaper-8-lcm`, sampled in a few LCM steps with guidance 1.5, and a
built-in negative prompt that suppresses the usual "looks drawn" artefacts
(cartoon, blur, grain, deformed anatomy, watermark). Three presets trade realism
for speed:

```bash
python -m cli image presets                                   # list them
python -m cli image generate "a portrait" --backend diffusion --preset realistic
python -m cli image generate "a portrait" --backend diffusion --preset detailed --negative "rain, fog"
```

| preset | model | steps | best for |
| --- | --- | --- | --- |
| `fast` | `stabilityai/sd-turbo` | 2 | speed over detail |
| `realistic` *(default)* | `Lykon/dreamshaper-8-lcm` | 6 | photorealistic pictures |
| `detailed` | `Lykon/dreamshaper-8` | 30 | maximum detail (slow on CPU) |

Realism has a cost on a CPU: the default model is ~4 GB and a generation takes
tens of seconds to a few minutes. The weights are cached under `data/models`
(git-ignored). Requested sizes are snapped to the model's 64px grid and resampled
back with a high-quality filter, so the output is crisp rather than blocky.
Everything about this path follows the project's hard rules:

- It is **optional and additive**: the base install has no torch, and the import
  is lazy, so nothing heavy is pulled unless you actually use it.
- The download **leaves the machine**, so it is an external action behind the
  same permission and human-approval gates as a web search — denied by default,
  granted with `--approve`. A cached model needs no download.
- Every image from it is labelled `learned=True` and names the model, the step
  count, and the seed — no pretending.
- When the extra or the weights are missing, the request fails with a clear
  message (and the window falls back to the procedural renderer); nothing breaks.

In the window, pick **Diffusion (learned)** in the text-to-image page, choose a
**Preset** (Photorealistic is the default), optionally set a **Negative prompt**,
press **Download model…** once, then **Generate**.

A real model can be dropped in behind the `ImageGenerator` protocol without
changing the window, the CLI, or the service; `--model`, `--preset`, `--steps`,
`--guidance`, and `--negative` choose another Diffusers model or override the
preset's settings.

In the GUI, open the composer's **＋** menu and choose **Generate an image from
text…**, or use the sidebar's **Generate an image from text**. Write a prompt (a
trailing size such as `768x512` sets the dimensions), optionally set the width,
height, and seed, and press **Generate**; the picture is shown and saved under the
writable data folder (`data/images`), and **Save a copy…** writes it wherever you
choose.

The CLI is the headless twin:

```bash
python -m cli image generate "a red spiral over a dark sea"
python -m cli image generate "wellhead schematic 768x512" --output shot.png
python -m cli image generate "the same again" --seed 42
python -m cli image backends
python -m cli image presets
```

## Continuous learning

Once the system runs, it must improve from real interaction without losing its
honesty:

- `core/interaction_reward.py` learns reward from real outcomes (not from a
  written condition).
- `core/knowledge_update.py` folds a confirmed statement into the citable source,
  with provenance tracking (`ProvenanceLedger`) and explicit decay: an old
  statement fades, and a statement contradicted by a stronger one is replaced and
  not silently kept.
- `core/continuous_learning.py` links the two: `ContinuousLearner` learns only
  from explicit feedback (`Feedback`) — never from the model's own output — and
  `RegressionMonitor` compares a new metric snapshot against the baseline and
  reports any regression.

Two rules preserve honesty: learning happens only from explicit feedback
(confirmation or correction), and a statement enters the source only if it shares
terms with what we already know. The regression check is the safety net: it can
always refuse a change.

Jarvis mode carries this into the conversation itself. Saying `that's right`
confirms the answer it just gave; `that's wrong, it's ...` replaces it with the
wording you supply. `JarvisMode` hands the last exchange to the `ContinuousLearner`
and re-indexes the retriever, so the correction is visible to the very next
question in the same run. When no learner is configured, it says so plainly rather
than recording feedback that could never change an answer — the `--no-feedback`
flag turns the path off, and nothing is ever learned from the model's own output.

```bash
python -m ai.learn_demo
python -m cli learn
```

## Asking the AI: what can it do and what has it learned

The `insight` command is the project's own answer to "how good is it, and what
does it know?". It answers two questions honestly, and it answers *your*
questions with citations.

```bash
python -m cli insight                       # learning + knowledge report
python -m cli insight "what is porosity?"   # ask the AI, with cited sources
python -m cli insight --interactive         # ask questions until 'quit'
python -m cli insight --json                # machine-readable report
python -m cli insight --no-model "what is permeability?"   # extractive, no model
```

**How well does it learn?** `ai/insight.py` reads the release manifest written by
`python -m ml.release` and reports the metric each layer was *trained and
measured* on: sandbox policy success, predictive-coding brain error, the
supervised outcome head, the knowledge-graph link-prediction score (Hit@1 against
its baseline), the learned reward model's calibration, petroleum-task and
procedural coverage, and the language model's size, capacity tier, and (when
recorded) its **held-out perplexity**. Because the numbers come from
a recorded run rather than a fresh claim, the report repeats a verified result; if
no manifest exists it says so instead of inventing one.

**What has it learned?** It lists the repositories the layers were trained on
(disciplines, courses, the petroleum corpus, the bulk knowledge base), the
learned statements promoted into the project
(`knowledge/learned_petroleum.json`), and the knowledge gained at runtime from
internet learning and Jarvis feedback.

**Asking it questions** uses the same grounded assistant as the rest of the
project: retrieve passages, cite them, and answer. With the bundled language model
it writes the prose; with `--no-model` it answers extractively from the retrieved
passages — either way every answer carries its sources, so a fresh clone is useful
before any training run.

## Continuous internet learning

The same governed web path can also *grow the corpus on its own* from trusted
petroleum sites, as a batch rather than one question at a time. The *Continuous
Internet Learner* takes topics from the project's own petroleum corpus, searches
the internet for each, keeps only results from trusted domains (SPE, OnePetro,
API, ISO, ASTM, Wikipedia, and government or academic hosts), and folds accepted
statements into the project's data store.

Every rule of the system still applies: search is denied until `--approve` grants
the external capability and records a human approval, a statement enters the store
only if it shares terms with existing knowledge (`KnowledgeUpdater`), each learned
statement keeps its source URL, and every run is a **dry run** until `--write` is
given, so you can inspect what would be added before it is saved.

```bash
python -m cli internet-learner --list-topics            # topics from the corpus
python -m cli internet-learner                          # denied by default
python -m cli internet-learner --approve                # search, but write nothing
python -m cli internet-learner --approve --write        # save accepted statements
python -m cli internet-learner --approve --write --max-topics 10 --per-topic 2
python -m cli internet-learner --topics "Porosity" "Permeability" --approve
```

### Fresh results on repeat searches

A search provider returns a *stable, relevance-ranked* list, so searching the
same topic twice returns the same top hits. The learner keeps a small cursor file
(`data/learned_cursor.json`) recording how far each query has been consumed;
because Wikipedia can only rank by relevance, the learner also prefers the most
recently edited article among the results that already pass the trusted-source
and topic-relevance gates. Each run therefore pages forward and finds new
material instead of repeating the same statements. A dry run never moves the
cursor, so previewing does not consume the window.

```bash
python -m cli internet-learner --approve --write        # continues where the last run stopped
python -m cli internet-learner --approve --write --reset-cursor   # start from the top again
```

### Choosing what to learn, step by step

Menu entry (5) opens the **interactive** mode, which walks the project's own
science tree one level at a time: **science branch → course → concept**. Each
level lists its children with numbers, and you can search at any level — a
single concept, a whole course, a whole branch, or everything — without having
to drill to the bottom first.

```
Science branches:
  1) Petroleum Disciplines (29 courses, 198 topics)
  2) Engineering Science (6 courses, 39 topics)
  3) Earth Sciences (7 courses, 45 topics)
  4) Foundations (6 courses, 39 topics)
  a) search all topics
  q) quit
Choose: 1
Petroleum Disciplines:
  1) Petrophysics (8 topics)
  ...
  s) search this branch
  b) back
  q) quit
Choose: 1
Petroleum Disciplines > Petrophysics:
  1) effective porosity
  2) absolute permeability
  ...
  s) search this course
  b) back
  q) quit
Choose: 1
```

A search never decides silently: it prints **every result** (title, source URL,
and snippet) marked `[accepted]` or `[skipped]`, then asks **`[s]ave / [i]gnore`**.
Saving writes the exact statements you just saw — the same preview, with no
second network call — into the project store.

Run it directly with `python -m cli internet-learner --interactive --approve`.

### Promoting reviewed knowledge into the project

By default the learner saves to the writable data store (`data/documents`),
which is not part of the repository — so a learned fact lives only on your
machine. When a batch has been reviewed and you want it to **ship with the
project**, run the interactive mode with `--promote` (the menu entry enables it):

```bash
python -m cli internet-learner --interactive --approve --promote
```

Saving then writes the accepted statements to `knowledge/learned_petroleum.json`,
a file committed to the repository and loaded at startup by
`core/learned_knowledge.py`. A fresh clone therefore already knows them, and
they behave exactly like the seeded corpus: citable documents with their source
URL. Promotion is idempotent — records are de-duplicated by their stable
document id — so re-running never duplicates a statement.

The learner only accepts an article whose title is genuinely about the topic
(a shared head-noun like "reservoir" or "screen" is not enough), rejects
symbolic formula fragments, and strips HTML entities and citation URLs from
snippets, so the promoted corpus stays on-topic and clean.

### The one acceptance policy

Every accept/refuse decision for newly learned knowledge is made by a single
module, `core.acceptance_policy.py`. It is the *only* gate: the same criteria
that reviewed the Petroleum Disciplines branch govern every statement learned
later, whether from a batch topic search or a single-question web answer. There
is no second, softer path for data that arrives afterwards.

A result is accepted only when it clears every gate, in order:

1. **Trusted source** — the host is on the explicit trusted list (SPE, OnePetro,
   API, ISO, ASTM, Wikipedia) or a government/academic host.
2. **Credible enough** — source credibility meets `MIN_CREDIBILITY` (0.6).
3. **Non-empty** — there is a snippet to store.
4. **Not a formula fragment** — symbolic markup is rejected.
5. **About the topic** — the article's title and the topic share at least
   `TITLE_COVERAGE` (0.5) of their words, in *both* directions, so an article
   that is mostly about something else is refused even when it shares a word.
6. **Anchored to seeded knowledge** — the statement must share terms with the
   project's *seeded* corpus (`KnowledgeUpdater`); already-learned statements are
   excluded so learned text cannot vouch for the next statement.

The thresholds are named constants, and the criteria are pinned by
`tests/test_acceptance_policy.py`, so changing acceptance is a visible,
deliberate change rather than a quiet drift.

## Answering from the web, with trust checks

The project can also answer a question it does not already know by searching the
internet — but only under the same governance as every other external action, and
only when the answer is trustworthy enough:

1. **Search the web** (`core/web_source.py`): a `WebSearchService` runs a
   `WebSearchProvider`. By default it reaches the **real internet** through the
   keyless `WikipediaSearchProvider`, and falls back to an offline
   `StaticSearchProvider` when there is no network; an `HttpSearchProvider` points
   at your own endpoint. Search is an **external tool**, so it is denied by
   default; it runs only when a `PermissionPolicy` grants the capability, the tool
   name is on the allow-list, and a live `HumanApprovalGate` approval exists.
   Every decision is written to the append-only `AuditLog`.
2. **Cite the sources** (`core/web_verifier.py`): each result keeps its URL,
   title, snippet, and a transparent credibility estimate over the domain suffix
   (government, standards bodies, and academic hosts rank above generic blogs).
3. **Compare with internal knowledge**: the verifier retrieves the project's own
   passages for the question and measures how much the web content overlaps them
   (`support`), and whether any source contradicts what is already known.
4. **Rate the answer**: a transparent trust score blends source credibility,
   source-to-source agreement, and internal support.
5. **Learn only when trusted**: the answer is folded into the citable corpus via
   `KnowledgeUpdater` (with provenance and decay) only when its trust clears the
   threshold, it does not contradict internal knowledge, and it connects to
   something already known.

```bash
python -m cli web --approve "What is the relationship between porosity and permeability?"
python -m cli web --approve            # prompts: type your own question
python -m cli web "What is a gamma ray log?"          # denied by default
python -m cli web --endpoint https://api.example.com/search "What is a gas kick?"
python -m ai.web_verify --json --approve "What is a gas kick?"
```

In the interactive menu, option **8) Ask the web and verify the answer** grants the
capability and then asks you to type the question, so you are never stuck with a
fixed example question.

### The conversation loop learns from the web too

Jarvis itself uses this same engine. When a question's top internal passage is
weak (score below a transparent threshold, `0.5` by default), the assistant
searches the web, compares the result with what it already knows, and folds a
trusted answer into the store it retrieves from — so the very next retrieval
already knows it. A strong internal hit never triggers a search, and an
off-topic web answer is shown but never learned. The toggle is a normal control.

```bash
python -m cli jarvis "what is the Klinkenberg correction?"     # answers from the web when unsure
python -m cli jarvis --web-threshold 0.5 "..."                 # search more eagerly
python -m cli jarvis --no-web "what is porosity?"              # internal knowledge only
```

Say **web off** / **web on** (or **offline**) in a session to toggle lookups.
The assistant does not decide this on its own: the search stays denied unless the
permission policy grants the external capability and a live approval exists, the
same gate as `cli web`.

Set `FAYD_SEARCH_ENDPOINT` (and optionally `FAYD_SEARCH_API_KEY`) to point the
command at a real search API; the application owns the URL and the key, and none
is stored in the project. With no endpoint it uses the keyless Wikipedia API, and
drops back to a small offline corpus when there is no network — so the whole path
works on a fresh clone.

## The bulk knowledge base (100 MB+)

The curated seeds above are deliberately small. For breadth across *every*
field, the project can build a large reference corpus from Wikipedia (keyless)
and retrieve it as additional, citation-carrying evidence:

```bash
python -m cli knowledge build --target-mb 100   # fetch, chunk, store (needs network)
python -m cli knowledge build-index             # fast full-text index (one-off, ~5 s)
python -m cli knowledge stats                   # show the real size in MB
python -m cli knowledge search "hydraulic fracturing"
```

- `core/knowledge_base.py`: the fetcher (`WikipediaFetcher`, batched MediaWiki
  API), `clean_wikitext`, the chunker, and the sharded JSONL store.
  `build_knowledge_base` walks each seed category **recursively through its
  subcategories** (`--depth`, default 2) — most Wikipedia content lives there —
  de-duplicates titles across the whole build, and is **resumable** through a
  `checkpoint.json`, so a long build can be stopped and restarted and stops
  exactly when the byte target is reached. It honours `Retry-After` and backs
  off on HTTP 429/503.
- **Verified:** a default build reaches **100.28 MB** (81,061 chunks, 11,869
  articles, 545 shards) across 40+ fields, well past the 100 MB floor.
- `core/knowledge_index.py`: a **SQLite FTS5** full-text index (standard library,
  no new dependency) built once beside the shards. `build-index` indexes 81k
  chunks in ~5 s, after which a query returns BM25-ranked chunk ids in
  milliseconds. Without the index, `BulkCorpusRetriever` falls back to a
  streaming scan; with it, an interactive turn never stalls on the corpus.
- `core/rag.py`: `BulkCorpusRetriever` keeps only the top matches, so a 100 MB+
  base never has to fit in memory at once. It serves queries from the FTS5 index
  when present and streams otherwise. It is wired into `RagRetriever` as an
  optional `bulk` source and used by `build_default_assistant(bulk=True)`.
  **Voice and Jarvis enable it automatically once the index exists**
  (`bulk_knowledge_available()`), so a keyless 100 MB+ base widens every
  interactive answer without a configuration step.
- **Where it lives:** under the writable data directory (`data/knowledge_base` by
  default; `FAYD_DATA_DIR` overrides). It is **git-ignored** and built on demand,
  so nothing large is ever committed. Pass `--store-dir` to place it elsewhere.
- **Separate from governed learning:** this text is reference material only. It
  is retrieved and cited (every chunk carries its source URL and the CC BY-SA
  licence), and it never feeds the acceptance policy or the belief layers. The
  governed internet learner remains the only path that can change internal
  knowledge.

> Reality check: free, specialised petroleum text is scarce. Reaching 100 MB
> therefore draws on broad fields (chemistry, physics, geology, engineering,
> mathematics, and more), which is why the default category set is wide. The
> target is a floor, not a guarantee, and `stats` always reports the true size.

## Importing your own documents (the Knowledge Import Center)

The bulk base above comes from Wikipedia. To bring in **your own** files and
folders — TXT, Markdown, CSV/TSV, JSON/JSONL, PDF, and Word — the window's **＋**
control opens the **Knowledge Import Center**, and the CLI offers the same engine:

```bash
python -m cli import-center                            # prompt for a file or folder
python -m cli import-center files report.pdf notes.txt data.csv  # scanned PDFs OCR'd automatically
python -m cli import-center --ocr files scanned.pdf    # force OCR for low-text pages
python -m cli import-center --no-ocr files digital.pdf # never OCR; text layer only
python -m cli import-center folder ./papers            # recursive, bounded batches
python -m cli import-center folder ./papers --job papers   # resumable
python -m cli import-center resume papers              # continue an interrupted job
python -m cli import-center jobs                       # list jobs
python -m cli import-center formats                    # supported / unsupported formats
```

The portable menu's **10) Import files and folders (Knowledge Import Center)**
entry runs the same interactive prompt, so it imports a path you type instead of
only listing the formats; the GUI sidebar's **Import files and folders** action
opens the window page instead.

The window page (**Upload files** / **Import from a local folder** / **Import
jobs**) and the CLI drive the *same* engine, `gui/import_center.py`
(`KnowledgeImporter`), which in turn drives the existing `ingest/` pipeline — no
second ingestion path exists to drift.

- **Two import paths, matching how people work.** *Upload files* imports one or
  many chosen files; *Import from a local folder* scans a folder (optionally its
  subfolders) and imports every supported file in bounded batches. The app runs
  on your machine, so a folder the picker returns is a real local path the
  process can read — there is no server to upload to and no path is ever built
  from free text.
- **Scanned PDFs are read by default, not a dead end.** A PDF with no text layer
  is OCR'd automatically once the optional `fayd-al-ruh[ocr]` extra plus Tesseract
  are installed — the scanned pages of a *mixed* PDF are recovered rather than
  silently dropped, and a digital page that merely carries short text is never
  OCR'd. OCR is also reachable where the user is when the extra is missing: the
  Import Center's **Read scanned PDFs (OCR)** checkbox (with a *force low-text
  pages* toggle and a language field, e.g. `ara+eng`), a **Retry with OCR** button
  on a failure, an **Attach a scanned PDF (OCR)…** entry in the composer's **＋**
  menu, and `--ocr` / `--no-ocr` / `--ocr-languages` on the CLI (an interactive
  run also offers to retry the scanned files it skipped). `--no-ocr` refuses OCR
  entirely; without the extra the file is reported, never mis-read.
- **Bounded memory.** Nothing reads a whole dataset by default: CSV/JSONL are
  streamed row by row, a large TXT is read line by line under a hard cap, a PDF
  page by page, a DOCX block by block, and a folder is walked one directory at a
  time. Measured: importing a 200-file folder peaked at **~0.2 MB** of traced
  memory; a 12.8 MB single text file peaked at **~22 MB** (the capped streamed
  read plus its chunk list).
- **Useful metadata, kept text-free.** Each document records its source path,
  filename/extension, content hash, media type, and ingestion timestamp in
  `ingest/metadata.py` (SQLite beside the store), so an unchanged file is
  recognised and re-imports are cheap.
- **Safe normalization.** Text is NFC-normalized and structural only; meaningful
  Arabic and English content, punctuation, and diacritics are preserved. Arabic
  letter forms are unified only when `--fold-arabic` is passed explicitly.
- **Dedup, retry, resume.** The dedup ledger skips unchanged files; a malformed
  file is recorded per file and never aborts the batch; failed files can be
  retried; a folder job is saved after every file and an interrupted job resumes
  from where it stopped.
- **Never executes content.** Every reader only *parses* bytes into text — no
  macro, script, or instruction inside an imported document is ever run. A
  `.doc`/`.rtf` is refused with a "save as .docx" hint rather than mis-read.
- **Licensing.** The pipeline records a source's licence when one is known, and
  imported text is *reference material only* — like the bulk base, it never feeds
  the acceptance policy or the belief layers, and RAG/training are not wired into
  this phase.

> **PDF and OCR.** A *digital* PDF needs the optional `fayd-al-ruh[pdf]` extra
> (`pypdf`); without it, the file is reported, not mis-read. A scanned PDF with no
> text layer is OCR'd **automatically** once the separate `fayd-al-ruh[ocr]` extra
> plus Tesseract are installed (force it with `--ocr`, refuse it with `--no-ocr`,
> and set `--ocr-languages ara+eng` for Arabic). Without the extra the file is
> reported honestly. OCR is a clearly separated optional path; a digital PDF is
> never OCR'd.
>
> The project is not published on PyPI, so `pip install fayd-al-ruh[ocr]` cannot
> work from a source checkout. Install the extras from the bundled requirements
> files instead -- and the window offers the same install from its **＋** menu
> ("Install OCR support…") and the Import Center, so a desktop user never has to
> open a terminal:
>
> ```bash
> python -m pip install -r requirements-pdf.txt   # digital PDFs
> python -m pip install -r requirements-ocr.txt   # scanned PDFs (OCR)
> ```
>
> The Tesseract engine and its language data are **system** packages that pip
> cannot install: `apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-ara`
> on Debian/Ubuntu, `brew install tesseract tesseract-lang` on macOS, or the
> UB-Mannheim installer on Windows. `python -m cli deps` reports what is present
> and what is missing for each feature.

## Code Studio (change the project's own code safely)

The Code Studio is the project's own "code host in a window": it edits the
project's **own source files** with a local coding model, then lets you test and
merge the change — without ever touching the running program until you say so.

```bash
python -m cli studio status          # coding-model + workspace status
python -m cli studio coder           # is qwen2.5-coder:7b pulled?
python -m cli studio new "add a --json flag to the report command"
python -m cli studio list
python -m cli studio files  add-a-json-flag
python -m cli studio read   add-a-json-flag cli/report.py
python -m cli studio write  add-a-json-flag cli/report.py --file /tmp/new.py
python -m cli studio ask    add-a-json-flag "add a --json flag, keep the text output"
python -m cli studio diff   add-a-json-flag
python -m cli studio check  add-a-json-flag     # runs the project's tests inside the workspace
python -m cli studio merge  add-a-json-flag     # refused unless the checks passed
python -m cli studio discard add-a-json-flag
```

In the window, the composer's **＋** menu and the sidebar both open the **Code
Studio** page. The sidebar entry mirrors the CLI menu's *Code Studio* item, and
the window and the CLI drive the **same** engine (`gui/code_studio.py` over
`core/code_studio.py`), so they share one set of workspaces, one diff, and one
merge gate. The CLI never reimplements git or agent logic.

How it works, and why it is safe:

- **A workspace is an isolated copy.** *New workspace* makes a `git worktree`
  under `data/studio/workspaces/<slug>` on its own `studio/<slug>` branch, cut
  from `main`. Every change — yours or the model's — lands there, so the running
  checkout is never edited and a half-finished change cannot break the program.
- **The coding model edits it.** *Ask the model* sends a plain-language request
  to a local coding model (`qwen2.5-coder:7b` through Ollama by default). It may
  `list_files`, `read_file`, `write_file`, or declare itself `done` — a strict
  JSON protocol, and every step it takes is shown so its work is reviewable
  rather than magic. Its writes are confined to the workspace by a
  deny-by-default path check and a text-extension allow-list, so it cannot escape
  the folder or drop a binary into the project.
- **Test, review, merge.** *Run checks* runs the project's own test command
  (`python -m pytest -q`) inside the workspace; *Show diff* is the change as a
  unified diff; and **Merge** is offered only when the checks have passed *and*
  `main` has not moved on, so a merge can never overwrite newer work. *Discard*
  throws the whole change away.

The coding model is a *local* model with a real, honest cost: it is only as good
as the model you pull, and the Studio says so rather than pretending. When Ollama
is not running, or the model is not pulled, every entry point reports the real
next step (`ollama pull qwen2.5-coder:7b`) instead of failing obscurely. Nothing
is downloaded behind your back: the model is fetched by `ollama pull`, which you
run yourself.

## Multimodal perception (optional)

Perception is an optional add-on, not a requirement: the core and the assistant
work with no image, and when an image is available it is turned into a short,
honest textual description that enters the same grounded path. `core/vision.py`
does not claim to recognise objects it cannot see; it measures measurable
features (dimensions, brightness, saturation, edge density) and attaches the
description as a citable document. The PPM format (`P6`) is simple and
dependency-free, so the whole path is tested on a fresh clone, and a real vision
encoder can be plugged in behind the `VisionEncoder` contract without touching
the assistant.

## Running

```bash
pip install -r requirements.txt
python main.py
```

Or through the unified interface (same behaviour, and it also works in the
portable build):

```bash
python -m cli            # interactive numbered menu
python -m cli gui        # open the ChatGPT-style graphical interface
python -m cli demo       # local demo
python -m cli ml --episodes 60
python -m cli llm --steps 4000
python -m cli eval
python -m cli jarvis "What is the relationship between porosity and permeability?"   # cited, source-grounded answer
python -m cli jarvis "list the files in notes/"    # Jarvis mode: route, act, and answer
python -m cli chat       # chat with the AI in text (Jarvis, no voice)
python -m cli chat --mind   # answer from the trained model itself (no sources)
python -m cli insight    # what the AI can do and what it has learned
python -m cli deps       # show/install the optional PDF and OCR features
python -m cli web --approve "What is a gas kick?"   # web answer, cited and trust-scored
python -m cli paths      # where assets are read and data is written
python -m cli selfcheck  # verify the bundled language model loads
```

## Portable mode

A package can be built to run on a machine with no Python and no intermediate
software (no CUDA, no external service), like DeepFaceLab:

```bash
pip install pyinstaller      # a build tool only, not a runtime dependency
python scripts/build.py      # output in dist/fayd-al-ruh/
```

- `fayd-al-ruh.spec`: the PyInstaller configuration; it bundles the language-model
  checkpoint.
- `scripts/build.py`: builds the package and then copies `run.sh` and `run.bat`
  beside it.
- `run.sh` / `run.bat`: a single launcher. It prefers the built executable if
  present, otherwise it runs the source via `python -m cli` — so the development
  environment stays as it is.

In portable mode, data is written **beside the executable** (`data/`), so the
folder can be copied to a USB stick and keep its state. It can be redirected with
`FAYD_DATA_DIR`.

| Context | Assets (read) | Data (write) |
|---|---|---|
| From source | repository root | repository root (`ml/artifacts`, `ml/llm/checkpoints`) |
| Portable build | inside the package (`_internal`) | beside the executable (`data/`) |

Details in `docs/PORTABLE.md`.

## Tests

```bash
python -m pytest -q
```

## Learning from interaction

The reward in sandbox environments is a hand-written success condition (a proxy
reward). To learn from actually observed outcomes:

```bash
python -m ml --episodes 400 --real-reward
```

- `core/interaction_reward.py`: `InteractionRewardModel` learns a calibration
  (confidence -> observed success rate) and contextual action values, from the
  evaluation corpus (`evaluation.scenarios`) and from human review
  (`outcomes_from_reviews`).
- `ml/interaction_learner.py`: the same Q-learning agent, but its reward is the
  observed rate of the action, not the task condition; and it is granted only
  when the action actually changes the state.

The output shows `outcomes/forecasts/ece/brier` and each task's success rate, so
the quality of the learned reward stays auditable instead of hidden.

## Planning with a learned model

A symbolic world model is learned from observed transitions, then used for
planning and for learning by imagination (Dyna-Q), always measuring success by
the real goal:

```bash
python -m ml --episodes 40 --model-based --seeds 3
```

- `ml/world_model.py`: `SymbolicWorldModel` infers action effects from
  observation only, and returns `unknown` for anything never seen.
- `ml/model_based_planner.py`: `plan_with_world_model` (search inside the learned
  model) and `DynaQAgent` (learning from real and imagined transitions).

Comparison at 40 episodes (mean of 3 seeds): the gain appears in multi-step
tasks — `plant-seed` 0.42->0.71, `boil-water` 0.67->0.92, `recharge-lamp`
0.83->0.96.

## The in-project language model

The project contains its own language model, written from scratch in `numpy` and
trained inside the repository, so no external service and no API keys are needed.

- `ml/llm/tokenizer.py`: byte-level encoding (safe for any script, no training),
  plus a byte-level **BPE** tokenizer (`ml/llm/bpe.py`) that learns subword merges
  from the corpus.
- `ml/llm/transformer.py`: a decoder-only transformer (multi-head causal
  attention, GELU, LayerNorm) with manual backprop and Adam — we verified the
  gradients numerically. It now supports **numbered capacity presets**
  (`--preset small|release-1|capable`), **weight tying**, **dropout**,
  **warmup+cosine learning-rate schedule**, and **gradient clipping**.
- `ml/llm/corpus.py`: an in-repo training corpus (English about the project's
  world and about petroleum engineering, a **corpus generated from the whole
  curriculum** — a sentence for every concept and relation in the disciplines and
  courses — **numbered procedural text** for every recorded procedure, and, when
  enabled, sentences from **every repository's documents**: the authored knowledge
  plus a relevance-filtered sample of the bulk knowledge base, with JSON examples
  in the schema format).
- `ml/llm/evaluate.py`: **honest evaluation** — a document-level train/validation/
  test split, held-out **perplexity**, and cross-tokenizer **bits per byte**, so a
  claim about the model is measurable and comparable rather than a single lucky run.
- `ml/llm/generator.py` and `ai/local_model.py`: text generation, and adaptation
  to the `CognitiveModel` interface so it replaces Ollama anywhere a model is
  accepted.
- `ml/llm/schema.py`: schema-guided decoding. The structure is fixed from
  `MODEL_OUTPUT_SCHEMA`, and the model suggests only leaf content, so it always
  returns valid JSON and is then validated by `LLMReasoner`.

```bash
python -m ml.llm                       # train and save the default checkpoint
python -m ml.llm --steps 4000          # train longer
python -m ml.llm --preset release-1    # the larger, released capacity tier
python -m ml.llm --generate "water"    # train, then sample a continuation
python -m ml.llm --kb-max-chunks 2000  # also train on the bulk knowledge base
python -m ml.llm --tokenizer-kind bpe  # smaller corpus: use the subword tokenizer
python -m ml.llm.evaluate --checkpoint ml/llm/checkpoints/fayd-llm.npz  # held-out perplexity
```

### Resuming training instead of restarting

By default every run builds a fresh model and overwrites the checkpoint. Pass
`--resume` to continue from the checkpoint at `--checkpoint` instead: the saved
weights and the tokenizer they were trained with are loaded, and training picks
up where it stopped rather than starting over.

```bash
python -m ml.llm --resume --steps 1000        # 1000 more steps on top of the saved model
python -m ml.release --resume --llm-steps 1000  # same, as part of a full release run
```

The GUI sidebar's **Train the language model** and **Train everything for a
release** buttons, and the CLI's `release` menu entry, all pass `--resume` by
default, so pressing them accumulates training instead of rebuilding from
scratch. Add `--resume` explicitly for the same behaviour on the command line.

The Adam optimiser's moment estimates and step count live in a small sidecar
next to the checkpoint (`fayd-llm.state.npz`). It is *run state*, not release
content, so it is never committed; a fresh run rebuilds it. When the sidecar is
missing, a resume still continues the weights but starts Adam's moments from
zero. When there is no checkpoint at all, `--resume` falls back to a fresh run
and says so. The warmup+cosine schedule spans the *new* total step count, so
asking for more steps extends the horizon along the same curve.

Capacity presets make growth a deliberate, recorded choice rather than an edit:

| Preset | Layers | Width | Heads | Context | Dropout | Tied |
|---|---|---|---|---|---|---|
| `small` (default) | 3 | 96 | 4 | 64 | 0.0 | no |
| `release-1` | 4 | 128 | 8 | 128 | 0.1 | yes |
| `capable` | 6 | 192 | 8 | 128 | 0.1 | yes |

Measured on the shared **document-level held-out test split**, using **bits per
byte** so tokenizers are compared on equal terms. These are recorded numbers, not
estimates — reproduce with `python -m ml.llm.evaluate`:

| Model | Preset | Params | Steps | Test perplexity | Test bits/byte |
|---|---|---|---|---|---|
| previous checkpoint | `small` | 391,200 | — | 5.781 | 2.570 |
| released checkpoint | `release-1` | 842,624 | 4000 | 4.265 | **2.125** |

Two honest caveats. First, capacity alone is not enough: at only 800 steps the
larger `release-1` model is *worse* than `small` (2.845 bits/byte), so the extra
parameters pay off only with the full schedule. Second, both models are small and
trained on a small in-repo corpus; the improvement is real but the scale is not.

Using it as a cognitive model:

```python
from ai.local_model import LocalLanguageModel

model = LocalLanguageModel(temperature=0.0)   # reads the committed checkpoint
response = controller.reason(model, request)
```

Wiring it into the decision path: when the request carries a `response_schema`,
the model suggests concepts, hypotheses, questions, and plans, and then
`LLMReasoner` validates them; it cannot modify beliefs or run tools:

```python
analysis = controller.analyze_with_model(model, request)   # the same validation path
```

> Explicit disclaimer: this is a real transformer but a small one, trained on a
> small in-repo corpus. It produces coherent text about the project's world, not
> a general model at frontier-model scale.

## Connecting a real language model

The model is passed by the application and no API keys are stored inside the
project:

```python
from core.model_interface import CallableModel

model = CallableModel("my-model", lambda request: client.generate(request))
response = controller.reason(model, request)
```

The wrapper returns text only; tools and external actions stay under the control
of the application and the user.

## Status

- Version: 2.2.0
- Python: 3.12 or newer
- License: MIT — see `LICENSE.txt`
- The release is trained and ready: all outputs are inside `ml/artifacts/` and
  `ml/llm/checkpoints/`, described by `ml/artifacts/release_manifest.json`.
  Reproduce them with `python -m cli release`.
