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
   generalisation.
3. **The in-project language model** (`ml/llm`): trained on a corpus that
   includes the whole curriculum.

It then writes a **manifest** at `ml/artifacts/release_manifest.json` recording:

- The project version (`core/version.py`), the training time, and the git revision.
- **The knowledge** trained on: number of disciplines, courses, concepts, and
  relations, and the curriculum coverage (1.0).
- **The corpus**: number of curriculum sentences, built examples, and characters.
- **The configuration**: episodes, language steps, seed, and replay size.
- **The results**: petroleum task success, and the language-model parameter count.

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
python -m cli ask "What is the relationship between porosity and permeability?"
python -m ai.ask --no-model "What is a well log?"   # extractive only
python -m ai.ask                                    # multi-turn session
```

## The subword tokenizer (BPE)

The in-project language model used to treat every UTF-8 byte as a separate token,
so a common word cost several tokens. We added a byte-level BPE tokenizer trained
inside the project (`ml/llm/bpe.py`): it merges frequent byte pairs, so common
words and English terms and punctuation become single tokens. On the project
corpus it compresses the text by about **5x** (from 64,206 bytes to 12,908
tokens), i.e. an effective context five times longer. The tokenizer is optional:
`--tokenizer-kind byte` restores the old behaviour.

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
`espeak`, `flite`, or macOS `say`; ASR uses `whisper` when it is installed. If
none is present, the project can still run a complete *text loop*: the recognizer
returns supplied text, and the synthesizer produces a real silent WAV clip. This
way the whole voice path is tested on a fresh clone, and a real ASR/TTS engine
can be plugged in later without touching the core. Use `--tts auto` / `--asr auto`
to pick the best installed engine, or name one explicitly (`--tts espeak`); an
explicitly named engine that is missing raises rather than faking audio.

```bash
python -m cli voice "What is the relationship between porosity and permeability?"  # a voice turn
python -m ai.voice --write-wav out.wav "permeability"                              # and write a WAV
python -m ai.voice --tts auto "permeability"                                       # speak with a real engine
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
  everything else to the grounded assistant. A proposed tool call must already be
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

In the interactive menu, option **5) Ask the web and verify the answer** grants the
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
python -m cli demo       # local demo
python -m cli ml --episodes 60
python -m cli llm --steps 4000
python -m cli eval
python -m cli ask "What is the relationship between porosity and permeability?"   # source-grounded assistant
python -m cli jarvis "list the files in notes/"    # Jarvis mode: route, act, and answer
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

- `ml/llm/tokenizer.py`: byte-level encoding (safe for any script, no training).
- `ml/llm/transformer.py`: a decoder-only transformer (multi-head causal
  attention, GELU, LayerNorm) with manual backprop and Adam — we verified the
  gradients numerically.
- `ml/llm/corpus.py`: an in-repo training corpus (English about the project's
  world and about petroleum engineering, and a **corpus generated from the whole
  curriculum** — a sentence for every concept and relation in the disciplines and
  courses, with JSON examples in the schema format).
- `ml/llm/generator.py` and `ai/local_model.py`: text generation, and adaptation
  to the `CognitiveModel` interface so it replaces Ollama anywhere a model is
  accepted.
- `ml/llm/schema.py`: schema-guided decoding. The structure is fixed from
  `MODEL_OUTPUT_SCHEMA`, and the model suggests only leaf content, so it always
  returns valid JSON and is then validated by `LLMReasoner`.

```bash
python -m ml.llm                       # train and save the default checkpoint
python -m ml.llm --steps 4000          # train longer
python -m ml.llm --generate "water"    # train, then sample a continuation
```

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

- Version: 2.1.0
- Python: 3.12 or newer
- License: MIT — see `LICENSE.txt`
- The release is trained and ready: all outputs are inside `ml/artifacts/` and
  `ml/llm/checkpoints/`, described by `ml/artifacts/release_manifest.json`.
  Reproduce them with `python -m cli release`.
