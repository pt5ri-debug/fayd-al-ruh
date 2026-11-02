# Fayd Al-Ruh — Component Inventory (Phase 0)

**Version:** 2.3.0 · **Audit date:** 2026-11-02

Every component below was confirmed by reading its source. "Key API" lists the
classes/functions a caller actually uses. Counts are measured.

---

## 1. `core/` — the cognitive architecture (303 modules, 82,356 lines)

### 1.1 Knowledge primitives

| Module | Key API | Notes |
| --- | --- | --- |
| `entity.py` | `Entity`, `EntityType` (Object/Person/Concept/Goal/Event/Memory/Question) | id `E-…`, properties, relations, confidence, activation |
| `relation.py` | `Relation` | typed, directed, confidence, evidence_ids, belief_id, metadata |
| `evidence.py` | `Evidence` | description, source, `supports`, strength, data; validated |
| `belief.py` | `Belief`, `BeliefStatus` | lifecycle status, confidence, evidence list |
| `concept.py` | `Concept` | normalised terms, `terms_for(language)`, `is_arabic` |
| `ontology.py` | `Ontology` | 495 concepts / 474 relations; token→concept index; `match_concepts`, `concepts_in`, `find_concepts`, `add_concept`, `add_concept_alias`, `concept_for_label` |
| `knowledge_graph.py` | `KnowledgeGraph` | entities/evidence/beliefs/relations with referential integrity |
| `graph_store.py` | `GraphStore` | dump/load graph via `Serializer` |
| `serializer.py` | `Serializer` | dataclass ⇄ primitive |
| `provenance.py` | `ProvenanceLedger`, `ProvenanceRecord`, `ProvenanceStage` | central provenance envelope |
| `knowledge_state.py`, `knowledge_store.py`, `knowledge_carry.py` | `KnowledgeStore`, `LearnedKnowledgeStore`, `capture_learned`, `restore_learned` | snapshots + cross-session learned relations |
| `acceptance_policy.py` | `rejection_reason` | the write gate |
| `consistency_checker.py` | `ConsistencyChecker`, `ConsistencyReport` | statement negation + polarity conflict |
| `contradiction_detector.py` | `ContradictionDetector` | antonymous-relation aware |
| `conflict_resolver.py`, `belief_conflict.py` | conflict records/resolution | |
| `evidence_processor.py`, `evidence_evaluator.py`, `evidence_weighting.py`, `evidence_trace.py` | evidence pipeline | |
| `relation_polarity.py` | `polarity_of`, `opposes`, `conflicts`, `find_polarity_conflicts` | the polarity vocabulary |
| `relation_consistency.py` | `relation_conflicts`, `RelationConflict` | reported, not hidden |
| `relation_inference.py` | `inferred_relations`, `InferredRelation` | entailment with steps |
| `relation_paths.py` | `find_connection`, `ConnectionPath`, `RelationStep` | shortest typed chain |
| `relation_connections.py` | `concept_connections` | multi-hop, question-scoped |
| `relation_facts.py` | `relation_facts`, `RelationFact` | stored edges as data |
| `relation_extraction.py` | `RelationExtractor`, `extract_from_store`, `merge_extractions` | typed relations from text (refuses rather than guesses) |

### 1.2 Ontology reasoning and retrieval

| Module | Key API |
| --- | --- |
| `ontology_reasoner.py` | `OntologyReasoner`, `RelationRule`, `OntologyInference` |
| `ontology_retrieval.py` | `OntologyGuidedRetriever` (hop-decayed walk) |
| `ontology_learner.py` | `OntologyLearner` (bounded, ceiling-respecting, decay/feedback) |
| `rag.py` | `RagRetriever`, `BulkCorpusRetriever`, `RetrievedPassage` |
| `documents.py` | `Document`, `DocumentStore`, `build_knowledge_documents` |
| `knowledge_base.py` | `KnowledgeBaseStore`, `build_knowledge_base`, `chunk_text`, `WikipediaFetcher` |
| `knowledge_index.py` | `KnowledgeTextIndex` (SQLite FTS5) |
| `embeddings.py` | `coembedding` space: `cooccurrence_counts`, PPMI + SVD, `semantic_space()`, `evaluate_held_out` |
| `tokenize.py` | `normalize_token`, `normalize_phrase`, `content_terms`, `token_spans` |
| `semantic_memory.py` | `SemanticMemoryIndex`, `hashed_embedding` |
| `meaning_model.py`, `semantic_parser.py`, `semantic_relation.py`, `language_reasoner.py` | language/semantic helpers |

### 1.3 Reasoning

| Module | Key API |
| --- | --- |
| `reasoning_engine.py` | `ReasoningEngine` (deduce/induce/abduce) |
| `deductive_reasoner.py`, `inductive_reasoner.py`, `abductive_reasoner.py` | the three reasoners |
| `inference.py`, `inference_rule.py`, `inference_result.py` | inference contracts |
| `uncertainty.py`, `uncertainty_propagator.py` | uncertainty |
| `reasoning_trace.py`, `reasoning_monitor.py`, `reasoning_explainer.py`, `reasoning_report.py` | trace + user-visible report |
| `hypothesis.py`, `hypothesis_engine.py`, `hypothesis_learner.py`, `hypothesis_test.py` | hypothesis lifecycle |
| `experiment*.py` (5) | experiment design/run/result |
| `inquiry.py`, `question.py`, `question_generator.py`, `information_gap.py`, `curiosity.py` | inquiry |
| `introspection.py`, `explanation.py`, `explanation_engine.py` | explanation |
| `counterfactual.py`, `counterfactual_reasoner.py`, `causal_intervention.py` | counterfactuals |
| `analogy.py`, `structure_mapping.py` | structure-mapping analogy |
| `temporal_reasoner.py`, `timeline.py`, `temporal_event.py`, `temporal_pattern.py`, `temporal_relation.py`, `time_prediction.py` | temporal |
| `prediction.py`, `prediction_engine.py`, `prediction_error.py` | prediction |
| `pattern_detector.py`, `pattern_learner.py` | patterns |

### 1.4 Causal

`causal_graph.py`, `causal_link.py`, `causal_chain.py`, `causal_reasoner.py`,
`causal_explainer.py`, `causal_discovery.py`, `causal_effect.py`,
`causal_experiment.py`, `causal_intervention.py`, `causal_paths.py`,
`causal_validator.py` — a complete causal layer; `causal_paths.CAUSAL_RELATION_TYPES`
is a **closed set** (17 types) so an association can never be read as a cause.

### 1.5 Planning, decision, action

`goal.py`, `plan.py`, `planner.py` (BFS + `order_steps` topological),
`plan_evaluator.py`, `action.py`, `action_model.py`, `action_policy.py`,
`action_executor.py`, `decision.py`, `decision_engine.py`,
`decision_explainer.py`, `decision_trace.py`, `utility.py`, `utility_model.py`,
`risk.py`, `risk_model.py`, `state.py`, `state_history.py`, `state_transition.py`,
`state_snapshot.py`, `environment.py`, `symbolic_simulation.py`.

### 1.6 Memory

`working_memory.py`, `episodic_memory.py`, `semantic_memory.py`,
`procedural_memory.py`, `memory.py` (`CognitiveMemory`, `MemoryItem`,
`MemoryRecall`), `memory_associator.py`, `memory_consolidator.py`,
`memory_decay.py`, `memory_retriever.py`, `memory_store.py`,
`retention_policy.py`.

### 1.7 The controller

`cognitive_controller.py` — `CognitiveController` (70 methods) is the hub.
Supporting: `cognitive_cycle.py`, `cognitive_loop.py`, `cognitive_state.py`,
`cognitive_context.py`, `cognitive_evaluator.py`, `cognitive_report.py`,
`cognitive_trace.py`, `cognition.py`, `cognition_catalog.py`, `self_model.py`,
`metacognition.py`, `self_improvement.py`, `architecture_learning.py`,
`error_monitor.py`, `confidence_model.py`, `confidence_monitor.py`.

### 1.8 The answer path

`assistant.py` (`GroundedAssistant`, `AssistantAnswer`, `build_default_assistant`),
`grounding.py` (`check_grounding`, `GroundingCheck`), `composition.py`
(`compose_model_answer`, `grounded_answer`, `with_lead`, `topic_phrase`),
`answer_lead.py`, `answer_plan.py`, `answer_decision.py`, `answer_belief.py`,
`answer_analogy.py`, `answer_temporal.py`, `answer_confidence.py`,
`answer_audit.py`, `answer_provenance.py`.

### 1.9 Learning

`learner.py`, `learning_engine.py`, `learning_signal.py`, `learning_metrics.py`,
`learning_history.py`, `adaptive_learning.py`, `online_learner.py`,
`continuous_learning.py`, `self_learning.py`, `strategy_learning.py`,
`skill_learning.py`, `knowledge_update.py`, `reinforcement.py`,
`interaction_reward.py`, `safe_learning.py`, `learning_records.py`,
`learning_pipeline.py`, `learning_audit.py`, `learning_isolation.py`,
`belief_learner.py`, `hypothesis_learner.py`, `pattern_learner.py`.

### 1.10 Characters

`character_learning.py`, `character_learning_store.py`, `character_config.py`,
`character_registry.py`, `character_engine.py`, `character_identity.py`,
`character_compare.py`, `character_consistency.py`, `character_memory_view.py`,
`character_memory_gate.py`, `character_seeding.py`, `character_isolation_gate.py`,
`character_identity_gate.py`, `personality_layer.py`, `mouth_profile.py`,
`knowledge_sharing.py`.

### 1.11 Models

`model_interface.py` (`CognitiveModel`, `ModelRequest`, `ModelResponse`,
`MultimodalInput`), `model_manager.py`, `model_registry.py`, `model_runtime.py`,
`model_files.py`, `model_settings.py`, `model_log.py`, `model_sources.py`,
`model_download.py`, `model_resources.py`, `model_preflight.py`,
`model_process.py`, `model_selection.py`, `model_advisor.py`,
`language_models.py`, `custom_models.py`, `shared_models.py`,
`trained_artifacts.py`, `optional_deps.py`.

### 1.12 Model mouths (local/remote)

`ultron.py`, `vision_model.py`, `banner.py`, `cloud_model.py`,
`presentation_model.py`, `coder_model.py`, `coder_llama.py`, `docs_model.py`,
`qwen_speech.py`, `piper_speech.py`, `audiocpp_speech.py`, `speech.py`,
`voices.py`, `voice_moods.py`, `reading.py`, `arabic_speech.py`,
`arabic_glossary.py`, `arabic_prose.py`, `voice_studio_profile.py`.

### 1.13 Perception and documents

`document_reader.py`, `docx_extract.py`, `pdf_extract.py`, `pdf_ocr.py`,
`text_encoding.py`, `text_normalize.py`, `vision.py`, `imagegen.py`,
`imagegen_diffusion.py`, `imageedit.py`, `perception.py`, `docs_studio.py`.

### 1.14 Governance and sandbox

`permissions.py`, `human_approval.py`, `audit_log.py`, `sandbox_runtime.py`,
`sandbox_environment.py`, `sandbox_tasks.py`, `progressive_environment.py`,
`tools.py`, `tool_agent.py`, `command_router.py`, `wake_word.py`,
`preferences.py`, `best_effort.py`, `progress.py`.

### 1.15 Studio / authoring

`code_studio.py`, `code_agent.py`, `presentation.py`, `presentation_layout.py`,
`pptx_engine.py`, `presentation_courses.py`, `scene3d*.py` (6),
`trained_artifacts.py`.

### 1.16 Session / runtime / system

`jarvis.py`, `jarvis_controls.py`, `conversation.py`, `paths.py`, `version.py`,
`updater.py`, `persistence.py`, `snapshot*.py`, `state_snapshot.py`,
`web_source.py`, `web_verifier.py`, `internet_learner.py`,
`knowledge_navigator.py`, `model_process.py`, `environment.py`.

---

## 2. `cognition/` — autonomous cognitive development (29 modules, 9,692 lines)

| Module | Key API |
| --- | --- |
| `proactive_mind.py` | `ProactiveMind.cycle` (8 engines in one bounded cycle) |
| `engines/` | curiosity, motivation, ideas, imagination, criticism, association, questions |
| `scheduler.py` | `ThoughtScheduler.due`, `SchedulerSettings`, `SchedulerState` |
| `runner.py` | `CognitiveRunner` (LIGHT/BALANCED/INTENSIVE, resource-aware deferral) |
| `metacognition.py` | `SelfModel`, `PersistentNovelty`, `FocusBuffer`, `StagnationDetector`, `EscalationPolicy`, `MetacognitiveController` |
| `consolidation.py` | `Consolidator`, `Pattern`, `ReviewSchedule`, `SkillRegistry` |
| `curriculum.py` | `AutonomousCurriculum`, `MultiSeedEvaluator` |
| `learning_system.py` | `CognitiveLearningSystem` (6 memory kinds composed) |
| `study.py` | `StudySession.run` + measured benefit report |
| `question_bank.py` | `QuestionBank`, `next_questions` (3110 questions, 1001 Arabic) |
| `knowledge_depth.py` | `knowledge_depth`, `weak_spots`, `harvest_relations`, `discipline_bridges`, `arabic_reach` |
| `training.py` | staged training (`SMOKE/QUICK/STANDARD/FULL`) |
| `versions.py` | `ModelRegistry`, `ModelVersion`, `should_accept`, `rollback` |
| `teacher_student.py`, `backends.py`, `asking.py` | teacher-student loop |
| `records.py`, `storage.py`, `reliability.py`, `evaluation.py`, `development.py` | records, stores, reliability, 10 indicators, held-out gate |

---

## 3. Character packages

| Package | Modules | Contents |
| --- | --- | --- |
| `companion/` | 18 | Natasha: `CompanionAgent`, personality/traits/moods/initiative, prompt builder, durable SQLite (`database.py`), relationship memory, polish, repetition, language |
| `hakeem/` | 7 | Hakeem: `HakeemAgent`, personality (boundaries, comparative/history methods), prompt builder, own model folder `data/hakeem`, own DB `data/hakeem.db` |

---

## 4. `ingest/` — local-first ingestion (19 modules)

`records.py`, `registry.py`, `interfaces.py`, `config.py`, `pipeline.py`,
`dedup.py`, `jobs.py`, `errors.py`, `delete.py`, `hashing.py`, `metadata.py`,
`source_registry.py`, `sql_job.py`, `connectors/{files,sql,sqlite,web}.py`.

---

## 5. `ml/` — learning stack (27 modules)

`models.py` (SoftmaxClassifier, MLP, LinearRegressor, Adam),
`dataset.py`, `trainer.py`, `learner.py`, `interaction_learner.py`,
`world_model.py`, `model_based_planner.py`, `cognitive_brain.py`,
`petroleum.py`, `procedural_tasks.py`, `knowledge_corpus.py`,
`general_learning.py`, `policy_adapter.py`, `release.py`,
`llm/` (bpe, tokenizer, corpus, transformer, generator, schedule, state,
evaluate, schema, `__main__`).

---

## 6. `ai/` — mouths and session wiring (29 modules)

`chat.py` (`ModelSwitcher`), `characters.py` (`CharacterMouth`,
`wrap_character`), `mouths.py` (`build_mouth`), `jarvis.py` (`build_session`),
`mind.py` (`MindAssistant`), `local_model.py`, plus `ultron*.py`,
`vision*.py`, `banner*.py`, `cloud*.py`, `natasha*.py`, `hakeem*.py`,
`ollama_*.py`, `presentation_author.py`, `insight.py`, `web_verify.py`,
`voice.py`, `speak.py`, `agent_demo.py`, `learn_demo.py`,
`internet_learner.py`.

---

## 7. `gui/` — desktop front end (48 modules, 35,810 lines)

| Group | Modules |
| --- | --- |
| Main window | `app.py` (4,279 lines), `options.py`, `service.py`, `actions.py`, `modes.py` |
| Shared widgets | `themes.py`, `typography.py`, `typography_settings.py`, `rich_text.py`, `markdown.py`, `chat_composer.py`, `sidebar.py`, `window_controls.py`, `dialog_base.py`, `message_footer.py`, `image_preview.py` |
| Dialogs (16) | `code_studio_dialog.py`, `presentation_dialog.py`, `presentation_finalize_dialog.py`, `presentation_course_dialog.py`, `docs_dialog.py`, `import_dialog.py`, `sql_import_dialog.py`, `image_gen_dialog.py`, `image_edit_dialog.py`, `scene3d_dialog.py`, `model_chat_dialog.py`, `model_manager_dialog.py`, `character_dialog.py`, `cognition_dialog.py`, `voice_studio_dialog.py` |
| Tk-free services | `code_studio.py`, `presentation.py`, `docs_studio.py`, `import_center.py`, `sql_import.py`, `image_gen.py`, `image_edit.py`, `scene3d.py`, `model_chat.py`, `model_manager.py`, `character_manager.py`, `cognition.py`, `voice_studio.py`, `attachments.py`, `character_evaluation.py` |

---

## 8. `cli/` — headless twin (25 modules, 11,409 lines)

`__main__.py` (69 commands), `menu.py` (99 items), `analogy.py`, `asking.py`,
`audio.py`, `characters.py`, `cognition.py`, `deps.py`, `docs.py`,
`embeddings.py`, `image.py`, `import_center.py`, `ingest.py`, `knowledge.py`,
`model_chat.py`, `models.py`, `piper_speech.py`, `presentation.py`,
`qwen_speech.py`, `relations.py`, `scene3d.py`, `shared_models.py`,
`studio.py`, `voice_studio.py`.

---

## 9. `storage/` and `evaluation/`

| Module | Key API |
| --- | --- |
| `storage/json_store.py` | `JsonStore` (atomic, percent-encoded keys) |
| `storage/sqlite_store.py` | `SqliteStore` (namespaced JSON values) |
| `storage/graph_database.py` | `GraphDatabase` (in-memory adapter) |
| `evaluation/runner.py` | assembles 55 metrics |
| `evaluation/benchmark.py` | `MetricSpec` catalogue |
| `evaluation/baseline.json` | reference (overall 0.987721) |
| `evaluation/scenarios.py` | fixed scenarios |
| `evaluation/metrics.py` | calibration/Brier/etc. |
| `evaluation/__main__.py` | `python -m evaluation [--update-baseline]` |

---

## 10. Duplication and near-duplication found

| Item | Finding |
| --- | --- |
| `core/cognition.py` | a thin adapter to `CognitiveController`; the *name* collides with the `cognition/` package but there is no behavioural duplication |
| `core/self_model.py` | declared identity (a domain frame), distinct from `cognition/metacognition.SelfModel` (learned self-model). **Two different things with the same name** — rename one to remove the ambiguity |
| `core/curiosity.py` vs `cognition/engines/curiosity.py` | the latter *wraps* the former (documented) — composition, not duplication |
| `core/learner.py` vs `core/learning_engine.py` vs `ml/learner.py` | three learners at different levels (symbolic cycle / learning result / RL policy). Names are close; behaviour differs |
| `core/model_resources.py` vs `cognition/scheduler.py` | both read machine load; no shared governor |
| `core/consistency_checker.py` vs `core/relation_consistency.py` vs `core/contradiction_detector.py` | three overlapping checks; each reads a different input (beliefs / relations / evidence). Consolidating them behind one validation service is a target, not a defect |

No module was found that is a byte-for-byte or behavioural duplicate of another.
