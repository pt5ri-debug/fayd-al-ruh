# Romance & Companion Studio — Feature Matrix

Every row is the **real** state of the shipped code, not an intention. "Requires
a model" means the feature is fully implemented but its live path needs a local
GGUF installed; with none installed the Studio opens, stores everything, and
reports the real next step. "Requires a voice engine" is the same for speech.

## Legend

- **✅ Implemented** — works and is tested.
- **🟡 Partially implemented** — works, with a named limit.
- **⚙️ Requires an installed model** — implemented; the live inference needs a
  local model.
- **🔊 Requires a voice engine** — implemented; live audio needs an engine.
- **🖥️ Requires additional hardware** — the target machine cannot do it well.
- **➖ Not applicable** — deliberately out of scope.

## Characters & personality

| Feature | State |
|---|---|
| Create / edit / duplicate / export / import / archive / restore / delete | ✅ |
| Independent persistent identity (id, name, nickname, fictional age, gender, bio, nationality, languages, appearance, relationship style) | ✅ |
| Adult-eligibility flag (off unless a fictional age ≥ 18 is set) | ✅ |
| Personality dials (introversion, playfulness, warmth, expressiveness, confidence, humor, initiative, affection, romantic intensity, response length, formality, Arabic dialect) | ✅ |
| Editable natural-language instructions | ✅ |
| Discover Characters (shipped, clearly-labelled example companions, added one at a time) | ✅ |
| Delete with explicit retention choice (keep / purge / anonymize) | ✅ |

## Conversation

| Feature | State |
|---|---|
| Nine conversation modes with dedicated system prompts | ✅ |
| Mode change without losing history; old messages never rewritten | ✅ |
| Per-mode context rules and response style | ✅ |
| Multi-character scenes (a cast; sequential generation) | ✅ |
| Persistent conversations and messages (SQLite) | ✅ |
| One local model answering a turn | ⚙️ |
| Cancellation, timeout, graceful failure | ⚙️ |

## Relationship engine

| Feature | State |
|---|---|
| Independent state per character | ✅ |
| Seven dimensions (familiarity, trust, closeness, comfort, affection, shared experiences, stability) | ✅ |
| Evidence-driven change (never random) | ✅ |
| Configurable stages (Stranger → … → Romantic Partner) | ✅ |
| Events with timestamps and message references | ✅ |
| Relationship-history / milestone screen | ✅ |
| Reset / edit relationship state | ✅ |
| Non-dependency, non-manipulation guardrails | ✅ |

## Memory

| Feature | State |
|---|---|
| Short-term (working context) | ✅ |
| Episodic | ✅ |
| Semantic personal | ✅ |
| Relationship memory | ✅ |
| Character memory (fiction continuity) | ✅ |
| Per-character and per-user isolation | ✅ |
| Importance + recency scoring, summarization, duplicate detection | ✅ |
| Search, retrieval, manual review, edit, delete, export | ✅ |
| Source-message references and timestamps | ✅ |
| Configurable retention | ✅ |
| Reuses the project's existing indexing rather than a second engine | ✅ |

## Local models

| Feature | State |
|---|---|
| GGUF discovery through the existing Model Manager | ✅ |
| Model selection, per-character and per-mode assignment | ✅ |
| Quantization / context / memory estimate surfaced | ✅ |
| CPU-first execution | ✅ |
| Model load / unload, runtime health, speed monitoring | ✅ |
| Cancellation, timeout, out-of-memory recovery | ✅ |
| Model compatibility validation (not "downloadable = compatible") | ✅ |
| Comparative model-evaluation facility | 🟡 (quality dimensions measured; a full head-to-head needs several installed models) |
| GPU offload on the target machine | 🖥️ (AMD R7 M360 / Intel HD 520 — CPU-first by design) |

## Voice & audio

| Feature | State |
|---|---|
| Per-character voice settings (voice, engine, language, dialect, speed, pitch, volume, preset) | ✅ |
| Only really-installed, compatible voices shown | ✅ |
| Text-to-speech of a reply, stop, interrupt | 🔊 |
| Speech-to-text / push-to-talk / voice mode | 🔊 |
| Companion adapter reusing the existing engines (no second TTS/STT) | ✅ |

## Dating & roleplay

| Feature | State |
|---|---|
| Scenario builder (title, category, location, setting, participants, roles, background, objectives, narrative style, content level, boundaries) | ✅ |
| Scenario categories (first meeting, coffee shop, virtual date, romantic evening, travel, daily life, friendship, emotional, adventure, multi-character) | ✅ |
| Interactive storytelling with continuity and user choices | ✅ |
| Pause / resume / branch / save / restart | ✅ |
| Multi-character turns with distinct speakers | ✅ |

## Storage & privacy

| Feature | State |
|---|---|
| Versioned, transactional SQLite schema with foreign keys and stable IDs | ✅ |
| Rollback-safe migrations; newer schema refused | ✅ |
| Local-only inference enforced; remote provider refused in local-only mode | ✅ |
| Model downloads gated by explicit permission (reusing the project's mechanism) | ✅ |
| Data export and deletion | ✅ |
| No conversation data in logs; no telemetry | ✅ |
| Optional local database encryption | ➖ (a managed key mechanism is out of scope for this phase; the database is local-only) |

## Interface

| Feature | State |
|---|---|
| Standalone Studio window from the main navigation | ✅ |
| Three collapsible columns; responsive layout | ✅ |
| Light and dark themes; live re-theming | ✅ |
| Arabic and English; RTL and LTR | ✅ |
| Keyboard navigation and accessibility | ✅ |
| No decorative placeholders; every control functional | ✅ |
