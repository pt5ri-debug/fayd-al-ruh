# Romance & Companion Studio — Database Schema (Phase 1)

The Studio keeps its data in **one versioned SQLite file**,
`data/romance/romance.db` (git-ignored; `FAYD_DATA_DIR` moves it). The schema is
written once, in `core/romance/schema.py`, and **derived from the record
dataclasses** in `core/romance/model.py`: a field's SQL type is read from its
annotation rather than duplicated, so there is one place a table's shape lives.

## Type mapping

| Python annotation | SQL type | Notes |
|---|---|---|
| `str` | `TEXT` | |
| `int` | `INTEGER` | |
| `float` | `REAL` | |
| `bool` | `INTEGER` | stored 0/1 |
| `tuple[...]` | `TEXT` | JSON array (round-trips to a tuple) |
| `dict[...]` | `TEXT` | JSON object |
| `X \| None` | same as `X`, nullable | an unspecified value is `NULL` |

A field named `order` is stored as the column `sort_order` (a SQL keyword), while
the Python attribute stays `order`.

## Conventions

- **Stable string ids.** Every record's primary key is `record_id`, minted once by
  `make_id` as `<PREFIX>-<hex>` (e.g. `CHR-3f9a1c2b`). It is never reused, so a
  deactivated record keeps its identity and every foreign key that points at it
  still resolves.
- **Audit columns.** `created_at` and `updated_at` are ISO-8601 **UTC** strings;
  `active` is the soft-delete flag (a record is deactivated, never deleted).
- **Foreign keys are on** (`PRAGMA foreign_keys=ON`). Every reference is
  `ON DELETE RESTRICT` — a record is deactivated, so history is never orphaned —
  except where a child has no meaning without its parent, which is
  `ON DELETE CASCADE` (a message without its conversation, a participant without
  its scenario, a session without its conversation).
- **Transactions.** Every write runs inside a transaction and rolls back on any
  failure, so a half-written record is never left behind.
- **Migrations.** `meta.schema_version` records the version on disk; the numbered
  steps in `MIGRATIONS` bring a file up to `SCHEMA_VERSION`. A database from a
  **newer** build is refused rather than read with a schema this build does not
  understand.

## Tables (schema version 1)

### Characters

| Table | Purpose | Key columns |
|---|---|---|
| `companion_characters` | the character library | `owner`, `name`, `nickname`, `fictional_age`, `adult_eligible`, `gender`, `biography`, `nationality`, `languages`, `appearance`, `avatar_path`, `relationship_style`, `default_mode`, `model_key`, `tags`, `favorite`, `archived`, `profile_version`, `metadata` |
| `character_personalities` | the personality dials (0..1) + instructions | `character_id`, `introversion`, `playfulness`, `warmth`, `expressiveness`, `confidence`, `humour`, `initiative`, `affection`, `romantic_intensity`, `formality`, `response_length`, `dialect`, `traits`, `instructions` |
| `character_profiles` | free-form interests/hobbies/styles | `character_id`, `kind`, `label`, `value` |
| `character_boundaries` | explicit conversation limits | `character_id`, `kind`, `value` |

### Relationship

| Table | Purpose | Key columns |
|---|---|---|
| `relationship_states` | the scored state, one row per (character, user) | `character_id`, `user_id`, `familiarity`, `trust`, `closeness`, `comfort`, `affection`, `shared_experiences`, `stability`, `stage` — **UNIQUE(character_id, user_id)** |
| `relationship_events` | append-only: why the state changed | `character_id`, `kind`, `title`, `detail`, `dimension`, `delta`, `message_id`, `conversation_id`, `occurred_at` |

### Memory

| Table | Purpose | Key columns |
|---|---|---|
| `companion_memories` | durable memory (episodic / semantic_personal / relationship / character) | `character_id`, `kind`, `content`, `importance`, `confidence`, `source`, `source_message_id`, `fictional`, `superseded_by` |

### Conversation

| Table | Purpose | Key columns |
|---|---|---|
| `companion_conversations` | one conversation (mode + scenario are conversation-scoped) | `character_id`, `title`, `mode`, `scenario_id`, `last_active` |
| `companion_messages` | append-only messages | `conversation_id`, `seq`, `role`, `speaker`, `mode`, `content` — **UNIQUE(conversation_id, seq)** |

### Scenario / roleplay

| Table | Purpose | Key columns |
|---|---|---|
| `companion_scenarios` | a dating/roleplay scenario | `title`, `category`, `description`, `location`, `setting_time`, `background`, `starting_situation`, `objectives`, `narrative_style`, `content_level`, `boundaries`, `is_real_place`, `metadata` |
| `scenario_participants` | a character's role in a scenario | `scenario_id`, `character_id`, `role`, `order` |
| `scenario_sessions` | a running roleplay (pause/resume/branch) | `scenario_id`, `conversation_id`, `state`, `branch_of` |

### Profiles, settings, safety

| Table | Purpose | Key columns |
|---|---|---|
| `companion_model_profiles` | a model assignment per character or mode | `scope`, `scope_id`, `model_key`, `context_length`, `max_tokens`, `temperature`, `top_p` |
| `character_voice_profiles` | a character's voice settings | `character_id`, `engine`, `voice_key`, `language`, `dialect`, `speed`, `pitch`, `volume`, `mood`, `enabled` |
| `companion_settings` | key/value Studio settings (`record_id` is the key) | `value` |
| `companion_safety_events` | append-only eligibility/refusal log | `character_id`, `conversation_id`, `kind`, `reason` |
| `companion_audit_entries` | append-only audit trail | `actor`, `action`, `entity`, `entity_id`, `detail`, `before`, `after` |

## Indexes

Every foreign key and common filter is indexed: character `owner`/`active`/
`archived`/`name`; `character_id` on every character-scoped table;
`companion_memories(kind, created_at)`; `companion_conversations(last_active)`;
`companion_scenarios(category, title)`; `companion_safety_events(kind)`; and
`companion_audit_entries(entity, entity_id, actor)`.

## Retention

Deleting a character takes an explicit retention choice (`core/romance/storage.py`):

- `keep` — soft-delete only; conversations and memories survive a restore.
- `purge` — remove the character's conversations (and, by cascade, their messages)
  and its memories, then soft-delete the character.
- `anonymize` — keep the rows but blank the character's identifying fields.

## Backup, restore, export

- `CompanionStorageService.backup()` writes a consistent copy (SQLite's online
  backup API) into `data/romance/backups`.
- `restore()` validates the candidate's `PRAGMA integrity_check` **before**
  replacing anything, and keeps the current file as `<name>.pre-restore`, so a bad
  backup cannot destroy a good database.
- `export()` writes every table (or one character's rows) to a JSON file, so a
  library can be moved between machines with no remote sync.

## Phase roadmap for the schema

Later phases append their own tables and a new numbered migration; nothing already
written changes. A database opened by an older build after a newer one has written
it is **refused**, which is the honest failure rather than a silent misread.
