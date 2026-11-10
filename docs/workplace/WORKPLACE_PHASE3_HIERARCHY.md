# Workplace — Phase 3: the configurable hierarchy and taxonomy

Phase 2 gave the Workplace its *shape* (the composer control, the window, the
institution, the ten divisions). Phase 3 completes the **configurable
hierarchy**: every part of the institutional structure and every division's
technical profile is now *data a user edits*, never a source change, and the
interface language and layout direction are settings.

## What Phase 3 adds

### 1. Each division's technical profile (the brief's §3)

A division must support its description and scope, its technical disciplines,
available personnel and qualifications, applicable standards, inspection and
testing capabilities, procedures and work instructions, equipment and resources,
project assignments, technical reports, historical records, and knowledge
documents.

Rather than ten separate tables (and a schema change for the eleventh), every
item is one row in a single `division_details` table, typed by `kind`:

| kind         | what it holds                                |
|--------------|----------------------------------------------|
| `discipline` | a technical discipline the division covers   |
| `personnel`  | a person and their qualification             |
| `standard`   | an applicable standard                       |
| `capability` | an inspection / testing capability           |
| `procedure`  | a procedure or work instruction              |
| `equipment`  | equipment or a resource                      |
| `project`    | a project assignment                         |
| `report`     | a technical report                           |
| `history`    | a historical record                          |
| `knowledge`  | a knowledge document                         |

So the profile is **fully configurable** and a new kind needs no migration. The
seed gives each division its own scope as a first `capability` item, so the
*shape* exists from the first moment and is editable.

### 2. The professional knowledge taxonomy (the brief's §4), stored and editable

The shipped taxonomy (`core/workplace/taxonomy.py`: 8 areas, 52 disciplines) is
the **default** set. Phase 3 stores it in a `taxonomy_nodes` table -- an area is
a node with no parent, a discipline is a node whose parent is its area -- so an
administrator **adds an area or a sub-discipline, renames one, or deactivates
one without touching source code**. The tree is two levels in one table and can
grow a third level later.

### 3. Interface language and layout direction

`core/workplace/settings.py` persists the Workplace's user preferences (interface
language `en`/`ar`, right-to-left layout, the default institution, whether
proposed records are marked) through the project's own `JsonStore`, so a choice
survives a restart. Choosing Arabic implies RTL unless the user explicitly turns
it off.

## Data model (added in Phase 3)

Two tables, appended by a **new numbered migration** (schema version 1 → 2);
nothing already written changes, and a database created by Phase 2 migrates
forward with no data loss.

```
division_details
  record_id PK | created_at | updated_at | active
  division_id FK -> divisions(record_id) ON DELETE RESTRICT
  kind | title | detail | reference | sort_order | metadata

taxonomy_nodes
  record_id PK | created_at | updated_at | active
  parent_id FK -> taxonomy_nodes(record_id) ON DELETE RESTRICT
  kind | key | name | name_ar | summary | sort_order | proposed | metadata
```

Every relationship is `ON DELETE RESTRICT` and every record is *deactivated*
rather than deleted, so a reorganisation never destroys the historical records
that reference a unit or a division.

## Surfaces

* **The window** (`gui/workplace_dialog.py`) gains a *Divisional profile* sidebar
  section (the selected division's profile, an *Add profile item…* button with a
  kind picker, and a *Remove selected item…* button), a *Professional taxonomy*
  section with *Add area…* and *Add discipline…*, and an interface-language /
  RTL block in *Settings*.
* **The CLI** (`python -m cli workplace …`) gains `profile`, `add-detail`,
  `remove-detail`, `kinds`, `move-division`, `add-taxonomy-area`,
  `add-taxonomy-discipline`, `rename-taxonomy`, `remove-taxonomy`, and
  `settings` — the headless twin of the window, driving the *same*
  `gui.workplace.WorkplaceService` and the *same* database.

## Reuse (no parallel system)

Phase 3 adds **no** new document engine, RAG, or model manager. It reuses:

* the project's `storage.json_store.JsonStore` for the settings;
* the Workplace's own `WorkplaceDatabase` (stdlib `sqlite3`), whose
  schema/migration discipline follows `core.memory_db`;
* the shared `gui.components`, `gui.sidebar`, and `gui.dialog_base` for the
  window.

## Honesty rules (unchanged)

Only publicly documented EPRI facts are seeded (`official=True`). The TSTC centre
and its ten divisions remain **proposed, configurable defaults** (`proposed=True`,
a `metadata` source note), never a claim of an officially approved structure. A
user's own taxonomy node is marked `proposed=False` and carries a `source: user`
note, so a reader can always tell a shipped default from a user's edit.

## Tests

`tests/test_workplace_phase3.py` (23 tests) pins the profile (add, remove, group,
the kind rule, the required title/detail, the real-division rule), the stored
taxonomy (seeded on first use, add/edit/remove an area, add a discipline, the
unique-key and real-area rules), the settings (defaults, persistence, the corrupt
store, the round trip, the read-only folder), the migration (v1 → v2 with no data
loss, a newer database refused), and the CLI (`profile`, `add-detail`, `kinds`,
taxonomy edit, settings, `move-division`). `tests/test_workplace_window.py` gains
the profile panel, the added item, and the language/direction controls (skipped
with no display).
