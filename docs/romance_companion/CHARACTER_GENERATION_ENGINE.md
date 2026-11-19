# FR-CCE — Fayd Al-Ruh Character Creation Engine

A native, procedural 3D human character-generation and customization engine,
integrated into the existing **Body & shape** window. It is not a separate
application and not a competing window: the Body & shape window is its frontend.

## What it is (and is not)

It generates four complete human bodies — **adult female**, **adult male**,
**girl**, **boy** — as real, single-skin geometry, sculpted and rendered **locally**
by the project's own 3D systems (no GPU, no network, no model download). It is a
*stylised parametric humanoid*, not a licensed likeness, a photograph, or a
commercial render engine, and the engine says so in its own `status()`.

## Architecture

| Component | Responsibility |
| --- | --- |
| `CharacterGenerationEngine` | The facade: create, morph, pose, render, save/restore |
| `CharacterTemplateManager` | The four base templates and the sex/age → template rule |
| `ProceduralMeshGenerator` | Builds a template's mesh (caches by parameter signature) |
| `BodyParameterSystem` | The dial catalogue, groups, clamping, and refusals |
| `MorphTargetEngine` | Applies a parameter and **measures** the geometry change |
| `SkeletonRigManager` | The 19-bone humanoid rig and linear-blend skin binding |
| `MaterialManager` | Skin / eye / hair materials; roughness, metallic; honest caps |
| `CharacterPresetManager` | Named body presets, saved and loaded |
| `CharacterPersistenceManager` | A character's own body, versioned and migrated |
| `CharacterPreviewRenderer` | Renders the body (rig, materials, lights) to a PNG |

All of it lives in `core/character_generation.py` and composes the project's
existing `core.scene3d` / `core.scene3d_figures` / `core.scene3d_render` /
`core.romance.body` — it never re-implements a mesh builder or a renderer.

## The four templates

| Key | Title | Build | Height | Age |
| --- | --- | --- | --- | --- |
| `woman` | Adult female | feminine / curvy | 1.75 m | adult |
| `girl` | Girl | feminine / child | 1.32 m | child |
| `man` | Adult male | masculine / athletic | 1.75 m | adult |
| `boy` | Boy | masculine / child | 1.38 m | child |

Each is a **real body at its age and sex**, not a scaled copy: the children are
shorter with different proportions and no adult bust. Every template fuses into
**one connected skin** (`Mesh.largest_shell` is a single piece).

## Parameters (58 dials in 16 groups)

Overall proportions, height, head, facial proportions, neck, shoulders, chest,
waist, abdomen, pelvis, arms, hands, thighs, legs, feet, and general body shape —
plus pose and the detail dials. Every numeric dial is bounded, and a value out of
range is **refused** (never silently clamped at the geometry level).

A parameter change moves **actual vertices**: `MorphTargetEngine.apply` returns a
`MorphResult` whose `moved_vertices` and `max_displacement` are measured by
comparing the built meshes. A slider that reports a new number but moves no vertex
counts as *no* change.

## Rig

A 19-bone humanoid rig — spine, neck, head, shoulders, upper arms, forearms,
hands, fingers, pelvis, upper legs, lower legs, feet, toes — with linear-blend
skinning. Changing a proportion rebuilds the rig **with** the body, so a longer
arm is a longer bone, and the rig stays compatible.

## Persistence

Every character has an independent body configuration, saved under
`data/romance/characters/bodies/<character_id>.json`, recording the template id,
character id, parameters, morph values, skeleton, materials, preset id, last
modified timestamp, engine version, and config version. A configuration written
by an earlier build is **migrated** on load; a corrupt one reads as absent. A
character's saved body is written only by `save`; **regenerating a template never
overwrites it**.

## Materials

Skin, eye, and hair materials with roughness, metallic, and lighting controls.
The project's CPU rasteriser has **no image sampler**, so texture maps and normal
maps are *not* supported — the engine's `capabilities()` says so plainly rather
than pretending. Nothing external is mandatory: no texture file is required.

## Performance

- CPU-only rendering (the project's own z-buffered rasteriser).
- Lazy generation: the mesh is built only when a render is asked for, on the
  window's worker thread.
- Geometry caching keyed by the parameter signature (`ProceduralMeshGenerator`).
- The window works in **parameter space**: a dial change re-renders (which builds
  the mesh) but never builds a mesh just to update a control.

## The Body & shape window (the frontend)

The existing `gui/romance_body.py` window is the engine's own frontend. It offers
the four templates (marked with the one in force), real-time preview, every body
parameter, camera rotation/zoom and front/side/rear views, lighting controls,
material preview (skin/hair colours), **Regenerate from template**, **Save custom
preset** / **Load custom preset**, **Save this character's body**, and **Restore
the saved body**. Selecting a character in the main application opens the body
builder on that character's matching template.

## Command line

```
python -m cli romance body --templates
python -m cli romance body --engine-status
python -m cli romance body <id> --template man --set height=1.9 --save-body --render out.png
python -m cli romance body <id> --body-config
python -m cli romance body <id> --save-preset tall
python -m cli romance body --body-presets
python -m cli romance body <id> --restore-body --summary
```

## Tests

`tests/test_character_generation.py` (40 tests) pins the four templates, geometry
validity, vertex deformation (not merely a slider value), the parameter limits,
the character mapping (including the minor negative control), independent
configurations, save/load persistence and migration, template regeneration,
rig compatibility, the materials, the render, offline operation, and that the
generator is a thin layer over the project's own figure builder.
`tests/test_romance_body_studio.py` adds the service and window wiring.
