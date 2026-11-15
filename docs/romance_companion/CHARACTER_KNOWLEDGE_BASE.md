# Per-companion knowledge bases

Every companion in the Romance & Companion Studio owns a **knowledge base of its
own** — the words, sentences, self-knowledge, and domain material that define how
*that* character talks and what *that* character knows. It is built locally, from
the character's own records, and it is what the companion draws on when it composes
a reply.

This document describes what the knowledge base is, how it is built, and how it
reaches a conversation.

## What it is

A companion's knowledge base is a real
[`core.knowledge_base.KnowledgeBaseStore`](../core/knowledge_base.py) — the same
sharded JSONL store the Import Center and the Knowledge Base Manager fill — that
lives inside the character's own folder:

```
data/romance/characters/<character_id>/knowledge_base/
    shard-00000.jsonl
    shard-00001.jsonl
    …
    manifest.json
    index.sqlite3        # the SQLite FTS5 index (core.knowledge_index)
```

Because it is a plain store, it is read the same way every other corpus in the
project is read, and it can be inspected, exported, backed up, or deleted with the
tools that already exist. There is **no second RAG engine**: the companion's
knowledge is retrieved through `core.rag.BulkCorpusRetriever` over
`core.knowledge_index.KnowledgeTextIndex`, exactly like the bulk knowledge base.

## What is in it

The builder (`core/romance/knowledge_pack.py`) renders eight sections from the
character's own records. The first four are **curated** (authored, finite, always
present); the rest are **expanded** (generated combinatorially until the byte
target is met).

| Section | What it holds |
| --- | --- |
| `voice_card` | The character's self-knowledge: identity, personality, interests, boundaries, and speech style, as one document. |
| `biography` | The character's own fictional biography, expanded into a life story with scenes. |
| `personality` | How each personality dial shows up in the character's own words. |
| `phrasebook` | Ready-made lines the character is likely to say, grouped by function (greeting, farewell, affection, teasing, comfort, encouragement, apology, surprise, laughter, thinking, agreement, disagreement, questions). |
| `interests` | One passage per interest and hobby, with the character's own take. |
| `knowledge` | General-life and archetype material, framed in the character's voice. |
| `dialogue` | Example exchanges (user → character and character → user) per conversation mode. |
| `scenes` | Dating- and roleplay-scene fragments. |

The pack is **bilingual**: English and Arabic passages are generated for every
section, so a companion can answer in the language the user writes in.

## How it is built

```python
from core.romance.character_files import CharacterFiles

files = CharacterFiles(character_id)          # the companion's own folder
pack = files.build_knowledge_pack(target_mb=50.0)
print(pack.describe())
```

* **Deterministic.** The generator is seeded by the character's own id, so a
  rebuild produces the same bytes; there is no randomness and no network.
* **Character-specific.** Every passage is rendered from the character's name,
  nickname, nationality, dialect, kinship, traits, interests, and boundaries, so
  two companions never share a pack.
* **Bounded.** The builder stops when the byte target is reached (default 50 MB,
  clamped to `TARGET_MB_RANGE`), and de-duplicates by content hash, so a rebuild
  adds nothing and a pack never contains an exact duplicate.
* **Isolated.** A pack lives under the character's own folder and is retrieved
  only through that character's own reader, so one companion can never read
  another's knowledge.
* **Offline.** Nothing here loads a model or opens a socket; it is pure file,
  index, and text work.

The builder writes in bounded batches and builds the FTS index **once** at the
end, so a 50 MB pack is written in a few seconds rather than one document at a
time.

## How it reaches a conversation

The Studio's prompt builder (`gui.romance.RomanceStudioService._memory_block`)
already retrieves a companion's own knowledge for the current question through
`CharacterFiles.reference_block(query)` and injects it into the prompt as
labelled **reference information** (never as instructions). With a knowledge base
built, that block is filled with the companion's own voice material and domain
knowledge, so the model composes its reply from the character's own words.

The block is bounded (passages and characters) so a large pack never floods the
context.

## Honesty

* The knowledge base is **generated**, deterministically, from the character's own
  records and a curated template/fact bank. It is not scraped from the internet
  and it contains no third-party text. Its value is *coverage of the character's
  voice and domain*, not novel facts.
* It is **not** a claim that the character is human, and it does not widen what a
  companion may say. The safety floor, the adult gate, and the "you are a virtual
  character" honesty rule all still hold above it.
* A companion with **no** pack behaves exactly as it did before: the reference
  block is simply empty, and the turn is unchanged.

## Command-line

```
python -m cli romance knowledge-pack CHR-… --target-mb 50   # build it
python -m cli romance knowledge-status CHR-…                # what is on disk
```

## Settings

| Constant | Meaning | Default |
| --- | --- | --- |
| `DEFAULT_TARGET_MB` | The size a pack is built to. | 50.0 |
| `TARGET_MB_RANGE` | The clamp on a requested target. | (1.0, 200.0) |
| `DEFAULT_LANGUAGES` | The languages a pack is built in. | `("en", "ar")` |
| `DEFAULT_BATCH` | Chunks per shard write. | 2000 |
