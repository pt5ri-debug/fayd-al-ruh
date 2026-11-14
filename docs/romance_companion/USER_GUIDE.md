# Romance & Companion Studio — User Guide

A local-first studio for fictional AI companions: independent characters with
persistent personalities, evolving relationships, durable memories, dating
simulation, and roleplay. **Everything stays on this machine** — no cloud
inference and no remote conversation service.

## Opening the Studio

Any of these opens the same window:

- The sidebar's **❤ Romance & Companion Studio** entry.
- The chat composer's **Companion** pill.
- The composer's **＋** menu → *Romance & Companion Studio*.
- Headless: `python -m cli romance …`.

## The window

Three columns, each collapsible:

- **Left sidebar** — *My Companions*, *Archived*, *Character Library*, *Recent
  Conversations*, *Conversation Mode*, *Dating Scenarios & Roleplay*, *Voice*,
  and *Privacy & Safety*.
- **Centre** — the selected character's avatar, name, summary, and relationship
  stage, the conversation transcript, and the message composer.
- **Right panel** — *Profile*, *Personality & Interests*, *Relationship* (with a
  closeness bar), *Milestones*, *Important Memories*, *Voice Configuration*, and
  *Active Scenario*. The header's **Right panel** control collapses it.

An honest disclosure sits at the top of the transcript, **outside the fiction**:
*these are fictional AI companions, not real people; everything stays on this
machine; adult content is off by default and every fictional participant must be
an adult (18+)*. A character's own words stay in character.

## Making a companion

1. **My Companions → New…** opens a form: a display name (required), an optional
   nickname, a **fictional age**, gender, biography, nationality, languages,
   appearance, relationship style, and a default conversation mode.
2. **Character Library** offers a large set of ready-made, fully-specified
   companions grouped by **relationship archetype** — *Family*, *Neighbours*,
   *Colleagues*, *Romantic partners*, *Courtesans & companions*, *Male escorts*,
   and *Inclusive & LGBTQ+*. Choose an archetype (and optionally type a search),
   pick a character to see its preview, then press **Add the chosen** to add just
   that one — or **Add the whole group** to add the whole archetype at once. Every
   library character is a stated adult (18+) and no real person is depicted.
3. Star a favourite (**★ Favorite**), move one to the archive (**Archive**) and
   back (**Restore**), or delete it (**Delete**) with an explicit retention
   choice for its conversations and memories — *keep*, *purge*, or *anonymize*.

> A fictional age is required for adult content. An unspecified or under-18 age
> is treated as **not adult-eligible**, and the safety floor refuses adult turns
> for it.

## Talking

Select a character, type in the composer, and press **Enter** (**Shift+Enter**
inserts a newline). Each turn:

1. passes the **safety floor** first (a refusal writes nothing),
2. stores your message,
3. runs your local model,
4. stores the reply **only when one was really produced**,
5. updates the **relationship** and records a **memory** from the exchange.

If no model is installed the turn reports the real next step — *train one with
`python -m cli training-center lm-train`* — and never fakes an answer.

## Conversation modes

Pick a mode in the sidebar; changing it **keeps the history** (a mode change
never rewrites old messages). Nine modes ship:

| Mode | For |
|---|---|
| General companion | everyday, friendly conversation |
| Romantic conversation | warm, affectionate, romantic |
| Dating simulation | a fictional date, scene by scene |
| Flirting and playful | light, teasing banter |
| Adult conversation | explicit (off by default; adults only) |
| Emotional support | a gentle, attentive space |
| Interactive storytelling | a collaborative story |
| Character roleplay | stay fully in character |
| Multi-character roleplay | a fictional scene with several characters |

## Adult content

Adult mode is **off by default**. To enable it, press **Adult mode** in the
*Privacy & Safety* block (or `python -m cli romance adult --on --confirm`). It
requires an explicit adult-eligibility confirmation **and** an adult-eligible
character. Sexual content involving minors, age-ambiguous participants,
coercion, exploitation, or non-consensual activity is refused outright, and the
disclosure makes clear these are fictional AI companions, never real people.

## The relationship

Each character has its own relationship with you, tracked on separate,
transparent dimensions — *familiarity, trust, closeness, comfort, affection,
shared experiences, stability* — that move **only** with real interaction
evidence, never at random. The relationship advances through configurable stages
(*Stranger → Acquaintance → Friend → Close Friend → Romantic Interest → Romantic
Partner*). **Milestones** lists the events with their timestamps and the message
that produced them. **Reset relationship** returns a character to the start.

## Memories

The right panel's *Important Memories* shows what a character remembers. The CLI
offers the full lifecycle: `memories`, `recall`, `remember`, `forget`,
`export-memories`. Memories are **per character** and **per user**, and one
character's memories are never visible to another. A memory carries its kind
(*episodic*, *semantic personal*, *relationship*, *character*), an importance
score, and a source-message reference.

## Voice

The *Voice* block lists the voices this machine really has (via the project's
existing speech stack). Pick one to assign it to the selected character, then
**Speak last reply** reads it aloud and **Stop** interrupts. When no engine is
installed the Studio says so and names the next step rather than pretending.

## Dating scenarios and roleplay

**Dating Scenarios & Roleplay → New scenario…** creates a fictional scenario
(title, category, content level, participants). Selecting one shows its details
and lets you start a scene. Categories include *first meeting*, *coffee shop*,
*virtual date*, *romantic evening*, *travel story*, *friendship development*,
and *multi-character stories*.

## Privacy

- The *Privacy & Safety* block shows whether inference is **local only**.
- Conversations, memories, and relationships live in a local SQLite database.
- Nothing is sent to a remote service, and **local-only mode refuses a remote
  provider** even if one is configured.
- **Export** a character (and **Export memories**) to take your data with you;
  **Delete** removes it with the retention choice you make.

## Command-line quick reference

```
python -m cli romance status                 # what the Studio holds right now
python -m cli romance characters             # your companions
python -m cli romance library-categories     # the ready-made archetypes
python -m cli romance library --category lovers   # the library, filtered
python -m cli romance add-library Ruby       # add one ready-made companion
python -m cli romance seed-library --category family  # add a whole archetype
python -m cli romance discover               # the shipped example companions
python -m cli romance add-example Layla      # add one
python -m cli romance new "Layla" --age 27 --interest coffee
python -m cli romance chat CHR-… "hello"     # one turn (opens a conversation)
python -m cli romance send CON-… "hello"     # a turn in an existing conversation
python -m cli romance relationship CHR-…     # the relationship state
python -m cli romance milestones CHR-…       # the relationship history
python -m cli romance remember CHR-… "likes tea" --kind semantic_personal
python -m cli romance voices                 # the voices this machine really has
python -m cli romance safety                 # the safety / local-only status
python -m cli romance adult --on --confirm   # enable adult mode
```

Exit codes: `0` success, `2` invalid arguments, `3` the Studio or a model is
unavailable, `4` an operation was refused, `5` an unexpected failure.
