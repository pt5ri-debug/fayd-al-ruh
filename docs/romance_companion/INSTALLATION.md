# Romance & Companion Studio — Installation & Configuration

The Studio is part of Fayd al-Ruh and adds **no new dependency** and **no
network requirement**. Installing the project installs the Studio.

## Install

```bash
git clone <the repository>
cd fayd-al-ruh
python -m pip install -r requirements.txt
```

That is enough to open the Studio, create characters, keep conversations,
memories, relationships, and scenarios, and drive everything from the CLI.
Inference and voice need their own engines (below); with none installed the
Studio still runs and reports the real next step.

## Opening it

- `python -m cli gui` → the sidebar's **❤ Romance & Companion Studio** entry (or
  the chat composer's **Companion** pill).
- Headless: `python -m cli romance status`.

## Configuration

### A language model (for real answers)

The Studio answers a turn through the project's **existing Model Manager**, so a
model is installed exactly as for the rest of Fayd al-Ruh:

```bash
python -m cli models list                 # what is installed / available
python -m cli models download <key> --approve   # a governed download
python -m cli models test <id>            # prove it answers
```

Assign a model to a character or to a conversation mode with
`CompanionModelRouter.assign_character` / `assign_mode` (the CLI and window do
this for you). CPU-first defaults are bounded for a 16 GB machine; a model larger
than the available memory is refused with a real reason rather than crashing.

### A voice engine (for speech)

Voices come from the project's existing speech stack, so install an engine as for
the rest of the app (`python -m cli deps --install speech`, or the window's
*Install* control). The Studio lists **only** the voices the machine really has,
filtered by engine and language.

### Storage location

All Studio data lives under the project's data directory (git-ignored),
`data/romance/` by default, resolved through `core.paths`. Point it elsewhere with
`FAYD_DATA_DIR` (the same variable the rest of the project honours).

## Privacy defaults

- Inference is **local only**; a remote provider is refused in local-only mode.
- Adult mode is **off by default** and needs an explicit confirmation.
- No conversation data is written to a log and nothing is sent anywhere.
- Export and delete are always available; deletion honours your retention choice.

See [`PRIVACY_AND_SAFETY.md`](PRIVACY_AND_SAFETY.md).

## Verifying the install

```bash
python -m pytest tests/test_romance_*.py -q     # the Studio's own suite
python -m cli romance status                    # the Studio's real state
python -m evaluation                            # the whole-project gate (PASS)
```

## Troubleshooting

| Symptom | Next step |
|---|---|
| "No answer — no model" on a turn | install/test a model: `python -m cli models list` then `download --approve` |
| The voice list is empty | install a speech engine: `python -m cli deps --install speech` |
| A turn is refused with a safety reason | the content level needs adult mode on (and an adult-eligible character), or the text hits a hard limit |
| The window will not open | it needs a display; the CLI (`python -m cli romance …`) works headless |
