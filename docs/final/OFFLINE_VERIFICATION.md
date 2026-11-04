# Offline Verification

**Requirement:** the base install must never require a network socket for normal
operation (Technical Gap **G-8**, priority **P0**). This document records what
was verified, how, and what remains untested.

---

## 1. Method

1. **Static egress inventory** — every `socket`, `urllib.request`, `requests`,
   `huggingface_hub`, and hard-coded URL in the source was located and traced to
   its gate.
2. **Runtime guard** — the new `core/offline.py` guard was installed and the core
   paths were run with all **external** connections blocked; the count of attempts
   is zero by construction (any attempt raises `NetworkDisabled`).
3. **Independent probe** — a separate script ran the answer path (EN + AR),
   memory, the cognitive controller, a transformer smoke step, a personality
   exchange and the learning coordinator with the guard in force.

The guard blocks **external** hosts only; **loopback is permitted**, because a
local model server (Ollama, the audio.cpp resident server) is part of *local*
operation. This is the honest meaning of "offline": no traffic to another machine.

## 2. Result — the core paths run offline

Re-run on merged `main` (`389c586`): with `FAYD_OFFLINE=1` and the guard in force,
**3/3 external hosts** (`example.com`, `8.8.8.8`, `pypi.org`) were refused
(`NetworkDisabled`) and **loopback stayed open** for local model servers.

| Path | Result |
| --- | --- |
| `answer_path` (grounded answer + citations) | PASS — answer produced, citations present, **0 network attempts** |
| `arabic_answer_path` | PASS |
| `long_term_memory` (store + read-back) | PASS |
| `cognitive_controller` (analogy) | PASS |
| `transformer_smoke` (forward pass) | PASS |
| `learning_coordinator` | PASS |

New tests: `tests/test_offline.py` (17 tests) — the switch, the guard (blocks a
remote socket/`getaddrinfo`/`create_connection`/`urlopen`/`requests`; **allows**
loopback), reversibility, and the four real local paths under the guard.

## 3. Network egress inventory (what can leave the machine)

| Path | Gate | Default |
| --- | --- | --- |
| `core/web_source.py` (search) | permission + approval + audit | **off** |
| `core/web_verifier.py` (verify) | same | **off** unless enabled |
| `core/knowledge_base.py` (bulk build) | explicit CLI command | **not built** |
| `core/updater.py` (update check) | explicit CLI/GUI action | **not run** |
| every `download_model` (Ultron/Vision/Banner/Natasha/Hakeem/shared/Piper/Qwen/audio.cpp/docs) | `approved=True` + permission | **not run** |
| `core/cloud_model.py` | permission + approval | **off** |
| `ingest/connectors/web.py` | registered only with `allow_web=True` | **off** |

No telemetry, analytics, crash reporter, or hidden beacon exists in the source
(searched for `telemetry|analytics|sentry|mixpanel|segment|posthog` — none).
The only hosts named in `core/` are the governed endpoints above plus
`huggingface.co` (downloads), `github.com` (the project/audio.cpp releases),
`en.wikipedia.org` (the governed search/build), and schema URLs that are never
fetched.

## 4. The hard-off switch

- Set `FAYD_OFFLINE=1` (also `true`/`yes`/`on`) to block every external
  connection for the whole run.
- Wired at the program entry points: `python -m cli`, `python -m ai.chat`,
  `python -m ai.jarvis`, `python -m gui`.
- `core.offline.describe()` reports whether it is in force.

## 5. Classification of every capability

| Capability | Classification |
| --- | --- |
| Grounded answer path (EN/AR) | **Fully offline and verified** |
| Long-term memory (4 kinds) | **Fully offline and verified** |
| Cognitive controller / ontology / analogy | **Fully offline and verified** |
| In-project transformer (train/forward) | **Fully offline and verified** |
| Learning coordinator | **Fully offline and verified** |
| Local document text/PDF/DOCX ingestion | **Offline** (verified for text/PDF/DOCX; pypdf present) |
| OCR of scanned PDFs | **Not tested here** — Tesseract absent (engine is optional) |
| Piper / Qwen3-TTS / Qwen3-ASR / audio.cpp voices | **Not tested here** — engines absent |
| Doc's Studio vision (Qwen3-VL) | **Not tested here** — weights absent |
| Diffusion image generation/editing | **Not tested here** — extra absent |
| GUI (Tk) | **Offline**; window tests skip without a display |
| Web search / Cloud mouth / model downloads | **Dependent on an external service — gated, off by default** |

## 6. Honesty statement

A local model *path* in configuration was **not** treated as proof of offline
operation. Each "fully offline" row above was executed with the guard in force
and no external attempt. Rows marked **not tested here** are absent engines on
this machine; they are **not** claimed to work offline.
