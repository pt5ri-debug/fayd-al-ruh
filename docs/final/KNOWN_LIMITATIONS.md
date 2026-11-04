# Known Limitations

This list is honest and current for the audited revision (2.3.0). Nothing here is
hidden or theoretical; each item says what was and was not verified.

---

## 1. Environment / hardware

| ID | Limitation |
| --- | --- |
| L-1 | The intended **Windows 11 / Intel i7 / AMD R7 M360 / Intel HD 520** target was **not available** for this audit. All performance figures come from a Linux container (4 CPUs, ~15 GiB RAM). Windows-specific paths (OneCore/SAPI voices, MCI microphone) were **not executed**. |
| L-2 | The GUI was exercised under a **virtual display** (`xvfb-run`), so the window tests **executed** (6718 passed, 16 skipped — no window test skipped); however **visual/interactive** verification by a human on a real desktop was **not** performed, and Windows-specific UI paths were not run. |

## 2. Optional engines not present here

These are optional and report a real next step when absent; their **offline
behaviour was not tested here**:

| ID | Engine |
| --- | --- |
| L-3 | **Tesseract OCR** (scanned-PDF OCR) — absent; needs the system package + language data |
| L-4 | **Piper** neural TTS — absent (weights) |
| L-5 | **Qwen3-TTS / Qwen3-ASR** — absent (extras + weights) |
| L-6 | **audio.cpp / VoiceTut-TTS** — absent (binary + weights) |
| L-7 | **Doc's Studio Qwen3-VL** vision model — absent (weights + mmproj) |
| L-8 | **Diffusion** image generation/editing (torch + diffusers) — absent |
| L-9 | **Ollama**-served models — no server running here |

## 3. Functional limitations

| ID | Limitation |
| --- | --- |
| L-10 | **Analogy latency:** an analogy question takes ~1.7 s (bounded structure-mapping search, ~200k states). Correct, but the slowest answer path. |
| L-11 | **Memory write latency:** ~40 ms per 1000 writes via `save_many` (batched transaction). |
| L-12 | **`semantic_separation` = 0.5427** — the learned semantic space is honestly modest because the authored corpus is small. |
| L-13 | **PDF Arabic** is recovered only as far as the format allows (glyphs are stored in visual order; a `rtl_hint` flags it). A format limitation, not a bug. |
| L-14 | **DOCX scope:** only the main body is read; headers/footers/footnotes/comments/text boxes/images and tracked deletions are deliberately excluded. |
| L-15 | **GGUF validation** is by magic bytes only; an optional SHA-256 is a documented future hardening. |

## 4. Design boundaries (not defects)

- The system is **not** AGI and has **no** consciousness or human-like
  understanding. It is a local cognitive software system: a symbolic graph, a
  retrieval/grounding path, bounded reasoning bridges, a memory store, a learning
  store, a planning/execution sandbox, and a governed self-improvement loop.
- It **never auto-verifies** a fresh learning record, **never** executes a tool
  outside the sandbox, and **never** deploys a production change without a human
  approval.
- The language model is **small** and trained in-project; its answers are labelled
  as the model's own words, not grounded in cited sources.

## 5. Untested here (summary)

- OCR, all optional speech engines, the vision model, and diffusion — absent.
- Windows voices/microphone — not executable here.
- Seven large models concurrently — deliberately not attempted.
- Target-hardware performance — not measured here.
