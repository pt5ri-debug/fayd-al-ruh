# Fayd Al-Ruh — Dependency Map (Phase 0)

**Version:** 2.3.0 · **Audit date:** 2026-11-02

Every third-party import was enumerated by AST-parsing all 897 Python files. Each
dependency is classified as **Fully local**, **Optional external**, **Internet-
dependent**, or **Replaceable with a local alternative**.

---

## 1. Third-party imports, measured

| Package | Import sites | Files | Class | Where |
| --- | ---: | ---: | --- | --- |
| `pytest` | 200 | 196 | dev-only | tests |
| `numpy` | 31 | 28 | **fully local** | `ml/`, `core/embeddings.py`, `core/llm` |
| `pptx` (python-pptx) | 19 | 4 | optional external (local lib) | `core/pptx_engine.py` |
| `llama_cpp` | 14 | 7 | optional external (local runtime) | model mouths |
| `huggingface_hub` | 11 | 11 | **internet** (download only) | model downloads |
| `PIL` (Pillow) | 9 | 5 | optional external (local lib) | image/OCR/docs |
| `requests` | 3 | 3 | **internet** (+ localhost) | `core/web_source.py`, `core/knowledge_base.py` (internet); `ai/ollama_client.py` (localhost Ollama — local) |
| `pypdf` | 2 | 2 | optional external (local lib) | PDF extraction |
| `piper` | 2 | 1 | optional external (local engine) | `core/piper_speech.py` |
| `torch` | 2 | 2 | optional external (local) | `imagegen_diffusion.py`, `qwen_speech.py` |
| `pytesseract` | 2 | 1 | optional external (local engine) | `core/pdf_ocr.py` |
| `pypdfium2` | 2 | 2 | optional external (local lib) | OCR/docs raster |
| `diffusers` | 1 | 1 | optional external (local) | image diffusion |
| `qwen_asr` | 1 | 1 | optional external (local) | `core/qwen_speech.py` |
| `qwen_tts` | 1 | 1 | optional external (local) | `core/qwen_speech.py` |
| `PyInstaller` | 1 | 1 | dev-only | `fayd-al-ruh.spec` |

\* `storage/json_store.py` imports `urllib.parse.quote/unquote` (stdlib string
escaping only) — **not** a network call. `core/{cloud_model,coder_model,
model_download,audiocpp_speech}.py` use `urllib.request` for a local server
(Ollama) or a governed download. The AST scan's `urllib` hit is not an internet
claim by itself.

### Base install (must be enough for a full offline answer)

`pyproject.toml` declares exactly two runtime dependencies: `numpy>=2.0` and
`requests>=2.31`. `requirements.txt` (what CI installs) adds `pypdf` and
`python-pptx` so CI can exercise the PDF and deck paths. **All four are local
libraries**; none requires a network at import time.

---

## 2. Classification per dependency

### Fully local (no network at any time)

- `numpy` — all numeric work, the transformer, the semantic space.
- `pypdf` — parses a PDF file on disk.
- `python-pptx` — writes an OOXML package on disk.
- `Pillow`, `pypdfium2` — local image work.
- `pytesseract` — drives the locally installed Tesseract binary.
- `piper` — local ONNX neural TTS.
- `torch` + `diffusers` — local diffusion (image gen/edit).
- `qwen_asr`, `qwen_tts` — local speech models.
- `llama_cpp` — local GGUF inference.
- `pytest`, `PyInstaller` — build/dev only.

### Optional external (a local library the user installs)

`pypdf`, `python-pptx`, `Pillow`, `pypdfium2`, `pytesseract`, `piper`, `torch`,
`diffusers`, `qwen_asr`, `qwen_tts`, `llama_cpp` — each is an **extra**, imported
lazily, and every module reports the real next step when it is absent. Removing
them all still leaves a working system (retrieval + reasoning + memory).

### Internet-dependent (only on an explicit, governed action)

- `huggingface_hub` — **download only**. Used by every `download_model`
  (`core/ultron.py`, `vision_model.py`, `banner.py`, `docs_model.py`,
  `qwen_speech.py`, `piper_speech.py`, `shared_models.py`, `companion/model.py`,
  `hakeem/model.py`). Each runs only with `approved=True` (permission + human
  approval gates). **Nothing imports it at module load.**
- `requests` — used by `core/web_source.py` (governed web search),
  `core/knowledge_base.py` (bulk Wikipedia build), and `core/updater.py`
  (program update check). All three are explicit actions.

### Replaceable with a local alternative

| Internet use | Local alternative | Status |
| --- | --- | --- |
| Web search (`web_source.py`) | the internal corpus + bulk knowledge base | ✅ already the default (search off by default since Phase 53) |
| Bulk Wikipedia build (`knowledge_base.py`) | authored seeds + local ingestion | ✅ authored seeds are always present |
| Model downloads (`huggingface_hub`) | copy a GGUF file into the model folder by hand; `core/custom_models.py` registers a hand-added GGUF | ✅ available |
| Program update (`updater.py`) | manual `git pull`, or a new portable bundle | ✅ documented |

---

## 3. Offline-operation assessment per capability

| Capability | Offline today? | Evidence | Target |
| --- | --- | --- | --- |
| Language models | ✅ | GGUF/Ollama run in-process/local; cloud is opt-in | add a "no socket opened" test |
| Embeddings | ✅ | `core/embeddings.py` learns from the local corpus | keep |
| Databases | ✅ | SQLite + JSON, stdlib | keep |
| Knowledge retrieval | ✅ | local corpus + seeds | keep |
| Memory | 🟡 | working/episodic in-process; `knowledge_carry` durable for relations | make episodic durable |
| OCR | ✅ | local Tesseract | keep |
| Speech recognition | ✅ | system + local Qwen3-ASR | keep |
| Text-to-speech | ✅ | system / Piper / Qwen3-TTS / audio.cpp, all local | keep |
| Image processing | ✅ | heuristic renderer + local diffusion | keep |
| Learning & training | ✅ | numpy in-project; LoRA (to add) runs locally | keep |
| Personality management | ✅ | local config + local stores | keep |
| Task execution | ✅ | local sandbox | keep |

**Conclusion:** the system is already **offline-capable for normal operation**.
The one gap is that this is not *enforced or tested*: nothing fails if a code path
starts reaching the network. See `TECHNICAL_GAPS.md` gap O-1.

---

## 4. Internal dependency direction (layering)

The import graph respects a clean layering, verified by inspection:

```
tests/          → everything
gui/ cli/       → ai/, core/, cognition/, ml/, storage/, ingest/, evaluation/
ai/             → core/, cognition/, companion/, hakeem/, ml/, storage/
cognition/      → core/, storage/, ml/ (backends), ai.local_model (lazy)
companion/ hakeem/ → core/ (paths, voices), storage/, own modules
ingest/         → core/ (knowledge_base, document_reader, pdf_ocr), storage/
ml/             → core/ (paths, progress, curriculum/corpus), storage/
core/           → storage/  (no import of gui/, cli/, ai/)
storage/        → stdlib only
```

**No cycles between packages** were found. `core/` never imports `gui/` or `cli/`
(the GUI is imported lazily by `cli/`; `cli/` never imports Tk). This is enforced
by the project's own convention and by `tests/test_packaging.py`.

---

## 5. System (non-Python) dependencies

| Dependency | Purpose | Optional? | Windows 11 |
| --- | --- | --- | --- |
| `git` | Code Studio worktrees; updater | yes (both report a next step) | ✅ (bundled with Git for Windows) |
| Tesseract OCR | scanned-PDF OCR | yes (`[ocr]` extra) | ✅ (UB-Mannheim installer) |
| espeak-ng / `say` / SAPI | system TTS | yes | ✅ (SAPI/OneCore built in) |
| audio.cpp (`audiocpp_cli`) | VoiceTut Arabic cloning TTS | yes | ✅ (native binary fetched by the module) |
| Xvfb | CI-only virtual display | CI only | n/a |

---

## 6. Dependency risks

| Risk | Severity | Mitigation (existing or target) |
| --- | --- | --- |
| `llama-cpp-python` build requires a compiler on Windows | Medium | extras are optional; Ollama is a fallback; `core/optional_deps.py` reports the pip command |
| `torch` is large (~2 GB) | Medium | only the `[image]`/`[qwen-*]` extras need it |
| `qwen-asr` vs `qwen-tts` pin different `transformers` versions | High (documented) | two separate extras; never install both |
| `huggingface_hub` download path is the only internet use for models | Low | governed by approval gate; a hand-copied GGUF works |
| Base install pulls `requests` even when search is off | Low | target: keep `requests` lazy in the base path (see gap O-2) |
