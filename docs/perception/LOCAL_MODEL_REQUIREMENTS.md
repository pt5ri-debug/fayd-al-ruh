# Local Model Requirements

All perception models are **optional** and use the project's existing **Local Model
Manager** (`core.model_manager`) — never a second registry. The rule is: perception
never loads every model at once, and no model is a mandatory dependency for the
system.

## The target machine

| Item | Value |
| --- | --- |
| OS | Windows 11 (this checkout runs on Linux x86_64 for development) |
| CPU | Intel Core i7 (4 cores visible here) |
| RAM | 16 GB (16.0 GB total, ~13.6 GB free here) |
| GPU | AMD Radeon R7 M360 2 GB + Intel HD 520 1 GB — **integrated/legacy; no verified CUDA** |
| Disk | ~24 GB free here |

**CPU-first.** No perception path assumes a GPU. `llama-cpp-python` on this build
has no verified CUDA/Metal backend for the target GPUs, so the vision model runs
on the CPU. Optional hardware acceleration must be **verified**, never assumed.

## Perception tasks → engines

`perception.models.PERCEPTION_TASKS` maps each perception task to the engine(s)
whose models suit it (a *view* of existing facts):

| Task | Engine(s) | Extra | Status here |
| --- | --- | --- | --- |
| `vision` | Doc's Studio (Qwen3-VL) | `[docs]` | **not installed** — `python -m pip install -r requirements-docs.txt` |
| `ocr` | Tesseract | `[ocr]` + system package | **available** — Tesseract 5.5.0, `eng` + `ara` |
| `speech-to-text` | Qwen3-ASR, `whisper` | `[qwen-asr]` / system | **not installed** — `python -m cli qwen-asr download --approve` |
| `text-to-speech` | Qwen3-TTS, Piper, audio.cpp, system | `[qwen-tts]`/`[piper]`/`[audiocpp]`/system | **not installed** — `python -m cli deps --install speech` |

`perception.models.perception_model_status()` reports each one honestly with its
real next step; `perception.validator.perception_status()` reports the runtime
engine availability.

## Memory estimates (honest, not guarantees)

| Model | Approx. need | Notes |
| --- | --- | --- |
| Qwen3-VL 4B Q4 (vision) | ~3–4 GB | run **sequentially** with any chat model |
| Qwen3-VL 2B Q4 (vision) | ~1.5–2 GB | the lighter option |
| Qwen3-ASR (0.6B/1.7B) | ~1–3 GB | CPU-friendly |
| Qwen3-TTS (0.6B/1.7B) | ~1–3 GB | |
| Piper (VITS ONNX) | small | CPU, per-voice ONNX file |
| Tesseract (OCR) | small | CPU, a system program |

`perception.models.can_load_vision_model()` reads the machine's **own** free memory
and returns a conservative verdict with a reason (an estimate, never a guarantee).

## Loading, unloading, and version tracking

* **Lazy loading**: a model is loaded on first use and reused within a session.
  `LlamaCppDocsClient._load()` and the speech factories load on demand.
* **Unloading**: the project's engines hold no global state, so dropping the
  client/recognizer/synthesizer releases it.
* **Version tracking**: `DocsModelStatus`, `perception.speech.recognizer_name()`,
  and `perception.ocr.ocr_engine_version()` report the model/engine and version.
* **Resource monitoring**: `perception.models.model_resource_report()` reads
  memory, disk, CPU count, and what the OS reports about GPUs
  (`core.model_resources`), each reading carrying its own provenance.

## Graceful handling of a missing model

A missing model is never an error: it is a `FAILED` `PerceptionResult` (or a
`silent` TTS clip) that names the real next step. `python -m cli perception status`
prints the whole picture, and `python -m cli deps` installs the pip half of a
feature.

## Downloads are separate from runtime

Every model download is a **one-time, human-approved** action (`--approve`), kept
apart from the runtime, so the system runs offline after installation:

```
python -m cli deps --install ocr --ocr-languages eng+ara   # OCR engine + data
python -m pip install -r requirements-docs.txt             # vision runtime
python -m cli docs download --approve                      # vision weights
python -m cli qwen-asr download --approve                  # STT weights
```

## Unsupported on this hardware

* **CUDA/GPU acceleration** for the vision/speech models: not verified on the
  target AMD/Intel GPUs — the models run on the CPU. Documented as a limitation,
  not assumed away.
* **QLoRA training** (Phase 6): refused without a CUDA GPU.
* **Speaker diarisation**: no compatible local implementation is bundled; the
  project reports it rather than claiming it.
