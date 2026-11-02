# Fayd Al-Ruh — Resource Assessment (Phase 0)

**Version:** 2.3.0 · **Audit date:** 2026-11-02

**Important scope note.** This Phase-0 audit ran inside a **Linux container**
(Debian 13, Python 3.13.15, 4 CPUs, 16 GiB RAM, no GPU). The *target* machine is
described as **Windows 11, Intel Core i7, 16 GB RAM, AMD Radeon R7 M360 2 GB,
Intel HD 520 1 GB**. The container figures below were **measured here**; the
target-machine figures are **reasoned guidance**, clearly labelled, because the
target hardware was not available to measure. No GPU claim is made without
evidence.

---

## 1. Measured environment (this audit host)

Reported by the project's own `core/model_resources.system_resources()`:

```
system            Linux
machine           x86_64
python            3.13.15
cpu_count         4
total_memory      16,767,217,664 bytes  (~15.6 GiB)
available_memory  14,183,550,976 bytes  (~13.2 GiB)
disk_free         25,698,811,904 bytes  (~23.9 GiB at data/)
gpu               available=False  ("nvidia-smi is not installed here, or reported no GPU")
process_memory    ~15 MB
memory_reading    measured
disk_reading      measured
```

Measured during the audit:

| Action | Observed cost |
| --- | --- |
| Import + seed ontology (495 concepts, 474 relations) + Arabic glossary + causal graph | < 2 s, well under 1 GiB |
| Load the trained in-project LLM checkpoint (6.2 MB npz, 842k params) | seconds, tens of MB |
| Full `pytest` run under Xvfb | ~20 min wall on 4 CPUs (see `TEST_BASELINE.md`) |
| `ruff check .` | seconds |
| `mypy` over 872 files | ~2 min |

---

## 2. Repository footprint on disk

| Asset | Size | Committed? |
| --- | ---: | --- |
| `ml/llm/checkpoints/fayd-llm.npz` | 6.2 MB | yes (release artifact) |
| `ml/artifacts/` (policies, brains, relation model, manifest) | 8.7 MB | yes |
| Full source tree (897 `.py`) | ~15 MB | yes |
| Bulk knowledge base (built on demand) | up to 100 MB+ | no (`data/`, git-ignored) |
| A local GGUF chat model (e.g. 4B Q4) | ~2.5 GB | no (downloaded) |
| A local vision model (Qwen3-VL 4B + mmproj) | ~3–4 GB | no |
| Piper voice (ONNX) | ~63 MB | no |
| VoiceTut-TTS GGUF | ~1.35 GB | no |

---

## 3. What runs simultaneously on a 16 GB machine

Rule of thumb: **one large model at a time.** A GGUF chat model holds its whole
file in RAM (plus KV cache), and loading two doubles the pressure.

| Combination | Feasible on 16 GB? | Notes |
| --- | --- | --- |
| Retrieval + reasoning only (no model) | ✅ comfortable | < 1 GiB; the symbolic path is cheap |
| One 1.7B Q4 chat model + retrieval | ✅ | ~1.5 GB weights + KV |
| One 4B Q4 chat model + retrieval | ✅ | ~2.5–3 GB weights + KV |
| One 7–8B Q4 chat model + retrieval | 🟡 tight | ~5–6 GB weights + KV; close other apps |
| Two large models at once | ❌ | not on 16 GB |
| A model + background thought cycles | 🟡 | the resource governor (Phase 1) must defer cycles |
| A model + LoRA training | ❌ | train **sequentially**, with the chat model unloaded |
| Qwen3-VL 4B + a chat model | ❌ | run the vision model alone |

**Conclusion:** the architecture must **serialize** large-model use. The pieces
exist (`cognition/runner` defers on load; `core/model_preflight` estimates the
need), and Phase 1 adds one arbiter.

---

## 4. GPU compatibility — the honest answer

The target GPUs are **AMD Radeon R7 M360 (2 GB)** and **Intel HD 520 (1 GB)**.

- The Radeon R7 M360 is a **GCN 1st-gen (2015)** part. Modern acceleration stacks
  target newer hardware: **ROCm** supports a limited set of newer AMD GPUs, and
  **CUDA is NVIDIA-only**. An R7 M360 is **not** on the supported list for a
  current ROCm or a practical PyTorch ROCm build.
- The Intel HD 520 (Skylake, 2015) has no useful acceleration for the local
  inference stacks this project uses (llama.cpp CPU/CUDA/Metal/Vulkan); an
  integrated part also shares the 16 GB system RAM, so it adds no memory.
- 2 GB of VRAM cannot hold a useful quantised model anyway (a 4B Q4 model is
  ~2.5 GB, before KV cache).

**Therefore: do not plan on GPU acceleration on the target machine.** Run
llama.cpp **on the CPU**, and treat any GPU as unavailable until measured with
the project's own `system_resources()` (which already reports
`gpu.available=False` honestly and never guesses).

This is not a limitation of the project — it is the hardware. The architecture
must **remain scalable** so a future CUDA-capable GPU can be added without
redesign; the model runtime adapters already abstract this.

### CPU-only model sizing (16 GB RAM, i7, CPU inference)

| Model | Q4 size | + KV (4k ctx) | Feasible | Expected speed |
| --- | ---: | ---: | --- | --- |
| 1.7B | ~1.2 GB | ~0.3 GB | ✅ | usable (seconds per reply) |
| 2B | ~1.5 GB | ~0.4 GB | ✅ | usable |
| 4B | ~2.5 GB | ~0.7 GB | ✅ | a few seconds per reply |
| 7–8B | ~5 GB | ~1 GB | 🟡 | slow on CPU (many seconds) |
| 12B+ | ~8 GB+ | ~1.5 GB | ❌ | not practical on CPU |

The project's own defaults already reflect this: Natasha/Hakeem default to
speed-first settings (`think=False`, `max_tokens=512`, `context=4096`,
`keep_alive="30m"`), and a CPU-friendly batch size.

---

## 5. Training feasibility

| Training target | On the target machine |
| --- | --- |
| In-project numpy transformer (842k params) | ✅ fast; full retrain in minutes |
| A **larger** in-project transformer (e.g. 10–40M params) | 🟡 minutes–hours on CPU |
| **LoRA on a 1.7B** quantised model (CPU) | 🟡 hours |
| **LoRA on a 4B** quantised model (CPU) | 🟡 many hours; feasible overnight |
| **LoRA on a 7B+** | ❌ impractical on CPU |
| Full fine-tune of any real model | ❌ not on this hardware |

**Recommendation:** keep Phase 6 (LoRA) to **small base models (1.7B–4B)** and
small adapters, train during idle periods, and always through the
version/rollback gate. The architecture's staged training (`SMOKE`→`FULL`)
already lets a run be split across sessions.

---

## 6. Memory optimisation opportunities (already present, extend)

1. **Lazy imports everywhere** — every optional engine (torch, llama_cpp, piper,
   qwen, diffusers) imports on use, not at module load. Measured: importing
   `core` costs little.
2. **Streaming ingestion** — `ingest/pipeline.py` never loads a whole dataset.
3. **Bounded retrieval** — `KnowledgeTextIndex` (FTS5) returns only winning
   chunks; `BulkCorpusRetriever` streams otherwise.
4. **Bounded memory** — `working_memory` (capacity 7), `memory_decay`,
   `retention_policy`.
5. **`best_effort` + bounded error record** (`MAX_RECORDED`) so a long run cannot
   grow without limit.
6. **Contentless FTS index** — ~60 MB without duplicating text.
7. **`keep_alive="30m"`** so the model is not reloaded per turn.
8. **`num_ctx`/`n_batch` tuning** already present in the Ollama/llama paths.

Target additions: the resource governor (Phase 1) to serialize loads, and a
durable-memory cap (Phase 2).

---

## 7. Sequential vs. schedulable operations

| Must run **sequentially** | May run during **idle** periods |
| --- | --- |
| Any two large model loads | Background thought cycles (`cognition/runner`) |
| A model load + LoRA training | Bulk knowledge-base build |
| Vision model + chat model | Embedding-space retraining |
| Anything holding the GPU (if present) | Consolidation / spaced review |

The scheduler already has `LIGHT`/`BALANCED`/`INTENSIVE` modes and a quiet window;
Phase 1's governor extends the same rule to model loads.

---

## 8. Scalability statement

The target architecture is **CPU-first and GPU-agnostic**:

- `core/model_runtime.py` exposes adapters (`llama.cpp`, `ollama`,
  `in-process`, `cloud`); adding a CUDA/Metal/Vulkan path is a new adapter, not a
  redesign.
- `core/model_preflight.estimate_memory` already takes the *context length* and
  *quantisation* into account and returns a band, so it scales to bigger models.
- `cognition/versions.py` is hardware-independent (it compares metrics, not
  machines).
- No component assumes a specific core count or VRAM.

A future machine with more RAM and a supported GPU can run the same code with a
larger model and a bigger LoRA adapter, with no architectural change.
