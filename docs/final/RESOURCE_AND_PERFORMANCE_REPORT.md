# Resource and Performance Report

**Environment actually measured:** Linux container, Python 3.13.15, 4 CPUs,
~15 GiB RAM.
**Intended target (not available here):** Windows 11, Intel Core i7, 16 GB RAM,
AMD Radeon R7 M360 2 GB, Intel HD Graphics 520 1 GB.

> The figures below are **measured on the audit machine**, not on the target.
> Target-specific figures are marked **not measured here**. CPU-first operation
> was confirmed; no GPU/CUDA path was assumed or required.

---

## 1. Measured figures (audit machine)

| Metric | Value | Notes |
| --- | --- | --- |
| `import core.assistant` | ~136 ms | incl. interpreter start |
| `build_default_assistant()` | ~530 ms | incl. interpreter start; seeds + retriever index |
| Idle RSS after build | ~23–60 MB | `ru_maxrss` after building the assistant |
| Peak RSS (answer path + transformer) | ~69 MB | whole probe process |
| First answer | ~162 ms | cold |
| Warm answer (definitional) | ~7–10 ms | e.g. "what is permeability?" |
| Arabic answer | ~7 ms | warm |
| Analogy answer | ~1.7 s | bounded structure-mapping search (~200k states) |
| 4 mixed answers | ~1.7 s | dominated by the single analogy question |
| 200 durable memory writes | ~1.0 s | ~5 ms each (per-write transaction) |
| Memory page (50) | ~2.8 ms | SQLite |
| Transformer build (small preset) | ~12 ms | `TransformerConfig.preset("small")` |
| Transformer forward (1×16) | ~3 ms | manual backprop implementation |
| `ruff check .` / `mypy` | clean | static gates |

## 2. Model loading / unloading

- Optional engines (Ultron/Vision/Banner/Natasha/Hakeem/Piper/Qwen/audio.cpp/
  Doc's Studio) load **lazily** and only on demand; none imports at module load,
  so the base install carries no model memory cost.
- The audio.cpp reader prefers a **resident server** (one model load per read)
  over a per-utterance process; the project documents ~6.6 s vs ~10 s per
  utterance on CPU. **Not re-measured here** (engine absent).
- **Seven large models were not run concurrently** — as the assignment requires.

## 3. Resource-exhaustion safety

- `core/model_resources` reports memory/disk honestly (an honest `None` where the
  platform cannot report it), and `core/model_preflight` gives an **estimated**
  memory band with its unknowns named — never a guaranteed figure.
- `core/model_process` records only pids **this program started** and signals only
  those, so a managed server can be stopped safely.
- `core.model_manager.active_models()` distinguishes *loaded* from *running*, so
  an optional component can be unloaded when it is not needed.

## 4. Concurrency

- Long work runs on a worker thread whose result is handed back through a queue
  the UI thread drains (Tk is not thread-safe), so the window never blocks.
- The Scene Studio publishes a render atomically (temp file renamed into place),
  so a concurrent reader never sees a truncated image.

## 5. Target-hardware expectations (not measured here)

On the intended Windows 11 / i7 / 16 GB machine the CPU-first figures above
should be comparable or slightly slower; the AMD R7 M360 / Intel HD 520 GPUs are
**not** used for the base paths (no CUDA). A full seven-model concurrent load is
**not** advised and was not attempted. Final acceptance should re-run the probe
on the target and record the real numbers.

## 6. Acceptance

Resource usage is **acceptable** for the target class: a few hundred milliseconds
to start, tens of MB idle, single-digit-ms warm answers, and a bounded ~1.7 s
worst case (one analogy). The one slow path (R-1) is documented, not hidden.
