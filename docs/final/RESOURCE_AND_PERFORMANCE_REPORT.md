# Resource and Performance Report

**Environment actually measured:** Linux container, Python 3.13.15, 4 CPUs,
~15 GiB RAM. Figures **re-measured on merged `main` (`389c586`)**.
**Intended target (not available here):** Windows 11, Intel Core i7, 16 GB RAM,
AMD Radeon R7 M360 2 GB, Intel HD Graphics 520 1 GB.

> The figures below are **measured on the audit machine**, not on the target.
> Target-specific figures are marked **not measured here**. CPU-first operation
> was confirmed; no GPU/CUDA path was assumed or required.

---

## 1. Measured figures (audit machine, merged `main`)

| Metric | Value | Notes |
| --- | --- | --- |
| `import ai.jarvis` | ~0.15 s | incl. interpreter start |
| `build_session(text_only=True)` | ~0.35 s | seeds + retriever index |
| Idle RSS after build | ~60.8 MB | `ru_maxrss` |
| First answer (cold, English) | ~0.17 s | "what is porosity?" |
| Warm answer (Arabic, all reasoning layers) | ~0.008–0.010 s | e.g. "هل التحميض يزيد النفاذية؟" |
| 4 mixed answers | < 1 s total | warm |
| Arabic answer share (body) | ~30–35 % | Arabic question → Arabic body |
| 1000 durable memory writes | ~40 ms | `MemoryDatabase.save_many` |
| Memory paged read (20) | ~0.81 ms | SQLite |
| Memory integrity check | `(True, 'ok')` | |
| Peak RSS (answer path) | ~60.8 MB | whole probe process |
| Full `pytest` suite | 6718 passed, 16 skipped, ~33 min | under `xvfb-run` |
| `mypy` | clean, 1016 files | static gate |
| `ruff check .` | clean | static gate |

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

Resource usage is **acceptable** for the target class: sub-second startup, ~61 MB
idle, single-digit-millisecond warm answers (including the Arabic path with all
reasoning layers), and microsecond-scale SQLite access. The one slow path (R-1,
the bounded analogy search) is documented, not hidden.
