# Romance & Companion Studio — Hardware Compatibility

This document states what the Studio assumes about the machine it runs on, and how
it stays honest about hardware it cannot measure. Nothing here is a benchmark; the
Studio's own evaluation facility measures the *real* machine at run time
(`core/romance/model_eval.py`).

## Target development machine

| Resource | Value |
|---|---|
| System RAM | 16 GB |
| GPU 1 | AMD Radeon R7 M360, 2 GB VRAM |
| GPU 2 | Intel HD 520, 1 GB graphics memory |

## Policy

- **CPU-first.** The Studio's safe defaults request `gpu_layers=0` and a bounded
  context (`SAFE_CONTEXT` = 4096, `SAFE_MAX_TOKENS` = 512). A GPU is used only when
  a build reports offload support; the router never assumes one.
- **Never both adapters at once.** The Studio does not try to combine the discrete
  and integrated GPUs; a GPU, when used, is the one the runtime reports.
- **Never all of system memory.** The context and output caps keep the KV cache
  small relative to the weights, and `memory_estimate()`/`preflight()` report the
  estimated need (with a low/high band and the unknowns named) *before* a load, so a
  user can decide rather than discover an out-of-memory failure.
- **Honest `None`.** A figure the platform cannot report is `None`, never a zero:
  `resources()` returns `available_memory_bytes=None` where it cannot be read.

## What "compatible" means here

A model is **not** called compatible because its file exists. The Studio checks:

1. the file is a recognised format and passes its integrity probe;
2. the runtime for that format is available on this machine;
3. the runtime's own `validate()` accepts the file;
4. the estimated memory fits (the `preflight` report).

Only then does `status()`/`health()` report the model ready to load — and even a
model that fits the estimate can still fail to load, which the report says plainly.

## Measuring the real machine

The evaluation facility (`core/romance.model_eval.ModelEvaluator`) is the headless
engine the Studio's window and CLI will drive in later phases. It runs the probe
battery through the resolved model and returns a `ModelEvaluationReport` with
per-dimension scores, total generation time, and the change in free system memory
across the run. When no model is installed it reports `not measured` with the real
next step instead of a score.

## VRAM

The local runtime used here does not report per-model VRAM to the Studio, so the
evaluation reports CPU use unless a build reported GPU offload. This is stated in
the report's `vram_note` rather than guessed.
