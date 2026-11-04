"""Controlled self-improvement (Phase 9).

A human-gated loop: **monitor → detect → propose → isolate → measure →
accept/reject → deploy → roll back**. Every stage reuses a proven component
(the event bus, the resource governor, the Code Studio worktrees, the model
version registry, the evaluation gate) rather than re-implementing one, and the
loop **never** merges without an explicit human approval and **never** touches
production code, model weights, or datasets on its own.

The package is dependency-free, Tk-free, offline, and CPU-first: importing it
loads no model, opens no socket, and writes nothing. The one facade the rest of
the project uses is :func:`core.self_improvement.improvement_loop`.
"""

from __future__ import annotations

__all__ = [
    "approval",
    "deployment",
    "evaluation",
    "experiments",
    "gate",
    "health",
    "history",
    "loop",
    "metrics",
    "monitor",
    "prioritization",
    "proposals",
    "regression",
    "rollback",
    "sandbox",
    "weakness",
]
