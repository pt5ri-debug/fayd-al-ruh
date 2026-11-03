"""Phase 6 -- controlled model-weight training for Fayd Al-Ruh.

The package is deliberately **additive**: nothing here trains anything on its own,
nothing touches a model weight by default, and every capability is a governed,
versioned, reversible operation that reuses the infrastructure already in the
project (the Model Manager, the Model Registry, the cognition version log, the
personality isolation policy, and the in-project numpy transformer).

The layers are:

* :mod:`ml.training.datasets` -- dataset governance: the record contract, the
  validation pipeline, duplicate/leakage detection, train/validation/test
  separation, and the **versioned, manifest-carrying snapshot**.
* :mod:`ml.training.isolation` -- dataset isolation: which personalities a dataset
  may draw from, and the **Natasha exclusion** for the project model's aggregate.
* :mod:`ml.training.compatibility` -- the model compatibility validator: whether a
  base model can actually be fine-tuned by a method on this machine.
* :mod:`ml.training.config` -- the training configuration manager and its
  validation.
* :mod:`ml.training.resources` -- the training resource monitor and the disk/RAM
  guards.
* :mod:`ml.training.adapters` -- the LoRA/PEFT adapter layer: a real low-rank
  adapter over the in-project transformer, and the optional ``peft`` backend.
* :mod:`ml.training.trainer` -- the adapter trainer (job lifecycle, checkpoints,
  cancellation, resumption).
* :mod:`ml.training.evaluation` -- the evaluation pipeline: baseline vs candidate
  on held-out data.
* :mod:`ml.training.registry` -- the model and adapter registry, the promotion
  manager, and the rollback manager.
* :mod:`ml.training.coordinator` -- the training coordinator that wires it all.
* :mod:`ml.training.gate` -- the held-out gate for the whole phase.

It is dependency-free (stdlib + numpy, already a base dependency), Tk-free, and
offline: importing it loads no model and touches no network.
"""

from __future__ import annotations

__all__: list[str] = []
