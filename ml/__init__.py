"""Learning package: a real, trainable ML stack for the project.

The project previously only adjusted symbolic confidences by hand-written
rules. This package adds an actual statistical learning layer on top of the
symbolic substrate:

- ``dataset``   : replay buffers, deterministic feature hashing, train/val splits
- ``models``    : softmax classifiers and a small MLP with backprop, plus an Adam optimiser
- ``trainer``   : a training loop with metrics, early stopping, and persistence
- ``learner``   : symbolic reinforcement learning over the sandbox tasks
- ``cognitive_brain`` : a predictive-coding loop wiring the above into one brain

Everything here is deterministic and dependency-light (numpy only), so it runs
in the same safe, replayable environments the rest of the project already uses.
"""

from __future__ import annotations

__all__ = [
    "dataset",
    "models",
    "trainer",
    "learner",
    "cognitive_brain",
]
