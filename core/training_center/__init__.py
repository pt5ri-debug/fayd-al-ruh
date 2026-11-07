"""The Training Center: one governed workspace over the project's own learning.

The Training Center is a *front end* over machinery the project already has. It
does not re-implement ingestion, the knowledge base, the governed datasets, the
controlled training pipeline, the evaluation gate, or the model catalog: it
composes them behind one interface so a user can see, in one place, what the
project has learned, what it has not, and what a training run would really do.

The package is deliberately **additive** and **honest**:

* nothing here trains anything on its own, and nothing here touches a model
  weight -- the controlled Phase-6 pipeline (:mod:`ml.training`) is the only path
  that may, and it is reached through its own coordinator;
* every figure a surface shows is read from the real stores (the knowledge base,
  the governed datasets, the adapter registry, the run log), never fabricated;
* a capability the project does not have is reported as unavailable with the real
  next step, never as a button that appears to work.

The modules are:

* :mod:`core.training_center.records` -- the small durable records (a learning
  example and its status, a history entry).
* :mod:`core.training_center.catalog` -- the one bilingual description of every
  control, so the window, the CLI, and the docs cannot drift.
* :mod:`core.training_center.datasets` -- the Dataset Studio over the governed
  ``ml.training`` builder.
* :mod:`core.training_center.qa_learning` -- question-and-answer learning with a
  review status, so nothing is trained on unreviewed data.
* :mod:`core.training_center.providers` -- the learning providers (Gemini/cloud,
  a local model, a character, a document) behind one small protocol.
* :mod:`core.training_center.jobs` -- the training-job manager over the Phase-6
  coordinator, with cancel and checkpoint resume.
* :mod:`core.training_center.resources` -- honest resource figures and estimates.
* :mod:`core.training_center.pipelines` -- the visual pipeline model and its
  bounded runner.
* :mod:`core.training_center.history` -- the durable history of everything.
* :mod:`core.training_center.dashboard` -- the dashboard's real statistics.
* :mod:`core.training_center.gate` -- the held-out gate for the whole feature.

Nothing here imports Tk, loads a model, or reaches the network on import.
"""

from __future__ import annotations

__all__: list[str] = []
