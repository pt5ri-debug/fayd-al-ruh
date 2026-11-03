"""The multi-agent layer: seven personalities as distinct cognitive agents.

Each of the project's seven characters (Jarvis, Ultron, Vision, Banner, Natasha,
Hakeem, Cloud) is a *character* in the chat; this package makes each one an
**agent** -- an independently addressable actor with its own identity, capability
set, working context, task history, and (via the existing stores) memory and
learning scope, all sharing the one Cognitive Core, Memory Architecture, Learning
System, and Reasoning Engine under the project's isolation policy.

Nothing here is a second implementation of what already exists. The layers are:

* :mod:`core.agents.capabilities` -- the deny-by-default capability set and the
  permission gate every action passes.
* :mod:`core.agents.registry` -- the Personality Registry: one addressable record
  per personality, built from the character identity + configuration that already
  exist.
* :mod:`core.agents.context` -- the independent working context per personality.
* :mod:`core.agents.goals` -- the goal/task manager and task history.
* :mod:`core.agents.memory_access` -- the personality memory access layer, over
  the project's own isolation policy and stores.
* :mod:`core.agents.learning_scope` -- the agent learning scope, over the Phase-3
  learning coordinator and the isolation policy.
* :mod:`core.agents.runtime` -- the shared agent runtime (context, execution,
  timeout, cancellation, resource limits, error handling).
* :mod:`core.agents.routing` -- the traceable task router.
* :mod:`core.agents.collaboration` -- bounded, explicit delegation between agents.
* :mod:`core.agents.conflict` -- conflict preservation and structured comparison.
* :mod:`core.agents.monitoring` -- per-personality evaluation and monitoring.
* :mod:`core.agents.gate` -- the held-out gate for the whole layer.

It is dependency-free, Tk-free, and offline: importing it loads no model and
touches no network.
"""

from __future__ import annotations

__all__: list[str] = []
