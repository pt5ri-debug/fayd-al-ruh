"""The held-out gate for the Phase-1 cognitive-core services.

Phases 1's services (the event bus, the resource governor, the knowledge
validator, the transactional integrator, and the query engine) each ship with
unit tests that pin the good case. This module is the *external* check: a
catalogue of hand-written judgements, each run against the **real** engine --
never a re-implementation -- with **negative controls**, so a regression that
the unit tests happen to still pass fails here.

Every judgement is a call to the function the claim is about:

* the event bus must deliver in order, isolate a raising subscriber (the
  publisher is unaffected and the other subscriber still runs), match a prefix
  on a boundary (``cycle`` must not match ``cyclone``), and drop the *oldest*
  event from a bounded queue while counting it;
* the resource governor must defer a request that exceeds the headroom budget,
  defer one whose need cannot be judged (an unknown is not plenty), grant one
  that fits, and never leave a deferral occupying a model slot;
* the knowledge validator must refuse to call an unsupported claim knowledge
  (the core rule), accept a well-evidenced one, report an opposing relation on
  the same pair as a contradiction, and -- the negative control -- **not** report
  an opposite-*direction* claim as a contradiction;
* the integrator must write an accepted claim, refuse an unsupported one, keep a
  conflicting claim beside the one it opposes, and roll the graph back exactly
  when a step fails;
* the query engine must find the shortest chain with its product confidence and
  return ``None`` rather than a fabricated path.

``knowledge_validation_accuracy`` is the share of judgements satisfied; an empty
catalogue scores ``0.0`` rather than a faked pass.
"""

from __future__ import annotations

from core.entity import Entity, EntityType
from core.event_bus import EventBus
from core.evidence import Evidence
from core.knowledge_graph import KnowledgeGraph
from core.knowledge_integration import KnowledgeIntegrator
from core.knowledge_query import KnowledgeQueryEngine
from core.knowledge_validation import KnowledgeClaim, KnowledgeValidator, ValidationVerdict
from core.relation import Relation
from core.resource_governor import GovernorReason, ResourceGovernor, ResourceRequest

GB = 1024**3


def _machine(available: int | None = 8 * GB):
    from core.model_resources import SystemResources

    return SystemResources(cpu_count=8, total_memory_bytes=16 * GB, available_memory_bytes=available)


def _claim(source="porosity", relation="increases", target="permeability", **kwargs) -> KnowledgeClaim:
    kwargs.setdefault("evidence", (Evidence("measured", "SPE", True, 0.9),))
    kwargs.setdefault("confidence", 0.8)
    return KnowledgeClaim(source, relation, target, **kwargs)


def _small_graph() -> tuple[KnowledgeGraph, dict[str, str]]:
    graph = KnowledgeGraph()
    ids: dict[str, str] = {}
    for name in ("porosity", "permeability", "flow_rate"):
        entity = Entity(EntityType.CONCEPT, name, id=f"E-{name}")
        graph.add_entity(entity)
        ids[name] = entity.id
    graph.add_relation(Relation(source=ids["porosity"], target=ids["permeability"], relation_type="increases", confidence=0.8))
    graph.add_relation(Relation(source=ids["permeability"], target=ids["flow_rate"], relation_type="controls", confidence=0.7))
    return graph, ids


def knowledge_validation_accuracy() -> float:
    """The share of held-out judgements the real Phase-1 services satisfy."""
    checks: list[bool] = []

    # -- the event bus -----------------------------------------------------
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(lambda event: seen.append(event.name))
    bus.publish("cycle.started")
    bus.publish("cycle.finished")
    checks.append(seen == ["cycle.started", "cycle.finished"])

    bus = EventBus()
    seen = []

    def _boom(_event):
        raise RuntimeError("observer failed")

    bus.subscribe(_boom)
    bus.subscribe(lambda event: seen.append(event.name))
    event = bus.publish("thing.happened")
    # The publisher is unaffected and the *other* subscriber still ran.
    checks.append(event.name == "thing.happened" and seen == ["thing.happened"])

    bus = EventBus()
    seen = []
    bus.subscribe(lambda event: seen.append(event.name), prefix="cycle")
    bus.publish("cyclone.started")  # negative control: a longer word must not match
    checks.append(seen == [])

    bus = EventBus(queue=True, queue_limit=2)
    seen = []
    bus.subscribe(lambda event: seen.append(event.name))
    for index in range(5):
        bus.publish(f"e{index}")
    bus.drain()
    checks.append(seen == ["e3", "e4"] and bus.dropped == 3)

    # -- the resource governor --------------------------------------------
    governor = ResourceGovernor(resources=lambda: _machine(available=8 * GB))
    # budget = 6 GB, so a 7 GB model is deferred.
    decision = governor.request(ResourceRequest("big model", memory_bytes=7 * GB, model=True))
    checks.append(decision.granted is False and decision.reason is GovernorReason.LOW_MEMORY)

    governor = ResourceGovernor(resources=lambda: _machine(available=None))
    # An unknown available memory is not plenty: a need that cannot be judged is deferred.
    decision = governor.request(ResourceRequest("needs memory", memory_bytes=1 * GB))
    checks.append(decision.granted is False and decision.reason is GovernorReason.UNKNOWN_MEMORY)

    governor = ResourceGovernor(resources=lambda: _machine())
    decision = governor.request(ResourceRequest("modest", memory_bytes=1 * GB))
    checks.append(decision.granted is True)

    governor = ResourceGovernor(resources=lambda: _machine(available=1 * GB), max_concurrent_models=1)
    governor.acquire(ResourceRequest("big", memory_bytes=4 * GB, model=True))
    # A deferral must occupy no slot.
    checks.append(governor.active_models == ())

    # -- the knowledge validator ------------------------------------------
    validator = KnowledgeValidator(known_relations={"increases", "reduces", "controls"})

    result = validator.validate(KnowledgeClaim("a", "increases", "b"))
    # THE RULE: an unsupported claim can never be classified as knowledge.
    checks.append(
        result.verdict is ValidationVerdict.NEEDS_REVIEW and result.storable is False and result.verified is False
    )

    result = validator.validate(_claim())
    checks.append(result.verdict is ValidationVerdict.ACCEPTED and result.storable is True and result.verified is False)

    graph, ids = _small_graph()
    graph.add_relation(Relation(source=ids["porosity"], target=ids["permeability"], relation_type="reduces", confidence=0.6))
    result = validator.validate(_claim(), graph=graph)
    checks.append(result.verdict is ValidationVerdict.NEEDS_REVIEW and any(
        finding.kind.value == "contradiction" for finding in result.findings
    ))

    # Negative control: an opposite-*direction* claim is a different claim, not a
    # contradiction.
    graph, ids = _small_graph()
    graph.add_relation(Relation(source=ids["permeability"], target=ids["porosity"], relation_type="reduces", confidence=0.6))
    result = validator.validate(_claim(), graph=graph)
    checks.append(not any(finding.kind.value == "contradiction" for finding in result.findings))

    # -- the integrator ----------------------------------------------------
    graph = KnowledgeGraph()
    record = KnowledgeIntegrator().integrate(_claim(), graph)
    checks.append(record.stored is True and len(graph.entities) == 2 and len(graph.relations) == 1)

    graph = KnowledgeGraph()
    record = KnowledgeIntegrator().integrate(KnowledgeClaim("a", "causes", "b"), graph)
    checks.append(record.stored is False and len(graph.entities) == 0)

    graph = KnowledgeGraph()
    integrator = KnowledgeIntegrator()
    integrator.integrate(_claim("porosity", "increases", "permeability"), graph)
    record = integrator.integrate(_claim("porosity", "reduces", "permeability"), graph)
    # The conflict is preserved: both relations are kept.
    checks.append(record.stored is True and record.conflict is True and len(graph.relations) == 2)

    class _Boom(KnowledgeValidator):
        def validate(self, claim, *, graph=None, now=None):  # type: ignore[override]
            raise RuntimeError("boom")

    graph = KnowledgeGraph()
    record = KnowledgeIntegrator(validator=_Boom()).integrate(_claim(), graph)
    checks.append(
        record.rolled_back is True and len(graph.entities) == 0 and len(graph.relations) == 0 and len(graph.evidence) == 0
    )

    # -- the query engine --------------------------------------------------
    graph, ids = _small_graph()
    engine = KnowledgeQueryEngine(graph)
    path = engine.path(ids["porosity"], ids["flow_rate"], max_hops=3)
    checks.append(
        path is not None and path.hops == 2 and abs(path.confidence - (0.8 * 0.7)) < 1e-6
    )
    # Negative control: a chain beyond the bound yields nothing, never a fabrication.
    checks.append(engine.path(ids["porosity"], ids["flow_rate"], max_hops=1) is None)

    if not checks:
        return 0.0
    return sum(1 for check in checks if check) / len(checks)


__all__ = ["knowledge_validation_accuracy"]
