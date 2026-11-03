"""The integration engine: transactional writes, honest refusals, preserved conflict."""

from __future__ import annotations

import pytest
from core.evidence import Evidence
from core.knowledge_graph import KnowledgeGraph
from core.knowledge_integration import (
    IntegrationRecord,
    KnowledgeIntegrator,
)
from core.knowledge_validation import (
    KnowledgeClaim,
    KnowledgeValidator,
    ValidationVerdict,
)


def _evidence(source: str = "SPE", strength: float = 0.9) -> Evidence:
    return Evidence(description=f"note from {source}", source=source, supports=True, strength=strength)


def _claim(source="porosity", relation="increases", target="permeability", **kwargs) -> KnowledgeClaim:
    kwargs.setdefault("evidence", (_evidence(),))
    kwargs.setdefault("confidence", 0.8)
    return KnowledgeClaim(source, relation, target, **kwargs)


# -- the happy path -------------------------------------------------------


def test_an_accepted_claim_is_written_and_reported():
    graph = KnowledgeGraph()
    integrator = KnowledgeIntegrator()
    record = integrator.integrate(_claim(), graph)
    assert record.stored is True
    assert record.verdict == ValidationVerdict.ACCEPTED.value
    assert len(graph.entities) == 2
    assert len(graph.relations) == 1
    assert len(graph.evidence) == 1
    assert record.relation_id
    assert "* update_graph" in record.describe()


def test_a_second_identical_claim_reinforces_rather_than_duplicates():
    graph = KnowledgeGraph()
    integrator = KnowledgeIntegrator()
    first = integrator.integrate(_claim(confidence=0.6), graph)
    before = graph.get_relation(first.relation_id).confidence
    second = integrator.integrate(_claim(confidence=0.9, evidence=(_evidence("OnePetro"),)), graph)
    assert len(graph.relations) == 1
    assert second.relation_id == first.relation_id
    relation = graph.get_relation(first.relation_id)
    # Reinforced: the merged confidence is the stronger of the two, and the
    # second item's evidence is appended rather than replacing the first's.
    assert relation.confidence >= before
    assert len(relation.evidence_ids) == 2


def test_entities_are_reused_by_name_across_claims():
    graph = KnowledgeGraph()
    integrator = KnowledgeIntegrator()
    integrator.integrate(_claim(), graph)
    integrator.integrate(_claim(relation="measures", source="porosity", target="grain_density"), graph)
    names = sorted(entity.name for entity in graph.entities.values())
    assert names == ["grain_density", "permeability", "porosity"]
    assert len(graph.entities) == 3


# -- refusals -------------------------------------------------------------


def test_an_unsupported_claim_is_not_stored_and_creates_no_entities():
    graph = KnowledgeGraph()
    integrator = KnowledgeIntegrator()
    record = integrator.integrate(KnowledgeClaim("a", "causes", "b"), graph)
    assert record.stored is False
    assert record.verdict == ValidationVerdict.NEEDS_REVIEW.value
    assert len(graph.entities) == 0
    assert len(graph.relations) == 0


def test_a_structurally_broken_claim_is_rejected():
    graph = KnowledgeGraph()
    record = KnowledgeIntegrator().integrate(KnowledgeClaim("", "causes", "b", evidence=(_evidence(),)), graph)
    assert record.stored is False
    assert record.verdict == ValidationVerdict.REJECTED.value
    assert len(graph.entities) == 0


# -- conflicts ------------------------------------------------------------


def test_a_conflicting_claim_is_stored_beside_the_one_it_opposes():
    graph = KnowledgeGraph()
    integrator = KnowledgeIntegrator()
    integrator.integrate(_claim("porosity", "increases", "permeability"), graph)
    record = integrator.integrate(_claim("porosity", "reduces", "permeability"), graph)
    assert record.stored is True
    assert record.conflict is True
    # Both relations are kept -- nothing is deleted.
    assert len(graph.relations) == 2
    types = {relation.relation_type for relation in graph.relations.values()}
    assert types == {"increases", "reduces"}


# -- transactional rollback -----------------------------------------------


class _BoomValidator(KnowledgeValidator):
    def validate(self, claim, *, graph=None, now=None):  # type: ignore[override]
        raise RuntimeError("validator boom")


def test_a_failing_step_rolls_the_graph_back_exactly():
    graph = KnowledgeGraph()
    integrator = KnowledgeIntegrator(validator=_BoomValidator())
    record = integrator.integrate(_claim(), graph)
    assert record.rolled_back is True
    assert "RuntimeError" in record.error
    assert len(graph.entities) == 0
    assert len(graph.relations) == 0
    assert len(graph.evidence) == 0


def test_rollback_leaves_a_prior_write_untouched():
    graph = KnowledgeGraph()
    good = KnowledgeIntegrator()
    good.integrate(_claim(), graph)
    before = (len(graph.entities), len(graph.relations), len(graph.evidence))

    failing = KnowledgeIntegrator(validator=_BoomValidator())
    failing.integrate(_claim("x", "causes", "y"), graph)
    assert (len(graph.entities), len(graph.relations), len(graph.evidence)) == before


# -- learner wiring -------------------------------------------------------


class _Learner:
    def __init__(self):
        self.calls = []

    def observe_relation(self, source, target, relation_type, confidence=0.5):
        self.calls.append((source, target, relation_type))
        return object()


def test_an_accepted_claim_with_labels_reaches_the_concept_learner():
    graph = KnowledgeGraph()
    learner = _Learner()
    integrator = KnowledgeIntegrator(learner=learner)
    integrator.integrate(
        _claim(source_label="porosity", target_label="permeability"), graph
    )
    assert learner.calls == [("porosity", "permeability", "increases")]


def test_a_claim_without_labels_does_not_reach_the_learner():
    graph = KnowledgeGraph()
    learner = _Learner()
    KnowledgeIntegrator(learner=learner).integrate(_claim(), graph)
    assert learner.calls == []


# -- history and observers ------------------------------------------------


def test_history_is_recorded_and_bounded():
    graph = KnowledgeGraph()
    integrator = KnowledgeIntegrator(history_limit=2)
    for index in range(4):
        integrator.integrate(_claim(target=f"t{index}"), graph)
    assert len(integrator.history) == 2


def test_an_observer_sees_each_record():
    seen: list[IntegrationRecord] = []
    integrator = KnowledgeIntegrator(on_event=seen.append)
    integrator.integrate(_claim(), KnowledgeGraph())
    assert len(seen) == 1
    assert seen[0].stored is True


def test_a_raising_observer_never_breaks_the_write():
    def boom(_record):
        raise RuntimeError("observer failed")

    graph = KnowledgeGraph()
    record = KnowledgeIntegrator(on_event=boom).integrate(_claim(), graph)
    assert record.stored is True


def test_describe_counts_attempts_and_conflicts():
    graph = KnowledgeGraph()
    integrator = KnowledgeIntegrator()
    integrator.integrate(_claim(), graph)
    integrator.integrate(_claim(relation="reduces"), graph)
    text = integrator.describe()
    assert "2 attempt(s)" in text
    assert "1 conflict(s)" in text


def test_record_to_primitive_is_json_safe():
    record = KnowledgeIntegrator().integrate(_claim(), KnowledgeGraph())
    payload = record.to_primitive()
    assert payload["stored"] is True
    assert isinstance(payload["steps"], list)


def test_a_negative_history_limit_is_refused():
    with pytest.raises(ValueError):
        KnowledgeIntegrator(history_limit=-1)
