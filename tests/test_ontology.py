from core.concept import Concept
from core.ontology import Ontology


def test_ontology_stores_and_finds_concepts():
    ontology = Ontology()
    concept = Concept("car", frozenset({"automobile"}))
    ontology.add_concept(concept)
    assert ontology.find_concepts("automobile") == [concept]


def _seeded_ontology() -> Ontology:
    ontology = Ontology()
    for name in ("a", "b", "c"):
        ontology.add_concept(Concept(name))
    return ontology


def test_add_relation_once_is_idempotent():
    ontology = _seeded_ontology()
    source, target = ontology.find_concepts("a")[0], ontology.find_concepts("b")[0]
    ontology.add_relation_once(source.id, target.id, "relates", 0.4)
    ontology.add_relation_once(source.id, target.id, "relates", 0.9)
    assert len(ontology.relations) == 1


def test_add_relation_once_survives_id_collisions(monkeypatch):
    from core import ontology as ontology_module
    from core.semantic_relation import SemanticRelation as RealRelation

    ids = iter(["SEM-dup", "SEM-dup", "SEM-fresh"])
    seen: list[str] = []

    def factory(source_id, target_id, relation_type, confidence):
        relation = RealRelation(source_id, target_id, relation_type, confidence)
        object.__setattr__(relation, "id", next(ids))
        seen.append(relation.id)
        return relation

    monkeypatch.setattr(ontology_module, "SemanticRelation", factory)
    ontology = _seeded_ontology()
    a, b, c = (ontology.find_concepts(n)[0] for n in ("a", "b", "c"))
    ontology.add_relation_once(a.id, b.id, "relates")
    ontology.add_relation_once(a.id, c.id, "relates")
    assert len(ontology.relations) == 2
    assert seen == ["SEM-dup", "SEM-dup", "SEM-fresh"]


def test_add_concept_once_survives_id_collisions(monkeypatch):
    from core import ontology as ontology_module
    from core.concept import Concept as RealConcept

    ids = iter(["CON-dup", "CON-dup", "CON-fresh"])

    def factory(label, aliases=frozenset()):
        concept = RealConcept(label, aliases)
        object.__setattr__(concept, "id", next(ids))
        return concept

    monkeypatch.setattr(ontology_module, "Concept", factory)
    ontology = Ontology()
    first = ontology.add_concept_once("alpha")
    second = ontology.add_concept_once("beta")
    assert {first.id, second.id} == {"CON-dup", "CON-fresh"}
    assert len(ontology.concepts) == 2


# ------------------------------------------- the reverse index (lookup speed)


def _seeded() -> Ontology:
    from core.domain_knowledge import seed_ontology
    from core.petroleum_corpus import seed_petroleum_corpus

    ontology = Ontology()
    seed_ontology(ontology)
    seed_petroleum_corpus(ontology)
    return ontology


def test_the_incremental_index_matches_a_full_rebuild():
    """Adding concepts one at a time must leave the same index as building it at once.

    Seeding adds concepts incrementally (a full rebuild per addition was quadratic
    and dominated the seed time), so the incremental path has to be *exactly* the
    same structure a from-scratch build produces.
    """
    ontology = _seeded()
    rebuilt = Ontology()
    for concept in ontology.concepts.values():
        rebuilt.concepts[concept.id] = concept
    rebuilt._ensure_index()
    assert set(rebuilt._exact_index) == set(ontology._exact_index)
    for term, concepts in rebuilt._exact_index.items():
        assert sorted(c.id for c in concepts) == sorted(
            c.id for c in ontology._exact_index[term]
        )
    assert set(rebuilt._prefix_index) == set(ontology._prefix_index)
    for key, entries in rebuilt._prefix_index.items():
        assert sorted((t, tuple(sorted(c.id for c in cs))) for t, cs in entries) == sorted(
            (t, tuple(sorted(c.id for c in cs))) for t, cs in ontology._prefix_index[key]
        )


def test_a_lookup_does_not_scan_the_whole_graph():
    """A term lookup is proportional to the term, not to the number of concepts.

    The graph is padded with thousands of unrelated concepts; a lookup must not get
    slower, which it would if it still walked every concept's every term (the
    30-second-per-query behaviour this replaced).
    """
    import time

    ontology = _seeded()
    ontology.find_concepts("porosity")  # build the index

    def timed() -> float:
        start = time.perf_counter()
        for _ in range(200):
            ontology.find_concepts("porosity")
            ontology.concepts_in("the porosity and permeability of the rock")
        return time.perf_counter() - start

    before = timed()
    from core.concept import Concept

    for index in range(2000):
        ontology.add_concept(Concept(f"filler{index}", frozenset({f"zz{index}"})))
    after = timed()
    # 2000 unrelated concepts must not change the cost of a lookup by more than a
    # generous factor (the index is a dict hit either way).
    assert after < before * 5 + 0.5, f"{before=} {after=}"


def test_an_alias_added_after_the_index_is_built_is_found():
    """``add_concept_alias`` refreshes the index in place (it must not go stale)."""
    ontology = _seeded()
    assert ontology.find_concepts("porosity")  # build the index first
    concept = ontology.find_concepts("porosity")[0]
    assert ontology.find_concepts("مصطلح جديد تماما") == []
    ontology.add_concept_alias(concept.id, "مصطلح جديد تماما")
    assert [c.id for c in ontology.find_concepts("مصطلح جديد تماما")] == [concept.id]
    assert [c.id for c in ontology.find_concepts("porosity")] == [concept.id]
    assert concept.id in ontology.concepts_in("مصطلح جديد تماما")


def test_concepts_in_remembers_a_repeated_text():
    """The same text is answered from the cache, and the cache is bounded."""
    from core import ontology as ontology_module

    ontology = _seeded()
    first = ontology.concepts_in("porosity controls permeability")
    assert ontology.concepts_in("porosity controls permeability") == first
    for index in range(ontology_module._TEXT_CACHE_SIZE + 10):
        ontology.concepts_in(f"filler text {index}")
    assert len(ontology._text_cache) <= ontology_module._TEXT_CACHE_SIZE
    # A cached answer is still correct after the cache was cycled.
    assert ontology.concepts_in("porosity controls permeability") == first


def test_a_new_concept_drops_the_remembered_answers():
    """A concept added after a text was cached is seen by the next lookup."""
    ontology = _seeded()
    assert ontology.concepts_in("زرقمفريد") == set()
    concept = ontology.add_concept_once("زرقمفريد")
    assert ontology.concepts_in("زرقمفريد") == {concept.id}
