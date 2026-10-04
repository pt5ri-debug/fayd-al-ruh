from core.concept import Concept


def test_concept_includes_label_and_aliases():
    assert "automobile" in Concept("car", frozenset({"automobile"})).terms
