from core.semantic_relation import SemanticRelation


def test_semantic_relation_has_type_and_confidence():
    assert SemanticRelation("A", "B", "is_a", .9).relation_type == "is_a"
