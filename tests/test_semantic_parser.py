from core.semantic_parser import SemanticParser


def test_parser_extracts_simple_semantic_relation():
    parsed = SemanticParser().parse("A cat is an animal.")
    assert parsed is not None and (parsed.subject, parsed.relation, parsed.object) == ("A cat", "is_a", "animal")
